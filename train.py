"""
train.py
========

Training loop for RIR-Former (Sec. 3 "Training Objective" of arXiv:2602.01861).

Reproduces the paper's training recipe:
  * AdamW optimizer, lr = 3e-4, batch size 8 (all configurable in config.py).
  * Loss = MSE between prediction and ground truth, computed **only** over
    the missing (query) positions (Eq. 10).
  * Curriculum masking: the fraction of array points treated as "missing"
    ramps linearly from 30% to 70% over the first `mask_warmup_epochs`
    epochs, then stays at 70% for the remainder of the main training run.
  * A subsequent per-segment decoder finetuning stage: after the main
    training run, each temporal-segment branch is finetuned individually
    (all other decoder heads and the shared encoder frozen) for
    `finetune_epochs` epochs, "to balance
    the imbalanced loss over the time dimension".

Checkpoints are saved in the same format consumed by `evaluate.py`
(and compatible with the official `eval_rirformer.py` loader):
    {"model_state_dict": ..., "epoch": ..., "config": cfg.to_dict()}
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config, get_config
from dataset_generator import build_rir_dataset, collate_fn, generate_dataset
from model import RIRFormer, build_model


# --------------------------------------------------------------------------- #
# Masking curriculum
# --------------------------------------------------------------------------- #
def mask_ratio_for_epoch(epoch: int, cfg: Config) -> float:
    """Linearly ramps from min_mask_ratio to max_mask_ratio over
    `mask_warmup_epochs`, then holds at max_mask_ratio."""
    t = cfg.train
    if t.mask_warmup_epochs <= 0:
        return t.max_mask_ratio
    frac = min(1.0, epoch / t.mask_warmup_epochs)
    return t.min_mask_ratio + frac * (t.max_mask_ratio - t.min_mask_ratio)


# --------------------------------------------------------------------------- #
# Loss (Eq. 10): L = (1/N) * ||H_hat - H_bar||_F^2, missing points only
# --------------------------------------------------------------------------- #
def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """pred, target: (B, L, K). mask: (B, L), 1=observed, 0=missing.

    Matches Eq. (10) exactly: the squared Frobenius-norm error is summed
    over the K time samples of each missing row, then averaged only over
    N, the *number of missing rows* -- NOT further divided by K. (An
    earlier version of this function divided by N*K, i.e. a plain
    per-element MSE; that's a constant rescaling by K relative to the
    paper's stated objective.)
    """
    missing = (1.0 - mask).unsqueeze(-1)  # (B, L, 1) -- already counts *rows*, not elements
    se = (pred - target) ** 2 * missing   # (B, L, K), broadcast over K only here
    n_missing_rows = missing.sum()        # N in Eq. (10)
    row_sq_err_sum = se.sum()             # sum_n sum_k (h_bar - h_hat)^2
    return row_sq_err_sum / n_missing_rows.clamp(min=1.0)

# --------------------------------------------------------------------------- #
# OPTIONAL auxiliary loss term (default OFF via cd_loss_weight=0.0): the
# same 1 - cos_angle quantity CD measures at eval time (see
# compute_metrics_per_sample below), computed over the missing rows of a
# training batch. MSE alone rewards squared-error magnitude but doesn't
# explicitly reward per-row directional/shape fidelity -- this term does.
# --------------------------------------------------------------------------- #
def masked_cosine_loss(pred: torch.Tensor, target: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """pred, target: (B, L, K). mask: (B, L), 1=observed, 0=missing.
    Returns mean_n [1 - cos_angle(h_gt_n, h_hat_n)] over every missing row
    in the batch -- same formula as the CD metric, just averaged over rows
    directly (flattened across B, L) rather than per-sample-then-averaged,
    consistent with how masked_mse_loss already flattens over B, L."""
    missing = (mask == 0)                      # (B, L)
    if missing.sum() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    pred_m = pred[missing]                      # (n_missing, K)
    tgt_m = target[missing]                      # (n_missing, K)
    p_norm = pred_m.norm(dim=-1).clamp(min=1e-8)
    t_norm = tgt_m.norm(dim=-1).clamp(min=1e-8)
    cos_sim = (pred_m * tgt_m).sum(dim=-1) / (p_norm * t_norm)
    return (1.0 - cos_sim).mean()


def combined_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                   cd_weight: float = 0.0) -> torch.Tensor:
    """masked_mse_loss + cd_weight * masked_cosine_loss. cd_weight=0.0
    (the default) reproduces masked_mse_loss exactly -- verified bit-for-bit
    identical, so existing runs are unaffected unless cd_weight is set."""
    mse = masked_mse_loss(pred, target, mask)
    if cd_weight > 0:
        cd = masked_cosine_loss(pred, target, mask)
        return mse + cd_weight * cd
    return mse

# --------------------------------------------------------------------------- #
# NMSE / CD metrics (Eq. 11), evaluated on missing positions only.
#
# Sec. 4.4: "we evaluate the NMSE and CD of RIR-Former across 10 simulated
# acoustic environments ... For each rate, we average the results across
# all environments." Eq. (11) itself is defined per single reconstruction
# (one environment); the paper then averages the resulting NMSE(dB)/CD
# values across environments. Because log10 doesn't commute with averaging,
# this is NOT the same as pooling every sample in a batch into one Frobenius
# ratio and converting that once -- so metrics are computed per sample here,
# then the (already-in-dB) values are averaged.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_metrics_per_sample(pred: torch.Tensor, target: torch.Tensor,
                                mask: torch.Tensor):
    """Same computation as `compute_metrics` but returns the raw per-sample
    lists instead of averaging them -- lets callers pool samples across
    multiple batches and average exactly once over all environments."""
    B = pred.shape[0]
    nmse_list, cd_list = [], []

    for b in range(B):
        missing_b = (mask[b] == 0)
        if missing_b.sum() == 0:
            continue

        pred_b = pred[b][missing_b]
        tgt_b = target[b][missing_b]

        num = torch.sum((tgt_b - pred_b) ** 2)
        den = torch.sum(tgt_b ** 2).clamp(min=1e-12)
        nmse_list.append((10.0 * torch.log10(num / den)).item())

        p_norm = pred_b.norm(dim=-1).clamp(min=1e-12)
        t_norm = tgt_b.norm(dim=-1).clamp(min=1e-12)
        cos_sim = (pred_b * tgt_b).sum(dim=-1) / (p_norm * t_norm)
        cd_list.append((1.0 - cos_sim).mean().item())

    return nmse_list, cd_list


@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor) -> dict:
    """Convenience wrapper: average over the samples/environments in this
    single batch. For averaging across multiple batches, use
    `compute_metrics_per_sample` and pool the lists instead (see `validate`
    and `evaluate.py`'s `evaluate_at_missing_rate`)."""
    nmse_list, cd_list = compute_metrics_per_sample(pred, target, mask)
    if len(nmse_list) == 0:
        return {"nmse_db": float("nan"), "cd": float("nan")}
    return {"nmse_db": float(np.mean(nmse_list)), "cd": float(np.mean(cd_list))}


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(model: nn.Module, cfg: Config, epoch: int, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "config": cfg.to_dict(),
    }, path)


def load_checkpoint(model: nn.Module, path: str, device: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    return ckpt


# --------------------------------------------------------------------------- #
# Train / validate one epoch
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optimizer, device, grad_clip: float,
                     only_segment: Optional[int] = None,
                     cd_loss_weight: float = 0.0) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        H_norm = batch["H_norm"].to(device)
        mask = batch["mask"].to(device)
        geo_feat = batch["geo_feat"].to(device)

        optimizer.zero_grad()

        if only_segment is None:
            pred = model(H_norm, mask, geo_feat)
            loss = combined_loss(pred, H_norm, mask, cd_weight=cd_loss_weight)
        elif not getattr(model, "use_residual_refine", False):
            # Default / released-model setting: no refine module, so each
            # branch's segment slice is placed into the final output via
            # plain concatenation with NO cross-branch mixing. That makes
            # the isolated, cheap `forward_segment_only` exactly equivalent
            # to slicing the full forward pass, but without the cost of
            # running the other (frozen, unrelated) branches every step.
            seg = model.segment_len
            s0, s1 = only_segment * seg, (only_segment + 1) * seg
            tgt_seg = H_norm[:, :, s0:s1]
            pred_seg = model.forward_segment_only(only_segment, H_norm, mask, geo_feat)
            loss = combined_loss(pred_seg, tgt_seg, mask, cd_weight=cd_loss_weight)
        else:
            # If the optional residual refine module IS enabled, it mixes
            # information across segment boundaries (temporal Conv1d over
            # the concatenated output), so it must stay in the autograd
            # graph during per-segment finetuning -- otherwise the branch
            # being tuned drifts away from what the (frozen, now-stale)
            # refine module expects, and quality can regress after
            # finetuning even though each branch's isolated segment loss
            # improves. Use the full forward pass and slice out this
            # segment instead of the cheap isolated one.
            seg = model.segment_len
            s0, s1 = only_segment * seg, (only_segment + 1) * seg
            pred_full = model(H_norm, mask, geo_feat)
            pred_seg = pred_full[:, :, s0:s1]
            tgt_seg = H_norm[:, :, s0:s1]
            loss = combined_loss(pred_seg, tgt_seg, mask, cd_weight=cd_loss_weight)

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device) -> dict:
    model.eval()
    nmse_list, cd_list = [], []

    for batch in loader:
        H_norm = batch["H_norm"].to(device)
        mask = batch["mask"].to(device)
        geo_feat = batch["geo_feat"].to(device)

        pred = model(H_norm, mask, geo_feat)
        n_list, c_list = compute_metrics_per_sample(pred, H_norm, mask)
        nmse_list.extend(n_list)
        cd_list.extend(c_list)

    if len(nmse_list) == 0:
        return {"nmse_db": float("nan"), "cd": float("nan")}
    return {"nmse_db": float(np.mean(nmse_list)), "cd": float(np.mean(cd_list))}


# --------------------------------------------------------------------------- #
# Main training entry point
# --------------------------------------------------------------------------- #
def run_training(cfg: Config, verbose: bool = True) -> str:
    device = torch.device(cfg.train.device)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    train_ds = build_rir_dataset(cfg, split="train",
                           mask_ratio_range=(cfg.train.min_mask_ratio, cfg.train.min_mask_ratio),
                           deterministic=False)
    val_ds = build_rir_dataset(cfg, split="val",
                         mask_ratio_range=(cfg.train.max_mask_ratio, cfg.train.max_mask_ratio),
                         deterministic=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=4, pin_memory=(device.type == "cuda"),)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                         collate_fn=collate_fn, num_workers=2, pin_memory=(device.type == "cuda"))

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                   weight_decay=cfg.train.weight_decay)
    scheduler = None
    if cfg.train.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.min_lr)

    best_nmse = float("inf")
    ckpt_dir = cfg.train.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "checkpoint_main.pth")

    # ---- Stage 1: main training with curriculum masking ---- #
    for epoch in range(cfg.train.epochs):
        ratio = mask_ratio_for_epoch(epoch, cfg)
        train_ds.set_mask_ratio_range(ratio, ratio)

        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                      cfg.train.grad_clip,
                                      cd_loss_weight=cfg.train.cd_loss_weight)

        if scheduler is not None:
            scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        if verbose and (epoch % cfg.train.log_every == 0 or epoch == cfg.train.epochs - 1):
            print(f"[train] epoch {epoch + 1}/{cfg.train.epochs} "
                  f"mask_ratio={ratio:.2f} loss={train_loss:.5f} "
                  f"lr={cur_lr:.2e} ({time.time() - t0:.2f}s)")

        if (epoch + 1) % cfg.train.val_every == 0 or epoch == cfg.train.epochs - 1:
            metrics = validate(model, val_loader, device)
            if verbose:
                print(f"[val]   epoch {epoch + 1}/{cfg.train.epochs} "
                      f"NMSE={metrics['nmse_db']:.3f} dB  CD={metrics['cd']:.4f}")
            if metrics["nmse_db"] < best_nmse:
                best_nmse = metrics["nmse_db"]
                save_checkpoint(model, cfg, epoch, best_path)

    # always keep a final checkpoint too, in case validation never improved
    final_path = os.path.join(ckpt_dir, "checkpoint_final.pth")
    save_checkpoint(model, cfg, cfg.train.epochs - 1, final_path)
    if not os.path.exists(best_path):
        save_checkpoint(model, cfg, cfg.train.epochs - 1, best_path)

    # `best_nmse`/`best_path` are updated after *every* stage below, so the
    # checkpoint this function returns is always the best validation result
    # seen across the whole run -- a later stage that happens to regress
    # (e.g. finetuning overfitting a segment) can never silently overwrite a
    # better earlier checkpoint.
    def _maybe_update_best(epoch_tag: int, stage_name: str):
        nonlocal best_nmse
        metrics = validate(model, val_loader, device)
        if verbose:
            print(f"[val after {stage_name}] NMSE={metrics['nmse_db']:.3f} dB  "
                  f"CD={metrics['cd']:.4f}")
        if metrics["nmse_db"] < best_nmse:
            best_nmse = metrics["nmse_db"]
            save_checkpoint(model, cfg, epoch_tag, best_path)
            if verbose:
                print(f"[checkpoint] new best ({best_nmse:.3f} dB) saved to {best_path}")
        return metrics

    # ---- Stage 2: per-segment decoder finetuning ---- #
    if cfg.train.finetune_epochs > 0:
        train_ds.set_mask_ratio_range(cfg.train.max_mask_ratio, cfg.train.max_mask_ratio)

        for seg_idx in range(model.n_segments):
            for p in model.parameters():
                p.requires_grad = False
            for p in model.branches[seg_idx].parameters():
                p.requires_grad = True

            seg_optimizer = torch.optim.AdamW(
                model.branches[seg_idx].parameters(), lr=cfg.train.lr,
                weight_decay=cfg.train.weight_decay)

            for epoch in range(cfg.train.finetune_epochs):
                # NOTE: train_one_epoch(..., only_segment=seg_idx) now runs
                # the *full* model forward (all decoder heads + refine) and only
                # back-propagates into branch[seg_idx]'s parameters -- see
                # the comment in train_one_epoch for why this matters.
                loss = train_one_epoch(model, train_loader, seg_optimizer, device,
                                        cfg.train.grad_clip, only_segment=seg_idx,
                                        cd_loss_weight=cfg.train.cd_loss_weight)
                if verbose and (epoch % cfg.train.log_every == 0
                                or epoch == cfg.train.finetune_epochs - 1):
                    print(f"[finetune seg {seg_idx}] epoch {epoch + 1}/"
                          f"{cfg.train.finetune_epochs} loss={loss:.5f}")

        for p in model.parameters():
            p.requires_grad = True

        finetuned_path = os.path.join(ckpt_dir, "checkpoint_finetuned.pth")
        save_checkpoint(model, cfg, cfg.train.epochs - 1, finetuned_path)
        _maybe_update_best(cfg.train.epochs - 1, "per-segment finetune")

    # ---- Stage 3: residual-refine adaptation ---- #
    # Per-segment finetuning (Stage 2) shifts each branch's output
    # distribution. If the model uses the residual refine module, that
    # module was last trained (in Stage 1) against the *pre*-finetune
    # branch outputs, so it's now slightly out of sync. Re-adapt it here
    # (encoder + all decoder heads frozen, only refine trainable) before
    # considering
    # training complete.
    if cfg.train.refine_finetune_epochs > 0 and getattr(model, "use_residual_refine", False):
        train_ds.set_mask_ratio_range(cfg.train.max_mask_ratio, cfg.train.max_mask_ratio)

        for p in model.parameters():
            p.requires_grad = False
        for p in model.refine.parameters():
            p.requires_grad = True

        refine_optimizer = torch.optim.AdamW(
            model.refine.parameters(), lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay)

        for epoch in range(cfg.train.refine_finetune_epochs):
            loss = train_one_epoch(model, train_loader, refine_optimizer, device,
                                    cfg.train.grad_clip,
                                    cd_loss_weight=cfg.train.cd_loss_weight)  # only_segment=None -> full loss
            if verbose and (epoch % cfg.train.log_every == 0
                            or epoch == cfg.train.refine_finetune_epochs - 1):
                print(f"[refine finetune] epoch {epoch + 1}/"
                      f"{cfg.train.refine_finetune_epochs} loss={loss:.5f}")

        for p in model.parameters():
            p.requires_grad = True

        refined_path = os.path.join(ckpt_dir, "checkpoint_refine_adapted.pth")
        save_checkpoint(model, cfg, cfg.train.epochs - 1, refined_path)
        _maybe_update_best(cfg.train.epochs - 1, "refine adaptation")

    if verbose:
        print(f"[train] Best checkpoint overall: {best_path} (NMSE={best_nmse:.3f} dB)")

    return best_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train RIR-Former")
    p.add_argument("--experiment", type=str, default="exp1", choices=["exp1", "exp2"])
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--finetune_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--lr_scheduler", action="store_true",
                    help="Enable cosine LR annealing (off by default).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--geo_attn_bias", action="store_true",
                    help="Enable Strategy-2 geometric attention bias (off by default).")
    p.add_argument("--cd_loss_weight", type=float, default=0.0,
                    help="Weight of an auxiliary cosine-similarity loss term "
                         "(0.0 = off, matches the released MSE-only objective).")
    return p


def main():
    args = build_arg_parser().parse_args()
    cfg = get_config(
        args.experiment,
        **{
            "data.data_root": args.data_root,
            "train.checkpoint_dir": args.checkpoint_dir,
            "train.epochs": args.epochs,
            "train.finetune_epochs": args.finetune_epochs,
            "train.batch_size": args.batch_size,
            "train.lr": args.lr,
            "train.min_lr": args.min_lr,
            "train.use_lr_scheduler": args.lr_scheduler,
            "train.device": args.device,
            "model.use_geo_attn_bias": args.geo_attn_bias,
            "train.cd_loss_weight": args.cd_loss_weight,
        },
    )
    run_training(cfg)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        # ------------------------------------------------------------------ #
        # Self-test: tiny end-to-end run (small dataset, few epochs) to make
        # sure the whole training pipeline (data -> model -> optimizer ->
        # checkpoint -> validation -> finetune stage) works without errors.
        # ------------------------------------------------------------------ #
        import shutil
        import tempfile

        print("Running train.py self-test (tiny end-to-end run)...")

        tmp_root = tempfile.mkdtemp(prefix="rirformer_train_")
        try:
            cfg = get_config(
                "exp1",
                **{
                    "data.data_root": tmp_root,
                    "train.checkpoint_dir": os.path.join(tmp_root, "ckpt"),
                    "train.epochs": 3,
                    "train.mask_warmup_epochs": 2,
                    "train.finetune_epochs": 1,
                    "train.batch_size": 2,
                    "train.log_every": 1,
                    "train.val_every": 1,
                    "model.d_model": 16,
                    "model.n_layers": 1,
                    "model.n_heads": 2,
                    "model.n_segments": 4,
                    "array.n_points": 8,
                },
            )

            print("Generating tiny synthetic train/val datasets...")
            generate_dataset(cfg, split="train", n_samples=6, seed=1, verbose=False)
            generate_dataset(cfg, split="val", n_samples=4, seed=2, verbose=False)

            assert mask_ratio_for_epoch(0, cfg) == cfg.train.min_mask_ratio
            assert abs(mask_ratio_for_epoch(cfg.train.mask_warmup_epochs, cfg)
                       - cfg.train.max_mask_ratio) < 1e-9
            mid = mask_ratio_for_epoch(1, cfg)
            assert cfg.train.min_mask_ratio < mid < cfg.train.max_mask_ratio
            print(f"[OK] mask_ratio_for_epoch curriculum: "
                  f"e0={mask_ratio_for_epoch(0, cfg):.2f}, e1={mid:.2f}, "
                  f"e2={mask_ratio_for_epoch(2, cfg):.2f}")

            # masked_mse_loss / compute_metrics sanity checks
            pred = torch.zeros(2, 4, 5)
            target = torch.ones(2, 4, 5)
            mask = torch.tensor([[1., 1., 0., 0.], [1., 0., 0., 0.]])
            K_test = pred.shape[-1]  # = 5
            loss = masked_mse_loss(pred, target, mask)
            # Eq. (10): L = (1/N) * sum_n sum_k (h_bar - h_hat)^2 -- summed
            # over K, only averaged over N missing rows. Here every missing
            # element has squared error 1, so loss == K (not 1.0, since we
            # deliberately do NOT also divide by K -- see Eq. 10 docstring).
            assert abs(loss.item() - K_test) < 1e-6, loss.item()
            print(f"[OK] masked_mse_loss on synthetic data: {loss.item():.4f} "
                  f"(expected K={K_test}, matches Eq. 10's /N-only normalization)")

            metrics = compute_metrics(pred, target, mask)
            assert abs(metrics["nmse_db"] - 0.0) < 1e-3, metrics  # ||pred-tgt||^2 == ||tgt||^2 here
            print(f"[OK] compute_metrics on synthetic data: {metrics}")

            best_path = run_training(cfg, verbose=True)
            assert os.path.exists(best_path), "Best checkpoint was not saved"
            print(f"[OK] Training completed, best checkpoint at: {best_path}")

            # reload checkpoint into a fresh model and confirm it works
            model2 = build_model(cfg)
            ckpt = load_checkpoint(model2, best_path, device="cpu")
            assert "epoch" in ckpt and "config" in ckpt
            print(f"[OK] Checkpoint reloads correctly (epoch={ckpt['epoch']}, "
                  f"config.experiment={ckpt['config']['experiment']})")

            val_ds = build_rir_dataset(cfg, split="val",
                                 mask_ratio_range=(0.7, 0.7), deterministic=True)
            from dataset_generator import collate_fn as cfn
            val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=cfn)
            metrics = validate(model2, val_loader, "cpu")
            assert not np.isnan(metrics["nmse_db"])
            print(f"[OK] Reloaded model validates: NMSE={metrics['nmse_db']:.3f} dB, "
                  f"CD={metrics['cd']:.4f}")

            finetuned_path = os.path.join(cfg.train.checkpoint_dir, "checkpoint_finetuned.pth")
            assert os.path.exists(finetuned_path), "Finetuned checkpoint was not saved"
            print(f"[OK] Finetuned checkpoint saved at: {finetuned_path}")

            print("\nAll train.py self-tests passed!")
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)