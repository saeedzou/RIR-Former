"""
evaluate.py
===========

Evaluation / inference script for RIR-Former, in the spirit of the official
`eval_rirformer.py` (single-file, loads a checkpoint, runs inference, does
not need training code) but extended to also **compute the metrics** used in
the paper (Sec. 4.3, Eq. 11):

    NMSE (dB) = 10 log10( ||H_gt - H_hat||_F^2 / ||H_gt||_F^2 )
    CD        = mean_n [ 1 - cos_angle(h_gt_n, h_hat_n) ]

evaluated only over the *missing* (reconstructed) array positions, at each
missing rate (MR) in {10%, ..., 90%} as in Fig. 4, and (optionally) at a
fixed MR=70% to reproduce a Table-1/2-style comparison summary.

Also reports average per-sample inference time, mirroring the "Inference"
column of Table 1 in the paper.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config, get_config
from dataset_generator import RIRDataset, collate_fn, generate_dataset
from model import RIRFormer, build_model
from train import compute_metrics_per_sample, load_checkpoint


# --------------------------------------------------------------------------- #
# Core evaluation loop for a single missing rate
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_at_missing_rate(model: RIRFormer, data_root: str, split: str,
                              missing_rate: float, batch_size: int,
                              device: str, mask_seed: int = 1234) -> Dict[str, float]:
    """Evaluates over every sample in `split` at a fixed missing rate.
    NMSE(dB)/CD are computed per sample (= per acoustic environment, Eq. 11)
    and then averaged across all environments, matching Sec. 4.4 ("we
    average the results across all environments") -- NOT pooled into one
    Frobenius ratio per batch, since log10 doesn't commute with averaging."""
    model.eval()
    ds = RIRDataset(data_root, split=split,
                     mask_ratio_range=(missing_rate, missing_rate),
                     deterministic=True, base_seed=mask_seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    nmse_list, cd_list, infer_times = [], [], []

    for batch in loader:
        H_norm = batch["H_norm"].to(device)
        mask = batch["mask"].to(device)
        geo_feat = batch["geo_feat"].to(device)

        t0 = time.time()
        pred = model(H_norm, mask, geo_feat)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        infer_times.append(elapsed / H_norm.shape[0])  # per-sample time

        n_list, c_list = compute_metrics_per_sample(pred, H_norm, mask)
        nmse_list.extend(n_list)
        cd_list.extend(c_list)

    return {
        "missing_rate": missing_rate,
        "nmse_db": float(np.mean(nmse_list)) if nmse_list else float("nan"),
        "cd": float(np.mean(cd_list)) if cd_list else float("nan"),
        "inference_time_s": float(np.mean(infer_times)) if infer_times else float("nan"),
        "n_samples": len(ds),
    }


def evaluate_all_rates(model: RIRFormer, cfg: Config, split: str = "test",
                        rates: List[float] = None, verbose: bool = True) -> List[Dict]:
    if rates is None:
        rates = cfg.data.eval_missing_rates

    device = cfg.train.device
    results = []
    for mr in rates:
        res = evaluate_at_missing_rate(model, cfg.data.data_root, split, mr,
                                        cfg.train.batch_size, device)
        results.append(res)
        if verbose:
            print(f"[eval] MR={mr * 100:5.1f}%  NMSE={res['nmse_db']:7.3f} dB  "
                  f"CD={res['cd']:.4f}  infer={res['inference_time_s'] * 1000:.3f} ms/sample")
    return results


def save_results_csv(results: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("missing_rate,nmse_db,cd,inference_time_s,n_samples\n")
        for r in results:
            f.write(f"{r['missing_rate']},{r['nmse_db']},{r['cd']},"
                     f"{r['inference_time_s']},{r['n_samples']}\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate RIR-Former")
    p.add_argument("--experiment", type=str, default="exp1", choices=["exp1", "exp2"])
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out_csv", type=str, default="eval_results.csv")
    p.add_argument("--fixed_mr", type=float, default=0.7,
                    help="Missing rate used for the single-number Table 1/2 style summary.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main():
    args = build_arg_parser().parse_args()
    cfg = get_config(args.experiment, **{
        "data.data_root": args.data_root,
        "train.batch_size": args.batch_size,
        "train.device": args.device,
    })

    model = build_model(cfg).to(args.device)
    load_checkpoint(model, args.ckpt_path, device=args.device)

    print(f"=== Sweeping missing rates on split='{args.split}' ===")
    results = evaluate_all_rates(model, cfg, split=args.split)
    save_results_csv(results, args.out_csv)
    print(f"Saved results to {args.out_csv}")

    print(f"\n=== Fixed MR={args.fixed_mr * 100:.0f}% summary (Table-style) ===")
    fixed = evaluate_at_missing_rate(model, cfg.data.data_root, args.split,
                                      args.fixed_mr, args.batch_size, args.device)
    print(f"Ours   NMSE={fixed['nmse_db']:.3f} dB  CD={fixed['cd']:.3f}  "
          f"Inference={fixed['inference_time_s']:.4f} s")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        # ------------------------------------------------------------------ #
        # Self-test: tiny synthetic test set + freshly-initialized (untrained)
        # model, just to confirm the full evaluation pipeline (data loading,
        # multi-missing-rate sweep, NMSE/CD metric computation, CSV export,
        # checkpoint load/save round trip) runs correctly end to end.
        # ------------------------------------------------------------------ #
        import shutil
        import tempfile

        print("Running evaluate.py self-test...")

        tmp_root = tempfile.mkdtemp(prefix="rirformer_eval_")
        try:
            cfg = get_config(
                "exp1",
                **{
                    "data.data_root": tmp_root,
                    "model.d_model": 16,
                    "model.n_layers": 1,
                    "model.n_heads": 2,
                    "model.n_segments": 4,
                    "array.n_points": 8,
                    "train.batch_size": 2,
                    "train.device": "cpu",
                    "data.eval_missing_rates": [0.3, 0.5, 0.7],
                },
            )

            print("Generating tiny synthetic test dataset...")
            generate_dataset(cfg, split="test", n_samples=4, seed=7, verbose=False)

            model = build_model(cfg)

            # save an (untrained) checkpoint and reload it, exactly as a real
            # trained checkpoint would be produced/consumed
            ckpt_path = os.path.join(tmp_root, "checkpoint_main.pth")
            torch.save({"model_state_dict": model.state_dict(), "epoch": 0,
                        "config": cfg.to_dict()}, ckpt_path)

            model2 = build_model(cfg)
            ckpt = load_checkpoint(model2, ckpt_path, device="cpu")
            assert ckpt["config"]["experiment"] == "exp1"
            print(f"[OK] Checkpoint saved & reloaded (epoch={ckpt['epoch']})")

            # single missing-rate evaluation
            single = evaluate_at_missing_rate(model2, tmp_root, "test", 0.7, 2, "cpu")
            assert single["n_samples"] == 4
            assert not np.isnan(single["nmse_db"])
            assert not np.isnan(single["cd"])
            assert single["inference_time_s"] >= 0
            print(f"[OK] evaluate_at_missing_rate(MR=0.7): NMSE={single['nmse_db']:.3f} dB, "
                  f"CD={single['cd']:.4f}, infer={single['inference_time_s'] * 1000:.3f} ms/sample")

            # sweep across multiple missing rates
            results = evaluate_all_rates(model2, cfg, split="test", verbose=True)
            assert len(results) == 3
            assert [r["missing_rate"] for r in results] == [0.3, 0.5, 0.7]
            for r in results:
                assert not np.isnan(r["nmse_db"]) and not np.isnan(r["cd"])
            print(f"[OK] evaluate_all_rates produced {len(results)} results")

            # CSV export
            csv_path = os.path.join(tmp_root, "results.csv")
            save_results_csv(results, csv_path)
            assert os.path.exists(csv_path)
            with open(csv_path) as f:
                lines = f.readlines()
            assert len(lines) == 1 + len(results)
            print(f"[OK] Results CSV written with {len(lines) - 1} data rows: {csv_path}")

            # sanity: a model that reconstructs everything perfectly should
            # give NMSE -> -inf (very negative) and CD -> 0
            class PerfectModel(torch.nn.Module):
                def forward(self, H_norm, mask, geo_feat):
                    return H_norm.clone()

            perfect = PerfectModel()
            perfect_res = evaluate_at_missing_rate(perfect, tmp_root, "test", 0.5, 2, "cpu")
            assert perfect_res["nmse_db"] < -50, perfect_res
            assert abs(perfect_res["cd"]) < 1e-4, perfect_res
            print(f"[OK] Perfect-reconstruction sanity check: "
                  f"NMSE={perfect_res['nmse_db']:.1f} dB, CD={perfect_res['cd']:.6f}")

            print("\nAll evaluate.py self-tests passed!")
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)