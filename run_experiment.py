"""
run_experiment.py
==================

Top-level orchestrator that reproduces Experiment 1 or Experiment 2 from the
paper end to end:

    1. generate_full_datasets.py  -> simulate 8000 train / 200 val / 10 test
                                      shoebox rooms with rir_generator
    2. train.py                    -> main curriculum-masked training run
                                      (200 epochs) + per-segment decoder
                                      finetuning (20 epochs/segment)
    3. evaluate.py                 -> NMSE/CD sweep over missing rates
                                      10%-90% + fixed-MR=70% summary table

Each stage can also be run standalone (they're separate scripts/CLIs), this
just chains them with paper-consistent defaults and prints a final summary.

Examples
--------
Full paper-scale run of Experiment 1 on a single GPU:
    python run_experiment.py --experiment exp1 --device cuda

Full paper-scale run of Experiment 2:
    python run_experiment.py --experiment exp2 --device cuda

Quick smoke run (tiny dataset/model/epochs) to sanity check your setup
before committing to the full ~8000-room / 200-epoch run:
    python run_experiment.py --experiment exp1 --quick

Resume without regenerating data you already have on disk:
    python run_experiment.py --experiment exp1 --skip_data_gen
"""

from __future__ import annotations

import argparse
import os
import time

from config import get_config
from generate_full_datasets import generate_all_splits
from train import run_training
from evaluate import evaluate_all_rates, evaluate_at_missing_rate, save_results_csv
from model import build_model
from train import load_checkpoint


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a full RIR-Former experiment "
                                             "(data generation + training + evaluation)")
    p.add_argument("--experiment", type=str, required=True, choices=["exp1", "exp2"])
    p.add_argument("--data_root", type=str, default=None,
                    help="Defaults to 'data/<experiment>'.")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                    help="Defaults to 'checkpoints/<experiment>'.")
    p.add_argument("--results_csv", type=str, default=None,
                    help="Defaults to 'results_<experiment>.csv'.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="Parallel worker processes for RIR simulation.")

    p.add_argument("--source", choices=["local", "hub"], default="local")
    p.add_argument("--hub_repo_id", type=str, default="saeedzou/rir-former-datasets")
    p.add_argument("--hub_config_name", type=str, default=None)

    # allow overriding paper defaults if you have limited compute
    p.add_argument("--n_train", type=int, default=8000)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--n_test", type=int, default=10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--finetune_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--lr_scheduler", action="store_true",
                    help="Enable cosine LR annealing (off by default).")

    p.add_argument("--skip_data_gen", action="store_true",
                    help="Reuse an already-generated dataset at --data_root.")
    p.add_argument("--skip_train", action="store_true",
                    help="Skip training, only run evaluation "
                         "(requires --ckpt_path or an existing checkpoint_finetuned.pth).")
    p.add_argument("--ckpt_path", type=str, default=None,
                    help="Checkpoint to evaluate; defaults to the finetuned "
                         "checkpoint produced by this run.")

    p.add_argument("--quick", action="store_true",
                    help="Tiny smoke-test settings (small dataset/model/epochs) "
                         "to validate the whole pipeline quickly before committing "
                         "to the full paper-scale run.")
    return p


def main():
    args = build_arg_parser().parse_args()

    data_root = args.data_root or os.path.join("data", args.experiment)
    checkpoint_dir = args.checkpoint_dir or os.path.join("checkpoints", args.experiment)
    results_csv = args.results_csv or f"results_{args.experiment}.csv"

    overrides = {
        "data.data_root": data_root,
        "data.source": args.source,
        "data.hub_repo_id": args.hub_repo_id,
        "data.hub_config_name": args.hub_config_name,
        "train.checkpoint_dir": checkpoint_dir,
        "train.device": args.device,
        "train.batch_size": args.batch_size,
        "train.lr": args.lr,
        "train.min_lr": args.min_lr,
        "train.use_lr_scheduler": args.lr_scheduler,
        "train.epochs": args.epochs,
        "train.finetune_epochs": args.finetune_epochs,
        "data.n_train_rooms": args.n_train,
        "data.n_val_rooms": args.n_val,
        "data.n_test_rooms": args.n_test,
    }

    if args.quick:
        # Small enough to finish in a couple of minutes on CPU, just to
        # confirm the end-to-end pipeline (sim -> train -> eval) works on
        # your machine before launching the full paper-scale run.
        overrides.update({
            "data.n_train_rooms": 40,
            "data.n_val_rooms": 10,
            "data.n_test_rooms": 4,
            "train.epochs": 3,
            "train.mask_warmup_epochs": 2,
            "train.finetune_epochs": 1,
            "train.batch_size": 4,
            "array.n_points": 16,
            "model.d_model": 64,
            "model.n_layers": 2,
        })

    cfg = get_config(args.experiment, **overrides)

    print("=" * 70)
    print(f"RIR-Former pipeline: experiment={args.experiment}  quick={args.quick}")
    print(f"  data_root       = {data_root}")
    print(f"  checkpoint_dir  = {checkpoint_dir}")
    print(f"  K (RIR length)  = {cfg.K}")
    print(f"  rooms           = train={cfg.data.n_train_rooms}, "
          f"val={cfg.data.n_val_rooms}, test={cfg.data.n_test_rooms}")
    print(f"  epochs          = {cfg.train.epochs} main + "
          f"{cfg.train.finetune_epochs}/segment finetune")
    print(f"  device          = {cfg.train.device}")
    print("=" * 70)

    # ---- Stage 1: dataset generation ---- #
    if not args.skip_data_gen and cfg.data.source == "local":
        t0 = time.time()
        generate_all_splits(cfg, workers=args.workers)
    elif cfg.data.source == "hub":
        print(f"[pipeline] Using Hugging Face Hub dataset "
            f"{cfg.data.hub_repo_id} (config={cfg.data.hub_config_name or cfg.experiment})")
    else:
        print("[pipeline] Skipping data generation (--skip_data_gen)")

    # ---- Stage 2: training ---- #
    # run_training() returns the checkpoint with the best validation NMSE
    # seen across *all* stages (main training, per-segment finetuning,
    # refine adaptation) -- not necessarily the last stage's checkpoint.
    if not args.skip_train:
        t0 = time.time()
        eval_ckpt = run_training(cfg, verbose=True)
        print(f"[pipeline] Training finished in {time.time() - t0:.1f}s")
    else:
        print("[pipeline] Skipping training (--skip_train)")
        eval_ckpt = args.ckpt_path or os.path.join(checkpoint_dir, "checkpoint_main.pth")

    if args.ckpt_path:
        eval_ckpt = args.ckpt_path

    # ---- Stage 3: evaluation ---- #
    print(f"[pipeline] Evaluating checkpoint: {eval_ckpt}")
    model = build_model(cfg).to(cfg.train.device)
    load_checkpoint(model, eval_ckpt, device=cfg.train.device)

    results = evaluate_all_rates(model, cfg, split="test")
    save_results_csv(results, results_csv)
    print(f"[pipeline] Full missing-rate sweep saved to {results_csv}")

    fixed = evaluate_at_missing_rate(model, cfg, "test", 0.7,
                                  cfg.train.batch_size, cfg.train.device)
    print("\n" + "=" * 70)
    print(f"FINAL SUMMARY [{args.experiment}] @ MR=70% "
          f"(compare against Table 1/2 in the paper)")
    print(f"  NMSE = {fixed['nmse_db']:.3f} dB")
    print(f"  CD   = {fixed['cd']:.3f}")
    print(f"  Inference time = {fixed['inference_time_s'] * 1000:.2f} ms/sample")
    print("=" * 70)


if __name__ == "__main__":
    main()