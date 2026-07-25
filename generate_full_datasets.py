"""
generate_full_datasets.py
==========================

Paper-scale dataset generation driver.

`dataset_generator.generate_dataset()` is single-threaded (one `rir_generator`
call per room). That's fine for the tiny datasets used in the file self-tests,
but generating the paper's full Monte-Carlo dataset -- 8000 train rooms + 200
val rooms + 10 test rooms (Sec. 4.1) -- is much faster done in parallel across
CPU cores, since each room's simulation is fully independent.

This script fans `generate_one_sample()` out across a process pool and writes
the same `.npy` sample format consumed by `dataset_generator.RIRDataset`, so
its output directory can be passed straight to `train.py` / `evaluate.py`
via `--data_root`.

Usage:
    python generate_full_datasets.py --experiment exp1 --data_root data/exp1
    python generate_full_datasets.py --experiment exp2 --data_root data/exp2

    # override sizes / worker count
    python generate_full_datasets.py --experiment exp1 \
        --n_train 8000 --n_val 200 --n_test 10 --workers 8
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from config import Config, get_config
from dataset_generator import generate_one_sample


def _generate_and_save(args):
    """Runs in a worker process: simulate one room and save it to disk."""
    cfg, seed, idx, out_dir = args
    rng = np.random.default_rng(seed + idx)
    sample = generate_one_sample(rng, cfg)
    path = os.path.join(out_dir, f"{idx:06d}.npy")
    np.save(path, sample, allow_pickle=True)
    return idx


def generate_split_parallel(cfg: Config, split: str, n_samples: int, seed: int,
                             workers: int, verbose: bool = True) -> str:
    out_dir = os.path.join(cfg.data.data_root, split)
    os.makedirs(out_dir, exist_ok=True)

    tasks = [(cfg, seed, idx, out_dir) for idx in range(n_samples)]
    t0 = time.time()
    done = 0

    if workers <= 1:
        for t in tasks:
            _generate_and_save(t)
            done += 1
            if verbose and done % max(1, n_samples // 20) == 0:
                print(f"[{split}] {done}/{n_samples} "
                      f"({time.time() - t0:.1f}s elapsed)")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_generate_and_save, t) for t in tasks]
            for fut in as_completed(futures):
                fut.result()  # re-raise any worker exception
                done += 1
                if verbose and done % max(1, n_samples // 20) == 0:
                    print(f"[{split}] {done}/{n_samples} "
                          f"({time.time() - t0:.1f}s elapsed)")

    if verbose:
        print(f"[{split}] done: {n_samples} samples in {time.time() - t0:.1f}s "
              f"-> {out_dir}")
    return out_dir


def generate_all_splits(cfg: Config, workers: int, verbose: bool = True):
    generate_split_parallel(cfg, "train", cfg.data.n_train_rooms,
                             seed=cfg.train.seed, workers=workers, verbose=verbose)
    generate_split_parallel(cfg, "val", cfg.data.n_val_rooms,
                             seed=cfg.train.seed + 1, workers=workers, verbose=verbose)
    generate_split_parallel(cfg, "test", cfg.data.n_test_rooms,
                             seed=cfg.train.seed + 2, workers=workers, verbose=verbose)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate full RIR-Former datasets")
    p.add_argument("--experiment", type=str, default="exp1", choices=["exp1", "exp2"])
    p.add_argument("--data_root", type=str, default=None,
                    help="Defaults to 'data/<experiment>' if not given.")
    p.add_argument("--n_train", type=int, default=8000)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--n_test", type=int, default=10)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_arg_parser().parse_args()
    data_root = args.data_root or os.path.join("data", args.experiment)

    cfg = get_config(
        args.experiment,
        **{
            "data.data_root": data_root,
            "data.n_train_rooms": args.n_train,
            "data.n_val_rooms": args.n_val,
            "data.n_test_rooms": args.n_test,
            "train.seed": args.seed,
        },
    )

    print(f"Generating '{args.experiment}' dataset at '{data_root}' "
          f"(train={args.n_train}, val={args.n_val}, test={args.n_test}, "
          f"workers={args.workers})")
    generate_all_splits(cfg, workers=args.workers)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        import shutil
        import tempfile

        print("Running generate_full_datasets.py self-test...")
        tmp_root = tempfile.mkdtemp(prefix="rirformer_fulldata_")
        try:
            cfg = get_config("exp1", **{
                "data.data_root": tmp_root,
                "array.n_points": 8,
            })

            # sequential
            out_dir = generate_split_parallel(cfg, "train", n_samples=6, seed=0,
                                               workers=1, verbose=False)
            files = sorted(os.listdir(out_dir))
            assert len(files) == 6
            print(f"[OK] sequential generation: {len(files)} files in {out_dir}")

            # parallel (workers>1 even though this sandbox only has 1 CPU --
            # ProcessPoolExecutor still works correctly with 1 or more workers)
            out_dir2 = generate_split_parallel(cfg, "val", n_samples=6, seed=1,
                                                workers=2, verbose=False)
            files2 = sorted(os.listdir(out_dir2))
            assert len(files2) == 6
            print(f"[OK] parallel generation (workers=2): {len(files2)} files in {out_dir2}")

            # sanity: sequential and parallel generation of the *same* seeded
            # indices produce identical samples (determinism is per-index seed,
            # not affected by worker scheduling order)
            out_dir3a = generate_split_parallel(cfg, "check_a", n_samples=4, seed=42,
                                                 workers=1, verbose=False)
            out_dir3b = generate_split_parallel(cfg, "check_b", n_samples=4, seed=42,
                                                 workers=2, verbose=False)
            a = np.load(os.path.join(out_dir3a, "000002.npy"), allow_pickle=True).item()
            b = np.load(os.path.join(out_dir3b, "000002.npy"), allow_pickle=True).item()
            assert np.allclose(a["rir"], b["rir"])
            assert np.allclose(a["mic_positions"], b["mic_positions"])
            print("[OK] sequential vs parallel generation produce identical samples "
                  "for the same index/seed")

            print("\nAll generate_full_datasets.py self-tests passed!")
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)