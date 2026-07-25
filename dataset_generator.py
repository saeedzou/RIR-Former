"""
dataset_generator.py
=====================

Generates the Monte-Carlo shoebox-room dataset described in Sec. 4.1 of the
RIR-Former paper, using `rir-generator` (https://github.com/audiolabs/rir-generator,
the Python port of Habets' image-method RIR generator, also cited as [30] in
the paper) for the RIR simulation itself.

For every simulated room we:
  1. sample room dimensions L=(Lx,Ly,Lz) and RT60,
  2. place L = M+N array points (either a fixed on-grid ULA for Experiment 1,
     or a randomly placed/spaced/oriented RSLA with random source for
     Experiment 2), all coplanar within a square ROI centered at the room's
     acoustic "origin" O,
  3. call `rir_generator.generate(...)` once for the *whole* array (a single
     call is far more efficient than one call per microphone), obtaining a
     (K, L) array of RIRs,
  4. store the *relative* (ROI-centered) microphone coordinates together with
     the RIRs -- the actual M/N split into "observed" vs "missing" points is
     performed later, at data-loading time (mirrors the official evaluation
     script `eval_rirformer.py`, which also keeps the full array in each
     sample file and builds the observation mask inside the Dataset).

Each sample is saved as a single `.npy` file containing a pickled dict:
    {
        "rir":            (L, K) float32,
        "mic_positions":  (L, 3) float32,   # relative to ROI center O
        "src_position":   (3,)  float32,   # relative to ROI center O
        "room_dims":      (3,)  float32,
        "rt60":           float,
    }

A `RIRDataset` (torch.utils.data.Dataset) is also provided for training /
evaluation: it loads these files, builds a (curriculum-able) random
observation mask, and returns per-sample-normalized tensors ready for
`model.py`.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import rir_generator as rir

from config import Config, get_config

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None
# --------------------------------------------------------------------------- #
# Geometry sampling
# --------------------------------------------------------------------------- #
def _sample_room_dims_and_rt60(rng: np.random.Generator, cfg: Config):
    Lx = rng.uniform(*cfg.room.length_range)
    Ly = rng.uniform(*cfg.room.width_range)
    Lz = rng.uniform(*cfg.room.height_range)
    rt60 = rng.uniform(*cfg.room.rt60_range)
    return np.array([Lx, Ly, Lz], dtype=np.float64), float(rt60)


def _sample_geometry_exp1(rng: np.random.Generator, cfg: Config):
    """Experiment 1: fixed source, on-grid ULA, fixed array center."""
    arr = cfg.array
    array_len = rng.uniform(*arr.array_length_range)

    n = arr.n_points
    offsets = np.linspace(-array_len / 2, array_len / 2, n)

    center = np.array(arr.fixed_array_center, dtype=np.float64)
    mic_rel = np.tile(center, (n, 1))
    axis_idx = {"x": 0, "y": 1, "z": 2}[arr.fixed_axis]
    mic_rel[:, axis_idx] += offsets

    src_rel = np.array(arr.fixed_source_pos, dtype=np.float64)
    return mic_rel, src_rel


def _sample_geometry_exp2(rng: np.random.Generator, cfg: Config):
    """Experiment 2: random source, grid-free RSLA (random length, spacing,
    orientation, position within the ROI)."""
    arr = cfg.array
    n = arr.n_points
    half_roi = arr.roi_size / 2.0
    array_len = rng.uniform(*arr.array_length_range)

    axis = rng.choice(["x", "y"])
    axis_idx = {"x": 0, "y": 1}[axis]
    other_idx = 1 - axis_idx

    # random, monotonically increasing spacing along the array axis
    raw = rng.uniform(0.2, 1.0, size=n)
    cum = np.cumsum(raw)
    cum -= cum.min()
    cum /= cum.max()
    offsets = (cum - 0.5) * array_len  # centered, spans [-len/2, len/2]

    # random center position of the array within the ROI (keep it inside)
    margin = min(array_len / 2.0, half_roi * 0.9)
    center_along = rng.uniform(-(half_roi - margin), (half_roi - margin)) \
        if half_roi - margin > 0 else 0.0
    center_perp = rng.uniform(-half_roi * 0.9, half_roi * 0.9)

    mic_rel = np.zeros((n, 3), dtype=np.float64)
    mic_rel[:, axis_idx] = offsets + center_along
    mic_rel[:, other_idx] = center_perp
    mic_rel[:, 2] = 0.0

    # random source position within the ROI
    src_rel = np.array([
        rng.uniform(-half_roi * 0.9, half_roi * 0.9),
        rng.uniform(-half_roi * 0.9, half_roi * 0.9),
        0.0,
    ], dtype=np.float64)

    return mic_rel, src_rel


def _to_absolute(rel_positions: np.ndarray, room_dims: np.ndarray) -> np.ndarray:
    """Map ROI-centered relative coordinates to absolute room coordinates.
    The ROI plane is placed at the room's horizontal & vertical center."""
    room_center = room_dims / 2.0
    return rel_positions + room_center[None, :]


def _within_room_with_margin(abs_positions: np.ndarray, room_dims: np.ndarray,
                              margin: float) -> bool:
    lo = abs_positions.min(axis=0)
    hi = abs_positions.max(axis=0)
    if np.any(lo < margin):
        return False
    if np.any(hi > (room_dims - margin)):
        return False
    return True


def generate_one_sample(rng: np.random.Generator, cfg: Config,
                         max_retries: int = 50) -> Dict[str, np.ndarray]:
    """Simulate a single room + array + RIRs. Retries with fresh geometry /
    room dims if the sampled positions would fall too close to a wall."""
    sampler = _sample_geometry_exp1 if cfg.experiment == "exp1" else _sample_geometry_exp2
    K = cfg.K

    for _ in range(max_retries):
        room_dims, rt60 = _sample_room_dims_and_rt60(rng, cfg)
        mic_rel, src_rel = sampler(rng, cfg)

        mic_abs = _to_absolute(mic_rel, room_dims)
        src_abs = _to_absolute(src_rel[None, :], room_dims)[0]

        all_pts = np.concatenate([mic_abs, src_abs[None, :]], axis=0)
        if not _within_room_with_margin(all_pts, room_dims, cfg.room.wall_margin):
            continue

        h = rir.generate(
            c=cfg.room.speed_of_sound,
            fs=cfg.data.fs,
            r=mic_abs.tolist(),
            s=src_abs.tolist(),
            L=room_dims.tolist(),
            reverberation_time=rt60,
            nsample=K,
        )  # shape (K, n_points)

        rir_arr = np.asarray(h, dtype=np.float32).T  # (n_points, K)

        return {
            "rir": rir_arr,
            "mic_positions": mic_rel.astype(np.float32),
            "src_position": src_rel.astype(np.float32),
            "room_dims": room_dims.astype(np.float32),
            "rt60": np.float32(rt60),
        }

    raise RuntimeError("Failed to sample a valid room geometry after "
                        f"{max_retries} retries.")


# --------------------------------------------------------------------------- #
# Dataset generation (simulate & save to disk)
# --------------------------------------------------------------------------- #
def generate_dataset(cfg: Config, split: str, n_samples: int,
                      seed: Optional[int] = None, verbose: bool = True) -> str:
    """Simulate `n_samples` rooms and save them under
    `<cfg.data.data_root>/<split>/*.npy`. Returns the output directory."""
    if seed is None:
        seed = cfg.train.seed
    rng = np.random.default_rng(seed + hash(split) % 10_000)

    out_dir = os.path.join(cfg.data.data_root, split)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(n_samples):
        sample = generate_one_sample(rng, cfg)
        path = os.path.join(out_dir, f"{i:06d}.npy")
        np.save(path, sample, allow_pickle=True)
        if verbose and (i + 1) % max(1, n_samples // 10) == 0:
            print(f"[dataset_generator] {split}: {i + 1}/{n_samples} samples generated")

    return out_dir


# --------------------------------------------------------------------------- #
# PyTorch Dataset
# --------------------------------------------------------------------------- #
try:
    import torch
    from torch.utils.data import Dataset

    class RIRDataset(Dataset):
        """
        Loads simulated `.npy` samples (full array of L RIRs + positions) and
        produces per-sample observation masks + normalized tensors.

        mask[i] == 1  -> point i is an *observed* microphone
        mask[i] == 0  -> point i is a *missing* target to be reconstructed

        If `deterministic=True`, the mask is derived from a fixed per-sample
        seed (used for reproducible evaluation across missing rates).
        Otherwise a fresh random mask ratio (sampled uniformly from
        `mask_ratio_range`) and split is drawn every time `__getitem__` is
        called (used for training, to realize the curriculum masking scheme
        described in Sec. 3 "Training Objective").
        """

        def __init__(self, root_dir: Optional[str], split: str,
                    mask_ratio_range=(0.7, 0.7), deterministic=False, base_seed=1234,
                    hub_repo_id: Optional[str] = None,
                    hub_config_name: Optional[str] = None):
            self.mask_ratio_range = mask_ratio_range
            self.deterministic = deterministic
            self.base_seed = base_seed
            self.source = "hub" if hub_repo_id else "local"

            if self.source == "hub":
                if load_dataset is None:
                    raise ImportError(
                        "`datasets` package required for hub loading: pip install datasets"
                    )
                self.hf_ds = load_dataset(hub_repo_id, hub_config_name, split=split)
                self.hf_ds.set_format(
                    type="numpy",
                    columns=["rir", "mic_positions", "src_position", "room_dims", "rt60"],
                )
                self.files = None
            else:
                self.dir = os.path.join(root_dir, split)
                self.files = sorted(glob.glob(os.path.join(self.dir, "*.npy")))
                if len(self.files) == 0:
                    raise RuntimeError(f"No .npy files found in: {self.dir}")

        def __len__(self):
            return len(self.hf_ds) if self.source == "hub" else len(self.files)

        def _load_raw(self, idx: int):
            if self.source == "hub":
                row = self.hf_ds[idx]
                sample = {
                    "rir": np.asarray(row["rir"], dtype=np.float32),
                    "mic_positions": np.asarray(row["mic_positions"], dtype=np.float32),
                    "src_position": np.asarray(row["src_position"], dtype=np.float32),
                    "room_dims": np.asarray(row["room_dims"], dtype=np.float32),
                    "rt60": np.float32(row["rt60"]),
                }
                path = f"hub://{idx}"
            else:
                path = self.files[idx]
                sample = np.load(path, allow_pickle=True).item()
            return sample, path

        def set_mask_ratio_range(self, lo: float, hi: float):
            self.mask_ratio_range = (lo, hi)

        def _make_mask(self, n_points: int, rng: np.random.Generator) -> np.ndarray:
            ratio = rng.uniform(*self.mask_ratio_range)
            n_missing = int(round(ratio * n_points))
            n_missing = min(max(n_missing, 1), n_points - 1)
            mask = np.ones(n_points, dtype=np.float32)
            missing_idx = rng.choice(n_points, size=n_missing, replace=False)
            mask[missing_idx] = 0.0
            return mask

        def __getitem__(self, idx: int):
            sample, path = self._load_raw(idx)

            H = sample["rir"].astype(np.float32)                # (L, K)
            geo = sample["mic_positions"].astype(np.float32)    # (L, 3)

            rng = (np.random.default_rng(self.base_seed + idx)
                   if self.deterministic else np.random.default_rng())
            mask_np = self._make_mask(H.shape[0], rng)

            H_t = torch.from_numpy(H)
            mask_t = torch.from_numpy(mask_np)

            # Per-sample amplitude normalisation, matching the official
            # eval_rirformer.py exactly: the norm is the peak absolute
            # amplitude of the MISSING/target rows only (mask==0), NOT the
            # whole array. `(1 - mask)` is 1 at missing positions, 0 at
            # observed ones, so `H_t * (1 - mask)` zeroes out the observed
            # rows before taking the max.
            norm = torch.abs(H_t * (1 - mask_t).unsqueeze(-1)).max().clamp(min=1e-8)
            H_norm = H_t / norm

            # Geometry preprocessing, matching the official
            # eval_rirformer.py exactly: shift all coordinates by the
            # sample's single global minimum (one scalar over the whole
            # (L, 3) array, not a per-axis shift) so that "Same geometry
            # preprocessing as training" is reproduced identically.
            geo_shifted = geo - geo.min()

            return {
                "H_norm": H_norm,
                "H_gt": H_norm,
                "norm": norm,
                "mask": mask_t,
                "geo_feat": torch.from_numpy(geo_shifted),
                "path": path,
            }

    def collate_fn(batch_list: List[dict]) -> dict:
        out = {}
        for k in batch_list[0]:
            if k == "path":
                out[k] = [b[k] for b in batch_list]
            else:
                out[k] = torch.stack([b[k] for b in batch_list])
        return out

    def build_rir_dataset(cfg: Config, split: str, **kwargs) -> "RIRDataset":
        """Picks local .npy files or the HF Hub dataset based on cfg.data.source."""
        if cfg.data.source == "hub":
            return RIRDataset(
                root_dir=None, split=split,
                hub_repo_id=cfg.data.hub_repo_id,
                hub_config_name=cfg.data.hub_config_name or cfg.experiment,
                **kwargs,
            )
        return RIRDataset(cfg.data.data_root, split=split, **kwargs)

except ImportError:  # torch not available -- generation still works
    RIRDataset = None
    collate_fn = None


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import shutil
    import tempfile

    print("Running dataset_generator.py self-test...")

    tmp_root = tempfile.mkdtemp(prefix="rirformer_data_")
    try:
        for exp in ("exp1", "exp2"):
            print(f"\n--- Experiment: {exp} ---")
            cfg = get_config(exp, **{"data.data_root": tmp_root})

            rng = np.random.default_rng(0)
            sample = generate_one_sample(rng, cfg)

            L = cfg.array.n_points
            K = cfg.K
            assert sample["rir"].shape == (L, K), sample["rir"].shape
            assert sample["mic_positions"].shape == (L, 3)
            assert sample["src_position"].shape == (3,)
            assert np.isfinite(sample["rir"]).all(), "RIR contains non-finite values"
            assert np.abs(sample["rir"]).max() > 0, "RIR is all-zero"
            print(f"[OK] single sample: rir{sample['rir'].shape}, "
                  f"rt60={sample['rt60']:.3f}, room_dims={sample['room_dims']}")

            # small dataset generation
            out_dir = generate_dataset(cfg, split="test_gen", n_samples=4,
                                        seed=42, verbose=False)
            files = sorted(glob.glob(os.path.join(out_dir, "*.npy")))
            assert len(files) == 4
            reloaded = np.load(files[0], allow_pickle=True).item()
            assert reloaded["rir"].shape == (L, K)
            print(f"[OK] generated & reloaded {len(files)} dataset files at {out_dir}")

            if RIRDataset is not None:
                ds = RIRDataset(tmp_root, split="test_gen",
                                 mask_ratio_range=(0.7, 0.7), deterministic=True)
                assert len(ds) == 4
                item = ds[0]
                assert item["H_norm"].shape == (L, K)
                assert item["mask"].shape == (L,)
                n_missing = (item["mask"] == 0).sum().item()
                assert n_missing == round(0.7 * L), n_missing
                # reproducibility check
                item2 = ds[0]
                assert torch.equal(item["mask"], item2["mask"]), \
                    "Deterministic dataset should produce identical masks"
                print(f"[OK] RIRDataset: H_norm{tuple(item['H_norm'].shape)}, "
                      f"missing={n_missing}/{L} (deterministic & reproducible)")

                # curriculum ratio range test (non-deterministic)
                ds2 = RIRDataset(tmp_root, split="test_gen",
                                  mask_ratio_range=(0.3, 0.3), deterministic=False)
                item3 = ds2[0]
                n_missing3 = (item3["mask"] == 0).sum().item()
                assert n_missing3 == round(0.3 * L), n_missing3
                print(f"[OK] mask ratio control works: missing={n_missing3}/{L} at ratio=0.3")

                from torch.utils.data import DataLoader
                loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
                batch = next(iter(loader))
                assert batch["H_norm"].shape == (2, L, K)
                assert batch["mask"].shape == (2, L)
                print(f"[OK] DataLoader batch shapes: H_norm{tuple(batch['H_norm'].shape)}, "
                      f"mask{tuple(batch['mask'].shape)}")

                # --- Verify preprocessing matches the official
                # eval_rirformer.py exactly (normalization by missing-only
                # peak amplitude, and geometry shifted by the sample's
                # global min). ---
                raw = np.load(ds.files[0], allow_pickle=True).item()
                H_raw = raw["rir"].astype(np.float32)
                geo_raw = raw["mic_positions"].astype(np.float32)
                mask_np = item["mask"].numpy()

                expected_norm = np.abs(H_raw * (1 - mask_np)[:, None]).max()
                assert abs(item["norm"].item() - expected_norm) < 1e-4, \
                    (item["norm"].item(), expected_norm)
                whole_array_max = np.abs(H_raw).max()
                print(f"[OK] Normalization uses missing-only peak amplitude "
                      f"({item['norm'].item():.6f}), matching eval_rirformer.py exactly "
                      f"(whole-array max is {whole_array_max:.6f} -- may coincide by chance "
                      f"in small samples, but the computation itself uses missing rows only)")

                expected_geo = geo_raw - geo_raw.min()
                assert np.allclose(item["geo_feat"].numpy(), expected_geo, atol=1e-6)
                assert abs(item["geo_feat"].numpy().min()) < 1e-6, \
                    "Shifted geometry should have a min of exactly 0"
                print(f"[OK] Geometry shifted by per-sample global min "
                      f"(new min={item['geo_feat'].numpy().min():.6f}), matching "
                      f"eval_rirformer.py exactly")
            else:
                print("[SKIP] torch not installed -- RIRDataset tests skipped")

        print("\nAll dataset_generator.py self-tests passed!")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)