"""
config.py
=========

Central configuration for the RIR-Former reimplementation
(Xu et al., "RIR-Former: Coordinate-Guided Transformer for Continuous
Reconstruction of Room Impulse Responses", arXiv:2602.01861).

All hyper-parameters that appear in the paper (Sec. 3 "Proposed Method" and
Sec. 4.1 "Experiment Setup") are collected here as dataclasses so that
`dataset_generator.py`, `model.py`, `train.py` and `evaluate.py` all share a
single source of truth.

Two experiment presets are provided, matching the paper:
    * "exp1": fixed source, on-grid Uniform Linear Array  (ULA)
    * "exp2": random source, grid-free Random-Spacing Linear Array (RSLA)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List


# --------------------------------------------------------------------------- #
# Room / acoustic simulation
# --------------------------------------------------------------------------- #
@dataclass
class RoomConfig:
    # Room length/width ~ U(4, 8) m, height ~ U(2.5, 4) m, RT60 ~ U(0.2, 0.8) s
    length_range: Tuple[float, float] = (4.0, 8.0)
    width_range: Tuple[float, float] = (4.0, 8.0)
    height_range: Tuple[float, float] = (2.5, 4.0)
    rt60_range: Tuple[float, float] = (0.2, 0.8)
    speed_of_sound: float = 340.0
    wall_margin: float = 0.15  # minimum distance (m) kept from any wall


# --------------------------------------------------------------------------- #
# Array / geometry configuration (differs between Experiment 1 and 2)
# --------------------------------------------------------------------------- #
@dataclass
class ArrayConfig:
    experiment: str = "exp1"          # "exp1" (ULA) or "exp2" (RSLA)
    n_points: int = 64                # L = M + N total array points
    roi_size: float = 3.0             # side length (m) of the square ROI
    array_length_range: Tuple[float, float] = (1.28, 3.0)

    # Experiment 1 specifics (fixed source, fixed array center, on-grid)
    fixed_array_center: Tuple[float, float, float] = (-1.5, 0.0, 0.0)
    fixed_source_pos: Tuple[float, float, float] = (1.5, 0.0, 0.0)
    fixed_axis: str = "y"              # array runs along this in-plane axis

    def K(self) -> int:
        """RIR length in samples, as specified per-experiment in the paper."""
        return 1024 if self.experiment == "exp1" else 2048


def get_array_config(experiment: str) -> ArrayConfig:
    if experiment == "exp1":
        return ArrayConfig(
            experiment="exp1",
            n_points=64,
            roi_size=3.0,
            array_length_range=(1.28, 3.0),
            fixed_array_center=(-1.5, 0.0, 0.0),
            fixed_source_pos=(1.5, 0.0, 0.0),
            fixed_axis="y",
        )
    elif experiment == "exp2":
        return ArrayConfig(
            experiment="exp2",
            n_points=64,
            roi_size=2.0,
            array_length_range=(1.28, 2.0),
        )
    else:
        raise ValueError(f"Unknown experiment '{experiment}', expected 'exp1' or 'exp2'.")


# --------------------------------------------------------------------------- #
# Model architecture (Sec. 3, Fig. 2)
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    d_model: int = 256
    n_layers: int = 3
    n_heads: int = 4
    n_segments: int = 4       # T: number of temporal segments / decoder branches
    pos_freqs: int = 6        # i = 6 in Eq. (8)
    dropout: float = 0.1
    use_residual_refine: bool = False  # OFF by default: the officially
        # released eval_rirformer.py model has no such module, even though
        # the paper's prose mentions one. Kept as an opt-in extra.


# --------------------------------------------------------------------------- #
# Training (Sec. 3 "Training Objective")
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 8
    epochs: int = 200
    mask_warmup_epochs: int = 10     # ratio increases 30% -> 70% over this many epochs
    min_mask_ratio: float = 0.3
    max_mask_ratio: float = 0.7
    finetune_epochs: int = 20        # per-segment decoder finetuning stage
    refine_finetune_epochs: int = 10  # re-adapt the residual refine module
                                       # after per-segment finetuning shifts
                                       # each branch's output distribution
    grad_clip: float = 0.0  # disabled by default -- the paper does not
                             # mention gradient clipping; kept as an
                             # opt-in safety option (set > 0 to enable)
    seed: int = 0
    device: str = "cpu"
    log_every: int = 10
    val_every: int = 1
    checkpoint_dir: str = "checkpoints"


# --------------------------------------------------------------------------- #
# Dataset generation / IO
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    fs: int = 8000
    n_train_rooms: int = 8000
    n_val_rooms: int = 200
    n_test_rooms: int = 10
    data_root: str = "data"
    eval_missing_rates: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    # NEW: where RIRDataset should read samples from
    source: str = "local"                 # "local" or "hub"
    hub_repo_id: str = "saeedzou/rir-former-datasets"
    hub_config_name: Optional[str] = None  # None -> defaults to cfg.experiment ("exp1"/"exp2")


# --------------------------------------------------------------------------- #
# Top level config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    experiment: str = "exp1"
    room: RoomConfig = field(default_factory=RoomConfig)
    array: ArrayConfig = field(default_factory=lambda: get_array_config("exp1"))
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @property
    def K(self) -> int:
        return self.array.K()

    def to_dict(self) -> dict:
        return asdict(self)


def get_config(experiment: str = "exp1", **overrides) -> Config:
    """
    Build a Config for the requested experiment ("exp1" or "exp2") with
    paper-matching defaults. `overrides` can set nested fields using
    dotted keys, e.g. get_config("exp1", **{"train.epochs": 5}).
    """
    cfg = Config(experiment=experiment, array=get_array_config(experiment))

    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)

    return cfg


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Running config.py self-test...")

    cfg1 = get_config("exp1")
    assert cfg1.K == 1024, f"Expected K=1024 for exp1, got {cfg1.K}"
    assert cfg1.array.fixed_source_pos == (1.5, 0.0, 0.0)
    assert cfg1.array.n_points == 64
    print(f"[OK] exp1 config: K={cfg1.K}, roi_size={cfg1.array.roi_size}, "
          f"array_len_range={cfg1.array.array_length_range}")

    cfg2 = get_config("exp2")
    assert cfg2.K == 2048, f"Expected K=2048 for exp2, got {cfg2.K}"
    assert cfg2.array.roi_size == 2.0
    print(f"[OK] exp2 config: K={cfg2.K}, roi_size={cfg2.array.roi_size}, "
          f"array_len_range={cfg2.array.array_length_range}")

    # dotted-key overrides
    cfg3 = get_config("exp1", **{"train.epochs": 5, "data.n_train_rooms": 32})
    assert cfg3.train.epochs == 5
    assert cfg3.data.n_train_rooms == 32
    print("[OK] dotted-key overrides work")

    # sanity: model config defaults match paper (Sec. 3 / Table architecture)
    assert cfg1.model.d_model == 256
    assert cfg1.model.n_heads == 4
    assert cfg1.model.n_layers == 3
    assert cfg1.model.pos_freqs == 6
    print("[OK] model config matches paper defaults")

    try:
        get_config("bogus")
        raise AssertionError("Expected ValueError for unknown experiment")
    except ValueError:
        print("[OK] invalid experiment name correctly raises ValueError")

    # round-trip to dict (used for checkpoint metadata)
    d = cfg1.to_dict()
    assert d["model"]["d_model"] == 256
    assert d["array"]["experiment"] == "exp1"
    print("[OK] Config.to_dict() round-trips correctly")

    print("All config.py self-tests passed!")