"""Typed configuration, loaded from YAML.

Every knob that changes a reported number lives here and is written into the
run directory next to the results, so a metric can always be traced back to the
settings that produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class DataConfig:
    feature_dir: str = "data/features/h-optimus-0"
    labels_csv: str = "data/tcga_hnsc_labels.csv"
    splits_path: str = "data/splits.json"
    bag_size: int | None = None      # None -> suggest_bag_size() from the cohort
    # Sub-bags drawn per slide per training epoch. 1 reproduces the first real
    # run, which saw only ~348 samples per fold and memorised them.
    samples_per_slide: int = 1
    n_classes: int = 3
    class_names: list[str] = field(default_factory=lambda: ["G1", "G2", "G3"])


@dataclass
class ModelConfig:
    feature_dim: int | None = None   # None -> read from the feature files
    window: int = 256
    stride: int = 64
    hidden: int = 256
    n_branches: int = 5
    dropout: float = 0.25
    branch_drop: float = 0.5
    ema_decay: float = 0.99
    feature_norm: bool = True


@dataclass
class LossConfig:
    beta: float = 1.0        # ASMIL attention stabilisation
    gamma: float = 0.1       # ACMIL branch diversity
    lambda_qwk: float = 0.2  # soft-QWK auxiliary
    # Penalty on uniform attention. 0 = off (what the first real run used, and
    # it collapsed to exact mean-pooling). See ASMILOrdLoss for why this is not
    # a standalone fix.
    lambda_attn_entropy: float = 0.0

    # Bag MixUp: swaps whole patches between slides (never interpolates
    # features) and interpolates the ordinal target. Off by default; enable
    # only if cross-validation says it helps.
    mixup_prob: float = 0.0
    mixup_alpha: float = 0.4


@dataclass
class OptimConfig:
    # Cohort size drives these more than anything else. A ~470-patient cohort
    # gives roughly 320 training slides per fold, so batch_size 8 yields ~40
    # optimiser steps per epoch; at 80 epochs that is ~3200 steps, which is the
    # right order for this model. Raising batch_size on a cohort this small
    # starves the run - it is steps, not epochs, that matter.
    lr: float = 5e-4
    weight_decay: float = 1e-4
    # Exempt the attention scorer from weight decay. See build_param_groups:
    # decay was the only force acting on the scorer once the head could fit the
    # data from the bag mean, and it shrank the scorer below its own init in
    # both real runs.
    scorer_no_decay: bool = False
    epochs: int = 80
    warmup_epochs: int = 5
    batch_size: int = 8
    grad_clip: float = 1.0
    num_workers: int = 4
    amp: bool = True
    balanced_sampling: bool = True
    eval_offsets: int = 6    # subspaces used for per-epoch validation

    # --- early stopping -------------------------------------------------
    # QWK on an ~80-slide fold moves about 0.02 when a single slide flips, so
    # min_delta sits just above that noise floor and the metric is median-
    # smoothed before the patience counter looks at it.
    patience: int = 20
    min_delta: float = 0.002
    smooth_window: int = 5
    plateau_window: int = 10
    plateau_slope: float | None = 0.001   # None disables trend-based stopping
    min_epochs: int = 20

    # --- discriminative learning rates ----------------------------------
    # All 1.0 collapses to a single param group, which is the default because
    # there is no projection bottleneck left to balance against.
    lr_mult_head: float = 1.0
    lr_mult_scorer: float = 1.0
    lr_mult_norm: float = 1.0

    # --- weight averaging -----------------------------------------------
    use_ema: bool = True
    ema_span_fraction: float = 0.25       # EMA window as a fraction of total steps


@dataclass
class Config:
    run_name: str = "asmil-ord-hoptimus0"
    output_dir: str = "runs"
    seed: int = 20260820
    encoder: str = "h-optimus-0"
    allow_noncommercial: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = _read_structured(Path(path))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        nested = {"data": DataConfig, "model": ModelConfig, "loss": LossConfig, "optim": OptimConfig}
        kwargs = {}
        known = {f.name for f in fields(cls)}
        for key, value in (raw or {}).items():
            if key not in known:
                raise KeyError(f"Unknown config key {key!r}. Valid keys: {sorted(known)}")
            kwargs[key] = nested[key](**value) if key in nested and value else value
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name


def _read_structured(path: Path) -> dict:
    text = path.read_text()
    if path.suffix in {".json"}:
        return json.loads(text)
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "YAML configs need PyYAML (pip install pyyaml), or use a .json config."
        ) from e
    return yaml.safe_load(text)
