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


@dataclass
class OptimConfig:
    # Cohort size drives these more than anything else. A ~470-patient cohort
    # gives roughly 320 training slides per fold, so batch_size 8 yields ~40
    # optimiser steps per epoch; at 80 epochs that is ~3200 steps, which is the
    # right order for this model. Raising batch_size on a cohort this small
    # starves the run - it is steps, not epochs, that matter.
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 80
    warmup_epochs: int = 5
    batch_size: int = 8
    grad_clip: float = 1.0
    patience: int = 20
    num_workers: int = 4
    amp: bool = True
    balanced_sampling: bool = True
    eval_offsets: int = 6    # subspaces used for per-epoch validation


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
