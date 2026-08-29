"""Patient-level splits with a locked-away test set.

The previous project reported its headline number on a validation set that had
also driven 35 Optuna trials, early stopping and best-epoch selection. That
number is a selection optimum, not a generalisation estimate, and it will not
survive technical due diligence.

This module enforces the alternative:

* one **locked test set**, carved out once and written to disk with a
  fingerprint, never seen during training or model selection;
* **k-fold cross-validation** over the remainder for every tuning decision;
* **patient-level** grouping throughout, so two slides from one patient can
  never straddle a split boundary.

The fingerprint is a hash of the sorted test patient IDs. ``load_splits``
re-checks it, so a silently re-generated split fails loudly rather than
quietly inflating a score.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


@dataclass
class Splits:
    test: list[str]
    folds: list[dict[str, list[str]]]        # [{"train": [...], "val": [...]}, ...]
    labels: dict[str, int]
    fingerprint: str
    n_classes: int


def _fingerprint(test_ids: list[str]) -> str:
    joined = "|".join(sorted(test_ids)).encode()
    return hashlib.sha256(joined).hexdigest()[:16]


def read_labels(labels_csv: str | Path, id_col: str = "patient_id", label_col: str = "label") -> dict[str, int]:
    """Read patient -> label, majority-voting if a patient appears more than once."""
    votes: dict[str, list[int]] = defaultdict(list)
    with open(labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            votes[row[id_col]].append(int(row[label_col]))
    return {pid: Counter(v).most_common(1)[0][0] for pid, v in votes.items()}


def make_splits(
    labels_csv: str | Path,
    out_path: str | Path,
    test_frac: float = 0.15,
    n_folds: int = 5,
    seed: int = 20260820,
    id_col: str = "patient_id",
    label_col: str = "label",
) -> Splits:
    labels = read_labels(labels_csv, id_col, label_col)
    pids = np.array(sorted(labels))                       # sorted => reproducible
    y = np.array([labels[p] for p in pids])
    n_classes = int(y.max()) + 1

    counts = Counter(y.tolist())
    rarest = min(counts.values())
    if rarest < n_folds:
        raise ValueError(
            f"Rarest class has {rarest} patients but n_folds={n_folds}. "
            "Reduce n_folds or merge classes - stratification cannot work otherwise."
        )

    holdout = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    dev_idx, test_idx = next(holdout.split(pids, y))
    test_ids = pids[test_idx].tolist()

    dev_pids, dev_y = pids[dev_idx], y[dev_idx]
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [
        {"train": dev_pids[tr].tolist(), "val": dev_pids[va].tolist()}
        for tr, va in kf.split(dev_pids, dev_y)
    ]

    splits = Splits(
        test=test_ids,
        folds=folds,
        labels=labels,
        fingerprint=_fingerprint(test_ids),
        n_classes=n_classes,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "fingerprint": splits.fingerprint,
                "seed": seed,
                "test_frac": test_frac,
                "n_folds": n_folds,
                "n_classes": n_classes,
                "class_counts": {str(k): int(v) for k, v in sorted(counts.items())},
                "test": splits.test,
                "folds": splits.folds,
                "labels": splits.labels,
            },
            f,
            indent=2,
        )
    return splits


def load_splits(path: str | Path) -> Splits:
    with open(path) as f:
        d = json.load(f)
    splits = Splits(
        test=d["test"],
        folds=d["folds"],
        labels=d["labels"],
        fingerprint=d["fingerprint"],
        n_classes=d["n_classes"],
    )
    if _fingerprint(splits.test) != splits.fingerprint:
        raise RuntimeError(
            "Test-set fingerprint mismatch - the split file has been edited or "
            "regenerated. Any metric computed against it is not comparable to "
            "previously reported numbers."
        )
    overlap = set(splits.test) & {p for f_ in splits.folds for p in f_["train"] + f_["val"]}
    if overlap:
        raise RuntimeError(f"{len(overlap)} test patients leaked into CV folds: {sorted(overlap)[:5]}")
    return splits
