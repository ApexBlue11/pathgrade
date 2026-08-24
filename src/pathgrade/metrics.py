"""Metrics for ordinal grading, with uncertainty attached.

Quadratic weighted kappa is the primary metric because it is what pathologists
use to quantify agreement with each other, which makes it the only number in
this file that a clinician can interpret without translation.

Two things are reported alongside it that usually are not:

* **Bootstrap confidence intervals.** On a few hundred slides a QWK point
  estimate has a spread of roughly +/-0.1. Quoting three decimal places without
  an interval implies a precision the cohort cannot support.
* **The human ceiling.** Inter-observer QWK for histologic grading tends to
  run 0.5-0.7 across several cancer types in the literature - grading is a
  genuinely noisy label. A model at 0.67 is not "67% of the way to solved";
  it may be at the noise floor of its own labels, and the honest framing of
  that is a selling point, not a weakness. NOTE: the 0.5-0.7 figure below is
  carried without a citation pinned to HNSCC specifically - verify against
  the literature before quoting it externally.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

# Commonly-reported inter-observer agreement range for histologic grading
# tasks generally. NOT a pinned citation for HNSCC specifically - see the
# module docstring above.
HUMAN_QWK_RANGE = (0.50, 0.70)


@dataclass
class GradingMetrics:
    qwk: float
    macro_f1: float
    balanced_accuracy: float
    accuracy: float
    adjacent_accuracy: float          # within one grade - the clinically tolerable error
    mean_absolute_error: float
    confusion: list[list[int]]
    per_class_f1: list[float]
    n: int
    qwk_ci: tuple[float, float] | None = None
    extras: dict = field(default_factory=dict)

    def summary(self) -> str:
        ci = f" [{self.qwk_ci[0]:.3f}, {self.qwk_ci[1]:.3f}]" if self.qwk_ci else ""
        return (
            f"QWK {self.qwk:.3f}{ci} | macroF1 {self.macro_f1:.3f} | "
            f"balAcc {self.balanced_accuracy:.3f} | adj-acc {self.adjacent_accuracy:.3f} | "
            f"MAE {self.mean_absolute_error:.3f} | n={self.n}"
        )


def compute_metrics(
    y_true, y_pred, n_classes: int = 3, bootstrap: int = 0, seed: int = 0
) -> GradingMetrics:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = list(range(n_classes))

    ci = None
    if bootstrap > 0 and len(y_true) > 1:
        ci = bootstrap_qwk_ci(y_true, y_pred, n_boot=bootstrap, seed=seed)

    return GradingMetrics(
        qwk=float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        accuracy=float((y_true == y_pred).mean()),
        adjacent_accuracy=float((np.abs(y_true - y_pred) <= 1).mean()),
        mean_absolute_error=float(np.abs(y_true - y_pred).mean()),
        confusion=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        per_class_f1=f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist(),
        n=int(len(y_true)),
        qwk_ci=ci,
    )


def bootstrap_qwk_ci(
    y_true, y_pred, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for QWK, resampling slides with replacement."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        # A resample can be single-class, where kappa is undefined; skip those.
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(cohen_kappa_score(y_true[idx], y_pred[idx], weights="quadratic"))
    if not scores:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def format_confusion(cm, class_names: list[str] | None = None) -> str:
    cm = np.asarray(cm)
    names = class_names or [f"C{i}" for i in range(len(cm))]
    width = max(len(n) for n in names) + 2
    header = " " * (width + 6) + "".join(f"{n:>7}" for n in names)
    lines = [header, " " * (width + 6) + "-" * (7 * len(names))]
    for i, name in enumerate(names):
        row = "".join(f"{v:>7}" for v in cm[i])
        lines.append(f"{'true ' + name:>{width + 5}} |{row}")
    return "\n".join(lines)


def aggregate_folds(fold_metrics: list[GradingMetrics]) -> dict:
    """Mean +/- std across CV folds - the honest model-selection number."""
    def ms(key):
        vals = [getattr(m, key) for m in fold_metrics]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "per_fold": vals}

    return {
        "qwk": ms("qwk"),
        "macro_f1": ms("macro_f1"),
        "balanced_accuracy": ms("balanced_accuracy"),
        "adjacent_accuracy": ms("adjacent_accuracy"),
        "mean_absolute_error": ms("mean_absolute_error"),
        "n_folds": len(fold_metrics),
    }


def contextualise(qwk: float) -> str:
    """One line putting a QWK next to the human agreement band."""
    lo, hi = HUMAN_QWK_RANGE
    if qwk < lo:
        return f"QWK {qwk:.3f} is below the typical inter-observer band ({lo:.2f}-{hi:.2f})."
    if qwk <= hi:
        return (
            f"QWK {qwk:.3f} sits inside the typical inter-observer band "
            f"({lo:.2f}-{hi:.2f}) - i.e. at the label noise floor, not obviously below it."
        )
    return f"QWK {qwk:.3f} exceeds the typical inter-observer band ({lo:.2f}-{hi:.2f})."
