"""Early stopping and weight averaging.

Both of these exist because of specific failure modes in the v1 project.

**Early stopping.** v1 used raw ``val_qwk > best_qwk`` to reset its patience
counter. On a ~80-slide validation fold a single slide flipping grade moves QWK
by roughly 0.02, so noise alone produces "improvements" that keep a dead run
alive for another dozen epochs. This version smooths the metric with a running
median before comparing, requires the gain to clear ``min_delta``, and stops on
a *flat trend* as well as on exhausted patience.

**Weight averaging.** v1 removed SWA, but its stated reason was an
implementation bug ("fixed denominator + best epochs appear before SWA
window"), not evidence that averaging hurts. Averaging weights across the tail
of training is well supported and nearly free, so it is back - implemented so
that the averaged weights are *evaluated* rather than assumed better, and used
only if they actually win on the fold.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class StopDecision:
    stop: bool
    improved: bool
    reason: str = ""
    best: float = float("nan")
    best_epoch: int = -1
    smoothed: float = float("nan")
    slope: float = float("nan")


@dataclass
class EarlyStopping:
    """Plateau-aware early stopping on a validation metric.

    Args:
        patience: epochs without a qualifying improvement before stopping.
        min_delta: improvement must exceed this to count. Set it near the noise
            floor of your metric, not to zero.
        mode: ``"max"`` for QWK/F1, ``"min"`` for a loss.
        smooth_window: running-median width. 1 disables smoothing.
        plateau_window: epochs used to fit the recent trend.
        plateau_slope: stop when |slope per epoch| falls below this and patience
            is at least half spent. Set to None to disable trend-based stopping.
        min_epochs: never stop before this, so warmup is not mistaken for a
            plateau.
    """

    patience: int = 20
    min_delta: float = 0.002
    mode: str = "max"
    smooth_window: int = 5
    plateau_window: int = 10
    plateau_slope: float | None = 0.001
    min_epochs: int = 20

    history: list[float] = field(default_factory=list, repr=False)
    smoothed_history: list[float] = field(default_factory=list, repr=False)
    best: float = field(init=False)
    best_epoch: int = -1
    num_bad: int = 0

    def __post_init__(self):
        if self.mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.best = -np.inf if self.mode == "max" else np.inf

    # ------------------------------------------------------------------
    def _is_better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def _smooth(self) -> float:
        w = max(1, self.smooth_window)
        return float(np.median(self.history[-w:]))

    def _slope(self) -> float:
        """Least-squares slope per epoch over the recent smoothed metric."""
        if len(self.smoothed_history) < max(3, self.plateau_window):
            return float("nan")
        y = np.asarray(self.smoothed_history[-self.plateau_window :], dtype=float)
        x = np.arange(len(y), dtype=float)
        return float(np.polyfit(x, y, 1)[0])

    # ------------------------------------------------------------------
    def step(self, value: float, epoch: int) -> StopDecision:
        self.history.append(float(value))
        smoothed = self._smooth()
        self.smoothed_history.append(smoothed)
        slope = self._slope()

        # Selection tracks the *raw* metric (we want the genuinely best weights),
        # while stopping is judged on the smoothed trend (we do not want noise
        # to grant a reprieve).
        improved = self._is_better(value)
        if improved:
            self.best = float(value)
            self.best_epoch = epoch

        smoothed_improved = (
            smoothed > max(self.smoothed_history[:-1], default=-np.inf) + self.min_delta
            if self.mode == "max"
            else smoothed < min(self.smoothed_history[:-1], default=np.inf) - self.min_delta
        )
        self.num_bad = 0 if smoothed_improved else self.num_bad + 1

        if epoch < self.min_epochs:
            return StopDecision(False, improved, "", self.best, self.best_epoch, smoothed, slope)

        if self.num_bad >= self.patience:
            return StopDecision(
                True, improved,
                f"no smoothed improvement for {self.num_bad} epochs",
                self.best, self.best_epoch, smoothed, slope,
            )

        if (
            self.plateau_slope is not None
            and not np.isnan(slope)
            and abs(slope) < self.plateau_slope
            and self.num_bad >= self.patience // 2
        ):
            return StopDecision(
                True, improved,
                f"plateau: |slope| {abs(slope):.5f}/epoch over {self.plateau_window} epochs "
                f"< {self.plateau_slope}",
                self.best, self.best_epoch, smoothed, slope,
            )

        return StopDecision(False, improved, "", self.best, self.best_epoch, smoothed, slope)

    def status(self) -> str:
        slope = self._slope()
        trend = "n/a" if np.isnan(slope) else f"{slope:+.5f}/ep"
        return f"best {self.best:.4f}@{self.best_epoch} | bad {self.num_bad}/{self.patience} | trend {trend}"


class WeightEMA:
    """Exponential moving average of model weights, evaluated on its own merits.

    ``decay`` is applied per optimiser step. A useful rule of thumb is that the
    average spans roughly ``1 / (1 - decay)`` steps, so pick it relative to your
    step count rather than copying a default: at 3000 total steps, 0.999 spans
    a third of training, while 0.9999 barely moves off the initialisation.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, start_step: int = 0):
        self.decay = decay
        self.start_step = start_step
        self.step_count = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step_count += 1
        if self.step_count <= self.start_step:
            for s, p in zip(self.shadow.parameters(), model.parameters()):
                s.copy_(p.detach())
            for s, b in zip(self.shadow.buffers(), model.buffers()):
                s.copy_(b)
            return
        d = self.decay
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)

    @property
    def model(self) -> nn.Module:
        return self.shadow

    def state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.shadow.state_dict().items()}


def suggest_ema_decay(total_steps: int, span_fraction: float = 0.25) -> float:
    """Decay whose averaging window covers ``span_fraction`` of training."""
    span = max(1.0, total_steps * span_fraction)
    return float(1.0 - 1.0 / span)
