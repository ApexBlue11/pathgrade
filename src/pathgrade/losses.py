"""Losses for ordinal slide grading.

The target here is *graded*, not categorical: G1 < G2 < G3. Treating it as
plain 3-way softmax throws that away, and the previous iteration of this
project patched around it with ``CE + alpha * (expected_grade - true)^2``,
which is not a proper ordinal likelihood and lets the model emit rank-
inconsistent posteriors (e.g. P(y>1) > P(y>0)).

We use **CORN** (Shi, Cao & Raschka, arXiv:2111.08851) instead. It factorises
the ordinal target through the chain rule::

    P(y > 0)            = sigmoid(f_0)
    P(y > 1)            = sigmoid(f_0) * sigmoid(f_1)

so rank consistency holds *by construction*, with no weight-sharing constraint
of the kind that limits CORAL. Each task j is trained only on the conditional
subset {y > j-1}.

``soft_qwk_loss`` is an optional auxiliary that differentiably relaxes the
primary metric (quadratic weighted kappa). Small weights help; large weights
make the model chase the metric's marginals rather than the tissue.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def corn_loss(logits: torch.Tensor, targets: torch.Tensor, n_classes: int) -> torch.Tensor:
    """CORN loss over ``n_classes - 1`` conditional binary tasks.

    Args:
        logits: [B, n_classes - 1] raw logits.
        targets: [B] integer labels in ``[0, n_classes)``.
    """
    losses = []
    for j in range(n_classes - 1):
        # Task j is trained on the conditional subset {y > j - 1}.
        subset = targets > (j - 1) if j > 0 else torch.ones_like(targets, dtype=torch.bool)
        if not subset.any():
            continue
        task_logits = logits[subset, j]
        task_labels = (targets[subset] > j).to(task_logits.dtype)
        losses.append(F.binary_cross_entropy_with_logits(task_logits, task_labels))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def corn_cumulative_probs(logits: torch.Tensor) -> torch.Tensor:
    """P(y > j) for each j, via the chain rule. Monotone non-increasing by design."""
    return torch.cumprod(torch.sigmoid(logits), dim=1)


def corn_class_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert CORN logits to a proper [B, n_classes] distribution."""
    cum = corn_cumulative_probs(logits)                       # [B, K-1]
    ones = torch.ones_like(cum[:, :1])
    upper = torch.cat([ones, cum], dim=1)                     # P(y > j-1)
    lower = torch.cat([cum, torch.zeros_like(cum[:, :1])], dim=1)  # P(y > j)
    return (upper - lower).clamp_min(0.0)


def corn_predict(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Label = number of cumulative probabilities above ``threshold``."""
    return (corn_cumulative_probs(logits) > threshold).sum(dim=1)


def soft_qwk_loss(probs: torch.Tensor, targets: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Differentiable relaxation of 1 - quadratic weighted kappa.

    Uses the predicted distribution directly as soft counts, so gradients reach
    every class rather than only the argmax.
    """
    device = probs.device
    idx = torch.arange(n_classes, device=device, dtype=probs.dtype)
    weights = (idx.view(-1, 1) - idx.view(1, -1)) ** 2 / (n_classes - 1) ** 2

    onehot = F.one_hot(targets, n_classes).to(probs.dtype)
    observed = onehot.t() @ probs                              # [K, K] soft confusion
    observed = observed / observed.sum().clamp_min(1e-8)

    hist_true = onehot.mean(dim=0)
    hist_pred = probs.mean(dim=0)
    expected = torch.outer(hist_true, hist_pred)
    expected = expected / expected.sum().clamp_min(1e-8)

    num = (weights * observed).sum()
    den = (weights * expected).sum().clamp_min(1e-8)
    return num / den


class ASMILOrdLoss(nn.Module):
    """Total objective: CORN + ASMIL stabilisation + branch diversity (+ soft QWK).

    ``beta`` and ``gamma`` follow the ASMIL / ACMIL papers; ``lambda_qwk``
    defaults to 0 so the metric relaxation is opt-in.
    """

    def __init__(
        self,
        n_classes: int = 3,
        beta: float = 1.0,
        gamma: float = 0.1,
        lambda_qwk: float = 0.0,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.beta = beta
        self.gamma = gamma
        self.lambda_qwk = lambda_qwk

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, aux: dict | None = None):
        aux = aux or {}
        parts = {"corn": corn_loss(logits, targets, self.n_classes)}

        if self.lambda_qwk > 0:
            parts["qwk"] = self.lambda_qwk * soft_qwk_loss(
                corn_class_probs(logits), targets, self.n_classes
            )
        if self.beta > 0 and "stabilisation" in aux:
            parts["stabilisation"] = self.beta * aux["stabilisation"]
        if self.gamma > 0 and "diversity" in aux:
            parts["diversity"] = self.gamma * aux["diversity"]

        total = torch.stack(list(parts.values())).sum()
        return total, {k: float(v.detach()) for k, v in parts.items()}
