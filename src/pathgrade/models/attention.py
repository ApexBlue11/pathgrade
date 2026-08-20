"""Attention modules for slide-level aggregation.

Two ideas from the 2025-26 literature drive this file:

* **No projection bottleneck.** nnMIL (arXiv:2511.14907) shows that projecting
  foundation-model features down before aggregation destroys the semantics the
  encoder learned. We therefore pool over the *full* embedding dimension and
  only ever subsample dimensions for the attention *scorer*.
* **Subspace attention.** The scorer sees a contiguous window of ``window``
  dimensions. During training a window is drawn at random, which regularises
  the scorer; at inference every window is evaluated and averaged, which buys
  an ensemble for free (no extra models to train).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e4  # fp16-safe additive mask


def build_window_offsets(dim: int, window: int, stride: int) -> torch.Tensor:
    """Anchored, wrap-around start offsets for the subspace windows.

    Using a fixed offset grid (rather than a fresh uniform draw each step) keeps
    train and inference consistent: every offset the scorer sees at test time is
    one it was trained on.
    """
    if window > dim:
        raise ValueError(f"window ({window}) cannot exceed feature dim ({dim})")
    offsets = list(range(0, dim, stride))
    return torch.tensor(offsets, dtype=torch.long)


def gather_window(x: torch.Tensor, offset: int, window: int) -> torch.Tensor:
    """Slice ``window`` dims starting at ``offset``, wrapping past the end."""
    dim = x.shape[-1]
    end = offset + window
    if end <= dim:
        return x[..., offset:end]
    return torch.cat([x[..., offset:], x[..., : end - dim]], dim=-1)


def normalised_sigmoid(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """NSF from ASMIL (arXiv:2603.06658): ``sigmoid(z_i) / sum_j sigmoid(z_j)``.

    Unlike softmax this can flatten the high-valued tail while still suppressing
    the low one, so the anchor branch stops a handful of tiles from monopolising
    the bag. It is deliberately *not* used on the online branch, where it
    causes vanishing gradients.
    """
    s = torch.sigmoid(scores)
    s = s.masked_fill(~mask.unsqueeze(-1), 0.0)
    return s / s.sum(dim=1, keepdim=True).clamp_min(1e-6)


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = scores.masked_fill(~mask.unsqueeze(-1), NEG_INF)
    return torch.softmax(scores, dim=1)


class GatedAttentionScorer(nn.Module):
    """ABMIL gated attention, scored on a subspace, with multiple branches.

    Args:
        window: number of feature dims the scorer sees.
        hidden: width of the gated attention MLP.
        n_branches: parallel attention heads (ACMIL-style). Each yields its own
            bag embedding; diversity between them is encouraged by a loss term.
    """

    def __init__(self, window: int, hidden: int = 256, n_branches: int = 5, dropout: float = 0.25):
        super().__init__()
        self.window = window
        self.n_branches = n_branches
        self.value = nn.Sequential(nn.Linear(window, hidden), nn.Tanh(), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(window, hidden), nn.Sigmoid(), nn.Dropout(dropout))
        self.score = nn.Linear(hidden, n_branches)

    def forward(self, x_win: torch.Tensor) -> torch.Tensor:
        """``x_win`` is [B, N, window]; returns raw scores [B, N, n_branches]."""
        return self.score(self.value(x_win) * self.gate(x_win))
