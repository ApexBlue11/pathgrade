"""Bag-level MixUp for ordinal slide grading.

v1 dropped MixUp as "biologically unsound for WSI feature space", and for the
usual formulation that objection is fair: convex-combining two feature vectors
invents tissue that does not exist, and a foundation encoder's manifold is not
something to interpolate across casually.

This is a different operation. Instead of averaging features, it **swaps whole
patches between two slides**, so every instance in a mixed bag is a real patch
from a real slide - nothing is synthesised. What changes is only which patches
share a bag.

That makes it well-motivated rather than merely tolerable, for two reasons:

* Tumour heterogeneity is real. Slides genuinely contain regions of differing
  differentiation, and a pathologist grading a heterogeneous slide is doing
  something close to this integration already.
* The target interpolates *along the grade axis*. Mixing a G1 bag with a G3 bag
  in equal parts yields a cumulative target that sits at G2, which is a real
  clinical category rather than a meaningless average. This supplies exactly the
  supervision an ordinal head wants and that a small, G1-poor cohort lacks.

Off by default (``mixup_prob: 0.0``). Turn it on if the CV score says so.
"""

from __future__ import annotations

import numpy as np
import torch


def bag_mixup(
    features: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    alpha: float = 0.4,
    prob: float = 0.5,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Swap patches between paired bags; return soft cumulative targets.

    Args:
        features: [B, N, D]
        mask: [B, N] bool, True for real patches
        labels: [B] integer grades
        alpha: Beta(alpha, alpha) mixing strength. Lower = milder mixing.
        prob: fraction of the batch to mix.

    Returns:
        ``(features, mask, cum_targets)`` with ``cum_targets`` of shape
        [B, n_classes - 1] holding soft P(y > j) targets.
    """
    b, n, _ = features.shape
    device = features.device

    # Hard cumulative targets: t[i, j] = 1 if label_i > j
    grades = torch.arange(n_classes - 1, device=device).unsqueeze(0)
    cum_targets = (labels.unsqueeze(1) > grades).float()

    if prob <= 0.0 or b < 2:
        return features, mask, cum_targets

    do_mix = torch.rand(b, device=device, generator=generator) < prob
    if not do_mix.any():
        return features, mask, cum_targets

    partner = torch.randperm(b, device=device, generator=generator)
    lam = torch.from_numpy(
        np.random.beta(alpha, alpha, size=b).astype(np.float32)
    ).to(device)
    lam = torch.where(do_mix, lam, torch.ones_like(lam))

    # Per-patch Bernoulli choice between self and partner.
    keep_self = torch.rand(b, n, device=device, generator=generator) < lam.unsqueeze(1)

    mixed_features = torch.where(keep_self.unsqueeze(-1), features, features[partner])
    mixed_mask = torch.where(keep_self, mask, mask[partner])

    # The effective mixing weight is the share of *valid* patches contributed by
    # each side, not the nominal lambda: bags differ in how much padding they
    # carry, and the label must follow the tissue actually present.
    from_self = (keep_self & mask).sum(dim=1).float()
    from_partner = (~keep_self & mask[partner]).sum(dim=1).float()
    lam_eff = from_self / (from_self + from_partner).clamp_min(1.0)
    lam_eff = torch.where(do_mix, lam_eff, torch.ones_like(lam_eff))

    mixed_targets = (
        lam_eff.unsqueeze(1) * cum_targets
        + (1.0 - lam_eff).unsqueeze(1) * cum_targets[partner]
    )

    # A bag that ended up entirely empty would produce an undefined target;
    # fall back to the unmixed sample in that (very rare) case.
    empty = ~mixed_mask.any(dim=1)
    if empty.any():
        mixed_features[empty] = features[empty]
        mixed_mask[empty] = mask[empty]
        mixed_targets[empty] = cum_targets[empty]

    return mixed_features, mixed_mask, mixed_targets


def describe_mix(cum_targets: torch.Tensor) -> torch.Tensor:
    """Expected grade implied by soft cumulative targets, for logging."""
    return cum_targets.sum(dim=1)
