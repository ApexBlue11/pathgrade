"""ASMIL-Ord: attention-stabilised MIL with a rank-consistent ordinal head.

Combines three lines of recent work:

* **ASMIL** (arXiv:2603.06658) - an EMA "anchor" copy of the attention scorer
  produces a normalised-sigmoid attention map that the online (softmax) branch
  is pulled toward via KL. This is what removes the epoch-to-epoch attention
  thrash that plagues plain ABMIL on small cohorts. The anchor is discarded at
  inference, so it costs nothing at deployment.
* **nnMIL** (arXiv:2511.14907) - pool over the full encoder dimension, score
  attention on a subspace, and average across subspaces at inference.
* **ACMIL** - several attention branches with a diversity penalty, so the bag
  representation is not hostage to a single attention pattern.

The classifier emits ``n_classes - 1`` CORN logits rather than softmax logits;
see :mod:`pathgrade.losses` for why that matters for a graded target.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import (
    GatedAttentionScorer,
    build_window_offsets,
    gather_window,
    masked_softmax,
    normalised_sigmoid,
)


@dataclass
class MILOutput:
    logits: torch.Tensor                      # [B, n_classes - 1] CORN logits
    attention: torch.Tensor                   # [B, N, n_branches]
    bag_embedding: torch.Tensor               # [B, feature_dim]
    aux: dict = field(default_factory=dict)   # stabilisation / diversity terms


class ASMILOrd(nn.Module):
    def __init__(
        self,
        feature_dim: int = 1536,
        n_classes: int = 3,
        window: int = 256,
        stride: int = 64,
        hidden: int = 256,
        n_branches: int = 5,
        dropout: float = 0.25,
        branch_drop: float = 0.5,
        ema_decay: float = 0.99,
        feature_norm: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.window = window
        self.n_branches = n_branches
        self.branch_drop = branch_drop
        self.ema_decay = ema_decay

        self.norm = nn.LayerNorm(feature_dim) if feature_norm else nn.Identity()
        self.online = GatedAttentionScorer(window, hidden, n_branches, dropout)

        # Anchor mirrors the online scorer but is EMA-updated, never trained.
        self.anchor = copy.deepcopy(self.online)
        for p in self.anchor.parameters():
            p.requires_grad_(False)

        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes - 1),
        )

        self.register_buffer("offsets", build_window_offsets(feature_dim, window, stride))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_anchor(self) -> None:
        """EMA step. Call once per optimiser step, after ``optimizer.step()``."""
        m = self.ema_decay
        for a, o in zip(self.anchor.parameters(), self.online.parameters()):
            a.mul_(m).add_(o.detach(), alpha=1.0 - m)
        for a, o in zip(self.anchor.buffers(), self.online.buffers()):
            a.copy_(o)

    # ------------------------------------------------------------------
    def _branch_keep_mask(self, batch: int, device) -> torch.Tensor:
        """Bernoulli-drop attention branches, always keeping at least one."""
        keep = torch.rand(batch, self.n_branches, device=device) >= self.branch_drop
        empty = ~keep.any(dim=1)
        if empty.any():
            rescue = torch.randint(0, self.n_branches, (int(empty.sum()),), device=device)
            keep[empty, rescue] = True
        return keep

    def _pool(self, h: torch.Tensor, alpha: torch.Tensor, keep: torch.Tensor | None):
        """alpha: [B, N, K] -> bag embedding [B, D] averaged over kept branches."""
        bags = torch.einsum("bnk,bnd->bkd", alpha, h)          # [B, K, D]
        if keep is None:
            return bags.mean(dim=1)
        w = keep.to(bags.dtype).unsqueeze(-1)                  # [B, K, 1]
        return (bags * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, n_offsets: int | None = None
    ) -> MILOutput:
        """
        Args:
            x: [B, N, feature_dim] patch embeddings.
            mask: [B, N] bool, True for real patches.
            n_offsets: at eval, evaluate only this many evenly-spaced subspaces.
                Cheaper per-epoch validation; leave ``None`` for the full
                ensemble when the number actually matters.
        """
        h = self.norm(x)

        if self.training:
            # One random subspace per step - this is the regulariser.
            idx = int(torch.randint(0, len(self.offsets), (1,)).item())
            offsets = self.offsets[idx : idx + 1]
        else:
            # Every subspace, averaged: the free ensemble.
            offsets = self.offsets
            if n_offsets is not None and 0 < n_offsets < len(offsets):
                step = len(offsets) / n_offsets
                pick = [int(i * step) for i in range(n_offsets)]
                offsets = offsets[torch.tensor(pick, device=offsets.device)]

        logit_sum, attn_ref, bag_ref = None, None, None
        stab_terms, div_terms = [], []

        for off in offsets.tolist():
            x_win = gather_window(x, off, self.window)
            scores = self.online(x_win)
            alpha = masked_softmax(scores, mask)                # [B, N, K]

            keep = self._branch_keep_mask(x.shape[0], x.device) if self.training else None
            bag = self._pool(h, alpha, keep)
            logits = self.head(bag)

            logit_sum = logits if logit_sum is None else logit_sum + logits
            if attn_ref is None:
                attn_ref, bag_ref = alpha, bag

            if self.training:
                with torch.no_grad():
                    alpha_a = normalised_sigmoid(self.anchor(x_win), mask)
                stab_terms.append(masked_kl(alpha_a, alpha, mask))
                div_terms.append(branch_diversity(alpha, mask))

        logits = logit_sum / len(offsets)
        aux = {}
        if stab_terms:
            aux["stabilisation"] = torch.stack(stab_terms).mean()
            aux["diversity"] = torch.stack(div_terms).mean()

        return MILOutput(logits=logits, attention=attn_ref, bag_embedding=bag_ref, aux=aux)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def patch_attention(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Per-patch importance averaged over branches and subspaces, for heatmaps.

        Unlike input-gradient saliency this is the quantity the model actually
        pools with, so it is the honest thing to put in front of a pathologist.
        """
        was_training = self.training
        self.eval()
        acc = None
        for off in self.offsets.tolist():
            alpha = masked_softmax(self.online(gather_window(x, off, self.window)), mask)
            a = alpha.mean(dim=-1)
            acc = a if acc is None else acc + a
        self.train(was_training)
        return (acc / len(self.offsets)).masked_fill(~mask, 0.0)


def masked_kl(target: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """KL(target || pred) over the patch axis, ignoring padding. ASMIL's L_AS."""
    m = mask.unsqueeze(-1)
    t = target.clamp_min(1e-8)
    p = pred.clamp_min(1e-8)
    kl = (t * (t.log() - p.log())).masked_fill(~m, 0.0)
    return kl.sum(dim=1).mean()


def branch_diversity(alpha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean pairwise cosine similarity between branch attention maps (ACMIL).

    Minimised, so branches are pushed to attend to different tissue.
    """
    k = alpha.shape[-1]
    if k < 2:
        return alpha.new_zeros(())
    a = alpha.masked_fill(~mask.unsqueeze(-1), 0.0)
    a = F.normalize(a, dim=1)                                   # over patches
    sim = torch.einsum("bnk,bnj->bkj", a, a)                    # [B, K, K]
    off_diag = ~torch.eye(k, dtype=torch.bool, device=a.device)
    return sim[:, off_diag].mean()
