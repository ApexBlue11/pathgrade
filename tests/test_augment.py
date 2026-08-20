"""Bag MixUp and discriminative learning rates."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathgrade.augment import bag_mixup, describe_mix
from pathgrade.config import Config
from pathgrade.losses import ASMILOrdLoss, corn_cumulative_probs, corn_loss_soft
from pathgrade.models import ASMILOrd
from pathgrade.train import build_param_groups


def _bags(b=8, n=64, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(b, n, d, generator=g),
        torch.ones(b, n, dtype=torch.bool),
        torch.randint(0, 3, (b,), generator=g),
    )


# --------------------------------------------------------------------------
def test_mixup_disabled_is_identity():
    x, m, y = _bags()
    fx, fm, t = bag_mixup(x, m, y, 3, prob=0.0)
    assert torch.equal(fx, x) and torch.equal(fm, m)
    assert torch.equal(t, (y.unsqueeze(1) > torch.arange(2)).float())


def test_hard_targets_encode_the_grade():
    x, m, _ = _bags()
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    _, _, t = bag_mixup(x, m, y, 3, prob=0.0)
    assert t[0].tolist() == [0.0, 0.0]      # G1: P(y>0)=0, P(y>1)=0
    assert t[1].tolist() == [1.0, 0.0]      # G2: P(y>0)=1, P(y>1)=0
    assert t[2].tolist() == [1.0, 1.0]      # G3: both 1


def test_mixed_bag_contains_only_real_patches():
    """The whole point: patches are swapped, never interpolated."""
    torch.manual_seed(0)
    x, m, y = _bags(b=4, n=32, d=8)
    fx, _, _ = bag_mixup(x, m, y, 3, prob=1.0, alpha=0.5)
    for i in range(fx.shape[0]):
        for j in range(fx.shape[1]):
            row = fx[i, j]
            # every patch must be byte-identical to some patch in the source batch
            assert (x[:, j] == row).all(dim=-1).any(), "patch was synthesised, not swapped"


def test_mixing_extremes_lands_between_them():
    """Half a G1 bag plus half a G3 bag should imply roughly G2."""
    torch.manual_seed(0)
    n = 4096
    x = torch.randn(2, n, 8)
    m = torch.ones(2, n, dtype=torch.bool)
    y = torch.tensor([0, 2])
    expected = []
    for _ in range(30):
        _, _, t = bag_mixup(x, m, y, 3, prob=1.0, alpha=1000.0)   # alpha huge -> lam ~ 0.5
        expected.append(describe_mix(t))
    mean_grade = torch.stack(expected).mean()
    assert 0.8 < float(mean_grade) < 1.2, f"expected ~1.0 (G2), got {mean_grade:.3f}"


def test_targets_stay_in_unit_interval():
    torch.manual_seed(0)
    x, m, y = _bags(b=16, n=128)
    _, _, t = bag_mixup(x, m, y, 3, prob=1.0)
    assert (t >= 0).all() and (t <= 1).all()


def test_padding_is_accounted_for_in_the_mix_weight():
    """A bag that is mostly padding must not claim half the label."""
    torch.manual_seed(0)
    n = 512
    x = torch.randn(2, n, 8)
    m = torch.zeros(2, n, dtype=torch.bool)
    m[0, :] = True            # full bag, grade 0
    m[1, :16] = True          # nearly empty bag, grade 2
    y = torch.tensor([0, 2])
    grades = [float(describe_mix(bag_mixup(x, m, y, 3, prob=1.0, alpha=1000.0)[2])[0]) for _ in range(20)]
    # Sample 0 keeps ~half its many patches and gains only ~8 from the tiny bag,
    # so its target must stay very close to G1.
    assert sum(grades) / len(grades) < 0.15


def test_mixed_bags_keep_some_patches():
    torch.manual_seed(0)
    x, m, y = _bags(b=8, n=64)
    _, fm, _ = bag_mixup(x, m, y, 3, prob=1.0)
    assert fm.any(dim=1).all(), "no bag may end up empty"


# --------------------------------------------------------------------------
def test_soft_corn_loss_is_minimised_at_the_target():
    torch.manual_seed(0)
    targets = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    logits = torch.zeros(3, 2, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.2)
    for _ in range(500):
        opt.zero_grad()
        corn_loss_soft(logits, targets).backward()
        opt.step()
    assert torch.allclose(corn_cumulative_probs(logits), targets, atol=0.02)


def test_soft_corn_stays_rank_consistent():
    logits = torch.randn(64, 2)
    cum = corn_cumulative_probs(logits)
    assert (cum[:, 0] >= cum[:, 1] - 1e-6).all()


def test_loss_uses_soft_path_and_skips_qwk_when_mixed():
    crit = ASMILOrdLoss(3, beta=1.0, gamma=0.1, lambda_qwk=0.2)
    logits, y = torch.randn(8, 2), torch.randint(0, 3, (8,))
    aux = {"stabilisation": torch.tensor(0.05), "diversity": torch.tensor(0.9)}
    _, hard_parts = crit(logits, y, aux)
    _, soft_parts = crit(logits, y, aux, cum_targets=torch.rand(8, 2))
    assert "qwk" in hard_parts and "qwk" not in soft_parts
    assert "corn" in soft_parts


# --------------------------------------------------------------------------
def test_param_groups_collapse_to_one_lr_by_default():
    cfg = Config()
    model = ASMILOrd(feature_dim=64, window=16, stride=8, hidden=32, n_branches=2)
    groups = build_param_groups(model, cfg)
    assert len({g["lr"] for g in groups}) == 1


def test_param_group_multipliers_apply():
    cfg = Config()
    cfg.optim.lr = 1e-3
    cfg.optim.lr_mult_head, cfg.optim.lr_mult_scorer, cfg.optim.lr_mult_norm = 2.0, 1.0, 0.5
    model = ASMILOrd(feature_dim=64, window=16, stride=8, hidden=32, n_branches=2)
    lrs = {g["name"]: g["lr"] for g in build_param_groups(model, cfg)}
    assert lrs["head"] == pytest.approx(2e-3)
    assert lrs["scorer"] == pytest.approx(1e-3)
    assert lrs["norm"] == pytest.approx(5e-4)


def test_param_groups_cover_every_trainable_parameter():
    cfg = Config()
    model = ASMILOrd(feature_dim=64, window=16, stride=8, hidden=32, n_branches=2)
    grouped = sum(p.numel() for g in build_param_groups(model, cfg) for p in g["params"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert grouped == trainable


def test_param_groups_exclude_the_frozen_anchor():
    cfg = Config()
    model = ASMILOrd(feature_dim=64, window=16, stride=8, hidden=32, n_branches=2)
    names = {g["name"] for g in build_param_groups(model, cfg)}
    assert "anchor" not in names
    anchor_ids = {id(p) for p in model.anchor.parameters()}
    grouped_ids = {id(p) for g in build_param_groups(model, cfg) for p in g["params"]}
    assert not (anchor_ids & grouped_ids)
