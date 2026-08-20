"""Invariants that must not regress.

These are the properties that were checked by hand while building the pipeline;
each one corresponds to a way the previous project could go wrong silently.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathgrade.data.dataset import SlideBagDataset, SlideCropIterator, balanced_sampler, suggest_bag_size
from pathgrade.data.splits import load_splits, make_splits
from pathgrade.encoders import REGISTRY, LicenceError, check_licence
from pathgrade.losses import (
    ASMILOrdLoss, corn_class_probs, corn_cumulative_probs, corn_loss, corn_predict, soft_qwk_loss,
)
from pathgrade.metrics import compute_metrics
from pathgrade.models.asmil_ord import ASMILOrd
from pathgrade.models.attention import gather_window, masked_softmax, normalised_sigmoid


# --------------------------------------------------------------------------
# Licensing - the failure mode that kills the product rather than the metric
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["uni", "uni2-h", "prov-gigapath", "virchow2", "h-optimus-1", "phikon-v2"])
def test_noncommercial_encoders_are_blocked(name):
    with pytest.raises(LicenceError):
        check_licence(name)


@pytest.mark.parametrize("name", ["h-optimus-0", "virchow", "hibou-l", "midnight"])
def test_commercial_encoders_load(name):
    assert check_licence(name).commercial_ok


def test_noncommercial_requires_explicit_optin():
    assert check_licence("uni", allow_noncommercial=True).name == "uni"


def test_gigapath_is_not_trusted_to_its_hf_tag():
    """HF tags prov-gigapath apache-2.0, but its model card forbids deployment."""
    assert REGISTRY["prov-gigapath"].commercial_ok is False


def test_virchow_v1_and_v2_differ():
    assert REGISTRY["virchow"].commercial_ok and not REGISTRY["virchow2"].commercial_ok


def test_h_optimus_0_and_1_differ():
    assert REGISTRY["h-optimus-0"].commercial_ok and not REGISTRY["h-optimus-1"].commercial_ok


# --------------------------------------------------------------------------
# Ordinal head
# --------------------------------------------------------------------------
def test_corn_is_rank_consistent():
    cum = corn_cumulative_probs(torch.randn(256, 2))
    assert (cum[:, 0] >= cum[:, 1] - 1e-6).all()


def test_corn_class_probs_form_a_distribution():
    p = corn_class_probs(torch.randn(128, 2))
    assert torch.allclose(p.sum(dim=1), torch.ones(128), atol=1e-5)
    assert (p >= 0).all()


def test_corn_predictions_on_saturated_logits():
    clear = torch.tensor([[8.0, 8.0], [-8.0, 0.0], [8.0, -8.0]])
    assert corn_predict(clear).tolist() == [2, 0, 1]


def test_corn_conditional_subset_excludes_lower_grades():
    """Task 1 must ignore samples with y == 0, or the ordinal factorisation is wrong."""
    logits = torch.randn(8, 2, requires_grad=True)
    corn_loss(logits, torch.zeros(8, dtype=torch.long), 3).backward()
    assert float(logits.grad[:, 1].abs().sum()) == 0.0


def test_corn_optimises_to_a_fit():
    torch.manual_seed(0)
    logits = torch.randn(64, 2, requires_grad=True)
    targets = torch.randint(0, 3, (64,))
    opt = torch.optim.Adam([logits], lr=0.5)
    for _ in range(200):
        opt.zero_grad()
        corn_loss(logits, targets, 3).backward()
        opt.step()
    assert (corn_predict(logits) == targets).float().mean() > 0.95


def test_soft_qwk_tracks_true_kappa():
    from sklearn.metrics import cohen_kappa_score

    t = torch.randint(0, 3, (300,))
    perfect = torch.stack([torch.where(t > 0, 8.0, -8.0), torch.where(t > 1, 8.0, -8.0)], dim=1)
    loss = float(soft_qwk_loss(corn_class_probs(perfect), t, 3))
    kappa = cohen_kappa_score(t.numpy(), corn_predict(perfect).numpy(), weights="quadratic")
    assert loss < 0.01 and kappa > 0.99


def test_combined_loss_sums_its_parts():
    crit = ASMILOrdLoss(3, beta=1.0, gamma=0.1, lambda_qwk=0.2)
    total, parts = crit(
        torch.randn(16, 2), torch.randint(0, 3, (16,)),
        {"stabilisation": torch.tensor(0.05), "diversity": torch.tensor(0.9)},
    )
    assert set(parts) == {"corn", "qwk", "stabilisation", "diversity"}
    assert abs(float(total) - sum(parts.values())) < 1e-5


# --------------------------------------------------------------------------
# Attention / model
# --------------------------------------------------------------------------
def test_window_wraps_around():
    x = torch.randn(2, 4, 1536)
    w = gather_window(x, 1500, 256)
    assert torch.allclose(w[..., :36], x[..., 1500:])
    assert torch.allclose(w[..., 36:], x[..., :220])


def test_attention_ignores_padding():
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[1, 5:] = False
    scores = torch.randn(2, 10, 3)
    for fn in (masked_softmax, normalised_sigmoid):
        a = fn(scores, mask)
        assert torch.allclose(a.sum(dim=1), torch.ones(2, 3), atol=1e-5)
        assert a[1, 5:].abs().max() < 1e-6


def test_model_is_padding_invariant():
    """Padding a bag must not change its prediction, or batching corrupts results."""
    torch.manual_seed(0)
    m = ASMILOrd(feature_dim=256, window=64, stride=16).eval()
    x = torch.randn(1, 300, 256)
    padded_x = torch.cat([x, torch.randn(1, 200, 256)], dim=1)
    padded_mask = torch.cat([torch.ones(1, 300, dtype=torch.bool), torch.zeros(1, 200, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        a = m(x, torch.ones(1, 300, dtype=torch.bool)).logits
        b = m(padded_x, padded_mask).logits
    assert torch.allclose(a, b, atol=1e-5)


def test_eval_is_deterministic_but_train_is_not():
    torch.manual_seed(0)
    m = ASMILOrd(feature_dim=256, window=64, stride=16)
    x, mask = torch.randn(2, 100, 256), torch.ones(2, 100, dtype=torch.bool)
    m.eval()
    with torch.no_grad():
        assert torch.allclose(m(x, mask).logits, m(x, mask).logits)
    m.train()
    assert not torch.allclose(m(x, mask).logits, m(x, mask).logits)


def test_anchor_is_ema_updated_and_never_trained():
    m = ASMILOrd(feature_dim=256, window=64, stride=16)
    assert all(not p.requires_grad for p in m.anchor.parameters())
    m.train()
    out = m(torch.randn(2, 64, 256), torch.ones(2, 64, dtype=torch.bool))
    out.logits.sum().backward()
    assert m.anchor.score.weight.grad is None
    before = m.anchor.score.weight.clone()
    m.update_anchor()
    assert not torch.equal(before, m.anchor.score.weight)


def test_no_projection_bottleneck():
    """Pooling happens at full encoder width - the v1 1024->256 proj is gone."""
    m = ASMILOrd(feature_dim=1536, window=256, stride=64)
    out = m.eval()(torch.randn(1, 50, 1536), torch.ones(1, 50, dtype=torch.bool))
    assert out.bag_embedding.shape[-1] == 1536


def test_subspace_count_matches_stride():
    m = ASMILOrd(feature_dim=1536, window=256, stride=64)
    assert len(m.offsets) == 24


def test_n_offsets_subsamples_the_ensemble():
    m = ASMILOrd(feature_dim=256, window=64, stride=16).eval()
    x, mask = torch.randn(1, 40, 256), torch.ones(1, 40, dtype=torch.bool)
    with torch.no_grad():
        assert not torch.allclose(m(x, mask).logits, m(x, mask, n_offsets=2).logits)


# --------------------------------------------------------------------------
# Splits - the evaluation-integrity failure mode
# --------------------------------------------------------------------------
@pytest.fixture
def cohort(tmp_path):
    rng = np.random.default_rng(0)
    labels = rng.choice(3, size=200, p=[0.15, 0.6, 0.25])
    pids = [f"P{i:04d}" for i in range(200)]
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "label"])
        for p, l in zip(pids, labels):
            w.writerow([p, int(l)])
    return csv_path, tmp_path


def test_test_set_never_appears_in_any_fold(cohort):
    csv_path, tmp = cohort
    s = make_splits(csv_path, tmp / "splits.json", test_frac=0.15, n_folds=5)
    dev = {p for f in s.folds for p in f["train"] + f["val"]}
    assert not (set(s.test) & dev)


def test_train_and_val_never_overlap(cohort):
    csv_path, tmp = cohort
    s = make_splits(csv_path, tmp / "splits.json", n_folds=5)
    for f in s.folds:
        assert not (set(f["train"]) & set(f["val"]))


def test_val_folds_tile_the_dev_set(cohort):
    csv_path, tmp = cohort
    s = make_splits(csv_path, tmp / "splits.json", n_folds=5)
    val_union = [p for f in s.folds for p in f["val"]]
    assert len(val_union) == len(set(val_union)) == 200 - len(s.test)


def test_tampered_split_file_is_rejected(cohort):
    csv_path, tmp = cohort
    make_splits(csv_path, tmp / "splits.json")
    d = json.loads((tmp / "splits.json").read_text())
    d["test"].append("P9999")
    (tmp / "bad.json").write_text(json.dumps(d))
    with pytest.raises(RuntimeError, match="fingerprint"):
        load_splits(tmp / "bad.json")


def test_splits_are_reproducible(cohort):
    csv_path, tmp = cohort
    a = make_splits(csv_path, tmp / "a.json", seed=1)
    b = make_splits(csv_path, tmp / "b.json", seed=1)
    assert a.test == b.test and a.fingerprint == b.fingerprint


def test_too_few_folds_for_rarest_class_raises(tmp_path):
    csv_path = tmp_path / "l.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "label"])
        for i in range(40):
            w.writerow([f"P{i}", 0 if i else 2])   # class 2 has a single patient
    with pytest.raises(ValueError, match="Rarest class"):
        make_splits(csv_path, tmp_path / "s.json", n_folds=5)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
@pytest.fixture
def features(tmp_path):
    import h5py

    rng = np.random.default_rng(0)
    d = tmp_path / "feat"
    d.mkdir()
    labels, counts = {}, {}
    for i in range(12):
        pid = f"P{i:04d}"
        n = int(rng.integers(100, 3000))
        counts[pid] = n
        labels[pid] = int(rng.integers(0, 3))
        with h5py.File(d / f"{pid}.h5", "w") as f:
            f.create_dataset("features", data=rng.standard_normal((n, 64)).astype(np.float32))
            f.create_dataset("coords", data=rng.integers(0, 9999, (n, 2)))
    return d, labels, counts


def test_bags_are_fixed_size_and_masked(features):
    d, labels, counts = features
    ds = SlideBagDataset(list(labels), labels, d, bag_size=512, train=False)
    for item in (ds[i] for i in range(len(ds))):
        assert item["features"].shape == (512, 64)
        assert int(item["mask"].sum()) == min(counts[item["patient_id"]], 512)
        assert item["features"][~item["mask"]].abs().sum() == 0


def test_missing_feature_file_fails_loudly(features):
    d, labels, _ = features
    with pytest.raises(FileNotFoundError, match="missing"):
        SlideBagDataset(list(labels) + ["GHOST"], {**labels, "GHOST": 0}, d, bag_size=64)


def test_crop_iterator_covers_every_patch_exactly_once(features):
    d, labels, counts = features
    pid = list(labels)[0]
    crops = list(SlideCropIterator(d / f"{pid}.h5", 512))
    assert sum(int(m.sum()) for _, m in crops) == counts[pid]


def test_balanced_sampler_rebalances():
    labels = [0] * 5 + [1] * 90 + [2] * 5
    draws = list(torch.utils.data.DataLoader(
        list(range(100)), batch_size=100, sampler=balanced_sampler(labels, seed=0)
    ))[0]
    seen = np.bincount([labels[i] for i in draws.tolist()], minlength=3)
    assert seen.min() > 10       # would be ~5 without rebalancing


def test_suggested_bag_size_is_sane(features):
    d, labels, _ = features
    bs = suggest_bag_size(d, list(labels))
    assert 256 <= bs <= 4096 and bs % 256 == 0


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def test_perfect_and_inverted_predictions():
    y = np.array([0, 1, 2] * 10)
    assert compute_metrics(y, y, 3).qwk == pytest.approx(1.0)
    assert compute_metrics(y, 2 - y, 3).qwk < 0


def test_adjacent_accuracy_tolerates_one_grade():
    y = np.array([0, 1, 2, 1])
    m = compute_metrics(y, np.array([1, 2, 1, 0]), 3)
    assert m.adjacent_accuracy == 1.0 and m.accuracy == 0.0


def test_bootstrap_ci_brackets_the_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 150)
    p = np.where(rng.random(150) < 0.75, y, rng.integers(0, 3, 150))
    m = compute_metrics(y, p, 3, bootstrap=400)
    assert m.qwk_ci[0] <= m.qwk <= m.qwk_ci[1]
