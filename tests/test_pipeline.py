"""Tests for streaming extraction, feature IO, and the training callbacks."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathgrade.callbacks import EarlyStopping, WeightEMA, suggest_ema_decay
from pathgrade.data.io import (
    feature_width, find_feature_file, patch_count, read_features, read_metadata, verify_cohort,
)
from pathgrade.models.asmil_ord import ASMILOrd
from pathgrade.preprocessing.gdc import (
    SlideRecord, one_slide_per_patient, patient_id_from_barcode, shard,
)
from pathgrade.preprocessing.stream_extract import write_features


# --------------------------------------------------------------------------
# Early stopping
# --------------------------------------------------------------------------
def test_does_not_stop_before_min_epochs():
    es = EarlyStopping(patience=2, min_epochs=10, plateau_slope=None)
    for epoch in range(1, 10):
        assert not es.step(0.5, epoch).stop        # dead flat, still too early


def test_stops_when_patience_exhausted():
    es = EarlyStopping(patience=5, min_epochs=5, smooth_window=1, plateau_slope=None)
    stopped_at = None
    for epoch in range(1, 30):
        value = 0.6 if epoch <= 5 else 0.4         # improves, then flatlines lower
        if es.step(value, epoch).stop:
            stopped_at = epoch
            break
    assert stopped_at is not None and stopped_at < 20


def test_noise_spike_does_not_reset_patience():
    """The v1 bug: a lucky single-epoch bounce kept dead runs alive.

    Smoothing means one spike cannot rescue a flat trend.
    """
    es = EarlyStopping(patience=6, min_epochs=5, smooth_window=5, min_delta=0.002, plateau_slope=None)
    rng = np.random.default_rng(0)
    stopped_at = None
    for epoch in range(1, 40):
        value = 0.50 + rng.normal(0, 0.01)          # flat + noise
        if epoch == 12:
            value = 0.62                            # one lucky spike
        if es.step(value, epoch).stop:
            stopped_at = epoch
            break
    assert stopped_at is not None, "should stop despite the spike"
    assert stopped_at < 30


def test_selection_still_tracks_the_raw_best():
    """Stopping uses the smoothed trend, but we keep the genuinely best weights."""
    es = EarlyStopping(patience=5, min_epochs=1, smooth_window=5, plateau_slope=None)
    for epoch, v in enumerate([0.4, 0.5, 0.9, 0.5, 0.5, 0.5], start=1):
        es.step(v, epoch)
    assert es.best == pytest.approx(0.9) and es.best_epoch == 3


def test_plateau_detection_fires_on_flat_trend():
    flat = EarlyStopping(patience=20, min_epochs=5, plateau_window=8, plateau_slope=0.001, smooth_window=3)
    stopped = any(flat.step(0.6, e).stop for e in range(1, 40))
    assert stopped, "a perfectly flat curve should trigger the plateau rule"


def test_plateau_does_not_fire_while_still_climbing():
    es = EarlyStopping(patience=20, min_epochs=5, plateau_window=8, plateau_slope=0.001, smooth_window=3)
    assert not any(es.step(0.3 + 0.01 * e, e).stop for e in range(1, 40))


def test_min_mode_for_losses():
    es = EarlyStopping(patience=3, min_epochs=1, mode="min", smooth_window=1, plateau_slope=None)
    assert es.step(1.0, 1).improved
    assert es.step(0.5, 2).improved
    assert not es.step(0.9, 3).improved


def test_rejects_bad_mode():
    with pytest.raises(ValueError):
        EarlyStopping(mode="sideways")


# --------------------------------------------------------------------------
# Weight EMA
# --------------------------------------------------------------------------
def test_ema_tracks_then_lags_the_model():
    model = ASMILOrd(feature_dim=64, window=16, stride=8, hidden=32, n_branches=2)
    ema = WeightEMA(model, decay=0.9, start_step=2)

    ema.update(model)                                   # warmup: copies
    p = next(model.parameters())
    assert torch.allclose(next(ema.model.parameters()), p)

    with torch.no_grad():
        for q in model.parameters():
            q.add_(1.0)
    ema.update(model)                                   # still warmup
    ema.update(model)                                   # now averaging
    with torch.no_grad():
        for q in model.parameters():
            q.add_(1.0)
    ema.update(model)
    assert not torch.allclose(next(ema.model.parameters()), next(model.parameters()))


def test_ema_state_dict_matches_model_keys():
    model = ASMILOrd(feature_dim=64, window=16, stride=8, hidden=32, n_branches=2)
    assert set(WeightEMA(model).state_dict()) == set(model.state_dict())


def test_ema_shadow_is_frozen():
    model = ASMILOrd(feature_dim=64, window=16, stride=8, hidden=32, n_branches=2)
    assert all(not p.requires_grad for p in WeightEMA(model).model.parameters())


def test_suggested_decay_scales_with_step_count():
    assert suggest_ema_decay(4000, 0.25) > suggest_ema_decay(400, 0.25)
    assert 0.9 < suggest_ema_decay(3000) < 1.0


# --------------------------------------------------------------------------
# Feature IO: .h5 and .pt must be interchangeable
# --------------------------------------------------------------------------
@pytest.fixture
def both_formats(tmp_path):
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((500, 64)).astype(np.float16)
    coords = rng.integers(0, 9999, (500, 2)).astype(np.int64)
    attrs = {"encoder": "h-optimus-0", "embed_dim": 64, "n_patches": 500}
    write_features(tmp_path / "P1.h5", feats, coords, attrs, "h5")
    write_features(tmp_path / "P2.pt", feats, coords, attrs, "pt")
    return tmp_path, feats, coords


def test_both_formats_read_identically(both_formats):
    tmp, feats, coords = both_formats
    h5_f, h5_c = read_features(tmp / "P1.h5", with_coords=True)
    pt_f, pt_c = read_features(tmp / "P2.pt", with_coords=True)
    assert np.allclose(h5_f, pt_f) and np.array_equal(h5_c, pt_c)
    assert h5_f.dtype == np.float32                     # promoted on read


def test_partial_read_selects_rows(both_formats):
    tmp, feats, _ = both_formats
    idx = np.array([0, 7, 99, 400])
    for name in ("P1.h5", "P2.pt"):
        got, _ = read_features(tmp / name, idx)
        assert got.shape == (4, 64)
        assert np.allclose(got, feats[idx].astype(np.float32))


def test_metadata_survives_both_formats(both_formats):
    tmp, _, _ = both_formats
    for name in ("P1.h5", "P2.pt"):
        assert read_metadata(tmp / name)["encoder"] == "h-optimus-0"


def test_counts_and_widths(both_formats):
    tmp, _, _ = both_formats
    for name in ("P1.h5", "P2.pt"):
        assert patch_count(str(tmp / name)) == 500
        assert feature_width(str(tmp / name)) == 64


def test_find_prefers_h5(both_formats, tmp_path):
    tmp, feats, coords = both_formats
    write_features(tmp / "P1.pt", feats, coords, {}, "pt")
    assert find_feature_file(tmp, "P1").suffix == ".h5"
    assert find_feature_file(tmp, "NOPE") is None


def test_verify_cohort_reports_missing(both_formats):
    tmp, _, _ = both_formats
    info = verify_cohort(tmp, ["P1", "P2", "GHOST"])
    assert info["n_present"] == 2 and info["missing"] == ["GHOST"]
    assert info["feature_dim"] == 64 and info["encoder"] == "h-optimus-0"


def test_verify_cohort_rejects_mixed_encoders(tmp_path):
    rng = np.random.default_rng(0)
    f, c = rng.standard_normal((10, 64)).astype(np.float16), np.zeros((10, 2), dtype=np.int64)
    write_features(tmp_path / "A.h5", f, c, {"encoder": "h-optimus-0", "embed_dim": 64}, "h5")
    write_features(tmp_path / "B.h5", f, c, {"encoder": "virchow", "embed_dim": 64}, "h5")
    with pytest.raises(ValueError, match="multiple encoders"):
        verify_cohort(tmp_path, ["A", "B"])


def test_verify_cohort_rejects_mixed_widths(tmp_path):
    rng = np.random.default_rng(0)
    c = np.zeros((10, 2), dtype=np.int64)
    write_features(tmp_path / "A.h5", rng.standard_normal((10, 64)).astype(np.float16), c, {}, "h5")
    write_features(tmp_path / "B.h5", rng.standard_normal((10, 128)).astype(np.float16), c, {}, "h5")
    with pytest.raises(ValueError, match="Inconsistent feature widths"):
        verify_cohort(tmp_path, ["A", "B"])


# --------------------------------------------------------------------------
# GDC helpers
# --------------------------------------------------------------------------
def test_patient_id_parsed_from_barcode():
    assert patient_id_from_barcode("TCGA-BA-4078-01Z-00-DX1") == "TCGA-BA-4078"


def test_diagnostic_vs_frozen_slides():
    dx = SlideRecord("i", "TCGA-BA-4078-01Z-00-DX1.abc.svs", 1, "TCGA-BA-4078")
    ts = SlideRecord("i", "TCGA-BA-4078-01A-01-TS1.abc.svs", 1, "TCGA-BA-4078")
    assert dx.is_diagnostic and not ts.is_diagnostic


def test_one_slide_per_patient_picks_smallest():
    recs = [
        SlideRecord("a", "TCGA-AA-1111-01Z-00-DX1.x.svs", 3_000_000_000, "TCGA-AA-1111"),
        SlideRecord("b", "TCGA-AA-1111-01Z-00-DX2.x.svs", 900_000_000, "TCGA-AA-1111"),
        SlideRecord("c", "TCGA-BB-2222-01Z-00-DX1.x.svs", 500_000_000, "TCGA-BB-2222"),
    ]
    picked = one_slide_per_patient(recs)
    assert len(picked) == 2
    assert next(r for r in picked if r.patient_id == "TCGA-AA-1111").file_id == "b"


def test_shards_are_disjoint_and_complete():
    items = list(range(97))
    parts = [shard(items, i, 5) for i in range(5)]
    flat = [x for p in parts for x in p]
    assert sorted(flat) == items and len(set(flat)) == len(items)
    assert max(len(p) for p in parts) - min(len(p) for p in parts) <= 1


def test_shard_index_is_validated():
    with pytest.raises(ValueError):
        shard(list(range(10)), 5, 5)


# --------------------------------------------------------------------------
# TPU-shape invariant
# --------------------------------------------------------------------------
def test_padded_batch_matches_unpadded_prefix():
    """XLA needs constant batch shapes; padding must not change real rows.

    encode_tiles pads the final short batch by repeating the last tile, then
    slices. This asserts the slicing is sound for any batch-independent model.
    """
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 16)).eval()
    real = torch.randn(5, 3, 8, 8)
    padded = torch.cat([real, real[-1:].expand(3, 3, 8, 8)], dim=0)
    with torch.no_grad():
        assert torch.allclose(net(real), net(padded)[:5], atol=1e-6)


# --------------------------------------------------------------------------
# encode_tiles ordering: parallel decode must not scramble features
# --------------------------------------------------------------------------
class _IndexEncoder:
    """Encodes each tile to its own first-pixel value, so order is verifiable."""

    class spec:
        embed_dim = 4
        name = "index"

    is_xla = False

    def __call__(self, batch):
        import torch as t
        vals = batch[:, 0, 0, 0].float()
        return t.stack([vals] * 4, dim=1)


class _FakeGrid:
    def __init__(self, n):
        self.coords = np.stack([np.arange(n), np.zeros(n, dtype=int)], axis=1)
        self.level = 0
        self.read_px = 4
        self.out_px = 4
        self.level0_px = 4
        self.base_mpp = 0.5
        self.scale_factor = 1.0


@pytest.mark.parametrize("n,batch_size,workers", [(384, 64, 12), (100, 64, 8), (7, 4, 4), (1, 8, 2)])
def test_encode_tiles_preserves_order(monkeypatch, n, batch_size, workers):
    """Tile i must land at row i regardless of thread scheduling."""
    import random as _random
    from pathgrade.preprocessing import stream_extract as se

    def fake_read_tile(slide, x, y, grid):
        # Sleep jitter forces threads to complete out of submission order.
        time.sleep(_random.random() * 0.001)
        return np.full((4, 4, 3), x % 251, dtype=np.uint8)

    monkeypatch.setattr(se, "read_tile", fake_read_tile)

    grid = _FakeGrid(n)
    feats = se.encode_tiles(
        object(), grid, _IndexEncoder(), lambda t: t,
        batch_size=batch_size, num_workers=workers, queue_depth=3,
    )
    assert feats.shape == (n, 4)
    expected = np.array([i % 251 for i in range(n)], dtype=np.float32)
    assert np.array_equal(feats[:, 0], expected), "features misaligned with tile order"


def test_encode_tiles_handles_ragged_final_batch(monkeypatch):
    """Padding the last short batch must not leak padded rows into the output."""
    from pathgrade.preprocessing import stream_extract as se

    monkeypatch.setattr(
        se, "read_tile",
        lambda slide, x, y, grid: np.full((4, 4, 3), x % 251, dtype=np.uint8),
    )
    n = 130                                     # 2 full batches of 64 + 2 leftover
    feats = se.encode_tiles(
        object(), _FakeGrid(n), _IndexEncoder(), lambda t: t,
        batch_size=64, num_workers=4, queue_depth=2,
    )
    assert feats.shape == (n, 4)
    assert np.array_equal(feats[:, 0], np.arange(n, dtype=np.float32) % 251)
