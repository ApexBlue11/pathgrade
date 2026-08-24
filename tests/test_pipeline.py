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
from pathgrade.data.dataset import SlideBagDataset
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


# --------------------------------------------------------------------------
# Parallel slide prefetch: concurrency must not exceed the disk budget
# --------------------------------------------------------------------------
def test_prefetcher_bounds_slides_on_disk(monkeypatch, tmp_path):
    """Downloads run concurrently, but never more than max_on_disk land at once."""
    import threading
    from pathgrade.preprocessing import stream_extract as se
    from pathgrade.preprocessing.gdc import SlideRecord

    on_disk = 0
    peak = 0
    lock = threading.Lock()

    def fake_download(record, cache_dir, **kw):
        nonlocal on_disk, peak
        with lock:
            on_disk += 1
            peak = max(peak, on_disk)
        time.sleep(0.02)
        p = Path(cache_dir) / record.file_name
        p.write_bytes(b"x")
        return p

    monkeypatch.setattr(se, "download_slide", fake_download)

    records = [SlideRecord(f"id{i}", f"S{i}.svs", 100, f"P{i}") for i in range(24)]
    pf = se.SlidePrefetcher(records, tmp_path, max_on_disk=3, workers=8).start()

    seen = []
    for record, path, _info in pf:
        seen.append(record.patient_id)
        Path(path).unlink()
        with lock:
            on_disk -= 1
        pf.release()

    # Delivery order is deliberately NOT guaranteed any more: results are
    # yielded as downloads complete so a slow one cannot block slides already
    # on disk. Completeness is what matters - slides are independent and each
    # writes its own file.
    assert sorted(seen) == sorted(r.patient_id for r in records), "every slide exactly once"
    assert peak <= 3, f"peak {peak} slides on disk exceeded max_on_disk=3"
    assert peak > 1, "downloads should overlap, not serialise"


def test_prefetcher_reports_failures_without_consuming_a_slot(monkeypatch, tmp_path):
    from pathgrade.preprocessing import stream_extract as se
    from pathgrade.preprocessing.gdc import SlideRecord

    def flaky(record, cache_dir, **kw):
        if record.patient_id == "P1":
            raise IOError("network went away")
        p = Path(cache_dir) / record.file_name
        p.write_bytes(b"x")
        return p

    monkeypatch.setattr(se, "download_slide", flaky)
    records = [SlideRecord(f"id{i}", f"S{i}.svs", 10, f"P{i}") for i in range(4)]
    pf = se.SlidePrefetcher(records, tmp_path, max_on_disk=2, workers=2).start()

    results = []
    for record, path, info in pf:
        results.append((record.patient_id, path is not None))
        if path:
            Path(path).unlink()
            pf.release()

    assert ("P1", False) in results
    assert len(results) == 4, "a failure must not stall the queue"


# --------------------------------------------------------------------------
# Inference-tensor regression
#
# Every slide failed on TPU with "Cannot set version_counter for inference
# tensor". Two causes, both here: the normalisation constants were created
# lazily inside an @torch.inference_mode() forward - making them inference
# tensors cached on the module - and the scaling used an in-place div_.
# --------------------------------------------------------------------------
def _stub_encoder(spec_name="h-optimus-0", monkeypatch=None):
    """A REAL PatchEncoder, with only the timm backbone swapped for a tiny stub.

    Constructing the genuine object matters: the bug lived in __init__ (or
    rather in its absence - the constants were built lazily in forward), so a
    hand-rolled stub would have tested nothing.
    """
    import timm
    import torch.nn as nn
    from pathgrade.encoders import PatchEncoder, get_spec

    spec = get_spec(spec_name)
    real_create = timm.create_model

    def tiny(*args, **kwargs):
        return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                             nn.Linear(3, spec.embed_dim))

    timm.create_model = tiny
    try:
        return PatchEncoder(spec, device="cpu")
    finally:
        timm.create_model = real_create


def test_normalisation_constants_are_buffers_not_inference_tensors():
    """Built in __init__, outside inference mode - this is the actual fix."""
    enc = _stub_encoder()
    buffers = dict(enc.named_buffers())
    assert "_mean" in buffers and "_std" in buffers
    assert not torch.is_inference(enc._mean)
    assert not torch.is_inference(enc._std)


def test_real_encoder_forward_accepts_uint8_under_inference_mode():
    """End-to-end reproduction of the TPU failure on CPU."""
    enc = _stub_encoder()
    batch = torch.randint(0, 256, (4, 224, 224, 3), dtype=torch.uint8)
    with torch.inference_mode():
        out = enc(batch)
    assert out.shape == (4, enc.spec.embed_dim)
    assert not torch.is_inference(enc._mean), "constants were poisoned"
    # A second call must still work; the original bug only bit after caching.
    assert enc(batch).shape == (4, enc.spec.embed_dim)


def test_normalise_survives_being_called_under_inference_mode():
    """The exact TPU failure: constants must not become inference tensors."""
    enc = _stub_encoder()
    x = torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8)
    with torch.inference_mode():
        out = enc._normalise(x)
    assert out.shape == (2, 3, 224, 224)
    # Constants must survive unpoisoned, so the next call still works.
    assert not torch.is_inference(enc._mean)
    again = enc._normalise(torch.randint(0, 256, (1, 224, 224, 3), dtype=torch.uint8))
    assert again.shape == (1, 3, 224, 224)


def test_normalise_does_not_mutate_its_input():
    """In-place ops are the other half of the inference-tensor trap."""
    enc = _stub_encoder()
    x = torch.randint(0, 256, (2, 8, 8, 3), dtype=torch.uint8)
    before = x.clone()
    enc._normalise(x)
    assert torch.equal(x, before)


def test_normalise_matches_the_reference_maths():
    from pathgrade.encoders import get_spec

    enc = _stub_encoder()
    spec = get_spec("h-optimus-0")
    x = torch.randint(0, 256, (3, 16, 16, 3), dtype=torch.uint8)
    expected = (
        x.permute(0, 3, 1, 2).float() / 255.0
        - torch.tensor(spec.mean).view(1, 3, 1, 1)
    ) / torch.tensor(spec.std).view(1, 3, 1, 1)
    assert torch.allclose(enc._normalise(x), expected, atol=1e-6)


def test_encode_tiles_output_is_not_an_inference_tensor(monkeypatch):
    """encode_tiles must hand back plain arrays usable by anything downstream."""
    from pathgrade.preprocessing import stream_extract as se

    monkeypatch.setattr(
        se, "read_tile",
        lambda slide, x, y, grid: np.full((4, 4, 3), x % 251, dtype=np.uint8),
    )

    class Enc:
        class spec:
            embed_dim = 4
            name = "stub"
        is_xla = False

        def __call__(self, batch):
            with torch.inference_mode():          # what the real encoder did
                vals = batch[:, 0, 0, 0].float()
                return torch.stack([vals] * 4, dim=1)

    feats = se.encode_tiles(object(), _FakeGrid(40), Enc(), lambda t: t,
                            batch_size=16, num_workers=4, queue_depth=2)
    assert feats.shape == (40, 4)
    assert np.isfinite(feats).all()


def test_width_probe_rejects_a_mismatched_architecture():
    """A wrong-width backbone must fail at construction, not mid-slide.

    Previously this surfaced as a numpy broadcast error inside encode_tiles,
    once per slide, with nothing naming the cause.
    """
    import timm
    import torch.nn as nn
    from pathgrade.encoders import PatchEncoder, get_spec

    spec = get_spec("h-optimus-0")               # declares 1536
    real = timm.create_model

    def wrong_width(*a, **k):
        return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 1408))

    timm.create_model = wrong_width
    try:
        with pytest.raises(RuntimeError, match="produced width 1408.*declares 1536"):
            PatchEncoder(spec, device="cpu")
    finally:
        timm.create_model = real


def test_width_probe_accepts_a_matching_architecture():
    enc = _stub_encoder()                        # stub is built at spec width
    assert enc.spec.embed_dim == 1536


@pytest.mark.parametrize("dim", [1536, 2560, 1024, 3072])
def test_random_arch_table_covers_every_registry_width(dim):
    from pathgrade.encoders import RANDOM_ARCH, REGISTRY

    widths = {s.embed_dim for s in REGISTRY.values()}
    assert widths <= set(RANDOM_ARCH), f"uncovered widths: {widths - set(RANDOM_ARCH)}"
    assert dim in RANDOM_ARCH


@pytest.fixture
def features(tmp_path):
    """A small cohort on disk, with a spread of patch counts."""
    import h5py

    rng = np.random.default_rng(1)
    d = tmp_path / "feat"
    d.mkdir()
    labels, counts = {}, {}
    for i in range(8):
        pid = f"Q{i:03d}"
        n = int(rng.integers(200, 1200))
        counts[pid] = n
        labels[pid] = int(rng.integers(0, 3))
        with h5py.File(d / f"{pid}.h5", "w") as f:
            f.create_dataset("features", data=rng.standard_normal((n, 32)).astype(np.float16))
            f.create_dataset("coords", data=rng.integers(0, 9999, (n, 2)))
    return d, labels, counts


# --------------------------------------------------------------------------
# In-RAM feature cache
#
# Features are re-read every epoch and HDF5 decompression dominates the step
# time, so the whole cohort is held in memory when it fits. The cached branch
# needs explicit tests: auto-detection reads /proc/meminfo, so on any non-Linux
# dev machine it silently stays off and the code path goes untested.
# --------------------------------------------------------------------------
def test_preloaded_dataset_matches_streaming_dataset(features):
    d, labels, counts = features
    ids = sorted(labels)
    streamed = SlideBagDataset(ids, labels, d, bag_size=512, train=False, preload=False)
    cached = SlideBagDataset(ids, labels, d, bag_size=512, train=False, preload=True)

    assert cached._cache is not None and streamed._cache is None
    for i in range(len(ids)):
        a, b = streamed[i], cached[i]
        assert a["patient_id"] == b["patient_id"]
        assert a["n_patches"] == b["n_patches"], "n_patches must survive the cached path"
        assert torch.equal(a["mask"], b["mask"])
        assert torch.allclose(a["features"], b["features"])


def test_preload_populates_every_slide(features):
    d, labels, _ = features
    ids = sorted(labels)
    ds = SlideBagDataset(ids, labels, d, bag_size=256, preload=True)
    assert set(ds._cache) == set(ids)
    assert all(v.dtype == np.float32 for v in ds._cache.values())


def test_preload_off_leaves_no_cache(features):
    d, labels, _ = features
    ds = SlideBagDataset(sorted(labels), labels, d, bag_size=256, preload=False)
    assert ds._cache is None


def test_preload_declined_when_cohort_dwarfs_memory(features, monkeypatch):
    """Auto-detection must stream rather than exhaust a small machine."""
    d, labels, _ = features
    monkeypatch.setattr(
        SlideBagDataset, "_estimated_bytes", lambda self: 10 ** 15
    )
    ds = SlideBagDataset(sorted(labels), labels, d, bag_size=256, preload=None)
    assert ds._cache is None


def test_cached_training_still_randomises_sub_bags(features):
    """Caching must not accidentally freeze the augmentation."""
    d, labels, _ = features
    big = [p for p in sorted(labels) if patch_count(str(find_feature_file(d, p))) > 600]
    if not big:
        pytest.skip("fixture has no slide larger than the bag")
    ds = SlideBagDataset(big[:1], labels, d, bag_size=256, train=True, preload=True)
    first, second = ds[0]["features"], ds[0]["features"]
    assert not torch.allclose(first, second), "sub-bag sampling should differ per call"


# --------------------------------------------------------------------------
# Multi-device encode: data parallelism must not change the answer
#
# The whole point of spreading tiles across eight XLA devices is throughput,
# so the one thing that must not move is the output. These pin the parallel
# path against the serial one that was validated on 60 real slides.
# --------------------------------------------------------------------------
import threading as _threading


class _SlowIndexEncoder(_IndexEncoder):
    """_IndexEncoder that records its own batch count and yields the GIL.

    The sleep is what lets a second replica pick work off the queue, which is
    what makes the distribution assertion meaningful rather than incidental.
    """

    def __init__(self, delay: float = 0.002):
        self.calls = 0
        self.delay = delay
        self._lock = _threading.Lock()

    def __call__(self, batch):
        with self._lock:
            self.calls += 1
        time.sleep(self.delay)
        return super().__call__(batch)


@pytest.mark.parametrize("n_devices", [2, 4, 8])
def test_encode_tiles_multi_device_preserves_order(monkeypatch, n_devices):
    """Tile i lands at row i no matter which replica encoded its batch."""
    from pathgrade.preprocessing import stream_extract as se

    monkeypatch.setattr(
        se, "read_tile",
        lambda slide, x, y, grid: np.full((4, 4, 3), x % 251, dtype=np.uint8),
    )
    n = 500
    replicas = [_SlowIndexEncoder() for _ in range(n_devices)]
    feats = se.encode_tiles(
        object(), _FakeGrid(n), replicas, lambda t: t,
        batch_size=32, num_workers=8, queue_depth=2,
    )
    assert feats.shape == (n, 4)
    assert np.array_equal(feats[:, 0], np.arange(n, dtype=np.float32) % 251)


def test_multi_device_output_is_identical_to_single_device(monkeypatch):
    """Byte-for-byte agreement with the serial path, ragged tail included."""
    from pathgrade.preprocessing import stream_extract as se

    monkeypatch.setattr(
        se, "read_tile",
        lambda slide, x, y, grid: np.full((4, 4, 3), (x * 7) % 251, dtype=np.uint8),
    )
    n = 333                                   # deliberately not a batch multiple
    grid = _FakeGrid(n)
    serial = se.encode_tiles(object(), grid, _IndexEncoder(), lambda t: t,
                             batch_size=64, num_workers=4, queue_depth=3)
    parallel = se.encode_tiles(object(), grid, [_SlowIndexEncoder(0.0) for _ in range(4)],
                               lambda t: t, batch_size=64, num_workers=4, queue_depth=3)
    assert np.array_equal(serial, parallel)


def test_multi_device_actually_spreads_work(monkeypatch):
    """More than one replica must do work, or the parallelism is decorative."""
    from pathgrade.preprocessing import stream_extract as se

    monkeypatch.setattr(
        se, "read_tile",
        lambda slide, x, y, grid: np.full((4, 4, 3), x % 251, dtype=np.uint8),
    )
    n, batch_size = 640, 32
    replicas = [_SlowIndexEncoder(0.004) for _ in range(4)]
    se.encode_tiles(object(), _FakeGrid(n), replicas, lambda t: t,
                    batch_size=batch_size, num_workers=8, queue_depth=2)

    assert sum(r.calls for r in replicas) == n // batch_size
    assert sum(1 for r in replicas if r.calls) >= 2


def test_multi_device_failure_raises_instead_of_returning_partial(monkeypatch):
    """A dead replica must not yield a half-filled array that looks valid.

    np.empty is uninitialised, so swallowing this would hand back plausible
    garbage embeddings - the worst possible failure for a grading model.
    """
    from pathgrade.preprocessing import stream_extract as se

    monkeypatch.setattr(
        se, "read_tile",
        lambda slide, x, y, grid: np.full((4, 4, 3), x % 251, dtype=np.uint8),
    )

    class _Broken(_IndexEncoder):
        def __call__(self, batch):
            raise RuntimeError("device fell over")

    with pytest.raises(RuntimeError, match="device fell over"):
        se.encode_tiles(object(), _FakeGrid(256), [_Broken(), _Broken()], lambda t: t,
                        batch_size=32, num_workers=4, queue_depth=2)


def test_single_encoder_still_accepted_unwrapped(monkeypatch):
    """The existing call signature must keep working untouched."""
    from pathgrade.preprocessing import stream_extract as se

    monkeypatch.setattr(
        se, "read_tile",
        lambda slide, x, y, grid: np.full((4, 4, 3), x % 251, dtype=np.uint8),
    )
    feats = se.encode_tiles(object(), _FakeGrid(64), _IndexEncoder(), lambda t: t,
                            batch_size=32, num_workers=4, queue_depth=2)
    assert feats.shape == (64, 4)


def test_xla_devices_empty_without_torch_xla():
    """Off TPU this must degrade to the single-device path, not explode."""
    from pathgrade.encoders import xla_devices

    devices = xla_devices()
    assert isinstance(devices, list)
    if not torch.cuda.is_available():
        try:
            import torch_xla  # noqa: F401
        except Exception:
            assert devices == []


def test_build_encoders_returns_single_replica_off_tpu(monkeypatch):
    """max_devices>1 on a CPU box must still give exactly one encoder."""
    from pathgrade import encoders as enc_mod

    built = []

    class _Stub:
        def __init__(self, spec, device=None, random_weights=False):
            built.append(device)
            self.spec, self.device = spec, torch.device("cpu")

    monkeypatch.setattr(enc_mod, "PatchEncoder", _Stub)
    out = enc_mod.build_encoders(enc_mod.get_spec("h-optimus-0"),
                                 device=torch.device("cpu"), max_devices=8)
    assert len(out) == 1, "non-XLA devices must not be replicated"


def test_build_encoders_refuses_to_thread_across_xla_devices(monkeypatch):
    """XLA multi-device threading is measured-broken; degrade, do not crash.

    cores_probe.py on a real v5e-8: 2 threads fine, 3/4 dead at four devices,
    7/8 dead at eight, all in SyncLiveTensorsGraph. Shipping that would fail
    every slide of a multi-hour extraction, so build_encoders keeps one replica
    unless explicitly forced.
    """
    from pathgrade import encoders as enc_mod

    class _Stub:
        def __init__(self, spec, device=None, random_weights=False):
            self.spec, self.device = spec, torch.device("xla:0")

    monkeypatch.setattr(enc_mod, "PatchEncoder", _Stub)
    monkeypatch.setattr(enc_mod, "xla_devices",
                        lambda: [torch.device(f"xla:{i}") for i in range(8)])
    monkeypatch.delenv("PATHGRADE_FORCE_XLA_THREADS", raising=False)

    out = enc_mod.build_encoders(enc_mod.get_spec("h-optimus-0"),
                                 device=torch.device("xla:0"), max_devices=8)
    assert len(out) == 1, "must not replicate across XLA devices by default"

    monkeypatch.setenv("PATHGRADE_FORCE_XLA_THREADS", "1")
    forced = enc_mod.build_encoders(enc_mod.get_spec("h-optimus-0"),
                                    device=torch.device("xla:0"), max_devices=8)
    assert len(forced) == 8, "explicit override must still work"


# --------------------------------------------------------------------------
# Prefetcher: a completed download must not wait behind a slow one
#
# Measured on a real 79-slide chunk: 2142 s of per-slide work against a 3158 s
# wall. Download times are skewed (median 16 s, max 63 s), and in-order
# delivery stalled the encoder behind the slowest one while other slides sat
# finished on disk. Slides are independent, so that ordering was pure cost.
# --------------------------------------------------------------------------
def test_prefetcher_delivers_out_of_order_when_one_download_is_slow(monkeypatch, tmp_path):
    """A fast slide must overtake a slow one instead of queueing behind it."""
    from pathgrade.preprocessing import stream_extract as se

    def fake_download(record, cache_dir):
        time.sleep(0.60 if record.patient_id == "P0" else 0.02)
        p = Path(cache_dir) / record.file_name
        p.write_bytes(b"x")
        return p

    monkeypatch.setattr(se, "download_slide", fake_download)
    records = [SlideRecord(f"id{i}", f"S{i}.svs", 100, f"P{i}") for i in range(6)]

    pf = se.SlidePrefetcher(records, tmp_path, max_on_disk=6, workers=6).start()
    order = []
    for record, path, _info in pf:
        order.append(record.patient_id)
        Path(path).unlink(missing_ok=True)
        pf.release()

    assert sorted(order) == sorted(r.patient_id for r in records), "every slide exactly once"
    assert order[-1] == "P0", "the slow download must not block the five fast ones"


def test_prefetcher_out_of_order_still_bounds_disk(monkeypatch, tmp_path):
    """Yielding as-completed must not loosen the slides-on-disk budget."""
    from pathgrade.preprocessing import stream_extract as se
    import threading as _t

    live = peak = 0
    lock = _t.Lock()

    def fake_download(record, cache_dir):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.02)
        p = Path(cache_dir) / record.file_name
        p.write_bytes(b"x")
        return p

    monkeypatch.setattr(se, "download_slide", fake_download)
    records = [SlideRecord(f"id{i}", f"S{i}.svs", 100, f"P{i}") for i in range(20)]

    pf = se.SlidePrefetcher(records, tmp_path, max_on_disk=3, workers=8).start()
    seen = 0
    for record, path, _info in pf:
        seen += 1
        Path(path).unlink(missing_ok=True)
        with lock:
            live -= 1
        pf.release()

    assert seen == 20
    assert peak <= 3, f"peak {peak} in flight exceeds max_on_disk=3"
    assert peak > 1, "downloads should still overlap"


# --------------------------------------------------------------------------
# Single-slide inference path: encode_slide / slide_thumbnail / grade_slide
#
# The commercial deployment never runs the cohort extraction CLI - it grades
# one uploaded slide and needs a grade plus an attention overlay back, with
# no separate thumbnail file. These pin the wiring between tiling, the shared
# encode_tiles used by cohort extraction, and GradePredictor.
# --------------------------------------------------------------------------
class _FakeOpenSlide:
    """Stands in for openslide.OpenSlide - never touches a real file."""

    closed = False

    def __init__(self, path):
        self.path = path

    def close(self):
        self.closed = True


class _StubEncoder:
    """Lightweight stand-in for PatchEncoder - avoids building a real 1B-param
    ViT-g just to test that encode_slide wires its arguments correctly."""

    def __init__(self, spec, device=None, random_weights=False):
        self.spec = spec
        self.is_xla = False

    def build_transform(self):
        return lambda t: t


def test_encode_slide_returns_extraction_shaped_output(monkeypatch):
    from pathgrade.preprocessing import single_slide as ss

    monkeypatch.setattr("openslide.OpenSlide", _FakeOpenSlide)
    monkeypatch.setattr(ss, "PatchEncoder", _StubEncoder)
    monkeypatch.setattr(ss, "build_grid", lambda *a, **k: _FakeGrid(37))

    def fake_encode_tiles(slide, grid, encoder, transform, batch_size, **kw):
        assert isinstance(slide, _FakeOpenSlide)
        return np.zeros((len(grid.coords), encoder.spec.embed_dim), dtype=np.float32)

    monkeypatch.setattr(ss, "encode_tiles", fake_encode_tiles)

    feats, coords, attrs = ss.encode_slide("fake.svs", device="cpu")

    assert feats.shape == (37, 1536)
    assert coords.shape == (37, 2)
    assert attrs["encoder"] == "h-optimus-0"
    assert attrs["n_patches"] == 37
    assert attrs["encoder_licence"] == "Apache-2.0"


def test_encode_slide_closes_the_slide_even_on_failure(monkeypatch):
    """A tiling error must not leak an open OpenSlide handle."""
    from pathgrade.preprocessing import single_slide as ss

    opened = {}

    class TrackedSlide(_FakeOpenSlide):
        def __init__(self, path):
            super().__init__(path)
            opened["handle"] = self

    monkeypatch.setattr("openslide.OpenSlide", TrackedSlide)
    monkeypatch.setattr(ss, "PatchEncoder", _StubEncoder)

    def explode(*a, **k):
        raise RuntimeError("corrupt tile")

    monkeypatch.setattr(ss, "build_grid", explode)

    with pytest.raises(RuntimeError, match="corrupt tile"):
        ss.encode_slide("fake.svs", device="cpu")
    assert opened["handle"].closed, "slide must be closed even when tiling raises"


def test_encode_slide_rejects_a_slide_with_no_tissue(monkeypatch):
    from pathgrade.preprocessing import single_slide as ss

    monkeypatch.setattr("openslide.OpenSlide", _FakeOpenSlide)
    monkeypatch.setattr(ss, "PatchEncoder", _StubEncoder)
    monkeypatch.setattr(ss, "build_grid", lambda *a, **k: _FakeGrid(0))

    with pytest.raises(ValueError, match="no tissue"):
        ss.encode_slide("fake.svs", device="cpu")


def test_slide_thumbnail_scales_to_the_requested_max_side(monkeypatch):
    from pathgrade.preprocessing import single_slide as ss

    class Sized(_FakeOpenSlide):
        level_dimensions = [(4000, 2000)]

        def get_thumbnail(self, size):
            return size          # return the requested size for inspection

    monkeypatch.setattr("openslide.OpenSlide", Sized)
    size = ss.slide_thumbnail("fake.svs", max_px=1000)
    assert size == (1000, 500), "aspect ratio must be preserved when scaling to max_px"


def test_grade_slide_wires_encode_predict_and_overlay_together(monkeypatch):
    """The one call a deployment makes: slide in, (prediction, image) out."""
    from pathgrade import inference as inf

    n = 20
    coords = np.stack([np.arange(n) * 10, np.zeros(n, dtype=int)], axis=1)
    attrs = {"level0_px": 224, "n_patches": n}

    monkeypatch.setattr(
        "pathgrade.preprocessing.single_slide.encode_slide",
        lambda path, device=None, **kw: (
            np.random.default_rng(0).standard_normal((n, 8)).astype(np.float32),
            coords, attrs,
        ),
    )
    monkeypatch.setattr(
        "pathgrade.preprocessing.single_slide.slide_thumbnail",
        lambda path, max_px=1536: __import__("PIL.Image", fromlist=["Image"]).new(
            "RGB", (64, 64)),
    )

    class _Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, x, mask):
            import types

            return types.SimpleNamespace(logits=torch.zeros(1, 2))

        def patch_attention(self, x, mask):
            return torch.ones(1, n) / n

    predictor = inf.GradePredictor([_Model()], config=type(
        "C", (), {"model": type("M", (), {"feature_dim": 8})()})(), device=torch.device("cpu"))
    monkeypatch.setattr(inf.GradePredictor, "from_run", classmethod(lambda cls, *a, **k: predictor))

    prediction, overlay = inf.grade_slide("fake.svs", "runs/whatever")

    assert prediction.n_patches == n
    assert overlay.size == (64, 64)


# --------------------------------------------------------------------------
# Attention collapse: the failure that shipped once and must not ship again
#
# The first real training run ended with attention EXACTLY uniform - top 1% of
# patches holding 1.0% of the mass, normalised entropy 1.0000 - so the model
# was mean-pooling and the heatmap sold as the product's explanation was flat
# noise. Nothing detected it because nothing was measuring it.
# --------------------------------------------------------------------------
def test_attention_guard_rejects_a_uniform_map():
    """A flat heatmap must be refused, not rendered as an explanation."""
    from pathgrade.inference import GradePrediction, attention_is_informative

    n = 3000
    pred = GradePrediction(
        grade=1, grade_label="G2", probabilities=np.array([0.2, 0.6, 0.2]),
        cumulative=np.array([0.87, 0.28]), confidence=0.6,
        attention=np.ones(n) / n, coords=np.zeros((n, 2), dtype=int),
        patch_size=224, n_patches=n,
    )
    ok, reason = attention_is_informative(pred)
    assert not ok
    assert "uniform" in reason.lower()


def test_attention_guard_accepts_a_concentrated_map():
    from pathgrade.inference import GradePrediction, attention_is_informative

    n = 3000
    a = np.ones(n)
    a[:30] = 200.0                       # top 1% carries most of the mass
    pred = GradePrediction(
        grade=1, grade_label="G2", probabilities=np.array([0.2, 0.6, 0.2]),
        cumulative=np.array([0.87, 0.28]), confidence=0.6,
        attention=a, coords=np.zeros((n, 2), dtype=int), patch_size=224, n_patches=n,
    )
    ok, _ = attention_is_informative(pred)
    assert ok


def test_model_reports_attention_entropy_so_collapse_is_visible_in_training():
    """Uniform attention must show as entropy ~1.0 in the training log."""
    from pathgrade.models.asmil_ord import _mean_normalised_entropy

    b, n, k = 2, 512, 5
    mask = torch.ones(b, n, dtype=torch.bool)

    uniform = torch.full((b, n, k), 1.0 / n)
    assert _mean_normalised_entropy(uniform, mask).item() == pytest.approx(1.0, abs=1e-3)

    peaked = torch.full((b, n, k), 1e-6)
    peaked[:, 0, :] = 1.0
    assert _mean_normalised_entropy(peaked, mask).item() < 0.2


def test_ambiguity_beats_fold_spread_at_flagging_a_borderline_slide():
    """The shipped `uncertainty` scored AUC 0.500 at detecting its own errors.

    `ambiguity` is distance to the CORN decision threshold, which is what
    actually separates a decisive call from a coin flip.
    """
    from pathgrade.inference import GradePrediction

    def mk(cum):
        return GradePrediction(
            grade=1, grade_label="G2", probabilities=np.array([0.2, 0.6, 0.2]),
            cumulative=np.array(cum), confidence=0.6, attention=np.ones(10) / 10,
            coords=np.zeros((10, 2), dtype=int), patch_size=224, n_patches=10,
        )

    decisive = mk([0.97, 0.03])
    borderline = mk([0.51, 0.49])
    assert decisive.ambiguity < 0.05
    assert borderline.ambiguity > 0.45
    assert borderline.ambiguity > decisive.ambiguity


def test_entropy_penalty_is_off_by_default_but_reports_raw_entropy():
    """Enabling the penalty must be a deliberate act, but the number is always logged."""
    from pathgrade.losses import ASMILOrdLoss

    logits = torch.zeros(4, 2, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 1])
    aux = {"stabilisation": torch.tensor(0.01), "diversity": torch.tensor(0.1),
           "attn_entropy": torch.tensor(0.97)}

    _, parts = ASMILOrdLoss(3).forward(logits, targets, aux)
    assert "attn_entropy" not in parts, "penalty must default to off"
    assert parts["attn_entropy_raw"] == pytest.approx(0.97), "raw entropy always reported"

    _, parts = ASMILOrdLoss(3, lambda_attn_entropy=0.5).forward(logits, targets, aux)
    assert parts["attn_entropy"] == pytest.approx(0.5 * 0.97)


def test_samples_per_slide_multiplies_the_training_set(tmp_path):
    """348 samples/epoch is what let the model memorise the cohort.

    Each slide holds several disjoint sub-bags; drawing more than one per
    epoch is real augmentation (different tissue, same label), not resampling.
    """
    from pathgrade.data.dataset import SlideBagDataset

    rng = np.random.default_rng(0)
    ids = [f"P{i}" for i in range(6)]
    labels = {p: i % 3 for i, p in enumerate(ids)}
    for p in ids:
        write_features(tmp_path / f"{p}.h5",
                       rng.standard_normal((800, 32)).astype(np.float16),
                       rng.integers(0, 999, (800, 2)).astype(np.int64),
                       {"encoder": "h-optimus-0", "embed_dim": 32}, "h5")

    one = SlideBagDataset(ids, labels, tmp_path, bag_size=128, train=True)
    four = SlideBagDataset(ids, labels, tmp_path, bag_size=128, train=True, samples_per_slide=4)
    assert len(one) == 6
    assert len(four) == 24, "4 sub-bags per slide should quadruple the epoch"

    # class balance must be preserved, or balanced sampling breaks
    from collections import Counter
    assert Counter(four.label_list) == {k: v * 4 for k, v in Counter(one.label_list).items()}

    # and the drawn bags must actually differ, otherwise it is just duplication
    a, b = four[0]["features"].numpy(), four[1]["features"].numpy()
    assert four[0]["patient_id"] == four[1]["patient_id"], "same slide, two draws"
    assert not np.array_equal(a, b), "repeated draws must sample different patches"


def test_samples_per_slide_is_ignored_at_eval(tmp_path):
    """Evaluation must stay deterministic and use every patch exactly once."""
    from pathgrade.data.dataset import SlideBagDataset

    rng = np.random.default_rng(1)
    ids = ["P0", "P1"]
    labels = {p: 1 for p in ids}
    for p in ids:
        write_features(tmp_path / f"{p}.h5",
                       rng.standard_normal((400, 32)).astype(np.float16),
                       rng.integers(0, 999, (400, 2)).astype(np.int64),
                       {"encoder": "h-optimus-0", "embed_dim": 32}, "h5")

    ds = SlideBagDataset(ids, labels, tmp_path, bag_size=128, train=False, samples_per_slide=8)
    assert len(ds) == 2, "eval must see each slide once regardless of the training knob"


def test_entropy_gradient_vanishes_at_uniform_attention():
    """Why the entropy penalty could not un-collapse attention.

    Uniform is the maximum of entropy, so the gradient there is ~0 and there
    is no downhill direction out of it at any coefficient. Enabling the
    penalty on a real run moved entropy 0.9975 -> 0.9998 (more uniform) while
    CV QWK fell. Pinned so nobody reaches for this lever again.
    """
    from pathgrade.models.asmil_ord import _mean_normalised_entropy

    n = 512
    mask = torch.ones(1, n, dtype=torch.bool)

    scores_uniform = torch.zeros(1, n, 1, requires_grad=True)
    a = torch.softmax(scores_uniform, dim=1)
    _mean_normalised_entropy(a, mask).backward()
    grad_at_uniform = scores_uniform.grad.abs().max().item()

    scores_peaked = torch.randn(1, n, 1) * 3.0
    scores_peaked.requires_grad_(True)
    a = torch.softmax(scores_peaked, dim=1)
    _mean_normalised_entropy(a, mask).backward()
    grad_when_peaked = scores_peaked.grad.abs().max().item()

    assert grad_at_uniform < 1e-6, "entropy is stationary at uniform - no escape direction"
    assert grad_when_peaked > grad_at_uniform * 100, "away from uniform there IS a gradient"


def test_scorer_can_be_exempted_from_weight_decay():
    """Decay was the only force acting on the scorer once the head fit the mean.

    It shrank the scorer below its own initialisation in both real runs
    (std 0.018, then 0.008, against an init of ~0.036), which is the mechanism
    behind the uniform attention.
    """
    from pathgrade.config import Config
    from pathgrade.train import build_param_groups

    model = ASMILOrd(feature_dim=64, window=16, stride=8, hidden=32, n_branches=2)

    cfg = Config()
    cfg.optim.weight_decay = 0.01
    cfg.optim.scorer_no_decay = False
    groups = {g["name"]: g for g in build_param_groups(model, cfg)}
    assert "weight_decay" not in groups["scorer"] or groups["scorer"]["weight_decay"] == 0.01

    cfg.optim.scorer_no_decay = True
    groups = {g["name"]: g for g in build_param_groups(model, cfg)}
    assert groups["scorer"]["weight_decay"] == 0.0, "scorer must be exempt when asked"
    assert "weight_decay" not in groups["head"], "the head must keep the global decay"
