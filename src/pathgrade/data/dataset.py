"""Slide dataset built on fixed-size sub-bags.

Whole slides have wildly different patch counts, which is why most MIL code is
stuck at batch size 1 (or pads every bag to a 10k ceiling and wastes most of
the compute on padding). nnMIL's fix is to sample a fixed number of patches per
slide, which makes bags stackable: real batches, class-balanced sampling and
large-batch optimisation all follow.

Training draws one random sub-bag per slide per epoch, which doubles as
augmentation - the model sees a different view of the same slide each time.
Evaluation instead partitions the slide into ``ceil(N / bag_size)`` disjoint
sub-bags and averages the predictions, so every patch is used exactly once and
the result is deterministic.

Features may be ``.h5`` or ``.pt``; see :mod:`pathgrade.data.io`.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .io import find_feature_file, feature_width, patch_count, read_features


def available_memory_bytes() -> int | None:
    """Free RAM, or None if it cannot be determined on this platform.

    Worth having as a real function rather than an inline /proc/meminfo read.
    That read is Linux-only and fails closed, so on Windows and macOS the
    preload cache silently never engaged - every sample re-read and gunzipped
    an HDF5 file on every epoch. It made local training slow enough to look
    like a compute problem when it was disk, and it is the second time this
    exact /proc/meminfo assumption has caused a bug in this project.
    """
    try:                                            # Linux
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass

    try:                                            # Windows
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = _Status()
        st.dwLength = ctypes.sizeof(_Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullAvailPhys)
    except Exception:
        pass

    try:                                            # POSIX fallback (macOS/BSD)
        import os as _os

        return _os.sysconf("SC_AVPHYS_PAGES") * _os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        pass
    return None


class SlideBagDataset(Dataset):
    """Reads ``<feature_dir>/<patient_id>.{h5,pt}``.

    Args:
        bag_size: patches per sub-bag. nnMIL suggests roughly half the median
            patch count; :func:`suggest_bag_size` computes that from the cohort.
        train: random sub-bag (augmentation) vs. deterministic first crop.
        return_coords: needed for heatmaps, skipped during training.
    """

    def __init__(
        self,
        patient_ids: list[str],
        labels: dict[str, int],
        feature_dir: str | Path,
        bag_size: int = 2048,
        train: bool = True,
        return_coords: bool = False,
        preload: bool | None = None,
        samples_per_slide: int = 1,
    ):
        self.feature_dir = Path(feature_dir)
        self.bag_size = bag_size
        self.train = train
        self.return_coords = return_coords
        self._cache: dict[str, np.ndarray] | None = None

        # How many independently-drawn sub-bags each slide contributes per
        # epoch. At 1 (the original behaviour) a 435-slide cohort gives ~348
        # training samples per fold and 43 gradient steps per epoch, and the
        # model drove training loss to ~0 - memorising the cohort - long
        # before validation QWK stopped improving.
        #
        # A slide of 3000 patches at bag_size 384 holds 8 disjoint sub-bags,
        # each a legitimate and differently-sampled view of the same label.
        # Drawing several per epoch is real augmentation, not resampling: it
        # multiplies gradient steps AND forces the model to reach the same
        # answer from different subsets of tissue, which is precisely the
        # pressure that was missing when attention collapsed to uniform.
        self.samples_per_slide = max(1, int(samples_per_slide)) if train else 1

        self.samples, missing = [], []
        for pid in patient_ids:
            path = find_feature_file(self.feature_dir, pid)
            if path is None:
                missing.append(pid)
            else:
                for _ in range(self.samples_per_slide):
                    self.samples.append((pid, int(labels[pid]), path))
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(patient_ids)} feature files missing under "
                f"{self.feature_dir} (first few: {missing[:5]}). Run extraction first."
            )

        if preload is None:
            preload = self._preload_is_worthwhile()
        if preload:
            self._build_cache()

    # ------------------------------------------------------------------
    def _unique_slides(self) -> dict:
        """pid -> path, one entry per slide.

        ``samples`` repeats a slide once per ``samples_per_slide``, so anything
        sizing or filling the cache must deduplicate first or it over-counts
        the memory estimate and re-reads the same file N times.
        """
        return {pid: path for pid, _lab, path in self.samples}

    def _estimated_bytes(self) -> int:
        total = 0
        for path in self._unique_slides().values():
            total += patch_count(str(path)) * feature_width(str(path)) * 4
        return total

    def _preload_is_worthwhile(self) -> bool:
        """Hold the whole cohort in RAM when it comfortably fits.

        Features are re-read every epoch, and decompressing HDF5 dominates the
        step time - a whole cohort at 3000 patches is only ~5 GB as float32,
        against the hundreds of GB a TPU VM carries. Auto-enables at under a
        quarter of available memory so a small machine still streams from disk.
        """
        try:
            need = self._estimated_bytes()
        except Exception:
            return False
        available = available_memory_bytes()
        if available is None:
            return False
        return need < available * 0.25

    def _build_cache(self) -> None:
        self._cache = {}
        for pid, path in self._unique_slides().items():
            feats, _ = read_features(path)
            self._cache[pid] = np.ascontiguousarray(feats, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def label_list(self) -> list[int]:
        return [lab for _, lab, _ in self.samples]

    def __getitem__(self, idx: int):
        pid, label, path = self.samples[idx]
        if self._cache is not None and not self.return_coords:
            cached = self._cache[pid]
            n = len(cached)
            take = self._select(n)
            feats = cached if take is None else cached[take]
            coords = None
        else:
            n = patch_count(str(path))
            take = self._select(n)
            feats, coords = read_features(path, take, with_coords=self.return_coords)

        m, d = feats.shape
        out = np.zeros((self.bag_size, d), dtype=np.float32)
        mask = np.zeros(self.bag_size, dtype=bool)
        out[:m] = feats
        mask[:m] = True

        item = {
            "features": torch.from_numpy(out),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(label, dtype=torch.long),
            "patient_id": pid,
            "n_patches": n,
        }
        if self.return_coords and coords is not None:
            padded = np.zeros((self.bag_size, 2), dtype=np.int64)
            padded[:m] = coords
            item["coords"] = torch.from_numpy(padded)
        return item

    def _select(self, n: int) -> np.ndarray | None:
        if n <= self.bag_size:
            return None                                   # read everything
        if self.train:
            # Sorted so HDF5 fancy indexing stays sequential on disk.
            return np.sort(np.random.choice(n, self.bag_size, replace=False))
        return np.arange(self.bag_size)


class SlideCropIterator:
    """Deterministic full-coverage evaluation: every patch used exactly once.

    Yields ``(features, mask)`` sub-bags for one slide. The caller averages the
    model's cumulative probabilities across crops.
    """

    def __init__(self, path: str | Path, bag_size: int):
        self.path = Path(path)
        self.bag_size = bag_size

    def __iter__(self):
        feats, _ = read_features(self.path)
        n, d = feats.shape
        n_crops = max(1, int(np.ceil(n / self.bag_size)))
        for c in range(n_crops):
            chunk = feats[c * self.bag_size : (c + 1) * self.bag_size]
            out = np.zeros((self.bag_size, d), dtype=np.float32)
            mask = np.zeros(self.bag_size, dtype=bool)
            out[: len(chunk)] = chunk
            mask[: len(chunk)] = True
            yield torch.from_numpy(out), torch.from_numpy(mask)


def balanced_sampler(labels: list[int], seed: int | None = None) -> WeightedRandomSampler:
    """Class-balanced sampling.

    Preferred over loss re-weighting: with sub-bags the epoch is already an
    arbitrary sample of slide views, so rebalancing *which* slides appear costs
    nothing and keeps the loss a clean likelihood.
    """
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True, generator=generator)


def suggest_bag_size(feature_dir: str | Path, patient_ids: list[str], cap: int = 4096) -> int:
    """Half the median patch count, rounded to a multiple of 256 and capped.

    Follows nnMIL's fingerprinting idea: let the cohort choose the setting
    rather than hard-coding a ceiling that happens to suit one dataset.
    """
    counts = [
        patch_count(str(p))
        for p in (find_feature_file(feature_dir, pid) for pid in patient_ids)
        if p is not None
    ]
    if not counts:
        return 2048
    half = int(np.median(counts) / 2)
    return min(max(256, int(round(half / 256)) * 256), cap)


def feature_dim(feature_dir: str | Path, patient_ids: list[str]) -> int:
    """Read the encoder width off disk so configs cannot drift from the data."""
    for pid in patient_ids:
        path = find_feature_file(feature_dir, pid)
        if path is not None:
            return feature_width(str(path))
    raise FileNotFoundError(f"No feature files found in {feature_dir}")
