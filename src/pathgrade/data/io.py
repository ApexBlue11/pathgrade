"""Read patch features from either ``.h5`` or ``.pt``.

``.h5`` is the default because it supports partial reads: sub-bag sampling only
ever touches a few thousand of a slide's rows, and HDF5 can fetch exactly those
instead of deserialising the whole array. ``.pt`` is supported because the
earlier Kaggle datasets used it, and re-extracting 456 GB of slides purely to
change container format would be absurd.

Both layouts carry the same two arrays: ``features`` [N, D] and ``coords`` [N, 2].
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

SUFFIXES = (".h5", ".pt")


def find_feature_file(feature_dir: str | Path, patient_id: str) -> Path | None:
    """Locate a patient's features, preferring HDF5 when both formats exist."""
    feature_dir = Path(feature_dir)
    for suffix in SUFFIXES:
        candidate = feature_dir / f"{patient_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _load_pt(path: Path):
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "features" not in obj:
        raise ValueError(f"{path} is not a dict with a 'features' key")
    return obj


@lru_cache(maxsize=4096)
def patch_count(path_str: str) -> int:
    path = Path(path_str)
    if path.suffix == ".h5":
        import h5py

        with h5py.File(path, "r") as f:
            return int(f["features"].shape[0])
    return int(_load_pt(path)["features"].shape[0])


@lru_cache(maxsize=4096)
def feature_width(path_str: str) -> int:
    path = Path(path_str)
    if path.suffix == ".h5":
        import h5py

        with h5py.File(path, "r") as f:
            return int(f["features"].shape[1])
    return int(_load_pt(path)["features"].shape[1])


def read_features(
    path: str | Path, indices: np.ndarray | None = None, with_coords: bool = False
):
    """Read features (and optionally coords), fetching only ``indices`` when possible.

    ``indices`` must be sorted and unique for the HDF5 path - h5py requires it
    for fancy indexing, and callers already sort to keep disk access sequential.
    """
    path = Path(path)

    if path.suffix == ".h5":
        import h5py

        with h5py.File(path, "r") as f:
            sel = slice(None) if indices is None else indices
            feats = np.asarray(f["features"][sel], dtype=np.float32)
            coords = np.asarray(f["coords"][sel]) if with_coords else None
    else:
        obj = _load_pt(path)
        feats = obj["features"].numpy().astype(np.float32, copy=False)
        coords = obj["coords"].numpy() if (with_coords and "coords" in obj) else None
        if indices is not None:
            feats = feats[indices]
            if coords is not None:
                coords = coords[indices]

    return (feats, coords) if with_coords else (feats, None)


def read_metadata(path: str | Path) -> dict:
    """Provenance block, if the writer stored one."""
    path = Path(path)
    if path.suffix == ".h5":
        import h5py

        with h5py.File(path, "r") as f:
            return dict(f.attrs)
    return dict(_load_pt(path).get("meta", {}))


def verify_cohort(feature_dir: str | Path, patient_ids: list[str]) -> dict:
    """Check a cohort is present, consistent in width, and from one encoder.

    Mixing encoders inside one feature directory produces a model that trains
    without error and means nothing, so this is worth failing on.
    """
    present, missing, widths, encoders, counts = [], [], set(), set(), []
    for pid in patient_ids:
        path = find_feature_file(feature_dir, pid)
        if path is None:
            missing.append(pid)
            continue
        present.append(pid)
        widths.add(feature_width(str(path)))
        counts.append(patch_count(str(path)))
        meta = read_metadata(path)
        if meta.get("encoder"):
            encoders.add(str(meta["encoder"]))

    if len(widths) > 1:
        raise ValueError(f"Inconsistent feature widths in {feature_dir}: {sorted(widths)}")
    if len(encoders) > 1:
        raise ValueError(
            f"Features in {feature_dir} came from multiple encoders: {sorted(encoders)}. "
            "Re-extract; a model trained across encoders is meaningless."
        )

    return {
        "n_present": len(present),
        "n_missing": len(missing),
        "missing": missing[:20],
        "feature_dim": widths.pop() if widths else None,
        "encoder": encoders.pop() if encoders else None,
        "median_patches": int(np.median(counts)) if counts else 0,
        "min_patches": int(min(counts)) if counts else 0,
        "max_patches": int(max(counts)) if counts else 0,
        "total_patches": int(sum(counts)),
    }
