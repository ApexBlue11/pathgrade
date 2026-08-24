"""Tile and encode exactly one slide, synchronously, for a live prediction.

``stream_extract.py`` is built for a *cohort*: it streams from GDC, journals
progress, resumes across sessions, and processes hundreds of slides
unattended. None of that applies when a deployment receives one slide and
needs an answer back in the same request. This module is the other half -
the same tiling and encoding logic, called directly on a local file with no
session machinery around it.

Reusing :func:`build_grid`, :func:`read_tile` and :func:`encode_tiles` here
rather than re-implementing them is the point: a served slide is tiled with
the identical MPP target, tissue threshold and patch size that produced the
training embeddings. A silent mismatch there - the classic way a model works
in development and fails on a partner's data - is a parameter difference away
if the two paths ever diverge, so there is deliberately only one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..encoders import DEFAULT_ENCODER, PatchEncoder, check_licence, resolve_device
from .stream_extract import encode_tiles
from .tiling import build_grid


def encode_slide(
    slide_path: str | Path,
    encoder_name: str = DEFAULT_ENCODER,
    device: str = "auto",
    max_patches: int = 8000,
    target_mpp: float = 0.5,
    tissue_frac: float = 0.35,
    batch_size: int = 64,
    decode_workers: int = 8,
    assume_mpp: float | None = None,
    random_weights: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Tile and encode one slide. Returns ``(features, coords, attrs)``.

    The return shape matches what extraction writes to disk - ``features``
    ``[N, D]``, ``coords`` ``[N, 2]`` at slide level 0 - so the result feeds
    straight into :meth:`GradePredictor.predict` without a round trip through
    a feature file.

    ``random_weights`` skips the gated H-optimus-0 download in favour of an
    architecture-matched random init, for exercising this path (a CI check,
    an integration smoke test) without a HuggingFace token. The resulting
    embeddings are not meaningful; see ``encoders.PatchEncoder``.
    """
    import openslide

    spec = check_licence(encoder_name)
    resolved = resolve_device(device)
    encoder = PatchEncoder(spec, device=resolved, random_weights=random_weights)
    transform = encoder.build_transform()

    slide = openslide.OpenSlide(str(slide_path))
    try:
        grid = build_grid(
            slide, out_px=spec.patch_px, target_mpp=target_mpp,
            tissue_frac=tissue_frac, assume_mpp=assume_mpp, max_patches=max_patches,
        )
        if len(grid.coords) == 0:
            raise ValueError(f"no tissue detected in {slide_path}")
        features = encode_tiles(
            slide, grid, encoder, transform, batch_size, num_workers=decode_workers,
        )
    finally:
        slide.close()

    attrs = {
        "encoder": spec.name,
        "encoder_licence": spec.licence,
        "embed_dim": int(features.shape[1]),
        "target_mpp": float(target_mpp),
        "base_mpp": float(grid.base_mpp),
        "patch_px": int(grid.out_px),
        "level0_px": int(grid.level0_px),
        "n_patches": int(features.shape[0]),
    }
    return features.astype(np.float32), grid.coords.astype(np.int64), attrs


def slide_thumbnail(slide_path: str | Path, max_px: int = 1536):
    """A display thumbnail straight from the slide - no separate file needed.

    Deliberately independent of the (smaller, tissue-detection-only) thumbnail
    :func:`build_grid` renders internally: this one is sized for a human to
    look at, not for masking, and callers who only want a prediction never pay
    for it.
    """
    import openslide

    slide = openslide.OpenSlide(str(slide_path))
    try:
        w0, h0 = slide.level_dimensions[0]
        ratio = min(max_px / max(w0, h0), 1.0)
        return slide.get_thumbnail((max(1, int(w0 * ratio)), max(1, int(h0 * ratio))))
    finally:
        slide.close()
