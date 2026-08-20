"""Resolve a WSI to a list of tissue patch coordinates at a target resolution.

Everything is expressed in **microns per pixel**, not magnification labels.
A slide scanned at "20x" on one vendor's scanner is not the same resolution as
another's, and mixing them silently is a classic source of a model that works
in development and fails on a partner's data. 0.5 um/px is what H-optimus-0,
Virchow and UNI all expect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .tissue import pen_mask, tissue_mask

DEFAULT_MPP = 0.25  # most TCGA diagnostic slides are 40x base


@dataclass
class TileGrid:
    coords: np.ndarray      # [N, 2] (x, y) at level 0
    level: int              # pyramid level to read from
    read_px: int            # patch size in that level's pixels
    out_px: int             # size after resize (what the encoder sees)
    level0_px: int          # footprint at level 0, for heatmap reconstruction
    base_mpp: float
    scale_factor: float


def slide_mpp(slide, fallback: float | None = DEFAULT_MPP) -> float:
    """Base microns-per-pixel, from OpenSlide properties or TIFF resolution tags."""
    import openslide

    props = slide.properties
    for key in (openslide.PROPERTY_NAME_MPP_X, "aperio.MPP", "openslide.mpp-x"):
        if key in props:
            try:
                v = float(props[key])
                if v > 0:
                    return v
            except (TypeError, ValueError):
                pass

    # Fall back to the TIFF resolution tag (typically pixels per centimetre).
    unit = props.get("tiff.ResolutionUnit", "").lower()
    xres = props.get("tiff.XResolution")
    if xres:
        try:
            res = float(xres)
            if res > 0:
                if unit == "centimeter":
                    return 10_000.0 / res
                if unit == "inch":
                    return 25_400.0 / res
        except (TypeError, ValueError):
            pass

    if fallback is None:
        raise ValueError(
            "Slide has no MPP metadata. Refusing to guess - encoding at the wrong "
            "resolution silently degrades every downstream result. Pass an explicit "
            "--assume-mpp if you know the scanner."
        )
    return fallback


def pick_level(slide, target_mpp: float, base_mpp: float) -> tuple[int, float]:
    """Highest pyramid level whose MPP does not exceed the target.

    Reading from a coarser level and upsampling invents detail; reading from a
    finer one and downsampling is safe, so we always err finer.
    """
    best_level, best_scale = 0, target_mpp / base_mpp
    for level, downsample in enumerate(slide.level_downsamples):
        level_mpp = base_mpp * downsample
        if level_mpp <= target_mpp + 1e-6:
            best_level = level
            best_scale = target_mpp / level_mpp
    return best_level, best_scale


def build_grid(
    slide,
    out_px: int = 224,
    target_mpp: float = 0.5,
    tissue_frac: float = 0.35,
    thumbnail_px: int = 2048,
    assume_mpp: float | None = DEFAULT_MPP,
    max_patches: int | None = None,
    seed: int = 0,
) -> TileGrid:
    """Non-overlapping tissue tiles, following the 20x / 256-equivalent convention."""
    base_mpp = slide_mpp(slide, assume_mpp)
    level, scale = pick_level(slide, target_mpp, base_mpp)

    read_px = max(1, int(round(out_px * scale)))
    level0_px = int(round(read_px * slide.level_downsamples[level]))

    w0, h0 = slide.level_dimensions[0]
    thumb_ratio = min(thumbnail_px / max(w0, h0), 1.0)
    thumb_w, thumb_h = max(1, int(w0 * thumb_ratio)), max(1, int(h0 * thumb_ratio))
    thumb = np.asarray(slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB"))

    keep = tissue_mask(thumb) & ~pen_mask(thumb)

    step = level0_px
    xs = np.arange(0, max(1, w0 - step + 1), step)
    ys = np.arange(0, max(1, h0 - step + 1), step)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    # Fraction of each tile's thumbnail footprint that is tissue.
    tw = max(1, int(round(step * thumb_ratio)))
    kept = []
    th, twid = keep.shape
    for x, y in coords:
        ty, tx = int(y * thumb_ratio), int(x * thumb_ratio)
        window = keep[ty : min(ty + tw, th), tx : min(tx + tw, twid)]
        if window.size and window.mean() >= tissue_frac:
            kept.append((x, y))

    coords = np.asarray(kept, dtype=np.int64).reshape(-1, 2)

    if max_patches is not None and len(coords) > max_patches:
        rng = np.random.default_rng(seed)
        coords = coords[np.sort(rng.choice(len(coords), max_patches, replace=False))]

    return TileGrid(
        coords=coords,
        level=level,
        read_px=read_px,
        out_px=out_px,
        level0_px=level0_px,
        base_mpp=base_mpp,
        scale_factor=scale,
    )


def read_tile(slide, x: int, y: int, grid: TileGrid):
    """Read one tile and resize it to the encoder's expected input size."""
    from PIL import Image

    tile = slide.read_region((int(x), int(y)), grid.level, (grid.read_px, grid.read_px))
    tile = tile.convert("RGB")
    if grid.read_px != grid.out_px:
        tile = tile.resize((grid.out_px, grid.out_px), Image.BILINEAR)
    return tile


def estimate_runtime(n_slides: int, patches_per_slide: int, patches_per_sec: float) -> str:
    total = n_slides * patches_per_slide
    hours = total / max(patches_per_sec, 1e-6) / 3600
    return f"{total:,} patches ~ {hours:.1f} GPU-hours at {patches_per_sec:.0f} patches/s"


def ceil_div(a: int, b: int) -> int:
    return -(-a // b) if b else 0


__all__ = [
    "TileGrid", "build_grid", "read_tile", "slide_mpp", "pick_level",
    "estimate_runtime", "ceil_div", "DEFAULT_MPP",
]
