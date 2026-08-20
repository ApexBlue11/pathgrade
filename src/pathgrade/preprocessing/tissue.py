"""Tissue detection on a WSI thumbnail.

Deliberately simple. The Frontiers 2026 practical-guidelines study found that
stain normalisation does not help when a modern foundation encoder sits
downstream - the encoders are already robust to staining variation - so the
only job here is to stop us spending GPU hours encoding glass, and to drop the
obvious artefacts (pen marks, out-of-focus white space) that would otherwise
enter the bag as noise.
"""

from __future__ import annotations

import numpy as np


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's method on a 1-D array, without pulling in scikit-image."""
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return float(values.max() if values.size else 0.0)

    centres = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not valid.any():
        return float(centres[len(centres) // 2])

    cum_mean = np.cumsum(hist * centres)
    mean_bg = np.divide(cum_mean, weight_bg, out=np.zeros_like(cum_mean), where=weight_bg > 0)
    total_mean = cum_mean[-1]
    mean_fg = np.divide(
        total_mean - cum_mean, weight_fg, out=np.zeros_like(cum_mean), where=weight_fg > 0
    )
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~valid] = -np.inf
    return float(centres[int(np.argmax(between))])


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB->HSV for a uint8 [H, W, 3] image. Returns floats in [0, 1]."""
    x = rgb.astype(np.float32) / 255.0
    mx = x.max(axis=-1)
    mn = x.min(axis=-1)
    diff = mx - mn

    hue = np.zeros_like(mx)
    nz = diff > 1e-8
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & (mx == b) & ~rm & ~gm
    hue[rm] = ((g[rm] - b[rm]) / diff[rm]) % 6
    hue[gm] = ((b[gm] - r[gm]) / diff[gm]) + 2
    hue[bm] = ((r[bm] - g[bm]) / diff[bm]) + 4
    hue /= 6.0

    sat = np.divide(diff, mx, out=np.zeros_like(mx), where=mx > 1e-8)
    return np.stack([hue, sat, mx], axis=-1)


def tissue_mask(
    thumbnail: np.ndarray,
    min_saturation: float = 0.05,
    max_value: float = 0.96,
    min_value: float = 0.05,
) -> np.ndarray:
    """Boolean tissue mask for a uint8 RGB thumbnail.

    Saturation separates stained tissue from grey/white glass far more reliably
    than luminance alone; the value bounds then remove blown-out background and
    the near-black borders scanners leave behind.
    """
    hsv = rgb_to_hsv(thumbnail)
    sat, val = hsv[..., 1], hsv[..., 2]

    thr = max(otsu_threshold(sat.ravel()), min_saturation)
    mask = (sat > thr) & (val < max_value) & (val > min_value)
    return remove_small_components(mask, min_size=max(16, mask.size // 20000))


def pen_mask(thumbnail: np.ndarray) -> np.ndarray:
    """Flag saturated green/blue/black marker ink so it never reaches the encoder."""
    hsv = rgb_to_hsv(thumbnail)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (hue > 0.20) & (hue < 0.45) & (sat > 0.35)
    blue = (hue > 0.50) & (hue < 0.75) & (sat > 0.35)
    black = val < 0.18
    return green | blue | black


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected components below ``min_size`` px (4-connectivity, iterative)."""
    if min_size <= 1 or not mask.any():
        return mask
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    out = np.zeros_like(mask)
    current = 0
    stack: list[tuple[int, int]] = []

    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or labels[sy, sx]:
                continue
            current += 1
            stack.append((sy, sx))
            labels[sy, sx] = current
            component = []
            while stack:
                y, x = stack.pop()
                component.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = current
                        stack.append((ny, nx))
            if len(component) >= min_size:
                ys, xs = zip(*component)
                out[np.array(ys), np.array(xs)] = True
    return out
