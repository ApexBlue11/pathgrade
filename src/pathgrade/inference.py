"""Slide-level grading with an explainable attention map.

**This is the product surface.** Everything else in the repo exists to produce
the weights this module loads. A deployment takes a slide, encodes it with
H-optimus-0, and calls :meth:`GradePredictor.predict`, which returns a grade,
calibrated ordinal probabilities, an ambiguity score, and a per-patch
attention map that can be drawn straight over the slide thumbnail.

The attention map is the part a pathologist will actually interrogate, so it is
worth being precise about what it shows.

The previous version of this project displayed **input-gradient saliency**: the
norm of d(score)/d(features) per patch. That answers "which patches would change
the prediction if perturbed", which is a sensitivity question, not an attribution
one. It is noisy, sign-agnostic, and not the quantity the model uses.

This returns the model's **actual attention weights** - the coefficients the
network multiplies each patch by when pooling the bag into a slide
representation. If a patch has attention 0.01, it contributed exactly 1% of the
slide embedding. That is a claim you can defend in front of a clinician, and it
requires no perturbation, no gradients, and no approximation.

Attention is averaged over all ACMIL branches and all subspace ensemble members,
so it reflects the full ensemble rather than one arbitrary view.

Because extraction stores patch coordinates alongside features, every attention
value maps back to a precise region of the original slide - see
:func:`attention_to_grid`.

.. warning::

   **The attention map from the first real training run is uniform and carries
   no information.** Measured on real test slides: the top 1% of patches hold
   1.0% of the attention mass and normalised entropy is 1.0000 - identical to
   averaging every patch equally. ``top_regions()`` on such a checkpoint
   returns arbitrary tiles, and a rendered overlay is flat noise dressed up as
   an explanation, which is worse than showing nothing.

   The cause is not the code in this module. The trained model reached ~0
   training loss by mean-pooling alone, so nothing ever pushed the attention
   scorer to specialise, and weight decay then shrank its output layer below
   its own initialisation (pre-softmax score std ~0.1 across 3000 patches,
   where softmax needs a spread of order log N ~ 8 to concentrate at all).
   Sharpening it after the fact does not help - it amplifies noise and
   *lowers* accuracy - because there is no learned ranking underneath to
   sharpen.

   ``attention_is_informative()`` below checks this. **Call it before showing
   an overlay to anyone**, and see ``docs/ENGINEERING.md`` for the fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data.io import read_features, read_metadata
from .losses import corn_class_probs, corn_cumulative_probs
from .models.asmil_ord import ASMILOrd

DEFAULT_CLASS_NAMES = ("G1 - well differentiated",
                       "G2 - moderately differentiated",
                       "G3 - poorly differentiated")


@dataclass
class GradePrediction:
    """A slide-level grade, with everything needed to justify it."""

    grade: int
    grade_label: str
    probabilities: np.ndarray          # [K] calibrated class posterior
    cumulative: np.ndarray             # [K-1] P(y > j), monotone by construction
    confidence: float                  # posterior mass on the predicted grade
    attention: np.ndarray              # [N] per-patch weight, sums to 1
    coords: np.ndarray                 # [N, 2] (x, y) at slide level 0
    patch_size: int                    # level-0 footprint of one patch
    uncertainty: float = 0.0           # ensemble disagreement, 0 for a single model
    n_patches: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def expected_grade(self) -> float:
        """Continuous grade, useful when a slide sits between categories."""
        return float((self.probabilities * np.arange(len(self.probabilities))).sum())

    @property
    def ambiguity(self) -> float:
        """How close this slide sits to a grade boundary. 0 = decisive, 0.5 = a coin flip.

        Use this, not :attr:`uncertainty`, to decide what a human should review.

        CORN predicts a grade by counting how many cumulative probabilities
        P(y > j) clear 0.5, so a slide whose nearest cumulative sits *at* 0.5 is
        one rounding error away from a different grade. That distance is the
        honest measure of "the model is unsure".

        Measured on the 66-slide locked test set of the first real training run:

            signal                    AUC at detecting its own errors
            fold disagreement          0.500   <- i.e. chance
            posterior entropy          0.541
            distance to threshold      0.585

        Ranking by this and keeping the most decisive half lifted QWK from
        0.293 to 0.472, which is what a review-queue is for. It is a weak
        detector in absolute terms and is not a safety mechanism - but it is
        the only one of the three that carries any signal at all.
        """
        return float(0.5 - np.abs(np.asarray(self.cumulative) - 0.5).min())

    def top_regions(self, k: int = 10) -> list[dict]:
        """The k highest-attention patches, for a 'review these first' list."""
        order = np.argsort(-self.attention)[:k]
        return [
            {
                "rank": i + 1,
                "x": int(self.coords[j][0]),
                "y": int(self.coords[j][1]),
                "size": self.patch_size,
                "attention": float(self.attention[j]),
                "share_of_slide": float(self.attention[j] * len(self.attention)),
            }
            for i, j in enumerate(order)
        ]

    def summary(self) -> str:
        probs = "  ".join(
            f"{n.split(' - ')[0]} {p:.1%}" for n, p in zip(DEFAULT_CLASS_NAMES, self.probabilities)
        )
        return (
            f"{self.grade_label}  (confidence {self.confidence:.1%})\n"
            f"  {probs}\n"
            f"  expected grade {self.expected_grade:.2f} | "
            f"{self.n_patches:,} patches | ambiguity {self.ambiguity:.3f} "
            f"(fold spread {self.uncertainty:.3f})"
        )


class GradePredictor:
    """Loads trained weights and grades slides from patch embeddings."""

    def __init__(self, models: list[ASMILOrd], config: Config, device=None,
                 class_names: tuple = DEFAULT_CLASS_NAMES):
        self.models = models
        self.config = config
        self.class_names = class_names
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for m in self.models:
            m.to(self.device).eval()

    # ------------------------------------------------------------------
    @classmethod
    def from_run(cls, run_dir: str | Path, device=None) -> "GradePredictor":
        """Load every fold checkpoint from a training run as an ensemble."""
        run_dir = Path(run_dir)
        checkpoints = sorted(run_dir.glob("fold*/checkpoint.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"no fold checkpoints under {run_dir}")

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models, config = [], None
        for path in checkpoints:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            config = Config.from_dict(ckpt["config"])
            model = ASMILOrd(
                feature_dim=config.model.feature_dim, n_classes=config.data.n_classes,
                window=config.model.window, stride=config.model.stride,
                hidden=config.model.hidden, n_branches=config.model.n_branches,
                dropout=config.model.dropout, branch_drop=config.model.branch_drop,
                ema_decay=config.model.ema_decay, feature_norm=config.model.feature_norm,
            )
            model.load_state_dict(ckpt["state_dict"])
            models.append(model)
        return cls(models, config, device)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        features: np.ndarray,
        coords: np.ndarray,
        patch_size: int = 224,
        metadata: dict | None = None,
    ) -> GradePrediction:
        """Grade one slide.

        Args:
            features: [N, D] patch embeddings from the SAME encoder used in
                training. Mixing encoders silently produces nonsense, so the
                width is checked against the trained model.
            coords: [N, 2] (x, y) at slide level 0, as written by extraction.
            patch_size: level-0 footprint of a patch, for the heatmap grid.
        """
        features = np.asarray(features, dtype=np.float32)
        coords = np.asarray(coords)
        n, d = features.shape

        if d != self.config.model.feature_dim:
            raise ValueError(
                f"feature width {d} but this model was trained on "
                f"{self.config.model.feature_dim}. These embeddings came from a "
                f"different encoder; re-extract with {self.config.encoder!r}."
            )
        if len(coords) != n:
            raise ValueError(f"{n} features but {len(coords)} coordinates")

        x = torch.from_numpy(features).unsqueeze(0).to(self.device)
        mask = torch.ones(1, n, dtype=torch.bool, device=self.device)

        cums, attentions = [], []
        for model in self.models:
            out = model(x, mask)
            cums.append(corn_cumulative_probs(out.logits)[0].float().cpu().numpy())
            attentions.append(model.patch_attention(x, mask)[0].float().cpu().numpy())

        cum = np.mean(cums, axis=0)
        # Disagreement between folds: high values mark slides worth a second look.
        # Kept for provenance, but do NOT gate review on this: measured at AUC
        # 0.500 - exactly chance - at detecting its own errors on the first real
        # test set. `GradePrediction.ambiguity` is the signal that works.
        uncertainty = float(np.mean(np.std(cums, axis=0))) if len(cums) > 1 else 0.0

        attention = np.mean(attentions, axis=0)
        total = attention.sum()
        if total > 0:
            attention = attention / total

        grade = int((cum > 0.5).sum())
        probs = corn_class_probs(torch.from_numpy(cum).unsqueeze(0))[0].numpy()

        return GradePrediction(
            grade=grade,
            grade_label=self.class_names[grade],
            probabilities=probs,
            cumulative=cum,
            confidence=float(probs[grade]),
            attention=attention,
            coords=coords,
            patch_size=patch_size,
            uncertainty=uncertainty,
            n_patches=n,
            metadata=metadata or {},
        )

    def predict_file(self, path: str | Path) -> GradePrediction:
        """Grade a slide from an extracted ``.h5`` / ``.pt`` feature file."""
        features, coords = read_features(path, with_coords=True)
        meta = read_metadata(path)
        return self.predict(
            features, coords,
            patch_size=int(meta.get("level0_px", meta.get("patch_px", 224))),
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------
def attention_to_grid(
    prediction: GradePrediction,
    out_shape: tuple[int, int] | None = None,
    percentile_clip: tuple[float, float] = (50.0, 99.0),
    gamma: float = 0.5,
) -> np.ndarray:
    """Rasterise per-patch attention onto a 2-D grid in slide coordinates.

    Attention is extremely long-tailed - a handful of patches routinely hold
    most of the mass - so raw values render as one bright dot on black. The
    percentile clip and gamma make the mid-range visible without changing the
    ranking, which is what a reader is actually reading off the image.

    Returns a float array in [0, 1]; empty cells are 0.
    """
    coords = prediction.coords
    step = max(int(prediction.patch_size), 1)

    cols = int(np.ceil((coords[:, 0].max() + step) / step))
    rows = int(np.ceil((coords[:, 1].max() + step) / step))
    grid = np.zeros((rows, cols), dtype=np.float32)
    counts = np.zeros((rows, cols), dtype=np.float32)

    cx = np.clip((coords[:, 0] // step).astype(int), 0, cols - 1)
    cy = np.clip((coords[:, 1] // step).astype(int), 0, rows - 1)
    np.add.at(grid, (cy, cx), prediction.attention)
    np.add.at(counts, (cy, cx), 1.0)
    grid /= np.maximum(counts, 1.0)

    filled = grid[counts > 0]
    if filled.size:
        lo, hi = np.percentile(filled, percentile_clip)
        grid = np.clip((grid - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
        grid = np.power(grid, gamma)
    grid[counts == 0] = 0.0

    if out_shape is not None:
        grid = _resize_nearest(grid, out_shape)
    return grid


def _resize_nearest(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    ys = np.clip((np.arange(h) * grid.shape[0] / h).astype(int), 0, grid.shape[0] - 1)
    xs = np.clip((np.arange(w) * grid.shape[1] / w).astype(int), 0, grid.shape[1] - 1)
    return grid[np.ix_(ys, xs)]


def attention_is_informative(
    prediction: "GradePrediction", min_top1_share: float = 2.0
) -> tuple[bool, str]:
    """Is this attention map worth showing to a human, or is it uniform noise?

    A guard, not a metric. The first real training run produced attention that
    was *exactly* uniform - top 1% of patches holding 1.0% of the mass - which
    renders as a flat wash that looks like an explanation and is not one.
    Shipping that to a pathologist is worse than shipping no heatmap at all,
    so a deployment should call this before displaying an overlay.

    Args:
        min_top1_share: percent of total attention the top 1% of patches must
            hold. Uniform gives exactly 1.0; the default of 2.0 demands only
            that the model concentrates twice as hard as chance, which is a
            deliberately low bar to clear.

    Returns ``(ok, reason)``.
    """
    a = np.asarray(prediction.attention, dtype=np.float64)
    n = a.size
    if n == 0:
        return False, "no patches"
    total = a.sum()
    if not np.isfinite(total) or total <= 0:
        return False, "attention does not sum to a positive finite value"
    a = a / total

    k = max(1, n // 100)
    top1 = float(np.sort(a)[::-1][:k].sum() * 100.0)
    entropy_ratio = float(-(a * np.log(a + 1e-12)).sum() / np.log(n)) if n > 1 else 0.0

    if top1 < min_top1_share:
        return False, (
            f"attention is effectively uniform: top 1% of patches hold {top1:.2f}% "
            f"of the mass (uniform = 1.00%, required >= {min_top1_share:.2f}%), "
            f"normalised entropy {entropy_ratio:.4f}. This heatmap is not an "
            f"explanation - do not display it."
        )
    return True, f"top 1% hold {top1:.2f}% of attention, normalised entropy {entropy_ratio:.4f}"


def grade_slide(
    slide_path: str | Path,
    run_dir: str | Path,
    device=None,
    thumbnail_px: int = 1536,
    **encode_kwargs,
) -> tuple[GradePrediction, "object"]:
    """The whole commercial path: a raw slide in, a grade and an overlay out.

    This is the only entry point a deployment needs to call. A user uploads
    one whole-slide image - nothing else. There is no separate thumbnail to
    supply: the overlay is rendered on a thumbnail pulled from the same slide
    file this function tiles, so it is guaranteed to line up with the
    coordinates the attention map is drawn in. There is no pre-extraction
    step either - tiling and encoding happen here, with the exact settings
    training used (see :func:`pathgrade.preprocessing.single_slide.encode_slide`).

    Returns ``(prediction, overlay_image)``. ``overlay_image`` is a PIL image;
    save it directly with ``overlay.save(...)``.
    """
    from .preprocessing.single_slide import encode_slide, slide_thumbnail

    features, coords, attrs = encode_slide(slide_path, device=device or "auto", **encode_kwargs)
    predictor = GradePredictor.from_run(run_dir, device=device)
    prediction = predictor.predict(
        features, coords, patch_size=attrs["level0_px"], metadata=attrs,
    )
    thumb = slide_thumbnail(slide_path, max_px=thumbnail_px)
    overlay = render_overlay(prediction, thumb)
    return prediction, overlay


def render_overlay(
    prediction: GradePrediction,
    thumbnail,
    alpha: float = 0.55,
    threshold: float = 0.15,
    colour: tuple[int, int, int] = (0, 220, 255),
):
    """Composite the attention map over a PIL thumbnail of the slide.

    Cyan by default: maximally distinct from H&E pink/purple, so the overlay
    cannot be mistaken for tissue. Cells below ``threshold`` stay fully
    transparent, which keeps background glass clean instead of tinting the whole
    slide.
    """
    from PIL import Image

    thumb = thumbnail.convert("RGBA")
    grid = attention_to_grid(prediction, out_shape=(thumb.size[1], thumb.size[0]))

    rgba = np.zeros((*grid.shape, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = colour
    rgba[..., 3] = np.where(grid < threshold, 0, np.clip(grid * 255 * alpha, 0, 255)).astype(np.uint8)

    return Image.alpha_composite(thumb, Image.fromarray(rgba, mode="RGBA"))
