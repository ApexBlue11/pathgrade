"""WSI -> patch embeddings. This is the expensive step; everything else is cheap.

Output is one HDF5 file per slide containing ``features`` [N, D], ``coords``
[N, 2] at level 0, and a provenance block in the file attributes: encoder name,
licence, resolution, tiling settings and code version. A model that ends up in
front of a regulator needs to be able to answer "what exactly produced this
vector", and that answer should live next to the vector, not in someone's
memory of which notebook they ran.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..encoders import DEFAULT_ENCODER, check_licence, describe_registry, PatchEncoder
from .tiling import build_grid, read_tile

SCHEMA_VERSION = 1


class _TileDataset(Dataset):
    """Reads tiles lazily. OpenSlide handles are per-worker: they are not fork-safe."""

    def __init__(self, slide_path: str, grid, transform):
        self.slide_path = slide_path
        self.grid = grid
        self.transform = transform
        self._slide = None

    def __len__(self):
        return len(self.grid.coords)

    def _handle(self):
        if self._slide is None:
            import openslide

            self._slide = openslide.OpenSlide(self.slide_path)
        return self._slide

    def __getitem__(self, i):
        x, y = self.grid.coords[i]
        tile = read_tile(self._handle(), int(x), int(y), self.grid)
        return self.transform(tile)


def extract_slide(
    slide_path: str | Path,
    out_path: str | Path,
    encoder: PatchEncoder,
    transform,
    target_mpp: float = 0.5,
    tissue_frac: float = 0.35,
    batch_size: int = 128,
    num_workers: int = 4,
    max_patches: int | None = None,
    assume_mpp: float | None = None,
    store_fp16: bool = True,
) -> dict:
    import openslide

    slide_path, out_path = str(slide_path), Path(out_path)
    slide = openslide.OpenSlide(slide_path)
    grid = build_grid(
        slide,
        out_px=encoder.spec.patch_px,
        target_mpp=target_mpp,
        tissue_frac=tissue_frac,
        assume_mpp=assume_mpp,
        max_patches=max_patches,
    )
    if len(grid.coords) == 0:
        slide.close()
        raise ValueError(f"No tissue tiles found in {slide_path}")

    loader = DataLoader(
        _TileDataset(slide_path, grid, transform),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    chunks = []
    t0 = time.time()
    for batch in loader:
        chunks.append(encoder(batch).cpu())
    feats = torch.cat(chunks).numpy()
    elapsed = time.time() - t0
    slide.close()

    if feats.shape[1] != encoder.spec.embed_dim:
        raise RuntimeError(
            f"Encoder produced width {feats.shape[1]} but the registry declares "
            f"{encoder.spec.embed_dim} for {encoder.spec.name!r}. Fix the registry "
            f"before training - a silent width change invalidates saved checkpoints."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".h5.partial")
    with h5py.File(tmp, "w") as f:
        f.create_dataset(
            "features",
            data=feats.astype(np.float16 if store_fp16 else np.float32),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset("coords", data=grid.coords.astype(np.int64), compression="gzip")
        f.attrs.update({
            "schema_version": SCHEMA_VERSION,
            "encoder": encoder.spec.name,
            "encoder_hub_id": encoder.spec.hf_hub_id,
            "encoder_licence": encoder.spec.licence,
            "commercial_ok": encoder.spec.commercial_ok,
            "embed_dim": int(feats.shape[1]),
            "pooling": encoder.spec.pooling,
            "target_mpp": float(target_mpp),
            "base_mpp": float(grid.base_mpp),
            "level": int(grid.level),
            "patch_px": int(grid.out_px),
            "level0_px": int(grid.level0_px),
            "tissue_frac": float(tissue_frac),
            "n_patches": int(feats.shape[0]),
            "source_slide": str(slide_path),
        })
    tmp.replace(out_path)

    return {
        "n_patches": int(feats.shape[0]),
        "embed_dim": int(feats.shape[1]),
        "seconds": round(elapsed, 1),
        "patches_per_sec": round(feats.shape[0] / max(elapsed, 1e-6), 1),
        "base_mpp": grid.base_mpp,
        "level": grid.level,
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Extract patch embeddings from whole-slide images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=describe_registry(),
    )
    p.add_argument("--slide-dir", required=True, help="directory of .svs/.tiff/.ndpi slides")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--pattern", default="*.svs")
    p.add_argument("--target-mpp", type=float, default=0.5)
    p.add_argument("--tissue-frac", type=float, default=0.35)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-patches", type=int, default=None,
                   help="cap per slide; omit to keep all tissue tiles")
    p.add_argument("--assume-mpp", type=float, default=None,
                   help="fallback MPP when the slide carries no metadata")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp32-store", action="store_true", help="store float32 instead of float16")
    p.add_argument("--allow-noncommercial", action="store_true",
                   help="permit a non-commercially-licensed encoder (research only)")
    p.add_argument("--limit", type=int, default=None, help="process at most N slides")
    args = p.parse_args(argv)

    spec = check_licence(args.encoder, args.allow_noncommercial)
    if not spec.commercial_ok:
        print(
            f"\n!! {spec.name} is {spec.licence}. Features and any model trained on them "
            f"are research-only and cannot ship.\n", file=sys.stderr,
        )

    slides = sorted(Path(args.slide_dir).rglob(args.pattern))
    if args.limit:
        slides = slides[: args.limit]
    if not slides:
        p.error(f"No slides matching {args.pattern!r} under {args.slide_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = PatchEncoder(spec, device=args.device)
    transform = encoder.build_transform()

    print(f"{spec.name} ({spec.embed_dim}-d, {spec.licence}) -> {out_dir}")
    print(f"{len(slides)} slides at {args.target_mpp} um/px, {spec.patch_px}px tiles\n")

    manifest, failures = [], []
    for i, slide_path in enumerate(slides, 1):
        out_path = out_dir / f"{slide_path.stem}.h5"
        if out_path.exists():
            print(f"[{i}/{len(slides)}] {slide_path.stem}: cached")
            continue
        try:
            info = extract_slide(
                slide_path, out_path, encoder, transform,
                target_mpp=args.target_mpp, tissue_frac=args.tissue_frac,
                batch_size=args.batch_size, num_workers=args.num_workers,
                max_patches=args.max_patches, assume_mpp=args.assume_mpp,
                store_fp16=not args.fp32_store,
            )
            manifest.append({"slide": slide_path.stem, **info})
            print(f"[{i}/{len(slides)}] {slide_path.stem}: {info['n_patches']:,} patches "
                  f"@ {info['patches_per_sec']}/s (mpp {info['base_mpp']:.3f}, L{info['level']})")
        except Exception as e:
            failures.append({"slide": slide_path.stem, "error": str(e)})
            print(f"[{i}/{len(slides)}] {slide_path.stem}: FAILED - {e}", file=sys.stderr)

    with open(out_dir / "extraction_manifest.json", "w") as f:
        json.dump({"encoder": asdict(spec), "slides": manifest, "failures": failures}, f, indent=2)

    total = sum(m["n_patches"] for m in manifest)
    print(f"\nDone. {len(manifest)} slides, {total:,} patches, {len(failures)} failures.")
    if failures:
        print("Failures recorded in extraction_manifest.json", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
