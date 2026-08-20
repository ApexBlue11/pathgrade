"""Streaming GDC -> embeddings pipeline: fetch a slide, encode it, delete it, repeat.

Built for Kaggle-shaped constraints. The full TCGA-HNSC diagnostic set is about
456 GB across 472 slides, which will never fit on a worker, so at most two
slides exist on disk at any moment: the one being encoded and the one being
fetched behind it.

Three things make this fit in a session budget:

* **Download and compute overlap.** A background thread fetches slide *n+1*
  while the accelerator encodes slide *n*. Without this the accelerator idles
  through every download, which on this workload is most of the wall clock.
* **Fixed batch shapes.** XLA compiles a static graph per input shape, so a
  ragged final batch triggers a full recompile on every slide. Batches are
  padded to a constant size and sliced afterwards, giving exactly one
  compilation for the whole run.
* **Sharding.** ``--shard i --num-shards n`` splits the slide list across
  parallel sessions, and the same flag drives per-core sharding on a multi-core
  TPU.

Progress is journalled, so a session that hits its time limit resumes cleanly.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from ..encoders import DEFAULT_ENCODER, PatchEncoder, check_licence, describe_registry, resolve_device
from .gdc import SlideRecord, download_slide, free_disk_gb, one_slide_per_patient, query_slides, shard, summarise
from .tiling import build_grid, read_tile


def write_features(path: Path, feats: np.ndarray, coords: np.ndarray, attrs: dict, fmt: str) -> None:
    """Write ``.pt`` (matches the older Kaggle dataset layout) or ``.h5``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")

    if fmt == "pt":
        torch.save(
            {
                "features": torch.from_numpy(feats),
                "coords": torch.from_numpy(coords),
                "meta": attrs,
            },
            tmp,
        )
    else:
        import h5py

        with h5py.File(tmp, "w") as f:
            f.create_dataset("features", data=feats, compression="gzip", compression_opts=4)
            f.create_dataset("coords", data=coords, compression="gzip")
            f.attrs.update(attrs)
    tmp.replace(path)


class SlidePrefetcher:
    """Downloads slides one step ahead of the encoder, on a background thread."""

    def __init__(self, records: list[SlideRecord], cache_dir: Path, depth: int = 1):
        self.records = records
        self.cache_dir = cache_dir
        self.queue: queue.Queue = queue.Queue(maxsize=depth)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self._stop = threading.Event()

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self._stop.set()
        # Drain so a blocked producer can notice the stop flag and exit.
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass

    def _run(self):
        for record in self.records:
            if self._stop.is_set():
                break
            try:
                t0 = time.time()
                path = download_slide(record, self.cache_dir)
                self.queue.put((record, path, time.time() - t0))
            except Exception as e:
                self.queue.put((record, None, str(e)))
        self.queue.put(None)             # sentinel

    def __iter__(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            yield item


@torch.inference_mode()
def encode_tiles(slide, grid, encoder: PatchEncoder, transform, batch_size: int) -> np.ndarray:
    """Encode every tile in ``grid``, padding batches to a constant size for XLA."""
    coords = grid.coords
    out = np.empty((len(coords), encoder.spec.embed_dim), dtype=np.float32)
    buf: list[torch.Tensor] = []
    written = 0

    def flush():
        nonlocal written, buf
        if not buf:
            return
        real = len(buf)
        batch = torch.stack(buf)
        if real < batch_size:
            pad = batch[-1:].expand(batch_size - real, *batch.shape[1:])
            batch = torch.cat([batch, pad], dim=0)
        feats = encoder(batch)
        if encoder.is_xla:
            import torch_xla.core.xla_model as xm

            xm.mark_step()
        out[written : written + real] = feats[:real].cpu().numpy()
        written += real
        buf = []

    for i in range(len(coords)):
        x, y = coords[i]
        buf.append(transform(read_tile(slide, int(x), int(y), grid)))
        if len(buf) == batch_size:
            flush()
    flush()
    return out


def process_slide(
    record: SlideRecord,
    slide_path: Path,
    encoder: PatchEncoder,
    transform,
    out_path: Path,
    args,
) -> dict:
    import openslide

    slide = openslide.OpenSlide(str(slide_path))
    try:
        grid = build_grid(
            slide,
            out_px=encoder.spec.patch_px,
            target_mpp=args.target_mpp,
            tissue_frac=args.tissue_frac,
            assume_mpp=args.assume_mpp,
            max_patches=args.max_patches,
        )
        if len(grid.coords) == 0:
            raise ValueError("no tissue tiles found")

        t0 = time.time()
        feats = encode_tiles(slide, grid, encoder, transform, args.batch_size)
        elapsed = time.time() - t0
    finally:
        slide.close()

    if feats.shape[1] != encoder.spec.embed_dim:
        raise RuntimeError(
            f"encoder produced width {feats.shape[1]}, registry declares {encoder.spec.embed_dim}"
        )

    attrs = {
        "encoder": encoder.spec.name,
        "encoder_licence": encoder.spec.licence,
        "commercial_ok": encoder.spec.commercial_ok,
        "embed_dim": int(feats.shape[1]),
        "target_mpp": float(args.target_mpp),
        "base_mpp": float(grid.base_mpp),
        "patch_px": int(grid.out_px),
        "level0_px": int(grid.level0_px),
        "level": int(grid.level),
        "n_patches": int(feats.shape[0]),
        "gdc_file_id": record.file_id,
        "slide_barcode": record.slide_barcode,
        "patient_id": record.patient_id,
    }
    store = feats.astype(np.float16 if not args.fp32_store else np.float32)
    write_features(out_path, store, grid.coords.astype(np.int64), attrs, args.format)

    return {
        "patient_id": record.patient_id,
        "n_patches": int(feats.shape[0]),
        "encode_seconds": round(elapsed, 1),
        "patches_per_sec": round(feats.shape[0] / max(elapsed, 1e-6), 1),
        "base_mpp": round(grid.base_mpp, 4),
        "gb": round(record.file_size / 1e9, 2),
    }


def run(args) -> int:
    spec = check_licence(args.encoder, args.allow_noncommercial)

    records = query_slides(args.project, diagnostic_only=not args.include_frozen)
    if args.one_per_patient:
        records = one_slide_per_patient(records)
    print(f"GDC {args.project}: {summarise(records)}")

    records = shard(records, args.shard, args.num_shards)
    if args.limit:
        records = records[: args.limit]
    print(f"this shard ({args.shard + 1}/{args.num_shards}): {summarise(records)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".pt" if args.format == "pt" else ".h5"

    todo = [r for r in records if not (out_dir / f"{r.patient_id}{suffix}").exists()]
    print(f"{len(records) - len(todo)} already extracted, {len(todo)} to go")
    if not todo:
        return 0

    device = resolve_device(args.device)
    print(f"device: {device}  |  encoder {spec.name} ({spec.embed_dim}-d, {spec.licence})")
    print(f"free disk: {free_disk_gb(cache_dir):.1f} GB\n")

    encoder = PatchEncoder(spec, device=device)
    transform = encoder.build_transform()

    journal_path = out_dir / f"journal_shard{args.shard}.jsonl"
    prefetcher = SlidePrefetcher(todo, cache_dir, depth=args.prefetch).start()

    done, failed, patches = 0, 0, 0
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    t_start = time.time()

    try:
        for record, slide_path, info in prefetcher:
            if deadline and time.time() > deadline:
                print(f"\nreached --max-hours ({args.max_hours}h); stopping cleanly.")
                break
            if slide_path is None:
                failed += 1
                print(f"  {record.patient_id}: DOWNLOAD FAILED - {info}", file=sys.stderr)
                continue

            out_path = out_dir / f"{record.patient_id}{suffix}"
            try:
                stats = process_slide(record, slide_path, encoder, transform, out_path, args)
                stats["download_seconds"] = round(float(info), 1)
                done += 1
                patches += stats["n_patches"]
                elapsed_h = (time.time() - t_start) / 3600
                rate = done / max(elapsed_h, 1e-6)
                print(
                    f"[{done + failed}/{len(todo)}] {record.patient_id}: "
                    f"{stats['n_patches']:>5,} patches @ {stats['patches_per_sec']:>6.0f}/s "
                    f"| dl {stats['download_seconds']:>5.0f}s enc {stats['encode_seconds']:>5.0f}s "
                    f"| {rate:.0f} slides/h"
                )
                with open(journal_path, "a") as f:
                    f.write(json.dumps(stats) + "\n")
            except Exception as e:
                failed += 1
                print(f"  {record.patient_id}: EXTRACT FAILED - {e}", file=sys.stderr)
            finally:
                # The whole point: reclaim the disk immediately.
                if not args.keep_slides:
                    Path(slide_path).unlink(missing_ok=True)
    finally:
        prefetcher.stop()

    hours = (time.time() - t_start) / 3600
    print(f"\n{done} slides, {patches:,} patches, {failed} failures in {hours:.2f} h")
    if done:
        print(f"projected for {len(records)} slides: {hours / done * len(records):.1f} h")
    return 1 if failed else 0


def build_parser():
    p = argparse.ArgumentParser(
        description="Stream slides from GDC, encode them, discard them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=describe_registry(),
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache-dir", default="/tmp/wsi_cache", help="scratch for in-flight slides")
    p.add_argument("--project", default="TCGA-HNSC")
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--format", choices=["pt", "h5"], default="h5")
    p.add_argument("--device", default="auto", help="auto | cuda | xla | cpu")

    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-hours", type=float, default=None,
                   help="stop cleanly before the session limit (Kaggle allows 9h)")

    p.add_argument("--max-patches", type=int, default=4000,
                   help="cap per slide. The dominant lever on encode time and storage")
    p.add_argument("--target-mpp", type=float, default=0.5)
    p.add_argument("--tissue-frac", type=float, default=0.35)
    p.add_argument("--batch-size", type=int, default=64,
                   help="keep constant on TPU; it defines the compiled graph")
    p.add_argument("--prefetch", type=int, default=1, help="slides to fetch ahead")

    p.add_argument("--one-per-patient", action="store_true", default=True)
    p.add_argument("--all-slides", dest="one_per_patient", action="store_false")
    p.add_argument("--include-frozen", action="store_true",
                   help="include TS/BS frozen sections (not advisable for grading)")
    p.add_argument("--keep-slides", action="store_true", help="do not delete after encoding")
    p.add_argument("--assume-mpp", type=float, default=None)
    p.add_argument("--fp32-store", action="store_true")
    p.add_argument("--allow-noncommercial", action="store_true")
    return p


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
