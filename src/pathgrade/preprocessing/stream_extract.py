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
import os
import queue
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ..encoders import (DEFAULT_ENCODER, PatchEncoder, build_encoders, check_licence,
                        with_pooling,
                        describe_registry, resolve_device)
from ..progress import ProgressReporter
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
    """Downloads slides ahead of the encoder, over several concurrent streams.

    Single-stream GDC throughput measured 25.6 MB/s from a Kaggle TPU VM, but
    four concurrent streams reached 143 MB/s aggregate - GDC does not rate-limit
    a single client, the per-connection ceiling is just low. Since download is
    the dominant cost of the whole job, this 5.6x is the most valuable
    optimisation in the pipeline.

    Concurrency is bounded by a semaphore counting slides *on disk*, not by the
    thread count: the consumer releases a slot only after deleting the slide it
    just encoded, so peak disk use stays at ``max_on_disk`` slides regardless of
    how fast downloads complete.
    """

    def __init__(self, records: list[SlideRecord], cache_dir: Path,
                 max_on_disk: int = 4, workers: int = 4):
        self.records = records
        self.cache_dir = cache_dir
        self.workers = max(1, workers)
        self.max_on_disk = max(1, max_on_disk)
        self.queue: queue.Queue = queue.Queue()
        self._slots = threading.Semaphore(self.max_on_disk)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self._stop.set()
        for _ in range(self.max_on_disk + self.workers):
            self._slots.release()          # unblock any waiting downloader

    def release(self):
        """Consumer calls this once it has deleted a slide from disk."""
        self._slots.release()

    def _fetch(self, record: SlideRecord):
        """Runs on a pool thread. The disk slot is already held on its behalf."""
        if self._stop.is_set():
            self._slots.release()
            return (record, None, "stopped")
        try:
            t0 = time.time()
            path = download_slide(record, self.cache_dir)
            return (record, path, time.time() - t0)
        except Exception as e:
            self._slots.release()          # nothing landed on disk
            return (record, None, str(e))

    def _run(self):
        """Producer. Acquires each disk slot *before* submitting, in record order.

        Acquiring inside the worker instead would deadlock: freed threads pick
        up later records, those grab the released slots, and the consumer waits
        forever on an earlier record that can no longer get a slot. Reserving in
        submission order keeps the in-flight set bounded and progress guaranteed.

        Results are delivered **as they complete, not in record order**, and
        that is worth a third of extraction wall clock. Measured on a 79-slide
        chunk: 2142 s of actual per-slide work against a 3158 s wall. Download
        durations are heavily skewed (median 16 s, max 63 s), and popping the
        oldest future blocked the encoder behind the single slowest download
        even when three other slides were already sitting on disk. Slides are
        independent - each writes its own file - so ordering buys nothing.
        """
        from concurrent.futures import FIRST_COMPLETED
        from concurrent.futures import wait as futures_wait

        pending: set = set()
        records = iter(self.records)

        def submit_next(pool) -> bool:
            record = next(records, None)
            if record is None:
                return False
            self._slots.acquire()           # blocks until the consumer frees disk
            if self._stop.is_set():
                self._slots.release()
                return False
            pending.add(pool.submit(self._fetch, record))
            return True

        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for _ in range(self.max_on_disk):
                    if not submit_next(pool):
                        break

                while pending:
                    done, _ = futures_wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.discard(future)
                        self.queue.put(future.result())
                    if self._stop.is_set():
                        break
                    for _ in range(len(done)):
                        if not submit_next(pool):
                            break
        finally:
            self.queue.put(None)            # sentinel

    def __iter__(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            yield item


def _encode_batch(encoder: PatchEncoder, start: int, tiles: list, batch_size: int,
                  out: np.ndarray) -> None:
    """Encode one batch of decoded tiles into ``out[start:start+len(tiles)]``.

    Shared by the single- and multi-device paths so the padding rule and the
    device round trip exist in exactly one place.

    ``.cpu()`` is what actually forces execution. XLA builds its graph lazily,
    so a benchmark - or a worker thread - that never materialises the result
    measures graph construction and nothing else. That mistake produced this
    repo's fictional 1226 patches/s.
    """
    batch = torch.from_numpy(np.stack(tiles))
    real = batch.shape[0]
    if real < batch_size:
        # Constant shape for XLA: pad by repeating, then slice the result.
        batch = torch.cat([batch, batch[-1:].expand(batch_size - real, *batch.shape[1:])])
    feats = encoder(batch)
    if encoder.is_xla:
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    out[start : start + real] = feats[:real].cpu().numpy()


def _encode_parallel(decode_one, n: int, encoders: list, batch_size: int,
                     num_workers: int, queue_depth: int, out: np.ndarray) -> None:
    """Data-parallel encode across several devices, one thread per device.

    The encoder is frozen and every batch is independent, so replicas need no
    collective communication - which is why threads are enough and a
    multi-process launcher is not required. Decoded batches go onto a bounded
    queue and whichever device is free takes the next one, so a slow device
    holds nobody up.

    Threads only help if the forward pass releases the GIL. It does: dispatch
    and the blocking device transfer are both C++. That is measured, not
    assumed here; see ``docs/ENGINEERING.md``.

    Each thread writes a disjoint slice of ``out``, so no lock is needed on the
    output. Failures are collected and re-raised on the caller's thread rather
    than left to strand a half-filled array as a valid-looking result.
    """
    batch_q: queue.Queue = queue.Queue(maxsize=len(encoders) * max(1, queue_depth))
    errors: list[BaseException] = []
    stop = threading.Event()

    def device_worker(encoder: PatchEncoder) -> None:
        while True:
            item = batch_q.get()
            if item is None:
                return
            if stop.is_set():
                continue          # keep draining so the producer never blocks
            try:
                with torch.no_grad():
                    _encode_batch(encoder, item[0], item[1], batch_size, out)
            except BaseException as e:      # noqa: BLE001 - re-raised below
                errors.append(e)
                stop.set()

    threads = [threading.Thread(target=device_worker, args=(e,), daemon=True)
               for e in encoders]
    for t in threads:
        t.start()

    max_inflight = max(batch_size * max(1, queue_depth) * len(encoders), batch_size)
    try:
        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
            pending: deque = deque()
            buffered: list[np.ndarray] = []
            start = 0

            def drain_one():
                nonlocal start, buffered
                buffered.append(pending.popleft().result())
                if len(buffered) == batch_size:
                    batch_q.put((start, buffered))
                    start += batch_size
                    buffered = []

            for i in range(n):
                if stop.is_set():
                    break
                pending.append(pool.submit(decode_one, i))
                if len(pending) >= max_inflight:
                    drain_one()
            while pending and not stop.is_set():
                drain_one()
            if buffered and not stop.is_set():
                batch_q.put((start, buffered))
    finally:
        for _ in threads:
            batch_q.put(None)
        for t in threads:
            t.join()

    if errors:
        raise errors[0]


@torch.no_grad()
def encode_tiles(
    slide,
    grid,
    encoder: PatchEncoder | list,
    transform,
    batch_size: int,
    num_workers: int = 8,
    queue_depth: int = 3,
) -> np.ndarray:
    """Encode every tile in ``grid``, decoding tiles on a thread pool.

    ``encoder`` may be a single ``PatchEncoder`` or a list of replicas on
    different devices; with a list the work is spread across all of them.

    Tile decoding, not the accelerator, is the usual bottleneck: a serial
    ``read_region`` + resize runs around 100 tiles/s, which would leave a TPU
    idle roughly 95% of the time. Threads (not processes) are the right tool
    here because OpenSlide releases the GIL inside ``read_region`` and Pillow
    releases it during resize, so they genuinely run in parallel - and an
    OpenSlide handle is documented as safe for concurrent reads, so all workers
    can share one open slide with no per-worker reopen cost.

    Batches are submitted ``queue_depth`` ahead so decoding for batch n+1
    overlaps the forward pass for batch n.
    """
    encoders = list(encoder) if isinstance(encoder, (list, tuple)) else [encoder]
    coords = grid.coords
    n = len(coords)
    out = np.empty((n, encoders[0].spec.embed_dim), dtype=np.float32)

    def decode_one(i: int) -> np.ndarray:
        return transform(read_tile(slide, int(coords[i][0]), int(coords[i][1]), grid))

    if len(encoders) > 1:
        _encode_parallel(decode_one, n, encoders, batch_size, num_workers, queue_depth, out)
        return out

    # ---- single device: the path validated on 60 real slides, left intact ----
    solo = encoders[0]

    def encode(start: int, tiles: list[np.ndarray]) -> None:
        # Stack as uint8 and let the accelerator normalise; float math in the
        # decode threads competes for the cores that are already the limit.
        _encode_batch(solo, start, tiles, batch_size, out)

    # Parallelism must be at the *tile* level. Submitting one task per batch and
    # decoding its tiles serially caps concurrency at the queue depth no matter
    # how many workers exist - measured 106 tiles/s that way versus 371 with
    # per-tile tasks on the same box.
    max_inflight = max(batch_size * max(1, queue_depth), batch_size)
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        pending: deque = deque()
        buffered: list[np.ndarray] = []
        start = 0

        def drain_one():
            nonlocal start, buffered
            buffered.append(pending.popleft().result())
            if len(buffered) == batch_size:
                encode(start, buffered)
                start += batch_size
                buffered = []

        for i in range(n):
            pending.append(pool.submit(decode_one, i))
            if len(pending) >= max_inflight:
                drain_one()
        while pending:
            drain_one()
        if buffered:
            encode(start, buffered)

    return out


def process_slide(
    record: SlideRecord,
    slide_path: Path,
    encoder: PatchEncoder | list,
    transform,
    out_path: Path,
    args,
) -> dict:
    import openslide

    # ``encoder`` may be a list of device replicas; they share one spec.
    spec = (encoder[0] if isinstance(encoder, (list, tuple)) else encoder).spec

    t_open = time.time()
    slide = openslide.OpenSlide(str(slide_path))
    try:
        # Timed separately because the 40-slide smoke run showed encode was only
        # 50% of wall clock, leaving ~25 s/slide attributed to nothing that was
        # being measured. Optimising the encoder cannot speed up uncounted work.
        t_grid = time.time()
        grid = build_grid(
            slide,
            out_px=spec.patch_px,
            target_mpp=args.target_mpp,
            tissue_frac=args.tissue_frac,
            assume_mpp=args.assume_mpp,
            max_patches=args.max_patches,
        )
        grid_seconds = time.time() - t_grid
        if len(grid.coords) == 0:
            raise ValueError("no tissue tiles found")

        t0 = time.time()
        feats = encode_tiles(
            slide, grid, encoder, transform, args.batch_size,
            num_workers=args.decode_workers, queue_depth=args.queue_depth,
        )
        elapsed = time.time() - t0
    finally:
        slide.close()

    if feats.shape[1] != spec.embed_dim:
        raise RuntimeError(
            f"encoder produced width {feats.shape[1]}, registry declares {spec.embed_dim}"
        )

    attrs = {
        "encoder": spec.name,
        # Which token pooling produced these vectors. Recorded because the
        # width alone does not identify it and a mismatched feature
        # directory is otherwise silent until training behaves oddly.
        "pooling": spec.pooling,
        "random_weights": bool(getattr(args, "random_weights", False)),
        "encoder_licence": spec.licence,
        "commercial_ok": spec.commercial_ok,
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
    t_write = time.time()
    store = feats.astype(np.float16 if not args.fp32_store else np.float32)
    write_features(out_path, store, grid.coords.astype(np.int64), attrs, args.format)
    write_seconds = time.time() - t_write

    return {
        "patient_id": record.patient_id,
        "n_patches": int(feats.shape[0]),
        "encode_seconds": round(elapsed, 1),
        "grid_seconds": round(grid_seconds, 1),
        "write_seconds": round(write_seconds, 1),
        "slide_seconds": round(time.time() - t_open, 1),
        "patches_per_sec": round(feats.shape[0] / max(elapsed, 1e-6), 1),
        "base_mpp": round(grid.base_mpp, 4),
        "gb": round(record.file_size / 1e9, 2),
    }


def run(args) -> int:
    spec = with_pooling(check_licence(args.encoder, args.allow_noncommercial),
                        getattr(args, "pooling", None))

    records = query_slides(args.project, diagnostic_only=not args.include_frozen)
    if args.one_per_patient:
        records = one_slide_per_patient(records)
    print(f"GDC {args.project}: {summarise(records)}")

    # Slides without a grade label cannot train anything, and downloading them
    # is the most expensive way to waste time in this pipeline.
    if args.labels_csv:
        from ..data.splits import read_labels

        labelled = set(read_labels(args.labels_csv))
        before = len(records)
        records = [r for r in records if r.patient_id in labelled]
        print(f"restricted to labelled patients: {len(records)}/{before} "
              f"({summarise(records)})")

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
    print(f"decode workers: {args.decode_workers}  |  batch {args.batch_size}")

    # Kaggle gives ~20 GB on /kaggle/working (persisted, becomes the dataset) and
    # ~60 GB on /kaggle/tmp (scratch). Slides must land on scratch or the output
    # volume fills after a handful of them.
    out_free, cache_free = free_disk_gb(out_dir), free_disk_gb(cache_dir)
    per_slide_mb = args.max_patches * spec.embed_dim * (4 if args.fp32_store else 2) / 1e6
    projected_gb = per_slide_mb * len(todo) / 1000
    print(f"free disk: output {out_free:.1f} GB, cache {cache_free:.1f} GB")
    print(f"projected output: {projected_gb:.1f} GB ({per_slide_mb:.1f} MB/slide x {len(todo)})")

    if str(out_dir.resolve()) == str(cache_dir.resolve()):
        print("!! cache-dir equals out-dir; slides will compete with embeddings for space",
              file=sys.stderr)
    if projected_gb > out_free - args.min_free_gb:
        print(
            f"!! projected output ({projected_gb:.1f} GB) may not fit in {out_free:.1f} GB. "
            f"Lower --max-patches, raise --num-shards, or write to a bigger volume.",
            file=sys.stderr,
        )
    print()

    encoders = build_encoders(spec, device=device, max_devices=args.tpu_cores,
                              random_weights=args.random_weights)
    transform = encoders[0].build_transform()
    if len(encoders) > 1:
        print(f"data-parallel across {len(encoders)} devices: "
              f"{[str(e.device) for e in encoders]}", flush=True)
    encoder = encoders if len(encoders) > 1 else encoders[0]

    journal_path = out_dir / f"journal_shard{args.shard}.jsonl"
    reporter = ProgressReporter(
        out_dir, total=len(todo), shard=args.shard, label="extract",
        webhook_url=args.webhook_url, notify_every=args.notify_every,
    )
    reporter.notify(f"START shard {args.shard}: {len(todo)} slides on {device}")
    prefetcher = SlidePrefetcher(
        todo, cache_dir, max_on_disk=args.prefetch, workers=args.download_workers
    ).start()

    done, failed, patches = 0, 0, 0
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    t_start = time.time()

    try:
        for record, slide_path, info in prefetcher:
            if deadline and time.time() > deadline:
                print(f"\nreached --max-hours ({args.max_hours}h); stopping cleanly.")
                break
            remaining = free_disk_gb(out_dir)
            if remaining < args.min_free_gb:
                print(
                    f"\nonly {remaining:.1f} GB left on the output volume "
                    f"(--min-free-gb {args.min_free_gb}); stopping cleanly so the "
                    f"finished embeddings survive.",
                    file=sys.stderr,
                )
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
                # Without this the heartbeat reported done:0 forever and the
                # webhook only fired on failures, so a run that was merely slow
                # looked exactly like one that had died.
                reporter.update(True, units=stats["n_patches"])
            except Exception as e:
                failed += 1
                print(f"  {record.patient_id}: EXTRACT FAILED - {e}", file=sys.stderr)
                reporter.update(False, extra={"last_error": f"{record.patient_id}: {e}"})
            finally:
                # The whole point: reclaim the disk immediately, then hand the
                # slot back so another download can start.
                if not args.keep_slides:
                    Path(slide_path).unlink(missing_ok=True)
                prefetcher.release()
    finally:
        prefetcher.stop()
        reporter.finish(f"{patches:,} patches")

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
    p.add_argument("--labels-csv", default=None,
                   help="restrict to patients present in this CSV. Skips slides "
                        "that cannot contribute to training")
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--pooling", choices=["cls", "cls_mean"], default=None,
                   help="override the encoder's token pooling. cls_mean concatenates "
                        "the CLS token with the mean of the patch tokens, doubling the "
                        "width; it is what Bioptimus recommend for downstream use, while "
                        "cls is the convention these models are benchmarked under.")
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
    p.add_argument("--prefetch", type=int, default=4,
                   help="max slides held on disk at once")
    p.add_argument("--download-workers", type=int, default=4,
                   help="concurrent GDC streams. Measured 25.6 MB/s single vs "
                        "143 MB/s at 4 streams, so this matters more than anything else")
    p.add_argument("--decode-workers", type=int, default=max(4, (os.cpu_count() or 8)),
                   help="threads decoding tiles. This, not the accelerator, is usually the limit")
    p.add_argument("--queue-depth", type=int, default=3, help="tile batches decoded ahead")
    p.add_argument("--tpu-cores", type=int, default=1,
                   help="XLA devices to encode across. A Kaggle TPU v5e-8 has 8 and the "
                        "loop historically used 1. Replicas are independent (the encoder "
                        "is frozen), so this is pure data parallelism. Ignored off TPU")
    p.add_argument("--min-free-gb", type=float, default=3.0,
                   help="stop cleanly when the output volume drops below this")

    p.add_argument("--one-per-patient", action="store_true", default=True)
    p.add_argument("--all-slides", dest="one_per_patient", action="store_false")
    p.add_argument("--include-frozen", action="store_true",
                   help="include TS/BS frozen sections (not advisable for grading)")
    p.add_argument("--keep-slides", action="store_true", help="do not delete after encoding")
    p.add_argument("--assume-mpp", type=float, default=None)
    p.add_argument("--fp32-store", action="store_true")
    p.add_argument("--allow-noncommercial", action="store_true")
    p.add_argument("--random-weights", action="store_true",
                   help="skip the pretrained download and use random weights. Exercises "
                        "the full pipeline without a gated fetch; embeddings are garbage")
    p.add_argument("--webhook-url", default=None,
                   help="Discord/Slack/Telegram hook for phone notifications")
    p.add_argument("--notify-every", type=int, default=25, help="slides between pings")
    return p


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
