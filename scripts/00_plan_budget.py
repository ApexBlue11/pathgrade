#!/usr/bin/env python
"""Step 0: query GDC and estimate the extraction budget before spending quota.

Queries the live GDC API for real slide counts and sizes, then projects
download, encode and storage cost across the hardware you might use. Throughput
numbers are assumptions - run one shard with --limit 5 first, read the actual
rates off the log, and re-run this with the measured values.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathgrade.encoders import get_spec
from pathgrade.preprocessing.gdc import one_slide_per_patient, query_slides, summarise

# Effective (not peak) throughput, bf16/fp16 inference including data stalls.
ACCELERATORS = {
    "tpu-v5e-8":  {"tflops": 550.0, "note": "Kaggle TPU VM v5e-8, ~35% MFU"},
    "tpu-v3-8":   {"tflops": 140.0, "note": "Kaggle TPU VM v3-8"},
    "gpu-t4x2":   {"tflops": 55.0,  "note": "Kaggle 2x T4, fp16"},
    "gpu-p100":   {"tflops": 12.0,  "note": "Kaggle P100, fp16"},
    "gpu-a100":   {"tflops": 180.0, "note": "A100 40GB, fp16"},
}

# ViT-g/14 at 224px: 40 layers x ~13.5 GFLOPs.
GFLOPS_PER_PATCH = {1536: 540.0, 2560: 350.0, 1024: 120.0, 3072: 540.0}


def fmt_hours(h: float) -> str:
    return f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="TCGA-HNSC")
    p.add_argument("--encoder", default="h-optimus-0")
    p.add_argument("--max-patches", type=int, default=4000)
    p.add_argument("--download-mbps", type=float, default=50.0, help="measured MB/s from GDC")
    p.add_argument("--decode-tiles-per-sec", type=float, default=1200.0, help="openslide+resize, all workers")
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--all-slides", action="store_true", help="every slide, not one per patient")
    args = p.parse_args()

    spec = get_spec(args.encoder)
    records = query_slides(args.project, diagnostic_only=True)
    if not args.all_slides:
        records = one_slide_per_patient(records)

    n = len(records)
    total_gb = sum(r.file_size for r in records) / 1e9
    patches = n * args.max_patches
    gflops = GFLOPS_PER_PATCH.get(spec.embed_dim, 400.0)
    total_flops = patches * gflops * 1e9

    print(f"\n{args.project}: {summarise(records)}")
    print(f"encoder {spec.name} ({spec.embed_dim}-d, {spec.licence})")
    print(f"cap {args.max_patches:,} patches/slide -> {patches:,} patches total\n")

    dl_h = total_gb * 1000 / args.download_mbps / 3600
    decode_h = patches / args.decode_tiles_per_sec / 3600
    store_gb = patches * spec.embed_dim * 2 / 1e9

    print(f"{'stage':<22}{'cost':>12}   note")
    print("-" * 74)
    print(f"{'download':<22}{fmt_hours(dl_h):>12}   {total_gb:.0f} GB at {args.download_mbps:.0f} MB/s")
    print(f"{'tile decode (CPU)':<22}{fmt_hours(decode_h):>12}   {args.decode_tiles_per_sec:.0f} tiles/s")
    print(f"{'output storage':<22}{store_gb:>10.1f} GB   fp16 embeddings\n")

    print(f"{'accelerator':<14}{'encode':>10}{'total*':>10}{'per shard':>12}   note")
    print("-" * 74)
    for name, meta in ACCELERATORS.items():
        enc_h = total_flops / (meta["tflops"] * 1e12) / 3600
        # Download overlaps with compute, so the wall clock is the slower leg
        # plus the pipeline's ramp-up, not the sum.
        total_h = max(dl_h, enc_h + decode_h) * 1.15
        print(
            f"{name:<14}{fmt_hours(enc_h):>10}{fmt_hours(total_h):>10}"
            f"{fmt_hours(total_h / args.num_shards):>12}   {meta['note']}"
        )

    print("\n* download overlaps with encoding, so total is the slower leg + 15% ramp,")
    print(f"  split across --num-shards {args.num_shards} parallel sessions.")

    # When download is the slower leg, extra patches are free until compute
    # catches up. On a fast accelerator that headroom is large, and more patches
    # per slide is strictly better for the downstream bags.
    print("\nFree patch headroom (largest cap before compute overtakes download):")
    for name, meta in ACCELERATORS.items():
        per_patch_h = gflops * 1e9 / (meta["tflops"] * 1e12) / 3600 + 1 / args.decode_tiles_per_sec / 3600
        free_total = dl_h / per_patch_h
        free_cap = int(free_total / n / 100) * 100
        verdict = "compute-bound already" if free_cap <= args.max_patches else f"raise cap to ~{free_cap:,}"
        print(f"  {name:<14}{verdict}")

    print("\nKaggle session limit is 9 h; pass --max-hours 8 so a shard stops cleanly")
    print("and resumes next session instead of being killed mid-slide.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
