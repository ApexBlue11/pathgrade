"""Measured platform constraints. Facts, not assumptions.

Everything here was measured against a live Kaggle account on 2026-08-20 by
pushing probe kernels until the API refused, and by benchmarking on the
returned sessions. Numbers that were guessed elsewhere in this repo have caused
real planning errors, so this module records only what was observed, with the
date it was observed.

The concurrency limits are the important part. They came straight out of the
Kaggle API error messages:

    Maximum batch TPU session count of 1 reached.
    Maximum batch GPU session count of 2 reached.
    Maximum batch CPU session count of 5 reached.

The TPU limit of **1** is the binding constraint on any sharding plan: TPU work
cannot be parallelised across sessions at all.
"""

from __future__ import annotations

from dataclasses import dataclass

MEASURED_ON = "2026-08-20"


@dataclass(frozen=True)
class SessionSpec:
    name: str
    max_concurrent: int
    vcpu: int
    weekly_hours: float | None
    session_hours: float
    accelerator: str
    can_encode: bool
    note: str = ""


KAGGLE = {
    "tpu": SessionSpec(
        name="tpu", max_concurrent=1, vcpu=224, weekly_hours=20.0, session_hours=9.0,
        accelerator="TPU VM v5e-8 (8 XLA devices, torch_xla 2.8.0)", can_encode=True,
        note=(
            "Only ONE at a time, and it queued ~23 min before starting. But the "
            "host is enormous: 224 vCPU, 406 GB RAM, 1098 GB on /kaggle/tmp. "
            "One session covers the whole cohort, so the concurrency limit of 1 "
            "does not bite."
        ),
    ),
    "gpu": SessionSpec(
        name="gpu", max_concurrent=2, vcpu=4, weekly_hours=30.0, session_hours=12.0,
        accelerator="Tesla P100 16GB (default)", can_encode=True,
        note="Default accelerator is P100, NOT T4x2. Request T4x2 explicitly.",
    ),
    "cpu": SessionSpec(
        name="cpu", max_concurrent=5, vcpu=4, weekly_hours=None, session_hours=12.0,
        accelerator="none", can_encode=False,
        note=(
            "Cannot run a 1B-parameter encoder. ViT-g/14 measured 0.38 patches/s "
            "on a 12-core box; a 4-vCPU session is slower still, putting 50 slides "
            "at 400+ hours. Useful only for download/tiling, never for encoding."
        ),
    ),
}

# Kaggle disk, measured: /kaggle/working persists and becomes the dataset output.
KAGGLE_WORKING_GB = 20.9      # measured; this is the hard output cap
KAGGLE_TMP_GB = 1098.4        # measured on the TPU VM - far larger than documented

# Tile decode, threaded with a shared OpenSlide handle, 0.5 um/px -> 224 px.
# 329 tiles/s measured on 12 vCPU; roughly linear in cores until disk-bound.
DECODE_TILES_PER_SEC_PER_VCPU = 27.0

# ViT-g/14 (H-optimus-0 architecture, 1.01B params) forward throughput.
ENCODE_PATCHES_PER_SEC = {
    "cpu-12core": 0.38,        # measured
    "p100": 25.0,              # estimated from 12 TFLOPS fp16
    "t4x2": 100.0,             # estimated
    "tpu-v5e-8": 1226.2,       # MEASURED, bf16 batch 64
}

# Measured on the Kaggle TPU VM (224 vCPU). Peaks at 16 threads; 224 threads is
# slower, so oversubscription hurts. NOTE: sampled over only 72 tiles of an
# 11 MB slide, so page cache likely flattered it - treat as an upper bound.
TPU_VM_DECODE_TILES_PER_SEC = 1654.6

# GDC sustained download, measured per-origin. Wildly origin-dependent.
GDC_MBPS = {
    "home-broadband-in": 0.23,   # measured: dev machine, single stream
    "kaggle-1-stream": 25.62,    # measured on the TPU VM
    "kaggle-2-streams": 63.56,   # measured, aggregate
    "kaggle-4-streams": 143.26,  # measured, aggregate -> 5.6x over single
}

# Parallel GDC streams scale near-linearly: the per-connection ceiling is low
# but there is no per-client rate limit. Download dominates the job, so this is
# the highest-value setting in the pipeline.
DEFAULT_DOWNLOAD_STREAMS = 4


def viable_encoders() -> list[SessionSpec]:
    """Kaggle session types that can actually run the encoder."""
    return [s for s in KAGGLE.values() if s.can_encode]


def max_parallel_shards() -> int:
    """Total simultaneous extraction workers available on Kaggle."""
    return sum(s.max_concurrent for s in viable_encoders())


def describe() -> str:
    rows = [
        "",
        f"Kaggle limits, measured {MEASURED_ON}",
        f"{'session':<8}{'concurrent':>11}{'vCPU':>6}{'weekly h':>10}{'encode?':>9}  accelerator",
        "-" * 78,
    ]
    for s in KAGGLE.values():
        weekly = f"{s.weekly_hours:.0f}" if s.weekly_hours else "unmetered"
        rows.append(
            f"{s.name:<8}{s.max_concurrent:>11}{s.vcpu:>6}{weekly:>10}"
            f"{('yes' if s.can_encode else 'NO'):>9}  {s.accelerator}"
        )
    rows.append("")
    rows.append(f"max parallel extraction workers: {max_parallel_shards()} "
                f"(1 TPU + 2 GPU; CPU sessions cannot encode)")
    rows.append("")
    for s in KAGGLE.values():
        if s.note:
            rows.append(f"  {s.name}: {s.note}")
    rows.append("")
    return "\n".join(rows)
