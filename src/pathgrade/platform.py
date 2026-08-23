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
#
# RETRACTION: this table used to carry "tpu-v5e-8": 1226.2 labelled MEASURED.
# It was fiction, and it is the most expensive wrong number this project has
# produced - it made the encoder look free and sent tuning at the decode path
# instead. XLA builds its graph lazily, so a benchmark that never materialises
# the result times graph *construction*. Nothing forced execution there.
#
# The corrected figure is per-device and comes from a run that calls .cpu() on
# every batch. Note it is one device: a v5e-8 exposes eight, and until
# `--tpu-cores` landed the extraction loop used exactly one of them.
ENCODE_PATCHES_PER_SEC = {
    "cpu-12core": 0.38,        # measured
    "p100": 25.0,              # estimated from 12 TFLOPS fp16
    "t4x2": 100.0,             # estimated
    "tpu-v5e-1core": 123.5,    # measured with a forced .cpu() - one of eight devices
}

# Aggregate across eight devices driven by eight threads in ONE process.
# MEASURED 2026-08-22 (kaggle/cores_probe.py) and the answer is: it does not
# work. Not "does not scale" - does not work.
#
#   devices  threads completed  real aggregate
#         1                1/1     123.5 /s
#         2                2/2     247   /s   (clean 2x in steady state)
#         4                1/4      73   /s
#         8                1/8      68   /s   <- slower than one device
#
# The failures are all the same crash, in the XLA graph executor itself:
#   torch_xla/csrc/xla_graph_executor.cpp:691 : Check failed: tensor_data
#     torch_xla::XLAGraphExecutor::SyncLiveTensorsGraph(...)
# mark_step() syncs every live tensor on a device rather than only the calling
# thread's, so concurrent threads tear each other's in-flight tensors out from
# under the forward pass. Surviving threads still ran at 123 patches/s, so this
# is a correctness failure and not contention.
#
# A trap worth recording: the probe's own summary reported "speedup 8.54,
# threads_scale: true" because it divided *intended* work by wall time without
# checking whether the threads finished. Same shape of error as the retracted
# 1226 figure - a throughput number that never verified the work happened.
ENCODE_PATCHES_PER_SEC_8CORE: float | None = None   # threads: broken, see above
XLA_MULTITHREAD_SAFE = False
# Using all eight needs one process per device (xla_multiprocessing.spawn);
# --shard/--num-shards already exist for exactly that split. Not yet built.

# Confirmed on the same session, one process:
XLA_DEVICES_VISIBLE = 8              # xla:0..xla:7, TPU:0..TPU:7
TPU_VM_RAM_GB = 405.7
HOPTIMUS0_FIRST_LOAD_SECONDS = 61.6  # gated fetch + build + XLA move + width probe

# Tile decode on the Kaggle TPU VM (224 vCPU). Peaks around 16 threads;
# 224 threads is slower, so oversubscription hurts.
#
# RETRACTION: previously 1654.6, sampled over 72 tiles of an 11 MB slide that
# sat entirely in page cache. Real slides run 1900-2500 tiles/s from any mount,
# which is why storage turned out never to have been the bottleneck.
TPU_VM_DECODE_TILES_PER_SEC = 2200.0   # midpoint of 1900-2500 on real slides

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
