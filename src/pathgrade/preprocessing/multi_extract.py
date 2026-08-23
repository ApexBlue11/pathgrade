"""Extraction with one process per TPU device.

A Kaggle TPU v5e-8 exposes eight XLA devices and the extraction loop has always
used one of them. Driving all eight from eight threads in a single process is
**measured-broken**, not merely slow: on 2026-08-22 seven of eight threads died
inside ``XLAGraphExecutor::SyncLiveTensorsGraph`` with ``Check failed:
tensor_data``, because ``mark_step()`` syncs every live tensor on a device
rather than only the calling thread's. See ``encoders.build_encoders``, which
refuses to replicate for that reason.

The supported model is one process per device, which is what
``torch_xla.distributed.xla_multiprocessing.spawn`` sets up - it handles the
TPU chip-visibility environment variables that are otherwise fiddly and
version-specific.

**This module lives in the package rather than in a Kaggle kernel script, and
that is load-bearing.** ``spawn`` re-imports the entry module in each child; a
kernel script re-imported that way would re-run the entire pipeline - preflight,
token check, training - eight times over. A library module is imported cleanly.
It also means the code ships inside ``pathgrade-src``, so every worker reads
the identical version from the mounted dataset.

Work is split with the existing ``--shard i --num-shards n`` flags, so each
process owns a disjoint slice of the cohort, writes its own journal and
heartbeat, and writes per-slide HDF5 files that cannot collide. Nothing is
shared and nothing is synchronised, because the encoder is frozen and every
slide is independent.
"""

from __future__ import annotations

import os
import sys
import time


def _world() -> tuple[int | None, int | None]:
    """(ordinal, world size) from the XLA runtime, or (None, None) off TPU."""
    try:
        import torch_xla.runtime as xr

        return xr.global_ordinal(), xr.world_size()
    except Exception:
        return None, None


def _worker(index: int, argv: list[str]) -> None:
    """One process, one XLA device, one shard of the slide list.

    The shard count is read from the runtime rather than passed in, because
    ``spawn`` is called with ``nprocs=None`` - torch_xla rejects an explicit
    count outright ("Unsupported nprocs (8). Please use nprocs=1 or None") and
    decides the device count itself. Guessing it here would silently mis-split
    the cohort if the runtime disagreed.
    """
    from .stream_extract import build_parser, run

    ordinal, world = _world()
    shard = index if ordinal is None else ordinal
    num_shards = world or int(os.environ.get("PATHGRADE_NPROCS", "8"))

    args = build_parser().parse_args(list(argv))
    args.shard = shard
    args.num_shards = num_shards
    args.device = "xla"
    args.tpu_cores = 1          # threading across devices is broken; never here
    index = shard
    print(f"[shard {shard}/{num_shards}] starting", flush=True)
    try:
        run(args)
    except BaseException as e:                      # noqa: BLE001
        # One dead shard must not abort the other seven. Each writes its own
        # slides, so the cohort is simply short by that shard's slice, which a
        # later run picks up because extraction skips what already exists.
        import traceback

        print(f"[shard {index}] FAILED {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


def prewarm(encoder_name: str) -> float:
    """Fetch the encoder once, before forking, so children hit a warm cache.

    Eight processes each calling ``timm.create_model(pretrained=True)`` against
    a cold cache would race on the same multi-gigabyte download.
    """
    from huggingface_hub import snapshot_download

    from ..encoders import get_spec

    repo = get_spec(encoder_name).hf_hub_id.split("hf-hub:")[-1]
    t0 = time.time()
    snapshot_download(repo, token=os.environ.get("HF_TOKEN"))
    return time.time() - t0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    from .stream_extract import build_parser

    args = build_parser().parse_args(argv)        # validate before spawning

    if not args.random_weights:
        secs = prewarm(args.encoder)
        print(f"encoder cached in {secs:.0f}s", flush=True)

    # Import only now. Touching torch_xla in the parent would claim the TPU
    # chips for the parent's lifetime and leave nothing for the children.
    import torch_xla.distributed.xla_multiprocessing as xmp

    # nprocs MUST be None. An explicit count is rejected outright:
    #   ValueError: Unsupported nprocs (8). Please use nprocs=1 or None
    #   (default). If None, spawn will use all available devices.
    # To cap it, set TPU_NUM_DEVICES in the environment instead.
    print("spawning one process per available XLA device", flush=True)
    xmp.spawn(_worker, args=(argv,), nprocs=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
