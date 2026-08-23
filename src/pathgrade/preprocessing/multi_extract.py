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

import json
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

    # The TPU topology environment decides whether multiprocess is even
    # possible, and on Kaggle it is pre-set. Children died with
    #   Could not find SliceBuilder port 8476 in any of the 0 ports provided
    #   in `tpu_process_addresses`="local"
    # which is libtpu in single-process mode being asked to run eight. Print
    # what is actually set before touching it - three attempts have now been
    # spent guessing at this layer.
    tpu_env = {k: v for k, v in sorted(os.environ.items())
               if k.startswith(("TPU_", "PJRT_", "XLA_", "CLOUD_TPU", "LIBTPU"))}
    print(f"TPU env before spawn: {json.dumps(tpu_env)}", flush=True)

    # torch_xla configures the multiprocess topology itself, but only for keys
    # it does not find already set. Kaggle's single-process defaults therefore
    # win and leave the children with nowhere to bind. Clear them and let
    # torch_xla derive the eight-way layout from scratch.
    if os.environ.get("PATHGRADE_CLEAR_TPU_ENV", "1") == "1":
        for key in ("TPU_PROCESS_ADDRESSES", "TPU_VISIBLE_CHIPS",
                    "TPU_VISIBLE_DEVICES", "TPU_PROCESS_BOUNDS",
                    "TPU_CHIPS_PER_PROCESS_BOUNDS", "CLOUD_TPU_TASK_ID",
                    "TPU_HOST_BOUNDS", "TPU_WORKER_ID", "TPU_WORKER_HOSTNAMES"):
            if key in os.environ:
                print(f"  clearing {key}={os.environ[key]!r}", flush=True)
                os.environ.pop(key, None)

    # Two constraints, both learned from real runs rather than documentation.
    #
    # nprocs MUST be None. An explicit count is rejected outright:
    #   ValueError: Unsupported nprocs (8). Please use nprocs=1 or None
    #   (default). If None, spawn will use all available devices.
    #
    # start_method is "spawn" rather than the default fork. CORRECTION: this
    # was first added believing forked children inherited an already-
    # initialised computation client, because all eight died with
    #   F runtime.cpp:21] Check failed: !g_computation_client_initialized
    #   InitializeComputationClient() can only be called once.
    # That hypothesis was WRONG - the same failure occurred with "spawn". The
    # stack trace names PrepareToExit(), so that check fires during teardown of
    # a child that never initialised successfully; the real error is the
    # SliceBuilder/tpu_process_addresses one above. "spawn" is kept because a
    # fresh interpreter is the cleaner arrangement, not because it fixed
    # anything.
    #
    # It is safe only because _worker lives in an importable library module:
    # "spawn" re-imports the entry module in each child (as __mp_main__, so the
    # __main__ guard does not re-fire), which would be catastrophic if the
    # entry point were the Kaggle kernel script.
    print("spawning one process per available XLA device", flush=True)
    xmp.spawn(_worker, args=(argv,), nprocs=None, start_method="spawn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
