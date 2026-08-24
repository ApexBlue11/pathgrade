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

    # UNRESOLVED after five real-TPU attempts. Recorded in full because
    # guessing at this layer has cost real sessions each time; the next person
    # (or model) should not re-derive it from nothing.
    #
    # Kaggle's actual environment (captured 2026-08-24, torch_xla 2.8.0,
    # v5litepod-8):
    #   TPU_PROCESS_ADDRESSES      local
    #   TPU_CHIPS_PER_HOST_BOUNDS  2,4,1
    #   TPU_HOST_BOUNDS            1,1,1
    #   TPU_WORKER_ID              0
    #   TPU_WORKER_HOSTNAMES       localhost
    #   TPU_RUNTIME_METRICS_PORTS  8431,8432,...,8438   (8 ports, correctly)
    #
    # Attempt 1 - nprocs=8: rejected outright, "Unsupported nprocs (8).
    #   Please use nprocs=1 or None."
    # Attempt 2 - nprocs=None, default fork: all 8 children died in
    #   XLAGraphExecutor::SyncLiveTensorsGraph, "Check failed:
    #   !g_computation_client_initialized". Diagnosed (wrongly, at the time)
    #   as fork inheriting an initialised client.
    # Attempt 3 - start_method="spawn": IDENTICAL failure. Disproved attempt
    #   2's diagnosis. The stack trace names PrepareToExit(), so the check
    #   fires during teardown of a child that never initialised - the real
    #   error, one line up, is
    #     Could not find SliceBuilder port 8476 in any of the 0 ports
    #     provided in `tpu_process_addresses`="local"
    #   i.e. libtpu in single-process mode being asked to run eight.
    # Attempt 4 - clear TPU_PROCESS_ADDRESSES/TPU_HOST_BOUNDS/TPU_WORKER_ID/
    #   TPU_WORKER_HOSTNAMES so torch_xla derives the topology fresh: a
    #   DIFFERENT and worse crash appeared -
    #     File ".../torch_xla/_internal/tpu.py", line 259, in configure_topology
    #       default_process_bounds = MeshShape.from_string(...)
    #     File ".../tpu.py", line 73, in from_string
    #       dims = tuple(int(d) for d in mesh.split(','))
    #     AttributeError: 'NoneType' object has no attribute 'split'
    #   Removing TPU_HOST_BOUNDS='1,1,1' most likely destroyed an input that
    #   configure_topology's own derivation needed, rather than clearing an
    #   obstruction - clearing state made this measurably worse, not better.
    #
    # Given four distinct failure modes from four distinct hypotheses, this is
    # far more likely a genuine defect or an unsupported configuration in this
    # torch_xla build on Kaggle's TPU than something fixable by more env
    # guessing. Left OFF by default (see PATHGRADE_NPROCS default in
    # pipeline_tpu.py) so it stops costing minutes on every chunk. A next
    # attempt should try TPU_PROCESS_ADDRESSES alone (the one value that
    # actually names the restriction) rather than the broad clear below, which
    # remains opt-in for exactly that purpose.
    tpu_env = {k: v for k, v in sorted(os.environ.items())
               if k.startswith(("TPU_", "PJRT_", "XLA_", "CLOUD_TPU", "LIBTPU"))}
    print(f"TPU env before spawn: {json.dumps(tpu_env)}", flush=True)

    if os.environ.get("PATHGRADE_CLEAR_TPU_ENV") == "1":
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
