#!/usr/bin/env python
"""Stage and push one kernel to Kaggle.

The kernel metadata files here declare a ``code_file`` whose name differs from
the script on disk (``pipeline.py`` vs ``pipeline_tpu.py``), and ``kaggle
kernels push`` insists on a directory containing exactly one file literally
named ``kernel-metadata.json``. This assembles that directory instead of
requiring it to be built by hand each time.

    python kaggle/push.py cores_probe
    python kaggle/push.py pipeline_tpu --dry-run

Two Windows-specific traps are handled:

* The CLI writes an upload-cache file whose path mirrors the source directory
  under ``%TEMP%/.kaggle/uploads`` and does **not** create the intermediate
  directories, so a push from an unseen path dies with ``[Errno 2]``. The
  parent is created up front.
* That mirrored path can exceed MAX_PATH, so staging happens in a short
  directory rather than under a deep scratch tree.

A push creates a new version and immediately queues a run with whatever
accelerator the metadata requests. With TPU concurrency of 1, do not push a TPU
kernel while another TPU run is queued or active - the new run will contend for
the same slot.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def stage(name: str) -> tuple[Path, dict]:
    meta_path = HERE / f"{name}.kernel-metadata.json"
    if not meta_path.exists():
        sys.exit(f"no metadata: {meta_path}")
    meta = json.loads(meta_path.read_text())

    script = HERE / f"{name}.py"
    if not script.exists():
        sys.exit(f"no script: {script}")

    # Refuse to publish something that cannot even be parsed. A kernel with a
    # syntax error still queues, still waits out the ~40 minute TPU queue, and
    # only then dies in two seconds - so the cost of shipping one is measured
    # in wall clock, not in the second it takes to check here. This exists
    # because exactly that happened on 2026-08-22.
    import ast

    try:
        ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    except SyntaxError as e:
        sys.exit(f"REFUSING TO PUSH: {script.name} line {e.lineno}: {e.msg}")

    # Short staging root: the CLI mirrors this path under %TEMP%/.kaggle/uploads
    # and a long one blows past MAX_PATH.
    root = Path(tempfile.gettempdir()) / "pgpush"
    out = root / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shutil.copy(script, out / meta["code_file"])
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

    # Pre-create the upload-cache mirror the CLI forgets to make.
    mirror = Path(tempfile.gettempdir()) / ".kaggle" / "uploads"
    drive, rest = os.path.splitdrive(str(out))
    mirror = mirror / (drive.replace(":", "_")) / Path(rest).parent.as_posix().lstrip("/")
    mirror.mkdir(parents=True, exist_ok=True)

    return out, meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="basename of the kernel script, e.g. cores_probe")
    p.add_argument("--dry-run", action="store_true", help="stage but do not push")
    p.add_argument("--as", dest="as_id", default=None,
                   help="publish under a different kernel slug, e.g. pathgrade-smoke. "
                        "Use for reduced-scale reproductions so the production "
                        "kernel's history and output are left alone")
    p.add_argument("--kernel-source", action="append", default=[], metavar="OWNER/SLUG",
                   help="mount another kernel's output at /kaggle/input/<slug>/. "
                        "Used to chain short extraction runs so each inherits the "
                        "slides its predecessors encoded")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="prepend an os.environ default to the staged script. Kaggle "
                        "has no way to pass environment variables to a kernel, so a "
                        "scale knob like PATHGRADE_LIMIT=24 has to be baked in")
    args = p.parse_args()

    out, meta = stage(args.name)

    if args.as_id:
        meta["id"] = f"{meta['id'].split('/')[0]}/{args.as_id}"
        meta["title"] = args.as_id

    if args.kernel_source:
        # Mounts a previous run's output at /kaggle/input/<slug>/. The pipeline
        # seeds its features directory from there, so a chain of short runs
        # accumulates the cohort without any of them having to survive for
        # hours - and without shipping gigabytes back through this machine.
        meta["kernel_sources"] = args.kernel_source

    if args.as_id or args.kernel_source:
        (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

    if args.env:
        script = out / meta["code_file"]
        header = ["import os  # injected by push.py --env"]
        for item in args.env:
            k, _, v = item.partition("=")
            header.append(f"os.environ.setdefault({k!r}, {v!r})")
        script.write_text("\n".join(header) + "\n" + script.read_text())
        print(f"  env:      {args.env}")
    accel = "TPU" if meta.get("enable_tpu") == "true" else (
        "GPU" if meta.get("enable_gpu") == "true" else "CPU")
    print(f"{meta['id']}  [{accel}]  code_file={meta['code_file']}")
    print(f"  datasets: {meta.get('dataset_sources', [])}")
    print(f"  staged:   {out}")
    if args.dry_run:
        print("  (dry run - not pushed)")
        return 0

    proc = subprocess.run(["kaggle", "kernels", "push", "-p", str(out)],
                          capture_output=True, text=True)
    print((proc.stdout or "") + (proc.stderr or ""))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
