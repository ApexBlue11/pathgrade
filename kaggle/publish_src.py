#!/usr/bin/env python
"""Publish a new version of the ``apexblue/pathgrade-src`` dataset.

Kernels import the pathgrade package from this dataset rather than from the
repo, so any change under ``src/`` is invisible to a run until this has been
executed. A kernel push and a dataset publish are independent: pushing new
kernel code against a stale dataset silently runs the old library.

    python kaggle/publish_src.py -m "multi-device encode"

Two things here are load-bearing:

* **The dataset holds a file the repo does not.** ``tcga_hnsc_labels.csv`` -
  the grade labels the whole project trains against - exists only in the
  dataset; it was never committed. Publishing a fresh directory built from the
  repo alone would delete it and break training after hours of extraction. So
  the current version is downloaded first and the repo is layered on top.
* **``--dir-mode`` defaults to ``skip``**, which silently ignores every
  subdirectory and would upload a dataset with no ``src/`` at all. ``zip`` is
  passed explicitly; Kaggle expands it server-side, which is why the file list
  shows individual paths.
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

REPO = Path(__file__).resolve().parents[1]
DATASET = "apexblue/pathgrade-src"
# Copied from the repo on every publish; anything else in the dataset is kept.
FROM_REPO = ["src", "configs", "scripts", "requirements.txt"]
# Must survive a publish. Absence means the download step failed - abort rather
# than ship a dataset that trains on nothing.
MUST_KEEP = ["tcga_hnsc_labels.csv"]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, text=True, **kw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--message", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stage = Path(tempfile.gettempdir()) / "pgsrc"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    print(f"downloading current {DATASET} (to preserve files absent from the repo) ...")
    proc = run(["kaggle", "datasets", "download", "-d", DATASET,
                "-p", str(stage), "--unzip"], capture_output=True)
    print((proc.stdout or "")[-400:] + (proc.stderr or "")[-400:])

    missing = [f for f in MUST_KEEP if not (stage / f).exists()]
    if missing:
        sys.exit(f"ABORT: {missing} not in the downloaded dataset. Publishing now "
                 f"would delete it. Check the download step before retrying.")
    print(f"preserved: {MUST_KEEP}")

    for item in FROM_REPO:
        src, dst = REPO / item, stage / item
        if not src.exists():
            print(f"  (skip {item}: not in repo)")
            continue
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".pytest_cache")) if src.is_dir() else shutil.copy(src, dst)
        print(f"  refreshed {item}")

    (stage / "dataset-metadata.json").write_text(json.dumps(
        {"title": "pathgrade-src", "id": DATASET, "licenses": [{"name": "unknown"}]},
        indent=2))

    # The CLI mirrors the source path under %TEMP%/.kaggle/uploads without
    # creating the intermediate directories, and dies with [Errno 2] if absent.
    mirror = Path(tempfile.gettempdir()) / ".kaggle" / "uploads"
    drive, rest = os.path.splitdrive(str(stage))
    (mirror / drive.replace(":", "_") / Path(rest).parent.as_posix().lstrip("/")
     ).mkdir(parents=True, exist_ok=True)

    n = sum(1 for _ in stage.rglob("*") if _.is_file())
    print(f"staged {n} files in {stage}")
    if args.dry_run:
        print("(dry run - not published)")
        return 0

    proc = run(["kaggle", "datasets", "version", "-p", str(stage),
                "-m", args.message, "-r", "zip", "-t"], capture_output=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out[-800:])
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
