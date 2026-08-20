"""Generate the Kaggle extraction notebook.

Kept as a generator rather than a hand-maintained .ipynb so the cells stay
diffable in review and cannot drift out of sync with the CLI flags.
"""

import json
from pathlib import Path

REPO = "https://github.com/YOUR_ORG/pathgrade.git"   # <- point at your fork

MD_INTRO = """\
# pathgrade — TCGA-HNSC feature extraction on Kaggle

Streams slides from GDC, encodes them with **H-optimus-0** (Apache-2.0), and
discards each slide immediately. No local WSI storage required.

**Why this notebook installs a package instead of defining classes in cells.**
Kaggle worker processes (DataLoader workers, and anything spawned rather than
forked) do not inherit the notebook's `__main__` namespace, so a class defined
in a cell cannot be unpickled by a worker — the classic
`AttributeError: Can't get attribute '...' on <module '__main__'>`. The usual
workaround is `%%writefile` to dump a module to disk first. Installing the
package with `pip install -e` does the same job properly: every symbol lives in
a real file on `sys.path`, so workers and threads import it cleanly with no
duplicated source.

**Disk layout matters here.**

| path | size | persists | use for |
|---|---|---|---|
| `/kaggle/working` | ~20 GB | yes, becomes the dataset | embeddings only |
| `/kaggle/tmp` | ~60 GB | no | in-flight slide downloads |

Slides average ~0.94 GB and the largest is ~4 GB, so the cache **must** live on
`/kaggle/tmp`. Pointing it at `/kaggle/working` fills the output quota after a
few slides.
"""

CELL_INSTALL = """\
# Install the package so worker processes can import it (see note above).
!git clone -q {repo} /kaggle/tmp/pathgrade || true
!pip install -q -e /kaggle/tmp/pathgrade

import pathgrade, sys
print("pathgrade", pathgrade.__version__, "| python", sys.version.split()[0])
""".format(repo=REPO)

CELL_DEVICE = """\
# TPU needs torch_xla; on a GPU/CPU session this cell is a no-op.
import os, torch

ACCEL = os.environ.get("PATHGRADE_ACCEL", "auto")   # "xla" | "cuda" | "auto"

if ACCEL == "xla":
    !pip install -q torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
    import torch_xla.core.xla_model as xm
    print("TPU device:", xm.xla_device(), "| cores:", xm.xrt_world_size())
else:
    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

import multiprocessing
print("vCPUs:", multiprocessing.cpu_count())
"""

CELL_DISK = """\
import shutil

for path in ["/kaggle/working", "/kaggle/tmp"]:
    total, used, free = shutil.disk_usage(path)
    print(f"{path:18s} free {free/1e9:6.1f} GB / total {total/1e9:6.1f} GB")

CACHE_DIR = "/kaggle/tmp/wsi_cache"      # scratch, never persisted
OUT_DIR   = "/kaggle/working/features"   # this becomes your Kaggle dataset
"""

CELL_PLAN = """\
# Query GDC and size the job before spending any quota.
!python /kaggle/tmp/pathgrade/scripts/00_plan_budget.py \\
    --max-patches 8000 --download-mbps 50 --num-shards 4
"""

CELL_TRIAL = """\
# ALWAYS run this first. Five slides tells you the real download and encode
# rates, which is the one number the budget estimate cannot guess.
!python /kaggle/tmp/pathgrade/scripts/01b_stream_extract.py \\
    --out-dir {out} --cache-dir {cache} \\
    --device $PATHGRADE_ACCEL --limit 5 --max-patches 8000 \\
    --batch-size 64 --decode-workers 8
"""

CELL_RUN = """\
# Full shard. Change SHARD to 0/1/2/3 in four parallel sessions.
SHARD, NUM_SHARDS = 0, 4

# Optional: a Discord/Slack/Telegram hook so progress reaches your phone.
WEBHOOK = ""   # e.g. "https://discord.com/api/webhooks/..."

!python /kaggle/tmp/pathgrade/scripts/01b_stream_extract.py \\
    --out-dir {out} --cache-dir {cache} \\
    --device $PATHGRADE_ACCEL \\
    --shard {{SHARD}} --num-shards {{NUM_SHARDS}} \\
    --max-patches 8000 --batch-size 64 --decode-workers 8 \\
    --max-hours 8 --min-free-gb 3 \\
    --webhook-url "{{WEBHOOK}}" --notify-every 25
"""

MD_MONITOR = """\
## Watching a run that takes hours

1. **Kaggle real-time logs.** Use *Save & Run All (Commit)*; the notebook viewer
   streams the log while it runs, so you do not need to keep this tab open.
2. **Webhook pings.** Set `WEBHOOK` above to a Discord/Slack/Telegram URL and
   the run pings every 25 slides plus once at the end. This is the only option
   that reaches a phone.
3. **Heartbeat files.** Every shard writes
   `heartbeat_shard<N>.json` next to its output after each slide, with rate and
   ETA. The cell below renders all shards at once — including from a previous,
   killed session, since the file persists in `/kaggle/working`.

A run that stops on `--max-hours` or `--min-free-gb` exits **cleanly** and skips
already-extracted slides next time, so resuming is just re-running the cell.
"""

CELL_HEARTBEAT = """\
from pathgrade.progress import format_heartbeats
print(format_heartbeats(OUT_DIR))
"""

CELL_VERIFY = """\
# Sanity-check the shard before publishing it as a dataset.
from pathgrade.data.io import verify_cohort
from pathlib import Path

pids = sorted(p.stem for p in Path(OUT_DIR).glob("*.h5"))
info = verify_cohort(OUT_DIR, pids)
for k, v in info.items():
    if k != "missing":
        print(f"  {k:18s} {v}")

size_gb = sum(p.stat().st_size for p in Path(OUT_DIR).glob("*.h5")) / 1e9
print(f"  {'output size':18s} {size_gb:.2f} GB")
"""


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.rstrip().split("\n")}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.rstrip().split("\n")}


def build():
    fmt = {"out": "$OUT_DIR", "cache": "$CACHE_DIR"}
    cells = [
        md(MD_INTRO),
        md("## 1 · Install"),
        code(CELL_INSTALL),
        md("## 2 · Accelerator\n\nSet `PATHGRADE_ACCEL` to `xla` for TPU or `cuda` for GPU."),
        code('import os\nos.environ["PATHGRADE_ACCEL"] = "xla"   # "cuda" on a GPU session'),
        code(CELL_DEVICE),
        md("## 3 · Disk"),
        code(CELL_DISK),
        md("## 4 · Size the job"),
        code(CELL_PLAN),
        md("## 5 · Trial run (do not skip)"),
        code(CELL_TRIAL.format(**fmt)),
        md("## 6 · Full shard"),
        code(CELL_RUN.format(**fmt)),
        md(MD_MONITOR),
        code(CELL_HEARTBEAT),
        md("## 7 · Verify before publishing"),
        code(CELL_VERIFY),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).parent / "kaggle_extract.ipynb"
    out.write_text(json.dumps(build(), indent=1))
    print(f"wrote {out}")
