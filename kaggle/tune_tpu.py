"""Hyperparameter search on the TPU VM's host CPU.

The first real training run used entirely default hyperparameters. This is the
tuning that was never done, run where the hardware suits it: the TPU VM host
carries 224 vCPU and ~405 GB RAM, which is what the search actually needs -
the TPU *devices* are irrelevant here, since a 530K-parameter head over
precomputed features is a CPU job and XLA would recompile on every shape change.

That RAM is the point. The feature cache holds the whole 435-slide cohort
(~8 GB as float32) in memory, so trials stop being disk-bound; on a 16 GB
laptop the cache cannot engage and a single trial takes hours.

Bounded and resumable, for the same reason extraction is: containers here have
been killed mid-run and committed nothing. The Optuna study is SQLite on
/kaggle/working, each finished trial is durable the moment it lands, and
mounting a previous run via ``kernel_sources`` continues the same study rather
than restarting it.

    python kaggle/push.py tune_tpu --kernel-source apexblue/pathgrade-c8-final
    python kaggle/push.py tune_tpu --as pathgrade-tune2 \
        --kernel-source apexblue/pathgrade-tune1        # continue the study

The locked test set is never read. The objective is mean CV QWK; evaluating the
winner is a separate, single, deliberate step.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()
WORK = Path("/kaggle/working")
OUT = WORK / "features"
TUNE = WORK / "tuning"
for d in (OUT, TUNE):
    d.mkdir(parents=True, exist_ok=True)

TRAIL = WORK / "tune_trail.txt"
with open(TRAIL, "w"):
    pass


def trail(step: str, detail: str = "") -> None:
    line = f"[{time.time() - T0:8.1f}s] {step} {detail}".rstrip()
    print(line, flush=True)
    try:
        with open(TRAIL, "a") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except OSError:
        pass


trail("BOOT", f"python {sys.version.split()[0]}")

TRIALS = int(os.environ.get("PATHGRADE_TRIALS", "60"))
EPOCHS = int(os.environ.get("PATHGRADE_TUNE_EPOCHS", "20"))
BUDGET_H = float(os.environ.get("PATHGRADE_TUNE_HOURS", "2.5"))


def find_src(marker="src/pathgrade/__init__.py", root="/kaggle/input"):
    for depth in ("*", "*/*", "*/*/*"):
        for hit in glob.glob(os.path.join(root, depth, marker)):
            return hit[: -len(marker) - 1]
    return None


SRC = find_src()
if SRC is None:
    trail("FATAL", "pathgrade-src not mounted")
    sys.exit("FATAL: pathgrade-src not mounted")
trail("SRC", SRC)

# Seed features and any previous study from mounted kernel outputs.
seeded = 0
for src_h5 in glob.glob("/kaggle/input/**/features/*.h5", recursive=True):
    dest = OUT / os.path.basename(src_h5)
    if not dest.exists():
        try:
            shutil.copy(src_h5, dest)
            seeded += 1
        except OSError as e:
            print(f"  seed failed {os.path.basename(src_h5)}: {e}", flush=True)
trail("SEEDED", f"{seeded} slides ({len(list(OUT.glob('*.h5')))} total)")

for prev in glob.glob("/kaggle/input/**/tuning/study.db", recursive=True):
    shutil.copy(prev, TUNE / "study.db")
    trail("RESUME", f"continuing study from {prev}")
    break

splits = next(iter(glob.glob("/kaggle/input/**/splits.json", recursive=True)), None)
labels = next(iter(glob.glob("/kaggle/input/**/labels_available.csv", recursive=True)), None)
if not splits or not labels:
    trail("FATAL", f"splits={splits} labels={labels}")
    sys.exit("FATAL: need splits.json and labels_available.csv from a previous run")
# Copy locally: load_splits verifies a SHA-256 fingerprint, so the file the
# study tunes against is pinned to the one the cohort was actually split with.
shutil.copy(splits, WORK / "splits.json")
shutil.copy(labels, WORK / "labels_available.csv")
trail("SPLITS", f"{splits} -> fingerprinted, test set stays sealed")

print("installing optuna ...", flush=True)
subprocess.run("pip install -q optuna 2>&1 | tail -2", shell=True, check=False)

cmd = [
    sys.executable, "-u", f"{SRC}/scripts/06_tune.py",
    "--feature-dir", str(OUT),
    "--splits", str(WORK / "splits.json"),
    "--labels", str(WORK / "labels_available.csv"),
    "--out", str(TUNE),
    "--study", str(TUNE / "study.db"),
    "--trials", str(TRIALS),
    "--epochs", str(EPOCHS),
    "--timeout-hours", str(BUDGET_H),
    "--workers", os.environ.get("PATHGRADE_TUNE_WORKERS", "16"),
]
trail("STEP", f"{TRIALS} trials, {EPOCHS} epochs each, {BUDGET_H}h budget")

# As a child, so a crash cannot take the finished trials with it - the study
# database is the deliverable and it is written trial by trial.
rc = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": f"{SRC}/src"}).returncode
trail("TUNE_RC", str(rc))

best = TUNE / "best_params.json"
if best.exists():
    print("\n" + json.dumps(json.loads(best.read_text()), indent=2), flush=True)
    trail("DONE", f"best written after {(time.time()-T0)/3600:.2f}h")
else:
    trail("NO_RESULT", "study.db may still hold completed trials; check it")
print(f"\ntotal elapsed {(time.time() - T0) / 3600:.2f} h", flush=True)
