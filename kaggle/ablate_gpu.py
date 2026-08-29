"""Controlled 2x2 ablation of the two fixes the diagnostic implied.

A 60-trial random search was tried twice and both sessions were cancelled with
one completed trial between them - a trial costs over an hour, so a full study
needs ~90 hours of a 20 h/week accelerator quota. The diagnostic had already
narrowed the question to two specific claims, so this tests those directly:
four interpretable runs instead of sixty opaque ones.

  scorer_no_decay - attention collapsed to exactly uniform because weight decay
                    was the only force acting on the scorer once the head could
                    fit the training set from the bag mean
  samples_per_slide - training saw ~348 samples per fold and drove training
                    loss to ~0

Reports attention entropy next to QWK, because an arm that lifts QWK while
leaving entropy at 1.0 has built a better mean-pooler, not a working attention
model - and the attention map is the product.

The locked test set is never read.
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
TUNE = WORK / "ablation"
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

EPOCHS = int(os.environ.get("PATHGRADE_ABLATE_EPOCHS", "25"))
ARMS = os.environ.get("PATHGRADE_ARMS", "baseline,scorer_no_decay,more_samples,both")
FOLDS = int(os.environ.get("PATHGRADE_ABLATE_FOLDS", "5"))
BAG = os.environ.get("PATHGRADE_ABLATE_BAG", "512")


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


cmd = [
    sys.executable, "-u", f"{SRC}/scripts/07_ablate.py",
    "--feature-dir", str(OUT),
    "--splits", str(WORK / "splits.json"),
    "--labels", str(WORK / "labels_available.csv"),
    "--out", str(TUNE),
    "--epochs", str(EPOCHS),
    "--folds", str(FOLDS),
    "--bag-size", str(BAG),
    "--arms", ARMS,
    "--workers", os.environ.get("PATHGRADE_ABLATE_WORKERS", "8"),
]
trail("STEP", f"arms={ARMS} epochs={EPOCHS} folds={FOLDS} bag={BAG}")

# As a child, so a crash cannot take the finished trials with it - the study
# database is the deliverable and it is written trial by trial.
rc = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": f"{SRC}/src"}).returncode
trail("ABLATE_RC", str(rc))

best = TUNE / "ablation.json"
if best.exists():
    print("\n" + json.dumps(json.loads(best.read_text()), indent=2), flush=True)
    trail("DONE", f"best written after {(time.time()-T0)/3600:.2f}h")
else:
    trail("NO_RESULT", "no ablation.json written")
print(f"\ntotal elapsed {(time.time() - T0) / 3600:.2f} h", flush=True)
