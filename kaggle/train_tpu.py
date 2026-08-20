"""Train ASMIL-Ord on the extracted H-optimus-0 embeddings.

Runs on the TPU VM but uses its **CPU**, not the XLA devices. That is a
deliberate choice, not a fallback: the model is ~530K parameters, while XLA
compiles a fresh graph per input shape and this training loop varies subspace
counts and crop counts by design. Recompilation would dominate. The TPU VM
happens to carry 224 vCPU and 406 GB RAM, which is far more than a model this
size needs, so the head trains in minutes with no compilation risk at all.

Produces a self-contained bundle in /kaggle/working/release: weights, metrics,
config, architecture, the exact source used, and a worked attention-map example.
"""
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

T0 = time.time()
WORK = Path("/kaggle/working")
RELEASE = WORK / "release"
RELEASE.mkdir(parents=True, exist_ok=True)


def find_dir(marker, root="/kaggle/input"):
    for depth in ("*", "*/*", "*/*/*", "*/*/*/*"):
        for hit in glob.glob(os.path.join(root, depth, marker)):
            return hit[: -len(marker) - 1] if marker else hit
    return None


SRC = find_dir("src/pathgrade/__init__.py")
if SRC is None:
    sys.exit("FATAL: pathgrade-src not mounted")
sys.path.insert(0, f"{SRC}/src")
print(f"source   : {SRC}")

# Features come from the extraction kernel's output, mounted as an input.
feature_dir = None
for candidate in glob.glob("/kaggle/input/*/features") + glob.glob("/kaggle/input/*/*/features"):
    if glob.glob(os.path.join(candidate, "*.h5")) or glob.glob(os.path.join(candidate, "*.pt")):
        feature_dir = candidate
        break
if feature_dir is None:
    hits = glob.glob("/kaggle/input/**/*.h5", recursive=True)
    if hits:
        feature_dir = str(Path(hits[0]).parent)
if feature_dir is None:
    sys.exit("FATAL: no extracted features found. Run pathgrade-extract-tpu first.")
print(f"features : {feature_dir}")

import multiprocessing

import numpy as np
import torch

torch.set_num_threads(min(32, multiprocessing.cpu_count()))
print(f"vCPU {multiprocessing.cpu_count()}, torch threads {torch.get_num_threads()}\n")

from pathgrade.config import Config
from pathgrade.data.io import verify_cohort
from pathgrade.data.splits import make_splits, read_labels
from pathgrade.evaluate import evaluate_run
from pathgrade.models.asmil_ord import ASMILOrd
from pathgrade.train import run_cv

# --------------------------------------------------------------- 1. cohort
print("=" * 68 + "\n1. COHORT\n" + "=" * 68)
labels_csv = f"{SRC}/tcga_hnsc_labels.csv"
labels = read_labels(labels_csv)
available = sorted(
    p.stem for p in Path(feature_dir).glob("*.h5")
) or sorted(p.stem for p in Path(feature_dir).glob("*.pt"))
usable = [p for p in available if p in labels]
print(f"{len(available)} extracted, {len(usable)} with labels")

info = verify_cohort(feature_dir, usable)
for k, v in info.items():
    if k != "missing":
        print(f"  {k:18s} {v}")

import collections

dist = collections.Counter(labels[p] for p in usable)
print(f"  class distribution  {dict(sorted(dist.items()))}  (G1/G2/G3)")

# Labels restricted to what actually extracted, so splits cannot reference
# a patient whose slide failed.
subset_csv = str(WORK / "labels_available.csv")
import csv

with open(subset_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["patient_id", "label"])
    for p in usable:
        w.writerow([p, labels[p]])

# ---------------------------------------------------------------- 2. splits
print("\n" + "=" * 68 + "\n2. SPLITS\n" + "=" * 68)
splits_path = str(WORK / "splits.json")
splits = make_splits(subset_csv, splits_path, test_frac=0.15, n_folds=5)
print(f"locked test : {len(splits.test)} patients (fingerprint {splits.fingerprint})")
print(f"cv folds    : {len(splits.folds)}")
for i, fold in enumerate(splits.folds):
    print(f"  fold {i}: {len(fold['train'])} train / {len(fold['val'])} val")

# --------------------------------------------------------------- 3. train
print("\n" + "=" * 68 + "\n3. TRAIN\n" + "=" * 68, flush=True)
cfg = Config()
cfg.run_name = "asmil-ord-hoptimus0"
cfg.output_dir = str(WORK / "runs")
cfg.encoder = "h-optimus-0"
cfg.data.feature_dir = feature_dir
cfg.data.labels_csv = subset_csv
cfg.data.splits_path = splits_path
cfg.optim.num_workers = 8
cfg.optim.amp = False              # CPU
summary = run_cv(cfg)

# ------------------------------------------------------------ 4. locked test
print("\n" + "=" * 68 + "\n4. LOCKED TEST SET\n" + "=" * 68, flush=True)
test_result = evaluate_run(cfg.run_dir, bootstrap=2000)

# ----------------------------------------------------- 5. attention example
print("\n" + "=" * 68 + "\n5. ATTENTION MAP (the product surface)\n" + "=" * 68)
from pathgrade.inference import GradePredictor, attention_to_grid

predictor = GradePredictor.from_run(cfg.run_dir, device=torch.device("cpu"))
demo_id = splits.test[0]
demo_path = Path(feature_dir) / f"{demo_id}.h5"
if not demo_path.exists():
    demo_path = Path(feature_dir) / f"{demo_id}.pt"

pred = predictor.predict_file(demo_path)
print(f"slide {demo_id} (true grade G{labels[demo_id] + 1})")
print(pred.summary())
print("\ntop 5 attended regions:")
for r in pred.top_regions(5):
    print(f"  #{r['rank']} ({r['x']:>7},{r['y']:>7})  attention {r['attention']:.5f}"
          f"  = {r['share_of_slide']:.1f}x average")

grid = attention_to_grid(pred)
np.savez_compressed(
    RELEASE / "attention_example.npz",
    patient_id=demo_id, attention=pred.attention, coords=pred.coords,
    grid=grid, probabilities=pred.probabilities, grade=pred.grade,
    true_grade=labels[demo_id], patch_size=pred.patch_size,
)
print(f"\nattention grid {grid.shape} saved")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap="inferno")
    ax.set_title(f"{demo_id}: predicted G{pred.grade + 1}, true G{labels[demo_id] + 1}")
    ax.axis("off")
    fig.colorbar(im, label="attention (share of slide embedding)")
    fig.savefig(RELEASE / "attention_example.png", dpi=130, bbox_inches="tight")
    print("attention_example.png written")
except Exception as e:
    print(f"(plot skipped: {e})")

# ------------------------------------------------------------- 6. release
print("\n" + "=" * 68 + "\n6. RELEASE BUNDLE\n" + "=" * 68)
model = ASMILOrd(
    feature_dim=cfg.model.feature_dim, n_classes=cfg.data.n_classes,
    window=cfg.model.window, stride=cfg.model.stride, hidden=cfg.model.hidden,
    n_branches=cfg.model.n_branches,
)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

for fold_dir in sorted(Path(cfg.run_dir).glob("fold*")):
    dest = RELEASE / fold_dir.name
    dest.mkdir(exist_ok=True)
    for name in ("checkpoint.pt", "history.json", "val_predictions.npz"):
        if (fold_dir / name).exists():
            shutil.copy(fold_dir / name, dest / name)

for name in ("config.json", "cv_summary.json", "test_results.json", "test_predictions.npz"):
    if (Path(cfg.run_dir) / name).exists():
        shutil.copy(Path(cfg.run_dir) / name, RELEASE / name)
shutil.copy(splits_path, RELEASE / "splits.json")
shutil.copy(subset_csv, RELEASE / "labels_available.csv")

# Archive the exact source that produced these weights.
shutil.make_archive(str(RELEASE / "source"), "zip", f"{SRC}/src")

manifest = {
    "encoder": "h-optimus-0",
    "encoder_licence": "Apache-2.0",
    "feature_dim": cfg.model.feature_dim,
    "architecture": "ASMIL-Ord",
    "trainable_params": trainable,
    "n_classes": cfg.data.n_classes,
    "class_names": cfg.data.class_names,
    "bag_size": cfg.data.bag_size,
    "subspace_ensemble": len(model.offsets),
    "cohort": {
        "n_extracted": len(available),
        "n_usable": len(usable),
        "class_distribution": {str(k): v for k, v in sorted(dist.items())},
        "median_patches": info.get("median_patches"),
        "total_patches": info.get("total_patches"),
    },
    "cv": summary,
    "test": test_result,
    "config": cfg.to_dict(),
    "elapsed_hours": round((time.time() - T0) / 3600, 3),
}
with open(RELEASE / "MANIFEST.json", "w") as f:
    json.dump(manifest, f, indent=2, default=str)

print(f"trainable params : {trainable:,}")
print(f"CV QWK           : {summary['qwk']['mean']:.4f} +/- {summary['qwk']['std']:.4f}")
print(f"TEST QWK         : {test_result['metrics']['qwk']:.4f} "
      f"CI {test_result['metrics']['qwk_ci95']}")
print(f"elapsed          : {manifest['elapsed_hours']:.2f} h")
print(f"\nrelease bundle: {sorted(p.name for p in RELEASE.iterdir())}")
