"""End-to-end: GDC -> H-optimus-0 embeddings -> trained model -> release bundle.

Extraction and training share one session on purpose. Kaggle allows a single
concurrent TPU session and each one queues for ~20 minutes, so splitting the
stages doubles the queue tax and forces the embeddings through a
publish-then-remount round trip for no benefit. Together they fit comfortably
inside the session cap: extraction is roughly two hours, training minutes.

Two failure properties matter here, because the expensive half runs first:

* **Extraction is idempotent.** Slides already present in the output are
  skipped, so a re-run resumes rather than restarting.
* **A training failure never destroys the extraction.** The training phase is
  guarded; if it raises, the traceback is printed, a FAILED marker is written,
  and the process still exits 0 so /kaggle/working - holding several GPU-hours
  of embeddings - is preserved as output. Check TRAINING_FAILED.txt before
  trusting a green run.

Output is flushed aggressively so `kaggle kernels logs` is useful while the run
is still going, instead of only after it ends.

RUN FROM THE KAGGLE UI. An API-pushed kernel does not inherit UI-attached
secrets, so HF_TOKEN would be invisible and the gated encoder would 401.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

T0 = time.time()
WORK = Path("/kaggle/working")
OUT = WORK / "features"
RELEASE = WORK / "release"
CACHE = Path("/kaggle/tmp/wsi")
for d in (OUT, RELEASE, CACHE):
    d.mkdir(parents=True, exist_ok=True)


def banner(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}", flush=True)


def find_src(marker="src/pathgrade/__init__.py", root="/kaggle/input"):
    for depth in ("*", "*/*", "*/*/*"):
        for hit in glob.glob(os.path.join(root, depth, marker)):
            return hit[: -len(marker) - 1]
    return None


# ------------------------------------------------------------ 0. PREFLIGHT
banner("0. PREFLIGHT")
MAX_PATCHES = int(os.environ.get("PATHGRADE_MAX_PATCHES", "3000"))
BATCH_SIZE = os.environ.get("PATHGRADE_BATCH", "256")
DECODE_WORKERS = os.environ.get("PATHGRADE_DECODE_WORKERS", "16")
SLIDE_LIMIT = os.environ.get("PATHGRADE_LIMIT")
RANDOM_WEIGHTS = os.environ.get("PATHGRADE_RANDOM_WEIGHTS") == "1"

SRC = find_src()
if SRC is None:
    sys.exit("FATAL: pathgrade-src dataset not mounted")
sys.path.insert(0, f"{SRC}/src")
print(f"source: {SRC}", flush=True)


def load_token():
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return f"env:{var}"
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            try:
                value = client.get_secret(key)
                if value:
                    os.environ["HF_TOKEN"] = value
                    os.environ["HUGGING_FACE_HUB_TOKEN"] = value
                    return f"kaggle-secret:{key}"
            except Exception as e:
                last = e
    except Exception as e:
        last = e
    for path in glob.glob("/kaggle/input/**/hf_token.txt", recursive=True):
        value = open(path).read().strip()
        if value:
            os.environ["HF_TOKEN"] = value
            os.environ["HUGGING_FACE_HUB_TOKEN"] = value
            return f"file:{path}"
    return None


token_source = load_token()
print(f"HF token: {token_source or 'NOT FOUND'}", flush=True)
if RANDOM_WEIGHTS and token_source is None:
    # Rehearsal mode never downloads weights, so a token would be pointless.
    print("REHEARSAL: no token needed (random weights)", flush=True)
elif token_source is None:
    sys.exit(
        "FATAL: no HF token visible.\n"
        "  1. accept terms at https://huggingface.co/bioptimus/H-optimus-0\n"
        "  2. create a read token at https://huggingface.co/settings/tokens\n"
        "  3. in THIS notebook: Add-ons > Secrets > add HF_TOKEN and tick it\n"
        "  4. Save & Run All from the UI - an API push cannot see secrets"
    )

print("installing openslide + timm ...", flush=True)
subprocess.run("pip install -q openslide-bin openslide-python timm 2>&1 | tail -2",
               shell=True, check=False)

if not RANDOM_WEIGHTS:
    from huggingface_hub import hf_hub_download

    cfg_file = hf_hub_download(
        "bioptimus/H-optimus-0", "config.json", token=os.environ["HF_TOKEN"]
    )
    print(f"H-optimus-0 reachable ({os.path.getsize(cfg_file)} B)", flush=True)

import multiprocessing

import numpy as np
import torch

print(f"vCPU {multiprocessing.cpu_count()}")
for path in ("/kaggle/working", "/kaggle/tmp"):
    if os.path.isdir(path):
        print(f"{path:16s} {shutil.disk_usage(path)[2] / 1e9:7.1f} GB free")
ACCEL = None
try:
    import torch_xla.core.xla_model as xm
    print(f"XLA: {xm.xla_device()}", flush=True)
    ACCEL = "xla"
except Exception as e:
    print(f"no XLA ({e})", flush=True)
    if torch.cuda.is_available():
        ACCEL = "cuda"
        print(f"CUDA: {torch.cuda.get_device_name(0)}", flush=True)

# Refuse to encode on a bare CPU session. H-optimus-0 is a 1B-parameter ViT-g;
# it runs at ~0.4 patches/s on CPU, so 435 slides would take several hundred
# hours. Far better to fail in ten seconds than to look busy for a whole
# session and produce nothing.
if ACCEL is None:
    sys.exit(
        "FATAL: no accelerator. This notebook is set to CPU. "
        "Settings > Accelerator > TPU VM, then Save & Run All. "
        "Encoding on CPU would take ~400 hours."
    )

# Settings measured on a real TPU session, not guessed. xla:0 is ONE of eight
# cores, and every batch pays a synchronous device-to-host transfer whose cost
# is latency rather than bandwidth, so larger batches barely help
# (64:114/s, 256:124/s, 512:114/s). At 124 patches/s, 3000 tiles per slide puts
# the cohort near 2.9 h - inside one session, where 6000 would be 5.8 h.

# ----------------------------------------------------------- 1. EXTRACTION
banner("1. EXTRACTION  (GDC -> H-optimus-0 embeddings)")
already = len(list(OUT.glob("*.h5")))
print(f"{already} slides already extracted (these are skipped)", flush=True)

from pathgrade.preprocessing.stream_extract import main as extract_main

extract_argv = [
    "--out-dir", str(OUT),
    "--cache-dir", str(CACHE),
    "--labels-csv", f"{SRC}/tcga_hnsc_labels.csv",
    "--encoder", "h-optimus-0",
    "--device", ACCEL,
    "--format", "h5",
    "--max-patches", str(MAX_PATCHES),
    "--batch-size", BATCH_SIZE,
    "--download-workers", "4",
    "--prefetch", "4",
    "--decode-workers", DECODE_WORKERS,
    "--max-hours", "6.5",
    "--min-free-gb", "4",
    "--notify-every", "25",
]
if SLIDE_LIMIT:
    extract_argv += ["--limit", SLIDE_LIMIT]
if RANDOM_WEIGHTS:
    # Dress-rehearsal mode: exercises extraction AND training end to end without
    # a gated fetch. Embeddings are noise, so any metric produced is meaningless
    # by construction - the point is to prove the plumbing, not the model.
    extract_argv += ["--random-weights"]
    print("!! REHEARSAL: random weights. Metrics below are meaningless.", flush=True)
print("argv:", " ".join(extract_argv), flush=True)
extract_code = extract_main(extract_argv)
extracted = sorted(p.stem for p in OUT.glob("*.h5"))
print(f"\nextraction finished: {len(extracted)} slides, exit={extract_code}", flush=True)

if not extracted:
    sys.exit("FATAL: no features extracted, nothing to train on")

# ------------------------------------------------------------- 2. TRAINING
banner("2. TRAINING  (TPU VM cpu - the head is ~530K params)")
try:
    import collections
    import csv

    torch.set_num_threads(min(32, multiprocessing.cpu_count()))

    from pathgrade.config import Config
    from pathgrade.data.io import verify_cohort
    from pathgrade.data.splits import make_splits, read_labels
    from pathgrade.evaluate import evaluate_run
    from pathgrade.inference import GradePredictor, attention_to_grid
    from pathgrade.models.asmil_ord import ASMILOrd
    from pathgrade.train import run_cv

    labels = read_labels(f"{SRC}/tcga_hnsc_labels.csv")
    usable = [p for p in extracted if p in labels]
    info = verify_cohort(str(OUT), usable)
    dist = collections.Counter(labels[p] for p in usable)
    print(f"{len(usable)} usable slides | classes {dict(sorted(dist.items()))}")
    for k, v in info.items():
        if k != "missing":
            print(f"  {k:18s} {v}")

    subset_csv = WORK / "labels_available.csv"
    with open(subset_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "label"])
        for p in usable:
            w.writerow([p, labels[p]])

    splits_path = WORK / "splits.json"
    splits = make_splits(str(subset_csv), str(splits_path), test_frac=0.15, n_folds=5)
    print(f"\nlocked test {len(splits.test)} (fingerprint {splits.fingerprint}), "
          f"{len(splits.folds)} folds", flush=True)

    cfg = Config()
    cfg.run_name = "asmil-ord-hoptimus0"
    cfg.output_dir = str(WORK / "runs")
    cfg.encoder = "h-optimus-0"
    cfg.data.feature_dir = str(OUT)
    cfg.data.labels_csv = str(subset_csv)
    cfg.data.splits_path = str(splits_path)
    cfg.optim.num_workers = 8
    cfg.optim.amp = False
    summary = run_cv(cfg)

    banner("3. LOCKED TEST SET")
    test_result = evaluate_run(cfg.run_dir, bootstrap=2000)

    banner("4. ATTENTION MAP  (the product surface)")
    predictor = GradePredictor.from_run(cfg.run_dir, device=torch.device("cpu"))
    demo_id = splits.test[0]
    pred = predictor.predict_file(OUT / f"{demo_id}.h5")
    print(f"slide {demo_id} (true G{labels[demo_id] + 1})")
    print(pred.summary())
    for r in pred.top_regions(5):
        print(f"  #{r['rank']} ({r['x']:>7},{r['y']:>7}) attention {r['attention']:.5f}"
              f"  {r['share_of_slide']:.1f}x average")

    grid = attention_to_grid(pred)
    np.savez_compressed(
        RELEASE / "attention_example.npz",
        patient_id=demo_id, attention=pred.attention, coords=pred.coords, grid=grid,
        probabilities=pred.probabilities, grade=pred.grade, true_grade=labels[demo_id],
        patch_size=pred.patch_size,
    )
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

    banner("5. RELEASE BUNDLE")
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
    shutil.make_archive(str(RELEASE / "source"), "zip", f"{SRC}/src")

    manifest = {
        "encoder": "h-optimus-0", "encoder_licence": "Apache-2.0",
        "architecture": "ASMIL-Ord", "trainable_params": trainable,
        "feature_dim": cfg.model.feature_dim, "max_patches_per_slide": MAX_PATCHES,
        "subspace_ensemble": len(model.offsets),
        "class_names": cfg.data.class_names,
        "cohort": {
            "n_extracted": len(extracted), "n_usable": len(usable),
            "class_distribution": {str(k): v for k, v in sorted(dist.items())},
            "median_patches": info.get("median_patches"),
            "total_patches": info.get("total_patches"),
        },
        "cv": summary, "test": test_result, "config": cfg.to_dict(),
        "elapsed_hours": round((time.time() - T0) / 3600, 3),
    }
    with open(RELEASE / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"trainable params : {trainable:,}")
    print(f"CV QWK           : {summary['qwk']['mean']:.4f} +/- {summary['qwk']['std']:.4f}")
    print(f"TEST QWK         : {test_result['metrics']['qwk']:.4f} "
          f"CI {test_result['metrics']['qwk_ci95']}")
    print(f"release          : {sorted(p.name for p in RELEASE.iterdir())}", flush=True)

except Exception:
    # The embeddings above cost hours; never let a training bug discard them.
    banner("TRAINING FAILED - embeddings preserved")
    traceback.print_exc()
    with open(WORK / "TRAINING_FAILED.txt", "w") as f:
        f.write(traceback.format_exc())
    print("\nFeatures in /kaggle/working/features are intact. Re-run to resume:")
    print("extraction skips slides already present, so only training repeats.")

print(f"\ntotal elapsed {(time.time() - T0) / 3600:.2f} h", flush=True)
