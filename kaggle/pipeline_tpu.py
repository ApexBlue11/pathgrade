"""End-to-end: GDC -> H-optimus-0 embeddings -> trained model -> release bundle.

Extraction and training live in one script so a single kernel can do either or
both. In practice extraction runs as a chain of short, bounded kernels rather
than one long session - two full-cohort attempts run to completion were killed
by the platform and lost all their output, while short bounded runs committed
reliably. See ``../docs/ENGINEERING.md`` for why, and ``../kaggle/README.md``
for how the chain is launched.

Two failure properties matter, because the expensive half runs first:

* **Extraction is idempotent and resumable.** Slides already present in the
  output are skipped, and each kernel seeds its output from the previous
  kernel's (mounted via ``kernel_sources``), so the chain accumulates the
  cohort rather than restarting it.
* **A training failure never destroys the extraction.** The training phase is
  guarded; if it raises, the traceback is printed, a FAILED marker is written,
  and the process still exits 0 so /kaggle/working - holding hours of
  embeddings - is preserved as output. Check TRAINING_FAILED.txt before
  trusting a green run. ``PATHGRADE_SKIP_TRAIN=1`` skips straight past this
  stage for an intermediate chunk that should only bank slides.

Output is flushed aggressively so `kaggle kernels logs` is useful while the run
is still going, instead of only after it ends.

The HF token arrives as a file in the attached ``apexblue/pathgrade-token``
dataset, not as a UI secret. That matters: an API-pushed kernel is not given
the secrets service at all, so a UI-attached ``HF_TOKEN`` is invisible to it
and the gated encoder 401s. Reading the token from a private dataset makes the
whole run launchable with ``kaggle kernels push``, with no human in the loop.
The UI secret still works and is still tried, just second.
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


# Every diagnostic kernel in this repo writes an fsync'd trail, and every one of
# them was debuggable. This kernel only printed, and it is the single one whose
# failures could never be explained. Kaggle's log endpoint has returned empty
# even for runs that finished successfully, so stdout is not a record. The trail
# starts at line one, so any future failure names the last step that began.
TRAIL = WORK / "pipeline_trail.txt"
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


def banner(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}", flush=True)
    trail("STAGE", text)


trail("BOOT", f"python {sys.version.split()[0]}")


def find_src(marker="src/pathgrade/__init__.py", root="/kaggle/input"):
    for depth in ("*", "*/*", "*/*/*"):
        for hit in glob.glob(os.path.join(root, depth, marker)):
            return hit[: -len(marker) - 1]
    return None


# ------------------------------------------------------------ 0. PREFLIGHT
banner("0. PREFLIGHT")
MAX_PATCHES = int(os.environ.get("PATHGRADE_MAX_PATCHES", "3000"))
BATCH_SIZE = os.environ.get("PATHGRADE_BATCH", "256")
# 16 is the proven value. Decode was never the bottleneck: real slides run
# 1900-2500 tiles/s against an encoder that consumes 123/s on one device, and
# oversubscription measurably hurts. Raise this only alongside real
# multi-device encoding.
DECODE_WORKERS = os.environ.get("PATHGRADE_DECODE_WORKERS", "16")
# XLA devices to encode across, via threads. MEASURED BROKEN above 2 (7 of 8
# threads die in SyncLiveTensorsGraph, and the survivor is slower than one
# device alone). Multi-process, one device each, was also tried and did not
# work on this platform - see docs/ENGINEERING.md for the full record. Single
# device is the proven path.
TPU_CORES = os.environ.get("PATHGRADE_TPU_CORES", "1")
# MEASURED, and the correction matters: download is the binding constraint, and
# 4 was starving it. Across chunks c1/c5/c6 per-stream throughput was a steady
# 30 MB/s, but the AGGREGATE over wall clock was only 14-17 MB/s - less than a
# single stream. The cause is structural, not bandwidth: the prefetcher's
# semaphore counts slides on disk, so after the opening burst of 4 a new
# download cannot start until the consumer finishes a slide and frees a slot.
# Downloads and encoding therefore barely overlap, which is why the 143 MB/s
# measured at 4 parallel streams never appeared in the real pipeline.
#
# Raising --download-workers alone does nothing; --prefetch is the semaphore and
# has to rise with it. Scratch is free: 16 slides of c6's unusually large 1.6 GB
# mean is 26 GB of the 1098 GB on /kaggle/tmp.
DOWNLOAD_WORKERS = os.environ.get("PATHGRADE_DOWNLOAD_WORKERS", "16")
PREFETCH = os.environ.get("PATHGRADE_PREFETCH", "16")
# Wall-clock budget for extraction. Deliberately short by default: the two
# runs that tried to do the whole cohort in one session were killed outright
# (86 min, 3 h 45) and lost everything, while a 34-minute run committed fine.
# Stopping cleanly well inside that window is what makes progress durable, and
# the seeding step above makes the next run continue where this one stopped.
MAX_EXTRACT_HOURS = os.environ.get("PATHGRADE_MAX_EXTRACT_HOURS", "0.85")
SKIP_TRAIN = os.environ.get("PATHGRADE_SKIP_TRAIN") == "1"
SLIDE_LIMIT = os.environ.get("PATHGRADE_LIMIT")
RANDOM_WEIGHTS = os.environ.get("PATHGRADE_RANDOM_WEIGHTS") == "1"
# Token pooling inside each 224px tile. Unset keeps the registry default
# ("cls"), which is the convention pathology foundation models are
# benchmarked under. "cls_mean" concatenates the patch-token mean, which is
# what Bioptimus recommend for downstream use, and doubles the width to
# 3072. The first 1536 columns are the CLS vector either way, so a
# cls_mean run can be compared against existing cls features on the same
# slides without re-encoding anything.
POOLING = os.environ.get("PATHGRADE_POOLING")

SRC = find_src()
if SRC is None:
    trail("FATAL", "pathgrade-src dataset not mounted")
    sys.exit("FATAL: pathgrade-src dataset not mounted")
sys.path.insert(0, f"{SRC}/src")
print(f"source: {SRC}", flush=True)


def _adopt(value: str, source: str) -> str:
    os.environ["HF_TOKEN"] = value
    os.environ["HUGGING_FACE_HUB_TOKEN"] = value
    return source


def load_token(why: dict):
    """Find an HF token, recording *why* each route failed.

    Route order is deliberate. The attached-dataset file comes before Kaggle
    secrets because it is the only route that works for an API-pushed kernel -
    the secrets service is not provisioned for one at all, and asking it first
    just buys a slow ConnectionError on the path we now use by default.

    ``why`` is filled in rather than discarded. The previous version bound the
    exception to a local named ``last`` and never read it, so the one run that
    failed here could not be diagnosed from its own output.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return f"env:{var}"

    hits = glob.glob("/kaggle/input/**/hf_token.txt", recursive=True)
    why["files_seen"] = hits
    for path in hits:
        try:
            value = open(path).read().strip()
            if value:
                return _adopt(value, f"file:{path}")
            why[f"file:{path}"] = "empty"
        except OSError as e:
            why[f"file:{path}"] = f"{type(e).__name__}: {e}"

    try:
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            try:
                value = client.get_secret(key)
                if value:
                    return _adopt(value, f"kaggle-secret:{key}")
            except Exception as e:
                why[f"secret:{key}"] = f"{type(e).__name__}: {e}"
    except Exception as e:
        why["secrets_import"] = f"{type(e).__name__}: {e}"
    return None


token_why: dict = {}
token_source = load_token(token_why)
print(f"HF token: {token_source or 'NOT FOUND'}", flush=True)
trail("TOKEN", token_source or f"NOT FOUND {json.dumps(token_why)[:300]}")
if RANDOM_WEIGHTS and token_source is None:
    # Rehearsal mode never downloads weights, so a token would be pointless.
    print("REHEARSAL: no token needed (random weights)", flush=True)
elif token_source is None:
    for k, v in token_why.items():
        print(f"  {k}: {v}", file=sys.stderr)
    trail("FATAL", "no HF token by any route")
    sys.exit(
        "FATAL: no HF token visible.\n"
        "  1. accept terms at https://huggingface.co/bioptimus/H-optimus-0\n"
        "  2. attach the apexblue/pathgrade-token dataset (contains hf_token.txt),\n"
        "     which is the route that works for an API-pushed kernel\n"
        "  3. or, for a UI run: Add-ons > Secrets > add HF_TOKEN and tick it"
    )

print("installing openslide + timm ...", flush=True)
subprocess.run("pip install -q openslide-bin openslide-python timm 2>&1 | tail -2",
               shell=True, check=False)

if not RANDOM_WEIGHTS:
    # Cheap gate on the gated repo: a 401 here costs seconds, whereas finding
    # out at the first slide costs the download that preceded it. This was the
    # one preflight check that could fail without leaving a trail entry.
    from huggingface_hub import hf_hub_download

    trail("STEP", "probing gated repo bioptimus/H-optimus-0")
    try:
        cfg_file = hf_hub_download(
            "bioptimus/H-optimus-0", "config.json", token=os.environ["HF_TOKEN"]
        )
    except Exception as e:
        trail("FATAL", f"H-optimus-0 unreachable: {type(e).__name__}: {e}")
        sys.exit(
            f"FATAL: cannot reach bioptimus/H-optimus-0 ({type(e).__name__}: {e}).\n"
            "  The token was found, so this is almost certainly the gate: accept the\n"
            "  terms at https://huggingface.co/bioptimus/H-optimus-0 with the same\n"
            "  account that issued the token."
        )
    print(f"H-optimus-0 reachable ({os.path.getsize(cfg_file)} B)", flush=True)
    trail("OK", "H-optimus-0 reachable")

import multiprocessing

import numpy as np
import torch

print(f"vCPU {multiprocessing.cpu_count()}")
for path in ("/kaggle/working", "/kaggle/tmp"):
    if os.path.isdir(path):
        print(f"{path:16s} {shutil.disk_usage(path)[2] / 1e9:7.1f} GB free")
# The accelerator probe runs in a THROWAWAY subprocess, and that detail is
# load-bearing. Creating an XLA device claims the TPU chips for the lifetime of
# the process that created them, so a parent that calls xm.xla_device() makes it
# impossible for a child to acquire the TPU afterwards. Extraction is run as a
# child (see stage 1), so the parent must never touch a device itself - it only
# asks a short-lived process whether one exists, and that process frees the
# chips when it exits.
ACCEL = None
_probe = subprocess.run(
    [sys.executable, "-c",
     "import torch_xla.core.xla_model as xm; print(xm.xla_device())"],
    capture_output=True, text=True, timeout=600,
)
if _probe.returncode == 0 and "xla" in _probe.stdout:
    ACCEL = "xla"
    print(f"XLA: {_probe.stdout.strip()} (probed out-of-process)", flush=True)
else:
    print(f"no XLA (rc={_probe.returncode}) {_probe.stderr.strip()[-300:]}", flush=True)
    if torch.cuda.is_available():
        ACCEL = "cuda"
        print(f"CUDA: {torch.cuda.get_device_name(0)}", flush=True)
trail("ACCEL", str(ACCEL))

# Refuse to encode on a bare CPU session. H-optimus-0 is a 1B-parameter ViT-g;
# it runs at ~0.4 patches/s on CPU, so 435 slides would take several hundred
# hours. Far better to fail in ten seconds than to look busy for a whole
# session and produce nothing.
if ACCEL is None:
    trail("FATAL", "no accelerator - notebook is set to CPU")
    sys.exit(
        "FATAL: no accelerator. This notebook is set to CPU. "
        "Settings > Accelerator > TPU VM, then Save & Run All. "
        "Encoding on CPU would take ~400 hours."
    )

# Settings measured on a real TPU session, not guessed. Every batch pays a
# synchronous device-to-host transfer whose cost is latency rather than
# bandwidth, so larger batches barely help (64:114/s, 256:124/s, 512:114/s).
#
# 124 patches/s is ONE device. A v5e-8 has eight, and --tpu-cores now spreads
# replicas across all of them; the encoder is frozen, so the replicas are
# independent and need no collectives. If replication fails at runtime,
# build_encoders falls back to however many it managed, down to one - the
# path validated on 60 real slides.

# ----------------------------------------------------------- 1. EXTRACTION
banner("1. EXTRACTION  (GDC -> H-optimus-0 embeddings)")

# Seed from any earlier run mounted as a kernel source or dataset.
#
# Two full-cohort attempts died mid-run and produced NO output at all - not the
# embeddings, not the log, not the trail fsync'd in the first second. The whole
# container went, at 86 minutes and again at 3 h 45. A 34-minute run finished
# cleanly. Nothing inside the container can explain a failure that discards the
# container, so rather than keep paying hours to learn nothing, runs are kept
# short and made to accumulate: each one inherits every slide its predecessors
# encoded, adds what it can inside PATHGRADE_MAX_EXTRACT_HOURS, and exits
# cleanly so Kaggle actually commits the result.
#
# Extraction was always idempotent - it skips slides already in out-dir - but
# that only helps if the previous output survived to be mounted. This is the
# missing half of that property.
#
# The glob is deliberately broad - it will take features from any mounted
# input - so it has to check what it is taking. Feature files record the token
# pooling that produced them, and cls and cls_mean differ in width (1536 vs
# 3072). Seeding one into a run configured for the other would build a feature
# directory of mixed widths, which fails late and confusingly if at all. Files
# written before that attribute existed are cls by definition.
WANT_POOLING = POOLING or "cls"


def _pooling_of(path):
    try:
        import h5py

        with h5py.File(path, "r") as f:
            got = f.attrs.get("pooling", "cls")
        return got.decode() if isinstance(got, bytes) else str(got)
    except Exception as e:                       # unreadable is not seedable
        return f"<unreadable: {type(e).__name__}>"


seeded = skipped = 0
for src_h5 in glob.glob("/kaggle/input/**/features/*.h5", recursive=True):
    dest = OUT / os.path.basename(src_h5)
    if dest.exists():
        continue
    got = _pooling_of(src_h5)
    if got != WANT_POOLING:
        if skipped < 5:
            print(f"  skipping {os.path.basename(src_h5)}: pooling={got!r}, "
                  f"this run wants {WANT_POOLING!r}", flush=True)
        skipped += 1
        continue
    try:
        shutil.copy(src_h5, dest)
        seeded += 1
    except OSError as e:
        print(f"  could not seed {os.path.basename(src_h5)}: {e}", flush=True)
if skipped:
    print(f"  seeding skipped {skipped} file(s) with the wrong pooling", flush=True)
    trail("SEED_SKIPPED", f"{skipped} wrong-pooling files")
if seeded:
    print(f"seeded {seeded} slides from previous runs", flush=True)
trail("SEEDED", f"{seeded} slides inherited")

already = len(list(OUT.glob("*.h5")))
print(f"{already} slides already extracted (these are skipped)", flush=True)

extract_argv = [
    "--out-dir", str(OUT),
    "--cache-dir", str(CACHE),
    "--labels-csv", f"{SRC}/tcga_hnsc_labels.csv",
    "--encoder", "h-optimus-0",
    "--device", ACCEL,
    "--format", "h5",
    "--max-patches", str(MAX_PATCHES),
    "--batch-size", BATCH_SIZE,
    "--download-workers", DOWNLOAD_WORKERS,
    "--prefetch", PREFETCH,
    "--decode-workers", DECODE_WORKERS,
    "--tpu-cores", TPU_CORES,
    "--max-hours", MAX_EXTRACT_HOURS,
    "--min-free-gb", "4",
    "--notify-every", "25",
]
if POOLING:
    extract_argv += ["--pooling", POOLING]
if SLIDE_LIMIT:
    extract_argv += ["--limit", SLIDE_LIMIT]
if RANDOM_WEIGHTS:
    # Dress-rehearsal mode: exercises extraction AND training end to end without
    # a gated fetch. Embeddings are noise, so any metric produced is meaningless
    # by construction - the point is to prove the plumbing, not the model.
    extract_argv += ["--random-weights"]
    print("!! REHEARSAL: random weights. Metrics below are meaningless.", flush=True)

# Progress needs a channel that does NOT live in /kaggle/working. On
# 2026-08-22 a run executed for 86 minutes, errored, and produced no output at
# all - not the embeddings, not the log, not even the fsync'd trail written in
# its first second. The whole volume went. A trail file cannot describe a
# failure that discards the trail, so anything worth knowing during a long run
# has to leave the machine while it is still running.
#
# ProgressReporter already speaks Discord/Slack/Telegram and swallows its own
# errors, so a dead webhook can never hurt the run. Supply it the same way as
# the token: a file in an attached private dataset, or a baked env var.
def find_webhook():
    if os.environ.get("PATHGRADE_WEBHOOK"):
        return os.environ["PATHGRADE_WEBHOOK"], "env"
    for path in glob.glob("/kaggle/input/**/webhook_url.txt", recursive=True):
        try:
            value = open(path).read().strip()
            if value.startswith("http"):
                return value, f"file:{path}"
        except OSError:
            pass
    return None, None


WEBHOOK, webhook_src = find_webhook()
# Never print the URL: a Discord/Slack hook URL is itself the credential.
trail("WEBHOOK", webhook_src or "none (long runs will be unobservable if output is lost)")
if WEBHOOK:
    extract_argv += ["--webhook-url", WEBHOOK, "--notify-every", "10"]

print("argv:", " ".join(a for a in extract_argv if not a.startswith("http")), flush=True)

# Extraction gets the same guarantee training has always had, and for the same
# reason: it is the expensive half. An unhandled exception at slide 400 used to
# kill the kernel outright, discarding both the 399 slides already encoded and
# any chance of training on them. Whatever landed on disk is worth keeping and
# worth training on - extraction is idempotent, so a later run resumes.
# Extraction runs OUT OF PROCESS, and this is the most important robustness
# property in the kernel.
#
# On 2026-08-22 a 435-slide run executed for 86 minutes, errored, and produced
# no output whatsoever - not the embeddings, not the log, not even the trail
# file fsync'd in its first second. A Python-level guard cannot catch that: a
# segfault in a C extension (OpenSlide and libtiff both parse untrusted TIFFs
# from 435 different scanners) kills the interpreter outright, and a container
# torn down that way never reaches Kaggle's output-commit step.
#
# As a child, the worst it can do is die. The parent stays alive, keeps every
# slide already written, records how the child died - including the signal, so
# a segfault is distinguishable from an exception - and still trains and
# commits output. stdout is inherited, so per-slide progress still streams.
child_env = {**os.environ, "PYTHONPATH": f"{SRC}/src"}

# Run as a child so a crash inside a C extension - OpenSlide and libtiff parse
# untrusted TIFFs from hundreds of different scanners - can only take down the
# child. Two full-cohort attempts run in-process were killed by the platform
# outright (86 min, then 3 h 45) and produced NO output at all, not even the
# trail file fsync'd in the first second; a bounded, child-isolated chunk of
# under an hour is what actually committed. See docs/ENGINEERING.md.
extract_code, extract_error = None, None
cmd = [sys.executable, "-u", "-m", "pathgrade.preprocessing.stream_extract"] + extract_argv
trail("STEP", f"extraction, budget {MAX_EXTRACT_HOURS}h")
try:
    extract_code = subprocess.run(cmd, env=child_env).returncode
    if extract_code is not None and extract_code < 0:
        # Negative means killed by a signal: -11 SIGSEGV, -9 SIGKILL (the OOM
        # killer). Resolving the name is best-effort - not every signal number
        # is in the enum on every platform, and raising here, inside the
        # handler for a crash, would lose the information we came for.
        try:
            import signal as _sig

            name = _sig.Signals(-extract_code).name
        except (ValueError, ImportError):
            name = "unknown signal"
        extract_error = f"extraction child killed by {name} ({extract_code})"
    elif extract_code not in (0, 1):
        # 1 is the documented "finished, but some slides failed" return.
        extract_error = f"extraction child exited {extract_code}"
except BaseException:
    extract_error = traceback.format_exc()

if extract_error:
    trail("EXTRACTION_FAILED", extract_error.strip().splitlines()[-1][:200])
    banner("EXTRACTION FAILED - keeping what was encoded")
    print(extract_error, flush=True)
    with open(WORK / "EXTRACTION_FAILED.txt", "w") as f:
        f.write(extract_error)
        f.flush()
        os.fsync(f.fileno())

extracted = sorted(p.stem for p in OUT.glob("*.h5"))
print(f"\nextraction finished: {len(extracted)} slides, exit={extract_code}"
      f"{' (RAISED)' if extract_error else ''}", flush=True)
trail("EXTRACTED", f"{len(extracted)} slides")

if not extracted:
    trail("FATAL", "no features extracted, nothing to train on")
    sys.exit("FATAL: no features extracted, nothing to train on")

# An intermediate chunk banks its embeddings and stops. Training on a partial
# cohort is not just wasted minutes - it is minutes spent inside a container
# that has twice been killed without committing anything, which would take the
# slides this run just paid for down with it. Bank first, train once at the end.
if SKIP_TRAIN:
    trail("SKIP_TRAIN", f"{len(extracted)} slides banked, exiting before training")
    banner(f"TRAINING SKIPPED - {len(extracted)} slides banked as output")
    print("")
    print(f"total elapsed {(time.time() - T0) / 3600:.2f} h", flush=True)
    raise SystemExit(0)

# ------------------------------------------------------------- 2. TRAINING
banner("2. TRAINING  (TPU VM cpu - the head is ~530K params)")
try:
    import collections
    import csv

    # UNVERIFIED tuning: the box has 224 vCPU and training is only ~5% of total
    # wall clock, so this is not where "fastest overall" is won. A 530K-param
    # head can also lose to thread-sync overhead, which is why this is raised
    # to 64 rather than 224.
    torch.set_num_threads(min(64, multiprocessing.cpu_count()))

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
    cfg.optim.num_workers = 16     # features are RAM-cached; this is cheap
    cfg.optim.amp = False

    # Env-overridable knobs, for testing the fix to the attention collapse
    # without editing and re-publishing the library. The first real run left
    # attention exactly uniform because the head reached ~0 training loss from
    # the bag mean alone, so nothing ever asked the scorer to specialise; these
    # are the levers on that. Defaults reproduce that original run exactly.
    def _envf(name, default):
        v = os.environ.get(name)
        return default if v is None else float(v)

    cfg.loss.lambda_attn_entropy = _envf("PATHGRADE_ATTN_ENTROPY", cfg.loss.lambda_attn_entropy)
    cfg.optim.scorer_no_decay = os.environ.get("PATHGRADE_SCORER_NO_DECAY") == "1"
    cfg.optim.weight_decay = _envf("PATHGRADE_WEIGHT_DECAY", cfg.optim.weight_decay)
    cfg.model.dropout = _envf("PATHGRADE_DROPOUT", cfg.model.dropout)
    if os.environ.get("PATHGRADE_SAMPLES_PER_SLIDE"):
        # Multiplies training samples per epoch. The first run saw ~348 samples
        # per fold and drove training loss to ~0; each slide holds several
        # disjoint sub-bags that are legitimately different views of one label.
        cfg.data.samples_per_slide = int(os.environ["PATHGRADE_SAMPLES_PER_SLIDE"])
    if os.environ.get("PATHGRADE_BAG_SIZE"):
        # Smaller bags make the bag mean a noisier estimate, which is the
        # structural reason attention had nothing to add: at 1536 of ~3000
        # patches, any two subsets have nearly the same mean.
        cfg.data.bag_size = int(os.environ["PATHGRADE_BAG_SIZE"])
    print(f"training knobs: attn_entropy={cfg.loss.lambda_attn_entropy} "
          f"weight_decay={cfg.optim.weight_decay} dropout={cfg.model.dropout} "
          f"bag_size={cfg.data.bag_size} "
          f"samples_per_slide={cfg.data.samples_per_slide} "
          f"scorer_no_decay={cfg.optim.scorer_no_decay}", flush=True)
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

except BaseException:
    # The embeddings above cost hours; never let a training bug discard them.
    trail("TRAINING_FAILED", "embeddings preserved")
    banner("TRAINING FAILED - embeddings preserved")
    traceback.print_exc()
    with open(WORK / "TRAINING_FAILED.txt", "w") as f:
        f.write(traceback.format_exc())
    print("\nFeatures in /kaggle/working/features are intact. Re-run to resume:")
    print("extraction skips slides already present, so only training repeats.")

print(f"\ntotal elapsed {(time.time() - T0) / 3600:.2f} h", flush=True)
