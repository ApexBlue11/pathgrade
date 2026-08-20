"""TCGA-HNSC feature extraction on a Kaggle TPU VM.

Streams every labelled slide from GDC, encodes it with H-optimus-0 on the TPU,
writes the embeddings to /kaggle/working, and deletes the slide. Nothing but
the embeddings survives the session.

RUN THIS FROM THE KAGGLE UI, NOT VIA API PUSH.
An API-pushed kernel does not inherit UI-attached secrets - the secrets service
is not even provisioned for it - so HF_TOKEN will be invisible and H-optimus-0,
which is gated, will 401. Open the notebook, attach the secret under
Add-ons > Secrets, then Save & Run All.
"""
import glob
import os
import subprocess
import sys
import time

T0 = time.time()

# --------------------------------------------------------------- source tree
def find_src(marker="src/pathgrade/__init__.py", root="/kaggle/input"):
    """Kaggle mounts datasets differently across images; search rather than guess."""
    for depth in ("*", "*/*", "*/*/*"):
        for hit in glob.glob(os.path.join(root, depth, marker)):
            return hit[: -len(marker) - 1]
    return None


SRC = find_src()
if SRC is None:
    sys.exit("FATAL: pathgrade-src dataset not mounted. Attach apexblue/pathgrade-src.")
sys.path.insert(0, f"{SRC}/src")
print(f"source: {SRC}")

# ---------------------------------------------------------------- HF token
def load_token():
    """Token from env, Kaggle secret, or a mounted file - whichever is present."""
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
            except Exception:
                continue
    except Exception:
        pass
    for path in glob.glob("/kaggle/input/*/hf_token.txt") + glob.glob("/kaggle/input/*/*/hf_token.txt"):
        value = open(path).read().strip()
        if value:
            os.environ["HF_TOKEN"] = value
            os.environ["HUGGING_FACE_HUB_TOKEN"] = value
            return f"file:{path}"
    return None


source = load_token()
print(f"HF token: {source or 'NOT FOUND'}")
if source is None:
    sys.exit(
        "FATAL: no HF token.\n"
        "  1. Accept terms at https://huggingface.co/bioptimus/H-optimus-0\n"
        "  2. Create a read token at https://huggingface.co/settings/tokens\n"
        "  3. In THIS notebook: Add-ons > Secrets > add HF_TOKEN, tick to attach\n"
        "  4. Save & Run All from the UI (an API push will not see the secret)"
    )

# ------------------------------------------------------------------- deps
print("\ninstalling openslide + timm ...")
subprocess.run(
    "pip install -q openslide-bin openslide-python timm 2>&1 | tail -2",
    shell=True, check=False,
)

# ------------------------------------------------------------- fail fast
# Pull the config before touching GDC: a bad token should cost seconds, not
# an hour of downloading.
from huggingface_hub import hf_hub_download

cfg = hf_hub_download("bioptimus/H-optimus-0", "config.json", token=os.environ["HF_TOKEN"])
print(f"H-optimus-0 reachable ({os.path.getsize(cfg)} B config)\n")

# ------------------------------------------------------------------ report
import multiprocessing
import shutil

print(f"vCPU {multiprocessing.cpu_count()}")
for path in ("/kaggle/working", "/kaggle/tmp"):
    if os.path.isdir(path):
        print(f"{path:16s} {shutil.disk_usage(path)[2] / 1e9:7.1f} GB free")

try:
    import torch_xla.core.xla_model as xm
    print(f"XLA device {xm.xla_device()}")
except Exception as e:
    print(f"no XLA: {e}")

# --------------------------------------------------------------------- run
OUT = "/kaggle/working/features"
CACHE = "/kaggle/tmp/wsi"          # 1 TB scratch, wiped at session end
os.makedirs(CACHE, exist_ok=True)

argv = [
    "--out-dir", OUT,
    "--cache-dir", CACHE,
    "--labels-csv", f"{SRC}/tcga_hnsc_labels.csv",
    "--encoder", "h-optimus-0",
    "--device", "xla",
    "--format", "h5",
    "--max-patches", "6000",       # 435 slides x 6000 x 1536 x 2 B = 8.0 GB
    "--batch-size", "64",          # constant shape: one XLA compilation
    "--download-workers", "4",     # 143 MB/s aggregate vs 25.6 single stream
    "--prefetch", "4",
    "--decode-workers", "16",      # measured peak; 224 threads is slower
    "--max-hours", "7.5",          # stop cleanly inside the session cap
    "--min-free-gb", "3",
    "--notify-every", "25",
]
print("argv:", " ".join(argv), "\n", flush=True)

from pathgrade.preprocessing.stream_extract import main

code = main(argv)

# ------------------------------------------------------------------ verify
print(f"\n{'=' * 66}\nVERIFY\n{'=' * 66}")
from pathlib import Path

from pathgrade.data.io import verify_cohort

pids = sorted(p.stem for p in Path(OUT).glob("*.h5"))
if pids:
    info = verify_cohort(OUT, pids)
    for k, v in info.items():
        if k != "missing":
            print(f"  {k:18s} {v}")
    size = sum(p.stat().st_size for p in Path(OUT).glob("*.h5")) / 1e9
    print(f"  {'output size':18s} {size:.2f} GB")
print(f"  {'elapsed':18s} {(time.time() - T0) / 3600:.2f} h")
print(f"\nexit={code}")
