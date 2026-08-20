"""Preflight: can this account actually pull H-optimus-0, and does the package import?

Runs on a CPU session on purpose. H-optimus-0 is a gated HuggingFace repo, so a
run that lacks a token fails at model load - which on a TPU session would mean
discovering it after a long queue wait and a chunk of quota. Two minutes of CPU
here removes that risk.
"""
import os
import sys
import traceback

def find_src(marker="src/pathgrade/__init__.py", root="/kaggle/input"):
    """Locate the mounted dataset.

    Kaggle has more than one input layout - /kaggle/input/<slug> on some
    images, /kaggle/input/datasets/<owner>/<slug> on others - so search for the
    package instead of hardcoding either.
    """
    import glob
    for depth in ("*", "*/*", "*/*/*"):
        for hit in glob.glob(os.path.join(root, depth, marker)):
            return hit[: -len(marker) - 1]
    return None


print("=== /kaggle/input tree ===")
for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
    depth = dirpath.count(os.sep) - 2
    if depth > 2:
        dirnames.clear()
        continue
    print("  " * depth, os.path.basename(dirpath) or dirpath)

SRC = find_src()
print()
print(f"resolved SRC = {SRC}")
print()
if SRC:
    sys.path.insert(0, f"{SRC}/src")
ok = {}


def check(name, fn):
    try:
        result = fn()
        ok[name] = True
        print(f"[PASS] {name}: {result}")
    except Exception as e:
        ok[name] = False
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)


# ------------------------------------------------------------------ package
def _import():
    import pathgrade
    from pathgrade.encoders import check_licence
    from pathgrade.preprocessing.stream_extract import build_parser
    spec = check_licence("h-optimus-0")
    return f"pathgrade {pathgrade.__version__}, encoder {spec.name} ({spec.licence})"


check("package imports", _import)


# -------------------------------------------------------------------- token
def _token():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return f"token from env ({len(tok)} chars)"

    # Report the real reason per candidate name. Swallowing these hid whether
    # the secret is missing, misnamed, or simply not attached to THIS notebook.
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError as e:
        raise RuntimeError(f"kaggle_secrets unavailable: {e}")

    client = UserSecretsClient()
    errors = {}
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN",
                "HUGGINGFACE_HUB_TOKEN", "hf_token"):
        try:
            value = client.get_secret(key)
            if value:
                os.environ["HF_TOKEN"] = value
                os.environ["HUGGING_FACE_HUB_TOKEN"] = value
                return f"found Kaggle secret {key!r} ({len(value)} chars)"
            errors[key] = "returned empty"
        except Exception as e:
            errors[key] = f"{type(e).__name__}: {str(e)[:160]}"

    detail = "\n      ".join(f"{k}: {v}" for k, v in errors.items())
    raise RuntimeError(
        "no HF token visible to this kernel. Tried:\n      " + detail
    )


check("huggingface token", _token)


# --------------------------------------------------------------- gated access
def _access():
    os.system("pip install -q -U huggingface_hub 2>&1 | tail -1")
    from huggingface_hub import hf_hub_download, model_info

    tok = os.environ.get("HF_TOKEN")
    info = model_info("bioptimus/H-optimus-0", token=tok)
    # Pull only the tiny config, not the 4.4 GB of weights.
    cfg = hf_hub_download("bioptimus/H-optimus-0", "config.json", token=tok)
    return f"gated={info.gated} config={os.path.basename(cfg)} ({os.path.getsize(cfg)} B)"


check("H-optimus-0 access", _access)


# ------------------------------------------------------------------ openslide
def _openslide():
    os.system("pip install -q openslide-bin openslide-python 2>&1 | tail -1")
    import openslide
    return f"openslide-python {openslide.__version__}, lib {openslide.__library_version__}"


check("openslide", _openslide)


# ----------------------------------------------------------------------- gdc
def _gdc():
    from pathgrade.preprocessing.gdc import one_slide_per_patient, query_slides
    from pathgrade.data.splits import read_labels

    slides = one_slide_per_patient(query_slides("TCGA-HNSC"))
    labels = read_labels(f"{SRC}/tcga_hnsc_labels.csv")
    trainable = [s for s in slides if s.patient_id in labels]
    gb = sum(s.file_size for s in trainable) / 1e9
    return f"{len(trainable)} labelled slides, {gb:.0f} GB"


check("GDC query + labels", _gdc)

print("\n" + "=" * 60)
failed = [k for k, v in ok.items() if not v]
if failed:
    print(f"PREFLIGHT FAILED: {failed}")
    print("Do NOT start the TPU extraction until these pass.")
else:
    print("PREFLIGHT PASSED - safe to start the TPU extraction.")
print("=" * 60)
