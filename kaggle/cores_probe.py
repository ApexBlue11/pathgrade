"""Measure what one process can actually do with a Kaggle TPU v5e-8.

This kernel exists because the repo's headline encoder number (1226 patches/s)
was fiction: it timed XLA graph *construction*, never execution, because nothing
forced a materialisation. Every figure here is taken with a `.cpu()` on the
result, so the device has demonstrably run the work before the clock stops.

Four questions, in order of how much they change the plan:

1. Does the HF token arrive via an attached dataset file rather than a UI
   secret? If yes, the whole pipeline becomes API-launchable and no longer
   depends on a human clicking Save & Run All.
2. Does H-optimus-0 actually download? It is gated and has never once been
   fetched in this project. Everything so far used --random-weights.
3. How many XLA devices does a single process address? The extraction loop uses
   xla:0 only, which platform.py says is one core of eight.
4. Does driving N devices from N Python threads actually scale? This is the
   crux. The forward pass has to release the GIL for threading to buy anything,
   and that is a property of torch_xla's C++ dispatch, not something worth
   guessing about.

Nothing here trains or writes embeddings. It is disposable.
"""
import glob
import json
import os
import sys
import threading
import time
from pathlib import Path

T0 = time.time()
WORK = Path("/kaggle/working")
TRAIL = WORK / "probe_trail.txt"
RESULT = WORK / "PROBE_RESULT.json"
with open(TRAIL, "w"):
    pass

R: dict = {"stages": {}}


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


def save() -> None:
    """Rewrite the result file after every stage, so a hang still leaves data."""
    R["elapsed_s"] = round(time.time() - T0, 1)
    try:
        with open(RESULT, "w") as fh:
            json.dump(R, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
    except OSError:
        pass


trail("BOOT", f"python {sys.version.split()[0]}")

# ---------------------------------------------------------------- 1. SOURCE
def find_src(marker="src/pathgrade/__init__.py", root="/kaggle/input"):
    for depth in ("*", "*/*", "*/*/*"):
        for hit in glob.glob(os.path.join(root, depth, marker)):
            return hit[: -len(marker) - 1]
    return None


SRC = find_src()
trail("SRC", str(SRC))
R["src"] = SRC
if SRC is None:
    trail("FATAL", "pathgrade-src not mounted")
    save()
    sys.exit("FATAL: pathgrade-src not mounted")
sys.path.insert(0, f"{SRC}/src")

# ----------------------------------------------------------------- 2. TOKEN
# The production kernel swallows the reason this fails (`last = e`, never
# read), which is why an earlier failure could not be diagnosed. Record it.
token_source, token_errors = None, {}
for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    if os.environ.get(var):
        token_source = f"env:{var}"
        break
if token_source is None:
    try:
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            try:
                value = client.get_secret(key)
                if value:
                    os.environ["HF_TOKEN"] = value
                    token_source = f"kaggle-secret:{key}"
                    break
            except Exception as e:
                token_errors[key] = f"{type(e).__name__}: {e}"
    except Exception as e:
        token_errors["import"] = f"{type(e).__name__}: {e}"
if token_source is None:
    for path in glob.glob("/kaggle/input/**/hf_token.txt", recursive=True):
        value = open(path).read().strip()
        if value:
            os.environ["HF_TOKEN"] = value
            os.environ["HUGGING_FACE_HUB_TOKEN"] = value
            token_source = f"file:{path}"
            break

# Never print the token itself - only where it came from and how long it is.
R["token"] = {
    "source": token_source,
    "length": len(os.environ.get("HF_TOKEN", "")),
    "secret_errors": token_errors,
    "files_seen": glob.glob("/kaggle/input/**/hf_token.txt", recursive=True),
}
trail("TOKEN", f"{token_source} (len {R['token']['length']})")
save()
if token_source is None:
    trail("FATAL", "no token by any route")
    save()
    sys.exit("FATAL: no token")

print("installing timm ...", flush=True)
os.system("pip install -q timm 2>&1 | tail -2")

# ------------------------------------------------------------ 3. TOPOLOGY
import multiprocessing

import torch

trail("STAGE", "topology")
topo = {"vcpu": multiprocessing.cpu_count(), "torch": torch.__version__}
try:
    import psutil

    topo["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
except Exception:
    pass

try:
    import torch_xla
    import torch_xla.core.xla_model as xm

    topo["torch_xla"] = torch_xla.__version__
    try:
        import torch_xla.runtime as xr

        for fn in ("global_runtime_device_count", "addressable_runtime_device_count",
                   "world_size", "local_process_count"):
            try:
                topo[fn] = getattr(xr, fn)()
            except Exception as e:
                topo[fn] = f"ERR {type(e).__name__}: {e}"
    except Exception as e:
        topo["xr_import"] = f"ERR {e}"
    for fn in ("get_xla_supported_devices", "xla_real_devices"):
        try:
            val = getattr(xm, fn)()
            topo[fn] = [str(d) for d in val] if val else val
        except Exception as e:
            topo[fn] = f"ERR {type(e).__name__}: {e}"
    topo["default_device"] = str(xm.xla_device())
except Exception as e:
    topo["xla_error"] = f"{type(e).__name__}: {e}"

R["topology"] = topo
trail("TOPOLOGY", json.dumps({k: v for k, v in topo.items() if k != "notes"})[:400])
save()

devices = topo.get("get_xla_supported_devices") or []
if isinstance(devices, str) or not devices:
    devices = [topo.get("default_device", "xla:0")]
R["n_devices_visible"] = len(devices)
trail("DEVICES", f"{len(devices)} visible: {devices}")
save()

# ------------------------------------------------------- 4. REAL ENCODER
from pathgrade.encoders import PatchEncoder, get_spec

spec = get_spec("h-optimus-0")
BATCH = int(os.environ.get("PROBE_BATCH", "64"))

trail("STAGE", "load H-optimus-0 on xla:0 (first real fetch in this project)")
t = time.time()
try:
    enc0 = PatchEncoder(spec, device=torch.device(devices[0]))
    R["stages"]["load_first_replica_s"] = round(time.time() - t, 1)
    trail("LOADED", f"{R['stages']['load_first_replica_s']}s  dim={spec.embed_dim}")
except Exception as e:
    import traceback

    R["stages"]["load_error"] = f"{type(e).__name__}: {e}"
    R["stages"]["load_traceback"] = traceback.format_exc()[-2000:]
    trail("FATAL", f"encoder load failed: {type(e).__name__}: {e}")
    save()
    sys.exit(1)
save()

import numpy as np

rng = np.random.default_rng(0)
sample = torch.from_numpy(rng.integers(0, 255, (BATCH, 224, 224, 3), dtype=np.uint8))


def bench(encoder, n_batches: int, tag: str) -> float:
    """Patches/s for one device. `.cpu()` forces execution - without it XLA
    builds a graph lazily and the timing is meaningless."""
    import torch_xla.core.xla_model as xm

    out = encoder(sample)
    xm.mark_step()
    out.cpu()                                  # warm-up: pays the compile
    t0 = time.time()
    for _ in range(n_batches):
        out = encoder(sample)
        xm.mark_step()
        out.cpu()
    dt = time.time() - t0
    rate = n_batches * BATCH / dt
    trail("BENCH", f"{tag}: {rate:.1f} patches/s ({n_batches} x {BATCH} in {dt:.1f}s)")
    return rate


N_BATCHES = int(os.environ.get("PROBE_BATCHES", "20"))

trail("STAGE", "single-device throughput")
try:
    single = bench(enc0, N_BATCHES, "1 device")
    R["single_device_patches_per_s"] = round(single, 1)
except Exception as e:
    R["single_device_error"] = f"{type(e).__name__}: {e}"
    trail("ERROR", f"single-device bench failed: {e}")
save()

# Sanity: the embedding must not be constant or NaN. A fast number from a
# broken forward pass is worse than a slow correct one.
try:
    v = enc0(sample).cpu().numpy()
    R["embedding_check"] = {
        "shape": list(v.shape),
        "finite": bool(np.isfinite(v).all()),
        "std": float(v.std()),
        "distinct_rows": int(len(np.unique(np.round(v[:, :8], 4), axis=0))),
    }
    trail("EMBED", json.dumps(R["embedding_check"]))
except Exception as e:
    R["embedding_check"] = f"ERR {e}"
save()

# --------------------------------------------------- 5. MULTI-DEVICE SCALING
# The question that decides the architecture: do N threads on N devices scale?
def scale_test(n: int) -> dict:
    trail("STAGE", f"replicating encoder onto {n} devices")
    replicas = [enc0]
    t = time.time()
    for d in devices[1:n]:
        replicas.append(PatchEncoder(spec, device=torch.device(d)))
    build_s = round(time.time() - t, 1)
    trail("REPLICAS", f"{len(replicas)} ready in {build_s}s")

    per_thread, errors = {}, {}

    def worker(i: int):
        try:
            per_thread[i] = bench(replicas[i], N_BATCHES, f"dev{i} of {n}")
        except Exception as e:
            errors[i] = f"{type(e).__name__}: {e}"

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(replicas))]
    t0 = time.time()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.time() - t0

    # Aggregate measured against wall clock, not the sum of per-thread rates:
    # if the GIL serialises them, per-thread rates stay high while the wall
    # clock does not improve, and only this ratio exposes that.
    total_patches = len(replicas) * N_BATCHES * BATCH
    agg = total_patches / wall
    out = {
        "n_devices": len(replicas),
        "build_s": build_s,
        "wall_s": round(wall, 1),
        "aggregate_patches_per_s": round(agg, 1),
        "per_thread_patches_per_s": {k: round(v, 1) for k, v in per_thread.items()},
        "errors": errors,
    }
    trail("SCALE", json.dumps({k: out[k] for k in ("n_devices", "aggregate_patches_per_s", "wall_s")}))
    return out, replicas


R["scaling"] = {}
n_visible = len(devices)
try:
    for n in [x for x in (2, 4, 8) if x <= n_visible]:
        res, _ = scale_test(n)
        R["scaling"][str(n)] = res
        save()
except Exception as e:
    import traceback

    R["scaling_error"] = f"{type(e).__name__}: {e}"
    R["scaling_traceback"] = traceback.format_exc()[-2000:]
    trail("ERROR", f"scaling failed: {e}")

# ------------------------------------------------------------- 6. VERDICT
base = R.get("single_device_patches_per_s")
best_n, best_rate = 1, base or 0.0
for k, v in R.get("scaling", {}).items():
    rate = v.get("aggregate_patches_per_s", 0)
    if rate > best_rate:
        best_n, best_rate = int(k), rate
R["verdict"] = {
    "single": base,
    "best_n_devices": best_n,
    "best_aggregate": best_rate,
    "speedup": round(best_rate / base, 2) if base else None,
    "threads_scale": bool(base and best_rate > 1.5 * base),
}
trail("VERDICT", json.dumps(R["verdict"]))

# What the measured rate implies for the real job, so the next decision is
# arithmetic rather than judgement.
if best_rate:
    for mp in (3000, 6000, 8000, 10000):
        total = 435 * mp
        R.setdefault("projection_hours", {})[str(mp)] = {
            "encode_h": round(total / best_rate / 3600, 2),
            "output_gb": round(435 * mp * 1536 * 2 / 1e9, 1),
        }
    trail("PROJECTION", json.dumps(R["projection_hours"]))

save()
trail("DONE", f"total {time.time() - T0:.0f}s")
print("\n" + json.dumps(R, indent=2, default=str)[:4000], flush=True)
