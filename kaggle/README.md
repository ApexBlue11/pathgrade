# Kaggle kernels

Scripts that run the pipeline on Kaggle, kept in the repo so a run is
reproducible from source rather than from someone's notebook history.

| file | session | purpose |
|---|---|---|
| `preflight.py` | CPU | verifies the dataset mount, HF token, gated model access, openslide and the GDC query before any quota is spent |
| `extract_tpu.py` | TPU | streams all labelled slides from GDC, encodes with H-optimus-0, writes embeddings |
| `train_tpu.py` | TPU VM (CPU) | training only, when features already exist |
| `pipeline_tpu.py` | TPU | **preferred**: extraction and training in one session |

## Prefer the combined pipeline

`pipeline_tpu.py` runs extraction and training back to back in a single
session. With one concurrent TPU session allowed and roughly 20 minutes of
queue per session, splitting the stages pays the queue tax twice and forces the
embeddings through a publish-then-remount round trip that buys nothing. Both
stages fit the session cap easily: extraction ~2 h, training minutes.

It is safe to re-run. Extraction skips slides already present, and the training
phase is guarded so a training bug prints a traceback, writes
`TRAINING_FAILED.txt` and still exits 0 - the embeddings, which cost hours, are
preserved as output either way. **A green run is not proof of success: check
for `TRAINING_FAILED.txt`.**

Output is flushed aggressively so `kaggle kernels logs <ref>` is useful while
the run is still going, rather than only once it ends. Confirmed: logs stream
with per-line timestamps during a RUNNING kernel, so a mistake an hour into
extraction surfaces immediately.

`watch.py` makes that practical - the raw command re-returns the entire log as
JSON on every call, so it diffs against what it has already shown:

```bash
python kaggle/watch.py apexblue/pathgrade-pipeline
python kaggle/watch.py apexblue/pathgrade-pipeline --grep "slides/h"
```

## Measured end to end

A dress rehearsal - the whole pipeline on 60 real slides with random weights -
established these on actual TPU sessions, not from micro-benchmarks:

| stage | measured | 435 slides |
|---|---|---|
| extraction | 87 slides/h at 3000 patches | ~5.0 h |
| training (5-fold CV) | 2.8 min for 60 slides | ~0.5 h |
| **total** | | **~5.5 h**, inside the 9 h cap |

Two numbers worth distrusting if you see them quoted anywhere earlier:

* **1226 patches/s for the encoder was fiction.** Without a `.cpu()` call XLA
  builds a graph lazily and never executes it, so that benchmark timed graph
  construction. Real throughput with the transfer the pipeline actually does is
  ~124 patches/s - which matches physics, since `xla:0` is one core of eight.
* **1654 tiles/s for decode was measured on an 11 MB cached slide.** Real
  slides run 1900-2500 tiles/s from any mount, so storage was never the limit.

Using all eight TPU cores is worth roughly 8x and is the obvious next
optimisation, but it is multi-device plumbing and belongs after a working model.

## Launch checklist

An API push cannot attach secrets, so the first push is deliberately made with
the accelerator set to CPU: the run starts immediately, fails in ~2 seconds at
the token check, and burns no TPU queue. Then, in the UI:

1. **Add-ons > Secrets** - tick `HF_TOKEN` (the secret is account-level; only
   the attachment is per notebook)
2. **Settings > Accelerator > TPU VM**
3. **Save & Run All**

Do not `kaggle kernels push` again afterwards - a new version can clear the
attachment.

## Order matters: TPU concurrency is 1

Only one TPU session may run at a time, so `train_tpu.py` must not be pushed
while extraction is queued or running - it would compete for the same slot.
Push it only once extraction reports COMPLETE.

`train_tpu.py` consumes the extraction output through `kernel_sources`, so
there is no manual dataset-publish step between the two.

## Why training uses the CPU of a TPU VM

The head is ~530K parameters. XLA compiles per input shape, and the training
loop varies subspace and crop counts by design, so it would recompile
constantly. The TPU VM carries 224 vCPU and 406 GB RAM, which is far more than
this model needs - it trains in minutes with no compilation risk. The TPU
*devices* are the right tool for extraction (a 1B-parameter encoder over
millions of tiles) and the wrong one for the aggregator.

## Secrets do not survive an API push

H-optimus-0 is a gated HuggingFace repo, so extraction needs an `HF_TOKEN`.
Kaggle secrets are attached **per notebook through the UI**, and a kernel pushed
with `kaggle kernels push` does not inherit that attachment — the secrets
service is not provisioned for it at all, and `UserSecretsClient` fails with
`ConnectionError: Connection error trying to communicate with service`.

Each `push` also creates a new version, which can clear an attachment made
earlier. So the working order is:

1. `kaggle kernels push -p .` to create or update the notebook
2. open it on kaggle.com
3. **Add-ons > Secrets**, add `HF_TOKEN`, tick to attach it to this notebook
4. **Save & Run All** from the UI
5. monitor with `kaggle kernels status` / `kaggle kernels output`

Do not push again after step 3 without redoing it.

## Measured session facts

See `src/pathgrade/platform.py`. Briefly: TPU allows **1** concurrent session,
GPU **2**, CPU **5**; CPU sessions cannot run a 1B-parameter encoder;
`/kaggle/tmp` holds ~1.1 TB of scratch while `/kaggle/working` caps at ~20 GB
and is the only path that persists.
