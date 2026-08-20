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
the run is still going, rather than only once it ends.

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
