# Running this on Kaggle

`pipeline_tpu.py` is the whole cohort pipeline in one script: mount the
labelled TCGA-HNSC slide list, stream each one from GDC, tile and encode it
with H-optimus-0, then train and evaluate on whatever has been extracted so
far. It's designed to run unattended on Kaggle's free TPU quota, launched and
monitored entirely through the API - no manual UI steps.

```bash
python kaggle/publish_src.py -m "what changed"   # only if src/ changed
python kaggle/push.py pipeline_tpu               # pushes AND starts the run
python kaggle/collect.py apexblue/pathgrade-pipeline
```

`publish_src.py` first, always: the kernel imports the `pathgrade` package
from a mounted dataset (`apexblue/pathgrade-src`), not from the kernel script
itself, so pushing new kernel code against a stale dataset silently runs the
old library. It also preserves `tcga_hnsc_labels.csv`, which lives in that
dataset but not in this git repo.

`push.py` handles two Windows/Kaggle-CLI quirks (a missing upload-cache
directory, and MAX_PATH on long staging paths) and refuses to publish a
script that fails `ast.parse` - a kernel with a syntax error still waits out
the TPU queue before failing in two seconds.

## The token

`HF_TOKEN` arrives as a file in a **private** Kaggle dataset
(`apexblue/pathgrade-token`, containing `hf_token.txt`), which the kernel
finds via `/kaggle/input/**/hf_token.txt`. This matters because Kaggle's
secrets service is not provisioned for API-pushed kernels at all -
`UserSecretsClient` fails with `ConnectionError: Connection error trying to
communicate with service` for every key - so a UI-attached secret is
invisible to anything launched by `kaggle kernels push`. Reading the token
from a dataset instead is what makes the whole run launchable from the API
with no human in the loop.

## Why extraction runs as a chain of short kernels, not one long one

Two attempts to extract the full 435-slide cohort in a single ~9-hour
session were killed by the platform partway through and produced **no
output at all** - not the embeddings, not the logs, not even a status file
flushed in the first second. A 34-minute run committed cleanly every time.

So extraction is capped at `PATHGRADE_MAX_EXTRACT_HOURS` (well under an
hour) and runs as a child process, isolating it from whatever OpenSlide does
on a malformed TIFF. Each kernel seeds its output from the previous one's,
mounted via `kernel_sources`:

```bash
python kaggle/push.py pipeline_tpu --as pathgrade-c2 \
    --kernel-source apexblue/pathgrade-c1 --env PATHGRADE_SKIP_TRAIN=1
```

Extraction was already idempotent (slides already on disk are skipped);
seeding from the prior kernel's output supplies the other half, so the chain
accumulates the full cohort with no single failure large enough to lose real
progress. `PATHGRADE_SKIP_TRAIN=1` on every chunk except the last: training
inside a container with a demonstrated failure mode would risk the slides
that same chunk just paid to extract. The full story, with the actual
tracebacks, is in [`docs/ENGINEERING.md`](../docs/ENGINEERING.md).

`collect.py` verifies a result rather than trusting a green status: it checks
for `TRAINING_FAILED.txt` and `EXTRACTION_FAILED.txt` before reading any
metric, because the training stage deliberately swallows its own failures
(prints a traceback, writes the marker, exits 0) so that hours of extracted
embeddings are never discarded by a training bug.

## Single TPU device, on purpose

A v5e-8 exposes eight XLA devices; this pipeline uses one.  Two ways of using
the rest were tried on real hardware and neither worked out on this
platform - `docs/ENGINEERING.md` has the failure modes. `build_encoders()` in
`encoders.py` will refuse a request to thread across devices and fall back
automatically, so a future retry can't silently corrupt a run either way.

## Why training runs on the TPU VM's CPU, not its TPU devices

The head is ~530K parameters. XLA compiles a graph per input shape, and the
training loop varies subspace and crop counts by design, so the TPU devices
would recompile constantly for a model that gains nothing from them. The TPU
VM's host carries 224 vCPU and ~405 GB RAM regardless - far more than a
530K-parameter model needs - and trains in under half an hour with zero
compilation risk. TPU devices are the right tool for the 1B-parameter
encoder; they are the wrong tool for the aggregator.

## Session facts

Kaggle allows **1** concurrent TPU session, **2** GPU, **5** CPU (CPU cannot
run the encoder at any usable speed). `/kaggle/working` persists and caps at
~21 GB; `/kaggle/tmp` is ~1.1 TB but wiped at session end. Full measured
throughput is in [`src/pathgrade/platform.py`](../src/pathgrade/platform.py)
and [`docs/ENGINEERING.md`](../docs/ENGINEERING.md).

`watch.py` tails a running kernel's log with per-line timestamps, useful
because the log streams live during a RUNNING kernel even though the
`kernels output` endpoint only returns files once a kernel is terminal:

```bash
python kaggle/watch.py apexblue/pathgrade-pipeline
```
