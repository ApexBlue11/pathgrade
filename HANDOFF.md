# pathgrade — session handoff

**Written 2026-08-22.** Everything a new session needs to continue without
re-deriving it. Read this before touching anything.

---

## 1. What this is and why

Commercialising an automated **HNSCC tumour grading** model (G1/G2/G3) from
whole-slide images. Owner: Surya (GitHub `ApexBlue11`, Kaggle `apexblue`).

**Goal is highest achievable accuracy**, subject to being legally shippable.
This is a product, not a paper.

### The reason the old project was scrapped

The predecessor (`ApexBlue11/H-MIL-Tumour-Grading`, JAX/Flax, QWK 0.6683) was
built on **UNI**. UNI's gated agreement bans commercial use of UNI *and* of
"models trained on outputs from the UNI model" — so every checkpoint was
permanently research-only. No retraining trick launders that. It also shipped
UNI-derived weights under MIT, which was not the author's to grant.

That repo still exists and **should be deleted or archived by the owner** — it
publicly offers UNI-derived weights under MIT. Same for the HuggingFace Space
`ApexBlue720/TumourGrading_Model`.

### Encoder licensing (verified Aug 2026 — re-verify before release)

| encoder | dim | licence | usable? |
|---|---|---|---|
| **H-optimus-0** | 1536 | Apache-2.0 | ✅ **chosen** |
| Virchow **v1** | 2560 | Apache-2.0 | ✅ backup |
| Hibou-L | 1024 | Apache-2.0 | ✅ |
| Midnight | 3072 | MIT | ⚠️ trained on TCGA — contaminates our eval |
| UNI / UNI2-h | 1024/1536 | CC-BY-NC-ND | ❌ derivatives banned |
| **Prov-GigaPath** | 1536 | HF tag says apache-2.0 | ❌ model card: *any* deployed use out of scope |
| Virchow**2**, H-optimus-**1** | | CC-BY-NC-ND | ❌ |
| Phikon / Phikon-v2 | | Owkin non-commercial | ❌ |

Two traps: **v1 is free, v2 is not** for both Paige and Bioptimus — the newer,
stronger model is the restricted one. `src/pathgrade/encoders.py` enforces this
in code; non-commercial encoders raise `LicenceError` without an explicit opt-in.

---

## 2. Architecture: ASMIL-Ord

`src/pathgrade/models/asmil_ord.py`, **529,927 trainable params**.

```
H-optimus-0 (frozen) -> [N x 1536] + coords
  gated attention scored on a 256-d subspace, x5 branches, branch-drop 0.5
    + EMA anchor with normalised-sigmoid attention, KL-pulled (ASMIL)
  pool over FULL 1536-d  ->  MLP  ->  2 CORN logits
```

Design choices and their sources:

- **No projection bottleneck** (nnMIL, arXiv:2511.14907). v1 spent 32% of its
  parameters on a 1024→256 `proj` that destroyed encoder semantics.
- **Attention stabilisation** (ASMIL, arXiv:2603.06658) — EMA anchor + NSF +
  KL. Targets exactly the "epoch-level F1 swings" v1's README complained about.
  Discarded at inference, so free at deployment.
- **24-way subspace ensemble** from one model (nnMIL) — replaces v1's three
  separately-trained seeds.
- **CORN ordinal head** (arXiv:2111.08851) — rank consistency by construction.
  v1 used `CE + α·(expected−true)²`, not a proper ordinal likelihood.
- **Fixed sub-bags** — makes bags stackable, enabling batch>1 and class-balanced
  sampling.

**Encoder choice dominates aggregator choice** (Frontiers 2026 practical
guidelines). The H-optimus-0 swap is the big lever; the above is second-order.

Optional, **off by default, enable only on CV evidence**: bag MixUp
(`loss.mixup_prob`, swaps whole patches between slides — never interpolates
features), discriminative LRs (`optim.lr_mult_*`).

---

## 3. The product surface — do not lose this

`src/pathgrade/inference.py`. This is what gets commercialised.

`GradePredictor.predict()` → grade, calibrated ordinal posterior, **uncertainty
from fold disagreement**, and a per-patch **attention map** aligned 1:1 with the
coords extraction stores. Plus `top_regions(k)` and `render_overlay()`.

The attention is the model's **real pooling weights**, not gradient saliency. A
patch with attention 0.01 contributed exactly 1% of the slide embedding. v1
showed `‖d(score)/d(features)‖`, which answers a sensitivity question, not an
attribution one.

Deployment path: **user slide → H-optimus-0 embeddings + coords →
`GradePredictor.predict` → grade + heatmap.** Only stage one needs a GPU.

---

## 4. Evaluation discipline (non-negotiable)

v1 reported QWK 0.6683 on a validation set that also drove 35 Optuna trials,
early stopping and best-epoch selection — and its `get_splits(seed=42)`
**ignored its own seed argument**, so all three "independent seeds" shared one
split. That number is a selection optimum.

Enforced in code now: one **locked test set** with a SHA-256 fingerprint
(`load_splits` raises if regenerated), **5-fold patient-level CV** for every
tuning decision, `evaluate.py` requires `--unlock`, and **bootstrap CIs** on QWK.

**Context for any number**: inter-pathologist QWK for HNSCC grading is itself
only ~0.50–0.70. `metrics.contextualise()` prints this. A model at 0.67 is at
the label noise floor, not 67% of the way to solved.

---

## 5. Cohort

- TCGA-HNSC: **472 diagnostic slides / 450 patients / 456 GB** (424 GB at one
  slide per patient), queried live from GDC.
- Labels: `apexblue/tumour-grading-model-tcga-manifest` → 502 patients.
- **Intersection: 435 patients**, classes **56 / 265 / 114** (G1/G2/G3).
  G4's 7 cases were already folded into G3 by the label file.
- Extraction uses `--labels-csv` to skip the 15 unlabelled slides.

---

## 6. Kaggle infrastructure — MEASURED, not assumed

Recorded in `src/pathgrade/platform.py`.

| | concurrent | vCPU | weekly | can encode? |
|---|---|---|---|---|
| TPU (v5e-8, 8 XLA devices) | **1** | 224 | 20 h | yes |
| GPU (P100 by default, not T4×2) | 2 | 4 | 30 h | yes |
| CPU | 5 | 4 | — | **no** (0.38 patches/s → 400+ h) |

- `/kaggle/working` **20.9 GB**, persists, becomes output.
- `/kaggle/tmp` **1098 GB**, wiped at session end. Slides must cache here.
- TPU VM: 224 vCPU, **396 GB RAM**.
- TPU queue: 20 min to **>1 h**, variable. Queue time is not billed.

### Measured throughput

Everything below with **real H-optimus-0 weights** unless noted.

| | measured |
|---|---|
| GDC download, 4 parallel streams | **143 MB/s** (25.6 single — 5.6× win) |
| tile decode | 1900–2500 tiles/s, same from any mount |
| encoder, one XLA device | **123.5 patches/s** (confirmed twice, forced `.cpu()`) |
| full pipeline, random weights | 87 slides/h @ 3000 patches |
| **full pipeline, real weights** | **73–90 slides/h** (40- and 79-slide chunks) |
| training, 5-fold CV | **2.8 min / 60 slides** (was 21.3 before the RAM cache) |

### Where a slide's time actually goes (c1, 79 slides, instrumented)

| stage | median | share of in-slide time |
|---|---|---|
| `grid_seconds` (tissue + tiling) | 2.5 s | 10% |
| `encode_seconds` | 24.5 s | **89%** |
| `write_seconds` (h5 + gzip) | 0.2 s | 1% |

But `slide_seconds` summed to 2142 s against a **3158 s** extraction wall, so
**~32% of the wall is the consumer waiting on downloads**. That is *not*
bandwidth: `SlidePrefetcher` yields results strictly in record order, so one
slow download (max seen 63 s) stalls every slide queued behind it, even those
already on disk. Fixing that head-of-line blocking — consuming completed
downloads out of order — is the cheapest remaining win, worth ~1.3× on
extraction. It was left alone during the extraction campaign because the
prefetcher is the component that produced bug #1.

A guess worth recording as wrong: before instrumenting, the missing time was
assumed to be tissue detection. It is 10%.

### Numbers that were WRONG — do not reuse

- **"1226 patches/s encoder"** — fiction. Without `.cpu()`, XLA builds a graph
  lazily and never executes it; that benchmark timed graph construction. Real
  is ~124 patches/s, which matches physics: `xla:0` is **one core of eight**.
- **"1654 tiles/s decode"** — measured on an 11 MB cached slide. Real slides
  run 1900–2500/s from any mount, so storage was never the bottleneck.

---

## 7. What is PROVEN vs UNPROVEN

### Proven on real TPU sessions with real TCGA slides

- Extraction end-to-end: GDC stream → tissue → tile → XLA encode → HDF5 with
  provenance. 3-slide and 60-slide runs, **0 failures**.
- **Full training half** (`pathgrade-rehearsal`, 60 real slides): splits, 5-fold
  CV, plateau early-stopping, EMA-vs-best-epoch, locked-set eval with bootstrap
  CIs, attention map, 15-artifact release bundle. Correctly reported QWK 0.000
  with everything on the majority class — the right answer for noise embeddings.
- Real pretrained weight loading (`pathgrade-load`, public 1.13B ViT-g):
  9.09 GB download 62 s, XLA move 1 s, forward 8 s, `PatchEncoder` + width probe
  16 s, peak RSS 13.3 GB of 396 GB.
- **116 unit tests**, including regression tests for every bug below and for
  the multi-device encode path (order preservation, byte-identical agreement
  with the serial path, work actually spreading across replicas, and a dead
  replica raising instead of returning uninitialised rows).

### NOT proven — be honest about this

- **The full 435-slide run has never completed.** Longest real run: 60 slides.
- **No real accuracy number exists yet.** Every QWK produced so far is from
  noise embeddings and is meaningless by construction.
- **H-optimus-0 itself has never been downloaded** — it is gated and needs the
  owner's token. All rehearsals used `--random-weights`. The public ViT-g test
  covers the same code path but not that exact repo.
- Training time at 435 slides is **extrapolated** (~0.5 h) from 60 slides, and
  that extrapolation assumed 3000 patches/slide. `bag_size` is derived from the
  cohort median, so raising `--max-patches` raises training cost roughly
  linearly in total patches.
- **Multi-device speedup is unmeasured.** The code is tested for correctness on
  CPU, but whether eight threads actually beat one on TPU depends on how much
  of a lazy-tensor ViT-g forward holds the GIL. `cores_probe.py` measures it.
- **The 8-stream download setting is a guess.** 4 streams measured 143 MB/s and
  nobody tried more; once encoding is 8x faster, download is the floor.
- No external validation cohort. CPTAC-HNSCC is the obvious second.

---

## 8. Bug history — and the lesson

Nine bugs, all mine, **none found by reasoning**. Every one came from running
the real thing on real data.

1. Prefetcher deadlock — semaphore acquired inside the worker, so freed threads
   took slots the in-order consumer was waiting on. Caught by a disk-bounding test.
2. `encode_tiles` parallelised across *batches* not *tiles* → 106 vs 371 tiles/s.
3. Inference-tensor crash — normalisation constants built lazily inside an
   `@torch.inference_mode()` forward became inference tensors cached on the
   module; XLA refused to version-count them. **Killed every slide.**
4. `--random-weights` picked `vit_giant_patch14_224` (1408 wide) for a 1536
   encoder. Fixed by `RANDOM_ARCH`; also added a construction-time width probe.
5. Rehearsal demanded an HF token it never uses (flags read after the gate).
6. Rehearsal metadata copied with `enable_tpu: false`.
7. `UnboundLocalError` in the RAM cache — **all 102 tests passed anyway**,
   because auto-detection reads `/proc/meminfo` and silently stays off on
   Windows, so the cached path was never executed.
8. Training 8× too slow — per-epoch HDF5 decompression, fixed by the RAM cache.
9. **The pipeline had no fsync trail** while every throwaway diagnostic did.
   The most important kernel was the least observable, which is why two of the
   owner's runs produced nothing to debug. Fixed.

### Session 2 (2026-08-22/23) added five more, same lesson

10. **Multi-device XLA threading is broken, and the benchmark said otherwise.**
    `cores_probe.py` reported `speedup 8.54, threads_scale: true`. It divided
    *intended* work by wall time without checking the threads finished. Seven
    of eight had died in `SyncLiveTensorsGraph`; the true figure was 68
    patches/s, **slower than one device**. Identical in shape to the retracted
    1226. Any throughput number must verify the work happened.
11. **`reporter.update()` was only called on failure.** The heartbeat read
    `done: 0` after 40 successful slides, and the webhook would only ever have
    fired on errors - so a slow run was indistinguishable from a dead one. This
    silently defeated the observability work built the same hour.
12. **Extraction had no exception guard** while training had one since day one.
    The *expensive* half was the unprotected one: a raise at slide 400 discarded
    399 encoded slides.
13. **A syntax error was pushed to Kaggle.** It still waits out the ~40 min TPU
    queue before dying in 2 s. `push.py` now refuses to publish anything that
    fails `ast.parse`.
14. **The parent must never touch an XLA device** if a child needs one -
    creating a device claims the TPU for that process's lifetime. The
    accelerator check runs in a throwaway subprocess.

**Wrong hypotheses I burned time on**: pip upgrading torch, HF cache filling the
root FS, `/kaggle/tmp` being slow, batch size, non-zero exit discarding output,
and (session 2) tissue detection being the unmeasured half of slide time - it
is 10%, the real answer was download head-of-line blocking.
All disproved by measurement. **Measure the real path; do not reason about it.**

One environment note that cost real time: in this Git-Bash-on-Windows shell,
`sed` and `py_compile` returned **stale cached content** immediately after a
successful write, so a correct fix looked like it had failed twice. Use the
Read/Edit file tools for edits that matter.

Kaggle's log endpoint has returned **zero bytes even for runs that COMPLETED
successfully** (`pathgrade-load`). An empty log proves nothing. Trust the
fsync'd trail file in the output.

---

## 9. Where things stand RIGHT NOW

**Updated 2026-08-23.** Extraction is running as a chain of short kernels.
Real H-optimus-0 embeddings exist for the first time.

### Launching no longer needs a human

`hf_token.txt` lives in the private dataset `apexblue/pathgrade-token`
(`isPrivate: true`). `load_token()` finds it at
`/kaggle/input/**/hf_token.txt`, so an API push both deploys and runs. Kaggle's
secrets service is confirmed unavailable to API-pushed kernels - it returns
`ConnectionError: Connection error trying to communicate with service` for
every key, which the pipeline now records instead of swallowing.

    python kaggle/publish_src.py -m "..."   # ONLY when src/ changes
    python kaggle/push.py pipeline_tpu ...  # pushes AND starts a run

`publish_src.py` exists because **`tcga_hnsc_labels.csv` is in the dataset but
not in git**; building the dataset from the repo alone deletes the labels and
breaks training hours later. It downloads the current version first and aborts
if the CSV is missing. `--dir-mode` also defaults to `skip`, which would
silently upload a dataset containing no `src/` at all.

`push.py` refuses to publish a script that fails `ast.parse`. A kernel with a
syntax error still waits out the full TPU queue before dying in two seconds.

### THE BIG PROBLEM: long runs are killed and lose everything

| run | duration | result |
|---|---|---|
| v6, 435 slides | 86 min | ERROR, **zero output** |
| smoke, 40 slides | 34 min | COMPLETE, full output |
| v7, 435 slides | 3 h 45 | ERROR, **zero output** |
| c1, chunked | 54 min | COMPLETE, full output |

"Zero output" means nothing at all - not the embeddings, not the log, not even
`pipeline_trail.txt`, which is fsync'd in the first second. Control fetches
against finished kernels return their files instantly, so the endpoint is fine
and retention lasts days. Extraction was moved into a **child process** to
survive a segfault in OpenSlide/libtiff; v7 still lost everything, which proves
the *whole container* is being killed, not the extraction process.

**The cause is still unknown and probably cannot be found from inside.** A
trail file cannot describe a failure that discards the trail. The fix for that
is `--webhook-url`, which `ProgressReporter` already supports for
Discord/Slack/Telegram and which `find_webhook()` will pick up from
`/kaggle/input/**/webhook_url.txt` or `PATHGRADE_WEBHOOK`. **Nobody has
supplied one yet.** Do that before attempting any long run again.

### So extraction runs as a chain of short, cumulative kernels

Each kernel seeds `/kaggle/working/features` from the previous kernel's output
mounted via `kernel_sources`, extracts for `PATHGRADE_MAX_EXTRACT_HOURS`
(default **0.85 h**), skips training, and exits cleanly so Kaggle commits.
Extraction was always idempotent; seeding supplies the missing half.

    python kaggle/push.py pipeline_tpu --as pathgrade-cN         --kernel-source apexblue/pathgrade-c<N-1>         --env PATHGRADE_SKIP_TRAIN=1

Chain so far: `pathgrade-smoke` (40) -> `pathgrade-c1` (**119 banked**) -> `c2` ...
Each chunk adds ~79 slides, so roughly four chunks reach 435.

**The final chunk drops `PATHGRADE_SKIP_TRAIN`** so it trains and produces the
release bundle. Then:

    python kaggle/collect.py apexblue/pathgrade-cN

A COMPLETE status is still not proof: check `TRAINING_FAILED.txt` and
`EXTRACTION_FAILED.txt`, which `collect.py` does.

### Verified real embeddings

119 files, 0.99 GB, all 3000 patches x 1536, coords aligned, attrs say
`encoder: h-optimus-0, random_weights: False`. 435 slides projects to ~3.6 GB
against the 20.9 GB cap.

### Training note

A 40-slide chunk failed training with `Rarest class has 4 patients but
n_folds=5`. That is a small-cohort artifact, not a bug - G1 has 56 patients at
435. It is why intermediate chunks skip training.

## 10. Roadmap once a real number exists

1. **Use all 8 TPU cores** — currently on 1 of 8, worth ~8×. Would allow
   6000–10000 patches/slide instead of 3000. Multi-device inference plumbing.
2. Raise `--max-patches` once (1) lands; 3000 was chosen purely to fit the session.
3. Turn on MixUp / discriminative LRs if CV says they help.
4. **External validation on CPTAC-HNSCC** — single-cohort CV overstates deployability.
5. Consider paying for a commercial licence to UNI2-h or Virchow2 if the
   accuracy gap justifies it (Modella AI, Paige, Bioptimus all license).
6. Regulatory: this is a medical device. FDA 510(k)/De Novo, EU IVDR Class C,
   UKCA. Budget for it.

---

## 11. Repo and assets

- GitHub: **https://github.com/ApexBlue11/pathgrade** (private, Apache-2.0)
- Kaggle source dataset: `apexblue/pathgrade-src` (kernels read the package from here)
- Kernels: `pathgrade-pipeline` (production), `-rehearsal`, `-repro`, `-perf`,
  `-perf2`, `-load`, `-diag`, `-exitcheck`, `-preflight` (diagnostics, disposable)
- `kaggle/watch.py` — tails a running kernel; `kaggle/collect.py` — verifies results

**Workflow constraint**: `kaggle kernels push` cannot attach secrets, and the
secrets service is not even provisioned for API-pushed kernels
(`ConnectionError: Connection error trying to communicate with service`). Push
the code by API, then attach and launch from the UI. (Empirically the secret
attachment *did* survive a later push — but do not rely on it.)
