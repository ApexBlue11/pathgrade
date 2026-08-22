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

| | measured |
|---|---|
| GDC download, 4 parallel streams | **143 MB/s** (25.6 single — 5.6× win) |
| tile decode | 1900–2500 tiles/s, same from any mount |
| **full pipeline** | **87 slides/h @ 3000 patches** |
| training, 5-fold CV | **2.8 min / 60 slides** (was 21.3 before the RAM cache) |
| **projected full cohort** | **~5.5 h** (5.0 extraction + 0.5 training) |

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
- **107 unit tests**, including regression tests for every bug below.

### NOT proven — be honest about this

- **The full 435-slide run has never completed.** Longest real run: 60 slides.
- **No real accuracy number exists yet.** Every QWK produced so far is from
  noise embeddings and is meaningless by construction.
- **H-optimus-0 itself has never been downloaded** — it is gated and needs the
  owner's token. All rehearsals used `--random-weights`. The public ViT-g test
  covers the same code path but not that exact repo.
- Training time at 435 slides is **extrapolated** (~0.5 h) from 60 slides.
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

**Wrong hypotheses I burned time on**: pip upgrading torch, HF cache filling the
root FS, `/kaggle/tmp` being slow, batch size, non-zero exit discarding output.
All disproved by measurement. **Measure the real path; do not reason about it.**

Kaggle's log endpoint has returned **zero bytes even for runs that COMPLETED
successfully** (`pathgrade-load`). An empty log proves nothing. Trust the
fsync'd trail file in the output.

---

## 9. Where things stand RIGHT NOW

**Blocked on one manual step.** The full run has never executed.

Kernel: **https://www.kaggle.com/code/apexblue/pathgrade-pipeline** (v5, has the
trail instrumentation).

- `HF_TOKEN` **is already ticked** — it survived the v5 push.
- **Accelerator must be set to TPU VM** (metadata ships CPU deliberately so the
  API auto-run fails in 2 s instead of burning an hour of TPU queue).
- Then **Save & Run All** from the UI. It is a background batch job; the machine
  can be shut down.

Expect ~5.5 h plus queue. On completion:

```bash
python kaggle/collect.py apexblue/pathgrade-pipeline
```

That checks for `TRAINING_FAILED.txt` and the expected artifacts before
reporting anything — a COMPLETE status is **not** proof of success, because the
training phase deliberately swallows failures to preserve hours of embeddings.

If it fails, pull `pipeline_trail.txt` from the output; it names the last step
that started.

---

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
