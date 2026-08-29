# pathgrade

**Commercially-clean whole-slide tumour grading.** Hierarchical attention MIL over
H-optimus-0 patch embeddings, with a rank-consistent ordinal head.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Encoder](https://img.shields.io/badge/encoder-H--optimus--0%20(Apache--2.0)-green.svg)](https://huggingface.co/bioptimus/H-optimus-0)
[![Tests](https://img.shields.io/badge/tests-135%20passing-brightgreen.svg)](tests/)

---



---

## explainable attention map


A production caller has one thing: a whole-slide image. Not a pre-extracted
feature file, not a separate thumbnail. One function call covers all of it -
tiling, encoding, prediction, and an attention overlay rendered on a
thumbnail pulled from that same slide, so it is guaranteed to line up with
the coordinates the attention map is drawn in:

```python
from pathgrade.inference import grade_slide

prediction, overlay = grade_slide("patient_042.svs", run_dir="runs/asmil-ord-hoptimus0")

print(prediction.summary())
# G2 - moderately differentiated  (confidence 71.4%)
#   G1 8.1%  G2 71.4%  G3 20.5%
#   expected grade 2.12 | 7,904 patches | ambiguity 0.062

for r in prediction.top_regions(5):        # "review these first"
    print(r["x"], r["y"], r["attention"])

overlay.save("explained.png")
```

Or as a CLI: `python scripts/05_predict_slide.py patient_042.svs --run-dir runs/asmil-ord-hoptimus0`.

`run_dir` is a completed training run's output directory, written by step 3
of the pipeline below. No checkpoint ships with this repository - `runs/` is
untracked because the weights are large, and because the attention map from
the first real run is not yet usable - measured, and documented under "Why
this attention map is defensible" below. Train a run first, or point `run_dir`
at your own; `GradePredictor.from_run` raises `FileNotFoundError` naming the
directory if it contains no `fold*/checkpoint.pt`.

If features are already extracted - the cohort pipeline below writes them -
`GradePredictor.predict_file("features/TCGA-BA-4078.h5")` skips straight to
prediction without re-tiling. `grade_slide` is the thin, deployment-shaped
wrapper: `pathgrade.preprocessing.single_slide.encode_slide` for tiling one
slide with the exact settings training used, then `GradePredictor.predict`,
then `render_overlay`.

A `GradePrediction` carries the grade, a calibrated ordinal posterior, an
**ambiguity** score, and a per-patch `attention` vector aligned 1:1 with the
`coords` written during extraction. That alignment is exactly why extraction
stores coordinates next to features.

`ambiguity` is distance to the CORN decision threshold - how close this slide
sits to a grade boundary. It is the signal to route a review queue on. The
older `uncertainty` field (disagreement between the five folds) is retained
for provenance but **measured at AUC 0.500 - exactly chance - at detecting its
own errors**, so do not gate anything on it. Ranking by `ambiguity` and keeping
the most decisive half of slides raised QWK from 0.293 to 0.472 on the first
real test set.

### Why this attention map is defensible

v1 displayed **input-gradient saliency**: the norm of `d(score)/d(features)` per
patch. That answers *"which patches would change the prediction if perturbed"* --
a sensitivity question, not an attribution one. It is noisy, sign-agnostic, and
not a quantity the model actually uses.

This returns the model's **real attention weights** -- the coefficients it
multiplies each patch by when pooling the bag into a slide representation. A
patch with attention 0.01 contributed exactly 1% of the slide embedding. No
gradients, no perturbation, no approximation, and it survives a clinician asking
"how do you know?".

Weights are averaged over all ACMIL branches and all 24 subspace ensemble
members, so the map reflects the full ensemble rather than one arbitrary view.

> **The checkpoint from the first real training run does not deliver this.**
> Its attention is *exactly* uniform - the top 1% of patches hold 1.0% of the
> mass, normalised entropy 1.0000 - so it is mean-pooling and the overlay is
> flat noise. Call `attention_is_informative(prediction)` before showing an
> overlay to anyone: a meaningless heatmap in front of a pathologist is worse
> than no heatmap.
>
> Measured since, against a **randomly initialised control**, and the honest
> statement is stronger than "collapsed": across 5 folds x 5 slides the trained
> attention is *no more peaked than random initialisation* (max/mean 1.128 vs
> 1.150), so the module never learned rather than having learned and degraded.
> Separately, the 24 subspaces disagree about which patches matter - pairwise
> correlation 0.087, top-30 overlap 1.3% against a 1.0% chance rate - so
> averaging them for display flattens what little structure each one has by
> 3.8x. That flattening is present at initialisation, so it is architectural.
> Both have to be fixed, and any fix has to beat the random-init control.
> Full evidence: [`docs/ENGINEERING.md`](docs/ENGINEERING.md).

Deployment path: **slide to H-optimus-0 embeddings + coords, then
`GradePredictor.predict`, giving grade + heatmap.** Only the first stage needs a GPU.

---

## Architecture

```
H-optimus-0  (frozen, Apache-2.0)
    |  224px tiles at 0.5 um/px
    v
[N x 1536] patch embeddings          <- no projection: pooling stays at full width
    |
    +-- gated attention scored on a 256-d subspace  ---- EMA anchor (NSF attention)
    |        x5 parallel branches                          |  KL stabilisation
    |        branch dropout 0.5                            |  training only
    v                                                       v
weighted pool over full 1536-d  ->  MLP  ->  2 CORN logits  ->  P(y>G1), P(y>G2)
```

**Design choices and where they come from:**

- **No projection bottleneck.** nnMIL (arXiv:2511.14907) shows that projecting
  foundation-model features down before aggregation destroys encoder semantics.
  The v1 model spent 32% of its parameters (262K of 823K) on a 1024→256 `proj`
  layer that did exactly that. Attention is scored on a subspace; pooling happens
  at full width.
- **Attention stabilisation.** ASMIL (arXiv:2603.06658) adds an EMA "anchor" copy
  of the attention scorer using a normalised-sigmoid map, pulling the online
  softmax branch toward it via KL. This targets precisely the "epoch-level F1
  swings" the v1 README complained about. The anchor is discarded at inference —
  zero deployment cost.
- **Subspace ensembling.** Attention is scored on one of 24 overlapping 256-d
  windows, drawn at random during training and *all averaged* at inference. That
  is a 24-way ensemble from a single trained model, replacing v1's three
  separately-trained seeds.
- **Rank-consistent ordinal head.** CORN (arXiv:2111.08851) factorises the grade
  through the chain rule, so P(y>G1) ≥ P(y>G2) holds *by construction*. v1 used
  `CE + α·(expected_grade − true)²`, which is not a proper ordinal likelihood and
  permits rank-inconsistent posteriors.
- **Fixed-size sub-bags.** Makes bags stackable, so batch size > 1 and
  class-balanced sampling both become available.

### Carried over from v1, and what was reconsidered

The v1 HPO notebook recorded a genuine list of hard-won fixes. Kept: warmup +
cosine decay, gradient clipping at global norm 1.0, `hidden_dim` 256, and the
underlying lesson behind Pre-LN — training instability was the real enemy,
though it is now addressed more directly by the ASMIL anchor.

Two were reconsidered:

- **SWA was dropped for the wrong reason.** The stated cause was an
  implementation bug ("fixed denominator + best epochs appear before SWA
  window"), not evidence that weight averaging hurts. It is back as
  `WeightEMA` — but the averaged weights are *evaluated against* the best-epoch
  weights on each fold and used only if they win, rather than assumed better.
- **Layer-wise learning rates were dropped.** v1 needed four LR tiers largely
  to nurse the `proj` bottleneck. With no projection layer there is much less to
  balance, so a single AdamW group is used until there is evidence otherwise.

MixUp stays out. v1 called it "biologically unsound for WSI feature space",
which is a reasonable prior, though feature-space mixing does have supporters in
recent MIL work. The sub-bag resampling already supplies strong augmentation, so
there is no need to relitigate it now.

**Encoder choice matters more than any of this.** The Frontiers 2026
practical-guidelines study found aggregator choice is largely secondary to
embedding quality. The move off a weaker encoder is the single biggest lever;
everything above is the second-order gain.

---

## Pipeline

```bash
pip install -r requirements.txt
```

**0. Plan the budget** — queries GDC live, before you spend any quota.

```bash
python scripts/00_plan_budget.py --max-patches 4000 --download-mbps 50
```

TCGA-HNSC is **472 diagnostic slides / 450 patients / 456 GB** (424 GB at one
slide per patient). That download dominates everything: on a TPU v5e-8 the
encode is ~29 min against ~2.4 h of transfer, so the job is **network-bound, not
compute-bound**. The planner prints how far you can raise `--max-patches` before
compute overtakes download — on v5e-8 that is ~10,000 patches/slide *for free*.

**1. Extract embeddings** — the only accelerator-expensive stage.

*Streaming (no local slides needed):* fetch, encode, delete, repeat. At most two
slides ever touch disk.

```bash
python scripts/01b_stream_extract.py --out-dir data/features/h-optimus-0 --device xla --max-patches 8000 --shard 0 --num-shards 4 --max-hours 8
```

*From local slides:*

```bash
python scripts/01_extract_features.py --slide-dir data/wsi --out-dir data/features/h-optimus-0
```

Writes one file per patient — `.h5` by default, `--format pt` for the older
Kaggle layout — holding `features [N, 1536]`, `coords [N, 2]`, and a provenance
block (encoder, licence, MPP, tiling). Both are resumable; already-extracted
slides are skipped, and a journal lets a killed session resume mid-shard.

### Running on Kaggle TPU

PyTorch runs on TPU via `torch_xla`; pass `--device xla`. Two details matter:

- **XLA compiles one graph per input shape.** A ragged final batch would
  recompile on every slide, so batches are padded to a constant size and sliced
  after. Keep `--batch-size` fixed.
- **`--shard i --num-shards n` splits slides across separate Kaggle sessions**,
  not across the eight XLA devices within one session - encoding across
  devices in a single process was tried and did not work out on this
  platform; see [`docs/ENGINEERING.md`](docs/ENGINEERING.md).

A single TPU session cannot run the full cohort in one sitting without risking
the whole run: see the survivability notes in the same file for why extraction
runs as a chain of short, resumable sessions rather than one long one.

**2. Build splits** — patient-level, with a locked test set.

```bash
python scripts/02_make_splits.py --labels-csv data/tcga_hnsc_labels.csv --out data/splits.json
```

**3. Train** — 5-fold CV. Never reads the test set.

```bash
python scripts/03_train_cv.py --config configs/hnsc_hoptimus0.yaml
```

**4. Evaluate** — once, at the end.

```bash
python scripts/04_evaluate_test.py --run-dir runs/asmil-ord-hoptimus0 --unlock
```

**5. Predict** — one slide, end to end. This is the path a deployment takes.

```bash
python scripts/05_predict_slide.py patient_042.svs --run-dir runs/asmil-ord-hoptimus0
```

**6. Tune** (optional) — Optuna over the CV folds. Never opens the test set.

```bash
python scripts/06_tune.py --feature-dir data/features/h-optimus-0     --splits data/splits.json --labels data/tcga_hnsc_labels.csv --trials 40
```

CPU-only against extracted features, so it costs no accelerator quota — but it
wants RAM: the feature cache holds the cohort in memory, and below that
threshold every sample re-reads a compressed HDF5 file each epoch and trials
take hours. `kaggle/tune_tpu.py` runs the same search on a TPU VM host for its
224 vCPU and ~405 GB.

**7. Audit the attention** — is the overlay an explanation or a decoration?

```bash
python scripts/08_attention_audit.py --run-dir runs/asmil-ord-hoptimus0     --feature-dir data/features/h-optimus-0
```

Scores the trained attention against **the same architecture untrained**.
That control is the whole point: softmax over a few thousand patches is never
perfectly flat, so peakedness on its own proves nothing, and only beating
random initialisation shows the module learned. It reports how peaked one
subspace's map is, whether the 24 subspaces agree on which patches matter,
and what the ensembled map a viewer actually sees looks like. Run it before
believing any overlay, including after a change meant to fix one. Reads no
labels and never touches the test set. (`scripts/07_ablate.py` is the other
optional step - a controlled 2x2 rather than a broad search.)

---

## Evaluation discipline

The v1 project reported **QWK 0.6683** on a validation set that had also driven
35 Optuna trials, early stopping, and best-epoch selection — and its
`get_splits(labels_csv, seed=42)` silently ignored its own `seed` argument, so
all three "independent seeds" shared one split. That number is a selection
optimum, not a generalisation estimate.

This repo enforces the alternative in code:

- One **locked test set**, carved out once and written with a SHA-256
  fingerprint. `load_splits` re-checks it and raises if the file was regenerated
  or edited, so a silently-changed split fails loudly instead of quietly
  inflating a score.
- **5-fold patient-level CV** for every tuning decision.
- `evaluate.py` requires `--unlock`, because consuming a test set is a one-way
  door.
- **Bootstrap confidence intervals** on QWK. On a few hundred slides the spread
  is roughly ±0.1; three decimal places without an interval implies a precision
  the cohort cannot support.

### Early stopping

v1 reset its patience counter on raw `val_qwk > best_qwk`. On an ~80-slide fold
a single slide changing grade moves QWK by about 0.02, so noise alone
manufactures "improvements" that keep a dead run alive. `EarlyStopping` instead:

- compares a **running median** of the metric, so one lucky epoch cannot grant a
  reprieve (there is a regression test for exactly this);
- requires gains to clear **`min_delta`**, set just above that noise floor;
- stops on a **flat trend** — a least-squares slope over the last 10 smoothed
  epochs below 0.001/epoch — not only on exhausted patience;
- refuses to stop before `min_epochs`, so warmup is never mistaken for a plateau;
- still **selects weights on the raw metric**, because you want the genuinely
  best epoch even though you judge termination on the trend.

On the synthetic end-to-end run this cut folds from 80 epochs to 30–34 with no
loss of accuracy.

### Read the number honestly

Inter-observer agreement on histologic grading tends to run **0.50–0.70** QWK
across several cancer types in the grading literature - grading is a
genuinely noisy label, not a ground truth with a single right answer.
`metrics.contextualise()` prints this range next to every result as context,
not as a benchmark to beat. **This specific 0.50–0.70 figure is carried in
this repo without a pinned citation for HNSCC specifically** and should be
verified against the literature before it appears in anything external. A
model scoring 0.67 should not be framed as "67% of the way to solved," and
should certainly not be framed as "beats pathologists" without a citation
that survives scrutiny.

---

## Status and limitations

**First real result, TCGA-HNSC, 435 slides, one locked test set, one seed:**

| | QWK | macro-F1 | balanced acc. | adjacent acc. |
|---|---|---|---|---|
| 5-fold CV (mean ± std) | 0.420 ± 0.073 | 0.514 | 0.534 | 0.989 |
| Locked test (n=66) | **0.293** [0.05, 0.50] 95% CI | 0.463 | 0.468 | 0.970 |

Read plainly, not as a headline:

- The test QWK sits **below** the typical inter-observer band this repo cites
  above, and its 95% CI is wide (a direct consequence of 66 test slides) -
  the honest statement is "not yet at reported human agreement, and not
  precisely enough measured to say by how much."
- **Adjacent accuracy near 97–99%** means the model is very rarely off by more
  than one grade even when it misses the exact class - errors are ordinal
  drift, not gross misclassification. That is the CORN head doing its job.
- The class split is uneven (G1 56 / G2 265 / G3 114); the confusion matrix
  shows G1 recall (3/9 in test) is the weakest cell, consistent with that
  imbalance rather than a modelling failure specific to G1.
- **Plain accuracy is not in that table, and the reason matters.** The model
  scores 0.530 on the locked test set; always predicting G2 scores 0.606,
  because 40 of the 66 test slides are G2. Accuracy rewards a constant
  predictor on a cohort this skewed, which is why the table reports QWK and
  balanced accuracy - measures a constant predictor cannot win. That same
  always-G2 baseline scores QWK 0.0 and balanced accuracy 0.333, against the
  model's 0.293 and 0.468. The model is genuinely better than the trivial
  answer; it is just not better on the metric that flatters it.
- Training used **3000 patches/slide**, a number chosen to fit a Kaggle
  session rather than tuned - see [`docs/ENGINEERING.md`](docs/ENGINEERING.md)
  for why raising it stayed out of scope this round.
- **The attention collapsed to uniform**, so this model is effectively
  mean-pooling and the score above is what mean-pooling achieves. A tuned
  mean-pooled logistic regression gets 0.251 on the same test set, which
  brackets how much the aggregator is currently contributing. Fixing the
  collapse is the first thing to try, and is diagnosed in full in
  `docs/ENGINEERING.md`.
- This is **one seed, one split**. Nothing here has been tuned against the
  test set - `evaluate.py` requires `--unlock` precisely so that stays true -
  but a single run is a starting point, not a validated estimate of what this
  architecture can do.

Also true:

- **No external validation.** Single-cohort CV overstates deployability.
  CPTAC-HNSCC is the obvious second cohort.
- **TCGA grade labels are noisy** and G1/G4 are rare, which caps achievable QWK
  independent of model quality.
- **This is not a medical device.** A tumour grading product is regulated: FDA
  510(k)/De Novo in the US, IVDR Class C in the EU, UKCA in the UK. Budget for it.
- Licence status was verified August 2026 and can change. Re-verify before any
  release. Nothing here is legal advice.

---

## Citation

If you use this work, cite the underlying methods — see [NOTICE](NOTICE) for the
full list (ASMIL, nnMIL, CORN, ACMIL, ABMIL) and the encoder licences.
