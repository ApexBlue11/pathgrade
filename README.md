# pathgrade

**Commercially-clean whole-slide tumour grading.** Hierarchical attention MIL over
H-optimus-0 patch embeddings, with a rank-consistent ordinal head.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Encoder](https://img.shields.io/badge/encoder-H--optimus--0%20(Apache--2.0)-green.svg)](https://huggingface.co/bioptimus/H-optimus-0)
[![Tests](https://img.shields.io/badge/tests-43%20passing-brightgreen.svg)](tests/)

---

## Why this exists

This is a clean-room rebuild of an earlier HNSC grading project that was built on
**UNI**. That project could not be commercialised, for a reason no amount of
retraining fixes: the UNI gated access agreement prohibits commercial use of UNI
*and of its derivatives*, and it expressly defines derivatives to include
**models trained on outputs of the UNI model**. Every checkpoint trained on UNI
embeddings inherits the restriction. The weights were also being offered under
MIT, which was not the original author's to grant.

So the encoder is replaced, the head is retrained from scratch, and nothing in
this repository derives from a non-commercial model.

### The licence trap this repo is designed to avoid

`src/pathgrade/encoders.py` refuses to load a non-commercially-licensed encoder
unless you explicitly pass `allow_noncommercial=True`. Two pairs are easy to get
wrong, and in both the *newer, stronger* model is the restricted one:

| | Commercially usable | Blocked |
|---|---|---|
| Bioptimus | **H-optimus-0** (Apache-2.0) | H-optimus-**1** (CC-BY-NC-ND) |
| Paige | **Virchow v1** (Apache-2.0) | Virchow**2** (CC-BY-NC-ND) |

**Prov-GigaPath deserves special mention.** Hugging Face tags it `apache-2.0`,
which covers the *code*. Its model card states that any deployed use case,
commercial or otherwise, is out of scope. For a product it is more restrictive
than UNI. The HF metadata tag is not diligence.

Full table: `python scripts/01_extract_features.py --help`.

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

**Encoder choice matters more than any of this.** The Frontiers 2026
practical-guidelines study found aggregator choice is largely secondary to
embedding quality. The move off a weaker encoder is the single biggest lever;
everything above is the second-order gain.

---

## Pipeline

```bash
pip install -r requirements.txt
```

**1. Extract embeddings** — the only GPU-expensive stage.

```bash
python scripts/01_extract_features.py --slide-dir data/wsi --out-dir data/features/h-optimus-0 --encoder h-optimus-0
```

Writes one HDF5 per slide: `features [N, 1536]`, `coords [N, 2]`, plus a
provenance block (encoder, licence, MPP, tiling settings) in the file attributes.
Resumable — already-extracted slides are skipped.

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

### Read the number honestly

Inter-pathologist QWK for HNSCC differentiation grading is itself only about
**0.50–0.70**. A model scoring 0.67 is not "67% of the way to solved" — it is at
the noise floor of its own labels. `metrics.contextualise()` prints this next to
every result. Framed correctly it is a selling point; framed as "beats
pathologists" it will not survive scrutiny.

---

## Status and limitations

- **Not validated on real data yet.** The pipeline is verified end-to-end on
  synthetic MIL data with a known signal (43 passing tests). Real TCGA-HNSC
  numbers require running steps 1–4 on actual slides.
- **No external validation.** Single-cohort CV overstates deployability.
  CPTAC-HNSCC is the obvious second cohort.
- **TCGA grade labels are noisy** and G1/G4 are rare, which caps achievable QWK.
- **This is not a medical device.** A tumour grading product is regulated: FDA
  510(k)/De Novo in the US, IVDR Class C in the EU, UKCA in the UK. Budget for it.
- Licence status was verified August 2026 and can change. Re-verify before any
  release. Nothing here is legal advice.

---

## Citation

If you use this work, cite the underlying methods — see [NOTICE](NOTICE) for the
full list (ASMIL, nnMIL, CORN, ACMIL, ABMIL) and the encoder licences.
