# Experiment backlog

Things to try against the weak result, in the order the evidence supports, with
what each one predicts and how it will be judged. Kept so that a change is
chosen because a measurement pointed at it, rather than because it is a
well-known technique.

Nothing here is a claim about what will work. Results move to
[`ENGINEERING.md`](ENGINEERING.md) once measured, whichever way they come out.

## How anything here gets judged

Two bars, both cheap, and a change has to clear the relevant one:

* **The random-init control.** `scripts/08_attention_audit.py` scores the
  trained attention against the same architecture untrained. The current model
  does not beat it (per-subspace max/mean 1.128 trained vs 1.150 untrained), so
  an attention change that does not move this has not done anything, whatever
  it does to QWK.
* **The mean-pool baseline.** A logistic regression on mean-pooled features
  scores **CV QWK 0.354**. The current model scores **0.420**, so the whole
  aggregator is worth +0.066 today. A change that improves QWK without widening
  that gap has improved the head, not the pooling.

Every number below is 5-fold CV on the development split. The locked test set
(fingerprint `eeaa1233f18fea05`) stays shut until there is something worth
spending it on.

## Where the current model loses its attention

Measured on the trained checkpoints, one slide, tracing the peaks through each
averaging stage:

| stage | attention max/mean |
|---|---|
| one branch, one subspace | 1.302 |
| averaged over 5 ACMIL branches (measured correlation -0.0005) | 1.152 |
| averaged over 24 nnMIL subspaces (correlation 0.087) | 1.036 |
| what `patch_attention` returns | 1.026 |

A single branch on a single subspace is peaked. Everything after it averages
maps that are close to uncorrelated, and averaging k uncorrelated maps cuts
peak structure by roughly sqrt(k). The branch diversity penalty (`gamma=0.1`)
explicitly rewards making the branches uncorrelated, and they are then
averaged.

The same averaging applies to the pooled bag embedding, not only to the
displayed map. That is consistent with the model scoring close to a mean-pooler
(0.420 against the 0.354 baseline).

---

## Tier 1 - what the measurements point at directly

### 1.1 Score attention on the full feature vector

`gather_window` slices raw contiguous dimensions, `x[..., offset:end]`, and
feeds them to **one shared** `nn.Linear(256, hidden)`. Window position *j* is
feature dim *(offset + j) mod 1536*, so the same weight column serves unrelated
features on different steps. Weights that suit every offset at once are
non-committal ones, which matches the finding that the trained scorer is
indistinguishable from its initialisation.

Standard ABMIL and CLAM project the whole vector through a *learned*
`Linear(D, 256)`. A raw slice is a different operation.

* **Change:** `window = feature_dim`, `stride = feature_dim` - one window over
  all 1536 dims, i.e. plain full-width gated attention. No code change needed;
  `build_window_offsets` already returns a single offset at that setting.
* **Predicts:** attention entropy falls below the random-init control. QWK may
  or may not move; entropy is the outcome being tested.
* **Status: tested, and the diagnosis holds.** Two folds, 15 epochs, bag 512,
  same seeds. Full-width attention scores max/mean **1.535** against a
  random-init control of 1.136; the sliced design scores 1.144 against its own
  control of 1.143, i.e. no better than untrained. Written up in
  [`ENGINEERING.md`](ENGINEERING.md).
* **But not sufficient.** The guard still refuses the map (top 1% holds 1.45%
  of mass, 2.00% required), and CV QWK went 0.432 to 0.401 - noise at two
  folds, but not a gain. Remaining items in this tier are now the live ones,
  and 2.1 matters more than it did, since the score did not move.

### 1.2 If subspace sampling is kept, keep features in their own positions

Should 1.1 show that subspace sampling still helps as a regulariser, sample a
random *mask* over features while keeping dimensionality at 1536, so position
*j* is always feature *j*. That is feature dropout, and it has no aliasing. The
alternative is one scorer per offset, at 24x the scorer parameters.

### 1.3 Stop averaging decorrelated maps for display

Independent of any retraining: `patch_attention` averages 5 branches x 24
subspaces. Options to compare on the existing checkpoints - a single branch and
subspace, the max across them, the branch the classifier weights most, or a
rank aggregate.

* **Predicts:** a map around 1.30 rather than 1.03, from weights that already
  exist.
* **Caveat:** more peaked is not the same as more correct. It still has to beat
  the random-init control, which a single-subspace random scorer also partly
  passes. This is presentation, not a fix for 1.1.

### 1.4 Revisit the diversity penalty

`gamma=0.1` pushes branches apart, and the branches are then averaged for both
the pooled embedding and the map. Sweep `gamma` over {0, 0.02, 0.1} and measure
entropy alongside QWK. ACMIL introduced the penalty against attention
*over-concentration*; the problem here is the opposite, so whether it helps on
this cohort is an open question rather than a given.

### 1.5 Attention entropy penalty, re-tested

An entropy penalty (`losses.lambda_attn_entropy`) is already implemented and
defaults to 0, because when it was tried it made things worse - entropy went
0.9975 to 0.9998 and CV QWK fell to 0.3873.

Worth re-testing, for two reasons. It was tried while 1.1 was still in place,
so the scorer could not learn anything for the penalty to sharpen. And it was
changed alongside three other things at once, so the attribution was never
clean.

Expect it to be weak on its own regardless. Uniform attention is the *maximum*
of the entropy, and the gradient of entropy with respect to the attention
logits is exactly zero at that maximum, so the penalty has least force
precisely where it is needed. It is a tiebreaker for after the scorer can
learn, not a fix by itself.

**Weighted variants, cheapest first:**

* **Plain weighted penalty.** `loss += lambda * normalised_entropy`, sweeping
  lambda over {0.01, 0.05, 0.2}. The knob that already exists.
* **Weight by whether the bag is right.** Apply the penalty only on bags the
  model is currently grading correctly, so it sharpens attention where there is
  signal instead of forcing confident-looking attention onto slides the model
  cannot grade at all.
* **Target a band, not a minimum.** Penalise `(H - H*)^2` for a target H* of
  perhaps 0.7-0.9 normalised, rather than driving H to 0. A hard minimum pushes
  all the mass onto a few tiles, which is the failure ACMIL exists to prevent;
  overshooting into over-concentration would be as useless as the flat map, and
  the audit script would show it as a max/mean in the hundreds.
* **Per-branch instead of per-bag.** Apply it to each branch's own map so a
  single sharp branch is not averaged away before the penalty sees it. This
  interacts with 1.4 and should be tested after it.

---

## Tier 2 - methods from the literature, in fit order

### 2.1 DTFD-MIL pseudo-bags

Splits each slide into several pseudo-bags carrying the slide label, trains a
tier-1 model on those and distils into a tier-2 model. It was designed for
cohorts of this size, and 435 slides with roughly 348 training samples per fold
is the regime it targets. This is the principled form of the
`data.samples_per_slide` idea already sitting in the config, which has never
been run.

* **Predicts:** better generalisation from more effective training samples. It
  does not by itself address the attention finding.

### 2.2 A state-space (Mamba) aggregator

Recent MIL work reports state-space aggregators outperforming transformer ones
and being less prone to overfitting, at linear rather than quadratic cost in
the number of patches - which matters at 3000 patches per slide.

* **Cost to be clear about:** this is a new aggregator, not a knob. It should
  be added as a switchable arm beside ASMIL-Ord and measured on the same folds,
  not swapped in.
* **Predicts:** unknown here. Published gains are mostly on larger cohorts.

### 2.3 Label-noise-robust losses

TCGA grade labels are noisy, and inter-observer agreement on grading is
reported at roughly 0.5-0.7 QWK, so some of the ceiling is the labels rather
than the model. Candidates in increasing order of intrusiveness: label
smoothing on the CORN targets, Generalized Cross Entropy, Symmetric Cross
Entropy.

### 2.4 MixUp

Left out of this project, carried over from v1's decision rather than tested
here. Mixup is reported to improve robustness to corrupted labels, which is the
situation this cohort is in, and feature-space mixing is used in recent MIL
work. Cheap to test, and currently an untested exclusion rather than a finding.

### 2.5 Flash attention

Worth stating plainly: this is a memory and speed optimisation whose output is
mathematically identical to standard attention. It will not change QWK. It
becomes relevant only if a transformer-style aggregator over all 3000 patches
is tried, where it is what makes the quadratic attention affordable. Filed as
an enabler, not an improvement.

---

## Tier 3 - the representation itself

### 3.1 CLS plus mean patch tokens

Features are currently the CLS token alone, 1536-d. Concatenating the mean of
the patch tokens gives 3072-d, and `encoders.py` already supports `cls_mean`
for other encoders in the registry.

* **Cost:** a full re-extraction, the only accelerator-expensive stage.
* **Judge first:** a linear probe on mean-pooled features already reaches
  0.354, so the features carry real signal. Test the aggregator fixes before
  spending quota on new ones.

### 3.2 A second cohort

CPTAC-HNSCC. Single-cohort CV overstates how well any of this transfers, and no
amount of architecture work substitutes for external validation.
