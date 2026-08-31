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

## The finding that reorders everything below

The trained model's predictions are **identical** to mean pooling.

Replacing the learned attention with a uniform 1/N weighting at inference -
same weights, same head, same slides, same chunking - changes nothing:

| model | learned attention | uniform pooling | paired delta | identical predictions |
|---|---|---|---|---|
| c8-final, 5 folds, bag 1536 | 0.4205 | 0.4205 | +0.0000 | 100.0% of slides |
| full-width arm, 2 folds, bag 512 | 0.4012 | 0.3996 | +0.0016 | 95.3% of slides |

The reported CV 0.4205 and locked-test 0.2930 are what mean pooling with this
head achieves. Five branches, 24 subspaces, the EMA anchor, the KL
stabilisation and the diversity penalty contribute exactly zero to accuracy.

The second row is the important one. That arm's attention *does* learn - it
beats its random-init control 1.535 to 1.136 - and it still moves QWK by
0.0016. So a working attention module and an accurate model are close to
independent goals on this cohort, and no amount of fixing the attention is
going to move the score on its own.

This splits the project cleanly in two, and the split is worth keeping in mind
for everything below:

* **The map** is an explainability problem. Tier 1 addresses it. Fixing it
  makes the overlay mean something; it will not raise QWK.
* **The score** is a different problem, and attention is not the lever. Tier 2
  and Tier 3 address it.

### What is being tested next, and what would falsify it

**Hypothesis A.** Grade is a diffuse, whole-slide property on this cohort, so
the mean is close to a sufficient statistic and patch selection has little to
offer any aggregator.

*Test:* linear probes over label-free pooling statistics - mean, max, p90,
std, and the mean of the top and bottom 10% of patches by embedding norm -
under the same 5-fold CV, paired per fold.

*Predicts:* if A holds, none of them beats the mean by more than fold noise.
An earlier partial result is consistent with A: mean scores 0.354 and mean
concatenated with std scores 0.274, i.e. adding a second statistic made it
worse rather than better.

*Falsified if:* any statistic beats the mean by a margin that holds up paired
across folds. That would show exploitable patch-level structure and put
attention back on the table as an accuracy lever.

*Result: not falsified.* 369 dev slides, 5-fold CV, logistic regression,
paired per fold against the mean:

| pooling | CV QWK | paired delta | folds improved | p |
|---|---|---|---|---|
| mean (baseline) | 0.3543 | -- | -- | -- |
| mean + top-10% by norm | 0.3688 | +0.0142 | 2/5 | 0.71 |
| top-10% by norm | 0.3246 | -0.0300 | 2/5 | 0.48 |
| mean + max | 0.3215 | -0.0328 | 1/5 | 0.16 |
| p90 | 0.3187 | -0.0358 | 2/5 | 0.35 |
| mean + p90 | 0.3074 | -0.0470 | 1/5 | 0.09 |
| std | 0.3055 | -0.0490 | 2/5 | 0.23 |
| max | 0.2895 | -0.0648 | 1/5 | 0.18 |
| bottom-10% by norm | 0.2735 | -0.0810 | 1/5 | 0.14 |

Every alternative is worse than the mean. The only nominal gain is sign-
inconsistent - 2 of 5 folds - and nowhere near separable from noise. Adding a
statistic to the mean generally *hurts*, which is what a small-sample
overfitting penalty looks like when the extra dimensions carry no signal.

**Limitation, and why this is not yet conclusive.** These rankings are
label-free: patches are ordered by embedding norm. A real attention module
ranks by a discriminative criterion it learned. So this rules out patch
selection by a generic saliency proxy, not patch selection in general.

**Hypothesis A2, closing that gap.** Fit an explicit instance-level classifier
on patches carrying their slide's label, rank each slide's patches by that
discriminative score, and pool the top of the ranking - attention constructed
the most direct way available, and the "a good instance classifier is all you
need" formulation. Fit on training folds only, inside the same CV.

*Predicts:* if A holds, a learned ranking does not beat the mean either.

*Falsified if:* it does, paired across folds - which would mean the signal is
there and the deep model simply failed to find it, a very different conclusion
from the ceiling being elsewhere.

*Result: not falsified.* 369 slides, 128 sampled patches each, 5-fold CV, the
instance classifier fit on training folds only:

| pooling | CV QWK | paired delta | folds improved | p |
|---|---|---|---|---|
| mean | 0.3468 | -- | -- | -- |
| instance softmax (soft attention) | 0.3474 | **+0.0006** | 2/5 | 0.986 |
| instance top-25% | 0.2526 | -0.0942 | 1/5 | 0.130 |
| instance top-10% | 0.2595 | -0.0873 | 2/5 | 0.397 |

A learned, explicitly discriminative ranking does not beat the mean. Weighting
softly by that ranking lands on the mean's score to within 0.0006. Selecting
hard on it is substantially worse, and worse the harder the selection.

The mean baseline reads 0.3468 here against 0.3543 in the previous table
because this run pools 128 sampled patches per slide rather than all ~3000;
the two are consistent, and every comparison above is paired within its own
run.

Fold-to-fold spread is worth seeing, because it is the reason single-fold
impressions were untrustworthy: instance top-10% scored 0.271, 0.083, 0.093,
0.482, 0.369 across the five folds. Three folds say it is far worse than the
mean and one says it is far better.

## Conclusion on Hypothesis A, from three independent directions

1. The trained deep model's predictions are **identical** to mean pooling -
   all 5 folds, 100% of slides, delta exactly 0.0000.
2. No **label-free** pooling statistic beats the mean; most are clearly worse,
   and adding one to the mean generally hurts.
3. No **learned discriminative** ranking beats the mean either. Soft weighting
   equals it to +0.0006 (p = 0.99); hard selection is worse.

On this cohort, with these features, patch selection has essentially nothing
to offer grading accuracy. Grade behaves as a diffuse, whole-slide property
and the mean is close to a sufficient statistic for it.

**What follows for the queue.** Tier 1 is still worth doing, but it is an
explainability project: it decides whether the overlay means anything, and it
will not move QWK. Anything aimed at the score has to act somewhere other than
the pooling - the head and loss (2.3, 2.4), the amount of usable training
signal (2.1), or the features themselves (3.1). Note the head is already
contributing: a logistic regression on mean-pooled features scores 0.354 while
the full model scores 0.420, so +0.066 is coming from the MLP, the CORN
formulation and the training recipe rather than from attention.

**The honest limit of this conclusion.** The instance classifier is linear,
trained on labels propagated from slide to patch, over 128 sampled patches per
slide. A stronger instance model on more patches could in principle find
structure this one missed. What makes the conclusion solid is not any single
line above but that three methods with different failure modes agree, one of
which is the trained network itself.

**Hypothesis B (the confound in the result above).** The full-width arm changed
two things at once - the scorer stopped being aliased across offsets *and* the
24-way subspace averaging disappeared, since one window means one offset. The
1.144 to 1.535 improvement cannot be attributed to de-aliasing alone.

*Test:* a third arm, `window=256, stride=1536`, which yields a single offset at
position 0. The scorer sees a fixed 256 dims - no aliasing, and no ensemble
either. That decomposes the total effect into two contrasts:

| contrast | isolates |
|---|---|
| sliced-24 to sliced-1 | aliasing plus ensembling, dimensionality held at 256 |
| sliced-1 to full-1 | dimensionality 256 to 1536, no aliasing either side |

*Predicts:* if aliasing is the cause, `sliced-1` lands near `full-1` (~1.5) and
well above its control. If the ensemble averaging was the cause, `sliced-1`
stays near `sliced-24` (~1.14).

*Note:* fully separating aliasing from ensembling needs a fourth arm with
per-offset scorers, which is a code change rather than a config one. Deferred
until the first two contrasts say whether it is worth it.

*Result.* Both contrasts are real and roughly equal:

| arm | dims | offsets | max/mean | control | over control | CV QWK |
|---|---|---|---|---|---|---|
| sliced-24 (shipped) | 256 | 24 | 1.144 | 1.143 | **+0.001** | 0.432 |
| sliced-1 | 256 | 1 | 1.315 | 1.142 | +0.173 | 0.425 |
| full-1 | 1536 | 1 | 1.535 | 1.136 | +0.399 | 0.401 |

+0.171 from removing aliasing and the ensemble at fixed dimensionality, +0.220
from full-width. The shipped design is the only arm that does not beat its own
control. Aliasing and ensembling are jointly established, not separately - the
per-offset-scorer arm would be needed for that, and given that attention is
not the accuracy lever it is hard to justify.

QWK declines monotonically as peakedness rises (0.432, 0.425, 0.401), which is
what Hypothesis A predicts: if the mean is near-sufficient, departing from
uniform weighting adds variance without signal. Two folds per arm, so a
pattern rather than a result.

**On statistical power, since the last comparison was underpowered.** Fold-to-
fold QWK sd is roughly 0.09, so a two-fold arm comparison cannot resolve
differences below about 0.1 - the 0.432 vs 0.401 gap reported earlier is not
separable from noise on its own. Two things follow. Comparisons are paired on
identical folds and seeds, and per-fold deltas are reported rather than only
means, since a consistent sign across folds is informative even when the mean
difference is small. And attention peakedness, not QWK, is the primary outcome
for Tier 1 - it is measured against a control and has a far larger effect size
(1.14 vs 1.54) than anything QWK is going to show at this sample size.

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
