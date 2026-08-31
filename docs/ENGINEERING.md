# Engineering notes

How the TCGA-HNSC cohort was actually extracted and trained on, what broke
along the way, and what that changed about the code. Kept separate from the
main README because it's a retrospective, not a getting-started guide.

---

## Platform: a free Kaggle TPU, measured rather than assumed

Kaggle grants one TPU v5e-8 session at a time (8 XLA devices, 224 vCPU,
~405 GB RAM), two GPU sessions, and five CPU sessions. `/kaggle/working`
persists and caps at ~21 GB; `/kaggle/tmp` is ~1.1 TB but wiped at session
end. These limits, and the throughput numbers below, are recorded in
[`src/pathgrade/platform.py`](../src/pathgrade/platform.py) rather than
asserted in prose, so code that plans around them can import the real values
instead of a comment someone forgot to update.

Measured on the real extraction campaign (not synthetic benchmarks - see
below for why that distinction mattered):

| stage | measured |
|---|---|
| H-optimus-0 encode, one XLA device | 123.5 patches/s |
| GDC download, one stream | ~30 MB/s |
| tile decode | 1900–2500 tiles/s |
| full pipeline, real weights | 73–90 slides/h at 3000 patches/slide |

### Where a slide's time actually goes

Per-stage instrumentation on a real 79-slide chunk, because a plausible guess
about the bottleneck turned out to be wrong twice:

| stage | share of *in-slide* time |
|---|---|
| tissue detection + tiling | 10% |
| H-optimus-0 encode | 89% |
| HDF5 write | 1% |

But in-slide time was only 30–68% of the actual wall clock across chunks. The
rest was the consumer blocked on downloads - not a bandwidth problem
(per-stream throughput held steady at ~30 MB/s throughout) but a structural
one: the download prefetcher's concurrency was tied to how many slides the
consumer had freed from disk, so downloads and encoding barely overlapped.
Raising both the parallel-stream count and the disk budget fixed it; the
first hypothesis (tissue detection) was 10% of nothing, and encode being 89%
of in-slide time didn't matter while in-slide time was a third of the wall.

---

## Two numbers that were wrong, and what that cost

Early micro-benchmarks reported **1226 patches/s** for the encoder and
**1654 tiles/s** for tile decode. Both were artifacts:

- The encoder figure never called `.cpu()` on the output. XLA's lazy tensors
  mean a result that's never materialised never actually runs on the
  device - the benchmark timed graph *construction*, not execution. Real
  throughput on one device is 123.5/s, about 10x lower.
- The decode figure was measured on an 11 MB slide sitting entirely in page
  cache. Real slides run 1900–2500 tiles/s from any mount.

Both numbers looked internally consistent and sent real debugging effort in
the wrong direction before either was caught by running the actual pipeline
end to end. The retained lesson, applied for the rest of the build: **a
throughput number is only real once something confirms the work happened**,
not once a timer stops.

That exact mistake recurred once, at higher stakes. A later probe reported
"8.54x speedup, threads scale" for running the encoder across all 8 TPU
devices from one process - by dividing *intended* work by wall-clock time
without checking whether the work had finished. Seven of eight threads had
actually died mid-run; the true aggregate was slower than a single device.
Same failure shape, same fix: verify completion, not just elapsed time.

---

## Getting real multi-device throughput did not work

A Kaggle TPU v5e-8 exposes eight independent chips, and the pipeline used
one. Two approaches were tried to use the rest:

**Threads across devices**, one process driving all eight XLA devices.
Broken by construction: `mark_step()` synchronises every live tensor *on a
device*, not just the calling thread's, so concurrent threads tear each
other's in-flight state apart. Confirmed on real hardware - two threads ran
cleanly, four and eight both crashed inside the graph executor.

**One process per device**, via `torch_xla`'s standard multiprocess spawn.
This is the documented, supported approach, and it still failed - five
consecutive attempts, each hitting a different failure inside torch_xla's own
topology configuration on this specific Kaggle image (`torch_xla` 2.8.0 on a
`v5litepod-8`, single-host `libtpu`). The most informative failure came from
correcting the environment: clearing what looked like a stale single-process
setting produced a *worse* crash, `AttributeError: 'NoneType' object has no
attribute 'split'` inside torch_xla's own mesh-shape derivation - evidence
that the value being cleared was actually load-bearing, not an obstruction.

Given four independent hypotheses producing four independent failures, this
reads as a genuine defect or an unsupported configuration in this specific
platform combination, not something a sixth guess at an environment variable
was likely to fix. The extraction pipeline runs single-device as a result;
`build_encoders()` in `encoders.py` refuses to attempt cross-device
threading and falls back automatically, with a regression test pinning that
behaviour. The commit history carries the four tracebacks for whoever
revisits this.

---

## Survivability: short, cumulative runs instead of one long one

Two attempts to extract the full 435-slide cohort in a single session were
killed by the platform partway through - one after 86 minutes, one after
3 hours 45 minutes - and both produced **no output at all**: not the
embeddings, not the log, not even a status file that was flushed and
`fsync`'d in the first second of the run. A 34-minute run, by contrast,
committed its output cleanly every time.

Two changes followed from that observation:

- **Extraction runs as a child process**, isolating it from whatever OpenSlide
  or libtiff does when parsing a malformed TIFF from one of several hundred
  different scanners. If the child dies, the parent - and everything already
  written to disk - survives.
- **Extraction is capped to well under an hour per run**, and each run seeds
  its output from the previous run's, mounted as a Kaggle dataset input.
  Extraction was already idempotent (it skips slides already on disk); this
  supplies the other half, so a chain of short kernels accumulates the full
  cohort with no single point of failure large enough to lose real progress.

Every chunk in that chain is also a subprocess launched by the same script
that runs training, with training explicitly skipped on intermediate chunks -
spending fifteen minutes training inside a container with a demonstrated
failure mode would risk the slides that same chunk had just paid to extract.

---

## Smaller lessons, kept because they generalise

- **A guarded stage needs the guard on both halves.** Training was wrapped in
  a broad exception handler from early on, so a training bug could never
  discard already-extracted embeddings. Extraction - the more expensive
  half - didn't have the same guard for longer than it should have.
- **A metric that's only updated on failure lies by omission.** A progress
  heartbeat that only recorded failed slides reported `0 done` after 40
  successful ones, and would have made a genuinely dead run indistinguishable
  from a merely slow one.
- **Validate before you spend a queue slot.** A syntax error, once pushed,
  still waits out a ~40-minute TPU queue before failing in two seconds. The
  push tooling now runs `ast.parse` first.
- **A platform log endpoint returning nothing is not evidence of nothing
  happening.** Kaggle's log API returned zero bytes for runs that had
  completed successfully. The pipeline writes its own append-only,
  `fsync`'d status file for exactly this reason, and every diagnostic here
  trusts that file over the log endpoint.

---

## Diagnosing the first real result

The first full training run scored QWK 0.293 on the locked test set (0.420
five-fold CV). That is weak, so the whole thing was taken apart against the
435 extracted slides, the five fold checkpoints and the saved predictions.

**Ruled out by measurement, not by argument:**

- *Magnification.* Slides at 0.25 µm/px read 444 px regions; slides at
  0.5 µm/px read 224 px. Both land at ~112 µm per tile, so physical scale is
  consistent across scanners.
- *Feature integrity.* All 435 files finite, 1536-d, real H-optimus-0 weights,
  sensible per-slide norms.
- *The ordinal head.* Zero rank-consistency violations across 66 test slides,
  and 0.5 is already the near-optimal threshold - sweeping 0.3–0.7, and
  sweeping expected-grade cuts, finds nothing better.
- *The aggregator versus its own baseline.* A mean-pooled logistic regression
  tuned by nested selection scores 0.251 on the same test set. The MIL model
  scores 0.293. A broken aggregator would land at or below the trivial
  baseline; this one beats it.

**The actual defects:**

**1. Attention collapsed to exactly uniform.** Top 1% of patches hold 1.0% of
the attention mass; normalised entropy 1.0000. The model is mean-pooling, and
the heatmap that is the product's whole explainability story is flat noise.

The mechanism is worth understanding because it is a trap any MIL model on a
small cohort can fall into. The classifier reaches near-zero training loss
(CORN term 0.668 → 0.008) using the bag *mean* alone. Once training loss is
flat, no gradient asks the attention scorer to specialise - and weight decay
keeps pulling on it regardless, until its output layer ends up with smaller
weights than it was initialised with (trained std 0.018 against an init of
0.036). Pre-softmax scores finish with std ≈ 0.1 spread across 3000 patches,
where a softmax needs spread on the order of log N ≈ 8 to concentrate at all.

Crucially, sharpening it afterwards does not rescue it: a temperature sweep on
the test set *lowers* QWK, because there is no learned ranking underneath to
sharpen. Attention entropy is now computed and logged every step so this is
visible while a run is happening, `loss.lambda_attn_entropy` can penalise
uniformity, and `inference.attention_is_informative()` refuses to let a flat
map be displayed as an explanation. The entropy penalty alone is explicitly
*half* a fix - concentrating attention does not make it concentrate on the
*right* patches; the other half is not overfitting in the first place.

**2. The uncertainty score was chance.** Disagreement between the five folds
scores AUC 0.5005 at detecting the model's own errors - mean spread 0.1583 on
correct predictions, 0.1593 on incorrect - while being advertised as flagging
slides for human review. Distance to the CORN decision threshold reaches AUC
0.585 and, used to keep the most decisive half of slides, lifts QWK from 0.293
to 0.472. That is now `GradePrediction.ambiguity`; the fold-spread field is
kept only for provenance.

**3. Validation QWK peaks at epoch 1–20 and then declines on every fold** while
training loss keeps falling. Classic overfitting on ~350 training slides.

**4. The CV number is inflated by ~0.13 QWK.** Best-epoch selection happens on
the validation fold, and that same fold's score is then reported: CV 0.420
against a locked-test 0.293. This is a milder instance of exactly the failure
the evaluation discipline exists to prevent, and the honest number to quote is
the locked-test one.

The through-line: **the accuracy ceiling here is set by the features and labels,
not the architecture**, but the *product* defects (a meaningless heatmap and a
meaningless confidence score) were real, shipped, and are fixed.

---

## What was tried against the weak result, and what it taught

**Attempt 1 — regularise harder and penalise uniform attention.** Bag size
1536→384, weight decay 1e-4→1e-2, dropout 0.25→0.4, attention-entropy penalty
0.5. Result: CV QWK 0.4205→0.3873, test 0.2930→0.2594. Worse on both.

More useful than the number is *why*. The entropy penalty did not merely fail
to help, it did nothing at all: normalised attention entropy went 0.9975 →
**0.9998** with the penalty active — more uniform, not less. Uniform attention
is the **maximum** of entropy, so the gradient of an entropy penalty vanishes
exactly at the collapse. There is no downhill direction out of a stationary
point at any coefficient. That is a property of the objective, not a tuning
problem, and there is now a test computing both gradients to keep the lever
from being reached for again.

The second half of that failure was self-inflicted: raising weight decay 100×
to fight overfitting attacks the very layer that was already being decayed to
death. The scorer's output layer went from std 0.018 (first run) to 0.008,
against an initialisation of ~0.036. And with four variables moved at once, the
regression could not have been attributed even if it had been informative.

The mechanism the evidence actually supports: once the head fits the training
set from the bag mean, **no gradient defends the attention scorer and weight
decay is the only force still acting on it**. So `optim.scorer_no_decay`
exempts it — which, unlike an entropy penalty, has a non-zero gradient at
uniform.

**Attempt 2 — the search that was never run.** v1 ran 35 Optuna trials; v2
shipped its first real number on pure defaults, so "the old model was better"
has never been a like-for-like comparison. `scripts/06_tune.py` searches what
the diagnostic implicated — `scorer_no_decay`, `lr_mult_scorer`, `bag_size`,
`samples_per_slide`, decay, dropout, branch drop, `n_branches`, `hidden`, lr,
batch size, `lambda_qwk` — with `lambda_attn_entropy` deliberately excluded,
since searching a lever that cannot move only wastes trials.

Two disciplines carried into it. The objective is **mean CV QWK and the locked
test set is never opened by the tuner** — tuning against the test set is
exactly how v1's 0.6683 became a selection optimum. And the study is SQLite,
written trial by trial, because a Kaggle container that dies takes anything
not already on disk with it.

**A measurement mistake worth recording**, since it is the third instance of
the same failure mode. Local tuning was estimated at ~14 min/trial from a
synthetic in-RAM benchmark. The real loop was hours per trial, because
`_preload_is_worthwhile()` read `/proc/meminfo`, which does not exist on
Windows, failed closed, and left every sample re-reading and gunzipping an
HDF5 file on every epoch. Disk, not compute — and the same `/proc/meminfo`
assumption had already caused one earlier bug here. The probe is now a real
cross-platform function with a test.

---

## The weight-decay explanation for the attention collapse is wrong

The diagnostic proposed that attention collapsed to uniform because, once the
classifier could fit the training set from the bag mean, no gradient defended
the attention scorer and weight decay was the only force still acting on it.
It was a tidy story and it is not true.

Both arms of the 2x2 ran 20 epochs on identical seeds, differing only in
`optim.scorer_no_decay`:

| arm | CV QWK (per-fold) | final attention entropy |
|---|---|---|
| baseline | 0.474 / 0.359 / 0.463 / 0.544 | 0.999289 |
| scorer exempted from decay | 0.474 / 0.359 / 0.463 / 0.544 | 0.999276 |

Identical to four decimals. The runs are not byte-identical - the final loss
differs around its sixth decimal - so the flag is plumbed through and does
reach the optimiser. The effect is simply negligible.

It was always going to be. AdamW applies decoupled decay as a shrink of
(1 - lr * wd) per step. At lr 5e-4 and wd 1e-4 that is 1 - 5e-8, so across a
whole run of roughly 800 steps the scorer loses about 0.004% of its
magnitude. The collapse this was meant to explain is a fall in the scorer's
output std from 0.036 to 0.008 - a 78% shrink, four orders of magnitude
larger than decay can deliver. The hypothesis could have been killed with a
calculator before an accelerator was booked, which is the lesson worth
keeping: check that a proposed cause is of the right order of magnitude
before spending quota measuring it.

So whatever flattens the scorer arrives through the gradient. That is a
different and harder problem, and it is still open.

A second bug surfaced alongside this. `scripts/07_ablate.py` read the entropy
by iterating the value returned by `train_fold`, which is a summary dict, not
the epoch list - iterating it yields string keys, so the comprehension matched
nothing and every arm reported `attn_entropy_final: null`. The number the
experiment existed to produce was silently absent, and the run would have been
read as "inconclusive" rather than "instrument broken". It now reads
`log["history"]` and prints a warning when a fold yields no entropy at all.

## The attention did not collapse. It never learned.

With the decay explanation dead, the next step was to measure the scorer
directly rather than propose another cause. Five folds x five slides, 25
measurements, comparing the trained checkpoints against a **randomly
initialised model of the same architecture**:

| | per-subspace max/mean | cross-subspace correlation | ensembled max/mean |
|---|---|---|---|
| trained | 1.128 +/- 0.022 | +0.045 +/- 0.033 | 1.034 +/- 0.009 |
| random init | 1.150 | +0.010 | 1.037 |

A randomly initialised attention module produces a *slightly more peaked*
map than the trained one. Whatever training did to the scorer, it did not
make it more informative than chance. The framing in the earlier diagnosis -
that attention "collapsed", implying it was once useful - is wrong. It was
never useful. Pre-softmax score std moves from 0.092 at init to 0.081 after
training; the module is essentially where it started.

**The subspace ensemble is a second, independent problem.** nnMIL scores
attention on a 256-d window of the 1536-d embedding and averages across 24
such windows at inference. That is sound only if different windows agree
about which patches matter. They do not:

* pairwise correlation between the 24 subspace maps: **0.087**
* overlap of their top-30 patches: **1.3%**, against a chance rate of 1.0%

The windows are picking near-independent patch sets, so averaging them is
averaging noise, and it shrinks peak structure by 3.8x - close to the sqrt(24)
= 4.9x expected from averaging decorrelated maps. Per subspace the map reaches
max/mean 1.137; ensembled it reaches 1.036. Crucially this flattening is
present at random initialisation too, so it is architectural, not a training
outcome.

The two failures compound, and neither fix works alone. Sharpening the
aggregation cannot help while each subspace is individually random. Making
the scorer learn cannot help while 24 decorrelated maps are averaged for
display. Any future attempt has to address both, and must be checked against
the random-init control above - a change that improves the map but not past
random initialisation has not achieved anything.

The measurement to keep repeating is that control. It cost one CPU-minute on
checkpoints that already existed, and it invalidated a hypothesis that had
already consumed a TPU session.

## The attention scorer was reading a sixth of each feature vector

`gather_window` takes a raw contiguous slice of the embedding,
`x[..., offset:end]`, and hands it to **one shared** `nn.Linear(256, hidden)`.
Window position *j* is feature dimension *(offset + j) mod 1536*, so the same
weight column serves unrelated features on different steps: dim 0 at one
offset, dim 64 at the next, dim 128 after that. The only weights that suit
every offset simultaneously are non-committal ones, which is exactly the
scorer that was measured - indistinguishable from its own initialisation.

This was a misreading of nnMIL's subspace idea. Standard ABMIL and CLAM score
attention on a *learned* projection of the whole vector, `Linear(D, 256)`. A
raw slice is a different operation with a different failure mode.

Tested by setting `window = stride = feature_dim`, which makes
`build_window_offsets` return a single offset covering all 1536 dims - plain
full-width gated attention, no code change. Two folds, 15 epochs, bag 512, same
seeds, judged by `scripts/08_attention_audit.py` against a randomly initialised
control (20 measurements, 2 folds x 10 slides):

| arm | attention max/mean | random-init control | CV QWK |
|---|---|---|---|
| sliced 256-d windows (current design) | 1.144 | 1.143 | 0.432 |
| full-width 1536-d | **1.535** | 1.136 | 0.401 |

The sliced design does not beat its own untrained control - 1.144 against
1.143. Full-width does, decisively. On one slide the map a viewer would
actually receive goes from max/mean 1.043 to 1.708, roughly twenty times
further from uniform, and with a single window there is no subspace averaging
left to flatten it.

**Two things this did not do, both worth stating.**

It did not produce a usable heatmap. `attention_is_informative` still returns
False: the top 1% of patches hold 1.45% of the attention mass against the 2.00%
the guard requires. Better than 1.04%, still not an explanation.

It did not improve accuracy. CV QWK went 0.432 to 0.401 across two folds at 15
epochs - within the noise of a two-fold comparison, but certainly not a gain.
So this fixes the mechanism that made attention unlearnable without yet moving
the number, and the honest reading is that the flat map and the weak score are
partly separate problems rather than one problem with one cause.

A method note, because it nearly caused a wrong conclusion. Normalised entropy
saturates: at 3000 patches a map with max/mean 1.3 still scores 0.9996, so
entropy alone cannot distinguish a working scorer from a broken one at this bag
size. The two arms read 0.99911 and 0.99699 - a difference easy to dismiss as
rounding, while max/mean showed 1.144 against 1.535. Peak ratio against a
control is the discriminating measurement; entropy is only useful for catching
the exactly-uniform case.

## Attention: what fixed it, and why fixing it did not help the score

The full-width result in the previous section changed two things at once - the
scorer stopped being aliased across offsets, *and* the 24-way subspace
averaging disappeared, since one window means one offset. A third arm,
`window=256, stride=1536`, gives a single fixed offset over 256 dims: no
aliasing, no ensemble, dimensionality unchanged. That splits the effect.

Two folds, 15 epochs, bag 512, identical seeds. Peakedness measured against a
randomly initialised control, 20 measurements per arm:

| arm | dims | offsets | aliased | ensembled | max/mean | control | over control | CV QWK |
|---|---|---|---|---|---|---|---|---|
| sliced-24 (shipped design) | 256 | 24 | yes | yes | 1.144 | 1.143 | **+0.001** | 0.432 |
| sliced-1 | 256 | 1 | no | no | 1.315 | 1.142 | +0.173 | 0.425 |
| full-1 | 1536 | 1 | no | no | 1.535 | 1.136 | +0.399 | 0.401 |

The shipped design is the only one that fails to beat its own untrained
control, by +0.001. Both single-offset variants clear it comfortably. The
total improvement splits about evenly: +0.171 from removing the aliasing and
the ensemble at fixed dimensionality, +0.220 from letting the scorer see all
1536 dims instead of 256.

Separating aliasing from ensembling within that first contrast needs a fourth
arm with one scorer per offset, which is a code change rather than a config
one. It has not been run, so "aliasing" and "averaging 24 decorrelated maps"
remain jointly, not individually, established.

**The part worth sitting with.** QWK moves the *other* way: 0.432, 0.425,
0.401 as peakedness climbs 1.144, 1.315, 1.535. Perfectly monotone across the
three arms, though on two folds each and well inside fold noise, so it is a
pattern rather than a result.

It is the pattern the headroom tests predict. If the mean is close to a
sufficient statistic for grade on this cohort - and three independent methods
say it is - then any departure from uniform weighting adds variance without
adding signal, and should cost a little accuracy. Sharpening the attention is
doing exactly that.

So the fix works and does not pay. The attention module can be made to learn,
which matters because the overlay is only honest if it reflects something the
model found. It should not be expected to raise the score, and on this
evidence a sharper map may cost a little. Those are two different projects and
this repository had been treating them as one.

## What's next

- External validation on a second cohort (CPTAC-HNSCC is the natural one) -
  single-cohort cross-validation overstates real-world deployability.
- Revisit multi-device throughput if a different `torch_xla`/platform
  combination becomes available, or if profiling shows the encode stage
  alone justifies the complexity.
- This is a medical device if shipped as a diagnostic aid: FDA 510(k)/De
  Novo in the US, IVDR Class C in the EU, UKCA in the UK. Out of scope for
  this repository, but not optional before a real release.
