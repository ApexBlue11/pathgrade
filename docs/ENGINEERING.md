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

## What's next

- External validation on a second cohort (CPTAC-HNSCC is the natural one) -
  single-cohort cross-validation overstates real-world deployability.
- Revisit multi-device throughput if a different `torch_xla`/platform
  combination becomes available, or if profiling shows the encode stage
  alone justifies the complexity.
- This is a medical device if shipped as a diagnostic aid: FDA 510(k)/De
  Novo in the US, IVDR Class C in the EU, UKCA in the UK. Out of scope for
  this repository, but not optional before a real release.
