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

## What's next

- External validation on a second cohort (CPTAC-HNSCC is the natural one) -
  single-cohort cross-validation overstates real-world deployability.
- Revisit multi-device throughput if a different `torch_xla`/platform
  combination becomes available, or if profiling shows the encode stage
  alone justifies the complexity.
- This is a medical device if shipped as a diagnostic aid: FDA 510(k)/De
  Novo in the US, IVDR Class C in the EU, UKCA in the UK. Out of scope for
  this repository, but not optional before a real release.
