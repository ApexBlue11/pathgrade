#!/usr/bin/env python
"""Pull a finished Kaggle run down and report whether it actually succeeded.

Run this after an unattended session:

    python kaggle/collect.py apexblue/pathgrade-pipeline

A COMPLETE status is not sufficient evidence of success. The pipeline
deliberately swallows training failures so that hours of embeddings survive a
training bug, exiting 0 with a TRAINING_FAILED.txt marker. This checks for that
marker, and for the artefacts that should exist, before saying anything worked.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

EXPECTED = ["MANIFEST.json", "cv_summary.json", "test_results.json", "splits.json"]


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return (proc.stdout or "") + (proc.stderr or "")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ref", nargs="?", default="apexblue/pathgrade-pipeline")
    p.add_argument("--out", default="runs/kaggle")
    args = p.parse_args()

    state = re.search(r"KernelWorkerStatus\.([A-Z]+)", run(["kaggle", "kernels", "status", args.ref]))
    state = state.group(1) if state else "UNKNOWN"
    print(f"{args.ref}: {state}")
    if state not in ("COMPLETE", "ERROR"):
        print("still running - come back later")
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"downloading to {out} ...")
    print(run(["kaggle", "kernels", "output", args.ref, "-p", str(out)])[-400:])

    # The marker that matters more than the exit status.
    failed = list(out.rglob("TRAINING_FAILED.txt"))
    if failed:
        print("\n" + "!" * 66)
        print("TRAINING FAILED - embeddings are intact, the model is not")
        print("!" * 66)
        print(failed[0].read_text()[:2000])
        return 1

    features = list(out.rglob("features/*.h5")) + list(out.rglob("features/*.pt"))
    print(f"\nfeature files: {len(features)}")

    manifests = list(out.rglob("MANIFEST.json"))
    if not manifests:
        missing = [n for n in EXPECTED if not list(out.rglob(n))]
        print(f"no MANIFEST.json. Missing artefacts: {missing}")
        print("The run did not reach the release stage.")
        return 1

    m = json.loads(manifests[0].read_text())
    cv, test = m.get("cv", {}), m.get("test", {}).get("metrics", {})
    print("\n" + "=" * 66)
    print("RESULTS")
    print("=" * 66)
    print(f"  encoder          {m.get('encoder')} ({m.get('encoder_licence')})")
    print(f"  architecture     {m.get('architecture')}, {m.get('trainable_params'):,} params")
    cohort = m.get("cohort", {})
    print(f"  cohort           {cohort.get('n_usable')} slides, "
          f"classes {cohort.get('class_distribution')}")
    print(f"  patches          {cohort.get('total_patches'):,} "
          f"(median {cohort.get('median_patches')}/slide)")
    if cv:
        print(f"  CV QWK           {cv['qwk']['mean']:.4f} +/- {cv['qwk']['std']:.4f}")
        print(f"  CV macro-F1      {cv['macro_f1']['mean']:.4f}")
    if test:
        ci = test.get("qwk_ci95") or ["?", "?"]
        print(f"  TEST QWK         {test['qwk']:.4f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
        print(f"  TEST macro-F1    {test['macro_f1']:.4f}")
        print(f"  TEST adj-acc     {test['adjacent_accuracy']:.4f}")
    print(f"  elapsed          {m.get('elapsed_hours')} h")

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from pathgrade.metrics import contextualise
        if test.get("qwk") is not None:
            print("\n  " + contextualise(test["qwk"]))
    except Exception:
        pass
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
