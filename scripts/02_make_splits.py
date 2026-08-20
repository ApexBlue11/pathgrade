#!/usr/bin/env python
"""Step 2: build patient-level CV folds plus a locked, fingerprinted test set."""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathgrade.data.splits import make_splits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--out", default="data/splits.json")
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260820)
    p.add_argument("--id-col", default="patient_id")
    p.add_argument("--label-col", default="label")
    args = p.parse_args()

    s = make_splits(
        args.labels_csv, args.out, args.test_frac, args.n_folds,
        args.seed, args.id_col, args.label_col,
    )
    dev = {pid for f in s.folds for pid in f["train"] + f["val"]}
    print(f"wrote {args.out}")
    print(f"  test  {len(s.test):4d} patients  (fingerprint {s.fingerprint})")
    print(f"  dev   {len(dev):4d} patients across {len(s.folds)} folds")
    print(f"  class counts: {dict(sorted(Counter(s.labels.values()).items()))}")
    print("\nThe test set is now locked. Use cross-validation for every tuning decision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
