#!/usr/bin/env python
"""Step 7 (optional): a controlled ablation of the fixes the diagnostic implied.

A 60-trial random search was the wrong instrument. One trial costs over an hour
even on a 224-vCPU host, so a full study needs ~90 hours of a 20 h/week
accelerator quota - and it answers "which of 12 knobs matter?" when the
diagnostic had already narrowed it to two specific, testable claims:

1. Attention collapsed to exactly uniform because, once the head could fit the
   training set from the bag mean, no gradient defended the attention scorer
   and weight decay was the only force still acting on it. If that is right,
   exempting the scorer (`optim.scorer_no_decay`) should raise attention
   entropy away from 1.0.

   MEASURED 2026-08-29, and this is WRONG. Both arms were run for 20 epochs
   on identical seeds: attention entropy ended at 0.999289 (baseline) versus
   0.999276 (scorer exempted), and per-fold val QWK was identical to four
   decimals. Exempting the scorer changes the final loss in its sixth decimal.

   The arithmetic says it could never have worked. AdamW's decoupled decay
   shrinks a weight by a factor of (1 - lr*wd) per step; at lr 5e-4 and wd
   1e-4 that is 1 - 5e-8, so over a whole run of ~800 steps the scorer can
   lose about 0.004% of its magnitude. The collapse being explained was a
   drop from std 0.036 to 0.008 - a 78% shrink. Decay is roughly four orders
   of magnitude too weak to be the cause, so whatever is flattening the
   scorer is arriving through the gradient, not through the regulariser.
   Axis 1 of this 2x2 is therefore dead; keep it only as a control.
2. Training saw ~348 samples per fold and drove training loss to ~0. If that is
   the overfitting, drawing several disjoint sub-bags per slide
   (`data.samples_per_slide`) should help generalisation.

Those are independent, so a 2x2 answers both and tells you whether they
interact - four runs instead of sixty, and every cell is interpretable.

Crucially this reports **attention entropy alongside QWK**. A config that
improves QWK while leaving entropy at 1.0 has produced a better mean-pooler,
not a working attention model, and the difference matters because the attention
map is the product.

The locked test set is never read here.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

# name -> the single thing it changes relative to the first real training run
ARMS = {
    "baseline":            {},
    "scorer_no_decay":     {"scorer_no_decay": True},
    "more_samples":        {"samples_per_slide": 4},
    "both":                {"scorer_no_decay": True, "samples_per_slide": 4},
}


def build_cfg(name: str, arm: dict, args):
    from pathgrade.config import Config

    cfg = Config()
    cfg.run_name = f"ablate-{name}"
    cfg.output_dir = args.out
    cfg.encoder = "h-optimus-0"
    cfg.data.feature_dir = args.feature_dir
    cfg.data.labels_csv = args.labels
    cfg.data.splits_path = args.splits
    cfg.optim.epochs = args.epochs
    cfg.optim.num_workers = args.workers
    cfg.optim.amp = False

    # A smaller bag than the 1536 the first run used. That run drew ONE sub-bag
    # of 1536 from ~3000 patches, so any two draws had nearly the same mean and
    # attention had nothing to contribute; a smaller bag is what makes
    # samples_per_slide a meaningful axis rather than a duplicate.
    cfg.data.bag_size = args.bag_size

    cfg.data.samples_per_slide = arm.get("samples_per_slide", 1)
    cfg.optim.scorer_no_decay = arm.get("scorer_no_decay", False)
    return cfg


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", default="runs/ablation")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--bag-size", type=int, default=512)
    p.add_argument("--folds", type=int, default=5, help="fewer folds = cheaper, noisier")
    p.add_argument("--arms", default=",".join(ARMS), help="comma-separated subset to run")
    args = p.parse_args()

    import numpy as np
    import torch
    from pathgrade.data.dataset import feature_dim
    from pathgrade.data.splits import load_splits
    from pathgrade.train import train_fold

    torch.set_num_threads(max(1, (__import__("os").cpu_count() or 4)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}

    for name in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if name not in ARMS:
            print(f"unknown arm {name!r}, skipping", flush=True)
            continue
        cfg = build_cfg(name, ARMS[name], args)
        splits = load_splits(cfg.data.splits_path)
        labels = splits.labels
        dev_ids = sorted({q for f in splits.folds for q in f["train"] + f["val"]})
        if cfg.model.feature_dim is None:
            cfg.model.feature_dim = feature_dim(cfg.data.feature_dir, dev_ids)
        cfg.run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 66}\narm: {name}  {ARMS[name] or '(unchanged)'}\n{'=' * 66}", flush=True)
        qwks, entropies = [], []
        for i, fold in enumerate(splits.folds[: args.folds]):
            log, metrics = train_fold(cfg, i, fold["train"], fold["val"], labels,
                                      torch.device("cpu"))
            qwks.append(float(metrics.qwk))
            # attn_entropy_raw is logged every epoch whether or not the penalty
            # is on; the last epoch's value is where the scorer ended up.
            #
            # train_fold returns a summary dict, not the epoch list - iterating
            # it directly yields string keys, so the old comprehension silently
            # produced nothing and every arm reported attn_entropy_final: null.
            # That is the one number this experiment exists to measure, so a
            # missing value is now loud rather than a null in the output file.
            ent = [e["parts"]["attn_entropy_raw"] for e in log["history"]
                   if isinstance(e, dict) and "attn_entropy_raw" in e.get("parts", {})]
            ent = [x for x in ent if x is not None]
            if ent:
                entropies.append(float(ent[-1]))
            else:
                print("  WARNING: no attn_entropy_raw in this fold's history - "
                      "the entropy column below is not measuring anything.",
                      flush=True)
            print(f"  fold {i}: qwk {metrics.qwk:.4f}"
                  + (f"  attn_entropy {ent[-1]:.4f}" if ent else ""), flush=True)

        results[name] = {
            "cv_qwk_mean": float(np.mean(qwks)), "cv_qwk_std": float(np.std(qwks)),
            "per_fold": qwks,
            "attn_entropy_final": float(np.mean(entropies)) if entropies else None,
            "config": {"bag_size": cfg.data.bag_size,
                       "samples_per_slide": cfg.data.samples_per_slide,
                       "scorer_no_decay": cfg.optim.scorer_no_decay,
                       "epochs": cfg.optim.epochs},
        }
        (out / "ablation.json").write_text(json.dumps(results, indent=2))
        print(f"  -> CV QWK {results[name]['cv_qwk_mean']:.4f}", flush=True)

    print("\n" + "=" * 78)
    print(f"{'arm':<20}{'CV QWK':>10}{'std':>8}{'attn entropy':>15}   (1.0 = uniform)")
    print("=" * 78)
    for name, r in results.items():
        ent = "n/a" if r["attn_entropy_final"] is None else f"{r['attn_entropy_final']:.4f}"
        print(f"{name:<20}{r['cv_qwk_mean']:>10.4f}{r['cv_qwk_std']:>8.4f}{ent:>15}")
    print("\nreference: first real run scored CV 0.4205 / locked test 0.2930 with")
    print("attention entropy 1.0000 - exactly uniform, i.e. mean-pooling.")
    print("An arm that lifts QWK but leaves entropy at 1.0 is a better mean-pooler.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
