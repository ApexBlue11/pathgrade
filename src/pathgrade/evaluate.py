"""Locked test-set evaluation. The only module permitted to read ``splits.test``.

Running this consumes the test set. Every time it runs and a decision is made
on the result, that set becomes a little more like a validation set, so the
CLI requires ``--unlock`` to make the act deliberate rather than incidental.

The fold models are ensembled by averaging cumulative probabilities. Unlike the
previous project's three same-split seeds, the folds were trained on genuinely
different subsets, so the ensemble reflects real disagreement rather than three
runs of the same experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data.io import find_feature_file
from .data.splits import load_splits
from .metrics import compute_metrics, contextualise, format_confusion
from .models.asmil_ord import ASMILOrd
from .train import predict_slide_full


def load_fold_models(run_dir: Path, device) -> tuple[list[ASMILOrd], Config]:
    checkpoints = sorted(run_dir.glob("fold*/checkpoint.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No fold checkpoints under {run_dir}. Train first.")

    models, cfg = [], None
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = Config.from_dict(ckpt["config"])
        model = ASMILOrd(
            feature_dim=cfg.model.feature_dim, n_classes=cfg.data.n_classes,
            window=cfg.model.window, stride=cfg.model.stride, hidden=cfg.model.hidden,
            n_branches=cfg.model.n_branches, dropout=cfg.model.dropout,
            branch_drop=cfg.model.branch_drop, ema_decay=cfg.model.ema_decay,
            feature_norm=cfg.model.feature_norm,
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models.append(model)
    return models, cfg


@torch.no_grad()
def ensemble_predict(models, ids, feature_dir, bag_size, device):
    """Returns cumulative probs [N, K-1] and per-slide disagreement across folds."""
    all_cums, spreads = [], []
    for pid in ids:
        path = find_feature_file(feature_dir, pid)
        per_model = torch.stack([
            predict_slide_full(m, path, bag_size, device) for m in models
        ])                                                    # [M, K-1]
        all_cums.append(per_model.mean(dim=0))
        spreads.append(per_model.std(dim=0).mean())
    return torch.stack(all_cums).numpy(), torch.stack(spreads).numpy()


def evaluate_run(run_dir: str | Path, bootstrap: int = 2000, device=None) -> dict:
    run_dir = Path(run_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models, cfg = load_fold_models(run_dir, device)
    splits = load_splits(cfg.data.splits_path)
    test_ids = splits.test

    cums, spreads = ensemble_predict(
        models, test_ids, cfg.data.feature_dir, cfg.data.bag_size, device
    )
    preds = (cums > 0.5).sum(axis=1)
    truth = np.array([splits.labels[pid] for pid in test_ids])

    metrics = compute_metrics(truth, preds, cfg.data.n_classes, bootstrap=bootstrap)

    result = {
        "run": str(run_dir),
        "n_models": len(models),
        "test_fingerprint": splits.fingerprint,
        "n_test": len(test_ids),
        "metrics": {
            "qwk": metrics.qwk, "qwk_ci95": metrics.qwk_ci,
            "macro_f1": metrics.macro_f1, "balanced_accuracy": metrics.balanced_accuracy,
            "accuracy": metrics.accuracy, "adjacent_accuracy": metrics.adjacent_accuracy,
            "mean_absolute_error": metrics.mean_absolute_error,
            "per_class_f1": metrics.per_class_f1, "confusion": metrics.confusion,
        },
        "mean_fold_disagreement": float(spreads.mean()),
    }

    with open(run_dir / "test_results.json", "w") as f:
        json.dump(result, f, indent=2)
    np.savez(
        run_dir / "test_predictions.npz",
        patient_ids=np.array(test_ids), cumulative=cums, pred=preds,
        true=truth, fold_spread=spreads,
    )

    print("=" * 72)
    print(f"LOCKED TEST SET  |  {len(test_ids)} patients  |  fingerprint {splits.fingerprint}")
    print(f"ensemble of {len(models)} fold models")
    print("=" * 72)
    print(metrics.summary())
    print()
    print(format_confusion(metrics.confusion, cfg.data.class_names))
    print()
    print("per-class F1: " + ", ".join(
        f"{n} {v:.3f}" for n, v in zip(cfg.data.class_names, metrics.per_class_f1)
    ))
    print(f"mean fold disagreement: {spreads.mean():.4f}  "
          f"(high values flag slides the ensemble is unsure about)")
    print()
    print(contextualise(metrics.qwk))
    print("=" * 72)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a trained run on the locked test set.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument(
        "--unlock", action="store_true",
        help="required. Consuming the test set is a one-way door: every look at it "
             "that informs a decision converts it into a validation set.",
    )
    args = p.parse_args(argv)

    if not args.unlock:
        p.error(
            "Refusing to read the locked test set without --unlock. "
            "Use cross-validation numbers for model selection."
        )
    evaluate_run(args.run_dir, bootstrap=args.bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
