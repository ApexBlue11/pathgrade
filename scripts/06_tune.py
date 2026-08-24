#!/usr/bin/env python
"""Step 6 (optional): tune hyperparameters on the CV folds with Optuna.

The first real training run used entirely default hyperparameters - zero
tuning trials - and scored QWK 0.293 on the locked test set. The predecessor
project ran 35 Optuna trials and reported 0.6683, but reported it *on the same
set those trials selected against*, which is how a selection optimum is
manufactured. This script does the tuning that was missing without repeating
that mistake.

**The locked test set is never loaded here.** The objective is the mean QWK
across the 5 CV folds, and `evaluate.py` still requires `--unlock` to touch the
test set afterwards. Tune here, then evaluate ONCE. If you find yourself
re-running the test set to compare tuned configs, the number stops meaning
anything - that is precisely what happened to v1.

Runs on CPU against already-extracted features, so it costs no accelerator
quota. Measured ~14 min/trial at bag_size 384 on 12 cores; smaller bags and
`--epochs` cut that further while searching.

    python scripts/06_tune.py --feature-dir runs/c8-final/features \
        --splits runs/c8-final/splits.json --labels runs/c8-final/labels_available.csv \
        --trials 40

The study is persisted to SQLite, so it resumes if interrupted.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")


def build_cfg(trial, args):
    """Sample one configuration. Search space chosen from the diagnostic.

    Every knob here is one the diagnostic implicated:

    * ``scorer_no_decay`` / ``lr_mult_scorer`` - attention collapsed to exactly
      uniform in both real runs because weight decay was the only force acting
      on the scorer once the head could fit the data from the bag mean.
    * ``bag_size`` / ``samples_per_slide`` - training saw ~348 samples per fold
      and drove training loss to ~0. Smaller bags make the bag mean a noisier
      target; more sub-bags per slide multiply gradient steps.
    * ``weight_decay`` / ``dropout`` / ``branch_drop`` - the overfitting itself.
    * ``lambda_qwk`` - the metric relaxation, never tuned.

    ``lambda_attn_entropy`` is deliberately NOT searched: it is measured to be
    incapable of un-collapsing attention, because uniform is the maximum of
    entropy and the gradient vanishes there.
    """
    from pathgrade.config import Config

    cfg = Config()
    cfg.run_name = f"tune-{trial.number:03d}"
    cfg.output_dir = args.out
    cfg.encoder = "h-optimus-0"
    cfg.data.feature_dir = args.feature_dir
    cfg.data.labels_csv = args.labels
    cfg.data.splits_path = args.splits

    cfg.data.bag_size = trial.suggest_categorical("bag_size", [256, 384, 512, 768, 1536])
    cfg.data.samples_per_slide = trial.suggest_int("samples_per_slide", 1, 6)

    cfg.model.dropout = trial.suggest_float("dropout", 0.1, 0.6)
    cfg.model.branch_drop = trial.suggest_float("branch_drop", 0.0, 0.7)
    cfg.model.hidden = trial.suggest_categorical("hidden", [128, 256, 384])
    cfg.model.n_branches = trial.suggest_int("n_branches", 2, 8)

    cfg.optim.lr = trial.suggest_float("lr", 5e-5, 3e-3, log=True)
    cfg.optim.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    cfg.optim.scorer_no_decay = trial.suggest_categorical("scorer_no_decay", [True, False])
    cfg.optim.lr_mult_scorer = trial.suggest_float("lr_mult_scorer", 0.5, 8.0, log=True)
    cfg.optim.batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    cfg.optim.epochs = args.epochs
    cfg.optim.num_workers = 0          # threads fight the CPU training loop
    cfg.optim.amp = False

    cfg.loss.lambda_qwk = trial.suggest_float("lambda_qwk", 0.0, 0.6)
    cfg.loss.lambda_attn_entropy = 0.0
    return cfg


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", default="runs/tuning")
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--study", default="runs/tuning/study.db")
    p.add_argument("--timeout-hours", type=float, default=None,
                   help="stop launching new trials after this long and write results. "
                        "Kaggle containers have been killed mid-run without committing "
                        "output, so a bounded study that exits cleanly beats an "
                        "open-ended one that vanishes")
    args = p.parse_args()

    import optuna
    import torch
    from pathgrade.train import run_cv

    torch.set_num_threads(max(1, (__import__("os").cpu_count() or 4)))
    Path(args.study).parent.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        cfg = build_cfg(trial, args)
        try:
            summary = run_cv(cfg)
        except Exception as e:                      # a bad corner of the space
            print(f"trial {trial.number} failed: {type(e).__name__}: {e}", flush=True)
            raise optuna.TrialPruned() from e
        qwk = float(summary["qwk"]["mean"])
        # Report attention entropy too: a config that scores well while leaving
        # attention uniform has produced a mean-pooler, which is worth knowing
        # even when the metric looks fine.
        trial.set_user_attr("qwk_std", float(summary["qwk"]["std"]))
        trial.set_user_attr("per_fold", summary["qwk"]["per_fold"])
        return qwk

    study = optuna.create_study(
        direction="maximize",
        study_name="pathgrade-asmil-ord",
        storage=f"sqlite:///{args.study}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=20260820),
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False,
                   timeout=None if args.timeout_hours is None else args.timeout_hours * 3600)

    print("\n" + "=" * 70)
    print(f"best CV QWK {study.best_value:.4f}  (baseline, untuned: 0.4205)")
    print("=" * 70)
    for k, v in sorted(study.best_params.items()):
        print(f"  {k:<20} {v}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "best_params.json").write_text(
        json.dumps({"cv_qwk": study.best_value, "params": study.best_params,
                    "n_trials": len(study.trials)}, indent=2))
    print(f"\nwritten to {Path(args.out) / 'best_params.json'}")
    print("The locked test set has NOT been touched. Evaluate it once, with the")
    print("winning config, using scripts/04_evaluate_test.py --unlock.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
