"""Cross-validated training for ASMIL-Ord.

Model selection happens on CV folds only. The locked test set defined in
``splits.json`` is never touched here - see :mod:`pathgrade.evaluate`, which is
the only entry point permitted to read it.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augment import bag_mixup
from .callbacks import EarlyStopping, WeightEMA, suggest_ema_decay
from .config import Config
from .data.dataset import (
    SlideBagDataset,
    SlideCropIterator,
    balanced_sampler,
    feature_dim,
    suggest_bag_size,
)
from .data.io import find_feature_file, verify_cohort
from .data.splits import load_splits
from .encoders import check_licence
from .losses import ASMILOrdLoss, corn_cumulative_probs
from .metrics import GradingMetrics, aggregate_folds, compute_metrics, format_confusion
from .models.asmil_ord import ASMILOrd


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_param_groups(model, cfg: Config) -> list[dict]:
    """Discriminative learning rates by module role.

    v1 used four LR tiers, largely to nurse its 1024->256 projection layer. With
    no projection there is far less to balance, so the multipliers default to
    1.0 and this collapses to a single group. They are exposed because the
    classifier head sits on a differently-scaled input than the attention
    scorer, and on a small cohort that can matter - but it should be turned on
    by CV evidence, not by default.
    """
    m = cfg.optim
    buckets: dict[str, list] = {"head": [], "scorer": [], "norm": [], "other": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("head"):
            buckets["head"].append(param)
        elif name.startswith("online"):
            buckets["scorer"].append(param)
        elif name.startswith("norm"):
            buckets["norm"].append(param)
        else:
            buckets["other"].append(param)

    mult = {
        "head": m.lr_mult_head,
        "scorer": m.lr_mult_scorer,
        "norm": m.lr_mult_norm,
        "other": 1.0,
    }
    return [
        {"params": params, "lr": m.lr * mult[key], "name": key}
        for key, params in buckets.items()
        if params
    ]


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Linear warmup then cosine decay to 1% of peak."""
    warmup = max(1, cfg.optim.warmup_epochs * steps_per_epoch)
    total = max(warmup + 1, cfg.optim.epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_loader(model, loader, device, n_offsets: int | None = None):
    """Fast path: one deterministic sub-bag per slide."""
    model.eval()
    cums, ys, pids = [], [], []
    for batch in loader:
        out = model(batch["features"].to(device), batch["mask"].to(device), n_offsets=n_offsets)
        cums.append(corn_cumulative_probs(out.logits).float().cpu())
        ys.append(batch["label"])
        pids.extend(batch["patient_id"])
    return torch.cat(cums), torch.cat(ys), pids


@torch.no_grad()
def predict_slide_full(model, path, bag_size: int, device, n_offsets: int | None = None):
    """Full coverage: every patch used exactly once, cumulative probs averaged.

    Averaging *cumulative* probabilities rather than logits keeps the result
    rank-consistent - the mean of monotone sequences is monotone.
    """
    model.eval()
    cums = []
    for feats, mask in SlideCropIterator(path, bag_size):
        out = model(feats.unsqueeze(0).to(device), mask.unsqueeze(0).to(device), n_offsets=n_offsets)
        cums.append(corn_cumulative_probs(out.logits)[0].float().cpu())
    return torch.stack(cums).mean(dim=0)


def evaluate_full(model, ids, labels, feature_dir, bag_size, device, n_classes=3, bootstrap=0):
    cums = torch.stack([
        predict_slide_full(model, find_feature_file(feature_dir, pid), bag_size, device)
        for pid in ids
    ])
    preds = (cums > 0.5).sum(dim=1).numpy()
    truth = np.array([labels[pid] for pid in ids])
    metrics = compute_metrics(truth, preds, n_classes, bootstrap=bootstrap)
    return metrics, cums.numpy(), preds, truth


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_fold(cfg: Config, fold: int, train_ids, val_ids, labels, device) -> tuple[dict, GradingMetrics]:
    set_seed(cfg.seed + fold)
    d = cfg.data

    train_ds = SlideBagDataset(train_ids, labels, d.feature_dir, d.bag_size, train=True,
                               samples_per_slide=d.samples_per_slide)
    val_ds = SlideBagDataset(val_ids, labels, d.feature_dir, d.bag_size, train=False)

    # Preloading removes the IO that worker processes exist to hide, and each
    # fork would otherwise copy the cache. Fall back to workers when streaming.
    n_workers = 0 if train_ds._cache is not None else cfg.optim.num_workers
    if train_ds._cache is not None:
        gb = sum(v.nbytes for v in train_ds._cache.values()) / 1e9
        print(f"  fold {fold}: {gb:.2f} GB of features held in RAM, dataloader workers off")

    sampler = balanced_sampler(train_ds.label_list, seed=cfg.seed + fold) if cfg.optim.balanced_sampling else None
    train_loader = DataLoader(
        train_ds, batch_size=cfg.optim.batch_size, sampler=sampler,
        shuffle=sampler is None, num_workers=n_workers,
        pin_memory=device.type == "cuda", drop_last=len(train_ds) > cfg.optim.batch_size,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.optim.batch_size, shuffle=False,
        num_workers=0 if val_ds._cache is not None else cfg.optim.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ASMILOrd(
        feature_dim=cfg.model.feature_dim, n_classes=d.n_classes, window=cfg.model.window,
        stride=cfg.model.stride, hidden=cfg.model.hidden, n_branches=cfg.model.n_branches,
        dropout=cfg.model.dropout, branch_drop=cfg.model.branch_drop,
        ema_decay=cfg.model.ema_decay, feature_norm=cfg.model.feature_norm,
    ).to(device)

    criterion = ASMILOrdLoss(d.n_classes, cfg.loss.beta, cfg.loss.gamma,
                             cfg.loss.lambda_qwk, cfg.loss.lambda_attn_entropy)
    # LambdaLR scales each group's own base_lr, so per-group multipliers survive
    # the warmup + cosine schedule.
    optimizer = torch.optim.AdamW(
        build_param_groups(model, cfg), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    scheduler = build_scheduler(optimizer, cfg, max(1, len(train_loader)))
    use_amp = cfg.optim.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_steps = max(1, len(train_loader)) * cfg.optim.epochs
    ema = None
    if cfg.optim.use_ema:
        ema = WeightEMA(
            model,
            decay=suggest_ema_decay(total_steps, cfg.optim.ema_span_fraction),
            start_step=max(1, len(train_loader)) * cfg.optim.warmup_epochs,
        )

    stopper = EarlyStopping(
        patience=cfg.optim.patience,
        min_delta=cfg.optim.min_delta,
        mode="max",
        smooth_window=cfg.optim.smooth_window,
        plateau_window=cfg.optim.plateau_window,
        plateau_slope=cfg.optim.plateau_slope,
        min_epochs=cfg.optim.min_epochs,
    )

    best = {"qwk": -2.0, "epoch": -1, "state": None}
    history, stop_reason = [], "completed all epochs"

    for epoch in range(1, cfg.optim.epochs + 1):
        model.train()
        epoch_loss, parts_acc, n_batches = 0.0, {}, 0
        for batch in train_loader:
            x = batch["features"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            cum_targets = None
            if cfg.loss.mixup_prob > 0:
                x, m, cum_targets = bag_mixup(
                    x, m, y, d.n_classes,
                    alpha=cfg.loss.mixup_alpha, prob=cfg.loss.mixup_prob,
                )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x, m)
                loss, parts = criterion(out.logits, y, out.aux, cum_targets=cum_targets)

            scaler.scale(loss).backward()
            if cfg.optim.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model.update_anchor()          # EMA anchor tracks the online scorer
            if ema is not None:
                ema.update(model)

            epoch_loss += float(loss.detach())
            for k, v in parts.items():
                parts_acc[k] = parts_acc.get(k, 0.0) + v
            n_batches += 1

        n_batches = max(n_batches, 1)
        cums, ys, _ = predict_loader(model, val_loader, device, n_offsets=cfg.optim.eval_offsets)
        preds = (cums > 0.5).sum(dim=1).numpy()
        m_val = compute_metrics(ys.numpy(), preds, d.n_classes)

        decision = stopper.step(m_val.qwk, epoch)
        if decision.improved:
            best = {
                "qwk": m_val.qwk,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

        history.append({
            "epoch": epoch,
            "loss": epoch_loss / n_batches,
            "parts": {k: v / n_batches for k, v in parts_acc.items()},
            "val_qwk": m_val.qwk,
            "val_qwk_smoothed": decision.smoothed,
            "val_macro_f1": m_val.macro_f1,
            "trend_slope": decision.slope,
            "lr": scheduler.get_last_lr()[0],
        })

        print(
            f"  fold {fold} ep {epoch:03d}/{cfg.optim.epochs} "
            f"loss {epoch_loss / n_batches:.4f} | QWK {m_val.qwk:.4f} "
            f"(smooth {decision.smoothed:.4f}) F1 {m_val.macro_f1:.4f} | {stopper.status()}"
        )

        if decision.stop:
            stop_reason = decision.reason
            print(f"  fold {fold}: early stop at epoch {epoch} - {stop_reason}")
            break

    # Best-epoch weights vs. the EMA average: pick the winner on this fold
    # rather than assuming either is better.
    model.load_state_dict(best["state"])
    m_final, cums, preds, truth = evaluate_full(
        model, val_ids, labels, d.feature_dir, d.bag_size, device, d.n_classes
    )
    selected, final_state = "best-epoch", best["state"]

    if ema is not None:
        ema_state = ema.state_dict()
        model.load_state_dict(ema_state)
        m_ema, cums_ema, preds_ema, truth_ema = evaluate_full(
            model, val_ids, labels, d.feature_dir, d.bag_size, device, d.n_classes
        )
        print(f"  fold {fold} weight selection: best-epoch {m_final.qwk:.4f} vs EMA {m_ema.qwk:.4f}")
        if m_ema.qwk > m_final.qwk:
            m_final, cums, preds, truth = m_ema, cums_ema, preds_ema, truth_ema
            selected, final_state = "ema", ema_state
        else:
            model.load_state_dict(best["state"])

    fold_dir = cfg.run_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": final_state,
            "config": cfg.to_dict(),
            "fold": fold,
            "weights_selected": selected,
            "val_qwk_subbag": best["qwk"],
            "val_qwk_full": m_final.qwk,
            "epoch": best["epoch"],
            "stop_reason": stop_reason,
            "epochs_run": len(history),
        },
        fold_dir / "checkpoint.pt",
    )
    with open(fold_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    np.savez(
        fold_dir / "val_predictions.npz",
        patient_ids=np.array(val_ids), cumulative=cums, pred=preds, true=truth,
    )

    return {
        "fold": fold,
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "stop_reason": stop_reason,
        "weights_selected": selected,
        "history": history,
    }, m_final


def run_cv(cfg: Config) -> dict:
    check_licence(cfg.encoder, cfg.allow_noncommercial)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits(cfg.data.splits_path)
    labels = splits.labels

    dev_ids = sorted({p for f in splits.folds for p in f["train"] + f["val"]})
    if cfg.data.bag_size is None:
        cfg.data.bag_size = suggest_bag_size(cfg.data.feature_dir, dev_ids)
    if cfg.model.feature_dim is None:
        cfg.model.feature_dim = feature_dim(cfg.data.feature_dir, dev_ids)

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(cfg.run_dir / "config.json")

    print(f"\n{cfg.run_name}  |  encoder {cfg.encoder} ({cfg.model.feature_dim}-d)")
    print(f"device {device} | bag_size {cfg.data.bag_size} | {len(splits.folds)} folds")
    print(f"test set held back: {len(splits.test)} patients (fingerprint {splits.fingerprint})\n")

    fold_metrics, fold_logs = [], []
    t0 = time.time()
    for i, fold in enumerate(splits.folds):
        log, metrics = train_fold(cfg, i, fold["train"], fold["val"], labels, device)
        fold_metrics.append(metrics)
        fold_logs.append(log)
        print(f"  fold {i} final (full coverage): {metrics.summary()}\n")

    summary = aggregate_folds(fold_metrics)
    summary["elapsed_minutes"] = round((time.time() - t0) / 60, 1)
    summary["bag_size"] = cfg.data.bag_size
    summary["feature_dim"] = cfg.model.feature_dim
    summary["folds"] = [
        {
            "fold": i,
            "best_epoch": lg["best_epoch"],
            "epochs_run": lg["epochs_run"],
            "stop_reason": lg["stop_reason"],
            "weights_selected": lg["weights_selected"],
            **{k: getattr(m, k) for k in
               ("qwk", "macro_f1", "balanced_accuracy", "adjacent_accuracy", "mean_absolute_error")},
        }
        for i, (lg, m) in enumerate(zip(fold_logs, fold_metrics))
    ]

    with open(cfg.run_dir / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("CROSS-VALIDATION SUMMARY  (model selection only - test set untouched)")
    for key in ("qwk", "macro_f1", "balanced_accuracy", "adjacent_accuracy"):
        s = summary[key]
        per = ", ".join(f"{v:.3f}" for v in s["per_fold"])
        print(f"  {key:<20} {s['mean']:.4f} +/- {s['std']:.4f}   [{per}]")
    print(f"  elapsed              {summary['elapsed_minutes']} min")
    print("=" * 72)
    return summary


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="Cross-validated ASMIL-Ord training.")
    p.add_argument("--config", required=True)
    p.add_argument("--run-name", default=None)
    args = p.parse_args(argv)

    cfg = Config.load(args.config)
    if args.run_name:
        cfg.run_name = args.run_name
    run_cv(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
