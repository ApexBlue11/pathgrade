#!/usr/bin/env python
"""Step 8: is the attention map worth showing to anyone?

The overlay is the product, so "does the attention mean anything" is not a
research nicety - it decides whether the thing being sold is an explanation or
a decoration. This answers it with a control rather than a threshold.

**The control is the point.** Peakedness alone proves nothing: a randomly
initialised scorer also produces a map that is not perfectly flat, because
softmax over a few thousand patches with any score variation at all gives some
spread. The only meaningful question is whether the *trained* map beats the
*same architecture untrained*. When this was first run the answer was no - the
trained attention was very slightly flatter than random init - which reframed
the problem from "attention collapsed during training" to "attention never
learned", and killed a fix aimed at the wrong thing.

Three numbers are reported, per fold:

``per-subspace max/mean``
    Peakedness of one subspace's map. 1.0 is perfectly uniform.
``cross-subspace correlation``
    Whether the 24 subspace windows agree about which patches matter. nnMIL's
    inference-time averaging is only sound if they do. Near 0.0 means the
    windows are picking near-independent patches and averaging them is
    averaging noise.
``ensembled max/mean``
    What ``patch_attention`` actually returns, and therefore what a viewer
    sees. Averaging k decorrelated maps shrinks peak structure by about
    sqrt(k), so this can be far flatter than any single subspace.

A change to the model counts as progress only if it moves the trained row
away from the random-init row. Reads no labels and never touches the test set.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")


def subspace_maps(model, x, mask):
    """Attention map per subspace window: [n_offsets, n_patches]."""
    import numpy as np
    import torch
    from pathgrade.models.attention import gather_window, masked_softmax

    out = []
    with torch.no_grad():
        for off in model.offsets.tolist():
            alpha = masked_softmax(model.online(gather_window(x, off, model.window)), mask)
            out.append(alpha[0].mean(dim=-1).numpy())
    return np.stack(out)


def summarise(M):
    """(per-subspace max/mean, mean pairwise correlation, ensembled max/mean)."""
    import numpy as np

    per = float(np.mean([w.max() / w.mean() for w in M]))
    if M.shape[0] < 2:
        # A single window (window == feature_dim) is plain full-width attention:
        # there is no second subspace to correlate against, and np.corrcoef
        # returns a 0-d scalar rather than a matrix. Undefined, not zero.
        corr = float("nan")
    else:
        C = np.corrcoef(M)
        corr = float(C[~np.eye(len(C), dtype=bool)].mean())
    avg = M.mean(axis=0)
    return per, corr, float(avg.max() / avg.mean())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="training run with fold*/checkpoint.pt")
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--slides", type=int, default=5, help="slides sampled per fold")
    p.add_argument("--seed", type=int, default=0, help="seed for the random-init control")
    args = p.parse_args()

    import h5py
    import numpy as np
    import torch
    from pathgrade.config import Config
    from pathgrade.models.asmil_ord import ASMILOrd

    files = sorted(Path(args.feature_dir).glob("*.h5"))[: args.slides]
    if not files:
        print(f"no .h5 features under {args.feature_dir}", file=sys.stderr)
        return 2
    checkpoints = sorted(Path(args.run_dir).glob("fold*/checkpoint.pt"))
    if not checkpoints:
        print(f"no fold checkpoints under {args.run_dir}", file=sys.stderr)
        return 2

    def build(cfg):
        return ASMILOrd(
            feature_dim=cfg.model.feature_dim, n_classes=cfg.data.n_classes,
            window=cfg.model.window, stride=cfg.model.stride, hidden=cfg.model.hidden,
            n_branches=cfg.model.n_branches, dropout=cfg.model.dropout,
            branch_drop=cfg.model.branch_drop, ema_decay=cfg.model.ema_decay,
            feature_norm=cfg.model.feature_norm,
        )

    bags = []
    for f in files:
        with h5py.File(f, "r") as h:
            x = torch.from_numpy(h["features"][:]).float().unsqueeze(0)
        bags.append((x, torch.ones(x.shape[:2], dtype=torch.bool)))

    trained_rows = []
    for ck_path in checkpoints:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        cfg = Config.from_dict(ck["config"])
        model = build(cfg)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        for x, mask in bags:
            trained_rows.append(summarise(subspace_maps(model, x, mask)))

    # The control: same architecture, same slides, no training.
    torch.manual_seed(args.seed)
    control = build(cfg)
    control.eval()
    control_rows = [summarise(subspace_maps(control, x, mask)) for x, mask in bags]

    t, c = np.array(trained_rows), np.array(control_rows)
    print(f"\n{len(checkpoints)} folds x {len(files)} slides = {len(t)} measurements\n")
    print(f"{'':<14}{'per-subspace':>14}{'cross-subspace':>17}{'ensembled':>12}")
    print(f"{'':<14}{'max/mean':>14}{'correlation':>17}{'max/mean':>12}")
    print("-" * 57)
    print(f"{'trained':<14}{t[:,0].mean():>14.4f}{t[:,1].mean():>+17.4f}{t[:,2].mean():>12.4f}")
    print(f"{'random init':<14}{c[:,0].mean():>14.4f}{c[:,1].mean():>+17.4f}{c[:,2].mean():>12.4f}")
    print("-" * 57)

    beat = t[:, 0].mean() > c[:, 0].mean()
    print(f"\ntrained attention is {'MORE' if beat else 'NOT more'} peaked than random init.")
    if not beat:
        print("The attention module has not learned anything usable. Peakedness on")
        print("its own is not evidence - this control is what distinguishes a")
        print("learned map from softmax noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
