"""Overlay training-loss curves of several variants for one subject.

    python scripts/plot_losses.py --subject dog --variants M0 M1
-> runs/<subject>/loss_<variants>.png  (50-step running mean of the per-step loss; for variants with
   ControlNet in the loop also the control-on / control-off curves separately)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from boothnet.utils import load_yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--variants", nargs="+", default=["M0", "M1", "M2", "M3"])
    ap.add_argument("--window", type=int, default=50)
    args = ap.parse_args()
    runs = ROOT / load_yaml("configs/train_base.yaml")["runs_dir"] / args.subject
    colors = {"M0": "tab:gray", "M1": "tab:blue", "M2": "tab:green", "M3": "tab:red"}
    fig, ax = plt.subplots(figsize=(8, 4))
    for v in args.variants:
        p = runs / v / "train_log.csv"
        if not p.exists():
            print(f"missing {p}")
            continue
        df = pd.read_csv(p)
        s = df.groupby("step")["loss"].mean().rolling(args.window, min_periods=1).mean()
        ax.plot(s.index, s.values, color=colors.get(v), lw=2, label=f"{v} (all)")
        if df["control_on"].nunique() > 1:
            for on, ls in [(1, "--"), (0, ":")]:
                ss = df[df.control_on == on].groupby("step")["loss"].mean().rolling(args.window, min_periods=1).mean()
                ax.plot(ss.index, ss.values, color=colors.get(v), lw=1.2, ls=ls,
                        label=f"{v} control {'on' if on else 'off'}")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(f"loss ({args.window}-step running mean)")
    ax.set_title(f"{args.subject}: training loss")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = runs / f"loss_{'_'.join(args.variants)}.png"
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
