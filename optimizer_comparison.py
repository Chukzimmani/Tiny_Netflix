"""
Four optimizers, same data, same k, same lambda, same initialisation.

    python experiments/optimizer_comparison.py
    python experiments/optimizer_comparison.py --synthetic

This is the figure that becomes the paper: convergence per epoch and per
second of wall clock, which are not the same ranking.
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
import data as data_mod          # noqa: E402
from mf import MF                # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

# Full-batch gradients are averaged over all ratings, so gd/momentum need
# far larger steps than sgd. These are tuned on the synthetic set — retune
# on real MovieLens before you trust the figure.
METHODS = {
    "sgd":      dict(lr=0.01,  color="#0A1F44"),
    "momentum": dict(lr=10.0,  color="#FFA500"),
    "gd":       dict(lr=50.0,  color="#7A8AA0"),
    "als":      dict(lr=0.01,  color="#C0392B"),   # lr unused
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()

    ds = data_mod.synthetic() if args.synthetic else data_mod.load()
    train, test = ds.split(test_frac=0.1)

    rows, results = [], {}
    for name, cfg in METHODS.items():
        m = MF(ds.n_users, ds.n_items, k=args.k, lam=args.lam,
               lr=cfg["lr"], seed=0)
        stamps = []
        t0 = time.perf_counter()
        m.fit(train, test, epochs=args.epochs, method=name, verbose=False,
              callback=lambda e, tr, te: stamps.append(time.perf_counter() - t0))
        h = np.array(m.history)
        results[name] = (h, stamps, cfg["color"])
        rows.append({"method": name, "lr": cfg["lr"],
                     "final_train_rmse": h[-1, 0],
                     "final_test_rmse": h[-1, 1],
                     "seconds": stamps[-1]})
        print(f"{name:9s} test RMSE {h[-1, 1]:.4f}   {stamps[-1]:6.1f}s")

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "optimizer_comparison.csv"),
                              index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    for name, (h, stamps, color) in results.items():
        axes[0].plot(range(1, len(h) + 1), h[:, 1], color=color, lw=2,
                     label=name)
        axes[1].plot(stamps, h[:, 1], color=color, lw=2, label=name)
    axes[0].set_xlabel("epoch")
    axes[1].set_xlabel("wall clock (seconds)")
    for ax in axes:
        ax.set_ylabel("held-out RMSE")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Same number of passes over the data", color="#0A1F44")
    axes[1].set_title("Same amount of your life", color="#0A1F44")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "optimizer_comparison.png"))
    print(f"saved -> {os.path.normpath(OUT)}/optimizer_comparison.png")


if __name__ == "__main__":
    main()
