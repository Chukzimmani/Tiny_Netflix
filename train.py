"""
Train the model on MovieLens and save the learned factors.

    python src/train.py                 # real data
    python src/train.py --synthetic     # fake data, no download needed
"""

import argparse
import os
import numpy as np

import data as data_mod
from mf import MF

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--method", default="sgd",
                    choices=["sgd", "gd", "momentum", "als"])
    ap.add_argument("--out", default=os.path.join(HERE, "model.npz"))
    args = ap.parse_args()

    ds = data_mod.synthetic() if args.synthetic else data_mod.load()
    print(f"{ds.n_users} users x {ds.n_items} movies, "
          f"{len(ds.r)} ratings, {100 * ds.density():.2f}% full")

    train, test = ds.split(test_frac=0.1)
    model = MF(ds.n_users, ds.n_items, k=args.k, lam=args.lam, lr=args.lr)
    model.fit(train, test, epochs=args.epochs, method=args.method)

    print(f"\nfinal held-out RMSE: {model.rmse(test):.4f}")
    print("(predict-the-average baseline: "
          f"{np.sqrt(np.mean((test[2] - np.mean(train[2])) ** 2)):.4f})")

    np.savez(args.out,
             P=model.P, Q=model.Q, bu=model.bu, bi=model.bi, mu=model.mu,
             item_ids=ds.item_ids, titles=np.array(ds.title, dtype=object),
             genres=np.array(ds.genres, dtype=object),
             counts=ds.counts(), history=np.array(model.history))
    print(f"saved -> {os.path.normpath(args.out)}")


if __name__ == "__main__":
    main()
