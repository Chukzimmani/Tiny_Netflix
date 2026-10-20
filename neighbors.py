"""
Nearest neighbours in latent space.

    python neighbors.py "Toy Story"
    python neighbors.py "Matrix" --n 10

The model was never told a genre. Two movies are "close" when their
learned vectors point the same direction. And because the predicted
rating is a dot product

        r_hat(u, i) = mu + b_u + b_i + p_u . q_i

two movies with nearly parallel q-vectors get scored almost identically
by EVERY user. So "near in latent space" literally means "people who
rate one this way rate the other that way." This list is read straight
off the geometry of the learned space -- no genres, no rules, just
angles between vectors.
"""

import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model.npz")


def load():
    z = np.load(MODEL, allow_pickle=True)
    return z["Q"], z["titles"], z["genres"], z["counts"]


def neighbours(title, n=8, min_count=20):
    Q, titles, genres, counts = load()

    # Unit-normalise every movie vector, so a dot product IS the cosine
    # of the angle between two movies: 1.0 = identical taste direction,
    # 0 = unrelated, negative = opposite taste.
    norms = np.linalg.norm(Q, axis=1, keepdims=True)
    U = Q / np.clip(norms, 1e-9, None)

    q = title.lower()
    hits = [i for i, t in enumerate(titles) if q in str(t).lower()]
    if not hits:
        print(f'No movie title contains "{title}".')
        return
    idx = hits[0]
    if len(hits) > 1:
        print(f'"{title}" matched {len(hits)} titles; using: {str(titles[idx])}\n')

    sims = U @ U[idx]              # cosine similarity to every other movie
    sims[idx] = -np.inf           # never return the query itself
    sims[counts < min_count] = -np.inf   # skip barely-rated obscurities

    order = np.argsort(-sims)[:n]
    print(f"Closest to  {titles[idx]}")
    print(f"            [{genres[idx]}]")
    print("-" * 60)
    for rank, j in enumerate(order, 1):
        print(f"{rank:2d}.  cos {sims[j]:+.3f}   {titles[j]}")
        print(f"              [{genres[j]}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("title", help="part of a movie title, e.g. 'Toy Story'")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--min_count", type=int, default=20)
    a = ap.parse_args()
    neighbours(a.title, a.n, a.min_count)
