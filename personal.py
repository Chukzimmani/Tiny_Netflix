"""
Your own results — the part that goes on camera.

Step 1:  python personal.py rate
         Rates widely-seen films one at a time (0.5-5, blank to skip).
         Saves to my_ratings.csv. Aim for 35-40 — you have k latent
         unknowns to pin down, so more ratings = a steadier vector.

Step 2:  python personal.py recommend
         Folds you into the trained model and prints:
           - a HELD-OUT honesty check (predicts a few of your own ratings
             it wasn't fit on — the number you can actually trust),
           - your position in the latent space,
           - what it thinks you'll like, and what you'd avoid.

The fold-in solves ONE ridge least-squares system for your taste vector p,
holding every movie vector Q fixed:  (Q_s^T Q_s + lam I) p = Q_s^T (r - base).
"""

import os
import sys
import numpy as np
import pandas as pd

from mf import MF

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model.npz")
MYRATINGS = os.path.join(HERE, "my_ratings.csv")

# Regularisation for the fold-in. Deliberately stronger than the model's
# training lambda: a single new user has only a few dozen ratings against
# k unknowns, so without this the vector memorises them (fit looks great,
# real predictions are poor). Validated by the held-out check below.
FOLD_LAM = 1.0

# Only ever recommend films with at least this many ratings — sparse-vector
# movies produce spurious matches (this floor was validated on the
# nearest-neighbours lists: flukes vanished once it was raised to 100).
MIN_COUNT = 100


def load_model():
    z = np.load(MODEL, allow_pickle=True)
    m = MF(1, len(z["Q"]), k=z["Q"].shape[1])
    m.P, m.Q = z["P"], z["Q"]
    m.bu, m.bi, m.mu = z["bu"], z["bi"], float(z["mu"])
    return m, z


def rate():
    _, z = load_model()
    titles, counts = z["titles"], z["counts"]
    # rate widely-seen films first — you're more likely to have opinions
    order = np.argsort(-counts)[:150]
    rng = np.random.default_rng()
    rng.shuffle(order)

    rows = []
    if os.path.exists(MYRATINGS):
        rows = pd.read_csv(MYRATINGS).to_dict("records")
        print(f"{len(rows)} ratings already saved — adding more.\n")
    seen = {r["item_index"] for r in rows}

    print("Rate 0.5 to 5. Blank = haven't seen it. 'q' = done. Aim for 35-40.\n")
    for idx in order:
        if int(idx) in seen:
            continue
        raw = input(f"  {titles[idx]}  ({counts[idx]} ratings): ").strip()
        if raw.lower() == "q":
            break
        if not raw:
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        if 0.5 <= val <= 5.0:
            rows.append({"item_index": int(idx),
                         "title": titles[idx], "rating": val})

    pd.DataFrame(rows).to_csv(MYRATINGS, index=False)
    print(f"\n{len(rows)} ratings saved -> {os.path.normpath(MYRATINGS)}")


def recommend(n=15):
    m, z = load_model()
    titles, genres, counts = z["titles"], z["genres"], z["counts"]
    mine = pd.read_csv(MYRATINGS)
    idx = mine.item_index.to_numpy()
    vals = mine.rating.to_numpy(dtype=float)
    print(f"folding in {len(idx)} of your ratings (lambda = {FOLD_LAM})...\n")

    # --- honesty check: hold out 5 of YOUR ratings, predict them cold ---
    # This is the number to trust. In-sample fit always looks better than
    # reality; predicting films you rated but weren't fit on is the truth.
    if len(idx) > 10:
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(idx))
        te, trn = perm[:5], perm[5:]
        p_h, b_h = m.fold_in(idx[trn], vals[trn], lam=FOLD_LAM)
        pred = m.mu + b_h + m.bi[idx[te]] + m.Q[idx[te]] @ p_h
        held = float(np.sqrt(np.mean((vals[te] - pred) ** 2)))
        print(f"held-out check: predicted 5 of my own ratings I was NOT fit "
              f"on -> RMSE {held:.3f}")
        print("(the honest measure of whether it learned my taste or just "
              "memorised it)\n")
    else:
        print("(rate more than 10 films to unlock the held-out honesty "
              "check)\n")

    # --- refit on ALL your ratings for the real recommendations ---
    p, b = m.fold_in(idx, vals, lam=FOLD_LAM)
    print("your latent vector:", np.round(p, 3))

    scores = m.mu + b + m.bi + m.Q @ p
    block = set(idx.tolist()) | set(np.flatnonzero(counts < MIN_COUNT).tolist())
    scores[list(block)] = -np.inf
    order = np.argsort(-scores)
    live = order[np.isfinite(scores[order])]

    print("\n--- what the math thinks you'll like ---")
    for t in live[:n]:
        print(f"  {scores[t]:4.2f}  {titles[t]}   [{genres[t]}]")

    print("\n--- and what it thinks you'd avoid (good on-camera moment) ---")
    for t in live[-5:][::-1]:
        print(f"  {scores[t]:4.2f}  {titles[t]}   [{genres[t]}]")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "recommend"
    rate() if cmd == "rate" else recommend()
