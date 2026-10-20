"""
The pictures.

    python plots.py latent          # movies in 2D via t-SNE, coloured by genre
    python plots.py latent pca      # same, but plain PCA (for comparison)
    python plots.py curves          # train/test RMSE per epoch

Nobody tells the model what a genre is. The clustering is the point.
Export the coordinates to CSV and Manim can animate the same points.

Note on t-SNE: it is used ONLY to lay out already-learned vectors for
display. It never touches the model — the factorization is still 100%
hand-written NumPy. t-SNE itself is honest maths worth a sentence on
camera: it places points so that near-neighbours in the real 32-D latent
space stay near-neighbours in 2-D, by minimising a divergence with its
own gradient descent.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model.npz")
FIGS = os.path.join(HERE, "figures")

NAVY, AMBER = "#0A1F44", "#FFA500"

# Distinctive, low-overlap genres, in PRIORITY order (most distinctive first).
# A movie tagged with several of these is assigned to the earliest one,
# which keeps the colour classes clean instead of double-counting.
DISTINCT = ["Documentary", "Animation", "Horror", "Romance"]
PALETTE = {"Documentary": "#1f77b4", "Animation": "#d62728",
           "Horror": "#0A1F44", "Romance": "#FFA500"}


def primary_genre(genre_str):
    """One distinctive genre per movie, or None if it carries none of them."""
    tags = str(genre_str).split("|")
    for g in DISTINCT:            # priority order breaks multi-genre ties
        if g in tags:
            return g
    return None


def pca_2d(X):
    """Two principal directions, by hand — it's just an eigen-decomposition."""
    Xc = X - X.mean(axis=0)
    C = Xc.T @ Xc / len(Xc)
    vals, vecs = np.linalg.eigh(C)
    top = vecs[:, np.argsort(-vals)[:2]]
    return Xc @ top, np.sort(-vals)[:2] * -1 / np.sum(vals)


def tsne_2d(X, seed=0):
    """t-SNE layout. perplexity is clamped to stay valid for small n."""
    from sklearn.manifold import TSNE
    n = len(X)
    perp = max(5, min(30, (n - 1) // 3))
    ts = TSNE(n_components=2, perplexity=perp, init="pca",
              learning_rate="auto", random_state=seed)
    return ts.fit_transform(X), perp


def latent(method="tsne", min_count=40, seed=0):
    z = np.load(MODEL, allow_pickle=True)
    Q, genres, counts, titles = z["Q"], z["genres"], z["counts"], z["titles"]

    keep = counts >= min_count
    Qk, gk, tk = Q[keep], genres[keep], titles[keep]
    print(f"{keep.sum()} movies with >= {min_count} ratings")

    if method == "tsne":
        coords, perp = tsne_2d(Qk, seed)
        xlab, ylab = "t-SNE direction 1", "t-SNE direction 2"
        sub = f"t-SNE, perplexity {perp}"
    else:
        coords, var = pca_2d(Qk)
        xlab = f"latent direction 1  ({100 * var[0]:.0f}% of variance)"
        ylab = f"latent direction 2  ({100 * var[1]:.0f}%)"
        sub = "PCA"

    labels = np.array([primary_genre(g) for g in gk], dtype=object)
    is_other = np.array([l is None for l in labels])

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=150)
    fig.patch.set_facecolor("white")

    # everything ambiguous fades into the background
    ax.scatter(coords[is_other, 0], coords[is_other, 1], s=9,
               c="#DDDDDD", edgecolors="none", zorder=1)

    for name in DISTINCT:
        sel = np.array([l == name for l in labels])
        if sel.sum() < 5:
            continue
        ax.scatter(coords[sel, 0], coords[sel, 1], s=36, color=PALETTE[name],
                   label=f"{name} ({sel.sum()})", edgecolors="white",
                   linewidths=0.4, alpha=0.9, zorder=2)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title("Movies in latent space — no genre was ever supplied",
                 color=NAVY, fontsize=13)
    ax.text(0.5, 1.005, sub, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=9, color="#888888")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIGS, "latent_space.png")
    fig.savefig(out)

    pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1],
                  "title": tk, "genres": gk,
                  "primary": labels}).to_csv(
        os.path.join(FIGS, "latent_space.csv"), index=False)
    print(f"saved -> {os.path.normpath(out)} (+ .csv for Manim)")


def curves():
    z = np.load(MODEL, allow_pickle=True)
    h = z["history"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(range(1, len(h) + 1), h[:, 0], color=NAVY, lw=2, label="train")
    ax.plot(range(1, len(h) + 1), h[:, 1], color=AMBER, lw=2, label="held out")
    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("Learning to predict what you'd have said", color=NAVY)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIGS, "learning_curve.png")
    fig.savefig(out)
    print(f"saved -> {os.path.normpath(out)}")


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "latent"
    if cmd == "curves":
        curves()
    else:
        method = sys.argv[2] if len(sys.argv) > 2 else "tsne"
        latent(method=method)
