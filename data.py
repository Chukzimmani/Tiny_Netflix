"""
Loading MovieLens and turning it into (user_index, item_index, rating) triplets.

Download ml-latest-small from https://grouplens.org/datasets/movielens/
and unzip it into data/ so you have:
    data/ml-latest-small/ratings.csv
    data/ml-latest-small/movies.csv
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "ml-latest-small")


class Dataset:
    def __init__(self, ratings, movies=None):
        self.ratings = ratings
        self.movies = movies

        self.user_ids = np.sort(ratings.userId.unique())
        self.item_ids = np.sort(ratings.movieId.unique())
        self.u_of = {v: n for n, v in enumerate(self.user_ids)}
        self.i_of = {v: n for n, v in enumerate(self.item_ids)}

        self.u = ratings.userId.map(self.u_of).to_numpy()
        self.i = ratings.movieId.map(self.i_of).to_numpy()
        self.r = ratings.rating.to_numpy(dtype=float)

        self.n_users = len(self.user_ids)
        self.n_items = len(self.item_ids)

        if movies is not None:
            m = movies.set_index("movieId")
            self.title = [
                m.title.get(mid, str(mid)) for mid in self.item_ids
            ]
            self.genres = [
                m.genres.get(mid, "") for mid in self.item_ids
            ]
        else:
            self.title = [f"item {n}" for n in range(self.n_items)]
            self.genres = [""] * self.n_items

    def split(self, test_frac=0.1, seed=0):
        """Random held-out split. Every user keeps most of their ratings."""
        rng = np.random.default_rng(seed)
        mask = rng.random(len(self.r)) < test_frac
        train = (self.u[~mask], self.i[~mask], self.r[~mask])
        test = (self.u[mask], self.i[mask], self.r[mask])
        return train, test

    def counts(self):
        """How many ratings each movie has — useful for filtering obscurities."""
        return np.bincount(self.i, minlength=self.n_items)

    def density(self):
        return len(self.r) / (self.n_users * self.n_items)


def load(path=DATA):
    ratings = pd.read_csv(os.path.join(path, "ratings.csv"))
    movies_path = os.path.join(path, "movies.csv")
    movies = pd.read_csv(movies_path) if os.path.exists(movies_path) else None
    return Dataset(ratings, movies)


def synthetic(n_users=300, n_items=200, k_true=4, density=0.25, seed=0):
    """
    A fake world with known latent structure — for checking the optimizer
    actually recovers signal before you touch the real data.
    """
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1, (n_users, k_true))
    Q = rng.normal(0, 1, (n_items, k_true))
    n = int(density * n_users * n_items)
    u = rng.integers(0, n_users, n)
    i = rng.integers(0, n_items, n)
    r = 3.5 + 0.5 * np.sum(P[u] * Q[i], axis=1) + rng.normal(0, 0.3, n)
    r = np.clip(np.round(r * 2) / 2, 0.5, 5.0)
    df = pd.DataFrame({"userId": u, "movieId": i, "rating": r})
    df = df.drop_duplicates(subset=["userId", "movieId"])
    return Dataset(df)
