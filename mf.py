"""
Matrix factorization, from scratch.

The whole idea in one line:

    predicted_rating(user u, movie i)  =  mu + b_u + b_i + p_u . q_i

where p_u and q_i are short vectors (k latent factors) that the optimizer
discovers on its own. No genres are ever supplied. No ML libraries.

We minimise, over the observed ratings only:

    L = sum_(u,i in observed) (r_ui - r_hat_ui)^2
        + lam * (||p_u||^2 + ||q_i||^2 + b_u^2 + b_i^2)

Four ways to descend that surface are implemented here:
    'sgd'       one rating at a time
    'gd'        full batch, every rating before a single step
    'momentum'  full batch with a velocity term
    'als'       alternating least squares (no gradient at all)
"""

import numpy as np


class MF:
    def __init__(self, n_users, n_items, k=20, lam=0.05, lr=0.01,
                 use_bias=True, seed=0):
        rng = np.random.default_rng(seed)
        scale = 0.1
        self.P = rng.normal(0, scale, (n_users, k))   # user factors
        self.Q = rng.normal(0, scale, (n_items, k))   # item factors
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.mu = 0.0
        self.k, self.lam, self.lr = k, lam, lr
        self.use_bias = use_bias
        self.history = []          # (train_rmse, test_rmse) per epoch

    # ---------- prediction ----------

    def predict(self, u, i):
        """Predict for arrays (or scalars) of user and item indices."""
        out = np.sum(self.P[u] * self.Q[i], axis=-1)
        if self.use_bias:
            out = out + self.mu + self.bu[u] + self.bi[i]
        return out

    def rmse(self, data):
        u, i, r = data
        return float(np.sqrt(np.mean((r - self.predict(u, i)) ** 2)))

    def loss(self, data):
        u, i, r = data
        err = r - self.predict(u, i)
        reg = self.lam * (
            np.sum(self.P ** 2) + np.sum(self.Q ** 2)
            + np.sum(self.bu ** 2) + np.sum(self.bi ** 2)
        )
        return float(np.sum(err ** 2) + reg)

    # ---------- training ----------

    def fit(self, train, test=None, epochs=20, method="sgd",
            beta=0.9, verbose=True, seed=0, callback=None):
        u, i, r = train
        self.mu = float(np.mean(r)) if self.use_bias else 0.0
        rng = np.random.default_rng(seed)

        vP = np.zeros_like(self.P)   # momentum velocities
        vQ = np.zeros_like(self.Q)
        vbu = np.zeros_like(self.bu)
        vbi = np.zeros_like(self.bi)

        for epoch in range(1, epochs + 1):
            if method == "sgd":
                order = rng.permutation(len(r))
                for n in order:
                    self._sgd_step(u[n], i[n], r[n])

            elif method in ("gd", "momentum"):
                gP, gQ, gbu, gbi = self._full_gradients(train)
                if method == "gd":
                    self.P -= self.lr * gP
                    self.Q -= self.lr * gQ
                    if self.use_bias:
                        self.bu -= self.lr * gbu
                        self.bi -= self.lr * gbi
                else:
                    vP = beta * vP + gP
                    vQ = beta * vQ + gQ
                    self.P -= self.lr * vP
                    self.Q -= self.lr * vQ
                    if self.use_bias:
                        vbu = beta * vbu + gbu
                        vbi = beta * vbi + gbi
                        self.bu -= self.lr * vbu
                        self.bi -= self.lr * vbi

            elif method == "als":
                self._als_step(train)

            else:
                raise ValueError(f"unknown method: {method}")

            tr = self.rmse(train)
            te = self.rmse(test) if test is not None else np.nan
            self.history.append((tr, te))
            if callback is not None:
                callback(epoch, tr, te)
            if verbose:
                tail = f"  test RMSE {te:.4f}" if test is not None else ""
                print(f"epoch {epoch:3d}   train RMSE {tr:.4f}{tail}")

        return self

    def _sgd_step(self, u, i, r):
        err = r - self.predict(u, i)
        pu, qi = self.P[u].copy(), self.Q[i].copy()
        self.P[u] += self.lr * (err * qi - self.lam * pu)
        self.Q[i] += self.lr * (err * pu - self.lam * qi)
        if self.use_bias:
            self.bu[u] += self.lr * (err - self.lam * self.bu[u])
            self.bi[i] += self.lr * (err - self.lam * self.bi[i])

    def _full_gradients(self, data):
        """dL/dtheta over every observed rating at once."""
        u, i, r = data
        err = r - self.predict(u, i)                 # (n_ratings,)

        gP = np.zeros_like(self.P)
        gQ = np.zeros_like(self.Q)
        np.add.at(gP, u, -2.0 * err[:, None] * self.Q[i])
        np.add.at(gQ, i, -2.0 * err[:, None] * self.P[u])
        gP += 2.0 * self.lam * self.P
        gQ += 2.0 * self.lam * self.Q

        gbu = np.zeros_like(self.bu)
        gbi = np.zeros_like(self.bi)
        if self.use_bias:
            np.add.at(gbu, u, -2.0 * err)
            np.add.at(gbi, i, -2.0 * err)
            gbu += 2.0 * self.lam * self.bu
            gbi += 2.0 * self.lam * self.bi

        n = len(r)
        return gP / n, gQ / n, gbu / n, gbi / n

    def _als_step(self, data):
        """
        Hold Q fixed, solve every p_u exactly (ridge regression); then swap.
        No learning rate anywhere — that's the point of the comparison.
        """
        u, i, r = data
        resid = r - (self.mu + self.bu[u] + self.bi[i]) if self.use_bias else r
        I = np.eye(self.k)

        by_user = _group(u, len(self.P))
        for uu, idx in by_user:
            Qs = self.Q[i[idx]]
            A = Qs.T @ Qs + self.lam * I
            b = Qs.T @ resid[idx]
            self.P[uu] = np.linalg.solve(A, b)

        by_item = _group(i, len(self.Q))
        for ii, idx in by_item:
            Ps = self.P[u[idx]]
            A = Ps.T @ Ps + self.lam * I
            b = Ps.T @ resid[idx]
            self.Q[ii] = np.linalg.solve(A, b)

    # ---------- using the trained model ----------

    def fold_in(self, item_idx, ratings, lam=None):
        """
        Fit a factor vector for a brand-new user (you) with Q frozen, by
        solving the ridge-regularised least-squares problem EXACTLY:

            min_(p,b)  sum_j (r_j - (mu + bi_j + b + q_j . p))^2
                       + lam (||p||^2 + b^2)

        One linear system -- the same exact solve ALS already does per user,
        no learning rate, no step count. Returns (p_new, b_new).

        lam defaults to the model's own lambda; pass a larger value when
        folding in a user with only a handful of ratings, to stop the
        vector from memorising them (see the held-out check in personal.py).
        """
        item_idx = np.asarray(item_idx)
        ratings = np.asarray(ratings, dtype=float)
        lam = self.lam if lam is None else lam

        Qs = self.Q[item_idx]
        base = (self.mu + self.bi[item_idx]) if self.use_bias else 0.0
        y = ratings - base

        if self.use_bias:
            X = np.column_stack([Qs, np.ones(len(item_idx))])   # solve [p ; b]
            A = X.T @ X + lam * np.eye(self.k + 1)
            sol = np.linalg.solve(A, X.T @ y)
            return sol[:-1], float(sol[-1])
        A = Qs.T @ Qs + lam * np.eye(self.k)
        return np.linalg.solve(A, Qs.T @ y), 0.0

    def recommend(self, p, b=0.0, exclude=(), n=10):
        scores = self.Q @ p
        if self.use_bias:
            scores = scores + self.mu + b + self.bi
        scores[list(exclude)] = -np.inf
        top = np.argsort(-scores)[:n]
        return top, scores[top]


def _group(keys, n_groups):
    """Yield (group_id, indices) pairs — a cheap groupby over an index array."""
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    bounds = np.flatnonzero(np.diff(ks)) + 1
    for chunk in np.split(order, bounds):
        if len(chunk):
            yield int(keys[chunk[0]]), chunk
