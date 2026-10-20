# A recommendation engine, from scratch

Linear algebra and optimization, doing real work. No scikit-learn, no
Surprise, no PyTorch — NumPy and a hand-written gradient.

The model is one line:

```
predicted_rating(user u, movie i)  =  mu + b_u + b_i + p_u · q_i
```

`p_u` and `q_i` are short vectors — the latent factors. Nobody supplies
genres. The optimizer discovers taste dimensions on its own by minimising
squared error over the ratings that actually exist:

```
L = Σ_(u,i observed) (r_ui − r̂_ui)²  +  λ(‖p_u‖² + ‖q_i‖² + b_u² + b_i²)
```

## Setup

```bash
pip install numpy pandas matplotlib
```

Download **ml-latest-small** from https://grouplens.org/datasets/movielens/
and unzip it so you have `data/ml-latest-small/ratings.csv` and `movies.csv`.
(~600 users, ~9,000 movies, 100,000 ratings.)

Every script also takes `--synthetic`, which fabricates a world with known
latent structure — use it to check the code before the download.

## Run order

```bash
python src/train.py                       # train, report held-out RMSE, save model.npz
python src/plots.py latent                # movies in 2D, coloured by genre
python src/plots.py curves                # learning curve
python src/personal.py rate               # rate 20–30 films yourself
python src/personal.py recommend          # your vector, your recommendations
python experiments/optimizer_comparison.py
```

## What's in each file

| file | what it does |
|---|---|
| `src/mf.py` | the model and four optimizers: `sgd`, `gd`, `momentum`, `als` |
| `src/data.py` | MovieLens → index triplets, train/test split, synthetic generator |
| `src/train.py` | training run, saves `model.npz` |
| `src/personal.py` | rate films yourself, fold in, get recommendations |
| `src/plots.py` | latent space (+ CSV for Manim) and learning curves |
| `experiments/optimizer_comparison.py` | four optimizers, per epoch and per second |

## The three outputs

**Video.** Every animation comes from the running code, not an illustration.
`figures/latent_space.csv` has x, y, title and genres per movie — feed it
straight into Manim and the clustering animates from real learned vectors.
The `personal.py recommend` output is the on-camera payoff: your own vector,
your own top ten, and the films it says you'd hate.

**GitHub.** This repo is the portfolio piece. Keep the README explaining
the math, commit the figures, add a notebook walkthrough when you're happy
with the results.

**Paper.** `experiments/optimizer_comparison.py` is the seed. The interesting
result is that convergence-per-epoch and convergence-per-second rank the
methods differently, and that ALS reaches a good RMSE in very few sweeps but
overfits fast at low λ. Sweep λ and k, repeat across seeds, and that's an
empirical study.

## Things to watch

- **Learning rates are not transferable.** Full-batch gradients here are
  averaged over all ratings, so `gd` and `momentum` need steps 1000× larger
  than `sgd`. The values in `optimizer_comparison.py` are tuned on the
  synthetic set — retune on real MovieLens or the figure is meaningless.
- **ALS uses `use_bias=True` by holding biases fixed** while it solves for
  P and Q. Fine for the comparison; mention it if you write it up.
- **Predicted scores aren't clipped to [0.5, 5].** Left raw on purpose so
  you can see the model overshoot. Clip before you screenshot anything.
- **Filter by rating count** before recommending, or the top ten fills with
  obscure films rated by four people. `personal.py` drops anything under 30.
