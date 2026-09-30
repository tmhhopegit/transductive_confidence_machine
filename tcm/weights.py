"""Feature weighting for the TCM by hill climbing (find_weights.m, TCM_Fitness.m)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .confidence import LOOCVResult, _labels, distances, loocv, max_k, zscore


def fitness(result: LOOCVResult, y) -> float:
    """How much better the TCM's correct predictions are than its wrong ones (TCM_Fitness.m).

    Each prediction's quality is credibility * confidence**2. The fitness is the mean
    quality of the correct predictions divided by the mean quality of the wrong ones.
    (The original wrote this as credibility * (1 - confidence)**2, because its
    "confidence" variable held the rejected label's p-value. The value is the same.)

    If there are no wrong predictions, or they all have quality 0, the fitness is inf.
    The original returned NaN there, and the search then rejected every later step,
    because Q1 >= NaN is false."""
    y = _labels(y)
    q = result.credibility * result.confidence ** 2
    right, wrong = q[result.pred == y], q[result.pred != y]
    if len(right) == 0:
        return 0.0
    if len(wrong) == 0 or wrong.mean() == 0:
        return float("inf")
    return float(right.mean() / wrong.mean())


@dataclass
class WeightSearch:
    w: np.ndarray                 # feature weights (>= 0), applied after z-scoring
    k: int                        # number of neighbours
    fitness: float
    result: LOOCVResult           # leave-one-out result with the final w and k
    history: list = field(default_factory=list)   # (iteration, fitness, k, steps since improvement)


def find_weights(X, y, k: int = 50, lr: float = 0.01, patience: int = 1000, k_step_probability: float = 0.01,
                 k_min: int = 5, drift: str = "none", method: str = "standard", w0=None,
                 max_iterations: int | None = None, seed: int | None = None, verbose: bool = False,
                 callback: Callable | None = None) -> WeightSearch:
    """Hill-climb feature weights (and k) to maximise `fitness` (find_weights.m).

    Each step adds Gaussian noise (sd lr) to the weights, clips them at 0, and, with
    probability k_step_probability, moves k up or down by 1. A step is kept if the
    fitness does not get worse. The search stops after `patience` steps in a row
    without improvement, or after max_iterations.

    drift="original" reproduces the original's mutation, randn*lr - lr. It has mean
    -lr, so the weights drift towards 0 whether or not that helps. The default
    ("none") uses randn*lr.

    Differences from the original:
    - k is kept within [k_min, max_k(y)]. The original bounded it by N-1 with N
      undefined, so it stopped with an error the first time k moved, about 1 step in
      100. N-1 would also have been too large: k must fit in the smaller class.
    - The starting k is clipped to max_k(y) instead of failing when the smaller class
      has fewer than 52 patients."""
    rng = np.random.default_rng(seed)
    y = _labels(y)
    Z = zscore(X)
    n_features = Z.shape[1]
    w = np.ones(n_features) if w0 is None else np.array(w0, dtype=np.float64)
    k_max = max_k(y)
    if k_max < 1:
        raise ValueError("each class needs at least 3 patients")
    k = int(np.clip(k, min(k_min, k_max), k_max))

    def evaluate(w, k):
        res = loocv(D=distances(Z, w, standardise=False), y=y, k=k, method=method)
        return fitness(res, y), res

    best, best_res = evaluate(w, k)
    history = [(0, best, k, 0)]
    since = 0
    it = 0
    while since < patience and (max_iterations is None or it < max_iterations):
        it += 1
        step = rng.standard_normal(n_features) * lr
        if drift == "original":
            step -= lr
        elif drift != "none":
            raise ValueError("drift must be 'none' or 'original'")
        tw = np.maximum(w + step, 0.0)
        tk = k
        if rng.random() < k_step_probability:
            tk = int(np.clip(k + rng.choice((-1, 1)), min(k_min, k_max), k_max))
        q, res = evaluate(tw, tk)
        if q >= best:
            w, k, best, best_res = tw, tk, q, res
            since = 0
        else:
            since += 1
        history.append((it, best, k, since))
        if verbose and it % 50 == 0:
            print(f"iteration {it:6d}  fitness {best:.4f}  k {k}  steps since improvement {since}", flush=True)
        if callback is not None:
            callback(it, w, k, best)
    return WeightSearch(w, k, best, best_res, history)
