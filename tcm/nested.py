"""Nested cross-validation of patient selection: an honest estimate of out-of-sample performance.

`select_patients` reports r on the patients it kept. They were kept *because* the model
predicts them well, so that r is optimistic, even on pure noise. The honest question is
different: if a model is fitted after dropping badly predicted training patients, how
well does it predict new patients?

A new patient's outcome is not known in advance, so they cannot be dropped for being
badly predicted. Every held-out patient is therefore scored:

    for each outer fold:
        training patients only:
            inner cross-validation: for each candidate fraction to keep,
                run the selection on the inner-training patients, fit on those kept,
                and predict *all* inner-test patients.
            choose the fraction with the lowest inner error (ties go to keeping more).
            run the selection on all the training patients down to that fraction and fit.
        predict all the held-out patients with that model.
        (baseline: the same fold, fitted on all the training patients, no selection.)

The outcome of a held-out patient is never used to decide anything, so the pooled
out-of-sample r and MSE are not inflated. The baseline uses the same folds, so the
comparison shows whether selection actually helps.

Optional, `membership=True`: selection may be meant to find the *kind* of patient the
model works for. That is only useful if the kind can be recognised before the outcome
is known. In each outer fold, a TCM is trained on the training patients' features to
tell kept from removed patients. It predicts which held-out patients would have been
kept, and the out-of-sample r and MSE are also reported for that predicted subgroup,
alongside the baseline model on the same subgroup. If the features cannot tell the
two kinds apart, the subgroup is a random subset and its r will be no better.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .confidence import predict as tcm_predict
from .selection import fit_ridge, fit_svm, predict_linear, select_patients


def _folds(n: int, n_folds: int, rng) -> list[np.ndarray]:
    order = rng.permutation(n)
    return [np.sort(f) for f in np.array_split(order, min(n_folds, n))]


def _fit(learner, X, y, penalty):
    return fit_ridge(X, y, penalty) if learner == "ridge" else fit_svm(X, y, penalty)


def _n_keep(fraction: float, n: int) -> int:
    return max(3, min(n, int(round(fraction * n))))


def _selection_masks(X, y, fractions, learner, penalty):
    """Run the selection once, down to the smallest fraction; return the mask for each fraction."""
    n = len(y)
    sizes = [_n_keep(f, n) for f in fractions]
    path = select_patients(X, y, min_patients=min(sizes), learner=learner, penalty=penalty)
    return [path.masks[n - size] for size in sizes]


def _scores(pred, y):
    pred, y = np.asarray(pred), np.asarray(y)
    mse = float(np.mean((pred - y) ** 2))
    r = float(np.corrcoef(pred, y)[0, 1]) if len(y) > 2 and np.std(pred) > 0 and np.std(y) > 0 else np.nan
    return r, mse


@dataclass
class RepeatResult:
    predictions: np.ndarray            # out-of-sample prediction for every patient (model after selection)
    baseline: np.ndarray               # out-of-sample prediction without selection (same folds)
    chosen_keep: list                  # fraction kept, chosen in each outer fold
    r: float
    mse: float
    baseline_r: float
    baseline_mse: float
    predicted_kept: np.ndarray | None = None   # membership=True: would this patient have been kept?
    subgroup_r: float = np.nan
    subgroup_mse: float = np.nan
    subgroup_baseline_r: float = np.nan
    subgroup_baseline_mse: float = np.nan


@dataclass
class NestedResult:
    repeats: list = field(default_factory=list)
    apparent_r: float = np.nan         # what select_patients reports on all the data at the typical chosen fraction
    apparent_keep: float = np.nan

    def _mean(self, name):
        return float(np.nanmean([getattr(r, name) for r in self.repeats]))

    @property
    def r(self): return self._mean("r")
    @property
    def mse(self): return self._mean("mse")
    @property
    def baseline_r(self): return self._mean("baseline_r")
    @property
    def baseline_mse(self): return self._mean("baseline_mse")
    @property
    def subgroup_r(self): return self._mean("subgroup_r")
    @property
    def subgroup_baseline_r(self): return self._mean("subgroup_baseline_r")

    def summary(self) -> str:
        keeps = np.concatenate([r.chosen_keep for r in self.repeats])
        lines = [
            f"out-of-sample, all held-out patients ({len(self.repeats)} repeat(s)):",
            f"  with selection     r = {self.r:.3f}   MSE = {self.mse:.4g}",
            f"  without selection  r = {self.baseline_r:.3f}   MSE = {self.baseline_mse:.4g}",
            f"  fraction kept (chosen per fold): median {np.median(keeps):.2f}, range {keeps.min():.2f}-{keeps.max():.2f}",
        ]
        if self.repeats[0].predicted_kept is not None:
            share = np.mean([r.predicted_kept.mean() for r in self.repeats])
            lines += [f"held-out patients a TCM on the features predicts would be kept ({share:.0%}):",
                      f"  with selection     r = {self.subgroup_r:.3f}",
                      f"  without selection  r = {self.subgroup_baseline_r:.3f}"]
        if self.apparent_keep < 1:
            lines.append(f"for comparison, the in-sample r that select_patients reports at {self.apparent_keep:.2f} kept:"
                         f" {self.apparent_r:.3f} (optimistic)")
        else:
            lines.append(f"(the inner loop mostly chose to keep everyone; leave-one-out r on all patients:"
                         f" {self.apparent_r:.3f})")
        return "\n".join(lines)


def nested_selection(X, y, keep=(1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5), outer_folds: int = 10,
                     inner_folds: int = 5, repeats: int = 1, learner: str = "ridge", penalty: float = 1.0,
                     criterion: str = "mse", membership: bool = False, membership_k: int = 5,
                     seed: int | None = 0, verbose: bool = False) -> NestedResult:
    """Nested cross-validation of `select_patients` (see the module docstring).

    keep:        candidate fractions of the training patients to keep. Give a single number
                 to fix it (then no inner loop is run). Include 1.0 so "no selection" can win.
    criterion:   "mse" or "r", computed on the pooled inner-test predictions.
    repeats:     repeat the whole procedure with different fold splits and average,
                 which reduces the dependence on one random split.
    learner:     "ridge" (fast) or "svm" (slow here: every selection step refits the
                 model once per patient, in every fold).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    N = len(y)
    fractions = sorted({float(f) for f in np.atleast_1d(keep)}, reverse=True)
    if any(not 0 < f <= 1 for f in fractions):
        raise ValueError("keep fractions must be in (0, 1]")
    if criterion not in ("mse", "r"):
        raise ValueError("criterion must be 'mse' or 'r'")
    if learner not in ("ridge", "svm"):
        raise ValueError("learner must be 'ridge' or 'svm'")
    rng = np.random.default_rng(seed)
    result = NestedResult()

    for rep in range(repeats):
        pred = np.full(N, np.nan)
        base = np.full(N, np.nan)
        kept_flag = np.zeros(N, bool) if membership else None
        chosen = []
        for fold, test in enumerate(_folds(N, outer_folds, rng)):
            train = np.setdiff1d(np.arange(N), test)
            Xtr, ytr = X[train], y[train]

            # ---- inner loop: choose how many training patients to keep
            if len(fractions) == 1:
                best = fractions[0]
            else:
                inner_pred = np.full((len(fractions), len(train)), np.nan)
                for itest in _folds(len(train), inner_folds, rng):
                    itrain = np.setdiff1d(np.arange(len(train)), itest)
                    masks = _selection_masks(Xtr[itrain], ytr[itrain], fractions, learner, penalty)
                    for c, m in enumerate(masks):
                        theta = _fit(learner, Xtr[itrain][m], ytr[itrain][m], penalty)
                        inner_pred[c, itest] = predict_linear(theta, Xtr[itest])
                scores = [_scores(p, ytr) for p in inner_pred]
                loss = [s[1] if criterion == "mse" else -s[0] for s in scores]
                loss = [np.inf if np.isnan(v) else v for v in loss]
                best = fractions[int(np.argmin(loss))]       # first minimum = largest fraction on ties
            chosen.append(best)

            # ---- outer: select on the training patients, predict every held-out patient
            mask = _selection_masks(Xtr, ytr, [best], learner, penalty)[0]
            pred[test] = predict_linear(_fit(learner, Xtr[mask], ytr[mask], penalty), X[test])
            base[test] = predict_linear(_fit(learner, Xtr, ytr, penalty), X[test])

            if membership:
                removed = ~mask
                small = int(min(mask.sum(), removed.sum()))
                if small < 2:                                  # nobody (or one patient) removed: all count as kept
                    kept_flag[test] = True
                else:
                    k = min(membership_k, small - 1)
                    kept_flag[test] = tcm_predict(Xtr, mask, X[test], k=k).pred
            if verbose:
                print(f"repeat {rep + 1} fold {fold + 1}: kept {best:.2f} of {len(train)} training patients", flush=True)

        r, mse = _scores(pred, y)
        br, bmse = _scores(base, y)
        rr = RepeatResult(pred, base, chosen, r, mse, br, bmse)
        if membership:
            rr.predicted_kept = kept_flag
            if kept_flag.sum() > 2:
                rr.subgroup_r, rr.subgroup_mse = _scores(pred[kept_flag], y[kept_flag])
                rr.subgroup_baseline_r, rr.subgroup_baseline_mse = _scores(base[kept_flag], y[kept_flag])
        result.repeats.append(rr)

    # the number the non-nested analysis would have reported
    keeps = np.concatenate([r.chosen_keep for r in result.repeats])
    result.apparent_keep = float(np.median(keeps))
    size = _n_keep(result.apparent_keep, N)
    path = select_patients(X, y, min_patients=size, learner=learner, penalty=penalty)
    result.apparent_r = float(path.r[N - size])
    return result


@dataclass
class PermutationTest:
    r: float                  # nested out-of-sample r on the real outcomes
    null_r: np.ndarray        # the same, with the outcomes shuffled
    p: float                  # (1 + #{null >= r}) / (1 + n)
    gain: float               # r - baseline_r on the real outcomes
    null_gain: np.ndarray
    p_gain: float             # is selection's improvement over no selection bigger than chance?


def permutation_test(X, y, n_permutations: int = 200, seed: int | None = 0, verbose: bool = False,
                     **nested_args) -> PermutationTest:
    """Is the nested out-of-sample r better than chance, and is selection's gain over no selection?

    Cross-validated r tends to fall below 0 when there is no signal (the model fitted
    without a patient is pulled slightly away from that patient's outcome), so 0 is
    not the right reference. Shuffling the outcomes and rerunning
    the whole nested procedure gives the right null distribution. With the ridge learner
    each permutation takes about as long as one nested run."""
    y = np.asarray(y, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    real = nested_selection(X, y, seed=seed, **nested_args)
    null_r, null_gain = np.empty(n_permutations), np.empty(n_permutations)
    for i in range(n_permutations):
        res = nested_selection(X, rng.permutation(y), seed=int(rng.integers(2**31)), **nested_args)
        null_r[i], null_gain[i] = res.r, res.r - res.baseline_r
        if verbose and (i + 1) % 20 == 0:
            print(f"permutation {i + 1}/{n_permutations}", flush=True)
    gain = real.r - real.baseline_r
    p = (1 + np.sum(null_r >= real.r)) / (1 + n_permutations)
    p_gain = (1 + np.sum(null_gain >= gain)) / (1 + n_permutations)
    return PermutationTest(real.r, null_r, float(p), float(gain), null_gain, float(p_gain))
