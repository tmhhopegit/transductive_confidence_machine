"""Backward patient selection for a linear regression (select_patients.m).

Start with all patients and fit a linear model with leave-one-out predictions.
Repeatedly drop the patient with the largest leave-one-out error, refit, and record
the correlation between the predictions and the outcome, until `min_patients` remain.

Caution: the correlation of the subset is optimistic by construction. The patients
are removed *because* the model predicts them badly, so the recorded r rises as
patients are removed, even on pure noise (see the tests). It describes how well
the model fits the patients it kept, not how well it would predict new ones. To
estimate the latter, the whole selection has to be repeated inside an outer
cross-validation: see nested.py.

Learners:
  "ridge"  least squares with a ridge penalty (fitrlinear 'Learner','leastsquares').
           The leave-one-out predictions are exact and come from one fit (via the hat matrix).
  "svm"    epsilon-insensitive linear regression, fitrlinear's default (the original
           used it, since it set no Learner). Solved by L-BFGS on a slightly smoothed loss
           and refitted for every left-out patient, so it is slower. MATLAB's own solver
           (SGD/BFGS with its tolerances) will not give identical numbers.
Both use fitrlinear's defaults: lambda = 1/n, with no penalty on the intercept. The
svm learner also uses epsilon = iqr(y)/13.49. Written as a sum over patients,
lambda = 1/n becomes a penalty of 1/2 |beta|^2 whatever n is, so every leave-one-out
fold has the same `penalty` (default 1 = n * lambda).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


def _design(X):
    X = np.asarray(X, dtype=np.float64)
    return np.column_stack([np.ones(len(X)), X])


def fit_ridge(X, y, penalty: float = 1.0) -> np.ndarray:
    """Ridge regression with an unpenalised intercept; returns [intercept, coefficients...]."""
    A = _design(X)
    P = np.eye(A.shape[1]) * penalty
    P[0, 0] = 0.0
    return np.linalg.solve(A.T @ A + P, A.T @ np.asarray(y, dtype=np.float64).ravel())


def predict_linear(theta, X) -> np.ndarray:
    return _design(X) @ theta


def loo_ridge(X, y, penalty: float = 1.0) -> np.ndarray:
    """Exact leave-one-out predictions of ridge regression with an intercept.

    Minimises sum(0.5 r^2) + penalty/2 |beta|^2 (fitrlinear 'leastsquares', lambda = penalty/n).
    Uses only the diagonal of the hat matrix: O(n p^2) rather than O(n^2 p)."""
    y = np.asarray(y, dtype=np.float64).ravel()
    A = _design(X)
    P = np.eye(A.shape[1]) * penalty
    P[0, 0] = 0.0
    M = np.linalg.inv(A.T @ A + P)
    h = np.einsum("ij,jk,ik->i", A, M, A)
    resid = y - A @ (M @ (A.T @ y))
    return y - resid / (1.0 - h)


def _svm_fit(X, y, penalty, eps, x0=None, smooth=None):
    n, p = X.shape
    smooth = smooth if smooth is not None else max(1e-6, 1e-3 * (np.std(y) + 1e-12))

    def f(theta):
        b, beta = theta[0], theta[1:]
        r = y - b - X @ beta
        a = np.abs(r) - eps
        # Huber-smoothed hinge: quadratic for 0 < a < smooth, linear beyond
        loss = np.where(a <= 0, 0.0, np.where(a < smooth, a ** 2 / (2 * smooth), a - smooth / 2))
        g = np.where(a <= 0, 0.0, np.where(a < smooth, a / smooth, 1.0)) * np.sign(r)
        val = loss.sum() + penalty / 2 * beta @ beta
        grad = np.concatenate([[-g.sum()], -(X.T @ g) + penalty * beta])
        return val, grad

    theta0 = np.zeros(p + 1) if x0 is None else x0
    if x0 is None:
        theta0[0] = np.median(y)
    return minimize(f, theta0, jac=True, method="L-BFGS-B").x


def _svm_eps(y):
    q75, q25 = np.percentile(y, [75, 25])
    return (q75 - q25) / 13.49


def fit_svm(X, y, penalty: float = 1.0, eps: float | None = None) -> np.ndarray:
    """Epsilon-insensitive linear regression; returns [intercept, coefficients...]."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    return _svm_fit(X, y, penalty, _svm_eps(y) if eps is None else eps)


def loo_svm(X, y, penalty: float = 1.0, eps: float | None = None) -> np.ndarray:
    """Leave-one-out predictions of epsilon-insensitive linear regression (fitrlinear's default)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    if eps is None:
        eps = _svm_eps(y)
    full = _svm_fit(X, y, penalty, eps)
    pred = np.empty(n)
    keep = np.ones(n, bool)
    for i in range(n):
        keep[i] = False
        theta = _svm_fit(X[keep], y[keep], penalty, eps, x0=full)
        pred[i] = theta[0] + X[i] @ theta[1:]
        keep[i] = True
    return pred


@dataclass
class Selection:
    masks: list          # boolean mask of the patients kept at each step (step 0 = everyone)
    r: np.ndarray        # correlation of leave-one-out predictions with y, on the kept patients
    n: np.ndarray        # number of patients kept
    removed: list        # index of the patient removed at each step (step 0: None)
    predictions: list    # leave-one-out predictions for the kept patients (NaN elsewhere)


def select_patients(X, y, min_patients: int = 9, learner: str = "ridge", penalty: float = 1.0,
                    verbose: bool = False) -> Selection:
    """Drop the worst-predicted patient one at a time (select_patients.m).

    The loop stops when min_patients remain. The original's `while length(find(sel)) > 9`
    also ends at 9.

    The original's lens(1) was never set (so it was 0); here n[0] is the full count."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    N = len(y)
    if learner == "ridge":
        fit = lambda Xs, ys: loo_ridge(Xs, ys, penalty)  # noqa: E731
    elif learner == "svm":
        fit = lambda Xs, ys: loo_svm(Xs, ys, penalty)  # noqa: E731
    else:
        raise ValueError("learner must be 'ridge' or 'svm'")

    sel = np.ones(N, bool)
    masks, rs, ns, removed, preds = [], [], [], [], []

    def record(drop):
        p = np.full(N, np.nan)
        p[sel] = fit(X[sel], y[sel])
        r = float(np.corrcoef(p[sel], y[sel])[0, 1])
        masks.append(sel.copy()); rs.append(r); ns.append(int(sel.sum())); removed.append(drop); preds.append(p)
        if verbose:
            print(f"{int(sel.sum()):5d} patients  r = {r:.4f}", flush=True)
        return p

    p = record(None)
    while sel.sum() > min_patients:
        err = np.abs(p - y)
        err[~sel] = -np.inf
        drop = int(np.argmax(err))
        sel[drop] = False
        p = record(drop)
    return Selection(masks, np.array(rs), np.array(ns), removed, preds)
