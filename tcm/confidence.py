"""Transductive confidence machine (TCM) with a k-nearest-neighbour strangeness measure.

For a two-class problem (labels 0/1), the strangeness of patient j is

    S_j = (sum of the k smallest distances from j to the other patients of its own class)
          / (sum of the k smallest distances from j to the patients of the other class)

A patient who sits among its own class has small S; one that sits among the other
class has large S. To classify patient i, each label l is tried in turn. With i
labelled l, the strangeness of every patient is recomputed, and the p-value of l is
the fraction of patients that are at least as strange as i:

    p(l) = #{j : S_j >= S_i} / N          (i itself included, so p >= 1/N)

The prediction is the label with the larger p-value. The credibility is that p-value,
and the confidence is 1 - (the other label's p-value).

The original code (transductive_confidence.m) counted the patients that are *less*
strange than i, #{j : S_j < S_i} / N, which is roughly 1 - p. TCM_loocv then picked the
label with the larger value, so it predicted the label under which the patient looks
*most* out of place. On two well-separated groups it scores far below chance (about
16% in the tests). `method="original"` reproduces it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist, squareform


# --------------------------------------------------------------------------- distances
def zscore(X: np.ndarray) -> np.ndarray:
    """Column z-scores with the sample standard deviation (as MATLAB's zscore).

    Constant columns become 0; MATLAB gives NaN, which would make every distance NaN."""
    X = np.asarray(X, dtype=np.float64)
    sd = X.std(axis=0, ddof=1)
    return np.divide(X - X.mean(axis=0), sd, out=np.zeros_like(X), where=sd > 0)


def distances(X: np.ndarray, w: np.ndarray | None = None, standardise: bool = True) -> np.ndarray:
    """Euclidean distances between patients after z-scoring each feature and scaling it by w."""
    Z = zscore(X) if standardise else np.asarray(X, dtype=np.float64)
    if w is not None:
        Z = Z * np.asarray(w, dtype=np.float64).reshape(1, -1)
    return squareform(pdist(Z))


def _labels(y) -> np.ndarray:
    y = np.asarray(y).ravel()
    values = np.unique(y)
    if not set(values.tolist()) <= {0, 1, False, True}:
        raise ValueError(f"labels must be 0/1 (or boolean); got {values}")
    return y.astype(bool)


def max_k(y) -> int:
    """The largest k for which every strangeness value is defined in a leave-one-out run.

    A patient j needs k other patients of each class. In the worst case j and the held-out
    patient i share the smaller class, and i is being tried with the other label, which
    leaves (n_small - 2)."""
    y = _labels(y)
    return int(min(y.sum(), (~y).sum()) - 2)


def _check_k(k: int, y: np.ndarray) -> int:
    k = int(k)
    if k < 1:
        raise ValueError("k must be at least 1")
    if k > max_k(y):
        raise ValueError(f"k={k} is too large: the smaller class has {min(y.sum(), (~y).sum())} patients,"
                         f" so k can be at most {max_k(y)}")
    return k


# --------------------------------------------------------------------------- strangeness
def strangeness(D: np.ndarray, y, k: int) -> np.ndarray:
    """k-NN strangeness of every patient under the labels y (see the module docstring)."""
    y = _labels(y)
    Dd = np.array(D, dtype=np.float64)
    np.fill_diagonal(Dd, np.inf)
    sums = {}
    for c in (False, True):
        cols = Dd[:, y == c]
        if cols.shape[1] < k + 1:       # members of c other than the patient itself
            raise ValueError(f"k={k} is larger than a class allows")
        part = np.partition(cols, k - 1, axis=1)[:, :k]
        sums[c] = part.sum(axis=1)
    same = np.where(y, sums[True], sums[False])
    other = np.where(y, sums[False], sums[True])
    with np.errstate(divide="ignore", invalid="ignore"):
        return same / other


TIE_TOLERANCE = 1e-9


def p_value(S: np.ndarray, subject: int, method: str = "standard") -> float:
    """p-value of `subject` given everyone's strangeness S (NaNs are ignored, as in the original).

    Strangeness values within a relative 1e-9 of the subject's count as ties (i.e. as
    equally strange). Otherwise rounding in the sums would decide ties arbitrarily,
    and ties are common with integer-valued features."""
    S = np.asarray(S)
    valid = ~np.isnan(S)
    s = S[subject]
    tol = TIE_TOLERANCE * abs(s) if np.isfinite(s) else 0.0
    if method == "standard":
        return float(np.sum(valid & (S >= s - tol)) / len(S))
    if method == "original":
        return float(np.sum(valid & (S < s - tol)) / len(S))
    raise ValueError("method must be 'standard' or 'original'")


def transductive_confidence(X, y, k: int, D=None, w=None, subject: int = 0, method: str = "standard") -> float:
    """p-value of patient `subject` with the labels y as given (transductive_confidence.m)."""
    if D is None:
        D = distances(X, w)
    return p_value(strangeness(D, y, k), subject, method)


# --------------------------------------------------------------------------- leave-one-out
@dataclass
class LOOCVResult:
    pred: np.ndarray          # predicted label per patient (bool)
    p_true: np.ndarray        # p-value of label 1 for each patient
    p_false: np.ndarray       # p-value of label 0
    credibility: np.ndarray   # max(p_true, p_false)
    confidence: np.ndarray    # 1 - min(p_true, p_false)

    def accuracy(self, y) -> float:
        return float(np.mean(self.pred == _labels(y)))


def loocv(X=None, y=None, k: int = 5, D=None, w=None, method: str = "standard") -> LOOCVResult:
    """Classify every patient by the TCM, trying both labels for it (TCM_loocv.m / TCM_loocv_par.m).

    Give either X (features; z-scored and weighted by w) or a precomputed distance matrix D.

    Computing strangeness from scratch for every patient and label takes O(N^3 log N)
    time. Here each patient's k+1 nearest neighbours in each class are sorted once,
    and then trying patient i with label l only changes the sums in which i appears.
    That brings the whole run down to O(N^2 k), and it gives the same results as the
    direct computation (see tests). The parallel version is no longer needed.

    Ties: when the two p-values are equal the prediction is 1, as in the original."""
    y = _labels(y)
    if D is None:
        if X is None:
            raise ValueError("give X or D")
        D = distances(X, w)
    D = np.asarray(D, dtype=np.float64)
    N = len(y)
    if D.shape != (N, N):
        raise ValueError(f"D must be {N}x{N}")
    k = _check_k(k, y)

    Dd = D.copy()
    np.fill_diagonal(Dd, np.inf)
    # For each patient j and class c: the k+1 nearest members of c (excluding j), sorted.
    near_d, near_i, sum_k = {}, {}, {}
    for c in (False, True):
        members = np.flatnonzero(y == c)
        part = np.argpartition(Dd[:, members], k, axis=1)[:, :k + 1]
        dist = np.take_along_axis(Dd[:, members], part, axis=1)
        order = np.argsort(dist, axis=1)
        near_d[c] = np.take_along_axis(dist, order, axis=1)          # (N, k+1)
        near_i[c] = members[np.take_along_axis(part, order, axis=1)]  # patient indices
        sum_k[c] = near_d[c][:, :k].sum(axis=1)

    p = {False: np.zeros(N), True: np.zeros(N)}
    for i in range(N):
        d_i = Dd[:, i]                                    # distance from every j to i (inf at j = i)
        for label in (False, True):
            sums = {}
            for c in (False, True):
                s = sum_k[c].copy()
                kth = near_d[c][:, k - 1].copy()
                if y[i] == c:                             # take i out of its own class
                    pos = np.argmax(near_i[c] == i, axis=1)
                    present = (near_i[c] == i).any(axis=1) & (pos < k)
                    s[present] += near_d[c][present, k] - d_i[present]
                    kth[present] = near_d[c][present, k]
                if label == c:                            # and put it into the class it is being tried in
                    closer = d_i < kth
                    s[closer] += d_i[closer] - kth[closer]
                sums[c] = s
            same = np.where(y, sums[True], sums[False])
            other = np.where(y, sums[False], sums[True])
            # patient i itself (its lists never contain i, since its self-distance is inf)
            same[i] = sum_k[label][i]
            other[i] = sum_k[not label][i]
            with np.errstate(divide="ignore", invalid="ignore"):
                S = same / other
            p[label][i] = p_value(S, i, method)

    p_true, p_false = p[True], p[False]
    pred = ~(p_false > p_true)
    credibility = np.maximum(p_true, p_false)
    confidence = 1 - np.minimum(p_true, p_false)
    return LOOCVResult(pred, p_true, p_false, credibility, confidence)


# --------------------------------------------------------------------------- reference version
def loocv_direct(D, y, k: int, method: str = "standard") -> LOOCVResult:
    """A line-by-line version of TCM_loocv.m: slow, and used only in the tests to check `loocv`."""
    y = _labels(y)
    N = len(y)
    p_true, p_false = np.zeros(N), np.zeros(N)
    for i in range(N):
        Y = y.copy()
        Y[i] = True
        p_true[i] = p_value(strangeness(D, Y, k), i, method)
        Y[i] = False
        p_false[i] = p_value(strangeness(D, Y, k), i, method)
    pred = ~(p_false > p_true)
    return LOOCVResult(pred, p_true, p_false, np.maximum(p_true, p_false), 1 - np.minimum(p_true, p_false))


# --------------------------------------------------------------------------- new patients
def predict(X_train, y_train, X_new, k: int = 5, w=None, method: str = "standard") -> LOOCVResult:
    """Classify new patients with a TCM built on training patients.

    Features are z-scored with the training patients' means and SDs (the new patients
    never influence the scaling). Each new patient is added to the training set on
    its own, once with each label. p(l) is then the fraction of the N+1 patients that
    are at least as strange as the new one."""
    y_train = _labels(y_train)
    X_train = np.asarray(X_train, dtype=np.float64)
    X_new = np.atleast_2d(np.asarray(X_new, dtype=np.float64))
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0, ddof=1)
    scale = np.where(sd > 0, sd, np.inf)
    Zt, Zn = (X_train - mu) / scale, (X_new - mu) / scale
    if w is not None:
        Zt, Zn = Zt * np.asarray(w).reshape(1, -1), Zn * np.asarray(w).reshape(1, -1)
    k = int(k)
    small = int(min(y_train.sum(), (~y_train).sum()))
    if k < 1 or k > small - 1:
        raise ValueError(f"k must be between 1 and {small - 1} (one less than the smaller class)")
    N = len(y_train)
    D = np.zeros((N + 1, N + 1))
    D[:N, :N] = squareform(pdist(Zt))
    p_true, p_false = np.zeros(len(Zn)), np.zeros(len(Zn))
    for t, z in enumerate(Zn):
        d = np.sqrt(((Zt - z) ** 2).sum(axis=1))
        D[N, :N] = D[:N, N] = d
        for label, out in ((True, p_true), (False, p_false)):
            y = np.append(y_train, label)
            out[t] = p_value(strangeness(D, y, k), N, method)
    pred = ~(p_false > p_true)
    return LOOCVResult(pred, p_true, p_false, np.maximum(p_true, p_false), 1 - np.minimum(p_true, p_false))
