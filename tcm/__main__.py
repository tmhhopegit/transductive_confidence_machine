"""Command line.

    python -m tcm loocv   data.csv --label outcome [--k 10] [--features a b c]
    python -m tcm weights data.csv --label outcome [--k 50] [--lr 0.01] [--patience 1000] [--out weights.csv]
    python -m tcm select  data.csv --target score [--learner ridge|svm] [--min 9] [--out selection.csv]
    python -m tcm nested  data.csv --target score [--keep 1 0.9 0.8 0.7] [--outer 10] [--inner 5]
                          [--repeats 5] [--membership] [--permutations 200] [--out predictions.csv]

data.csv has a header row, one row per patient, and numeric columns. Rows with any
missing value in the columns used are dropped (with a message).
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from .confidence import loocv
from .nested import nested_selection, permutation_test
from .selection import select_patients
from .weights import find_weights


def read_table(path, target, features=None):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = [h.strip() for h in rows[0]], rows[1:]
    if target not in header:
        sys.exit(f"column '{target}' not found; columns are: {', '.join(header)}")
    features = features or [h for h in header if h != target]
    missing = [c for c in features if c not in header]
    if missing:
        sys.exit(f"columns not found: {', '.join(missing)}")
    cols = [header.index(c) for c in features + [target]]

    def num(v):
        try:
            return float(v)
        except ValueError:
            return np.nan
    data = np.array([[num(r[c]) if c < len(r) else np.nan for c in cols] for r in body])
    ok = ~np.isnan(data).any(axis=1)
    if not ok.all():
        print(f"dropping {int((~ok).sum())} rows with missing values", file=sys.stderr)
    return data[ok, :-1], data[ok, -1], features


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tcm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("loocv", "weights", "select", "nested"):
        p = sub.add_parser(name)
        p.add_argument("csv")
        p.add_argument("--label" if name in ("loocv", "weights") else "--target", required=True, dest="target")
        p.add_argument("--features", nargs="+")
        p.add_argument("--out")
        if name in ("loocv", "weights"):
            p.add_argument("--k", type=int, default=10 if name == "loocv" else 50)
            p.add_argument("--method", choices=("standard", "original"), default="standard")
        if name == "weights":
            p.add_argument("--lr", type=float, default=0.01)
            p.add_argument("--patience", type=int, default=1000)
            p.add_argument("--max-iterations", type=int)
            p.add_argument("--seed", type=int)
        if name == "select":
            p.add_argument("--learner", choices=("ridge", "svm"), default="ridge")
            p.add_argument("--min", type=int, default=9)
        if name == "nested":
            p.add_argument("--learner", choices=("ridge", "svm"), default="ridge")
            p.add_argument("--keep", type=float, nargs="+", default=[1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5])
            p.add_argument("--outer", type=int, default=10)
            p.add_argument("--inner", type=int, default=5)
            p.add_argument("--repeats", type=int, default=1)
            p.add_argument("--criterion", choices=("mse", "r"), default="mse")
            p.add_argument("--membership", action="store_true")
            p.add_argument("--permutations", type=int, default=0)
            p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    X, y, features = read_table(a.csv, a.target, a.features)

    if a.cmd == "loocv":
        r = loocv(X, y, a.k, method=a.method)
        print(f"accuracy {r.accuracy(y):.3f} ({len(y)} patients, k={a.k})")
        if a.out:
            np.savetxt(a.out, np.column_stack([y, r.pred, r.p_true, r.p_false, r.credibility, r.confidence]),
                       delimiter=",", header="label,pred,p_true,p_false,credibility,confidence", comments="", fmt="%g")
    elif a.cmd == "weights":
        r = find_weights(X, y, k=a.k, lr=a.lr, patience=a.patience, method=a.method,
                         max_iterations=a.max_iterations, seed=a.seed, verbose=True)
        print(f"fitness {r.fitness:.4f}, k {r.k}, leave-one-out accuracy {r.result.accuracy(y):.3f}")
        for f, w in sorted(zip(features, r.w), key=lambda t: -t[1]):
            print(f"  {f:30s} {w:.3f}")
        if a.out:
            with open(a.out, "w", newline="") as f:
                csv.writer(f).writerows([["feature", "weight"], *zip(features, r.w), ["k", r.k]])
    elif a.cmd == "nested":
        args = dict(keep=a.keep, outer_folds=a.outer, inner_folds=a.inner, repeats=a.repeats, learner=a.learner,
                    criterion=a.criterion, membership=a.membership)
        res = nested_selection(X, y, seed=a.seed, verbose=True, **args)
        print(res.summary())
        if a.permutations:
            pt = permutation_test(X, y, n_permutations=a.permutations, seed=a.seed, **args)
            print(f"permutation test ({a.permutations} shuffles): p = {pt.p:.3f} for r;"
                  f" p = {pt.p_gain:.3f} for selection's gain over no selection")
        if a.out:
            rep = res.repeats[0]
            cols = [y, rep.predictions, rep.baseline] + ([rep.predicted_kept] if a.membership else [])
            head = "outcome,prediction_with_selection,prediction_without_selection" + (
                ",predicted_kept" if a.membership else "")
            np.savetxt(a.out, np.column_stack(cols), delimiter=",", header=head, comments="", fmt="%g")
    else:
        s = select_patients(X, y, min_patients=a.min, learner=a.learner, verbose=True)
        print("note: r rises as badly predicted patients are removed, even for pure noise;"
              " use 'nested' for an out-of-sample estimate (see README)")
        if a.out:
            with open(a.out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "n_patients", "r", "removed_row"])
                for i, (n, r, d) in enumerate(zip(s.n, s.r, s.removed)):
                    w.writerow([i, n, f"{r:.6f}", "" if d is None else d + 1])


if __name__ == "__main__":
    main()
