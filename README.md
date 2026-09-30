# tcm

A Python package focused on probabilistic classification, which has three parts:

- a **transductive confidence machine** (TCM) that classifies patients into two groups with a k-nearest-neighbour "strangeness" measure;
- a hill-climbing search for **feature weights** (and k) that make the TCM's correct predictions more credible and confident than its wrong ones;
- a **backward patient selection** for a linear regression, with a **nested cross-validation** that estimates how well it predicts new patients.

Needs only numpy and scipy.

## Layout

```
refactored/
  tcm/
    confidence.py   strangeness, p_value, transductive_confidence, loocv (fast), loocv_direct (reference),
                    predict (new patients)
    weights.py      fitness, find_weights
    selection.py    select_patients, loo_ridge, loo_svm, fit_ridge, fit_svm
    nested.py       nested_selection, permutation_test
    __main__.py     command line
  examples/demo.py  a walk through on synthetic patients
  tests/test_tcm.py
 ```

## Use

```python
from tcm import loocv, find_weights, select_patients, nested_selection, permutation_test

r = loocv(X, y, k=10)                 # X: patients x features, y: 0/1
r.pred, r.credibility, r.confidence, r.p_true, r.p_false
r.accuracy(y)

s = find_weights(X, y, k=50, lr=0.01, patience=1000, seed=0, verbose=True)
s.w, s.k, s.fitness, s.result, s.history

sel = select_patients(X, score, min_patients=9, learner="ridge")   # or "svm"
sel.r, sel.n, sel.masks, sel.removed          # in-sample r: optimistic, see below

res = nested_selection(X, score, repeats=5)  # out-of-sample, with and without selection
print(res.summary())
pt = permutation_test(X, score, n_permutations=200, repeats=5)
pt.p, pt.p_gain
```

From the command line, with a CSV that has a header row:

```
python -m tcm loocv   data.csv --label outcome --k 10 --out predictions.csv
python -m tcm weights data.csv --label outcome --k 50 --out weights.csv
python -m tcm select  data.csv --target score --learner ridge --out selection.csv
python -m tcm nested  data.csv --target score --repeats 5 --membership --permutations 200 --out nested.csv
```

Run `python examples/demo.py` to see everything on synthetic data. To run the tests: `pytest tests`.

## Nested patient selection

`select_patients` drops the patients the model predicts worst and reports r on the rest. That r cannot be trusted. On pure noise it rises from 0.11 to 0.94 as patients are dropped (demo, 120 patients down to 30).

`nested_selection` asks the question that matters: *if the model is fitted after dropping badly predicted patients, how well does it predict new patients?* A new patient's outcome isn't known in advance, so they can't be dropped for being badly predicted, and every held-out patient is scored:

```
for each outer fold (default 10):
    training patients only:
        inner cross-validation (default 5 folds): for each candidate fraction to keep
        (default 1, .95, .9, .85, .8, .7, .6, .5), run the selection on the inner-training
        patients, fit on those kept, and predict ALL the inner-test patients
        -> choose the fraction with the lowest error (ties go to keeping more; 1 = no selection)
        run the selection on all training patients down to that fraction; fit
    predict ALL held-out patients
    baseline: the same fold fitted on all training patients, without selection
```

No held-out outcome is used to decide anything; a test checks this directly. The result reports the pooled out-of-sample r and MSE with and without selection on the same folds, the fraction kept in each fold, and the in-sample r that `select_patients` would have reported, for contrast.

- `keep=0.8` fixes the fraction and skips the inner loop.
- `repeats=5` averages over different fold splits.
- `criterion="r"` chooses the fraction by correlation instead of MSE.

Cross-validated r tends to fall slightly *below* 0 when there is no signal, so 0 is the wrong reference point. `permutation_test` reruns the whole nested procedure on shuffled outcomes. It gives:
- `p`: is the out-of-sample r better than chance?
- `p_gain`: is selection's improvement over no selection bigger than chance?

If the aim is to find a *kind* of patient for whom the model holds, that kind has to be recognisable before the outcome is known. With `membership=True`, each outer fold does the following:
1. Train a TCM on the training patients' features to tell kept from removed patients.
2. Predict which held-out patients would have been kept.
3. Report r for that predicted subgroup, with and without selection.

If the features can't tell the two apart, the subgroup is effectively random and its r won't improve.

