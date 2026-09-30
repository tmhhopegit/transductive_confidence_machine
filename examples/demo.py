"""A walk through the package on synthetic patients.

    python examples/demo.py

Two groups of 60 patients, 8 features, of which only the first two differ between the groups.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcm import find_weights, loocv, nested_selection, permutation_test, select_patients  # noqa: E402

rng = np.random.default_rng(0)
y = np.r_[np.zeros(60, bool), np.ones(60, bool)]
X = rng.normal(size=(120, 8))
X[y, 0] += 1.5
X[y, 1] -= 1.0

# 1. Leave-one-out TCM classification
r = loocv(X, y, k=10)
print(f"TCM leave-one-out accuracy: {r.accuracy(y):.2f}")
print(f"  with the original (inverted) p-value: {loocv(X, y, k=10, method='original').accuracy(y):.2f}")
print(f"  mean credibility {r.credibility.mean():.2f}, mean confidence {r.confidence.mean():.2f}")

# 2. Feature weights
s = find_weights(X, y, k=50, lr=0.05, patience=200, seed=0, verbose=True)
print(f"\nweights (features 1 and 2 are the informative ones): {np.round(s.w, 2)}")
print(f"k = {s.k}, fitness {s.history[0][1]:.2f} -> {s.fitness:.2f}, accuracy {s.result.accuracy(y):.2f}")

# 3. Patient selection for a regression, and why its r cannot be taken at face value
score = rng.normal(size=120)                     # an outcome unrelated to X
sel = select_patients(X, score, min_patients=30)
print(f"\npatient selection on pure noise: r = {sel.r[0]:.2f} with all {sel.n[0]} patients,"
      f" {sel.r[-1]:.2f} with the {sel.n[-1]} that are left")

# 4. The honest version: selection nested inside cross-validation
nested = nested_selection(X, score, keep=0.5)          # force half out, as an extreme case
print(f"nested out-of-sample r on the same noise, keeping half the training patients: {nested.r:.2f}")

# 5. A case built to favour selection: 25 patients, recognisable from feature 8, whose outcome
#    does not follow the model. Even so, the gain from dropping patients is small and the
#    permutation test cannot tell it from chance; the in-sample r would suggest otherwise.
outcome = X[:, :3] @ [1.0, -1.0, 0.5] + rng.normal(size=120) * 0.7
odd = np.arange(120) < 25
X2 = X.copy()
X2[odd, 7] += 4
outcome[odd] = rng.normal(size=25) * 4
res = nested_selection(X2, outcome, repeats=5, membership=True)
print("\n" + res.summary())
pt = permutation_test(X2, outcome, n_permutations=50)
print(f"permutation test: p = {pt.p:.3f} for r; p = {pt.p_gain:.3f} for selection's gain over no selection")
