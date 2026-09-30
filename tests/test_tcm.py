import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcm import (distances, find_weights, fitness, loo_ridge, loo_svm, loocv, max_k,  # noqa: E402
                 select_patients, strangeness, transductive_confidence, zscore)
from tcm.confidence import loocv_direct  # noqa: E402


def two_groups(n0=40, n1=40, shift=2.5, p=3, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n0, bool), np.ones(n1, bool)]
    X = rng.normal(size=(n0 + n1, p))
    X[y, 0] += shift
    return X, y


def test_zscore_matches_matlab_and_handles_constant_columns():
    X = np.array([[1.0, 5, 2], [2, 5, 4], [4, 5, 9]])
    Z = zscore(X)
    assert np.allclose(Z[:, 0], (X[:, 0] - X[:, 0].mean()) / X[:, 0].std(ddof=1))
    assert np.all(Z[:, 1] == 0)


def test_strangeness_by_hand():
    D = distances(np.array([[0.0], [1], [3], [10], [11]]), standardise=False)
    y = np.array([0, 0, 0, 1, 1])
    S = strangeness(D, y, 1)
    assert np.isclose(S[0], 1 / 10) and np.isclose(S[2], 2 / 7) and np.isclose(S[3], 1 / 7)


@pytest.mark.parametrize("seed", range(5))
def test_fast_loocv_equals_direct_computation(seed):
    rng = np.random.default_rng(seed)
    n0, n1 = rng.integers(10, 35, 2)
    X, y = two_groups(n0, n1, rng.uniform(0, 3), 4, seed)
    if seed == 4:
        X = np.round(X)                     # many tied distances
    D = distances(X)
    k = int(rng.integers(1, max_k(y) + 1))
    for method in ("standard", "original"):
        a, b = loocv(D=D, y=y, k=k, method=method), loocv_direct(D, y, k, method)
        assert np.allclose(a.p_true, b.p_true) and np.allclose(a.p_false, b.p_false)
        assert np.array_equal(a.pred, b.pred)


def test_single_p_value_matches_loocv():
    X, y = two_groups(15, 15)
    Y = y.copy()
    Y[3] = True
    assert np.isclose(transductive_confidence(X, Y, 4, subject=3), loocv(X, y, 4).p_true[3])


def test_standard_tcm_classifies_and_the_original_rule_is_inverted():
    X, y = two_groups()
    assert loocv(X, y, 5).accuracy(y) > 0.8
    assert loocv(X, y, 5, method="original").accuracy(y) < 0.3


def test_credibility_and_confidence():
    X, y = two_groups()
    r = loocv(X, y, 5)
    assert np.all((r.credibility >= 1 / len(y)) & (r.credibility <= 1))
    assert np.all(r.confidence >= 1 - r.credibility)


def test_k_is_checked():
    X, y = two_groups(6, 20)
    assert max_k(y) == 4
    with pytest.raises(ValueError):
        loocv(X, y, 5)


def test_fitness_edge_cases():
    X, y = two_groups(shift=20)
    r = loocv(X, y, 3)
    assert r.accuracy(y) == 1 and fitness(r, y) == float("inf")
    X, y = two_groups(shift=1.0)
    assert 0 < fitness(loocv(X, y, 5), y) < float("inf")


def test_weight_search_prefers_informative_features():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(50, bool), np.ones(50, bool)]
    X = rng.normal(size=(100, 6))
    X[y, 0] += 1.5
    r = find_weights(X, y, k=60, lr=0.05, patience=100, max_iterations=400, seed=0)
    assert r.k <= max_k(y)                                   # the starting k was clipped
    assert r.fitness >= r.history[0][1]
    assert np.argmax(r.w) == 0 and np.all(r.w >= 0)


def test_loo_ridge_is_exact():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(25, 3))
    y = X @ [1, -2, 0.5] + rng.normal(size=25) * 0.5 + 3
    ref = []
    for i in range(25):
        keep = np.arange(25) != i
        A = np.c_[np.ones(24), X[keep]]
        P = np.eye(4)
        P[0, 0] = 0
        th = np.linalg.solve(A.T @ A + P, A.T @ y[keep])
        ref.append(th[0] + X[i] @ th[1:])
    assert np.allclose(loo_ridge(X, y), ref)
    assert np.corrcoef(loo_svm(X, y), y)[0, 1] > 0.9


def test_patient_selection_runs_down_to_the_minimum_and_inflates_r_on_noise():
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(40, 4)), rng.normal(size=40)
    s = select_patients(X, y, min_patients=9)
    assert s.n[0] == 40 and s.n[-1] == 9 and len(s.r) == 32
    assert s.removed[0] is None and len(set(s.removed[1:])) == 31
    assert s.r[-1] > s.r[0] + 0.3                            # pure noise, yet r climbs


def test_command_line(tmp_path):
    from tcm.__main__ import main
    X, y = two_groups(20, 20)
    path = tmp_path / "d.csv"
    np.savetxt(path, np.column_stack([X, y]), delimiter=",", header="a,b,c,outcome", comments="")
    main(["loocv", str(path), "--label", "outcome", "--k", "5", "--out", str(tmp_path / "o.csv")])
    main(["weights", str(path), "--label", "outcome", "--k", "5", "--max-iterations", "5", "--seed", "0"])
    main(["select", str(path), "--target", "c", "--min", "30", "--out", str(tmp_path / "s.csv")])
    assert (tmp_path / "o.csv").exists() and (tmp_path / "s.csv").exists()


# ------------------------------------------------------------------ nested selection
def test_tcm_predicts_new_patients():
    from tcm import predict
    X, y = two_groups(50, 50, seed=0)
    Xn, yn = two_groups(50, 50, seed=1)
    r = predict(X, y, Xn, k=5)
    assert (r.pred == yn).mean() > 0.8
    assert (predict(X, y, Xn, k=5, method="original").pred == yn).mean() < 0.3


def test_nested_selection_is_not_inflated_on_noise():
    from tcm import nested_selection
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(80, 5)), rng.normal(size=80)
    res = nested_selection(X, y, keep=0.5)             # force half the patients out
    assert res.apparent_r > 0.6                        # what the non-nested analysis reports
    assert res.r < 0.2                                 # what it is worth on new patients
    assert all(np.isfinite(rep.predictions).all() for rep in res.repeats)


def test_nested_selection_never_uses_held_out_outcomes():
    from tcm.nested import nested_selection
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3))
    y = X @ [1.0, -1.0, 0.5] + rng.normal(size=40)
    a = nested_selection(X, y, keep=(1.0, 0.8), outer_folds=4, inner_folds=3, seed=5)
    fold = np.sort(np.random.default_rng(5).permutation(40)[:10])     # the first outer fold
    y2 = y.copy()
    y2[fold] = rng.normal(size=10) * 100               # change only that fold's outcomes
    b = nested_selection(X, y2, keep=(1.0, 0.8), outer_folds=4, inner_folds=3, seed=5)
    assert np.allclose(a.repeats[0].predictions[fold], b.repeats[0].predictions[fold])


def test_nested_selection_keeps_everyone_when_there_are_no_outliers_and_baseline_matches():
    from tcm import nested_selection
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 4))
    y = X @ [1.0, -1.0, 0.5, 0] + rng.normal(size=100) * 0.5
    res = nested_selection(X, y, keep=1.0)
    assert np.allclose(res.repeats[0].predictions, res.repeats[0].baseline)
    assert res.r > 0.8


def test_nested_selection_with_membership_and_permutation_test():
    from tcm import nested_selection, permutation_test
    rng = np.random.default_rng(3)
    X = rng.normal(size=(90, 4))
    y = X[:, :3] @ [1.0, -1.0, 0.5] + rng.normal(size=90) * 0.5
    bad = np.arange(90) < 20
    X[bad, 3] += 4                                     # an identifiable subgroup ...
    y[bad] = rng.normal(size=20) * 4                   # ... for which the model does not hold
    res = nested_selection(X, y, membership=True, repeats=2)
    assert res.repeats[0].predicted_kept is not None and "predicts would be kept" in res.summary()
    pt = permutation_test(X, y, n_permutations=19, outer_folds=5, inner_folds=3)
    assert pt.p <= 0.1 and len(pt.null_r) == 19


def test_nested_command_line(tmp_path):
    from tcm.__main__ import main
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 3))
    y = X @ [1.0, 0.5, 0] + rng.normal(size=40)
    path = tmp_path / "d.csv"
    np.savetxt(path, np.column_stack([X, y]), delimiter=",", header="a,b,c,score", comments="")
    main(["nested", str(path), "--target", "score", "--outer", "4", "--inner", "3", "--membership",
          "--permutations", "3", "--out", str(tmp_path / "n.csv")])
    assert (tmp_path / "n.csv").exists()
