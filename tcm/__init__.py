"""TCM: a transductive confidence machine (k-NN strangeness) for two-class patient data,
with leave-one-out validation, feature-weight search, and backward patient selection.

    from tcm import loocv, find_weights, select_patients, nested_selection
"""
from .confidence import distances, loocv, max_k, p_value, strangeness, transductive_confidence, zscore
from .confidence import predict
from .nested import nested_selection, permutation_test
from .selection import fit_ridge, fit_svm, loo_ridge, loo_svm, predict_linear, select_patients
from .weights import find_weights, fitness

__all__ = ["distances", "loocv", "max_k", "p_value", "strangeness", "transductive_confidence", "zscore",
           "fitness", "find_weights", "select_patients", "loo_ridge", "loo_svm", "predict",
           "nested_selection", "permutation_test", "fit_ridge", "fit_svm", "predict_linear"]
__version__ = "1.0.0"
