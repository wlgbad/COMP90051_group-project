"""
Medium-complexity algorithm: RBF-kernel SVM (sklearn SVC, one-vs-one).

    k(x, x') = exp(-gamma * ||x - x'||^2)

Tuned hyperparameters
    C          soft-margin penalty (regularisation strength = 1/C)
    gamma_mult gamma = gamma_mult * gamma_scale, with
               gamma_scale = 1 / (n_features * Var(Z_train)) computed on the
               training fold (sklearn's 'scale' heuristic, written by hand).
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC

ALGORITHM = "rbf_svm"

# Initial grid C in {0.1, 1, 10} x gamma_mult in {0.1, 1, 10} picked edge values
# (C = 10 most often). A wider scan of mean inner-validation macro-F1
# (C in 1..1000, gamma_mult in 0.01..1) put the optimum near C ~ 100,
# gamma_mult ~ 0.3, so the grid was re-centred there. Only inner splits were
# used for this; outer test sets were never consulted.
GRID = {
    "C": [30.0, 100.0, 300.0],
    "gamma_mult": [0.1, 0.3, 1.0],
}

SMOKE_GRID = {
    "C": [1.0, 10.0],
    "gamma_mult": [1.0],
}


def gamma_scale(Z_train: np.ndarray) -> float:
    return 1.0 / (Z_train.shape[1] * Z_train.var())


def make_model(params: dict, Z_train: np.ndarray) -> SVC:
    return SVC(
        kernel="rbf",
        C=params["C"],
        gamma=params["gamma_mult"] * gamma_scale(Z_train),
        cache_size=1000,
    )
