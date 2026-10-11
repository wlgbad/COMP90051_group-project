"""
Verify the team's hand-written metrics against sklearn.metrics (allowed in tests
only), plus sanity checks for shared_features, raw-mean failure, the shared
failure patterns and the inner temporal splits.

    python -m tests.test_metrics
"""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm

from medium.common import PATTERNS_PATH, load_data, per_class_f1
from shared_features import FeaturePipeline, construct_features, feature_names
from shared_metrics import classification_metrics_from_scratch
from shared_protocol import make_inner_temporal_splits, sensor_feature_positions
from shared_robustness import (corrupt_raw_with_outer_train_mean,
                               corrupt_standardised_with_zero, load_failure_patterns)


def test_metrics_match_sklearn() -> None:
    rng = np.random.default_rng(0)
    labels = list(range(6))
    for _ in range(200):
        n = int(rng.integers(1, 300))
        y_true = rng.integers(0, 6, n)
        y_pred = np.where(rng.random(n) < rng.random(), y_true, rng.integers(0, 6, n))
        m, cm = classification_metrics_from_scratch(y_true, y_pred)

        assert np.array_equal(cm, skm.confusion_matrix(y_true, y_pred, labels=labels))
        assert np.isclose(m["accuracy"], skm.accuracy_score(y_true, y_pred))
        for key, avg_fn in (("macro_precision", skm.precision_score),
                            ("macro_recall", skm.recall_score), ("macro_f1", skm.f1_score)):
            ref = avg_fn(y_true, y_pred, labels=labels, average="macro", zero_division=0)
            assert np.isclose(m[key], ref), key
        assert np.allclose(per_class_f1(cm), skm.f1_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0))
    print("shared_metrics + per_class_f1 match sklearn on 200 random cases")


def test_features_failure_and_splits() -> None:
    df, X_raw, y, batch, splits = load_data()
    split = splits[0]
    tr, te = split["train_indices"], split["test_indices"]

    Z = construct_features(X_raw)
    assert Z.shape == (len(X_raw), len(feature_names())) and np.isfinite(Z).all()
    Z_tr = FeaturePipeline().fit(X_raw[tr]).transform(X_raw[tr])
    assert np.allclose(Z_tr.mean(axis=0), 0, atol=1e-4)

    # Raw-mean failure touches exactly the failed sensors' 8 columns.
    train_mean = X_raw[tr].mean(axis=0)
    failed = (3, 11)
    X_fail = corrupt_raw_with_outer_train_mean(X_raw[te], failed, train_mean)
    cols = [p for s in failed for p in sensor_feature_positions(s)]
    assert np.allclose(X_fail[:, cols], train_mean[cols])
    others = np.setdiff1d(np.arange(128), cols)
    assert np.array_equal(X_fail[:, others], X_raw[te][:, others])

    # The two shared corruption modes agree on standardised raw inputs.
    mu, sd = X_raw[tr].mean(0), X_raw[tr].std(0)
    a = (X_fail - mu) / sd
    b = corrupt_standardised_with_zero((X_raw[te] - mu) / sd, failed)
    assert np.allclose(a, b, atol=1e-4)

    patterns = load_failure_patterns(PATTERNS_PATH)
    assert {k: len(v) for k, v in patterns.items()} == {0: 1, 1: 16, 2: 20, 4: 20, 8: 20}

    for inner in make_inner_temporal_splits(df, tr):
        it, iv = inner["train_indices"], inner["val_indices"]
        assert np.isin(it, tr).all() and np.isin(iv, tr).all()
        assert batch[it].max() < batch[iv].min()
        assert set(np.unique(y[it])) == set(range(6)), inner["inner_split"]
    print(f"features {Z.shape[1]} dims, raw-mean failure, patterns and inner splits OK")


if __name__ == "__main__":
    test_metrics_match_sklearn()
    test_features_failure_and_splits()
