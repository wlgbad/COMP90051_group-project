"""
Shared feature construction (proposed for all three models; needs group sign-off).

Spec 2.c does not allow feeding the raw table through a model with only simple
preprocessing, so every model should take FeaturePipeline output instead of the
standardised x1..x128. Standardisation uses shared_protocol.TrainOnlyStandardScaler.

Raw layout: sensor j (1..16) owns raw columns 8(j-1) .. 8j-1 (0-based), in order
    DR, |DR|, EMAi_0.001, EMAi_0.01, EMAi_0.1, EMAd_0.001, EMAd_0.01, EMAd_0.1

All ratios are written as differences of signed logs, slog(a) - slog(b) with
slog(x) = sign(x) * log1p(|x|), which behaves like log(a/b) for large responses
but never divides by zero.

Families (396 features in total):
    base_slog      128  slog of every raw feature
    dr_pair        120  slog(DR_i) - slog(DR_j), i < j  (cancels concentration)
    dr_comp         16  |DR_i| / sum_j |DR_j|          (array composition profile)
    trans_balance   48  slog|EMAi_a| - slog|EMAd_a|    (rise vs decay, per alpha)
    trans_alpha     64  slog|EMA_0.1| - slog|EMA_0.001|, slog|EMA_0.01| - slog|EMA_0.001|
                         for EMAi and EMAd               (rough response-time constant)
    array_stats      4  mean, std, skew, range of slog(DR) over the 16 sensors
    max_sensor      16  one-hot index of the max-responding sensor

Usage (sensor failure is applied to the RAW columns first, so derived
features inherit it; use corruption_mode="raw_mean" in shared_robustness)
    pipe = FeaturePipeline().fit(X_raw_train)
    Z_train = pipe.transform(X_raw_train)
    predict_fn = lambda X_raw: model.predict(pipe.transform(X_raw))
"""

from __future__ import annotations

import numpy as np

from shared_protocol import FEATURES_PER_SENSOR as N_PER_SENSOR
from shared_protocol import N_SENSORS, TrainOnlyStandardScaler

DR, ABS_DR = 0, 1
EMAI = (2, 3, 4)  # alpha = 0.001, 0.01, 0.1
EMAD = (5, 6, 7)
ALPHAS = ("0.001", "0.01", "0.1")


def slog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _pairs() -> list[tuple[int, int]]:
    return [(i, j) for i in range(N_SENSORS) for j in range(i + 1, N_SENSORS)]


def construct_features(X_raw: np.ndarray) -> np.ndarray:
    """Deterministic (stateless) feature construction: (n, 128) -> (n, 396)."""
    if X_raw.ndim != 2 or X_raw.shape[1] != N_SENSORS * N_PER_SENSOR:
        raise ValueError(f"Expected (n, 128) raw features, got {X_raw.shape}")

    S = X_raw.reshape(-1, N_SENSORS, N_PER_SENSOR)
    log_dr = slog(S[:, :, DR])
    abs_dr = np.abs(S[:, :, DR])

    blocks = [slog(X_raw)]

    i_idx, j_idx = np.array(_pairs()).T
    blocks.append(log_dr[:, i_idx] - log_dr[:, j_idx])

    blocks.append(abs_dr / (abs_dr.sum(axis=1, keepdims=True) + 1e-12))

    log_emai = slog(np.abs(S[:, :, EMAI]))  # (n, 16, 3)
    log_emad = slog(np.abs(S[:, :, EMAD]))
    blocks.append((log_emai - log_emad).reshape(len(X_raw), -1))

    alpha_ratios = []
    for log_ema in (log_emai, log_emad):
        alpha_ratios.append(log_ema[:, :, 2] - log_ema[:, :, 0])
        alpha_ratios.append(log_ema[:, :, 1] - log_ema[:, :, 0])
    blocks.append(np.concatenate(alpha_ratios, axis=1))

    mean = log_dr.mean(axis=1)
    std = log_dr.std(axis=1)
    centred = log_dr - mean[:, None]
    skew = (centred**3).mean(axis=1) / (std**3 + 1e-12)
    span = log_dr.max(axis=1) - log_dr.min(axis=1)
    blocks.append(np.column_stack([mean, std, skew, span]))

    onehot = np.zeros((len(X_raw), N_SENSORS))
    onehot[np.arange(len(X_raw)), abs_dr.argmax(axis=1)] = 1.0
    blocks.append(onehot)

    Z = np.concatenate(blocks, axis=1)
    if not np.isfinite(Z).all():
        raise ValueError("Non-finite constructed features.")
    return Z


def feature_names() -> list[str]:
    kinds = ["DR", "absDR", "EMAi0.001", "EMAi0.01", "EMAi0.1",
             "EMAd0.001", "EMAd0.01", "EMAd0.1"]
    names = [f"slog_s{s + 1}_{k}" for s in range(N_SENSORS) for k in kinds]
    names += [f"drpair_s{i + 1}_s{j + 1}" for i, j in _pairs()]
    names += [f"drcomp_s{s + 1}" for s in range(N_SENSORS)]
    names += [f"balance_s{s + 1}_a{a}" for s in range(N_SENSORS) for a in ALPHAS]
    for kind in ("EMAi", "EMAd"):
        names += [f"alpha_{kind}_0.1vs0.001_s{s + 1}" for s in range(N_SENSORS)]
        names += [f"alpha_{kind}_0.01vs0.001_s{s + 1}" for s in range(N_SENSORS)]
    names += ["dr_mean", "dr_std", "dr_skew", "dr_range"]
    names += [f"maxsensor_s{s + 1}" for s in range(N_SENSORS)]
    return names


FAMILY_SIZES = {
    "base_slog": 128, "dr_pair": 120, "dr_comp": 16, "trans_balance": 48,
    "trans_alpha": 64, "array_stats": 4, "max_sensor": 16,
}


def feature_families() -> np.ndarray:
    """Family label of each constructed column (for per-family diagnostics)."""
    return np.repeat(list(FAMILY_SIZES), list(FAMILY_SIZES.values()))


class FeaturePipeline:
    """construct_features -> StandardScaler, fitted on raw training rows."""

    def fit(self, X_raw_train: np.ndarray) -> "FeaturePipeline":
        self.scaler_ = TrainOnlyStandardScaler().fit(construct_features(X_raw_train))
        return self

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        return self.scaler_.transform(construct_features(X_raw))

    def fit_transform(self, X_raw_train: np.ndarray) -> np.ndarray:
        Z = construct_features(X_raw_train)
        self.scaler_ = TrainOnlyStandardScaler().fit(Z)
        return self.scaler_.transform(Z)


assert len(feature_names()) == sum(FAMILY_SIZES.values()) == 396
