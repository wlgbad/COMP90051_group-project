"""
Classification metrics implemented from scratch.
"""

from __future__ import annotations
import numpy as np

N_CLASSES = 6


def confusion_matrix_from_scratch(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = N_CLASSES,
) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1

    return cm


def metrics_from_confusion_matrix(cm: np.ndarray) -> dict[str, float]:
    cm = np.asarray(cm, dtype=np.float64)
    n_classes = cm.shape[0]

    total = cm.sum()
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0

    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)

    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        denom = precision[c] + recall[c]
        f1[c] = 2 * precision[c] * recall[c] / denom if denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }


def classification_metrics_from_scratch(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = N_CLASSES,
) -> tuple[dict[str, float], np.ndarray]:
    cm = confusion_matrix_from_scratch(y_true, y_pred, n_classes)
    return metrics_from_confusion_matrix(cm), cm
