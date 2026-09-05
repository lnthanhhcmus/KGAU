"""Classification metrics for evaluating KG models."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_prob=None, zero_division=0) -> dict:
    """Compute standard binary classification metrics for triple classification."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    unique_classes = np.unique(y_true)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=zero_division),
        'recall': recall_score(y_true, y_pred, zero_division=zero_division),
        'f1': f1_score(y_true, y_pred, zero_division=zero_division),
    }

    if y_prob is None:
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0
        return metrics

    if unique_classes.size < 2:
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0
        return metrics

    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics['roc_auc'] = 0.0

    try:
        metrics['pr_auc'] = average_precision_score(y_true, y_prob)
    except Exception:
        metrics['pr_auc'] = 0.0

    return metrics


def find_global_threshold(y_true, y_prob, n_thresholds=100) -> float:
    """Find a global threshold that maximizes validation accuracy.

    When multiple thresholds tie on accuracy, prefer the one with higher F1 so
    precision/recall are not degenerate on balanced label splits.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_true.size == 0:
        raise ValueError('y_true must not be empty')
    if y_prob.size == 0:
        raise ValueError('y_prob must not be empty')

    if np.allclose(y_prob.min(), y_prob.max()):
        return float(y_prob[0])

    thresholds = np.linspace(float(y_prob.min()), float(y_prob.max()), n_thresholds)
    best_acc = -1.0
    best_f1 = -1.0
    best_t = float(thresholds[0])
    for t in thresholds:
        y_pred = (y_prob > t).astype(int)
        acc = float((y_pred == y_true).mean())
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        if acc > best_acc or (acc == best_acc and f1 > best_f1):
            best_acc = acc
            best_f1 = f1
            best_t = float(t)
    return best_t


def classification_metrics_from_probs(y_true, y_prob, threshold=None, n_thresholds=100, zero_division=0) -> dict:
    """Convenience helper that thresholds probabilities before computing metrics."""

    if threshold is None:
        threshold = find_global_threshold(y_true, y_prob, n_thresholds=n_thresholds)
    y_pred = (np.asarray(y_prob) > threshold).astype(int)
    return classification_metrics(y_true, y_pred, y_prob=y_prob, zero_division=zero_division)


# Backward-compatible alias.
binary_classification_metrics = classification_metrics
