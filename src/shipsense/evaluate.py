"""Metrics shared by every model so comparisons are apples-to-apples."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, roc_auc_score, roc_curve,
)


def best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Pick the cut-off that maximises Youden's J (TPR - FPR) on the VALIDATION set.

    Why not maximise F1? With ~57% of orders late, F1 is maximised by flagging
    almost everything as late - a useless model. Youden's J balances catching late
    orders against false alarms. Tuning on validation (never test) keeps test honest.
    """
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    return float(np.clip(thresholds[int(np.argmax(tpr - fpr))], 0.0, 1.0))


def score(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, pred)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "threshold": round(threshold, 3),
    }
