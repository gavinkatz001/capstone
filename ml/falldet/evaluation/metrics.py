"""Evaluation metrics for fall detection: TPR, FPR, AUC, confusion matrix."""

import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute all metrics at a given threshold.

    Args:
        y_true: (N,) binary ground truth labels.
        y_prob: (N,) predicted probabilities (after sigmoid).
        threshold: classification threshold.

    Returns:
        Dictionary of metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity / recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # F2 score: weights recall 2x over precision
    f2 = ((1 + 4) * precision * tpr) / (4 * precision + tpr) if (4 * precision + tpr) > 0 else 0.0

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0

    return {
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "precision": float(precision),
        "specificity": float(specificity),
        "f1": float(f1),
        "f2": float(f2),
        "roc_auc": float(roc_auc),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_tpr: float = 0.95,
    max_fpr: float = 0.10,
    n_steps: int = 100,
) -> tuple[float, dict]:
    """Sweep thresholds to find best operating point.

    Strategy: Find the threshold that achieves target_tpr with minimum FPR.
    If no threshold achieves target_tpr within max_fpr, return the best trade-off.

    Returns:
        (best_threshold, metrics_at_threshold)
    """
    thresholds = np.linspace(0.0, 1.0, n_steps + 1)
    best_threshold = 0.5
    best_score = -1.0

    for t in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=t)
        # Primary: achieve target TPR. Secondary: minimize FPR.
        if m["tpr"] >= target_tpr:
            score = -m["fpr"]  # lower FPR is better
            if score > best_score:
                best_score = score
                best_threshold = t

    # If nothing achieved target TPR, maximize F2 (recall-weighted)
    if best_score == -1.0:
        for t in thresholds:
            m = compute_metrics(y_true, y_prob, threshold=t)
            if m["f2"] > best_score:
                best_score = m["f2"]
                best_threshold = t

    return best_threshold, compute_metrics(y_true, y_prob, best_threshold)


def compute_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute ROC curve data for plotting."""
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr_arr, tpr_arr)
    return {
        "fpr": fpr_arr.tolist(),
        "tpr": tpr_arr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": float(roc_auc),
    }
