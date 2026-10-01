"""Evaluation helpers.

Every method produces an anomaly score per window (higher means more anomalous).
Three kinds of numbers are reported:

* AUROC and AUPRC: threshold free, computed on the test scores.
* F1 at a calibrated threshold: the threshold is a percentile of the scores on
  held-out NORMAL validation windows. No test label is used to choose it.
* Oracle best F1: the best F1 over all possible thresholds, chosen with the test
  labels. This is an optimistic upper bound and is reported only for reference.

Two calibration modes are available:

* "normal-file": threshold from a held-out part of the anomaly-free recording.
* "recording": every test recording is normalised with statistics of its own first
  n_calib rows, which the SKAB benchmark defines as anomaly free, and the threshold
  is a percentile of the pooled calibration scores. No test label is used.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calibrated_threshold(val_scores, quantile):
    return float(np.percentile(val_scores, quantile))


def threshold_metrics(scores, y, threshold):
    pred = (scores > threshold).astype(int)
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "flagged_fraction": float(pred.mean()),
    }


def threshold_free_metrics(scores, y):
    return {
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
    }


def oracle_best_f1(scores, y):
    precision, recall, _ = precision_recall_curve(y, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(f1.max())


def all_anomalous_metrics(y):
    """Trivial predictor that flags every window. Any useful method must beat it."""
    pred = np.ones_like(y)
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "flagged_fraction": 1.0,
    }


def evaluate_scores(val_scores, test_scores, y, quantile):
    thr = calibrated_threshold(val_scores, quantile)
    out = {"threshold": thr}
    out.update(threshold_metrics(test_scores, y, thr))
    out.update(threshold_free_metrics(test_scores, y))
    out["oracle_best_f1"] = oracle_best_f1(test_scores, y)
    return out


def recording_normalized_scores(scores, rec_ids, end_idx, n_calib):
    """Z-score every recording with the mean and std of its first n_calib rows."""
    z = np.empty(len(scores), dtype=float)
    for r in np.unique(rec_ids):
        m = rec_ids == r
        calib = scores[m & (end_idx < n_calib)]
        z[m] = (scores[m] - calib.mean()) / (calib.std() + 1e-12)
    return z


def evaluate_recording_calibrated(scores, y, rec_ids, end_idx, n_calib, quantile):
    z = recording_normalized_scores(scores, rec_ids, end_idx, n_calib)
    thr = float(np.percentile(z[end_idx < n_calib], quantile))
    out = {"threshold": thr}
    out.update(threshold_metrics(z, y, thr))
    out.update(threshold_free_metrics(z, y))
    out["oracle_best_f1"] = oracle_best_f1(z, y)
    return out


def position_prior_metrics(y, end_idx, start=570, stop=970):
    """Diagnostic only: flag rows start..stop-1 of every recording.

    The valve1 recordings of SKAB contain one contiguous anomaly of 337 to 405 rows that
    begins between row 566 and row 578 in every file. This rule uses that structure and no signal at all.
    """
    pred = ((end_idx >= start) & (end_idx < stop)).astype(int)
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "flagged_fraction": float(pred.mean()),
    }
