# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Parts IV and VI: how a model is judged and how two models are compared. The metrics are scikit-learn's, the corrected resampled t is the project's own ``corrected_resampled_t`` and ``naive_paired_t``, and
the optimism of choosing the best of many models is simulated on models that know nothing."""

import numpy as np
from sklearn.metrics import (
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from cypmode.challenge.model_comparison import corrected_resampled_t, naive_paired_t


def draw_scores(prevalence: float, separation: float, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Labels and scores: the scores of the positives are normal with mean ``separation``, those of the negatives with mean 0, both with standard deviation 1."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < prevalence).astype(int)
    if labels.min() == labels.max():
        labels[0], labels[1] = 0, 1
    return labels, rng.normal(separation * labels, 1.0)


def metrics_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    called = (scores > threshold).astype(int)
    return {"AUC": float(roc_auc_score(labels, scores)), "MCC": float(matthews_corrcoef(labels, called)) if called.min() != called.max() else 0.0,
            "precision": float(precision_score(labels, called, zero_division=0)), "recall": float(recall_score(labels, called, zero_division=0)), "positives called": int(called.sum())}


def spread(prevalence: float, separation: float, n: int, threshold: float, repeats: int = 300) -> dict:
    """The standard deviation of the AUC and of the MCC over ``repeats`` fresh samples of the same size: how much a score moves by sampling alone."""
    auc, mcc = [], []
    for seed in range(repeats):
        labels, scores = draw_scores(prevalence, separation, n, 1000 + seed)
        m = metrics_at(labels, scores, threshold)
        auc.append(m["AUC"])
        mcc.append(m["MCC"])
    return {"AUC mean": float(np.mean(auc)), "AUC sd": float(np.std(auc, ddof=1)), "MCC mean": float(np.mean(mcc)), "MCC sd": float(np.std(mcc, ddof=1))}


def compare(differences, n_train: int, n_test: int) -> dict:
    """The naive paired t and the corrected resampled t of the same differences (6.5)."""
    naive, corrected = naive_paired_t(differences), corrected_resampled_t(differences, n_train, n_test)
    return {"mean difference": corrected.mean_difference, "naive t": naive.t, "naive p": naive.p_value, "corrected t": corrected.t, "corrected p": corrected.p_value,
            "variance factor": 1 / len(differences) + n_test / n_train, "naive factor": 1 / len(differences), "folds better": corrected.n_better, "splits": corrected.n_folds}


def selection_optimism(n_models: int, n: int, repeats: int = 200, seed: int = 0) -> dict:
    """Models that know nothing: the best of ``n_models`` on a validation set looks better than chance, and on fresh data it is exactly at chance (6.1, 6.2)."""
    rng = np.random.default_rng(seed)
    best_validation, test = [], []
    for _ in range(repeats):
        labels_v, labels_t = rng.integers(0, 2, n), rng.integers(0, 2, n)
        if labels_v.min() == labels_v.max() or labels_t.min() == labels_t.max():
            continue
        scores_v, scores_t = rng.normal(size=(n_models, n)), rng.normal(size=(n_models, n))
        validation = [roc_auc_score(labels_v, s) for s in scores_v]
        chosen = int(np.argmax(validation))
        best_validation.append(validation[chosen])
        test.append(roc_auc_score(labels_t, scores_t[chosen]))
    return {"best validation AUC": float(np.mean(best_validation)), "its AUC on fresh data": float(np.mean(test))}


def calls_by_multiple(prevalence: float, separation: float, n: int, multiples, seed: int = 11) -> list[dict]:
    """Section 8.4: the project calls the top share (a multiple of the positive rate) of the scored compounds positive, with its own ``call_top_fraction``; the metrics at each multiple."""
    from cypmode.challenge.nested import call_top_fraction

    labels, scores = draw_scores(prevalence, separation, n, seed)
    rows = []
    for m in multiples:
        called = call_top_fraction(scores, min(1.0, m * prevalence))
        rows.append({"multiple of the positive rate": float(m), "share called": float(called.mean()), "MCC": float(matthews_corrcoef(labels, called)) if called.min() != called.max() else 0.0,
                     "precision": float(precision_score(labels, called, zero_division=0)), "recall": float(recall_score(labels, called, zero_division=0))})
    return rows
