# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Make cross-validation resemble the blinded test set.

Random CV is optimistic-in-the-wrong-way for this challenge: the 750 test
compounds are analog expansions of top screening hits, so most of them have a
close training neighbor, whereas a random validation fold of the training data
mostly does not (median nearest-training Tanimoto about 0.44 for a CYP3A4 CV
fold vs 0.51 for the test set; 33% vs 54% of compounds at or above 0.5). Models
that read across from close analogs are therefore under-credited by plain CV.

This module re-weights validation predictions so the distribution of
nearest-training-neighbor similarity matches the test set's (a one-dimensional
covariate-shift correction, far better conditioned than full-fingerprint
importance weights: an adversarial train-vs-test classifier separates the two
sets with AUC 0.96 and leaves an effective sample size of ~220), and can also
re-weight classes to a different positive rate.

It does NOT close the whole gap: even with both corrections, the official
LightGBM baseline's CV MCC stays about 0.08-0.13 below its leaderboard score,
so absolute CV numbers must never be read as test forecasts -- only the
ordering of candidate models under matched conditions is meaningful.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

DEFAULT_BINS = np.array([0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])


def tanimoto_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Pairwise Tanimoto similarity of binary fingerprint matrices (rows = compounds)."""
    b = a if b is None else b
    intersection = a @ b.T
    union = a.sum(axis=1)[:, None] + b.sum(axis=1)[None, :] - intersection
    return (intersection / np.maximum(union, 1e-9)).astype(np.float32)


def nearest_neighbor_similarity(query: np.ndarray, pool: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    """Highest Tanimoto similarity of every query row to the pool rows. With
    ``exclude_self`` the query and pool are the same compounds in the same order
    and each row's own entry is ignored.
    """
    similarity = tanimoto_matrix(query, pool)
    if exclude_self:
        np.fill_diagonal(similarity, -1.0)
    return similarity.max(axis=1)


def similarity_weights(validation_nn: np.ndarray, test_nn: np.ndarray, bins: np.ndarray = DEFAULT_BINS) -> np.ndarray:
    """Per-validation-compound weights that make the histogram of nearest-
    neighbor similarity match the test set's. Bins the validation set never
    populates carry no weight, so callers should treat the weights as relative. If
    the two profiles share no bin at all the weights fall back to uniform.
    """
    validation_nn = np.asarray(validation_nn, dtype=float)
    test_hist = np.histogram(test_nn, bins)[0] / max(len(test_nn), 1)
    validation_hist = np.histogram(validation_nn, bins)[0] / max(len(validation_nn), 1)
    bin_weight = np.where(validation_hist > 0, test_hist / np.maximum(validation_hist, 1e-12), 0.0)
    weights = bin_weight[np.clip(np.digitize(validation_nn, bins) - 1, 0, len(bin_weight) - 1)]
    if weights.sum() <= 0:  # no overlap between the two profiles: matching is impossible, so do not pretend
        return np.ones(len(validation_nn))
    return weights


def prior_weights(y: np.ndarray, target_prevalence: float) -> np.ndarray:
    """Class weights that shift the positive rate of ``y`` to ``target_prevalence``."""
    y = np.asarray(y).astype(int)
    observed = y.mean()
    if not 0 < observed < 1:
        raise ValueError("y must contain both classes")
    return np.where(y == 1, target_prevalence / observed, (1 - target_prevalence) / (1 - observed))


def weighted_mcc(y: np.ndarray, predicted: np.ndarray, weights: np.ndarray | None = None) -> float:
    y = np.asarray(y).astype(int)
    predicted = np.asarray(predicted).astype(int)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    tp = (w * ((y == 1) & (predicted == 1))).sum()
    tn = (w * ((y == 0) & (predicted == 0))).sum()
    fp = (w * ((y == 0) & (predicted == 1))).sum()
    fn = (w * ((y == 1) & (predicted == 0))).sum()
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denominator) if denominator > 0 else 0.0


def weighted_auc(y: np.ndarray, score: np.ndarray, weights: np.ndarray | None = None) -> float:
    return float(roc_auc_score(y, score, sample_weight=weights))


class SimilarityMatcher:
    """``matcher(rows, reference_rows)`` -> weights that make the nearest-neighbor-similarity
    profile of ``rows`` (scored against ``reference_rows`` as their training set) look like
    the blinded test set's profile against the same reference.

    The test profile is computed against random 80% subsets of the reference (a CV
    fold's training portion holds about 80% of the pool), averaged over ``n_draws``.
    Everything is derived from fingerprints; no label and no hand-set number is involved.
    """

    def __init__(self, fingerprints: np.ndarray, test_rows: np.ndarray, bins: np.ndarray = DEFAULT_BINS, n_draws: int = 5, seed: int = 0):
        self.fingerprints, self.test_rows, self.bins, self.n_draws, self.seed = fingerprints, np.asarray(test_rows), bins, n_draws, seed
        self._test_profile: dict[bytes, np.ndarray] = {}

    def _profile(self, reference_rows: np.ndarray) -> np.ndarray:
        key = np.asarray(reference_rows, dtype=np.int64).tobytes()
        if key not in self._test_profile:
            similarity = tanimoto_matrix(self.fingerprints[self.test_rows], self.fingerprints[reference_rows])
            rng = np.random.default_rng(self.seed)
            self._test_profile[key] = np.concatenate([similarity[:, rng.random(similarity.shape[1]) < 0.8].max(axis=1) for _ in range(self.n_draws)])
        return self._test_profile[key]

    def __call__(self, rows: np.ndarray, reference_rows: np.ndarray) -> np.ndarray:
        nn = nearest_neighbor_similarity(self.fingerprints[rows], self.fingerprints[reference_rows])
        return similarity_weights(nn, self._profile(reference_rows), self.bins)
