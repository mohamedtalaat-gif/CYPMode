# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Parts VI to VIII, the logic: a confusion matrix under another prevalence (6.4), the false alarms of two ways to compare many models (6.6), a hold-out chosen by similarity (6.8), a small model on a few blocks (7.1, 7.2),
the coefficients of a stack that stops where it happens to be (8.3), the calibration of a score (8.5) and the checks of a file of calls (8.6)."""

import numpy as np
import pandas as pd
from scipy import stats

from cypmode.challenge.model_comparison import corrected_resampled_t
from cypmode.challenge.multiplicity import holm

TEST_COLUMNS = ["SMILES", "Molecule_Name", "CYP2D6_is_TDI", "CYP3A4_is_TDI"]


def oof_scores(frame: pd.DataFrame, candidate: str, repeat: int = 0) -> tuple[np.ndarray, np.ndarray]:
    return frame["y"].to_numpy().astype(int), frame[f"{candidate}|{repeat}"].to_numpy()


def weighted_mcc(y, scores, call_rate: float, target_prevalence: float | None = None) -> float:
    """The MCC when the top ``call_rate`` of the compounds is called, with the two classes weighted as if the positive rate were ``target_prevalence`` (the training rate when None)."""
    y = np.asarray(y).astype(int)
    pi = y.mean()
    w1, w0 = (1.0, 1.0) if target_prevalence is None else (target_prevalence / pi, (1 - target_prevalence) / (1 - pi))
    called = scores >= np.quantile(scores, 1 - call_rate)
    tp, fp = w1 * (called & (y == 1)).sum(), w0 * (called & (y == 0)).sum()
    fn, tn = w1 * (~called & (y == 1)).sum(), w0 * (~called & (y == 0)).sum()
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denominator) if denominator else 0.0


def false_alarm_rates(k: int, folds: int, rho: float, repeats: int, seed: int) -> dict:
    """k models of the same true skill scored on the same folds. The folds of one model are not independent (their training sets overlap), which is modelled by a part rho of the variance of its fold scores
    that is common to all its folds. The share of experiments in which some pair is declared different at 0.05 by Tukey's HSD, which treats the scores as independent, and by the corrected resampled t with the correction of Holm."""
    rng = np.random.default_rng(seed)
    tukey = corrected = 0
    for _ in range(repeats):
        scores = rng.normal(0, np.sqrt(rho), k)[None, :] + rng.normal(0, np.sqrt(1 - rho), (folds, k))
        if (stats.tukey_hsd(*scores.T).pvalue[np.triu_indices(k, 1)] < 0.05).any():
            tukey += 1
        pairs = [corrected_resampled_t(list(scores[:, i] - scores[:, j]), 4000, 1000).p_value for i in range(k) for j in range(i + 1, k)]
        if (np.array(holm(pairs)) < 0.05).any():
            corrected += 1
    return {"Tukey HSD": tukey / repeats, "corrected t with Holm": corrected / repeats}


def holdout_by_similarity(similarity: pd.DataFrame, threshold: float) -> dict:
    train = similarity[similarity["set"].str.startswith("training")]["max similarity"].to_numpy()
    test = similarity[similarity["set"].str.startswith("test")]["max similarity"].to_numpy()
    kept = train[train >= threshold]
    return {"kept": len(kept), "share kept": float(len(kept) / len(train)), "KS distance to the test set": float(stats.ks_2samp(kept, test).statistic) if len(kept) > 5 else float("nan"), "median kept": float(np.median(kept)) if len(kept) else float("nan"), "median test": float(np.median(test))}


def cross_validated(matrix: np.ndarray, labels: np.ndarray, trees: int, folds: int = 5, seed: int = 0) -> dict:
    """AUC and the MCC at the 0.5 cut of a gradient-boosted model, by stratified cross-validation."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import matthews_corrcoef, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    out, called = np.zeros(len(labels)), np.zeros(len(labels), dtype=int)
    for fit, test in StratifiedKFold(folds, shuffle=True, random_state=seed).split(matrix, labels):
        model = HistGradientBoostingClassifier(max_iter=trees, learning_rate=0.1, random_state=seed).fit(matrix[fit], labels[fit])
        out[test] = model.predict_proba(matrix[test])[:, 1]
        called[test] = model.predict(matrix[test])
    return {"AUC": float(roc_auc_score(labels, out)), "MCC": float(matthews_corrcoef(labels, called))}


def stack_coefficients(scores: np.ndarray, y: np.ndarray, tol: float) -> dict:
    """The coefficients of a logistic blend of the candidate scores when the solver is allowed to stop at a tolerance ``tol``, next to the converged ones: a solver that stops early stops where it happens to be."""
    from sklearn.linear_model import LogisticRegression

    early = LogisticRegression(C=1.0, tol=tol, max_iter=1000).fit(scores, y)
    converged = LogisticRegression(C=1.0, tol=1e-12, max_iter=100000).fit(scores, y)
    return {"early": early.coef_[0], "converged": converged.coef_[0], "iterations": int(early.n_iter_[0]), "iterations to converge": int(converged.n_iter_[0])}


def calibration(y, scores, method: str, bins: int, seed: int = 0) -> dict:
    """The reliability curve of a score and its expected calibration error, for the raw score or after a calibration fitted on one half and applied to the other."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    y, scores = np.asarray(y).astype(int), np.asarray(scores, dtype=float)
    p = scores.copy()
    if method != "raw":
        order = np.random.default_rng(seed).permutation(len(y))
        a, b = order[: len(y) // 2], order[len(y) // 2:]
        model = IsotonicRegression(out_of_bounds="clip").fit(scores[a], y[a]) if method == "isotonic" else LogisticRegression().fit(scores[a].reshape(-1, 1), y[a])
        p[b] = model.predict(scores[b]) if method == "isotonic" else model.predict_proba(scores[b].reshape(-1, 1))[:, 1]
        keep = b
    else:
        keep = np.arange(len(y))
    edges = np.quantile(p[keep], np.linspace(0, 1, bins + 1))
    index = np.clip(np.searchsorted(edges, p[keep], side="right") - 1, 0, bins - 1)
    table = pd.DataFrame({"predicted": [p[keep][index == i].mean() for i in range(bins) if (index == i).any()], "observed": [y[keep][index == i].mean() for i in range(bins) if (index == i).any()], "n": [int((index == i).sum()) for i in range(bins) if (index == i).any()]})
    return {"table": table, "ece": float((table["n"] * (table["predicted"] - table["observed"]).abs()).sum() / table["n"].sum()), "brier": float(np.mean((p[keep] - y[keep]) ** 2))}


def check_calls(frame: pd.DataFrame, test: pd.DataFrame) -> dict[str, bool]:
    """The six checks that the audit makes on a TDI file of calls against the blinded test file."""
    have = all(c in frame.columns for c in TEST_COLUMNS)
    return {"750 rows": len(frame) == len(test), "the four columns are there": have, "no missing value": bool(frame[TEST_COLUMNS].notna().all().all()) if have else False,
            "calls are boolean": all(set(frame[c].unique()) <= {True, False} for c in ("CYP2D6_is_TDI", "CYP3A4_is_TDI") if c in frame.columns) and have,
            "names in the order of the test file": frame["Molecule_Name"].tolist() == test["Molecule_Name"].tolist() if "Molecule_Name" in frame.columns else False, "SMILES equal the test file": frame["SMILES"].tolist() == test["SMILES"].tolist() if "SMILES" in frame.columns else False}


def _similarity_matrix(isoform: str, n: int):
    from functools import lru_cache  # noqa: F401

    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    from simulators import chem

    table = chem.labelled(isoform).head(n).reset_index(drop=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in table["SMILES"]]
    matrix = np.array([DataStructs.BulkTanimotoSimilarity(f, fps) for f in fps])
    return matrix, table["label"].to_numpy().astype(int)


def _knn_scores(similarity, y, train, test, k: int) -> np.ndarray:
    sims = similarity[np.ix_(test, train)]
    top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    weights = np.take_along_axis(sims, top, axis=1)
    return (weights * y[train][top]).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)


def _cv_auc(similarity, y, indices, k: int, folds: int, seed: int) -> float:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    scores = np.zeros(len(indices))
    for fit, test in StratifiedKFold(folds, shuffle=True, random_state=seed).split(indices, y[indices]):
        scores[test] = _knn_scores(similarity, y, indices[fit], indices[test], k)
    return float(roc_auc_score(y[indices], scores))


def selection_bias(isoform: str, n: int, folds: int, seed: int, ks=range(1, 31)) -> dict:
    """Nearest-neighbour models with k = 1..30 neighbours, scored by cross-validation. The best of them looks better under the same cross-validation that chose it than under a nested one, in which the choice is made inside each outer training set and the outer test set is never used for it."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    similarity, y = _similarity_matrix(isoform, n)
    everyone = np.arange(len(y))
    curve = pd.DataFrame({"k": list(ks), "cross-validated AUC": [_cv_auc(similarity, y, everyone, k, folds, seed) for k in ks]})
    outer = np.zeros(len(y))
    for fit, test in StratifiedKFold(5, shuffle=True, random_state=seed + 1).split(everyone, y):
        best = max(ks, key=lambda k: _cv_auc(similarity, y, everyone[fit], k, folds, seed))
        outer[test] = _knn_scores(similarity, y, everyone[fit], everyone[test], best)
    return {"curve": curve, "best k": int(curve.loc[curve["cross-validated AUC"].idxmax(), "k"]), "best cross-validated AUC": float(curve["cross-validated AUC"].max()), "nested AUC": float(roc_auc_score(y, outer))}
