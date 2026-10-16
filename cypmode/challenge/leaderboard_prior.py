"""Positive rate of the blinded test set, recovered from the public leaderboard.

Every leaderboard row publishes accuracy (A), precision (P) and recall (R) for
one prediction file. Those three numbers pin down the positive rate p of the
scored set: per unit of p*n, FN = 1-R, TP = R, predicted positives = R/P,
FP = R/P - R, so the error rate is p * ((1-R) + (R/P - R)) = 1 - A, i.e.

    p = (1 - A) / (1 - R + R/P - R)

All rows score the same labels, so p must agree across rows; the spread is
reported as a check (it was ~0.000 on the real boards). The median over rows is
the estimate. This replaces typing a prevalence into code: it is derived from
the exported board, with the row count and spread kept as provenance.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def implied_prevalence(accuracy, precision, recall) -> np.ndarray:
    accuracy, precision, recall = (np.asarray(v, dtype=float) for v in (accuracy, precision, recall))
    with np.errstate(divide="ignore", invalid="ignore"):
        errors_per_unit = (1 - recall) + (recall / precision - recall)
        return np.where((precision > 0) & (recall > 0) & (errors_per_unit > 0), (1 - accuracy) / errors_per_unit, np.nan)


def estimate_from_board(board: pd.DataFrame) -> dict:
    """``board`` is one per-isoform TDI leaderboard export (columns Accuracy, Precision, Recall)."""
    p = implied_prevalence(board["Accuracy"], board["Precision"], board["Recall"])
    p = p[np.isfinite(p)]
    if len(p) == 0:
        raise ValueError("no usable rows (every row needs non-zero precision and recall)")
    return {"prevalence": float(np.median(p)), "spread": float(np.std(p)), "n_rows": len(p)}


def estimate_from_csv(path: str | Path) -> dict:
    return estimate_from_board(pd.read_csv(path).dropna(subset=["Username"]))
