# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Section 8.7: the out-of-fold predictions of the candidates of the nested plan, as stored. A candidate is scored on compounds that no model of it was fitted on; here the stored scores are ranked, called at a share,
and read as a confusion matrix, an ROC curve and an AUC per outer fold. Nothing is refitted."""

import numpy as np
import pandas as pd


def auc_of(y: np.ndarray, scores) -> float:
    """The area under the ROC curve as the probability that a positive outranks a negative (ties count one half)."""
    ranks = pd.Series(scores).rank().to_numpy()
    positives = int(y.sum())
    negatives = len(y) - positives
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def folds_of(frame: pd.DataFrame) -> int:
    return int(frame[[c for c in frame.columns if c.startswith("fold|")]].to_numpy().max()) + 1


def candidate_table(oof: dict, candidates: pd.DataFrame, enzyme: str, fold: str = "all folds", needed=()) -> pd.DataFrame:
    """One row per candidate that uses every block of ``needed``: the mean, lowest and highest AUC over the repeats, on all compounds or on those of one outer fold."""
    frame = oof[enzyme]
    y = frame["y"].to_numpy()
    used = dict(zip(candidates["candidate"], candidates["blocks"].str.split("+").map(lambda parts: {p for p in parts if p}), strict=True))
    rows = []
    for column in frame.columns:
        if column == "y" or column.startswith("fold|"):
            continue
        name, repeat = column.rsplit("|", 1)
        if not set(needed) <= used.get(name, set()):
            continue
        keep = np.ones(len(y), dtype=bool) if fold == "all folds" else (frame[f"fold|{repeat}"] == int(fold)).to_numpy()
        rows.append({"candidate": name, "repeat": int(repeat), "auc": auc_of(y[keep], frame[column].to_numpy()[keep]), "compounds": int(keep.sum())})
    if not rows:
        return pd.DataFrame(columns=["candidate", "mean", "low", "high", "compounds", "family", "blocks"])
    long = pd.DataFrame(rows)
    table = long.groupby("candidate").agg(mean=("auc", "mean"), low=("auc", "min"), high=("auc", "max"), compounds=("compounds", "first")).reset_index()
    return table.merge(candidates, on="candidate", how="left").sort_values("mean", ascending=False).reset_index(drop=True)


def roc_points(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    hits = y[order]
    return np.concatenate([[0.0], np.cumsum(1 - hits) / (len(hits) - hits.sum())]), np.concatenate([[0.0], np.cumsum(hits) / hits.sum()])


def confusion(y: np.ndarray, scores: np.ndarray, share: float) -> dict:
    called = scores >= np.quantile(scores, 1 - share)
    tp, fp = int((called & (y == 1)).sum()), int((called & (y == 0)).sum())
    fn, tn = int((~called & (y == 1)).sum()), int((~called & (y == 0)).sum())
    denominator = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"called positive": tp + fp, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "MCC": round((tp * tn - fp * fn) / denominator, 4) if denominator else 0.0}


def fold_table(oof: dict, enzyme: str, candidate: str, repeats: int = 3) -> pd.DataFrame:
    """The AUC of one candidate on the compounds that each outer fold scored, one column per repeat."""
    frame = oof[enzyme]
    y = frame["y"].to_numpy()
    rows = []
    for fold in range(folds_of(frame)):
        row = {"outer fold": fold}
        for repeat in range(repeats):
            keep = (frame[f"fold|{repeat}"] == fold).to_numpy()
            row[f"repeat {repeat}"] = round(auc_of(y[keep], frame[f"{candidate}|{repeat}"].to_numpy()[keep]), 4)
        row["compounds"] = int(keep.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def block_summary(oof: dict, candidates: pd.DataFrame, blocks: pd.DataFrame, enzyme: str, fold: str) -> pd.DataFrame:
    """For every feature block, the mean AUC of the candidates that use it against those that do not (descriptive, not a gate)."""
    everything = candidate_table(oof, candidates, enzyme, fold)
    rows = []
    for block in blocks["block"]:
        uses = everything["blocks"].str.split("+").map(lambda parts, b=block: b in parts)
        if uses.sum() and (~uses).sum():
            rows.append({"feature block": block, "candidates that use it": int(uses.sum()), "mean AUC with it": round(float(everything.loc[uses, "mean"].mean()), 4), "mean AUC without it": round(float(everything.loc[~uses, "mean"].mean()), 4)})
    out = pd.DataFrame(rows)
    out["difference"] = (out["mean AUC with it"] - out["mean AUC without it"]).round(4)
    return out.sort_values("difference", ascending=False)
