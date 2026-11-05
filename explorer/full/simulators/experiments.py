# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The experiments of the ledger behind a section, replayed: the verdicts of the families of ideas that the section tells, their paired differences with the corrected p that the gate computed, and the lines of the logs.
Nothing is refitted; the one live control is the level of the gate, which the stored p values are compared with."""

import re

import numpy as np
import pandas as pd

from simulators import store


def family_of(log: str) -> str:
    """``aux_neural_f10_CYP3A4.log`` -> ``aux_neural``: the family is the log without the enzyme and the replicate suffix."""
    return re.sub(r"(_f10|_r\d+)?_(CYP\w+)\.log$|\.log$", "", log)


def verdicts(families: list[str]) -> pd.DataFrame:
    ledger = store.table("ledger")
    return ledger[ledger["stem"].isin(families)][["stem", "name", "kind", "verdict"]].reset_index(drop=True)


def comparisons(families: list[str], enzyme: str | None = None) -> pd.DataFrame:
    paired = store.table("paired")
    paired = paired.assign(family=paired["log"].map(family_of))
    paired = paired[paired["family"].isin(families)]
    return paired if enzyme in (None, "both") else paired[paired["enzyme"] == enzyme]


def passes(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """The comparisons marked with the rule of the gate at a level alpha: a positive gain and a p below alpha."""
    return frame.assign(passes=(frame["auc_gain"] > 0) & (frame["p"] < alpha))


def lines(families: list[str], limit: int = 40) -> pd.DataFrame:
    excerpts = store.table("excerpts")
    excerpts = excerpts.assign(family=excerpts["log"].map(family_of))
    return excerpts[excerpts["family"].isin(families)][["log", "text"]].head(limit)


def share_met(frame: pd.DataFrame) -> float:
    return float((frame["verdict"] == "MET").mean()) if len(frame) else float("nan")


def gain_summary(frame: pd.DataFrame) -> dict:
    return {"comparisons": len(frame), "mean AUC gain": float(frame["auc_gain"].mean()) if len(frame) else float("nan"), "best AUC gain": float(frame["auc_gain"].max()) if len(frame) else float("nan"), "smallest p": float(frame["p"].min()) if len(frame) else float("nan"),
            "gains that are positive": int((frame["auc_gain"] > 0).sum()), "folds better (mean share)": float(np.mean(frame["better"] / frame["folds"])) if len(frame) else float("nan")}
