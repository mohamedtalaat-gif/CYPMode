# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The small tables of the explorer, read once: ``table("ledger")`` finds ``data/ledger.csv``, ``data/public/ledger.csv`` or ``data/public/ledger.csv.gz``."""

from functools import cache
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


@cache
def table(name: str) -> pd.DataFrame:
    for candidate in (DATA / f"{name}.csv", DATA / "public" / f"{name}.csv", DATA / "public" / f"{name}.csv.gz"):
        if candidate.exists():
            return pd.read_csv(candidate)
    raise FileNotFoundError(f"no table named {name!r} under {DATA}")


@cache
def candidate_names(*enzymes: str) -> list[str]:
    """The candidates that have stored out-of-fold scores for every one of the enzymes, so that a choice is never silently replaced by another when the enzyme changes."""
    sets = [{c.split("|")[0] for c in table(f"oof_{e}").columns if c.endswith("|0") and not c.startswith("fold")} for e in enzymes]
    return sorted(set.intersection(*sets))
