# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The row of the submitted entries on the final leaderboards of the organisers, read from the two files that the Space of the challenge exports.

``python -m cypmode.post_deadline.board CLASSIFICATION.csv STRUCTURE.csv`` copies the two files to ``data/post_deadline/inputs/leaderboards`` (not tracked), finds the row whose model report link points at this
repository, and writes ``data/post_deadline/final_board.json`` (tracked): for each track the rank, the number of entries, the tier, the values with their bootstrap standard deviations, the first entry of the board and
the last entry of the first tier, with the SHA-256 of the file that was read. The entry is found by its link and never by its alias, so that the file names no alias.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from cypmode.post_deadline.released import ROOT

INPUTS = ROOT / "data/post_deadline/inputs/leaderboards"
RESULT = ROOT / "data/post_deadline/final_board.json"
OWN_LINK = "mohamedtalaat-gif/CYPMode"
SOURCE = "https://openadmet.ghost.io/take-me-heme-binding-modes-wrapping-up-the-cyp-inhibition-challenge/"
VALUES = {"classification": ["MA-MCC", "MA-Accuracy", "MA-Precision", "MA-Recall", "MA-F1 Score"], "structure": ["LDDT-PLI", "BiSyRMSD", "LDDT-LP"]}
PRIMARY = {"classification": "MA-MCC", "structure": "LDDT-PLI"}


def _tier(label: str) -> int:
    return int(str(label).split()[-1])


def track_entry(board: pd.DataFrame, track: str, link: str = OWN_LINK) -> dict:
    """The row of the entry whose report link contains `link`, with the position of the board around it. Exactly one row must match."""
    mine = board[board["Model Report Link"].astype(str).str.contains(link, regex=False)]
    if len(mine) != 1:
        raise SystemExit(f"{track}: {len(mine)} rows carry a report link to {link}; exactly one is expected")
    row = mine.iloc[0]
    tiers = board["Significance (tiers)"].map(_tier)
    first_tier_last = board[tiers == 1].iloc[-1]
    primary = PRIMARY[track]
    values = {name: {"mean": float(row[name]), "std": float(row[f"{name} std"])} for name in VALUES[track]}
    return {
        "rank": int(row["Rank"]),
        "entries": len(board),
        "tier": _tier(row["Significance (tiers)"]),
        "tiers": int(tiers.max()),
        "first_tier_entries": int((tiers == 1).sum()),
        "values": values,
        "open_code": str(row["Open Code"]),
        "first": {primary: float(board.iloc[0][primary])},
        "last_of_first_tier": {primary: float(first_tier_last[primary])},
    }


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(classification: Path, structure: Path, inputs: Path = INPUTS, result: Path = RESULT) -> dict:
    inputs.mkdir(parents=True, exist_ok=True)
    out = {"source": SOURCE, "tracks": {}}
    for track, path in (("classification", classification), ("structure", structure)):
        kept = inputs / f"{track}_leaderboard.csv"
        shutil.copyfile(path, kept)
        out["tracks"][track] = {**track_entry(pd.read_csv(kept), track), "file_sha256": sha256_of(kept)}
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(out, indent=1) + "\n")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("classification", type=Path)
    parser.add_argument("structure", type=Path)
    args = parser.parse_args()
    out = build(args.classification, args.structure)
    for track, entry in out["tracks"].items():
        print(f"{track}: rank {entry['rank']} of {entry['entries']}, tier {entry['tier']} of {entry['tiers']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
