# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Re-analyse the saved per-fold tables of the older significance experiments with the paired corrected t (Holm).

Each ``*_significance_raw.csv`` holds the per-(repeat, fold) MCC of every variant. The earlier summaries compared the variants with
Tukey's HSD, which treats correlated folds as independent groups. This writes ``*_significance_paired.csv`` next to each table from the
same raw scores, using ``stacking_significance.paired_comparison_summary``; no model is refitted.

Usage: python -m cypmode.challenge.recompute_paired_summaries
"""

from pathlib import Path

import pandas as pd

from cypmode.challenge.stacking_significance import paired_comparison_summary

COMPARISONS = Path("data/tdi_track/outputs/comparisons")


def recompute(directory: Path = COMPARISONS) -> list[Path]:
    written = []
    for raw in sorted(directory.glob("*_significance_raw.csv")):
        table = pd.read_csv(raw)
        parts = []
        for isoform, group in table.groupby("isoform", sort=True):
            summary = paired_comparison_summary(group)
            summary.insert(0, "isoform", isoform)
            parts.append(summary)
        out = directory / raw.name.replace("_raw.csv", "_paired.csv")
        pd.concat(parts, ignore_index=True).to_csv(out, index=False)
        written.append(out)
    return written


def main() -> None:
    for path in recompute():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
