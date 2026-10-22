# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Multiplicity of the additions tested together: Holm-adjusted p-values next to the pre-registered gates.

Every addition is judged by its own pre-registered gate (corrected resampled t, p < 0.05 at both confidence cut-offs, inside the assembled stack). Testing many additions raises the
chance that one of them passes by luck, so the family of tests of one batch is also reported with the Holm step-down adjustment. The gate of an addition needs its p-value at both
cut-offs, so the p-value of the addition is the larger of the two (the intersection-union test); Holm then adjusts those over the family. The pre-registered gate decides what
enters the model; the adjusted values say how much of the evidence survives the number of attempts, and are reported in the audit, the method report and the notebook.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

STACKED = re.compile(r"z0=(\S+)\s+stacked[^:\n]*:\s+AUC\s+([+-]\d+\.\d+)\s+\(p=(\d+\.\d+)")
SELECTOR = re.compile(r"z0=(\S+)\s+outer AUC: .*?\(([+-]\d+\.\d+), p=(\d+\.\d+)")
ISOFORM = re.compile(r"^(CYP\d[A-Z]\d) GATE \(", re.MULTILINE)
BATCH = ("aux_*_CYP*.log", "monroe_embeddings_CYP*.log", "population_screen_CYP2D6.log", "nonneg_stack_CYP*.log")


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the order given."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m, adjusted, running = len(p_values), [0.0] * len(p_values), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def collect(logs: Path, patterns: tuple[str, ...] = BATCH) -> pd.DataFrame:
    """One row per (log, isoform) with the stacked or selector test at each z0 and the intersection-union p-value."""
    rows = []
    for pattern in patterns:
        for path in sorted(logs.glob(pattern)):
            text = path.read_text(errors="ignore")
            found = {z0: (float(d), float(p)) for z0, d, p in (STACKED.findall(text) or SELECTOR.findall(text))}
            if not found:
                continue
            iso = (ISOFORM.findall(text) or [path.stem.rsplit("_", 1)[-1]])[0]
            rows.append({"log": path.name, "isoform": iso, **{f"diff_z{z}": d for z, (d, _) in found.items()}, **{f"p_z{z}": p for z, (_, p) in found.items()}, "p_both": max(p for _, p in found.values()), "gain_positive_at_both": all(d > 0 for d, _ in found.values())})
    table = pd.DataFrame(rows)
    if len(table):
        table["holm_adjusted"] = holm(table["p_both"].tolist())
    return table
