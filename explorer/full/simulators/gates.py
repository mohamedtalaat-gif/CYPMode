# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Sections 6.7 and 6.5: the gate that every addition passes before it enters a stack. The constants are the project's (``GATE_ALPHA`` and ``Z_CUTS``), the test is its corrected resampled t, and the
correction for many comparisons is its own ``holm``."""

import numpy as np
from scipy import stats

from cypmode.challenge.addition_experiment import GATE_ALPHA, Z_CUTS
from cypmode.challenge.model_comparison import corrected_resampled_t
from cypmode.challenge.multiplicity import holm


def verdict(differences: dict[float, list[float]], n_train: int, n_test: int) -> dict:
    """The rule of the gate: at every cut-off z0 the mean difference is positive and the corrected p is below ``GATE_ALPHA``; all cut-offs must pass."""
    rows = []
    for z0 in Z_CUTS:
        c = corrected_resampled_t(differences[z0], n_train, n_test)
        rows.append({"z0": z0, "mean difference": c.mean_difference, "corrected t": c.t, "corrected p": c.p_value, "passes": bool(c.p_value < GATE_ALPHA and c.mean_difference > 0)})
    return {"rows": rows, "met": all(r["passes"] for r in rows), "alpha": GATE_ALPHA}


def false_alarms(rho: float, splits: int = 15, repeats: int = 40000, seed: int = 0) -> dict:
    """An addition with no true gain gives two signed t statistics, one per cut-off, with the correlation ``rho``. The share of such additions that pass under three rules: one cut-off fixed in advance,
    either of the two (chosen after seeing both) and both (the rule of the project)."""
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=repeats)
    cut = stats.t.ppf(1 - GATE_ALPHA / 2, df=splits - 1) / stats.t.ppf(0.8413447, df=splits - 1)  # a two-sided p below alpha, in units of the standard normal
    passed = z > cut
    return {"one cut-off fixed in advance": float(passed[:, 0].mean()), "either of the two cut-offs": float(passed.any(axis=1).mean()), "both cut-offs (the project's rule)": float(passed.all(axis=1).mean()), "alpha": GATE_ALPHA}


def holm_table(p_values: list[float]) -> list[dict]:
    adjusted = holm(p_values)
    return [{"comparison": i + 1, "raw p": p, "Holm-adjusted p": a, "still below alpha": bool(a < GATE_ALPHA)} for i, (p, a) in enumerate(zip(p_values, adjusted, strict=True))]
