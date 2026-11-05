# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Sections 0.1 and 1.4: what an inhibited enzyme does to the exposure to a drug. The static model of a drug-drug interaction (the ratio of the areas under the curve, AUCR) for a reversible inhibitor, and
the same with an inactivated enzyme, whose level falls until its loss by inactivation balances its synthesis."""

import numpy as np

STRONG, MODERATE, WEAK = 5.0, 2.0, 1.25  # the ratios of the AUC that name a strong, a moderate and a weak inhibitor in the guidance of the regulators


def aucr(fm: float, i_over_ki: float, enzyme_fraction: float = 1.0) -> float:
    """AUCR = 1 / (fm * E / (1 + I/Ki) + (1 - fm)): fm is the share of the clearance by this enzyme, E the fraction of the enzyme that is left (1 when nothing inactivates it)."""
    return 1.0 / (fm * enzyme_fraction / (1.0 + i_over_ki) + (1.0 - fm))


def classify(ratio: float) -> str:
    return "strong" if ratio >= STRONG else "moderate" if ratio >= MODERATE else "weak" if ratio >= WEAK else "no relevant"


def inactivation_rate(kinact: float, inhibitor: float, k_half: float) -> float:
    """The first-order rate of loss of the enzyme by inactivation: k_obs = kinact I / (K_I + I) (1.6)."""
    return kinact * inhibitor / (k_half + inhibitor)


def enzyme_fraction(t, k_deg: float, k_obs: float):
    """E(t)/E0 when the enzyme is made at the rate that keeps it constant without the inhibitor and is lost at k_deg + k_obs with it."""
    steady = k_deg / (k_deg + k_obs)
    return steady + (1.0 - steady) * np.exp(-(k_deg + k_obs) * np.asarray(t, dtype=float))
