# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Section 2.5: measurement noise and the confidence z. The noise of the shift follows from the two standard deviations, and z measures how far the shift is from the threshold in units of that noise. The
computation is the project's own ``confidence_z`` applied to one row."""

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from cypmode.challenge.label_confidence import STD_SUFFIX, confidence_z
from cypmode.challenge.tdi_targets import (
    ASSAY_FLOOR,
    SHIFT_THRESHOLD,
    direct_column,
    tdi_column,
)

ISOFORM = "CYP3A4"


def evaluate(d: float, t: float, sd_d: float, sd_t: float) -> dict:
    """The shift, its standard deviation, z and the probability that the label is wrong, for one compound."""
    row = pd.DataFrame({direct_column(ISOFORM): [d], tdi_column(ISOFORM): [t], direct_column(ISOFORM) + STD_SUFFIX: [sd_d], tdi_column(ISOFORM) + STD_SUFFIX: [sd_t]})
    z = float(confidence_z(row, ISOFORM)[0])
    variance = sd_t**2 + (sd_d**2 if d >= ASSAY_FLOOR else 0.0)
    shift = t - max(d, ASSAY_FLOOR)
    return {"shift": shift, "sd of the shift": math.sqrt(variance), "z": z, "probability the label is wrong": float(norm.sf(z)), "threshold": SHIFT_THRESHOLD, "d counts in the noise": d >= ASSAY_FLOOR}


def committee(d: float, t: float, sd_d: float, sd_t: float, draws: int = 20000, seed: int = 0) -> dict:
    """Section 9.3: the label as a threshold of two noisy numbers. Draw both measurements from normal distributions around what was measured and count how often the label is positive; the analytic value
    is the normal probability that the shift lies above the threshold."""
    rng = np.random.default_rng(seed)
    shift = rng.normal(t, sd_t, draws) - np.maximum(rng.normal(d, sd_d, draws), ASSAY_FLOOR)
    base = evaluate(d, t, sd_d, sd_t)
    analytic = float(norm.sf((SHIFT_THRESHOLD - base["shift"]) / base["sd of the shift"]))
    return {"simulated share positive": float((shift > SHIFT_THRESHOLD).mean()), "analytic probability": analytic, "label as measured": bool(base["shift"] > SHIFT_THRESHOLD), "z": base["z"]}


def weights_by_mode(z, mode: str) -> dict:
    """Section 9.8: what the project's ``training_weights`` do with a vector of |z| under one mode: the rows kept and the weight of each."""
    from cypmode.challenge.label_confidence import training_weights

    keep, weight = training_weights(np.asarray(z, dtype=float), mode)
    return {"rows kept": int(keep.sum()), "rows": len(keep), "total weight": float(weight[keep].sum()), "keep": keep, "weight": weight}
