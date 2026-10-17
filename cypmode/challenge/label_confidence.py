"""How confidently is a compound's TDI label assigned? -- the scored set only contains confident labels.

The challenge FAQ says predictions are requested for every test compound, "but only compounds whose label
can be assigned with confidence contribute to the score". The label is a threshold on the shift

    s = t - max(d, 4)        positive  <=>  s > log10(2)

of two measured pIC50 values, each with a posterior standard deviation in the training file. A compound
whose shift sits within noise of the threshold has a label that is close to a coin flip, and the official
score almost certainly leaves such compounds out. The rule used here is the natural one and the exact
official rule is unknown, so every analysis is repeated across a range of cut-offs:

    z = (s - log10(2)) / sd(s),     sd(s)^2 = sd(t)^2 + [d >= 4] sd(d)^2,     confident <=> |z| >= z0

Evidence that this matters: the official LightGBM baseline's cross-validated MCC rises with z0
(CYP2D6 0.046 on all rows, 0.096 at z0 = 1.96, 0.140 at 2.5; CYP3A4 0.236 -> 0.289 -> 0.297), i.e. in the
direction of its leaderboard score (0.176 / 0.382), so evaluating on every row measures a noisier problem
than the one that is scored.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from cypmode.challenge.tdi_targets import (
    ASSAY_FLOOR,
    SHIFT_THRESHOLD,
    direct_column,
    tdi_column,
)

STD_SUFFIX = "_std"


def confidence_z(df: pd.DataFrame, isoform: str) -> np.ndarray:
    """|z| of every row's label against the threshold (NaN where the TDI arm is not measured)."""
    t = df[tdi_column(isoform)].to_numpy(dtype=float)
    d = df[direct_column(isoform)].to_numpy(dtype=float)
    sd_t = df[tdi_column(isoform) + STD_SUFFIX].to_numpy(dtype=float)
    sd_d = df[direct_column(isoform) + STD_SUFFIX].to_numpy(dtype=float)
    shift = t - np.maximum(np.where(np.isnan(d), ASSAY_FLOOR, d), ASSAY_FLOOR)
    variance = sd_t**2 + np.where(d >= ASSAY_FLOOR, np.nan_to_num(sd_d) ** 2, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(shift - SHIFT_THRESHOLD) / np.sqrt(variance)


def confident_weights(weight_fn: Callable | None, z: np.ndarray, z0: float) -> Callable:
    """Wrap an evaluation weight function so rows with |z| < z0 (or unknown) get weight 0.

    ``z`` is indexed like the plan's rows (one entry per compound). Rows outside the pool carry NaN and are
    never evaluated, so they are harmless.
    """

    def wrapped(rows: np.ndarray, reference_rows: np.ndarray) -> np.ndarray:
        base = np.ones(len(rows)) if weight_fn is None else weight_fn(rows, reference_rows)
        keep = np.nan_to_num(z[rows], nan=0.0) >= z0
        return base * keep

    return wrapped


def training_weights(z: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """(row mask, sample weights) a model should train with, from the |z| of its TRAINING rows only.

    ``""``       every row, weight 1 (the usual behaviour)
    ``"drop<z0>"`` rows with |z| >= z0 only, weight 1   (e.g. "drop1.0")
    ``"soft"``   every row, weight min(1, |z| / 2) floored at 0.1 -- a smooth version of the same idea
    """
    z = np.nan_to_num(z, nan=0.0)
    if mode == "":
        return np.ones(len(z), dtype=bool), np.ones(len(z))
    if mode.startswith("drop"):
        keep = z >= float(mode.removeprefix("drop"))
        return keep, np.ones(len(z))
    if mode == "soft":
        return np.ones(len(z), dtype=bool), np.clip(z / 2.0, 0.1, 1.0)
    raise ValueError(f"unknown training weighting {mode!r}")
