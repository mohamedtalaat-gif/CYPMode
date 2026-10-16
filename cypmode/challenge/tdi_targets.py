"""The TDI label is a deterministic function of two measured pIC50 values, and
the training file carries both of them -- this module makes that explicit so
models can learn the continuous quantity instead of its thresholded shadow.

The challenge FAQ defines the label from the direct-inhibition pIC50 ``d`` and
the TDI-arm pIC50 ``t`` (assay floor 4, a 2-fold shift = log10(2) = 0.301):

- positive           d > 4 and t - d > 0.301
- inferred positive  d < 4 and t > 4.301
- negative / assigned negative: everything else

Both positive branches collapse into one threshold on a single continuous
number, the shift measured against the floor-clipped direct value:

    s = t - max(d, 4)        positive  <=>  s > log10(2)

(d < 4: s > 0.301 <=> t > 4.301; d > 4: s > 0.301 <=> t - d > 0.301.) This
reproduces every given training label exactly (checked in the tests against
the real file), so ``s`` is a lossless, far more informative regression
target than the boolean.

Rows lacking a direct measurement need care. For CYP3A4 about a third of the
labeled training rows have no direct pIC50 but DO have a TDI-arm pIC50 -- with
a median of 5.4, i.e. mostly potent -- and every one of them is labeled
negative, because the label formula has no baseline to compute a shift from.
The blinded test set was assayed in full dose-response mode in both arms, so
no test compound has that property: those rows follow a different labeling
convention than the one being scored.
"""

import math

import numpy as np
import pandas as pd

ASSAY_FLOOR = 4.0
SHIFT_THRESHOLD = math.log10(2.0)

ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
TDI_ISOFORMS = ("CYP3A4", "CYP2D6")

BOTH_ARMS = "both_arms"
NO_DIRECT = "no_direct_label_by_convention"
NO_LABEL = "unlabeled"


def direct_column(isoform: str) -> str:
    return f"{isoform}_pIC50_direct_inhibition"


def tdi_column(isoform: str) -> str:
    return f"{isoform}_pIC50_TDI_condition"


def label_column(isoform: str) -> str:
    return f"{isoform}_is_TDI"


def shift_target(direct, tdi):
    """s = t - max(d, 4); NaN wherever either arm is missing."""
    return np.asarray(tdi, dtype=float) - np.maximum(np.asarray(direct, dtype=float), ASSAY_FLOOR)


def label_from_shift(shift) -> np.ndarray:
    """The challenge label implied by s (NaN-safe: NaN stays NaN, as float 0/1)."""
    shift = np.asarray(shift, dtype=float)
    return np.where(np.isnan(shift), np.nan, (shift > SHIFT_THRESHOLD).astype(float))


def label_from_arms(direct: float, tdi: float) -> bool:
    """The FAQ's literal four-way rule, kept separate from ``shift_target`` so
    the equivalence of the two is something the tests verify, not assume.
    """
    if direct > ASSAY_FLOOR:
        return bool(tdi - direct > SHIFT_THRESHOLD)
    return bool(tdi > ASSAY_FLOOR + SHIFT_THRESHOLD)


def training_view(df: pd.DataFrame, isoform: str) -> pd.DataFrame:
    """One row per training compound with this isoform's ``label``, ``direct``,
    ``tdi``, ``shift`` and a ``population`` tag (BOTH_ARMS / NO_DIRECT / NO_LABEL).

    ``shift_imputed`` additionally fills the NO_DIRECT rows with t - 4, the value
    they would have if their (unmeasured) direct pIC50 were below the floor --
    the only reading consistent with their negative label when t is low; it is
    NOT a measured value and exists so a regressor can still learn from the TDI
    arm of those compounds.
    """
    label = df[label_column(isoform)]
    direct = df[direct_column(isoform)]
    tdi = df[tdi_column(isoform)]
    view = pd.DataFrame({"label": label, "direct": direct, "tdi": tdi}, index=df.index)
    view["shift"] = shift_target(direct, tdi)
    population = pd.Series(NO_LABEL, index=df.index, dtype=object)
    population[label.notna() & direct.notna() & tdi.notna()] = BOTH_ARMS
    population[label.notna() & direct.isna()] = NO_DIRECT
    view["population"] = population
    view["shift_imputed"] = view["shift"]
    no_direct = (population == NO_DIRECT) & tdi.notna()
    view.loc[no_direct, "shift_imputed"] = tdi[no_direct] - ASSAY_FLOOR
    return view
