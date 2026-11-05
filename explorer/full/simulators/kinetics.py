# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part I: the kinetics that the notebook derives, as functions of their parameters, with the formulas of the notebook: elimination (1.1), the Michaelis-Menten rate (1.2), reversible inhibition (1.5),
time-dependent inhibition (1.6), the Hill equation and the pIC50 scale (1.7) and the shift between the two arms of the challenge (1.8)."""

import math

import numpy as np
import pandas as pd

from cypmode.challenge.tdi_targets import SHIFT_THRESHOLD


def elimination(c0: float, half_life: float, interval: float, doses: int, t_max: float, points: int = 400) -> pd.DataFrame:
    """C(t) = C0 exp(-k t) with k = ln 2 / half-life (1.1), for one dose and for ``doses`` doses given every ``interval`` (the curves add: the equation is linear)."""
    k = math.log(2) / half_life
    t = np.linspace(0, t_max, points)
    single = c0 * np.exp(-k * t)
    repeated = sum(np.where(t >= n * interval, c0 * np.exp(-k * (t - n * interval)), 0.0) for n in range(doses))
    return pd.DataFrame({"time": t, "one dose": single, "repeated doses": repeated})


def accumulation(half_life: float, interval: float) -> float:
    """The factor by which repeated dosing raises the concentration at steady state: 1 / (1 - exp(-k tau))."""
    return 1.0 / (1.0 - math.exp(-math.log(2) / half_life * interval))


def michaelis_menten(s, vmax: float, km: float):
    return vmax * np.asarray(s, dtype=float) / (km + np.asarray(s, dtype=float))


def inhibited_rate(s, inhibitor: float, vmax: float, km: float, ki: float, mode: str, ki_complex: float | None = None):
    """v = Vmax [S] / (alpha Km + alpha' [S]) with alpha = 1 + [I]/Ki and alpha' = 1 + [I]/Ki' (1.5); the mode fixes alpha and alpha'."""
    alpha = 1 + inhibitor / ki
    if mode == "competitive":
        alpha_prime = 1.0
    elif mode == "non-competitive":
        alpha_prime = alpha
    elif mode == "uncompetitive":
        alpha, alpha_prime = 1.0, 1 + inhibitor / ki
    elif mode == "mixed":
        alpha_prime = 1 + inhibitor / (ki_complex if ki_complex else 4 * ki)
    else:
        raise ValueError(mode)
    s = np.asarray(s, dtype=float)
    return vmax * s / (alpha * km + alpha_prime * s)


def tdi_activity(inhibitor: float, kinact: float, big_ki: float, minutes) -> tuple[float, np.ndarray]:
    """kobs = kinact [I] / (K_I + [I]) and the activity that is left, exp(-kobs t) (1.6)."""
    kobs = kinact * inhibitor / (big_ki + inhibitor)
    return kobs, np.exp(-kobs * np.asarray(minutes, dtype=float))


def hill(concentration, ic50: float, slope: float):
    """The activity that is left, y = 1 / (1 + ([I]/IC50)^h) (1.7)."""
    return 1.0 / (1.0 + (np.asarray(concentration, dtype=float) / ic50) ** slope)


def pic50(ic50_molar: float) -> float:
    return -math.log10(ic50_molar)


def ic50_molar(pic50_value: float) -> float:
    return 10 ** (-pic50_value)


def arm_shift(pic50_direct: float, pic50_tdi: float) -> dict:
    """The two arms in IC50 and in fold (1.8): a positive shift of log10(2) is a two-fold gain in potency after the head start."""
    shift = pic50_tdi - pic50_direct
    return {"direct IC50 (uM)": ic50_molar(pic50_direct) * 1e6, "TDI IC50 (uM)": ic50_molar(pic50_tdi) * 1e6, "IC50 shift (fold)": 10**shift, "pIC50 shift": shift, "log10(2)": SHIFT_THRESHOLD}
