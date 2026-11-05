# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Sections 1.3 and 1.9: the catalytic cycle of a P450 and what binding to the iron looks like. The steps are the textbook cycle of the heme iron; the difference spectra are schematic (typical positions
of the peak and the trough, widths chosen for display) and the titration is a simulation of the hyperbola that a real titration follows."""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

CYCLE = [
    ("Resting state", "Fe(III), low spin", "A water molecule is the sixth ligand of the iron; the Soret band is near 418 nm."),
    ("Substrate binds", "Fe(III), high spin", "The substrate displaces the water; the iron moves to high spin (a Type I shift, peak near 390 nm) and its reduction potential rises, so that an electron can be taken."),
    ("First electron", "Fe(II), high spin", "NADPH-cytochrome P450 reductase gives the first electron."),
    ("Oxygen binds", "Fe(II)-O2", "Dioxygen binds as the sixth ligand."),
    ("Second electron", "Fe(III)-peroxo", "The second electron (from the reductase, or from cytochrome b5) reduces the bound oxygen."),
    ("First proton", "Fe(III)-hydroperoxo", "A proton gives the hydroperoxo species (called compound 0)."),
    ("Second proton, the O-O bond breaks", "Fe(IV)=O, a radical cation of the porphyrin", "A second proton leaves a molecule of water and the O-O bond is cleaved: this is compound I, the oxidant."),
    ("Oxygen transfer", "Fe(III) with the product bound", "Compound I takes a hydrogen atom from the substrate and the oxygen rebounds onto it: the substrate is hydroxylated."),
    ("Product leaves", "Fe(III), low spin", "The product leaves, water returns to the sixth position and the enzyme is back in the resting state."),
]

TYPE_II_NOTE = ("A Type II ligand puts the lone pair of a nitrogen on the iron as its sixth ligand. The iron stays low spin, the substrate cannot displace the ligand, the reduction potential is not raised and "
                "the cycle does not leave the first step: the enzyme is inhibited. The difference spectrum has its peak near 430 nm and its trough near 390 to 410 nm.")


def difference_spectrum(kind: str, fraction_bound: float, wavelengths) -> np.ndarray:
    """A schematic difference spectrum (the bound minus the free enzyme): Type I has its peak near 390 nm and its trough near 420 nm, Type II the reverse order, a peak near 430 and a trough near 392."""
    w = np.asarray(wavelengths, dtype=float)
    peak, trough = (390.0, 420.0) if kind == "Type I" else (430.0, 392.0)
    return fraction_bound * (np.exp(-0.5 * ((w - peak) / 12.0) ** 2) - 0.8 * np.exp(-0.5 * ((w - trough) / 14.0) ** 2))


def titration(k_s: float, delta_max: float, noise: float, seed: int, points: int = 12) -> tuple[pd.DataFrame, float, float]:
    """A binding titration: dA = dA_max L / (K_s + L) at ligand concentrations from K_s/10 to 10 K_s, with normal noise; returns the points, the fitted K_s and its standard error."""
    rng = np.random.default_rng(seed)
    ligand = np.geomspace(k_s / 10.0, k_s * 10.0, points)
    signal = delta_max * ligand / (k_s + ligand) + rng.normal(0.0, noise, points)
    (_fit_max, fit_ks), cov = curve_fit(lambda x, a, k: a * x / (k + x), ligand, signal, p0=[delta_max, k_s], maxfev=10000)
    return pd.DataFrame({"ligand": ligand, "signal": signal}), float(fit_ks), float(np.sqrt(cov[1, 1]))
