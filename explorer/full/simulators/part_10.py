# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part X, quantum chemistry and the physics features: the Schroedinger equation (10.1), a self-consistent field (10.2), the exchange of a density (10.3), a Hueckel calculation (10.4) and the experiments with the physics block (10.6)."""

import altair as alt
import numpy as np
import pandas as pd

from simulators import molecule, quantum, replay

POTENTIALS = ["particle in a box of width a", "harmonic oscillator of frequency w", "Morse well of depth D", "double well of height h"]


def c_10_1(mo):
    return {"potential": mo.ui.dropdown(options=POTENTIALS, value=POTENTIALS[1], label="Potential"), "parameter": mo.ui.slider(start=0.5, stop=8.0, step=0.1, value=2.0, label="the parameter of the potential (a, w, D or h)", show_value=True)}


def v_10_1(mo, v):
    r = quantum.schrodinger_1d(v["potential"], float(v["parameter"]))
    scale = np.diff(r["energies"]).mean() * 0.8
    frames = [pd.DataFrame({"x": r["x"], "y": r["v"]}).assign(curve="potential")]
    for k in range(len(r["energies"])):
        frames.append(pd.DataFrame({"x": r["x"], "y": r["energies"][k] + scale * r["states"][:, k] / np.abs(r["states"][:, k]).max()}).assign(curve=f"state {k + 1}"))
    data = pd.concat(frames)
    top = float(r["energies"].max() + 2 * scale)
    chart = alt.Chart(data[data["y"] < top]).mark_line().encode(x="x:Q", y=alt.Y("y:Q", scale=alt.Scale(domain=[0, top])), color="curve:N")
    table = pd.DataFrame({"state": range(1, len(r["energies"]) + 1), "energy, numerical": r["energies"], "energy, exact": r["exact"]})
    return mo.vstack([mo.md("The Schroedinger equation -1/2 psi'' + V psi = E psi solved by turning the second derivative into differences on a grid and diagonalising the matrix: the eigenvalues are the allowed energies and the eigenvectors the waves (drawn at the height of their energy). "
                            "Only a confined particle has levels; where the exact energies are known they are given beside the numerical ones."), chart.properties(width=560, height=300), mo.ui.table(table, selection=None, pagination=False)])


def c_10_2(mo):
    return {"t": mo.ui.slider(start=0.1, stop=2.0, step=0.05, value=1.0, label="t, the hopping between the two sites", show_value=True), "u": mo.ui.slider(start=0.0, stop=12.0, step=0.1, value=6.0, label="U, the repulsion of two electrons on one site", show_value=True),
            "asymmetry": mo.ui.slider(start=0.0, stop=0.4, step=0.02, value=0.2, label="how much the starting density favours site 1", show_value=True), "mixing": mo.ui.slider(start=0.1, stop=1.0, step=0.1, value=0.5, label="mixing of the new density into the old", show_value=True)}


def v_10_2(mo, v):
    r = quantum.hubbard_dimer_scf(float(v["t"]), float(v["u"]), float(v["asymmetry"]), float(v["mixing"]))
    curve = pd.DataFrame({"iteration": range(1, len(r["history"]) + 1), "energy": r["history"]})
    return mo.vstack([mo.md(f"Two electrons on two sites. The field that each one feels depends on where the other is, so the density is guessed, the orbitals are found in the field it makes, the new density is mixed in and the loop is repeated: it {'converged' if r['converged'] else 'did not converge'} in **{r['iterations']}** iterations to E = **{r['energy']:.4f}**; "
                            f"the exact energy is {r['exact']:.4f}, so what the mean field misses (the correlation energy) is **{r['energy'] - r['exact']:.4f}**. Once U is larger than 2t the converged density of the spin up electron sits on one site: {np.round(r['density up'], 3)}."), alt.Chart(curve).mark_line(point=True).encode(x="iteration:Q", y=alt.Y("energy:Q", scale=alt.Scale(zero=False))).properties(width=480, height=220)])


def c_10_3(mo):
    return {"zeta": mo.ui.slider(start=0.4, stop=1.6, step=0.01, value=1.0, label="zeta, the exponent of the 1s density (1 is the hydrogen atom)", show_value=True)}


def v_10_3(mo, v):
    r = quantum.hydrogenic(float(v["zeta"]))
    grid = np.linspace(0.4, 1.6, 61)
    curve = pd.DataFrame([{"zeta": z, "E exact (one electron)": quantum.hydrogenic(z)["E exact"], "E with the LDA exchange": quantum.hydrogenic(z)["E LDA"]} for z in grid]).melt("zeta", ["E exact (one electron)", "E with the LDA exchange"], "energy", "Hartree")
    return mo.vstack([mo.md(f"The density integrates to {r['electrons']:.4f} electron. Its Hartree energy is {r['Hartree']:.4f}; for one electron the exact exchange cancels it ({r['exact exchange']:.4f}), while the local-spin-density exchange (the electron has all its spin up) gives {r['LDA exchange']:.4f} and leaves a self-interaction of {r['Hartree'] + r['LDA exchange']:.4f}. "
                            f"Total energy at this zeta: exact **{r['E exact']:.4f}**, with the LDA exchange **{r['E LDA']:.4f}**. The minima (-0.5000 at zeta = 1 and about -0.4565 at zeta = 0.9555) show the error that the approximation makes on the simplest atom."),
                      alt.Chart(curve).mark_line().encode(x="zeta:Q", y=alt.Y("Hartree:Q", scale=alt.Scale(zero=False)), color="energy:N").properties(width=520, height=260)])


def c_10_4(mo):
    return {"smiles": mo.ui.text(value="c1ccc2[nH]ccc2c1", label="SMILES", full_width=True)}


def v_10_4(mo, v):
    r = quantum.huckel(v["smiles"])
    mol = molecule.parse(v["smiles"])
    if r is None or mol is None:
        return mo.md("That is not a SMILES that RDKit can read.")
    if not r["atoms"]:
        return mo.md("No conjugated bond: there is no pi system to treat.")
    levels = pd.DataFrame({"level": range(1, len(r["levels"]) + 1), "energy (alpha + x beta)": r["levels"], "occupied": [i < r["occupied"] for i in range(len(r["levels"]))]})
    return mo.vstack([mo.Html(molecule.drawing(mol, (), size=(360, 260))), mo.md(f"The Hueckel method keeps only the pi electrons: {r['atoms']} conjugated atoms, {r['electrons']} electrons, the levels below (x in units of beta, the most bonding first). The gap between the highest occupied and the lowest empty level is **{r['gap']:.3f} |beta|** "
                                                                                 "(benzene: 2). A smaller gap means a molecule that is easier to oxidise. The tight-binding method GFN2-xTB of the physics block is the same idea with every valence orbital and parameters fitted to many molecules."),
                      alt.Chart(levels).mark_rule(strokeWidth=4).encode(y=alt.Y("energy (alpha + x beta):Q"), x=alt.value(0), x2=alt.value(120), color="occupied:N").properties(width=140, height=260)])


SIMS = {"10.1": (c_10_1, v_10_1), "10.2": (c_10_2, v_10_2), "10.3": (c_10_3, v_10_3), "10.4": (c_10_4, v_10_4),
        "10.6": (replay.controls, lambda mo, v: replay.page(mo, v, ["physics_experiment", "physics_compact", "physics_tierb", "population_physics"], "Section 10.6 put the quantities of the physics block in the models at four levels of theory and two refinements."))}
