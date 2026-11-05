# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part I, biology and chemistry: the catalytic cycle (1.3), a drug interaction with an inactivated enzyme (1.4), binding to the heme iron (1.9) and the alerts of reactive metabolites (1.10)."""

import altair as alt
import numpy as np
import pandas as pd

from simulators import alerts, ddi, heme, molecule, store


def c_1_3(mo):
    return {"step": mo.ui.slider(start=1, stop=len(heme.CYCLE), step=1, value=1, label="Step of the cycle", show_value=True), "ligand": mo.ui.checkbox(value=False, label="a Type II ligand (a nitrogen on the iron)")}


def v_1_3(mo, v):
    n = len(heme.CYCLE)
    angle = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    ring = pd.DataFrame({"x": np.cos(angle), "y": np.sin(angle), "step": [str(i + 1) for i in range(n)], "name": [c[0] for c in heme.CYCLE], "current": [i + 1 == v["step"] for i in range(n)]})
    chart = alt.Chart(ring).mark_circle().encode(x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-1.4, 1.4])), y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-1.4, 1.4])), size=alt.condition("datum.current", alt.value(900), alt.value(260)),
                                                  color=alt.condition("datum.current", alt.value("#c62828"), alt.value("#90a4ae")), tooltip=["step", "name"])
    labels = alt.Chart(ring).mark_text(color="white", fontWeight="bold").encode(x="x:Q", y="y:Q", text="step:N")
    name, iron, text = heme.CYCLE[v["step"] - 1]
    body = f"**Step {v['step']}: {name}.** The iron is **{iron}**. {text}"
    if v["ligand"]:
        body += "\n\n" + heme.TYPE_II_NOTE
    return mo.vstack([mo.md(body), (chart + labels).properties(width=380, height=380)])


def c_1_4(mo):
    return {"fm": mo.ui.slider(start=0.05, stop=0.99, step=0.01, value=0.8, label="fm", show_value=True), "i": mo.ui.slider(start=0.1, stop=100.0, step=0.1, value=5.0, label="free inhibitor concentration I (uM)", show_value=True),
            "ki": mo.ui.slider(start=0.1, stop=100.0, step=0.1, value=20.0, label="Ki of the reversible binding (uM)", show_value=True), "kinact": mo.ui.slider(start=0.0, stop=0.5, step=0.005, value=0.05, label="kinact, the rate of inactivation at saturation (1/h)", show_value=True),
            "k_half": mo.ui.slider(start=0.1, stop=100.0, step=0.1, value=5.0, label="K_I, the concentration that gives half of kinact (uM)", show_value=True), "half_life": mo.ui.slider(start=4.0, stop=120.0, step=1.0, value=36.0, label="half-life of the enzyme (h)", show_value=True)}


def v_1_4(mo, v):
    k_deg, k_obs = float(np.log(2) / v["half_life"]), ddi.inactivation_rate(v["kinact"], v["i"], v["k_half"])
    t = np.linspace(0, 10 * v["half_life"], 300)
    e = ddi.enzyme_fraction(t, k_deg, k_obs)
    frame = pd.DataFrame({"hours": t, "enzyme left": e, "AUC ratio, reversible only": ddi.aucr(v["fm"], v["i"] / v["ki"]), "AUC ratio, with inactivation": [ddi.aucr(v["fm"], v["i"] / v["ki"], x) for x in e]})
    ratios = alt.Chart(frame.melt("hours", ["AUC ratio, reversible only", "AUC ratio, with inactivation"], "model", "AUC ratio")).mark_line().encode(x="hours:Q", y="AUC ratio:Q", color="model:N")
    enzyme = alt.Chart(frame).mark_line(color="#6a1b9a").encode(x="hours:Q", y=alt.Y("enzyme left:Q", scale=alt.Scale(domain=[0, 1])))
    final = float(frame["AUC ratio, with inactivation"].iloc[-1])
    return mo.vstack([mo.md(f"The enzyme is made at the rate that keeps it constant and lost at k_deg = {k_deg:.4f}/h; the inhibitor adds the loss k_obs = kinact I / (K_I + I) = **{k_obs:.4f}/h**, so the level falls to **{k_deg / (k_deg + k_obs):.1%}** of the normal one. "
                            f"The exposure to the victim drug is multiplied by {frame['AUC ratio, reversible only'].iloc[0]:.2f} by the reversible binding alone and by **{final:.2f}** when the inactivation has run its course ({ddi.classify(final)}). The effect grows with time, which is why a time-dependent inhibitor is judged after days, not hours."),
                      enzyme.properties(width=560, height=150, title="the enzyme that is left"), ratios.properties(width=560, height=240)])


def c_1_9(mo):
    return {"kind": mo.ui.radio(options=["Type I", "Type II"], value="Type II", label="Binding", inline=True), "ks": mo.ui.slider(start=0.1, stop=50.0, step=0.1, value=5.0, label="Ks, the dissociation constant (uM)", show_value=True),
            "noise": mo.ui.slider(start=0.0, stop=0.05, step=0.001, value=0.004, label="noise of the absorbance difference", show_value=True), "seed": mo.ui.number(start=0, stop=999, step=1, value=1, label="seed")}


def v_1_9(mo, v):
    w = np.linspace(360, 470, 221)
    spectra = pd.concat([pd.DataFrame({"wavelength (nm)": w, "difference": heme.difference_spectrum(v["kind"], f, w), "bound": f"{f:.0%} bound"}) for f in (0.25, 0.5, 1.0)])
    spectrum = alt.Chart(spectra).mark_line().encode(x="wavelength (nm):Q", y="difference:Q", color="bound:N")
    points, ks_fit, se = heme.titration(v["ks"], 0.1, v["noise"], int(v["seed"]))
    curve = pd.DataFrame({"ligand": np.geomspace(points["ligand"].min(), points["ligand"].max(), 100)}).assign(signal=lambda f: 0.1 * f["ligand"] / (v["ks"] + f["ligand"]))
    fit = alt.Chart(points).mark_point(filled=True, size=60).encode(x=alt.X("ligand:Q", scale=alt.Scale(type="log")), y="signal:Q") + alt.Chart(curve).mark_line(color="gray").encode(x="ligand:Q", y="signal:Q")
    return mo.vstack([mo.md(f"{v['kind']} binding shows a peak and a trough in the difference spectrum, with the peak {'near 390 nm' if v['kind'] == 'Type I' else 'near 430 nm'} (the curves are schematic: typical positions, widths chosen for display). "
                            f"A titration reads the size of the peak at growing ligand concentrations; the hyperbola gives Ks. Fitted here: **{ks_fit:.2f} uM** +- {se:.2f} (true value {v['ks']:.2f})."), spectrum.properties(width=560, height=220), fit.properties(width=560, height=220)])


def c_1_10(mo):
    options = {n: s for n, s in zip(store.table("compounds")["Molecule_Name"].head(300), store.table("compounds")["SMILES"].head(300), strict=True)}
    return {"smiles": mo.ui.text(value="c1ccc2OCOc2c1CCN", label="SMILES", full_width=True), "pick": mo.ui.dropdown(options={"(use the SMILES above)": "", **options}, value="(use the SMILES above)", label="or a compound of the training file")}


def v_1_10(mo, v):
    smiles = v["pick"] or v["smiles"]
    mol = molecule.parse(smiles)
    if mol is None:
        return mo.md("That is not a SMILES that RDKit can read.")
    found = alerts.alerts_found(smiles)
    if not found:
        return mo.vstack([mo.Html(molecule.drawing(mol, (), indices=False)), mo.md("No alert of the list matches. " + str(alerts.context(smiles)))])
    table = pd.DataFrame(found).drop(columns="atoms")
    return mo.vstack([mo.Html(molecule.drawing(mol, found[0]["atoms"], indices=False)), mo.md(f"The first alert, **{found[0]['alert']}**, is highlighted. {len(found)} alert(s) match; a P450 can turn {sum(f['can be bioactivated'] for f in found)} of them into a reactive intermediate. "
                                                                                       f"What decides the risk is how readily the molecule is metabolised and how unprotected the alert is: {alerts.context(smiles)}."), mo.ui.table(table, selection=None, pagination=False)])


SIMS = {"1.3": (c_1_3, v_1_3), "1.4": (c_1_4, v_1_4), "1.9": (c_1_9, v_1_9), "1.10": (c_1_10, v_1_10)}
