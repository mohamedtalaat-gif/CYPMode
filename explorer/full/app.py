# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="CYPMode: the notebook, interactive", html_head_file="head.html")


@app.cell(hide_code=True)
def _():
    import json
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    here = Path(mo.notebook_location())
    for _path in (here, here.parents[1]):  # the simulators, and the project's own modules that they call
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    return alt, here, json, mo, np, pd


@app.cell(hide_code=True)
def _(here, json, pd):
    sections = json.loads((here / "data/sections.json").read_text())["parts"]
    label_points = pd.read_csv(here / "data/label_points.csv")
    compounds = pd.read_csv(here / "data/compounds.csv")
    return compounds, label_points, sections


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # CYPMode, the notebook made interactive

    Each tab is a part of the notebook and each entry is one of its sections, in the order of the notebook. Every entry is **live**: it has a simulator that runs the project's own code on what you give it
    (a molecule, a pair of potencies, a noise level, a pocket in three dimensions) and shows what the notebook shows for the real data. Engines that take hours (the co-folding models, the training of the models of the project)
    are not run here; where a section depends on one, its page replays what the engine really produced and says so. The models that some pages fit are small ones, on a few hundred compounds, made to teach an idea.
    """)


# -- 2.1 a molecule as text and as a graph ------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(compounds, mo):
    source = mo.ui.radio(options=["type a SMILES", "a compound of the training file"], value="type a SMILES", label="Molecule", inline=True)
    typed = mo.ui.text(value="c1ccc(cc1)Cn1ccnc1", label="SMILES", full_width=True)
    named = mo.ui.dropdown(options=dict(zip(compounds["Molecule_Name"].head(400), compounds["SMILES"].head(400), strict=True)), value=compounds["Molecule_Name"].iloc[0], label="Compound (first 400 of the file)")
    return named, source, typed


@app.cell(hide_code=True)
def _(named, source, typed):
    from simulators import molecule

    smiles = typed.value if source.value == "type a SMILES" else named.value
    mol = molecule.parse(smiles)
    bits = molecule.morgan_bits(mol) if mol is not None else {}
    return bits, mol, molecule, smiles


@app.cell(hide_code=True)
def _(bits, mo):
    bit_pick = mo.ui.dropdown(options={f"bit {b} ({len(envs)} place{'s' if len(envs) != 1 else ''})": b for b, envs in bits.items()}, value=None, label="A fingerprint bit") if bits else mo.ui.dropdown(options=[], label="A fingerprint bit")
    return (bit_pick,)


@app.cell(hide_code=True)
def _(bit_pick, bits, mo, mol, molecule, named, smiles, source, typed):
    if mol is None:
        sim_2_1 = mo.vstack([source, typed if source.value == "type a SMILES" else named, mo.callout("This text is not a molecule that RDKit can read.", kind="warn")])
    else:
        _donor = molecule.donor(smiles)
        _motifs = molecule.type_ii_motifs(mol)
        _highlight = ()
        _text = ""
        if bit_pick.value is not None:
            _center, _radius = bits[bit_pick.value][0]
            _highlight, _ = molecule.environment(mol, _center, _radius)
            _text = f"Bit {bit_pick.value} stands for the neighbourhood of atom {_center} within {_radius} bonds (highlighted)."
        elif _donor is not None:
            _highlight = (_donor["atom index"],)
            _text = f"Highlighted: the nitrogen that the restraint of the structure track puts on the iron (atom {_donor['atom index']}, named {_donor['Boltz-2 atom name']} in the Boltz-2 input)."
        sim_2_1 = mo.vstack([
            source, typed if source.value == "type a SMILES" else named,
            mo.hstack([mo.Html(molecule.drawing(mol, _highlight, indices=True)), mo.ui.table([{"property": k, "value": v} for k, v in molecule.summary(mol).items()], selection=None, pagination=False)], align="start"),
            mo.md(_text or "No Type II nitrogen: the restraint would not apply to this molecule."),
            mo.md(f"**Type II motifs found:** {', '.join(sorted({m['motif'] for m in _motifs})) or 'none'}. **Fingerprint:** {len(bits)} bits set of 2,048."),
            bit_pick,
            mo.accordion({"The graph: atoms": mo.ui.table(molecule.atom_rows(mol), selection=None, page_size=8), "The graph: bonds": mo.ui.table(molecule.bond_rows(mol), selection=None, page_size=8)}),
        ])
    return (sim_2_1,)


# -- 2.3 the label rule -----------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    d_rule = mo.ui.slider(start=1.0, stop=8.0, step=0.01, value=5.0, label="d, the direct pIC50", show_value=True)
    t_rule = mo.ui.slider(start=1.0, stop=8.0, step=0.01, value=5.6, label="t, the TDI-arm pIC50", show_value=True)
    enzyme_rule = mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Real compounds of", inline=True)
    return d_rule, enzyme_rule, t_rule


@app.cell(hide_code=True)
def _(alt, d_rule, enzyme_rule, label_points, mo, np, pd, t_rule):
    from simulators import label_rule

    _r = label_rule.evaluate(d_rule.value, t_rule.value)
    _real = label_points[label_points["isoform"] == enzyme_rule.value].assign(label=lambda f: f["label"].map({0: "negative", 1: "positive"}))
    _floor, _threshold = label_rule.ASSAY_FLOOR, label_rule.SHIFT_THRESHOLD
    _line = pd.DataFrame({"d": [1.0, _floor, _floor, 8.0], "t": [_floor + _threshold, _floor + _threshold, _floor + _threshold, 8.0 + _threshold], "segment": ["flat", "flat", "rising", "rising"]})
    _dots = alt.Chart(_real).mark_circle(size=14, opacity=0.35).encode(x=alt.X("d:Q", scale=alt.Scale(domain=[1, 8]), title="d (direct pIC50)"), y=alt.Y("t:Q", scale=alt.Scale(domain=[1, 8]), title="t (TDI-arm pIC50)"), color=alt.Color("label:N", scale=alt.Scale(domain=["negative", "positive"], range=["#8aa0b8", "#d9534f"])))
    _boundary = alt.Chart(_line).mark_line(color="black", strokeDash=[5, 3]).encode(x="d:Q", y="t:Q", detail="segment:N")
    _you = alt.Chart(pd.DataFrame({"d": [d_rule.value], "t": [t_rule.value]})).mark_point(size=160, color="black", filled=True, shape="diamond").encode(x="d:Q", y="t:Q")
    _verdict = "**positive**: a time-dependent inhibitor" if _r["single threshold"] else "**negative**"
    sim_2_3 = mo.vstack([
        mo.hstack([d_rule, t_rule], justify="start"),
        mo.md(f"The shift is s = t - max(d, 4) = **{_r['shift']:+.3f}**, the threshold is log10(2) = {_r['threshold']:.5f}, so the label is {_verdict}. {_r['branch'].capitalize()}. "
              f"The two forms of the rule agree: **{'yes' if _r['agree'] else 'NO'}**."),
        enzyme_rule,
        (_dots + _boundary + _you).properties(width=560, height=360),
        mo.md(f"The dots are the {len(_real)} real {enzyme_rule.value} compounds with both arms; the dashed line is the boundary of the rule. The diamond is your pair."),
    ])
    return (sim_2_3,)


# -- 2.5 measurement noise and the confidence z -----------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    d_noise = mo.ui.slider(start=1.0, stop=8.0, step=0.01, value=5.0, label="d", show_value=True)
    t_noise = mo.ui.slider(start=1.0, stop=8.0, step=0.01, value=5.5, label="t", show_value=True)
    sd_d_noise = mo.ui.slider(start=0.01, stop=1.5, step=0.01, value=0.25, label="sd(d)", show_value=True)
    sd_t_noise = mo.ui.slider(start=0.01, stop=1.5, step=0.01, value=0.25, label="sd(t)", show_value=True)
    return d_noise, sd_d_noise, sd_t_noise, t_noise


@app.cell(hide_code=True)
def _(alt, d_noise, mo, np, pd, sd_d_noise, sd_t_noise, t_noise):
    from scipy.stats import norm
    from simulators import confidence

    _c = confidence.evaluate(d_noise.value, t_noise.value, sd_d_noise.value, sd_t_noise.value)
    _x = np.linspace(_c["shift"] - 4 * _c["sd of the shift"], _c["shift"] + 4 * _c["sd of the shift"], 400)
    _curve = pd.DataFrame({"true shift": _x, "density": norm.pdf(_x, _c["shift"], _c["sd of the shift"])})
    _other = _curve[(_curve["true shift"] < _c["threshold"]) if _c["shift"] >= _c["threshold"] else (_curve["true shift"] >= _c["threshold"])]
    _chart = (alt.Chart(_curve).mark_line().encode(x="true shift:Q", y="density:Q")
              + alt.Chart(_other).mark_area(opacity=0.45, color="#d9534f").encode(x="true shift:Q", y="density:Q")
              + alt.Chart(pd.DataFrame({"x": [_c["threshold"]]})).mark_rule(color="black", strokeDash=[5, 3]).encode(x="x:Q")).properties(width=560, height=240)
    sim_2_5 = mo.vstack([
        mo.hstack([d_noise, t_noise], justify="start"), mo.hstack([sd_d_noise, sd_t_noise], justify="start"),
        mo.md(f"The shift is **{_c['shift']:+.3f}** with standard deviation **{_c['sd of the shift']:.3f}**"
              f"{' (both measurements are noisy)' if _c['d counts in the noise'] else ' (d is below the floor of 4, so only the noise of t counts)'}. "
              f"The threshold is {_c['threshold']:.3f}, so z = **{_c['z']:.2f}**, and the chance that the true shift lies on the other side, the red area, is **{_c['probability the label is wrong']:.1%}**."),
        _chart,
    ])
    return (sim_2_5,)


# -- 1.1 elimination ----------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    half_life = mo.ui.slider(start=0.5, stop=24.0, step=0.5, value=6.0, label="half-life (h)", show_value=True)
    interval = mo.ui.slider(start=2.0, stop=24.0, step=1.0, value=12.0, label="a dose every (h)", show_value=True)
    doses = mo.ui.slider(start=1, stop=10, step=1, value=6, label="number of doses", show_value=True)
    return doses, half_life, interval


@app.cell(hide_code=True)
def _(alt, doses, half_life, interval, mo):
    from simulators import kinetics

    _curve = kinetics.elimination(100.0, half_life.value, interval.value, doses.value, interval.value * doses.value + 2 * half_life.value).melt("time", var_name="curve", value_name="concentration")
    _chart = alt.Chart(_curve).mark_line().encode(x=alt.X("time:Q", title="time (h)"), y=alt.Y("concentration:Q", title="concentration (arbitrary units)"), color="curve:N").properties(width=560, height=260)
    sim_1_1 = mo.vstack([
        mo.hstack([half_life, interval, doses], justify="start"),
        mo.md(f"The rate constant is k = ln 2 / {half_life.value} h = **{0.6931471805599453 / half_life.value:.3f} per hour**. At steady state repeated dosing multiplies the concentration by **{kinetics.accumulation(half_life.value, interval.value):.2f}**; "
              "a dose every half-life doubles it, a dose every ten half-lives hardly changes it."),
        _chart,
    ])
    return kinetics, sim_1_1


# -- 1.2 Michaelis-Menten ------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    vmax_mm = mo.ui.slider(start=1.0, stop=20.0, step=0.5, value=10.0, label="Vmax", show_value=True)
    km_mm = mo.ui.slider(start=0.1, stop=20.0, step=0.1, value=2.0, label="Km", show_value=True)
    return km_mm, vmax_mm


@app.cell(hide_code=True)
def _(alt, kinetics, km_mm, mo, np, pd, vmax_mm):
    _s = np.linspace(0.0, 40.0, 300)
    _frame = pd.DataFrame({"substrate": _s, "rate": kinetics.michaelis_menten(_s, vmax_mm.value, km_mm.value)})
    _marks = pd.DataFrame({"substrate": [km_mm.value], "rate": [vmax_mm.value / 2]})
    _chart = (alt.Chart(_frame).mark_line().encode(x=alt.X("substrate:Q", title="[S]"), y=alt.Y("rate:Q", title="v", scale=alt.Scale(domain=[0, 20])))
              + alt.Chart(_marks).mark_point(size=120, filled=True, color="#d9534f").encode(x="substrate:Q", y="rate:Q")
              + alt.Chart(pd.DataFrame({"y": [vmax_mm.value]})).mark_rule(strokeDash=[5, 3]).encode(y="y:Q")).properties(width=560, height=260)
    sim_1_2 = mo.vstack([mo.hstack([vmax_mm, km_mm], justify="start"),
                         mo.md(f"The rate is v = Vmax [S] / (Km + [S]). The red point is [S] = Km = {km_mm.value}, where the rate is exactly half of Vmax = {vmax_mm.value}; the dashed line is Vmax, which the curve only approaches."), _chart])
    return (sim_1_2,)


# -- 1.5 reversible inhibition --------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    mode_ri = mo.ui.dropdown(options=["competitive", "non-competitive", "uncompetitive", "mixed"], value="competitive", label="Kind of inhibitor")
    inhibitor_ri = mo.ui.slider(start=0.0, stop=20.0, step=0.5, value=4.0, label="[I]", show_value=True)
    ki_ri = mo.ui.slider(start=0.5, stop=10.0, step=0.5, value=2.0, label="Ki", show_value=True)
    return inhibitor_ri, ki_ri, mode_ri


@app.cell(hide_code=True)
def _(alt, inhibitor_ri, kinetics, ki_ri, mo, mode_ri, np, pd):
    _s = np.linspace(0.0, 40.0, 300)
    _frame = pd.concat([pd.DataFrame({"substrate": _s, "rate": kinetics.michaelis_menten(_s, 10.0, 2.0), "curve": "no inhibitor"}),
                        pd.DataFrame({"substrate": _s, "rate": kinetics.inhibited_rate(_s, inhibitor_ri.value, 10.0, 2.0, ki_ri.value, mode_ri.value), "curve": "with the inhibitor"})])
    _alpha = 1 + inhibitor_ri.value / ki_ri.value
    _what = {"competitive": f"the apparent Km grows by alpha = {_alpha:.2f} and Vmax is unchanged (enough substrate wins)", "non-competitive": f"Vmax falls by alpha = {_alpha:.2f} and Km is unchanged",
             "uncompetitive": f"Vmax and Km both fall by alpha' = {_alpha:.2f}", "mixed": "the inhibitor binds the free enzyme and the complex with different strength (here Ki' = 4 Ki), so both constants change"}[mode_ri.value]
    sim_1_5 = mo.vstack([mo.hstack([mode_ri, inhibitor_ri, ki_ri], justify="start"), mo.md(f"For a {mode_ri.value} inhibitor {_what}. The enzyme here has Vmax = 10 and Km = 2."),
                         alt.Chart(_frame).mark_line().encode(x=alt.X("substrate:Q", title="[S]"), y=alt.Y("rate:Q", title="v", scale=alt.Scale(domain=[0, 10.5])), color="curve:N").properties(width=560, height=260)])
    return (sim_1_5,)


# -- 1.6 time-dependent inhibition -----------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    kinact_td = mo.ui.slider(start=0.01, stop=0.5, step=0.01, value=0.1, label="kinact (1/min)", show_value=True)
    ki_td = mo.ui.slider(start=0.5, stop=20.0, step=0.5, value=5.0, label="K_I (uM)", show_value=True)
    inhibitor_td = mo.ui.slider(start=0.0, stop=50.0, step=0.5, value=5.0, label="[I] (uM)", show_value=True)
    return inhibitor_td, ki_td, kinact_td


@app.cell(hide_code=True)
def _(alt, inhibitor_td, kinetics, ki_td, kinact_td, mo, np, pd):
    _minutes = np.linspace(0, 60, 200)
    _frame = pd.concat([pd.DataFrame({"minutes": _minutes, "activity left": kinetics.tdi_activity(_i, kinact_td.value, ki_td.value, _minutes)[1], "[I] (uM)": f"{_i:g}"}) for _i in sorted({inhibitor_td.value, ki_td.value, 5 * ki_td.value})])
    _kobs, _ = kinetics.tdi_activity(inhibitor_td.value, kinact_td.value, ki_td.value, [0.0])
    sim_1_6 = mo.vstack([mo.hstack([kinact_td, ki_td, inhibitor_td], justify="start"),
                         mo.md(f"kobs = kinact [I] / (K_I + [I]) = **{_kobs:.4f} per minute**, a half-life of the activity of **{(0.6931471805599453 / _kobs) if _kobs > 0 else float('inf'):.1f} minutes**. "
                               f"kobs is half of kinact at [I] = K_I = {ki_td.value} uM and can never exceed kinact = {kinact_td.value}. The curves are your [I], K_I and five times K_I."),
                         alt.Chart(_frame).mark_line().encode(x=alt.X("minutes:Q", title="minutes of pre-incubation"), y=alt.Y("activity left:Q", scale=alt.Scale(domain=[0, 1])), color="[I] (uM):N").properties(width=560, height=260)])
    return (sim_1_6,)


# -- 1.7 the Hill equation and the pIC50 scale -------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    pic50_hill = mo.ui.slider(start=2.0, stop=9.0, step=0.05, value=6.0, label="pIC50", show_value=True)
    slope_hill = mo.ui.slider(start=0.3, stop=4.0, step=0.1, value=1.0, label="Hill slope", show_value=True)
    noise_hill = mo.ui.slider(start=0.0, stop=0.3, step=0.01, value=0.05, label="noise of the readout", show_value=True)
    seed_hill = mo.ui.number(start=0, stop=999, step=1, value=1, label="random seed")
    return noise_hill, pic50_hill, seed_hill, slope_hill


@app.cell(hide_code=True)
def _(alt, kinetics, mo, noise_hill, np, pd, pic50_hill, seed_hill, slope_hill):
    from scipy.optimize import curve_fit

    _ic50 = kinetics.ic50_molar(pic50_hill.value)
    _c = np.logspace(-9, -3, 200)
    _doses = np.logspace(-8, -4, 8)
    _rng = np.random.default_rng(int(seed_hill.value))
    _y = np.clip(kinetics.hill(_doses, _ic50, slope_hill.value) + _rng.normal(0, noise_hill.value, _doses.size), -0.1, 1.2)
    try:
        _fit, _ = curve_fit(lambda lc, lic50, h: 1 / (1 + (10 ** lc / 10 ** lic50) ** h), np.log10(_doses), _y, p0=[-5.0, 1.0], maxfev=5000)
        _fit_text = f"Eight noisy points fitted by the Hill equation give pIC50 = **{-_fit[0]:.2f}** (the true value is {pic50_hill.value:.2f}) and slope {_fit[1]:.2f}. Change the seed to repeat the experiment: this is the measurement noise of section 2.5."
        _fit_curve = pd.DataFrame({"conc": _c, "activity": 1 / (1 + (_c / 10 ** _fit[0]) ** _fit[1]), "curve": "fitted"})
    except RuntimeError:
        _fit_text, _fit_curve = "The fit did not converge for these noisy points.", pd.DataFrame({"conc": [], "activity": [], "curve": []})
    _lines = pd.concat([pd.DataFrame({"conc": _c, "activity": kinetics.hill(_c, _ic50, slope_hill.value), "curve": "true"}), _fit_curve])
    _chart = (alt.Chart(_lines).mark_line().encode(x=alt.X("conc:Q", scale=alt.Scale(type="log"), title="[I] (mol/L)"), y=alt.Y("activity:Q", scale=alt.Scale(domain=[-0.1, 1.2])), color="curve:N")
              + alt.Chart(pd.DataFrame({"conc": _doses, "activity": _y})).mark_point(size=60, filled=True, color="black").encode(x="conc:Q", y="activity:Q")).properties(width=560, height=260)
    sim_1_7 = mo.vstack([mo.hstack([pic50_hill, slope_hill], justify="start"), mo.hstack([noise_hill, seed_hill], justify="start"),
                         mo.md(f"pIC50 = {pic50_hill.value:.2f} means IC50 = **{_ic50 * 1e6:.3g} uM**; at [I] = IC50 the activity is one half whatever the slope. {_fit_text}"), _chart])
    return (sim_1_7,)


# -- 1.8 the two arms and the shift between them ---------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    direct_arm = mo.ui.slider(start=3.0, stop=8.0, step=0.05, value=5.0, label="pIC50, direct arm", show_value=True)
    tdi_arm = mo.ui.slider(start=3.0, stop=8.0, step=0.05, value=5.4, label="pIC50, TDI arm", show_value=True)
    return direct_arm, tdi_arm


@app.cell(hide_code=True)
def _(direct_arm, kinetics, mo, tdi_arm):
    _r = kinetics.arm_shift(direct_arm.value, tdi_arm.value)
    sim_1_8 = mo.vstack([mo.hstack([direct_arm, tdi_arm], justify="start"),
                         mo.md(f"Direct arm: IC50 = **{_r['direct IC50 (uM)']:.3g} uM**. TDI arm (after the head start): IC50 = **{_r['TDI IC50 (uM)']:.3g} uM**. The compound became **{_r['IC50 shift (fold)']:.2f} times** more potent, a shift of {_r['pIC50 shift']:+.2f} in pIC50. "
                               f"The label of section 2.3 asks for more than log10(2) = {_r['log10(2)']:.3f}, a factor of 2 (with the direct arm at or above the floor of 4).")])
    return (sim_1_8,)


# -- 3.2 Tanimoto similarity (and 3.1 the fingerprint) ---------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    smiles_a = mo.ui.text(value="c1ccc(cc1)Cn1ccnc1", label="Molecule A (SMILES)", full_width=True)
    smiles_b = mo.ui.text(value="c1ccc(cc1)Cn1cncn1", label="Molecule B (SMILES)", full_width=True)
    return smiles_a, smiles_b


@app.cell(hide_code=True)
def _(mo, molecule, smiles_a, smiles_b):
    from simulators import fingerprints

    _a, _b = molecule.parse(smiles_a.value), molecule.parse(smiles_b.value)
    if _a is None or _b is None:
        sim_3_2 = mo.vstack([smiles_a, smiles_b, mo.callout("One of the two texts is not a molecule that RDKit can read.", kind="warn")])
    else:
        _t = fingerprints.tanimoto(_a, _b)
        sim_3_2 = mo.vstack([smiles_a, smiles_b, mo.hstack([mo.Html(molecule.drawing(_a, size=(300, 240))), mo.Html(molecule.drawing(_b, size=(300, 240)))]),
                             mo.md(f"A sets {_t['bits of A']} bits and B sets {_t['bits of B']}; {_t['shared']} bits are set in both, {_t['union']} in at least one. The Tanimoto similarity is {_t['shared']} / {_t['union']} = **{_t['similarity']:.3f}** "
                                   f"(RDKit gives {_t['rdkit']:.3f}). Identical molecules give 1, molecules that share no neighbourhood give 0; the project calls a compound an analogue at 0.5 or more.")])
    return fingerprints, sim_3_2


# -- 3.3 descriptors ---------------------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    smiles_desc = mo.ui.text(value="CC(C)(C)OC(=O)NCCCCCC(=O)NCc1cccnc1", label="SMILES", full_width=True)
    return (smiles_desc,)


@app.cell(hide_code=True)
def _(fingerprints, mo, molecule, smiles_desc):
    _m = molecule.parse(smiles_desc.value)
    sim_3_3 = mo.vstack([smiles_desc, mo.callout("This text is not a molecule that RDKit can read.", kind="warn")]) if _m is None else mo.vstack([smiles_desc, mo.hstack([mo.Html(molecule.drawing(_m, size=(380, 280))), mo.ui.table(fingerprints.descriptor_rows(_m), selection=None, pagination=False)], align="start")])
    return (sim_3_3,)


# -- 4.10 how a model is judged ------------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    prevalence_m = mo.ui.slider(start=0.02, stop=0.6, step=0.01, value=0.3, label="share of positives", show_value=True)
    separation_m = mo.ui.slider(start=0.0, stop=3.0, step=0.1, value=1.2, label="how far apart the scores are", show_value=True)
    n_m = mo.ui.slider(start=30, stop=3000, step=10, value=300, label="compounds scored", show_value=True)
    threshold_m = mo.ui.slider(start=-1.0, stop=3.0, step=0.05, value=0.8, label="call positive above", show_value=True)
    return n_m, prevalence_m, separation_m, threshold_m


@app.cell(hide_code=True)
def _(alt, mo, n_m, pd, prevalence_m, separation_m, threshold_m):
    from simulators import statistics as stat

    _labels, _scores = stat.draw_scores(prevalence_m.value, separation_m.value, int(n_m.value), 7)
    _m = stat.metrics_at(_labels, _scores, threshold_m.value)
    _spread = stat.spread(prevalence_m.value, separation_m.value, int(n_m.value), threshold_m.value, repeats=120)
    _frame = pd.DataFrame({"score": _scores, "label": ["positive" if _l else "negative" for _l in _labels]})
    _hist = alt.Chart(_frame).mark_bar(opacity=0.6).encode(x=alt.X("score:Q", bin=alt.Bin(maxbins=40)), y=alt.Y("count()", stack=None), color=alt.Color("label:N", scale=alt.Scale(domain=["negative", "positive"], range=["#8aa0b8", "#d9534f"]))).properties(width=560, height=200)
    _rule = alt.Chart(pd.DataFrame({"x": [threshold_m.value]})).mark_rule(color="black", strokeDash=[5, 3]).encode(x="x:Q")
    sim_4_10 = mo.vstack([
        mo.hstack([prevalence_m, separation_m], justify="start"), mo.hstack([n_m, threshold_m], justify="start"),
        mo.md(f"AUC **{_m['AUC']:.3f}**, MCC **{_m['MCC']:.3f}**, precision {_m['precision']:.2f}, recall {_m['recall']:.2f}, {_m['positives called']} compounds called positive. "
              f"Draw the same experiment again and again: the AUC moves by about **{_spread['AUC sd']:.3f}** and the MCC by about **{_spread['MCC sd']:.3f}** (one standard deviation over 120 fresh samples of {int(n_m.value)} compounds). "
              "Make the sample smaller and both grow; the MCC always moves more, because it uses one threshold and the AUC uses all of them."),
        _hist + _rule,
    ])
    return sim_4_10, stat


# -- 5.1 the Type II motif ---------------------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    smiles_t2 = mo.ui.text(value="c1cc(c(cc1F)F)C(Cn2cncn2)(Cn3cncn3)O", label="SMILES (the default is fluconazole)", full_width=True)
    return (smiles_t2,)


@app.cell(hide_code=True)
def _(mo, molecule, smiles_t2):
    from cypmode.metrics.motifs import TYPE_II_MOTIFS as _MOTIFS

    _m = molecule.parse(smiles_t2.value)
    if _m is None:
        sim_5_1 = mo.vstack([smiles_t2, mo.callout("This text is not a molecule that RDKit can read.", kind="warn")])
    else:
        _found = molecule.type_ii_motifs(_m)
        _atoms = tuple(sorted({a for f in _found for a in f["atoms"]}))
        _donor = molecule.donor(smiles_t2.value)
        sim_5_1 = mo.vstack([smiles_t2, mo.hstack([mo.Html(molecule.drawing(_m, _atoms)), mo.ui.table([{"motif": k, "SMARTS pattern": v, "matches": sum(1 for f in _found if f["motif"] == k)} for k, v in _MOTIFS.items()], selection=None, pagination=False)], align="start"),
                             mo.md(("The nitrogen that would be put on the iron is atom " + str(_donor["atom index"]) + f" (named {_donor['Boltz-2 atom name']} in the Boltz-2 input)." if _donor else "No unsubstituted aromatic nitrogen inside a Type II motif: the restraint of the structure track would not apply.") + " The highlighted atoms are the matches of the patterns.")])
    return (sim_5_1,)


# -- 6.1 and 6.2 why a score on the data used to choose lies ------------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    k_models = mo.ui.slider(start=1, stop=200, step=1, value=30, label="models tried", show_value=True)
    n_valid = mo.ui.slider(start=30, stop=500, step=10, value=80, label="compounds in the validation set", show_value=True)
    return k_models, n_valid


@app.cell(hide_code=True)
def _(k_models, mo, n_valid, stat):
    _o = stat.selection_optimism(int(k_models.value), int(n_valid.value), repeats=80)
    sim_6_1 = mo.vstack([mo.hstack([k_models, n_valid], justify="start"),
                         mo.md(f"All {int(k_models.value)} models are random numbers: they know nothing. The best of them scores an AUC of **{_o['best validation AUC']:.3f}** on the data used to choose it, and **{_o['its AUC on fresh data']:.3f}** on new data, which is chance. "
                               "The gap is the optimism of choosing; it grows with the number of models and shrinks with the size of the validation set. This is why the project scores a choice only on folds that took no part in it (nested cross-validation).")])
    return (sim_6_1,)


# -- 6.5 comparing two models: the corrected resampled t -----------------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    differences_box = mo.ui.text_area(value="0.012, 0.004, 0.019, 0.008, 0.015, 0.001, 0.010, 0.007, 0.013, 0.006", label="Per-fold differences in AUC (model A minus model B), separated by commas", full_width=True)
    n_train_ct = mo.ui.number(start=50, stop=100000, step=50, value=3000, label="compounds in one training set")
    n_test_ct = mo.ui.number(start=10, stop=50000, step=10, value=750, label="compounds in one test set")
    return differences_box, n_test_ct, n_train_ct


@app.cell(hide_code=True)
def _(differences_box, mo, n_test_ct, n_train_ct, stat):
    try:
        _d = [float(x) for x in differences_box.value.replace("\n", ",").split(",") if x.strip()]
        _r = stat.compare(_d, int(n_train_ct.value), int(n_test_ct.value))
        _body = mo.md(f"{_r['splits']} splits, A is better in {_r['folds better']}. Mean difference **{_r['mean difference']:+.4f}**. The naive paired t uses the factor 1/J = {_r['naive factor']:.3f} for the variance and gives t = {_r['naive t']:.2f}, p = **{_r['naive p']:.4f}**. "
                      f"The corrected t uses 1/J + n_test/n_train = {_r['variance factor']:.3f} and gives t = {_r['corrected t']:.2f}, p = **{_r['corrected p']:.4f}**. "
                      "The folds share training data, so their differences are correlated and the naive test is too optimistic; the project uses only the corrected one (section 6.5). Both are the project's own functions.")
    except ValueError:
        _body = mo.callout("Give at least two numbers, separated by commas.", kind="warn")
    sim_6_5 = mo.vstack([differences_box, mo.hstack([n_train_ct, n_test_ct], justify="start"), _body])
    return (sim_6_5,)



# -- 2.9 standardising structures ------------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    smiles_std = mo.ui.text(value="CC(=O)Nc1ccc(O)cc1.[Na+].[Cl-]", label="SMILES (try a salt, a charged amine, a mixture)", full_width=True)
    return (smiles_std,)


@app.cell(hide_code=True)
def _(mo, molecule, smiles_std):
    from cypmode.challenge.structure_standardization import standardized_parent_smiles

    _m = molecule.parse(smiles_std.value)
    _parent = standardized_parent_smiles(smiles_std.value) if _m is not None else None
    if _m is None or _parent is None:
        sim_2_9 = mo.vstack([smiles_std, mo.callout("This text is not a molecule that RDKit can read.", kind="warn")])
    else:
        _p = molecule.parse(_parent)
        sim_2_9 = mo.vstack([smiles_std, mo.hstack([mo.vstack([mo.md("**as given**"), mo.Html(molecule.drawing(_m, size=(300, 240)))]), mo.vstack([mo.md(f"**standard parent**: `{_parent}`"), mo.Html(molecule.drawing(_p, size=(300, 240)))])]),
                             mo.md("The project's function `standardized_parent_smiles` strips salts and solvents and neutralises charges, so that two strings of the same compound become one string before any comparison of compounds.")])
    return (sim_2_9,)


# -- 4.1 generalisation and overfitting ---------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    degree_of = mo.ui.slider(start=0, stop=14, step=1, value=3, label="degree of the polynomial", show_value=True)
    n_of = mo.ui.slider(start=10, stop=120, step=5, value=30, label="training points", show_value=True)
    noise_of = mo.ui.slider(start=0.05, stop=0.6, step=0.05, value=0.3, label="noise", show_value=True)
    seed_of = mo.ui.number(start=0, stop=999, value=1, label="random seed")
    return degree_of, n_of, noise_of, seed_of


@app.cell(hide_code=True)
def _(alt, degree_of, mo, n_of, noise_of, seed_of):
    from simulators import learning

    _table = learning.overfitting(14, int(n_of.value), noise_of.value, int(seed_of.value))
    _points, _fit = learning.polynomial_curve(int(degree_of.value), int(n_of.value), noise_of.value, int(seed_of.value))
    _fit_long = _fit.melt("x", var_name="curve", value_name="y")
    _left = (alt.Chart(_fit_long).mark_line().encode(x="x:Q", y=alt.Y("y:Q", scale=alt.Scale(domain=[-2.2, 2.2])), color="curve:N") + alt.Chart(_points).mark_point(color="black", filled=True).encode(x="x:Q", y="y:Q")).properties(width=330, height=240, title="the fit and the truth")
    _long = _table.melt("degree", var_name="data", value_name="error")
    _right = (alt.Chart(_long).mark_line(point=True).encode(x="degree:Q", y=alt.Y("error:Q", scale=alt.Scale(type="log")), color="data:N") + alt.Chart(_table[_table["degree"] == degree_of.value].assign(error=0.0)).mark_rule(strokeDash=[4, 3]).encode(x="degree:Q")).properties(width=330, height=240, title="the error at every degree")
    _row = _table[_table["degree"] == degree_of.value].iloc[0]
    sim_4_1 = mo.vstack([mo.hstack([degree_of, n_of], justify="start"), mo.hstack([noise_of, seed_of], justify="start"),
                         mo.md(f"Degree {int(degree_of.value)}: error on the data it was fitted to **{_row['training error']:.3f}**, error on new data **{_row['error on new data']:.3f}**. The training error only falls as the model grows; the error on new data falls and then rises. "
                               "That gap is why a score on the training data lies (section 6.1)."), mo.hstack([_left, _right])])
    return learning, sim_4_1


# -- 4.2 the split of a decision tree ------------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    sep_tree = mo.ui.slider(start=0.0, stop=4.0, step=0.1, value=2.0, label="how far apart the two classes are", show_value=True)
    n_tree = mo.ui.slider(start=40, stop=1000, step=20, value=300, label="compounds", show_value=True)
    return n_tree, sep_tree


@app.cell(hide_code=True)
def _(alt, learning, mo, n_tree, pd, sep_tree):
    _points, _impurity, _best = learning.tree_split(sep_tree.value, int(n_tree.value), 5)
    _hist = (alt.Chart(_points).mark_bar(opacity=0.6).encode(x=alt.X("x:Q", bin=alt.Bin(maxbins=40)), y=alt.Y("count()", stack=None), color=alt.Color("class:N", scale=alt.Scale(domain=["negative", "positive"], range=["#8aa0b8", "#d9534f"])))
             + alt.Chart(pd.DataFrame({"x": [_best]})).mark_rule(color="black", strokeDash=[5, 3]).encode(x="x:Q")).properties(width=560, height=170)
    _curve = (alt.Chart(_impurity).mark_line().encode(x="threshold:Q", y=alt.Y("impurity after the split:Q", scale=alt.Scale(domain=[0, 0.5]))) + alt.Chart(pd.DataFrame({"x": [_best]})).mark_rule(color="black", strokeDash=[5, 3]).encode(x="x:Q")).properties(width=560, height=170)
    sim_4_2 = mo.vstack([mo.hstack([sep_tree, n_tree], justify="start"),
                         mo.md(f"A tree asks 'is x below a?' and chooses the a that leaves the two groups as pure as possible (the Gini impurity 2p(1-p), averaged over the groups). The best question here is a = **{_best:.2f}**, "
                               f"which leaves an impurity of **{_impurity['impurity after the split'].min():.3f}** (0.5 means no information, 0 a perfect split). The further apart the classes, the lower the curve."), _hist, _curve])
    return (sim_4_2,)


# -- 4.3 gradient boosting ---------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    trees_gb = mo.ui.slider(start=1, stop=200, step=1, value=10, label="trees added", show_value=True)
    rate_gb = mo.ui.slider(start=0.02, stop=1.0, step=0.02, value=0.3, label="learning rate", show_value=True)
    depth_gb = mo.ui.slider(start=1, stop=4, step=1, value=2, label="depth of each tree", show_value=True)
    return depth_gb, rate_gb, trees_gb


@app.cell(hide_code=True)
def _(alt, depth_gb, learning, mo, pd, rate_gb, trees_gb):
    _b = learning.boosting(int(trees_gb.value), rate_gb.value, int(depth_gb.value), 0.25, 3)
    _fit_long = _b["fit"].melt("x", var_name="curve", value_name="y")
    _left = (alt.Chart(_fit_long).mark_line().encode(x="x:Q", y=alt.Y("y:Q", scale=alt.Scale(domain=[-2.2, 2.2])), color="curve:N") + alt.Chart(_b["points"]).mark_point(color="black", filled=True).encode(x="x:Q", y="y:Q")).properties(width=330, height=240, title=f"after {int(trees_gb.value)} trees")
    _err = _b["errors"].melt("trees", var_name="data", value_name="error")
    _right = (alt.Chart(_err).mark_line().encode(x="trees:Q", y="error:Q", color="data:N") + alt.Chart(pd.DataFrame({"x": [trees_gb.value]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")).properties(width=330, height=240, title="the error as trees are added")
    _row = _b["errors"].iloc[int(trees_gb.value) - 1]
    sim_4_3 = mo.vstack([mo.hstack([trees_gb, rate_gb, depth_gb], justify="start"),
                         mo.md(f"Each new small tree is fitted to what the trees before it still get wrong, and added with weight equal to the learning rate. After {int(trees_gb.value)} trees: error on the training data **{_row['training error']:.3f}**, on new data **{_row['error on new data']:.3f}**. "
                               "Too many trees at a high learning rate fits the noise; the project's models stop early and are judged only on data they did not see."), mo.hstack([_left, _right])])
    return (sim_4_3,)


# -- 4.4 forests -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    max_trees_rf = mo.ui.slider(start=5, stop=200, step=5, value=100, label="trees in the forest", show_value=True)
    noise_rf = mo.ui.slider(start=0.05, stop=0.6, step=0.05, value=0.3, label="noise", show_value=True)
    return max_trees_rf, noise_rf


@app.cell(hide_code=True)
def _(alt, learning, max_trees_rf, mo, noise_rf):
    _t = learning.forest(int(max_trees_rf.value), noise_rf.value, 4).rename(columns={"error on new data": "the forest"}).melt("trees", var_name="model", value_name="error on new data")
    sim_4_4 = mo.vstack([mo.hstack([max_trees_rf, noise_rf], justify="start"),
                         mo.md("One deep tree fits the noise and its answer changes a lot with the sample (high variance). Many trees, each fitted to a resample of the data, disagree in their mistakes, and their average has a smaller error: the line falls as trees are added and flattens."),
                         alt.Chart(_t).mark_line(point=True).encode(x=alt.X("trees:Q", scale=alt.Scale(type="log")), y="error on new data:Q", color="model:N").properties(width=560, height=260)])
    return (sim_4_4,)


# -- 6.3 nested cross-validation ---------------------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    n_nested = mo.ui.slider(start=500, stop=6000, step=100, value=3000, label="compounds", show_value=True)
    outer_nested = mo.ui.slider(start=3, stop=10, step=1, value=5, label="outer folds", show_value=True)
    inner_nested = mo.ui.slider(start=2, stop=10, step=1, value=5, label="inner folds", show_value=True)
    repeats_nested = mo.ui.slider(start=1, stop=5, step=1, value=3, label="repeats", show_value=True)
    candidates_nested = mo.ui.slider(start=1, stop=60, step=1, value=24, label="candidate models", show_value=True)
    return candidates_nested, inner_nested, n_nested, outer_nested, repeats_nested


@app.cell(hide_code=True)
def _(candidates_nested, inner_nested, mo, n_nested, outer_nested, repeats_nested):
    from simulators import nested

    _p = nested.plan(int(n_nested.value), int(outer_nested.value), int(inner_nested.value), int(repeats_nested.value), int(candidates_nested.value))
    sim_6_3 = mo.vstack([mo.hstack([n_nested, outer_nested, inner_nested], justify="start"), mo.hstack([repeats_nested, candidates_nested], justify="start"),
                         mo.md(f"Every outer fold holds back **{_p['outer test size']}** compounds and keeps **{_p['outer training size']}**. On those the choice among {int(candidates_nested.value)} models is made by an inner cross-validation "
                               f"(each fit uses **{_p['inner training size']}** compounds and scores **{_p['inner test size']}**), which is **{_p['fits in the inner loop of one outer fold']}** fits; with the refits **{_p['fits per outer fold']}** per outer fold. "
                               f"The outer score is computed on compounds that played no part in the choice. {int(repeats_nested.value)} repeats of {int(outer_nested.value)} folds give **{_p['outer scores']}** outer scores and **{_p['total fits']:,}** fits in all; "
                               f"the t-test of section 6.5 then uses the factor 1/J + n_test/n_train = **{_p['correction factor of the t-test']:.3f}**.")])
    return nested, sim_6_3


# -- 6.7 the gate ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    gate_box_1 = mo.ui.text_area(value="0.012, 0.015, 0.011, 0.014, 0.013, 0.016, 0.012, 0.010, 0.015, 0.013, 0.014, 0.012, 0.011, 0.016, 0.013", label="Differences at z0 = 1.0 (15 outer scores, stack with the addition minus stack without)", full_width=True)
    gate_box_2 = mo.ui.text_area(value="0.011, 0.014, 0.012, 0.013, 0.012, 0.015, 0.011, 0.010, 0.014, 0.012, 0.013, 0.011, 0.012, 0.015, 0.012", label="Differences at z0 = 1.96", full_width=True)
    rho_gate = mo.ui.slider(start=0.0, stop=0.99, step=0.01, value=0.9, label="correlation of the two cut-offs", show_value=True)
    p_box = mo.ui.text(value="0.003, 0.02, 0.04, 0.2, 0.5", label="p-values of several comparisons (Holm)", full_width=True)
    return gate_box_1, gate_box_2, p_box, rho_gate


@app.cell(hide_code=True)
def _(alt, gate_box_1, gate_box_2, mo, p_box, pd, rho_gate):
    from simulators import gates

    def _nums(box):
        return [float(x) for x in box.value.replace("\n", ",").split(",") if x.strip()]

    try:
        _v = gates.verdict({1.0: _nums(gate_box_1), 1.96: _nums(gate_box_2)}, 3000, 750)
        _verdict = mo.vstack([mo.ui.table(_v["rows"], selection=None, pagination=False), mo.md(f"The gate (`GATE_ALPHA` = {_v['alpha']} at both cut-offs, mean difference above zero): **{'MET' if _v['met'] else 'NOT MET'}**.")])
    except ValueError:
        _verdict = mo.callout("Give at least two numbers in each box.", kind="warn")
    _f = gates.false_alarms(rho_gate.value)
    _fa = pd.DataFrame({"rule": [k for k in _f if k != "alpha"], "share of useless additions that pass": [v for k, v in _f.items() if k != "alpha"]})
    try:
        _holm = mo.ui.table(gates.holm_table([float(x) for x in p_box.value.split(",") if x.strip()]), selection=None, pagination=False)
    except ValueError:
        _holm = mo.callout("Give p-values between 0 and 1, separated by commas.", kind="warn")
    sim_6_7 = mo.vstack([
        mo.md("**1. The gate on your own differences.** The addition passes only when the corrected p is below the level and the mean difference is positive at both cut-offs."), gate_box_1, gate_box_2, _verdict,
        mo.md("**2. Why both cut-offs must pass.** An addition that adds nothing still has noisy gains. If the analyst may choose the cut-off after seeing both, more of them look good than the level promises; if both must pass, fewer do."), rho_gate,
        alt.Chart(_fa).mark_bar().encode(x=alt.X("share of useless additions that pass:Q", axis=alt.Axis(format="%")), y=alt.Y("rule:N", sort=None)).properties(width=560, height=130),
        mo.md("**3. Many comparisons.** The Holm correction (the project's own function) raises the p-values so that the chance of any false pass stays at the level."), p_box, _holm,
    ])
    return gates, sim_6_7


# -- 7.3 reading across: the nearest neighbours ----------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(compounds, fingerprints):
    from simulators import neighbours

    train_prints = neighbours.prepare(compounds["SMILES"])
    return neighbours, train_prints


@app.cell(hide_code=True)
def _(mo):
    smiles_nn = mo.ui.text(value="c1ccc(cc1)Cn1ccnc1", label="SMILES of a compound", full_width=True)
    k_nn = mo.ui.slider(start=1, stop=15, step=1, value=5, label="neighbours", show_value=True)
    enzyme_nn = mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Labels of", inline=True)
    weighted_nn = mo.ui.checkbox(value=False, label="weight the vote by similarity")
    return enzyme_nn, k_nn, smiles_nn, weighted_nn


@app.cell(hide_code=True)
def _(compounds, enzyme_nn, k_nn, mo, molecule, neighbours, smiles_nn, train_prints, weighted_nn):
    _found = neighbours.nearest(smiles_nn.value, train_prints, int(k_nn.value))
    if not _found:
        sim_7_3 = mo.vstack([smiles_nn, mo.callout("This text is not a molecule that RDKit can read.", kind="warn")])
    else:
        _col = f"{enzyme_nn.value}_is_TDI"
        _rows = [{"similarity": round(sim, 3), "compound": compounds["Molecule_Name"].iloc[i], "SMILES": compounds["SMILES"].iloc[i], f"{enzyme_nn.value} label": compounds[_col].iloc[i]} for i, sim in _found]
        _share = neighbours.vote([r[f"{enzyme_nn.value} label"] for r in _rows], [r["similarity"] for r in _rows], weighted_nn.value)
        _read = "none of the neighbours has a label for this enzyme" if _share is None else f"a vote of the neighbours says **{_share:.0%}** positive"
        sim_7_3 = mo.vstack([smiles_nn, mo.hstack([k_nn, enzyme_nn, weighted_nn], justify="start"), mo.ui.table(_rows, selection=None, pagination=False),
                             mo.md(f"The most similar training compound has a Tanimoto similarity of **{_found[0][1]:.2f}**; {_read}. The closer the best neighbour, the more this reading is worth; below about 0.4 it is close to a guess (section 7.3).")])
    return (sim_7_3,)



# -- 8.4 turning scores into calls --------------------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    prevalence_calls = mo.ui.slider(start=0.03, stop=0.5, step=0.01, value=0.31, label="share of positives among the scored compounds", show_value=True)
    separation_calls = mo.ui.slider(start=0.3, stop=3.0, step=0.1, value=1.3, label="how far apart the scores are", show_value=True)
    n_calls = mo.ui.slider(start=100, stop=5000, step=100, value=750, label="compounds scored", show_value=True)
    return n_calls, prevalence_calls, separation_calls


@app.cell(hide_code=True)
def _(alt, mo, n_calls, np, pd, prevalence_calls, separation_calls, stat):
    _rows = pd.DataFrame(stat.calls_by_multiple(prevalence_calls.value, separation_calls.value, int(n_calls.value), np.round(np.arange(0.4, 2.51, 0.1), 2)))
    _best = _rows.iloc[int(_rows["MCC"].idxmax())]
    _long = _rows.melt("multiple of the positive rate", value_vars=["MCC", "precision", "recall"], var_name="measure", value_name="value")
    sim_8_4 = mo.vstack([mo.hstack([prevalence_calls, separation_calls, n_calls], justify="start"),
                         mo.md("The model gives every compound a score; the call is 'positive' for the top share of the scores. The project fixes that share as a **multiple of the positive rate** of the scored set (section 8.4): at 1 it calls as many positives as there should be, "
                               f"above 1 it calls more and trades precision for recall. Here the best multiple is **{_best['multiple of the positive rate']:.1f}** (MCC {_best['MCC']:.3f}); the curve is flat near its top, which is why a smoothed choice is used and not the single best point. "
                               "Draw fewer compounds and the best multiple jumps about: the MCC is noisy."),
                         alt.Chart(_long).mark_line(point=True).encode(x="multiple of the positive rate:Q", y="value:Q", color="measure:N").properties(width=560, height=260)])
    return (sim_8_4,)


# -- 9.3 the label as a threshold of two noisy numbers -------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    d_cm = mo.ui.slider(start=3.0, stop=8.0, step=0.05, value=5.0, label="d measured", show_value=True)
    t_cm = mo.ui.slider(start=3.0, stop=8.0, step=0.05, value=5.4, label="t measured", show_value=True)
    sd_d_cm = mo.ui.slider(start=0.05, stop=1.2, step=0.05, value=0.3, label="sd(d)", show_value=True)
    sd_t_cm = mo.ui.slider(start=0.05, stop=1.2, step=0.05, value=0.3, label="sd(t)", show_value=True)
    return d_cm, sd_d_cm, sd_t_cm, t_cm


@app.cell(hide_code=True)
def _(d_cm, mo, sd_d_cm, sd_t_cm, t_cm):
    from simulators import confidence as _confidence

    _c = _confidence.committee(d_cm.value, t_cm.value, sd_d_cm.value, sd_t_cm.value)
    sim_9_3 = mo.vstack([mo.hstack([d_cm, t_cm], justify="start"), mo.hstack([sd_d_cm, sd_t_cm], justify="start"),
                         mo.md(f"Both measurements are uncertain, so the label is uncertain. Draw 20,000 pairs around what was measured and apply the rule to each: **{_c['simulated share positive']:.1%}** of the draws are positive. The normal-curve probability of section 2.5 gives "
                               f"**{_c['analytic probability']:.1%}**. As measured the label is **{'positive' if _c['label as measured'] else 'negative'}** (z = {_c['z']:.2f}). A model that outputs this probability, instead of one label, keeps what the noise "
                               "says; this is the committee of section 9.3.")])
    return (sim_9_3,)


# -- 9.8 training on label confidence --------------------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    enzyme_w = mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Compounds of", inline=True)
    mode_w = mo.ui.dropdown(options={"every row, weight 1": "", "drop rows with z below 1.0": "drop1.0", "drop rows with z below 1.96": "drop1.96", "soft weights: min(1, z/2), at least 0.1": "soft"}, value="every row, weight 1", label="Training rule")
    return enzyme_w, mode_w


@app.cell(hide_code=True)
def _(alt, enzyme_w, label_points, mo, mode_w, pd):
    from simulators import confidence as _conf

    _z = label_points[label_points["isoform"] == enzyme_w.value]["z"].to_numpy()
    _all = pd.DataFrame([{"rule": name, "rows kept": (_r := _conf.weights_by_mode(_z, m))["rows kept"], "share kept": _r["rows kept"] / _r["rows"], "total weight": round(_r["total weight"], 1)} for name, m in [("every row, weight 1", ""), ("drop z < 1.0", "drop1.0"), ("drop z < 1.96", "drop1.96"), ("soft", "soft")]])
    _chosen = _conf.weights_by_mode(_z, mode_w.value)
    _hist = alt.Chart(pd.DataFrame({"z": _z.clip(max=6), "kept": _chosen["keep"], "weight": _chosen["weight"]})).mark_bar().encode(x=alt.X("z:Q", bin=alt.Bin(maxbins=40), title="confidence z of the training compounds"), y=alt.Y("sum(weight):Q", title="training weight"), color=alt.Color("kept:N", scale=alt.Scale(domain=[True, False], range=["#0e9f8a", "#c9d1de"]))).properties(width=560, height=220)
    sim_9_8 = mo.vstack([mo.hstack([enzyme_w, mode_w], justify="start"),
                         mo.md(f"With this rule **{_chosen['rows kept']} of {_chosen['rows']}** {enzyme_w.value} compounds with both arms train the model, with a total weight of {_chosen['total weight']:.0f}. The bars show where in z the weight sits. "
                               "A compound whose label the noise could flip (small z) is dropped or down-weighted; the cost is fewer or lighter examples (section 9.8). The project tested these rules by the gate and kept none."),
                         _hist, mo.ui.table(_all.to_dict("records"), selection=None, pagination=False)])
    return (sim_9_8,)


# -- 9.11 the ledger of the gates and the correction for the number of attempts -------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(here, pd):
    paired_real = pd.read_csv(here / "data/public/paired.csv")
    return (paired_real,)


@app.cell(hide_code=True)
def _(mo, paired_real):
    enzyme_l = mo.ui.dropdown(options=["both enzymes", "CYP3A4", "CYP2D6"], value="both enzymes", label="Enzyme")
    cut_l = mo.ui.dropdown(options=["both cuts"] + [f"z0 = {z}" for z in sorted(paired_real["z0"].unique())], value="both cuts", label="Confidence cut")
    floor_l = mo.ui.slider(start=0.0, stop=0.03, step=0.0005, value=0.0099, label="gain at least", show_value=True)
    alpha_l = mo.ui.slider(start=0.001, stop=0.25, step=0.001, value=0.05, label="corrected p below", show_value=True)
    holm_l = mo.ui.checkbox(value=False, label="also correct for the number of comparisons (Holm)")
    return alpha_l, cut_l, enzyme_l, floor_l, holm_l


@app.cell(hide_code=True)
def _(alpha_l, alt, cut_l, enzyme_l, floor_l, holm_l, mo, np, paired_real):
    from cypmode.challenge.multiplicity import holm as _holm

    _rows = paired_real
    if enzyme_l.value != "both enzymes":
        _rows = _rows[_rows["enzyme"] == enzyme_l.value]
    if cut_l.value != "both cuts":
        _rows = _rows[_rows["z0"] == float(cut_l.value.split("= ")[1])]
    _rows = _rows.reset_index(drop=True)
    _p = _holm(_rows["p"].tolist()) if holm_l.value else _rows["p"].tolist()
    _rows = _rows.assign(p_used=_p, passes=lambda f: (f["auc_gain"] >= floor_l.value) & (f["p_used"] < alpha_l.value), minus_log10_p=lambda f: -np.log10(f["p_used"].clip(lower=1e-6)))
    _chart = (alt.Chart(_rows).mark_circle(size=70, opacity=0.75).encode(x=alt.X("auc_gain:Q", title="gain in AUC over the base"), y=alt.Y("minus_log10_p:Q", title="-log10 of the p used"),
              color=alt.Color("passes:N", scale=alt.Scale(domain=[True, False], range=["#0e9f8a", "#9aa5b8"])), tooltip=["log", "comparison", "enzyme", "z0", "auc_gain", "p"]).properties(width=560, height=340)
              + alt.Chart({"values": [{"x": floor_l.value}]}).mark_rule(color="#c44", strokeDash=[5, 4]).encode(x="x:Q") + alt.Chart({"values": [{"y": float(-np.log10(alpha_l.value))}]}).mark_rule(color="#c44", strokeDash=[5, 4]).encode(y="y:Q"))
    if cut_l.value == "both cuts":
        _g = _rows.groupby(["log", "comparison"]).agg(cuts=("z0", "nunique"), passing=("passes", "sum")).reset_index()
        _win = _g[(_g["cuts"] >= 2) & (_g["passing"] == _g["cuts"])]
        _text, _table = f"{len(_win)} of {len(_g)} comparisons pass at every cut they were scored at.", _win[["log", "comparison"]]
    else:
        _win = _rows[_rows["passes"]]
        _text, _table = f"{len(_win)} of {len(_rows)} comparisons pass at this cut.", _win[["log", "comparison", "auc_gain", "p"]]
    sim_9_11 = mo.vstack([mo.md("Each point is one comparison of the project: an idea added to a base stack, scored on the same folds, tested with the corrected resampled t. The registered rule asks for a gain of at least the design floor and a corrected p below 0.05 at both cuts; "
                                "move the thresholds, or add the correction for the number of attempts (the project's Holm function), and see how many ideas would have passed."),
                          mo.hstack([enzyme_l, cut_l], justify="start"), mo.hstack([floor_l, alpha_l, holm_l], justify="start"), mo.md(_text), _chart, mo.ui.table(_table.reset_index(drop=True).to_dict("records"), selection=None, page_size=8)])
    return (sim_9_11,)


# -- 10.5 the physics block, live -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    smiles_ph = mo.ui.text(value="c1ccc(cc1)Cn1ccnc1", label="SMILES (up to 40 heavy atoms)", full_width=True)
    return (smiles_ph,)


@app.cell(hide_code=True)
def _(mo, molecule, smiles_ph):
    from simulators import physics

    _ok = molecule.parse(smiles_ph.value) is not None
    _out = physics.run(smiles_ph.value) if _ok else None
    if _out is None:
        sim_10_5 = mo.vstack([smiles_ph, mo.callout("This text is not a molecule that RDKit can read, or it has more than 40 heavy atoms.", kind="warn")])
    else:
        _head = {k: _out["features"].get(k) for k in physics.HEADLINE}
        sim_10_5 = mo.vstack([smiles_ph, mo.hstack([mo.Html(molecule.drawing(_out["molecule"], atom_colors=physics.charge_colors(_out["atoms"]), size=(380, 300))),
                                                   mo.md(f"**GFN2-xTB on a conformer ensemble** (the project's `compute_molecule`)\n\n- HOMO {(_head['phys_homo_ev'] or 0):.2f} eV, LUMO {(_head['phys_lumo_ev'] or 0):.2f} eV, gap **{(_head['phys_gap_ev'] or 0):.2f} eV**\n"
                                                         f"- dipole {(_head['phys_dipole_d'] or 0):.2f} debye\n- {len(_out['features'])} of the {_out['n_quantities']} quantities apply to this molecule (the others, such as those of a missing functional group, are undefined).\n\n"
                                                         "Atoms are coloured by partial charge: red positive, blue negative.")], align="start"),
                             mo.accordion({"Per atom: charge, Fukui functions, valence, exposed surface": mo.ui.table(physics.atom_rows(_out["atoms"]), selection=None, page_size=12),
                                           "Every quantity of the molecule": mo.ui.table(physics.feature_rows(_out["features"]), selection=None, page_size=15)})])
    return (sim_10_5,)



# -- the results of the finished project, from the public bundle ------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(here, pd):
    _root = here / "data/public"
    ledger = pd.read_csv(_root / "ledger.csv")
    excerpts = pd.read_csv(_root / "excerpts.csv")
    selectors = pd.read_csv(_root / "selectors.csv")
    poses_bench = pd.read_csv(_root / "poses.csv")
    rules_bench = pd.read_csv(_root / "rules.csv")
    return excerpts, ledger, poses_bench, rules_bench, selectors


# -- 0.5 the ledger in one table ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(ledger, mo):
    kind_filter = mo.ui.multiselect(options=sorted(ledger["kind"].unique()), value=sorted(ledger["kind"].unique()), label="Kind of idea")
    verdict_filter = mo.ui.radio(options=["all", "MET", "NOT MET"], value="all", label="Verdict", inline=True)
    log_pick = mo.ui.dropdown(options=sorted(ledger["log"].unique()), value=min(ledger["log"].unique()), label="Experiment (log)")
    return kind_filter, log_pick, verdict_filter


@app.cell(hide_code=True)
def _(excerpts, kind_filter, ledger, log_pick, mo, paired_real, verdict_filter):
    _met = int((ledger["verdict"] == "MET").sum())
    _rows = ledger[ledger["kind"].isin(kind_filter.value)]
    if verdict_filter.value != "all":
        _rows = _rows[_rows["verdict"] == verdict_filter.value]
    _verdicts = ledger[ledger["log"] == log_pick.value][["name", "verdict"]].reset_index(drop=True)
    _pairs = paired_real[paired_real["log"] == log_pick.value][["enzyme", "z0", "comparison", "auc_gain", "p", "better", "folds"]].reset_index(drop=True)
    _lines = excerpts[excerpts["log"] == log_pick.value]["text"].tolist()
    sim_0_5 = mo.vstack([
        mo.hstack([mo.stat(label="Registered gates", value=str(len(ledger))), mo.stat(label="MET", value=str(_met), caption="the idea joined the stack or the recipe"), mo.stat(label="NOT MET", value=str(len(ledger) - _met), caption="written down as a negative result")], justify="start", gap=2),
        mo.md("Every ruling of a gate was written down **before** its result existed. Read the verdicts, then open one experiment to see its paired comparisons and the lines of its log."),
        mo.hstack([kind_filter, verdict_filter], justify="start", gap=2, align="start"),
        mo.md(f"{len(_rows)} of {len(ledger)} verdicts."), mo.ui.table(_rows[["kind", "name", "verdict", "log"]].reset_index(drop=True).to_dict("records"), selection=None, page_size=10),
        mo.md("**Read one experiment**"), log_pick, mo.ui.table(_verdicts.to_dict("records"), selection=None), mo.ui.table(_pairs.to_dict("records"), selection=None), mo.md("**Lines of the log**\n\n```\n" + "\n".join(_lines) + "\n```"),
    ])
    return (sim_0_5,)


# -- 8.2 the selection rules, fold by fold -------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo, selectors):
    sel_enzyme = mo.ui.dropdown(options=sorted(selectors["enzyme"].unique()), value=min(selectors["enzyme"].unique()), label="Enzyme")
    sel_fold = mo.ui.dropdown(options=["all folds"] + [str(k) for k in sorted(selectors["outer"].unique())], value="all folds", label="Outer fold")
    sel_metric = mo.ui.radio(options=["outer_score", "outer_mcc", "optimism"], value="outer_score", label="Measure", inline=True)
    return sel_enzyme, sel_fold, sel_metric


@app.cell(hide_code=True)
def _(alt, mo, sel_enzyme, sel_fold, sel_metric, selectors):
    _rows = selectors[selectors["enzyme"] == sel_enzyme.value]
    if sel_fold.value != "all folds":
        _rows = _rows[_rows["outer"] == int(sel_fold.value)]
    _box = alt.Chart(_rows).mark_boxplot(extent="min-max", size=24).encode(y=alt.Y("selector:N", title=None), x=alt.X(f"{sel_metric.value}:Q", title=sel_metric.value))
    _points = alt.Chart(_rows).mark_circle(size=30, opacity=0.5, color="#0e9f8a").encode(y="selector:N", x=f"{sel_metric.value}:Q", tooltip=["selector", "repeat", "outer", sel_metric.value])
    _means = _rows.groupby("selector")[sel_metric.value].agg(["mean", "std", "count"]).round(4).reset_index()
    sim_8_2 = mo.vstack([
        mo.md("The nested cross-validation chooses a candidate inside every training part and scores it on the part it never saw. One point is one repeat and outer fold; the optimism is the inner score minus the outer score, the amount by which choosing inside the loop flatters a rule. "
              "The scores are the ones of the nested report (weighted to follow the test set's prior and its similarity to the training compounds), not those of the gates."),
        mo.hstack([sel_enzyme, sel_fold, sel_metric], justify="start", gap=2), (_box + _points).properties(width=560, height=40 * _rows["selector"].nunique() + 30), mo.ui.table(_means.to_dict("records"), selection=None)])
    return (sim_8_2,)


# -- 11.8 choosing among poses -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo, poses_bench):
    ligand_pose = mo.ui.dropdown(options=["all ligands"] + sorted(poses_bench["compound"].unique()), value="all ligands", label="Ligand")
    line_pose = mo.ui.slider(start=0.5, stop=5.0, step=0.1, value=2.0, label="A good pose is within (A)", show_value=True)
    return ligand_pose, line_pose


@app.cell(hide_code=True)
def _(alt, ligand_pose, line_pose, mo, poses_bench, rules_bench):
    _poses = poses_bench if ligand_pose.value == "all ligands" else poses_bench[poses_bench["compound"] == ligand_pose.value]
    _rules = rules_bench if ligand_pose.value == "all ligands" else rules_bench[rules_bench["compound"] == ligand_pose.value]
    _strip = alt.Chart(_poses).mark_circle(size=60, opacity=0.7).encode(x=alt.X("true_rmsd:Q", title="RMSD to the crystal pose (A)"), y=alt.Y("engine:N", title=None), color="compound:N", tooltip=["compound", "engine", "sample", "true_rmsd"]).properties(width=560, height=140)
    _rule = alt.Chart({"values": [{"x": line_pose.value}]}).mark_rule(color="#c44", strokeDash=[5, 4]).encode(x="x:Q")
    _share = _poses.assign(good=_poses["true_rmsd"] <= line_pose.value).groupby("engine")["good"].agg(["mean", "count"]).rename(columns={"mean": "share of poses within the line", "count": "poses"}).round(3).reset_index()
    _by_rule = _rules.groupby("rule")["rmsd"].agg(["mean", "median", "count"]).round(3).reset_index()
    sim_11_8 = mo.vstack([
        mo.md("Ten ligands with a crystal structure of CYP3A4 were predicted again and again, and each pose is measured by its RMSD to the crystal pose (lower is better; 2 A is the usual line for a good pose). A **rule** picks one pose per ligand to submit: "
              "the model's own first sample, the pose that agrees best with the others (minimum Bayes risk), and so on. These are the stored poses and picks; nothing is predicted again."),
        mo.hstack([ligand_pose, line_pose], justify="start", gap=2), _strip + _rule, mo.md("**Poses within the line, by engine**"), mo.ui.table(_share.to_dict("records"), selection=None), mo.md("**RMSD of the pose each rule submits**"), mo.ui.table(_by_rule.to_dict("records"), selection=None)])
    return (sim_11_8,)



# -- 8.7 the out-of-fold predictions ------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(here, pd):
    from simulators import oof as oof_sim

    _root = here / "data/public"
    oof_tables = {e: pd.read_csv(_root / f"oof_{e}.csv.gz") for e in ("CYP3A4", "CYP2D6")}
    oof_candidates = pd.read_csv(_root / "candidates.csv").fillna({"blocks": ""})
    oof_blocks_tbl = pd.read_csv(_root / "blocks.csv")
    return oof_blocks_tbl, oof_candidates, oof_sim, oof_tables


@app.cell(hide_code=True)
def _(mo, oof_blocks_tbl, oof_sim, oof_tables):
    oof_enzyme = mo.ui.dropdown(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Enzyme")
    oof_fold = mo.ui.dropdown(options=["all folds"] + [str(k) for k in range(oof_sim.folds_of(oof_tables["CYP3A4"]))], value="all folds", label="Outer fold")
    oof_blocks_pick = mo.ui.multiselect(options={f"{row.block}: {row.label}": row.block for row in oof_blocks_tbl.itertuples()}, label="Feature blocks (candidates that use all of the selected)")
    oof_top = mo.ui.slider(start=5, stop=60, step=5, value=20, label="Candidates shown", show_value=True)
    return oof_blocks_pick, oof_enzyme, oof_fold, oof_top


@app.cell(hide_code=True)
def _(alt, mo, oof_blocks_pick, oof_candidates, oof_enzyme, oof_fold, oof_sim, oof_tables, oof_top):
    oof_ranking = oof_sim.candidate_table(oof_tables, oof_candidates, oof_enzyme.value, oof_fold.value, tuple(oof_blocks_pick.value))
    if oof_ranking.empty:
        oof_view_ranking = mo.md("No candidate with an outer-fold prediction for every compound uses all of the selected feature blocks.")
    else:
        _shown = oof_ranking.head(int(oof_top.value))
        _dots = alt.Chart(_shown).mark_circle(size=70, color="#0e9f8a").encode(x=alt.X("mean:Q", title="mean AUC over the repeats", scale=alt.Scale(zero=False)), y=alt.Y("candidate:N", sort=None, title=None), tooltip=["candidate", "family", "mean", "low", "high"])
        _range = alt.Chart(_shown).mark_rule(color="#12203a").encode(x="low:Q", x2="high:Q", y=alt.Y("candidate:N", sort=None))
        oof_view_ranking = mo.vstack([mo.md(f"{len(oof_ranking)} candidates have an outer-fold prediction for every compound of {oof_enzyme.value}; the dot is the mean AUC over the repeats and the line runs from the lowest to the highest repeat."),
                                      (_dots + _range).properties(width=560, height=18 * len(_shown) + 30), mo.ui.table(_shown[["candidate", "family", "blocks", "mean", "low", "high"]].round(4).to_dict("records"), selection=None, page_size=8)])
    return oof_ranking, oof_view_ranking


@app.cell(hide_code=True)
def _(mo, oof_ranking):
    oof_cand = mo.ui.dropdown(options=oof_ranking["candidate"].tolist() or ["(none)"], value=(oof_ranking["candidate"].iloc[0] if len(oof_ranking) else "(none)"), label="Candidate")
    oof_repeat = mo.ui.dropdown(options=[0, 1, 2], value=0, label="Repeat")
    oof_share = mo.ui.slider(start=0.02, stop=0.8, step=0.01, value=0.3, label="Call the top share positive", show_value=True)
    return oof_cand, oof_repeat, oof_share


@app.cell(hide_code=True)
def _(alt, mo, np, oof_cand, oof_enzyme, oof_fold, oof_repeat, oof_share, oof_sim, oof_tables, pd):
    if oof_cand.value == "(none)":
        oof_view_candidate = mo.md("Pick feature blocks that some candidate uses.")
    else:
        _frame = oof_tables[oof_enzyme.value]
        _keep = np.ones(len(_frame), dtype=bool) if oof_fold.value == "all folds" else (_frame[f"fold|{oof_repeat.value}"] == int(oof_fold.value)).to_numpy()
        _y = _frame["y"].to_numpy()[_keep]
        _scores = _frame[f"{oof_cand.value}|{oof_repeat.value}"].to_numpy()[_keep]
        _fpr, _tpr = oof_sim.roc_points(_y, _scores)
        _at = oof_sim.confusion(_y, _scores, oof_share.value)
        _roc = alt.Chart(pd.DataFrame({"false positive rate": _fpr, "true positive rate": _tpr})).mark_line(color="#0e9f8a").encode(x="false positive rate:Q", y="true positive rate:Q")
        _diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="#9aa5b8", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
        _dot = alt.Chart(pd.DataFrame({"false positive rate": [_at["FP"] / (_at["FP"] + _at["TN"])], "true positive rate": [_at["TP"] / (_at["TP"] + _at["FN"])]})).mark_circle(size=120, color="#c44").encode(x="false positive rate:Q", y="true positive rate:Q")
        oof_view_candidate = mo.vstack([mo.hstack([oof_cand, oof_repeat, oof_share], justify="start"),
                                        mo.hstack([(_roc + _diag + _dot).properties(width=300, height=300), mo.vstack([mo.md(f"AUC **{oof_sim.auc_of(_y, _scores):.4f}** on {len(_y)} compounds, {int(_y.sum())} positive ({np.mean(_y):.1%})."), mo.ui.table([_at], selection=None), mo.md("The red point is the chosen call rate on the ROC curve.")])], align="start")])
    return (oof_view_candidate,)


@app.cell(hide_code=True)
def _(mo, oof_blocks_tbl, oof_candidates, oof_cand, oof_enzyme, oof_fold, oof_sim, oof_tables):
    if oof_cand.value == "(none)":
        oof_view_folds = mo.md("")
    else:
        _folds = oof_sim.fold_table(oof_tables, oof_enzyme.value, oof_cand.value)
        _blocks = oof_sim.block_summary(oof_tables, oof_candidates, oof_blocks_tbl, oof_enzyme.value, oof_fold.value)
        oof_view_folds = mo.vstack([mo.md(f"**{oof_cand.value} in every outer fold of {oof_enzyme.value}**: the AUC on the compounds that each fold scored, one column per repeat. The spread between folds is the noise that a corrected test has to allow for (section 6.5)."),
                                    mo.ui.table(_folds.to_dict("records"), selection=None),
                                    mo.md("**What the feature blocks show.** For each block, the mean AUC of the candidates that use it against those that do not. The difference is descriptive: it mixes the block with the family of the model and with the other blocks that the same candidates use, and it is not a gate."),
                                    mo.ui.table(_blocks.to_dict("records"), selection=None, page_size=10)])
    return (oof_view_folds,)


@app.cell(hide_code=True)
def _(mo, oof_blocks_pick, oof_enzyme, oof_fold, oof_top, oof_view_candidate, oof_view_folds, oof_view_ranking):
    sim_8_7 = mo.vstack([mo.md("Every candidate of the nested plan, scored on compounds it never saw: each compound of the training pool is predicted once per repeat by a model fitted without it. These are the stored predictions; nothing is refitted. "
                               "Choose an enzyme, an outer fold and the feature blocks, then a candidate and a share of compounds to call positive."),
                         mo.hstack([oof_enzyme, oof_fold, oof_top], justify="start", gap=2), oof_blocks_pick, oof_view_ranking, oof_view_candidate, oof_view_folds])
    return (sim_8_7,)



# -- the parts whose simulators are registered by section number (simulators/part_*.py) ------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    import importlib
    import pkgutil

    import simulators
    from simulators import framework

    parts = {name: importlib.import_module(f"simulators.{name}") for _, name, _ in sorted(pkgutil.iter_modules(simulators.__path__)) if name.startswith("part_")}
    ctl = mo.ui.dictionary({name: framework.controls(mo, module.SIMS) for name, module in parts.items()})
    return ctl, framework, parts


@app.cell(hide_code=True)
def _(ctl, framework, mo, parts):
    part_views = {}
    for _name, _module in parts.items():
        part_views.update(framework.views(mo, ctl[_name], _module.SIMS))
    return (part_views,)



# -- the notebook, part by part --------------------------------------------------------------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo, part_views, sections, sim_1_1, sim_1_2, sim_1_5, sim_1_6, sim_1_7, sim_1_8, sim_2_1, sim_2_3, sim_2_5, sim_2_9, sim_3_2, sim_3_3, sim_4_1, sim_4_10, sim_4_2, sim_4_3, sim_4_4, sim_5_1, sim_6_1, sim_6_3, sim_6_5, sim_6_7, sim_7_3, sim_8_4, sim_9_11, sim_9_3, sim_9_8, sim_10_5, sim_0_5, sim_8_2, sim_11_8, sim_8_7):
    live = {"1.1": sim_1_1, "1.2": sim_1_2, "1.5": sim_1_5, "1.6": sim_1_6, "1.7": sim_1_7, "1.8": sim_1_8, "2.1": sim_2_1, "2.3": sim_2_3, "2.5": sim_2_5, "2.9": sim_2_9, "3.2": sim_3_2, "3.3": sim_3_3,
            "4.1": sim_4_1, "4.2": sim_4_2, "4.3": sim_4_3, "4.4": sim_4_4, "4.10": sim_4_10, "5.1": sim_5_1, "6.1": sim_6_1, "6.3": sim_6_3, "6.5": sim_6_5, "6.7": sim_6_7, "7.3": sim_7_3,
            "8.4": sim_8_4, "9.3": sim_9_3, "9.8": sim_9_8, "9.11": sim_9_11, "10.5": sim_10_5, "0.5": sim_0_5, "8.2": sim_8_2, "11.8": sim_11_8, "8.7": sim_8_7, **part_views}
    _tabs = {}
    for _part in sections:
        _n_live = sum(1 for _s in _part["sections"] if _s["number"] in live)
        _items = []
        for _s in _part["sections"]:
            _field = f" · {_s['field']}" if _s["field"] else ""
            if _s["number"] in live:
                _items.append(mo.accordion({f"{_s['number']} {_s['title']} · LIVE{_field}": live[_s["number"]]}))
            else:
                _items.append(mo.md(f"{_s['number']} {_s['title']}{_field} (read it in the notebook, anchor `{_s['anchor']}`)"))
        _head = mo.md(f"**{_part['label']}. {_part['title']}**: {len(_part['sections'])} sections, {_n_live} with a simulator." if _part["sections"] else f"**{_part['label']}. {_part['title']}**")
        _tabs[_part["label"].replace("Part ", "")] = mo.vstack([_head, *_items])
    mo.vstack([mo.md(f"**{sum(len(p['sections']) for p in sections)} sections in {len(sections)} parts; {len(live)} have a simulator.**"), mo.ui.tabs(_tabs)])


if __name__ == "__main__":
    app.run()
