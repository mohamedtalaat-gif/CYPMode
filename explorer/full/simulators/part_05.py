# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part V, the Type II motif screen: the amine alert (5.2), a flag against the panel labels (5.3), the rank test on potencies (5.4), the iron-nitrogen distance (5.5), the burial of the ligand (5.6), the screen against the time-dependent shift (5.7) and the application (5.8)."""

import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

from cypmode.metrics.motifs import detect_motifs
from cypmode.validation.constants import COORDINATION_DISTANCE_CUTOFF_A
from simulators import molecule, motifs, store

PRESETS = {"triethylamine (a trialkylamine)": "CCN(CC)CC", "verapamil (a drug that inactivates CYP3A4 through its amine)": "COc1ccc(CCN(C)CCCC(C#N)(C(C)C)c2ccc(OC)c(OC)c2)cc1OC",
           "N,N-dimethylacetamide (an amide: excluded)": "CC(=O)N(C)C", "N,N-dimethylaniline (an aryl amine: excluded)": "CN(C)c1ccccc1", "a sulfonamide (excluded)": "CS(=O)(=O)N(C)C", "an enamine (excluded)": "CN(C)C=CC"}


def c_5_2(mo):
    return {"preset": mo.ui.dropdown(options={"(type one below)": "", **PRESETS}, value="(type one below)", label="A molecule"), "smiles": mo.ui.text(value="CCN(CC)CC", label="SMILES", full_width=True)}


def v_5_2(mo, v):
    smiles = v["preset"] or v["smiles"]
    mol = molecule.parse(smiles)
    if mol is None:
        return mo.md("That is not a SMILES that RDKit can read.")
    found = motifs.amine_alert(smiles)
    kind = f"The nitrogen of a {found['pattern'].replace('_', ' ')} is highlighted." if found["alert"] else "No nitrogen passes the pattern: an amide, an aryl, a sulfonamide or an enamine nitrogen is excluded, because its lone pair is not free."
    return mo.vstack([mo.Html(molecule.drawing(mol, found["atoms"], indices=False)), mo.md(f"**Alert: {'yes' if found['alert'] else 'no'}.** {kind} The Type II (iron-binding) motifs of the same molecule: {sorted(detect_motifs(smiles)) or 'none'}.")])


def c_5_3(mo):
    names = sorted({n for s in store.table("panel_groups")["motifs"].dropna() for n in s.split(";") if n})
    return {"isoform": mo.ui.dropdown(options=sorted(store.table("panel_groups")["isoform"].unique()), value="CYP3A4", label="Enzyme of the panel"), "counted": mo.ui.multiselect(options=names, value=names, label="Motifs that make the flag")}


def v_5_3(mo, v):
    groups = store.table("panel_groups")
    groups = groups[groups["isoform"] == v["isoform"]].fillna({"motifs": ""})
    counted = set(v["counted"])
    flag = groups["motifs"].map(lambda s: bool(counted & set(s.split(";"))))
    t = motifs.two_by_two(np.repeat(flag.to_numpy(), groups["n"]), np.repeat(groups["inhibitor"].to_numpy() == 1, groups["n"]))
    return mo.vstack([mo.md(f"Of the {int(groups['n'].sum()):,} {v['isoform']} compounds of the panel, **{t['flagged and positive'] + t['flagged and negative']:,}** carry a flagged motif and **{t['precision']:.1%}** of them are inhibitors, against {t['not flagged and positive'] / max(t['not flagged and positive'] + t['not flagged and negative'], 1):.1%} of the others. "
                            f"Odds ratio **{t['odds ratio']:.2f}**, one-sided Fisher p **{t['p (Fisher, one-sided)']:.2g}**; the flag finds {t['recall']:.1%} of the inhibitors."), mo.ui.table(pd.DataFrame([t]).T.reset_index().rename(columns={"index": "quantity", 0: "value"}), selection=None, pagination=False)])


def c_5_4(mo):
    return {"floor": mo.ui.slider(start=3.0, stop=8.0, step=0.1, value=3.0, label="only compounds with a pIC50 of at least", show_value=True)}


def v_5_4(mo, v):
    data = store.table("octant")
    data = data[data["pIC50"] >= v["floor"]]
    pos, neg = data[data["motif"]]["pIC50"].to_numpy(), data[~data["motif"]]["pIC50"].to_numpy()
    if len(pos) < 3 or len(neg) < 3:
        return mo.md("Too few compounds in one of the groups.")
    r = motifs.rank_test(pos, neg)
    ecdf = pd.concat([pd.DataFrame({"pIC50": np.sort(x), "share": np.arange(1, len(x) + 1) / len(x), "group": name}) for x, name in ((pos, "motif"), (neg, "no motif"))])
    return mo.vstack([mo.md(f"{len(pos)} compounds with a Type II motif and {len(neg)} without, on a screen that the rule never saw. Median pIC50 **{r['median of the first group']:.2f}** against **{r['median of the second group']:.2f}**; the chance that a motif compound is the more potent of a random pair is **{r['probability that a value of the first group is higher']:.3f}** "
                            f"(0.5 would be no difference); one-sided Mann-Whitney p = **{r['p']:.2g}**."), alt.Chart(ecdf).mark_line(interpolate="step-after").encode(x="pIC50:Q", y=alt.Y("share:Q", title="share of the group at or below"), color="group:N").properties(width=520, height=260)])


def c_5_5(mo):
    return {"cutoff": mo.ui.slider(start=1.5, stop=6.0, step=0.05, value=float(COORDINATION_DISTANCE_CUTOFF_A), label="distance that counts as coordination (angstrom)", show_value=True)}


def v_5_5(mo, v):
    table = store.table("fe_n").fillna({"motif": ""})
    rows = []
    for name, group in table.groupby("compound"):
        matched = group[group["motif"] != ""]
        closest = matched["distance"].min() if len(matched) else float("nan")
        rows.append({"compound": name, "closest motif nitrogen (A)": closest, "closest nitrogen of any kind (A)": group["distance"].min(), "coordinated": bool(closest <= v["cutoff"])})
    summary = pd.DataFrame(rows)
    chart = alt.Chart(summary.melt("compound", ["closest motif nitrogen (A)", "closest nitrogen of any kind (A)"], "which", "distance")).mark_bar().encode(x="compound:N", y="distance:Q", color="which:N", xOffset="which:N") + alt.Chart(pd.DataFrame({"y": [v["cutoff"]]})).mark_rule(color="#c62828", strokeDash=[4, 3]).encode(y="y:Q")
    return mo.vstack([mo.md("A replay of the four Boltz-2 predictions of the compounds with a cryo-EM structure (the model is not run here). The call is about the nitrogen that the motif flags, not the nearest nitrogen of any kind: for vardenafil the two differ."),
                      mo.ui.table(summary, selection=None, pagination=False), chart.properties(width=520, height=260), mo.accordion({"Every nitrogen of every ligand": mo.ui.table(table, selection=None, page_size=10)})])


def c_5_6(mo):
    return {"compound": mo.ui.dropdown(options=sorted(store.table("pocket")["compound"]), value="ketoconazole", label="Compound")}


def v_5_6(mo, v):
    pocket = store.table("pocket")
    row = pocket[pocket["compound"] == v["compound"]].iloc[0]
    chart = alt.Chart(pocket).mark_bar().encode(x="compound:N", y=alt.Y("burial_fraction:Q", scale=alt.Scale(domain=[0, 1])), color=alt.condition(alt.datum.compound == v["compound"], alt.value("#c62828"), alt.value("#90a4ae")))
    return mo.vstack([mo.md(f"A replay of the burial measured on the predicted structure: the surface of {v['compound']} is {row['unbound_sasa_A2']:.0f} A^2 alone and {row['bound_sasa_A2']:.0f} A^2 in the complex, so **{1 - row['bound_sasa_A2'] / row['unbound_sasa_A2']:.1%}** of it is covered by the protein. "
                            "A buried ligand is held in the pocket, a more exposed one is held less certainly."), chart.properties(width=420, height=220)])


def c_5_7(mo):
    names = sorted({n for s in store.table("flags")["motifs"].dropna() for n in s.split(";") if n})
    return {"isoform": mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Time-dependent label of", inline=True), "counted": mo.ui.multiselect(options=names, value=names, label="Type II motifs that make the flag"), "amine": mo.ui.checkbox(value=False, label="also flag a trialkylamine")}


def v_5_7(mo, v):
    flags = store.table("flags")
    label = flags[f"{v['isoform']}_is_TDI"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    keep = label.notna()
    flag = flags["motifs"].fillna("").map(lambda s: bool(set(v["counted"]) & set(s.split(";"))))
    if v["amine"]:
        flag = flag | flags["amine"].astype(bool)
    t = motifs.two_by_two(flag[keep].to_numpy(), label[keep].to_numpy() == 1)
    return mo.vstack([mo.md(f"On the {int(keep.sum()):,} training compounds that have a {v['isoform']} label, the flag marks {t['flagged and positive'] + t['flagged and negative']:,}; **{t['precision']:.1%}** of them are time-dependent inhibitors against "
                            f"{t['not flagged and positive'] / max(t['not flagged and positive'] + t['not flagged and negative'], 1):.1%} of the rest (odds ratio {t['odds ratio']:.2f}, one-sided p {t['p (Fisher, one-sided)']:.2g}). A rule that finds strong reversible binders need not find time-dependent ones: compare with 5.3."),
                      mo.ui.table(pd.DataFrame([t]).T.reset_index().rename(columns={"index": "quantity", 0: "value"}), selection=None, pagination=False)])


def c_5_8(mo):
    return {"smiles": mo.ui.text(value="Cc1ncc(n1CCO)[N+](=O)[O-]", label="A molecule to screen as the application does", full_width=True)}


def v_5_8(mo, v):
    source = Path(__file__).resolve().parents[3] / "app.py"
    tabs = re.findall(r'"([^"]+)"', re.search(r"st\.tabs\(\s*\[(.*?)\]", source.read_text(), re.DOTALL).group(1)) if source.exists() and "st.tabs" in source.read_text() else []
    mol = molecule.parse(v["smiles"])
    found = sorted(detect_motifs(v["smiles"])) if mol is not None else []
    return mo.vstack([mo.md(f"The application of the project (`app.py`, run with Streamlit) has the tabs **{', '.join(tabs) or '(not found)'}**. Its first tab is this screen: the Type II motifs of the molecule are {found or 'none'}, and the trialkylamine alert is **{motifs.amine_alert(v['smiles'])['alert']}**."),
                      mo.Html(molecule.drawing(mol, (), indices=False)) if mol is not None else mo.md("That is not a SMILES that RDKit can read.")])


SIMS = {"5.2": (c_5_2, v_5_2), "5.3": (c_5_3, v_5_3), "5.4": (c_5_4, v_5_4), "5.5": (c_5_5, v_5_5), "5.6": (c_5_6, v_5_6), "5.7": (c_5_7, v_5_7), "5.8": (c_5_8, v_5_8)}
