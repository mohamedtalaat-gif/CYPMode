# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part III, molecules as numbers: the folding of a fingerprint (3.1), the blocks of the feature store (3.4 and 3.7), a three-point pharmacophore key (3.5) and chemical series (3.6)."""

import altair as alt
import pandas as pd

from simulators import chem, store

DEFAULT = "c1ccc(cc1)Cn1ccnc1CCN"


def c_3_1(mo):
    return {"smiles": mo.ui.text(value=DEFAULT, label="SMILES", full_width=True), "radius": mo.ui.slider(start=0, stop=4, step=1, value=2, label="radius of the environments", show_value=True),
            "bits": mo.ui.dropdown(options=[64, 128, 256, 512, 1024, 2048, 4096], value=256, label="length of the folded fingerprint")}


def v_3_1(mo, v):
    now = chem.morgan_folding(v["smiles"], int(v["radius"]), int(v["bits"]))
    curve = pd.DataFrame([{"bits": b, **chem.morgan_folding(v["smiles"], int(v["radius"]), b)} for b in (64, 128, 256, 512, 1024, 2048, 4096)])
    chart = alt.Chart(curve.melt("bits", ["environments", "bits set"], "count of", "number")).mark_line(point=True).encode(x=alt.X("bits:O"), y="number:Q", color="count of:N")
    return mo.vstack([mo.md(f"A fingerprint of radius {int(v['radius'])} has **{now['environments']}** different environments here; folded into {int(v['bits'])} bits they set **{now['bits set']}** bits, so **{now['environments that share a bit']}** environments share a bit with another one. "
                            "Folding is why two different substructures can look the same to a model; a longer fingerprint folds less."), chart.properties(width=520, height=220)])


def c_3_4(mo):
    return {"smiles": mo.ui.text(value=DEFAULT, label="SMILES", full_width=True)}


def v_3_4(mo, v):
    widths = chem.block_widths(v["smiles"])
    chart = alt.Chart(widths.melt("block", ["width", "non-zero for this molecule"], "what", "columns")).mark_bar().encode(y=alt.Y("block:N", sort=None), x=alt.X("columns:Q", scale=alt.Scale(type="symlog")), color="what:N", yOffset="what:N")
    return mo.vstack([mo.md("Each block is a different way to describe the same molecule: counts of environments, pairs of atoms, torsions, keys, a graph of pharmacophore types, 217 descriptors, the flags of the motif and the alerts. The width is the number of columns the block adds to a table; most of it is zero for any one molecule."),
                      mo.ui.table(widths, selection=None, pagination=False), chart.properties(width=560, height=320)])


def c_3_5(mo):
    return {"smiles": mo.ui.text(value=DEFAULT, label="SMILES", full_width=True), "bin": mo.ui.slider(start=0.5, stop=4.0, step=0.5, value=1.0, label="width of a distance bin (angstrom)", show_value=True)}


def v_3_5(mo, v):
    now = chem.pharmacophore_keys(v["smiles"], float(v["bin"]))
    if not now["triples"]:
        return mo.md("The molecule has fewer than three pharmacophore features, or no conformer could be embedded.")
    curve = pd.DataFrame([{"bin (angstrom)": b, "distinct keys": chem.pharmacophore_keys(v["smiles"], b)["distinct keys"]} for b in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)])
    return mo.vstack([mo.md(f"One conformer (seed 42) has the features {now['features']}: **{now['triples']}** triples of them, **{now['distinct keys']}** distinct keys with distance bins of {v['bin']} angstrom. "
                            "This is a teaching version of the idea of the project's three-point fingerprint: the features come from RDKit's definitions and the bins are chosen here, not those of the pharmacophore library the project uses."),
                      alt.Chart(curve).mark_line(point=True).encode(x="bin (angstrom):Q", y="distinct keys:Q").properties(width=420, height=200)])


def c_3_6(mo):
    return {"n": mo.ui.slider(start=100, stop=1500, step=100, value=500, label="compounds (the first of the file)", show_value=True), "similarity": mo.ui.slider(start=0.3, stop=0.9, step=0.05, value=0.6, label="Tanimoto similarity that puts two compounds in one cluster", show_value=True)}


def v_3_6(mo, v):
    now = chem.series(int(v["n"]), float(v["similarity"]))
    curve = pd.DataFrame([{"similarity": s, "clusters": chem.series(int(v["n"]), s)["clusters"]} for s in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)])
    return mo.vstack([mo.md(f"The Butina algorithm groups the compounds so that each cluster has a centre and every member is at least {v['similarity']:.2f} similar to it. Here: **{now['clusters']}** clusters, **{now['singletons']}** of them single compounds, the largest with **{now['largest cluster']}**; "
                            f"the compounds have **{now['distinct scaffolds']}** distinct Murcko scaffolds. A series is made by changing one part of a molecule at a time, so a random split of the data puts relatives in training and test, which is the reason for the splits of Part VI."),
                      alt.Chart(curve).mark_line(point=True).encode(x="similarity:Q", y="clusters:Q").properties(width=420, height=200)])


def c_3_7(mo):
    names = list(chem.CHEAP_BLOCKS)
    return {"smiles": mo.ui.text(value=DEFAULT, label="Width measured on this molecule (SMILES)", full_width=True), "blocks": mo.ui.multiselect(options=names, value=["morgan_c2", "rdkit_desc", "motif"], label="Blocks")}


def v_3_7(mo, v):
    widths = chem.block_widths(v["smiles"]).merge(store.table("blocks")[["block", "label", "candidates"]], on="block", how="left")
    chosen = widths[widths["block"].isin(v["blocks"])]
    return mo.vstack([mo.md(f"The feature store computes every block once for every compound and the candidates of the stack pick from them. The blocks you chose add **{int(chosen['width'].sum()):,}** columns; "
                            "`candidates` is the number of candidates of the final stack that use the block."), mo.ui.table(widths, selection=None, pagination=False)])


SIMS = {"3.1": (c_3_1, v_3_1), "3.4": (c_3_4, v_3_4), "3.5": (c_3_5, v_3_5), "3.6": (c_3_6, v_3_6), "3.7": (c_3_7, v_3_7)}
