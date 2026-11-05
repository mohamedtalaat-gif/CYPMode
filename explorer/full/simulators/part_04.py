# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part IV, machine learning from zero: nearest neighbours (4.5), the arms with noise and the floor (4.6), a small network (4.7), how far a message reaches in a graph (4.8) and a principal-component analysis (4.9)."""

import warnings

import altair as alt
import numpy as np
import pandas as pd

from simulators import chem, molecule, store


def c_4_5(mo):
    return {"isoform": mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Enzyme", inline=True), "row": mo.ui.slider(start=0, stop=399, step=1, value=5, label="the compound to score (of the first 400 labelled)", show_value=True),
            "k": mo.ui.slider(start=1, stop=30, step=1, value=5, label="k, the number of neighbours", show_value=True), "weighted": mo.ui.checkbox(value=True, label="weight the votes by the similarity")}


def v_4_5(mo, v):
    found, estimate = chem.nearest(v["isoform"], int(v["row"]), int(v["k"]), bool(v["weighted"]))
    table = chem.labelled(v["isoform"]).head(800).reset_index(drop=True)
    own = table.loc[int(v["row"]), "label"]
    auc = chem.knn_leave_one_out_auc(v["isoform"], int(v["k"]), bool(v["weighted"]))
    return mo.vstack([mo.md(f"The {int(v['k'])} most similar labelled compounds vote; the estimate is **{estimate:.3f}** and the compound's own label is **{int(own)}**. Over the first 400 labelled compounds, each scored by the others, the estimate has an AUC of **{auc:.3f}**."),
                      mo.ui.table(found, selection=None, pagination=False)])


def c_4_6(mo):
    return {"sd_t": mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.3, label="error of the TDI-arm prediction (pIC50 units)", show_value=True), "sd_d": mo.ui.slider(start=0.0, stop=1.5, step=0.05, value=0.6, label="error of the direct-arm prediction", show_value=True),
            "isoform": mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Enzyme", inline=True)}


def v_4_6(mo, v):
    points = store.table("label_points")
    points = points[points["isoform"] == v["isoform"]].dropna(subset=["label"])
    grid = np.linspace(0.0, 1.5, 16)
    curve = pd.DataFrame([{"error of the direct arm": s, "labels recovered": chem.noisy_rule_accuracy(points, s, v["sd_t"], clip, 3), "direct arm floored at 4": clip} for s in grid for clip in (True, False)])
    clipped, plain = (chem.noisy_rule_accuracy(points, v["sd_d"], v["sd_t"], clip, 3) for clip in (True, False))
    return mo.vstack([mo.md(f"The label depends on t - max(d, 4). With predicted arms that carry these errors the rule gives back **{clipped:.1%}** of the labels when the direct arm is floored at 4 and **{plain:.1%}** when it is not: below the floor the direct arm does not matter, so its error is harmless there. "
                            "That is the reason for floor-clipping the arm before it is predicted."), alt.Chart(curve).mark_line(point=True).encode(x="error of the direct arm:Q", y=alt.Y("labels recovered:Q", scale=alt.Scale(zero=False)), color="direct arm floored at 4:N").properties(width=520, height=240)])


def c_4_7(mo):
    return {"hidden": mo.ui.slider(start=1, stop=32, step=1, value=8, label="neurons in the hidden layer", show_value=True), "activation": mo.ui.radio(options=["relu", "tanh"], value="tanh", label="activation", inline=True),
            "noise": mo.ui.slider(start=0.05, stop=0.5, step=0.05, value=0.2, label="noise of the two moons", show_value=True), "epochs": mo.ui.slider(start=10, stop=600, step=10, value=300, label="passes over the data", show_value=True)}


def v_4_7(mo, v):
    from sklearn.datasets import make_moons
    from sklearn.neural_network import MLPClassifier

    x, y = make_moons(300, noise=float(v["noise"]), random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = MLPClassifier((int(v["hidden"]),), activation=v["activation"], max_iter=int(v["epochs"]), random_state=0).fit(x, y)
    g = np.linspace(-2, 3, 60), np.linspace(-1.5, 2, 60)
    xx, yy = np.meshgrid(*g)
    surface = pd.DataFrame({"x": xx.ravel(), "y": yy.ravel(), "p": net.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]})
    field = alt.Chart(surface).mark_rect().encode(x=alt.X("x:Q", bin=alt.Bin(maxbins=60), axis=None), y=alt.Y("y:Q", bin=alt.Bin(maxbins=60), axis=None), color=alt.Color("mean(p):Q", scale=alt.Scale(scheme="redblue", domain=[0, 1])))
    dots = alt.Chart(pd.DataFrame({"x": x[:, 0], "y": x[:, 1], "class": y.astype(str)})).mark_circle(size=40, stroke="black", strokeWidth=0.6).encode(x="x:Q", y="y:Q", fill="class:N")
    loss = alt.Chart(pd.DataFrame({"pass": range(len(net.loss_curve_)), "loss": net.loss_curve_})).mark_line().encode(x="pass:Q", y="loss:Q")
    return mo.vstack([mo.md(f"A network with one hidden layer of {int(v['hidden'])} neurons ({v['activation']}) trained by backpropagation on 300 points: it classifies **{net.score(x, y):.1%}** of them correctly after {len(net.loss_curve_)} passes. "
                            "One neuron can only draw a straight line; more neurons bend it. The loss falls as the weights are corrected pass after pass."), mo.hstack([(field + dots).properties(width=300, height=250), loss.properties(width=260, height=250)])])


def c_4_8(mo):
    return {"smiles": mo.ui.text(value="CC(C)Cc1ccc(cc1)C(C)C(=O)O", label="SMILES", full_width=True), "atom": mo.ui.slider(start=0, stop=29, step=1, value=0, label="the atom whose state is followed", show_value=True), "steps": mo.ui.slider(start=0, stop=6, step=1, value=2, label="message-passing steps T", show_value=True)}


def v_4_8(mo, v):
    mol = molecule.parse(v["smiles"])
    if mol is None:
        return mo.md("That is not a SMILES that RDKit can read.")
    atom = min(int(v["atom"]), mol.GetNumAtoms() - 1)
    reach = chem.within_bonds(v["smiles"], atom, int(v["steps"]))
    sizes = pd.DataFrame([{"steps": t, "atoms taken in": len(chem.within_bonds(v["smiles"], atom, t))} for t in range(7)])
    return mo.vstack([mo.Html(molecule.drawing(mol, reach, indices=True)), mo.md(f"After T = {int(v['steps'])} steps the state of atom {atom} has taken in **{len(reach)}** of the {mol.GetNumAtoms()} atoms: exactly those within {int(v['steps'])} bonds (highlighted). "
                                                                                 "The network learns what to do with the messages; how far they reach is fixed by the graph."), mo.ui.table(sizes, selection=None, pagination=False)])


def c_4_9(mo):
    return {"isoform": mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Enzyme", inline=True), "n": mo.ui.dropdown(options=[200, 300, 400], value=300, label="compounds"), "components": mo.ui.slider(start=1, stop=40, step=1, value=10, label="components kept", show_value=True)}


def v_4_9(mo, v):
    scores, ratio, labels = chem.pca_of_descriptors(v["isoform"], int(v["n"]))
    kept = float(ratio[:int(v["components"])].sum())
    scree = pd.DataFrame({"component": np.arange(1, 41), "cumulative variance": np.cumsum(ratio)[:40]})
    plot = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1], "label": np.where(labels == 1, "positive", "negative")})
    return mo.vstack([mo.md(f"The {scores.shape[1]} informative descriptors of {int(v['n'])} compounds are summarised by components in the order of the variance they carry: the first {int(v['components'])} keep **{kept:.1%}** of it. The first component alone carries {ratio[0]:.1%}."),
                      alt.Chart(scree).mark_line(point=True).encode(x="component:Q", y=alt.Y("cumulative variance:Q", scale=alt.Scale(domain=[0, 1]))).properties(width=420, height=200),
                      alt.Chart(plot).mark_circle(opacity=0.6, size=40).encode(x="PC1:Q", y="PC2:Q", color="label:N").properties(width=420, height=300)])


SIMS = {"4.5": (c_4_5, v_4_5), "4.6": (c_4_6, v_4_6), "4.7": (c_4_7, v_4_7), "4.8": (c_4_8, v_4_8), "4.9": (c_4_9, v_4_9)}
