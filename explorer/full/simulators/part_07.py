# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part VII, the first TDI models and what they taught: the baseline and the staircase of blocks on a small sample (7.1, 7.2), a shortcut (7.4), the decision threshold (7.5) and the experiments of the ledger behind 7.6 to 7.11."""

import altair as alt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from simulators import chem, replay, store, validation


def _matrix(isoform: str, n: int, stage: int):
    from rdkit import Chem

    from cypmode.challenge import feature_store as fs
    from cypmode.metrics.amines import has_trialkylamine

    desc, labels = chem._descriptor_matrix(isoform, n)
    table = chem.labelled(isoform).head(n)
    mols = [Chem.MolFromSmiles(s) for s in table["SMILES"]]
    blocks = [desc]
    if stage >= 1:
        blocks.append(np.asarray(fs.BLOCKS["motif"](mols), dtype=float))
    if stage >= 2:
        blocks.append(np.array([[float(has_trialkylamine(s))] for s in table["SMILES"]]))
    if stage >= 3:
        blocks.append(np.asarray(fs.BLOCKS["morgan_bits"](mols), dtype=float))
    return np.hstack(blocks), labels.astype(int)


def c_7_1(mo):
    return {"run": mo.ui.run_button(label="Fit and score"), "isoform": mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Enzyme", inline=True), "n": mo.ui.dropdown(options=[200, 300, 400], value=300, label="compounds"), "trees": mo.ui.slider(start=20, stop=200, step=20, value=60, label="trees", show_value=True)}


def v_7_1(mo, v):
    if not v["run"]:
        return mo.md("Press the button: a gradient-boosted model on the RDKit descriptors of a few hundred training compounds, scored by five-fold cross-validation, the way the baseline of the organisers is built.")
    matrix, labels = _matrix(v["isoform"], int(v["n"]), 0)
    r = validation.cross_validated(matrix, labels, int(v["trees"]))
    return mo.md(f"{v['isoform']}, {int(v['n'])} compounds, {matrix.shape[1]} descriptors, {int(v['trees'])} trees: AUC **{r['AUC']:.3f}**, MCC at the cut 0.5 **{r['MCC']:.3f}**. A model this small is a reminder that the score of a baseline moves with the sample: compare with 0.3. The organisers' baseline gave MCC 0.264 on CYP3A4 and 0.055 on CYP2D6 on the whole test set.")


def c_7_2(mo):
    return {"run": mo.ui.run_button(label="Fit the four stages"), "isoform": mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Enzyme", inline=True), "n": mo.ui.dropdown(options=[200, 300, 400], value=300, label="compounds")}


def v_7_2(mo, v):
    if not v["run"]:
        return mo.md("Press the button: the staircase of 7.2 on a sample, one block of features added at a time (descriptors; the motif flags; the amine alert; a Morgan fingerprint), five-fold cross-validation at every step.")
    names = ["M0: descriptors", "M1: + motif flags", "M2: + amine alert", "M3: + Morgan fingerprint"]
    rows = []
    for stage, name in enumerate(names):
        matrix, labels = _matrix(v["isoform"], int(v["n"]), stage)
        rows.append({"stage": name, **validation.cross_validated(matrix, labels, 60, seed=1)})
    table = pd.DataFrame(rows)
    return mo.vstack([mo.md("The steps are small beside the spread of a score on this few compounds, which is what the notebook found on the whole file: the staircase shows what was tried, not a ranking."), mo.ui.table(table, selection=None, pagination=False),
                      alt.Chart(table.melt("stage", ["AUC", "MCC"])).mark_bar().encode(x="stage:N", y="value:Q", color="variable:N", xOffset="variable:N").properties(width=520, height=240)])


def c_7_4(mo):
    return {"isoform": mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Enzyme", inline=True)}


def v_7_4(mo, v):
    points = store.table("label_points")
    points = points[points["isoform"] == v["isoform"]].dropna(subset=["label"])
    rows = [{"feature": name, "AUC for the label": roc_auc_score(points["label"].astype(int), points[col])} for name, col in (("d, the direct pIC50", "d"), ("t, the TDI-arm pIC50", "t"), ("the shift s = t - max(d, 4)", "shift"))]
    return mo.vstack([mo.md("The direct pIC50 d is known for the training compounds and is a part of the label rule, so a model given d (or a model of d) looks excellent on the training data and has nothing to offer on the blinded compounds, which have no measurement at all. "
                            "A gain that comes from a quantity that the test set will not have is a shortcut. The AUC that each measurement gives for the label of the compounds that have both arms:"), mo.ui.table(pd.DataFrame(rows), selection=None, pagination=False)])


def c_7_5(mo):
    return {"isoform": mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Enzyme", inline=True), "candidate": mo.ui.dropdown(options=store.candidate_names("CYP3A4", "CYP2D6"), value=store.candidate_names("CYP3A4", "CYP2D6")[0], label="Candidate")}


def v_7_5(mo, v):
    oof = store.table(f"oof_{v['isoform']}")
    candidate = v["candidate"] if f"{v['candidate']}|0" in oof.columns else next(c.split("|")[0] for c in oof.columns if c.endswith("|0") and not c.startswith("fold"))
    y, s = validation.oof_scores(oof, candidate)
    rates = np.linspace(0.02, 0.7, 35)
    curve = pd.DataFrame({"call rate": rates, "MCC": [validation.weighted_mcc(y, s, r) for r in rates]})
    best = curve.loc[curve["MCC"].idxmax()]
    return mo.vstack([mo.md(f"Calling the top share of the compounds: the MCC of `{candidate}` is highest at a call rate of **{best['call rate']:.2f}** ({best['MCC']:.3f}); the positive rate of the data is {y.mean():.2f}. The cut that a model of probabilities suggests (0.5) is rarely the best one for MCC when the classes are unbalanced."),
                      (alt.Chart(curve).mark_line().encode(x="call rate:Q", y="MCC:Q") + alt.Chart(pd.DataFrame({"x": [float(y.mean())]})).mark_rule(strokeDash=[4, 3], color="gray").encode(x="x:Q")).properties(width=520, height=240)])


def _replayed(families, intro):
    return (replay.controls, lambda mo, v: replay.page(mo, v, families, intro))


SIMS = {"7.1": (c_7_1, v_7_1), "7.2": (c_7_2, v_7_2), "7.4": (c_7_4, v_7_4), "7.5": (c_7_5, v_7_5),
        "7.6": _replayed(["committee_label_model"], "Section 7.6 tested a committee of models as a measure of how sure a prediction is, against the real mistakes of the model."),
        "7.7": _replayed(["aux_ext_adme", "aux_ext_cont", "aux_ext_tox", "aux_pxr", "aux_gate_shift"], "Section 7.7 gave the models outside data as an auxiliary signal: other enzymes, other assays, other endpoints."),
        "7.8": _replayed(["physics_experiment", "physics_compact", "physics_tierb", "population_physics"], "Section 7.8 added quantum-chemical features and three-point pharmacophores."),
        "7.9": _replayed(["embeddings", "aux_mlp", "aux_mlp_aux", "monroe_embeddings"], "Section 7.9 tried learned representations: embeddings of pretrained networks and a shared-trunk network."),
        "7.10": _replayed(["aux_pocket_cyp3a4", "aux_pocket_cyp2d6", "aux_site_cyp3a4", "aux_site_cyp2d6", "pose_awareness"], "Section 7.10 asked whether a predicted structure helps a classifier: features of the pocket and of the site of a pose."),
        "7.11": _replayed(["nonneg_stack", "softmax_selector"], "Section 7.11 assembled the first models and tested the architecture before building the final stack.")}
