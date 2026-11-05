# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part IX, the additions that were tested: the selection-aware model in miniature (9.1), pairs of near neighbours (9.2), the weights that make training resemble the test set (9.5), the redundancy of a candidate (9.12), the functional groups that go with a label (9.15) and the experiments of the ledger behind the other sections."""

import altair as alt
import numpy as np
import pandas as pd

from simulators import additions, molecule, replay, store


def c_9_1(mo):
    return {"seed": mo.ui.number(start=0, stop=99, step=1, value=0, label="seed of the made-up population")}


def v_9_1(mo, v):
    grid = pd.DataFrame([{"non-hit share of the scored set": w, **additions.selection_auc(w, int(v["seed"]))} for w in np.linspace(0.0, 0.9, 10)])
    chart = alt.Chart(grid.melt("non-hit share of the scored set", ["fitted on the hits alone", "multiplied by the gate P(hit)"], "model", "AUC")).mark_line(point=True).encode(x="non-hit share of the scored set:Q", y=alt.Y("AUC:Q", scale=alt.Scale(zero=False)), color="model:N")
    return mo.vstack([mo.md("A made-up population in which a label exists only for the compounds that an earlier screen promoted. A model fitted on the labelled compounds answers a question about that selected population; when the set it is scored on holds compounds the screen would never have promoted, "
                            "multiplying by a gate P(hit) can help, and when it holds none it cannot. This is the structure of the CYP2D6 problem in miniature; the real result, with its dependence on the unknown share of non-hits, is in the ledger below."),
                      chart.properties(width=540, height=260), replay.page(mo, {"enzyme": "CYP2D6", "alpha": 0.05}, ["selection_aware_v2", "selection_validation"], "The experiments of the project on the selection-aware model:")])


def c_9_2(mo):
    return {"isoform": mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Enzyme", inline=True), "threshold": mo.ui.slider(start=0.4, stop=0.95, step=0.01, value=0.7, label="two compounds are a pair when they are at least this similar", show_value=True)}


def v_9_2(mo, v):
    pairs, summary = additions.similar_pairs(v["isoform"], float(v["threshold"]))
    if not summary["pairs"]:
        return mo.md("No pair of the first 800 labelled compounds is that similar.")
    different = pairs[pairs["labels differ"]].sort_values("similarity", ascending=False)
    shown = different.iloc[0] if len(different) else None
    body = [mo.md(f"Among the first 800 labelled compounds there are **{summary['pairs']}** pairs at least {v['threshold']:.2f} similar; in **{summary['pairs with different labels']}** of them ({summary['pairs with different labels'] / summary['pairs']:.0%}) the two labels differ. "
                  "A pair that differs by one change and in its label is where structure-activity knowledge lives; a pair model learns the effect of the change instead of the label of each compound. The most similar pair with different labels:")]
    if shown is not None:
        body += [mo.hstack([mo.Html(molecule.drawing(molecule.parse(shown["SMILES 1"]), (), size=(300, 240))), mo.Html(molecule.drawing(molecule.parse(shown["SMILES 2"]), (), size=(300, 240)))]), mo.md(f"{shown['compound']} and {shown['its neighbour']}, similarity {shown['similarity']:.2f}.")]
    body.append(replay.page(mo, {"enzyme": "both", "alpha": 0.05}, ["pair_transfer", "pair_transfer_physics", "mmp_transfer", "series_consensus"], "The experiments of the project on pairs and series:"))
    return mo.vstack(body)


def c_9_5(mo):
    return {"c": mo.ui.slider(start=-3.0, stop=1.0, step=0.5, value=-1.0, label="log10 of C, the freedom of the classifier of test against training", show_value=True)}


def v_9_5(mo, v):
    r = additions.test_likeness_weights(10.0 ** float(v["c"]))
    weights = pd.DataFrame({"weight": r["weights"]})
    return mo.vstack([mo.md(f"A classifier that tells the test compounds from the training compounds by their structures alone has an AUC of **{r['AUC test against training']:.3f}** (0.5 would be no difference). Each training compound gets the weight p / (1 - p) that the classifier implies; "
                            f"the effective sample size of the weighted training set is **{r['effective sample size']:.0f}** of {r['training compounds']}: a few compounds that look like the test set carry the weight, which is the price of resembling it."),
                      alt.Chart(weights).mark_bar().encode(x=alt.X("weight:Q", bin=alt.Bin(maxbins=40), scale=alt.Scale(type="symlog")), y="count()").properties(width=520, height=220),
                      replay.page(mo, {"enzyme": "both", "alpha": 0.05}, ["test_likeness"], "The experiment of the project on test-likeness weights:")])


def c_9_12(mo):
    names = store.candidate_names("CYP3A4", "CYP2D6")
    return {"candidate": mo.ui.dropdown(options=names, value=names[0], label="Candidate"), "isoform": mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Enzyme", inline=True)}


def v_9_12(mo, v):
    oof = store.table(f"oof_{v['isoform']}")
    candidate = v["candidate"] if f"{v['candidate']}|0" in oof.columns else next(c.split("|")[0] for c in oof.columns if c.endswith("|0") and not c.startswith("fold"))
    r = additions.redundancy(oof, candidate)
    gain = r["AUC of the others with it"] - r["AUC of the others"]
    return mo.vstack([mo.md(f"`{candidate}` ranks the compounds like the mean of the other candidates (rank correlation **{r['rank correlation with the others']:.2f}**). Alone its AUC is {r['AUC alone']:.3f}; the others without it {r['AUC of the others']:.3f}; with it {r['AUC of the others with it']:.3f}, a change of **{gain:+.4f}**. "
                            "An idea that adds nothing to a stack is usually one that the stack already holds in other words.")])


def c_9_15(mo):
    return {"isoform": mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Enzyme", inline=True), "min_count": mo.ui.slider(start=5, stop=100, step=5, value=20, label="a group must be in at least this many compounds", show_value=True)}


def v_9_15(mo, v):
    table = additions.group_enrichment(v["isoform"], 1500, int(v["min_count"]))
    return mo.vstack([mo.md(f"The functional groups of RDKit against the {v['isoform']} label of the first 1500 labelled compounds. The p values are corrected for the {len(table)} groups that were looked at (Holm): a group that looks enriched but does not survive the correction is what the ledger of the project calls a near miss."), mo.ui.table(table.round(4), selection=None, page_size=12)])


def _r(families, intro):
    return (replay.controls, lambda mo, v: replay.page(mo, v, families, intro))


SIMS = {"9.1": (c_9_1, v_9_1), "9.2": (c_9_2, v_9_2), "9.4": _r(["aux_chemprop", "aux_chemprop_foundation", "aux_tabpfn_chemeleon", "aux_tabicl_chemeleon", "embeddings"], "Section 9.4 tried a frozen foundation model, CheMeleon: its embeddings as features, with tabular learners on them, and a network that starts from its weights."),
        "9.5": (c_9_5, v_9_5), "9.6": _r(["arms_experiment"], "Section 9.6 asked whether a model should spend no capacity below the floor of the assay: the arms are clipped there before they are predicted."),
        "9.7": _r(["xiso_transfer"], "Section 9.7 tried to carry the arms of one enzyme to another."), "9.9": _r(["aux_neural", "aux_mlp", "aux_mlp_aux", "aux_chemprop"], "Section 9.9 gave the stack a neural pair: a multitask network and plain Chemprop."),
        "9.10": _r(["aux_tabpfn", "aux_tabicl", "aux_tabicl_monroe", "monroe_embeddings", "aux_atomjepa", "aux_atomjepa_head", "aux_basic_centre", "aux_hypothesis"], "Section 9.10 tested the other additions in the same way: tabular foundation models, embeddings, a readout of a pretrained network and the basic centre."),
        "9.12": (c_9_12, v_9_12), "9.13": _r(["aux_pocket_cyp3a4", "aux_pocket_cyp2d6", "aux_site_cyp3a4", "aux_site_cyp2d6"], "Section 9.13 put the compound in the pocket: features of the pocket and of the site of a predicted pose."),
        "9.14": _r(["pose_pharmacophore_model", "aux_hypothesis"], "Section 9.14 built pharmacophore models: sites from the pocket, hypotheses from the ligands."), "9.15": (c_9_15, v_9_15)}
