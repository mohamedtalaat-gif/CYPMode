# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part II, the data: the columns of the files (2.2), the rows that have a label and no direct arm (2.4), the block of compounds measured on every enzyme (2.6), how the test set differs from the training set (2.7),
the other public sources (2.8), the positive rate that three published numbers give (2.10) and the one that a classifier gives without any board (2.11)."""

import altair as alt
import numpy as np
import pandas as pd

from cypmode.challenge.external_activity import connectivity_key
from cypmode.challenge.leaderboard_prior import implied_prevalence
from cypmode.challenge.tdi_targets import ASSAY_FLOOR, SHIFT_THRESHOLD
from simulators import datasets, store


def c_2_2(mo):
    return {"file": mo.ui.radio(options=["training file", "blinded test file"], value="training file", label="File", inline=True), "word": mo.ui.text(value="", label="Words in the name of the column")}


def v_2_2(mo, v):
    columns = store.table("columns")
    shown = columns[(columns["file"] == v["file"]) & columns["column"].str.contains(v["word"], case=False, regex=False)].drop(columns="file")
    return mo.vstack([mo.md(f"The {v['file']} has **{int((columns['file'] == v['file']).sum())}** columns; **{len(shown)}** match. The blinded file carries the structures and the names and nothing else: it holds no potency and no label."), mo.ui.table(shown, selection=None, page_size=12)])


def c_2_4(mo):
    return {"isoform": mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Enzyme", inline=True), "d": mo.ui.slider(start=1.0, stop=6.0, step=0.1, value=4.0, label="the direct pIC50 that the missing arm is assumed to have", show_value=True)}


def v_2_4(mo, v):
    rows = store.table("no_direct")
    rows = rows[rows["isoform"] == v["isoform"]]
    both = store.table("label_points")
    both = both[both["isoform"] == v["isoform"]]
    if rows.empty:
        return mo.md(f"No {v['isoform']} compound of the file has a label and no direct arm.")
    called = datasets.positives_if_direct_is(rows["t"], v["d"])
    frame = pd.concat([rows.assign(group="label but no direct arm")[["t", "group"]], both.assign(group="both arms measured")[["t", "group"]]])
    chart = alt.Chart(frame).mark_bar(opacity=0.6).encode(x=alt.X("t:Q", bin=alt.Bin(maxbins=30), title="t, the TDI-arm pIC50"), y=alt.Y("count()", stack=None), color="group:N")
    rule = alt.Chart(pd.DataFrame({"t": [max(v["d"], ASSAY_FLOOR) + SHIFT_THRESHOLD]})).mark_rule(color="black", strokeDash=[4, 3]).encode(x="t:Q")
    return mo.vstack([mo.md(f"**{len(rows)}** {v['isoform']} compounds have a label and no direct arm, and **{int((rows['label'] == 1).sum())}** of them are positive. Were their direct potency {v['d']:.1f}, the written rule (a shift above log10 2 = {SHIFT_THRESHOLD:.3f} over max(d, {ASSAY_FLOOR:g})) "
                            f"would call **{called}** positive: the labels follow another convention, no direct measurement and therefore negative. The dashed line is the value of t above which the rule would call a compound positive."), (chart + rule).properties(width=560, height=260)])


def c_2_6(mo):
    return {"window": mo.ui.slider(start=50, stop=800, step=10, value=300, label="the window (compounds)", show_value=True), "share": mo.ui.slider(start=0.1, stop=0.9, step=0.05, value=0.3, label="the minimum share with a measurement on another enzyme", show_value=True)}


def v_2_6(mo, v):
    numbering = store.table("numbering")
    share, mask = datasets.block_of(numbering, int(v["window"]), float(v["share"]))
    other = numbering.drop(columns="Molecule_Name").notna().any(axis=1).to_numpy()
    inside, outside = (float(other[mask].mean()) if mask.any() else float("nan")), float(other[~mask].mean())
    chart = alt.Chart(share).mark_line().encode(x=alt.X("position:Q", title="compounds in the order of their numbers"), y=alt.Y("share:Q", title="share with a measurement on another enzyme")) + alt.Chart(pd.DataFrame({"y": [v["share"]]})).mark_rule(strokeDash=[4, 3], color="gray").encode(y="y:Q")
    band = alt.Chart(share[share["in the block"]]).mark_area(opacity=0.25, color="#c62828").encode(x="position:Q", y=alt.Y("share:Q"))
    return mo.vstack([mo.md(f"The finder takes the longest run of compounds where the rolling share stays above {v['share']:.2f}. For a window of {int(v['window'])} it finds a block of **{int(mask.sum())}** compounds: {inside:.0%} of them carry a measurement on another enzyme against {outside:.0%} outside the block. "
                            "A measurement exists only where an earlier screen promoted the compound, which is the selection that section 9.1 models."), (chart + band).properties(width=600, height=260)])


def c_2_7(mo):
    return {"threshold": mo.ui.slider(start=0.2, stop=1.0, step=0.01, value=0.6, label="a close relative has a similarity of at least", show_value=True)}


def v_2_7(mo, v):
    sim = store.table("similarity")
    share = sim.assign(close=sim["max similarity"] >= v["threshold"]).groupby("set")["close"].mean().rename("share with a close relative").reset_index()
    chart = alt.Chart(sim).mark_bar(opacity=0.55).encode(x=alt.X("max similarity:Q", bin=alt.Bin(maxbins=40)), y=alt.Y("count()", stack=None), color="set:N")
    rule = alt.Chart(pd.DataFrame({"x": [v["threshold"]]})).mark_rule(color="black", strokeDash=[4, 3]).encode(x="x:Q")
    return mo.vstack([mo.md("The highest Tanimoto similarity (Morgan fingerprints of radius 2) of a compound to the compounds of the other set. The test compounds sit closer to the training compounds than the training compounds sit to each other, "
                            "which is why the evaluation of the project weights its folds to resemble the test set (section 6.4)."), mo.ui.table(share, selection=None, pagination=False), (chart + rule).properties(width=600, height=260)])


def c_2_8(mo):
    return {"smiles": mo.ui.text(value="CC(C)Cc1ccc(cc1)C(C)C(=O)O", label="A compound of another source (SMILES)", full_width=True)}


def v_2_8(mo, v):
    keys = store.table("keys")
    key = connectivity_key(v["smiles"])
    where = sorted(keys[keys["key"] == key]["set"].unique()) if key else []
    verdict = "not a structure that RDKit can read" if not key else (f"**in the {' and '.join(where)} set of the challenge**: it would leak" if where else "in neither set of the challenge")
    return mo.vstack([mo.md(f"The connectivity key of the compound (the first block of the InChIKey, which ignores stereochemistry) is `{key}`: {verdict}. "
                            "The table gives, for each other public source, what it measures, how many compounds it holds and how many connectivity keys it shares with the challenge."), mo.ui.table(store.table("sources"), selection=None, pagination=False)])


def c_2_10(mo):
    return {"accuracy": mo.ui.slider(start=0.5, stop=1.0, step=0.001, value=0.806, label="accuracy A of a file", show_value=True), "precision": mo.ui.slider(start=0.05, stop=1.0, step=0.001, value=0.665, label="precision P", show_value=True),
            "recall": mo.ui.slider(start=0.05, stop=1.0, step=0.001, value=0.672, label="recall R", show_value=True), "isoform": mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Board", inline=True)}


def v_2_10(mo, v):
    p = float(implied_prevalence(v["accuracy"], v["precision"], v["recall"]))
    board = store.table("board_rows")
    board = board[board["isoform"] == v["isoform"]].assign(**{"positive rate": lambda f: implied_prevalence(f["Accuracy"], f["Precision"], f["Recall"])}).dropna()
    chart = alt.Chart(board).mark_bar().encode(x=alt.X("positive rate:Q", bin=alt.Bin(maxbins=30)), y="count()")
    return mo.vstack([mo.md(f"p = (1 - A) / (1 - 2R + R/P) = **{p:.4f}**. Every row of one board is scored on the same labels, so the {len(board)} rows of the {v['isoform']} board must all give the same p: "
                            f"their median is **{board['positive rate'].median():.4f}** and their standard deviation {board['positive rate'].std():.5f}."), chart.properties(width=520, height=200)])


def c_2_11(mo):
    return {"isoform": mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Enzyme", inline=True), "rate": mo.ui.slider(start=0.05, stop=0.6, step=0.01, value=0.3, label="the true positive rate of the new file", show_value=True),
            "n": mo.ui.slider(start=100, stop=2000, step=50, value=750, label="compounds in the new file", show_value=True)}


def v_2_11(mo, v):
    oof = store.table(f"oof_{v['isoform']}")
    column = next(c for c in oof.columns if c.endswith("|0") and not c.startswith("fold"))
    estimates, m0, m1 = datasets.estimate_positive_rate(oof[column], oof["y"], v["rate"], int(v["n"]), 300, seed=7)
    chart = alt.Chart(pd.DataFrame({"estimate": estimates})).mark_bar().encode(x=alt.X("estimate:Q", bin=alt.Bin(maxbins=30)), y="count()") + alt.Chart(pd.DataFrame({"x": [v["rate"]]})).mark_rule(color="#c62828").encode(x="x:Q")
    return mo.vstack([mo.md(f"The classifier is the stored out-of-fold score of `{column.split('|')[0]}`: its mean is {m1:.3f} on the positives and {m0:.3f} on the negatives. A new file of {int(v['n'])} compounds at a true rate of {v['rate']:.2f} (red) "
                            f"has a mean score that gives the rate through pi = (mean - m0) / (m1 - m0): over 300 files the estimates average **{np.mean(estimates):.3f}** with a standard deviation of **{np.std(estimates):.3f}**. The new files are resampled from the stored scores, so the estimate is optimistic."), chart.properties(width=560, height=220)])


SIMS = {"2.2": (c_2_2, v_2_2), "2.4": (c_2_4, v_2_4), "2.6": (c_2_6, v_2_6), "2.7": (c_2_7, v_2_7), "2.8": (c_2_8, v_2_8), "2.10": (c_2_10, v_2_10), "2.11": (c_2_11, v_2_11)}
