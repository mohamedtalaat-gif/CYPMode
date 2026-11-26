# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part VI, measuring without selection bias: a score under the prevalence of the test set (6.4), why Tukey's HSD was dropped (6.6), a hold-out chosen by similarity (6.8), the place of the stack on a second published data set (6.9) and the columns against the compounds (6.10)."""

import altair as alt
import numpy as np
import pandas as pd

from simulators import store, validation


def _candidates(*enzymes: str) -> list[str]:
    return store.candidate_names(*(enzymes or ("CYP3A4", "CYP2D6")))


def c_6_2(mo):
    return {"isoform": mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Enzyme", inline=True), "n": mo.ui.dropdown(options=[200, 300, 400], value=300, label="compounds"), "folds": mo.ui.dropdown(options=[3, 5, 10], value=5, label="folds of the cross-validation"), "seed": mo.ui.number(start=0, stop=99, step=1, value=0, label="seed")}


def v_6_2(mo, v):
    r = validation.selection_bias(v["isoform"], int(v["n"]), int(v["folds"]), int(v["seed"]))
    chart = alt.Chart(r["curve"]).mark_line(point=True).encode(x="k:Q", y=alt.Y("cross-validated AUC:Q", scale=alt.Scale(zero=False)))
    return mo.vstack([mo.md(f"Nearest-neighbour models with 1 to 30 neighbours, scored by {int(v['folds'])}-fold cross-validation on {int(v['n'])} compounds. The best, k = {r['best k']}, scores **{r['best cross-validated AUC']:.3f}** under the very cross-validation that chose it; with the choice made inside each outer training set "
                            f"(a nested cross-validation, 6.3) the same procedure scores **{r['nested AUC']:.3f}**. The gap is the optimism of choosing the best of many on the data that scores them, and it grows with the number of models tried."), chart.properties(width=520, height=240)])


def c_6_4(mo):
    return {"enzyme": mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP2D6", label="Enzyme", inline=True), "candidate": mo.ui.dropdown(options=_candidates(), value=_candidates()[0], label="Candidate (its stored out-of-fold scores)"),
            "prevalence": mo.ui.slider(start=0.03, stop=0.6, step=0.01, value=0.1, label="positive rate of the set it is scored on", show_value=True), "z0": mo.ui.slider(start=0.0, stop=3.0, step=0.1, value=1.0, label="keep labels with a confidence of at least z0", show_value=True)}


def v_6_4(mo, v):
    oof = store.table(f"oof_{v['enzyme']}")
    candidate = v["candidate"] if f"{v['candidate']}|0" in oof.columns else _candidates(v["enzyme"])[0]
    y, scores = validation.oof_scores(oof, candidate)
    rates = np.linspace(0.03, 0.6, 20)
    curve = pd.DataFrame([{"call rate": r, "MCC": validation.weighted_mcc(y, scores, r, p), "scored as": label} for r in rates for p, label in ((None, f"the training rate ({y.mean():.2f})"), (v["prevalence"], f"the rate you chose ({v['prevalence']:.2f})"))])
    points = store.table("label_points")
    points = points[points["isoform"] == v["enzyme"]]
    kept = float((points["z"].abs() >= v["z0"]).mean())
    return mo.vstack([mo.md(f"The same out-of-fold scores of `{candidate}`, with the compounds weighted as if the set had a positive rate of {v['prevalence']:.2f}: the best MCC moves and so does the call rate that gives it. "
                            f"A validation that reads the training rate would choose the wrong cut. Labels with a confidence of at least {v['z0']:.1f} are **{kept:.0%}** of the {len(points)} compounds that have both arms."), alt.Chart(curve).mark_line().encode(x="call rate:Q", y="MCC:Q", color="scored as:N").properties(width=560, height=260)])


def c_6_6(mo):
    return {"k": mo.ui.slider(start=2, stop=8, step=1, value=5, label="models of the same skill", show_value=True), "folds": mo.ui.slider(start=5, stop=15, step=1, value=10, label="folds", show_value=True),
            "rho": mo.ui.slider(start=0.0, stop=0.9, step=0.1, value=0.2, label="the part of the variance that all folds of a model share (they train on overlapping data)", show_value=True)}


def v_6_6(mo, v):
    grid = pd.DataFrame([{"rho": r, **validation.false_alarm_rates(int(v["k"]), int(v["folds"]), r, 80, 5)} for r in np.linspace(0.0, 0.9, 7)]).melt("rho", ["Tukey HSD", "corrected t with Holm"], "method", "share of experiments with a false difference")
    now = validation.false_alarm_rates(int(v["k"]), int(v["folds"]), float(v["rho"]), 150, 5)
    rule = alt.Chart(pd.DataFrame({"y": [0.05]})).mark_rule(strokeDash=[4, 3], color="gray").encode(y="y:Q")
    return mo.vstack([mo.md(f"{int(v['k'])} models with exactly the same skill, scored on {int(v['folds'])} folds: at a dependence of {v['rho']:.1f} Tukey's HSD declares some pair different in **{now['Tukey HSD']:.0%}** of the experiments and the corrected resampled t with Holm in **{now['corrected t with Holm']:.0%}** (a sound test gives 5%). "
                            "Tukey's HSD assumes independent scores; the scores of the folds of one model are not."), (alt.Chart(grid).mark_line(point=True).encode(x="rho:Q", y=alt.Y("share of experiments with a false difference:Q", scale=alt.Scale(domain=[0, 1])), color="method:N") + rule).properties(width=520, height=260)])


def c_6_8(mo):
    return {"threshold": mo.ui.slider(start=0.2, stop=0.9, step=0.01, value=0.55, label="hold out the training compounds with a relative at least this similar", show_value=True)}


def v_6_8(mo, v):
    sim = store.table("similarity")
    r = validation.holdout_by_similarity(sim, float(v["threshold"]))
    train = sim[sim["set"].str.startswith("training")]
    kept = train[train["max similarity"] >= v["threshold"]].assign(set="the hold-out")
    both = pd.concat([sim[sim["set"].str.startswith("test")].assign(set="the test compounds"), kept[["set", "max similarity"]]])
    keys = store.table("keys")
    shared = len(set(keys[keys["set"] == "training"]["key"]) & set(keys[keys["set"] == "test"]["key"]))
    chart = alt.Chart(both).mark_bar(opacity=0.55).encode(x=alt.X("max similarity:Q", bin=alt.Bin(maxbins=40)), y=alt.Y("count()", stack=None), color="set:N")
    return mo.vstack([mo.md(f"A hold-out of the training compounds that have a close relative resembles the test set when its distribution of similarities does. At {v['threshold']:.2f} it keeps **{r['share kept']:.0%}** of the training compounds ({r['kept']}); the distance between its distribution and that of the test compounds is **{r['KS distance to the test set']:.3f}** "
                            f"(0 would be identical; medians {r['median kept']:.2f} and {r['median test']:.2f}). Leakage check: **{shared}** connectivity keys are in both the training and the test file."), chart.properties(width=560, height=240)])


def c_6_9(mo):
    return {"enzyme": mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Enzyme", inline=True),
            "auprc": mo.ui.slider(start=0.5, stop=1.0, step=0.005, value=0.8, label="the test AUPRC of a model of your own", show_value=True)}


def v_6_9(mo, v):
    board = store.table("ext_leaderboard")
    board = board[board["isoform"] == v["enzyme"]]
    entries, ours = board[~board["ours"]], board[board["ours"]].iloc[0]

    def place(score: float) -> int:
        return 1 + int((entries["auprc"] > score).sum())

    bins = store.table("ext_similarity_bins")
    bins = bins[bins["isoform"] == v["enzyme"]]
    permuted = store.table("ext_permutation_control")
    permuted = permuted[permuted["isoform"] == v["enzyme"]]
    mine = float(v["auprc"])
    bars = alt.Chart(entries).mark_bar(color="#9aa5b1").encode(x=alt.X("auprc:Q", scale=alt.Scale(domain=[0.5, 1.0]), title="test AUPRC"), y=alt.Y("entry:N", sort="-x", title=None))
    rules = alt.Chart(pd.DataFrame({"auprc": [float(ours["auprc"]), mine], "mark": ["this stack", "your model"]})).mark_rule(strokeWidth=3).encode(x="auprc:Q", color=alt.Color("mark:N", legend=alt.Legend(title=None)))
    by_bin = alt.Chart(bins)
    curve = by_bin.mark_line(point=True).encode(x=alt.X("bin:N", sort=list(bins["bin"]), title="similarity to the training set"), y=alt.Y("auc:Q", scale=alt.Scale(domain=[0.5, 1.0])))
    band = by_bin.mark_rule().encode(x=alt.X("bin:N", sort=list(bins["bin"])), y="auc_low:Q", y2="auc_high:Q")
    return mo.vstack([mo.md(f"The Veith {v['enzyme']} inhibition set of the Therapeutics Data Commons has **{len(entries)}** published entries, with a median test AUPRC of **{entries['auprc'].median():.3f}** and a best of **{entries['auprc'].max():.3f}**. "
                            f"This stack scores a mean of **{ours['auprc']:.3f}** over five seeds, which would stand **{place(float(ours['auprc']))}** among the published entries. A model of your own that scores {mine:.3f} would stand **{place(mine)}** (1 is the best). "
                            f"With the labels permuted the test AUC is {permuted['auc'].mean():.3f} (mean of {len(permuted)} permutations; 0.5 is chance). The label of this set is direct inhibition, so a good place supports the features and the protocol, not the claim that time-dependent inhibition is predicted well."),
                      (bars + rules).properties(width=520, height=max(160, 16 * len(entries))),
                      mo.md("The AUC of the deployed stack by the similarity of a test compound to the training set, with its 95% interval:"), (curve + band).properties(width=520, height=200)])


def c_6_10(mo):
    return {"enzyme": mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Enzyme", inline=True),
            "cut": mo.ui.radio(options=["1.0", "1.96"], value="1.0", label="confidence cut z0 of the labels", inline=True),
            "view": mo.ui.dropdown(options=["the 23 candidates by width", "keeping a fraction of the columns", "adding columns of noise"], value="the 23 candidates by width", label="What to look at")}


def v_6_10(mo, v):
    column = "auc_z1" if v["cut"] == "1.0" else "auc_z196"
    spectrum = store.table("ratio_spectrum")
    spectrum = spectrum[spectrum["isoform"] == v["enzyme"]]
    rows = ", ".join(f"{r.feature_set}: {int(r.columns_read)} columns, rank {int(r['rank'])} of {int(r.rows)} compounds, {int(r.components_for_90)} components hold 90% of the variance" for _, r in spectrum.iterrows())
    intro = f"The columns of one learner against the compounds that it is fitted on ({v['enzyme']}; labels at z0 = {v['cut']}). The matrix of the learners: {rows}."
    if v["view"].startswith("the 23"):
        c = store.table("ratio_candidates")
        c = c[c["isoform"] == v["enzyme"]].assign(ratio=lambda d: d["n_columns"] / d["rows"], AUC=lambda d: d["auc_z1.0" if v["cut"] == "1.0" else "auc_z1.96"])
        rho = c["ratio"].corr(c["AUC"], method="spearman")
        chart = alt.Chart(c).mark_circle(size=70).encode(x=alt.X("ratio:Q", scale=alt.Scale(type="symlog"), title="columns per compound"), y=alt.Y("AUC:Q", scale=alt.Scale(zero=False), title="nested outer AUC"), color="family:N", tooltip=["candidate", "n_columns", "AUC"])
        body = [mo.md(f"The {len(c)} candidates of the final stack use between {int(c['n_columns'].min())} and {int(c['n_columns'].max())} columns for {int(c['rows'].iloc[0])} compounds, so between {c['ratio'].min():.2f} and {c['ratio'].max():.2f} columns per compound. "
                      f"The rank correlation between the ratio and the AUC is **{rho:+.2f}** across the seven families, which differ in width and in skill, so it describes the grid and does not test the effect of width alone."), chart.properties(width=520, height=260)]
    elif v["view"].startswith("keeping"):
        f = store.table("ratio_fraction")
        f = f[f["isoform"] == v["enzyme"]]
        chart = alt.Chart(f).mark_line(point=True).encode(x=alt.X("n_columns:Q", title="columns kept"), y=alt.Y(f"{column}:Q", scale=alt.Scale(zero=False), title="mean nested outer AUC"))
        low, high = f.iloc[0], f.iloc[-1]
        body = [mo.md(f"One boosted-tree learner on the widest block, fitted on a random fraction of its columns inside the same nested folds: from {int(low['n_columns'])} columns ({low['fraction']:.0%}) to {int(high['n_columns'])} ({high['fraction']:.0%}) "
                      f"the mean AUC goes from **{low[column]:.3f}** to **{high[column]:.3f}**, over {int(low['folds'])} outer folds."), chart.properties(width=520, height=240)]
    else:
        n = store.table("ratio_noise")
        n = n[n["isoform"] == v["enzyme"]].assign(label=lambda d: d["candidate"] + ", " + d["kind"] + " x" + d["multiple"].astype(str))
        chart = alt.Chart(n).mark_bar().encode(y=alt.Y("label:N", sort="-x", title=None), x=alt.X(f"{column}:Q", scale=alt.Scale(domain=[0.5, 1.0]), title="mean nested outer AUC"), color="kind:N")
        base = n[n["kind"] == "none"][column]
        body = [mo.md(f"The same learners with columns of noise added (a copy of the columns with each shuffled, or Gaussian noise), up to four times the width: the unchanged learners score {', '.join(f'{x:.3f}' for x in base)}; "
                      f"the worst case with noise scores **{n[column].min():.3f}**, over {int(n['folds'].iloc[0])} outer folds each."), chart.properties(width=520, height=max(160, 22 * len(n)))]
    return mo.vstack([mo.md(intro), *body])


SIMS = {"6.2": (c_6_2, v_6_2), "6.4": (c_6_4, v_6_4), "6.6": (c_6_6, v_6_6), "6.8": (c_6_8, v_6_8), "6.9": (c_6_9, v_6_9), "6.10": (c_6_10, v_6_10)}
