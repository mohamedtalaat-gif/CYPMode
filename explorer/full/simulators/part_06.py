# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part VI, measuring without selection bias: a score under the prevalence of the test set (6.4), why Tukey's HSD was dropped (6.6) and a hold-out chosen by similarity (6.8)."""

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


SIMS = {"6.2": (c_6_2, v_6_2), "6.4": (c_6_4, v_6_4), "6.6": (c_6_6, v_6_6), "6.8": (c_6_8, v_6_8)}
