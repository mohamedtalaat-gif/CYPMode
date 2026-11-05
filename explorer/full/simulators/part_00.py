# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part 0, orientation: the interaction of two drugs in numbers (0.1), the map of what was explored (0.2), how much a score moves when a board scores half of a set (0.3), the jobs of the system (0.4), the files
that the notebook explains (0.6) and the claims with the sections that explain them (0.7)."""

import altair as alt
import numpy as np
import pandas as pd

from simulators import ddi, store
from simulators import statistics as stats


def c_0_1(mo):
    return {"fm": mo.ui.slider(start=0.05, stop=0.99, step=0.01, value=0.8, label="fm, the share of the clearance of the drug that this enzyme carries", show_value=True),
            "strength": mo.ui.slider(start=-1.0, stop=2.0, step=0.05, value=0.7, label="log10 of I/Ki, how strongly the inhibitor holds the enzyme", show_value=True)}


def v_0_1(mo, v):
    ratio_i = 10.0 ** v["strength"]
    r = ddi.aucr(v["fm"], ratio_i)
    grid = np.linspace(-1.0, 2.0, 121)
    lines = pd.concat([pd.DataFrame({"log10 I/Ki": grid, "AUC ratio": [ddi.aucr(fm, 10.0 ** x) for x in grid], "fm": label}) for fm, label in ((0.3, "fm 0.3"), (0.6, "fm 0.6"), (0.9, "fm 0.9"), (v["fm"], "your fm"))])
    chart = alt.Chart(lines).mark_line().encode(x="log10 I/Ki:Q", y=alt.Y("AUC ratio:Q", scale=alt.Scale(type="log")), color="fm:N")
    here = alt.Chart(pd.DataFrame({"log10 I/Ki": [v["strength"]], "AUC ratio": [r]})).mark_point(size=120, color="black", filled=True).encode(x="log10 I/Ki:Q", y="AUC ratio:Q")
    bands = alt.Chart(pd.DataFrame({"AUC ratio": [ddi.WEAK, ddi.MODERATE, ddi.STRONG]})).mark_rule(strokeDash=[4, 3], color="gray").encode(y="AUC ratio:Q")
    return mo.vstack([mo.md(f"With fm = {v['fm']:.2f} and I/Ki = {ratio_i:.2f} the exposure to the victim drug is multiplied by **{r:.2f}**: a **{ddi.classify(r)}** inhibitor (the dashed lines are the ratios 1.25, 2 and 5 that name a weak, a moderate and a strong one). "
                                   "A drug that this enzyme clears almost alone (fm near 1) is the one an inhibitor can raise most: the ratio cannot exceed 1/(1 - fm)."), (chart + bands + here).properties(width=560, height=320)])


def c_0_2(mo):
    kinds = ["(all)", *sorted(store.table("ledger")["kind"].unique())]
    return {"kind": mo.ui.dropdown(options=kinds, value="(all)", label="Kind of idea")}


def v_0_2(mo, v):
    ledger = store.table("ledger")
    by_kind = ledger.groupby("kind").agg(families=("stem", "nunique"), verdicts=("name", "size"), met=("verdict", lambda s: int((s == "MET").sum()))).assign(**{"not met": lambda f: f["verdicts"] - f["met"]}).reset_index()
    chart = alt.Chart(by_kind.melt("kind", ["met", "not met"], "verdict", "count")).mark_bar().encode(y=alt.Y("kind:N", sort="-x"), x="count:Q", color=alt.Color("verdict:N", scale=alt.Scale(domain=["met", "not met"], range=["#2e7d32", "#9e9e9e"])))
    chosen = ledger if v["kind"] == "(all)" else ledger[ledger["kind"] == v["kind"]]
    families = chosen.groupby("stem").agg(verdicts=("name", "size"), met=("verdict", lambda s: int((s == "MET").sum()))).reset_index()
    return mo.vstack([mo.md(f"The ledger holds **{len(ledger)}** pre-registered verdicts in **{ledger['stem'].nunique()}** families of ideas; **{int((ledger['verdict'] == 'MET').sum())}** of them are MET. Choose a kind to list its families."),
                      chart.properties(width=520, height=170), mo.ui.table(by_kind, selection=None, pagination=False), mo.md(f"**{len(families)} families** for {v['kind']}:"), mo.ui.table(families, selection=None, page_size=10)])


def c_0_3(mo):
    return {"separation": mo.ui.slider(start=0.2, stop=2.5, step=0.05, value=0.9, label="separation of the scores of the two classes", show_value=True),
            "prevalence": mo.ui.slider(start=0.05, stop=0.5, step=0.01, value=0.3, label="positive rate of the scored set", show_value=True),
            "threshold": mo.ui.slider(start=-1.0, stop=3.0, step=0.05, value=0.8, label="the threshold of the calls", show_value=True)}


def v_0_3(mo, v):
    frames = []
    for n, label in ((375, "half of the test set (the live board)"), (750, "the whole test set (the final ranking)")):
        mcc = [stats.metrics_at(*stats.draw_scores(v["prevalence"], v["separation"], n, 5000 + s), v["threshold"])["MCC"] for s in range(300)]
        frames.append(pd.DataFrame({"MCC": mcc, "scored": label}))
    both = pd.concat(frames)
    sd = both.groupby("scored")["MCC"].std()
    chart = alt.Chart(both).mark_bar(opacity=0.55).encode(x=alt.X("MCC:Q", bin=alt.Bin(maxbins=30)), y=alt.Y("count()", stack=None), color="scored:N")
    return mo.vstack([mo.md("The same model scored on 300 different samples of each size. The live board of the challenge scores a part of the test set and the final ranking the whole of it, so the number on the board is "
                            f"a sample of a wider spread: its standard deviation here is **{sd.iloc[0]:.3f}** against **{sd.iloc[1]:.3f}** for the whole set."), chart.properties(width=600, height=260)])


def c_0_4(mo):
    return {"words": mo.ui.text(value="", label="Words in the description of the job"), "first": mo.ui.number(start=0, stop=999, step=1, value=0, label="from job number"), "last": mo.ui.number(start=0, stop=999, step=1, value=999, label="to job number")}


def v_0_4(mo, v):
    jobs = store.table("jobs")
    shown = jobs[(jobs["number"] >= v["first"]) & (jobs["number"] <= v["last"]) & jobs["title"].str.contains(v["words"], case=False, regex=False, na=False)]
    return mo.vstack([mo.md(f"The system is run by a queue of **{len(jobs)}** jobs, each a short script that calls the modules and says in its first comment line what it does. **{len(shown)}** match."), mo.ui.table(shown, selection=None, page_size=12)])


def c_0_6(mo):
    return {"query": mo.ui.text(value="queue", label="A file name, a module or a word of its path")}


def v_0_6(mo, v):
    cover = store.table("coverage")
    shown = cover[cover["path"].str.contains(v["query"], case=False, regex=False, na=False)]
    return mo.vstack([mo.md(f"The builder of the notebook fails when a file of the repository is named in no section. **{len(cover)}** files are covered; **{len(shown)}** match, with the sections that explain each."), mo.ui.table(shown, selection=None, page_size=12)])


def c_0_7(mo):
    claims = list(dict.fromkeys(store.table("claims")["claim"]))
    return {"claim": mo.ui.dropdown(options=claims, value=claims[0], label="A claim of the innovation summary")}


def v_0_7(mo, v):
    claims = store.table("claims")
    rows = claims[claims["claim"] == v["claim"]]
    return mo.vstack([mo.md(f"**{v['claim']}**"), mo.md("The notebook explains it, from first principles, and computes its number in:"), mo.ui.table(rows[["section", "title"]], selection=None, pagination=False),
                      mo.md(f"All {len(claims['claim'].unique())} claims, each with its sections, are in the table at the end of section 0.7.")])


SIMS = {"0.1": (c_0_1, v_0_1), "0.2": (c_0_2, v_0_2), "0.3": (c_0_3, v_0_3), "0.4": (c_0_4, v_0_4), "0.6": (c_0_6, v_0_6), "0.7": (c_0_7, v_0_7)}
