# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="CYPMode explorer")


@app.cell(hide_code=True)
def _():
    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    return alt, mo, np, pd


@app.cell(hide_code=True)
def _(mo, pd):
    # Everything the explorer shows comes from five small tables and a manifest in ./public (built by `python -m cypmode.explorer.data`); nothing is fitted or computed again here.
    _root = mo.notebook_location() / "public"

    def _read(name):
        return pd.read_csv(str(_root / f"{name}.csv"))

    ledger = _read("ledger")
    paired = _read("paired")
    selectors = _read("selectors")
    poses = _read("poses")
    rules = _read("rules")
    return ledger, paired, poses, rules, selectors


@app.cell(hide_code=True)
def _(ledger, mo, paired, poses, selectors):
    mo.md(
        f"""
        # CYPMode explorer

        Time-dependent inhibition of CYP3A4 and CYP2D6, and the structure of CYP3A4 with a ligand: the results of the finished project, to look at and to question. Every ruling of a gate
        was written down **before** its result existed; this page lets you read the verdicts, move the thresholds to see what a different rule would have let through, and compare the
        selection rules and the poses. It only reads: {len(ledger)} verdicts, {len(paired)} paired comparisons, {len(selectors)} rows of nested cross-validation and
        {len(poses)} benchmark poses, all from the public part of the repository. Moving a slider here never changes a registered verdict.
        """
    )


@app.cell(hide_code=True)
def _(ledger, mo):
    _met = int((ledger["verdict"] == "MET").sum())
    _total = len(ledger)
    mo.hstack(
        [
            mo.stat(label="Registered gates", value=str(_total)),
            mo.stat(label="MET", value=str(_met), caption="the idea joined the stack or the recipe"),
            mo.stat(label="NOT MET", value=str(_total - _met), caption="written down as a negative result"),
        ],
        justify="start",
        gap=2,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. The ledger of gates
    """)


@app.cell(hide_code=True)
def _(ledger, mo):
    kind_filter = mo.ui.multiselect(options=sorted(ledger["kind"].unique()), value=sorted(ledger["kind"].unique()), label="Kind of idea")
    verdict_filter = mo.ui.radio(options=["all", "MET", "NOT MET"], value="all", label="Verdict", inline=True)
    mo.hstack([kind_filter, verdict_filter], justify="start", gap=2, align="start")
    return kind_filter, verdict_filter


@app.cell(hide_code=True)
def _(kind_filter, ledger, mo, verdict_filter):
    _rows = ledger[ledger["kind"].isin(kind_filter.value)]
    if verdict_filter.value != "all":
        _rows = _rows[_rows["verdict"] == verdict_filter.value]
    mo.vstack(
        [
            mo.md(f"{len(_rows)} of {len(ledger)} verdicts. The log of a verdict holds its numbers: the paired difference, the corrected p and the rule that was registered."),
            mo.ui.table(_rows[["kind", "name", "verdict", "log"]].reset_index(drop=True), selection=None, page_size=12),
        ]
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. What a different rule would have let through

    Each point is one paired comparison of the project: an idea added to a base stack, scored on the same folds, with the corrected resampled t. The **registered** rule asked for a gain of
    at least the design floor and a corrected p below 0.05 **at both** confidence cuts. The sliders move those two thresholds; the registered values are the defaults.
    """)


@app.cell(hide_code=True)
def _(mo, paired):
    enzyme = mo.ui.dropdown(options=["both enzymes", "CYP3A4", "CYP2D6"], value="both enzymes", label="Enzyme")
    cut = mo.ui.dropdown(options=["both cuts"] + [f"z0 = {z}" for z in sorted(paired["z0"].unique())], value="both cuts", label="Confidence cut")
    floor = mo.ui.slider(start=0.0, stop=0.03, step=0.0005, value=0.0099, label="Gain at least", show_value=True)
    alpha = mo.ui.slider(start=0.001, stop=0.25, step=0.001, value=0.05, label="Corrected p below", show_value=True)
    mo.vstack([mo.hstack([enzyme, cut], justify="start", gap=2), mo.hstack([floor, alpha], justify="start", gap=2)])
    return alpha, cut, enzyme, floor


@app.cell(hide_code=True)
def _(alpha, alt, cut, enzyme, floor, mo, np, paired):
    _rows = paired
    if enzyme.value != "both enzymes":
        _rows = _rows[_rows["enzyme"] == enzyme.value]
    if cut.value != "both cuts":
        _rows = _rows[_rows["z0"] == float(cut.value.split("= ")[1])]
    _rows = _rows.assign(passes=(_rows["auc_gain"] >= floor.value) & (_rows["p"] < alpha.value), minus_log10_p=-np.log10(_rows["p"].clip(lower=1e-6)))
    _chart = (
        alt.Chart(_rows)
        .mark_circle(size=70, opacity=0.75)
        .encode(
            x=alt.X("auc_gain:Q", title="gain in AUC over the base"),
            y=alt.Y("minus_log10_p:Q", title="-log10 of the corrected p"),
            color=alt.Color("passes:N", title="passes", scale=alt.Scale(domain=[True, False], range=["#0e9f8a", "#9aa5b8"])),
            tooltip=["log", "comparison", "enzyme", "z0", "auc_gain", "p", "better", "folds"],
        )
        .properties(width=620, height=380)
    )
    _lines = alt.Chart({"values": [{"x": floor.value}]}).mark_rule(color="#c44", strokeDash=[5, 4]).encode(x="x:Q") + alt.Chart({"values": [{"y": float(-np.log10(alpha.value))}]}).mark_rule(
        color="#c44", strokeDash=[5, 4]
    ).encode(y="y:Q")
    if cut.value == "both cuts":
        _grouped = _rows.groupby(["log", "comparison"]).agg(cuts=("z0", "nunique"), passing=("passes", "sum")).reset_index()
        _winners = _grouped[(_grouped["cuts"] >= 2) & (_grouped["passing"] == _grouped["cuts"])]
        _text = f"{len(_winners)} comparison(s) pass at every cut they were scored at."
        _table = _winners[["log", "comparison"]].reset_index(drop=True)
    else:
        _winners = _rows[_rows["passes"]]
        _text = f"{len(_winners)} comparison(s) pass at this cut."
        _table = _winners[["log", "comparison", "auc_gain", "p"]].reset_index(drop=True)
    mo.vstack([mo.md(_text), _chart + _lines, mo.ui.table(_table, selection=None, page_size=8)])


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. The selection rules, fold by fold

    The nested cross-validation chooses a candidate inside every training part and scores it on the part it never saw. One point is one repeat and outer fold; the optimism is the
    inner score minus the outer score, the amount by which choosing inside the loop flatters a rule. The scores are the ones of the nested report, which weights the compounds to
    follow the test set's prior and its similarity to the training compounds; they are not the scores of the gates, which use the two confidence cuts.
    """)


@app.cell(hide_code=True)
def _(mo, selectors):
    sel_enzyme = mo.ui.dropdown(options=sorted(selectors["enzyme"].unique()), value=min(selectors["enzyme"].unique()), label="Enzyme")
    sel_metric = mo.ui.radio(options=["outer_score", "outer_mcc", "optimism"], value="outer_score", label="Measure", inline=True)
    mo.hstack([sel_enzyme, sel_metric], justify="start", gap=2)
    return sel_enzyme, sel_metric


@app.cell(hide_code=True)
def _(alt, mo, sel_enzyme, sel_metric, selectors):
    _rows = selectors[selectors["enzyme"] == sel_enzyme.value]
    _box = alt.Chart(_rows).mark_boxplot(extent="min-max", size=24).encode(y=alt.Y("selector:N", title=None), x=alt.X(f"{sel_metric.value}:Q", title=sel_metric.value))
    _points = alt.Chart(_rows).mark_circle(size=30, opacity=0.5, color="#0e9f8a").encode(y="selector:N", x=f"{sel_metric.value}:Q", tooltip=["selector", "repeat", "outer", sel_metric.value])
    _means = _rows.groupby("selector")[sel_metric.value].agg(["mean", "std", "count"]).round(4).reset_index()
    mo.vstack([(_box + _points).properties(width=620, height=40 * _rows["selector"].nunique() + 30), mo.ui.table(_means, selection=None)])


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. The poses of the benchmark

    Ten ligands with a crystal structure of CYP3A4 were predicted again and again, and each pose is measured by the RMSD to the crystal pose (lower is better; 2 Å is the usual line for a
    good pose). A **rule** picks one pose per ligand to submit: the model's own first sample, the pose that agrees best with the others, and so on.
    """)


@app.cell(hide_code=True)
def _(mo, poses):
    ligand = mo.ui.dropdown(options=["all ligands"] + sorted(poses["compound"].unique()), value="all ligands", label="Ligand")
    line = mo.ui.slider(start=0.5, stop=5.0, step=0.1, value=2.0, label="A good pose is within (Å)", show_value=True)
    mo.hstack([ligand, line], justify="start", gap=2)
    return ligand, line


@app.cell(hide_code=True)
def _(alt, ligand, line, mo, poses, rules):
    _poses = poses if ligand.value == "all ligands" else poses[poses["compound"] == ligand.value]
    _rules = rules if ligand.value == "all ligands" else rules[rules["compound"] == ligand.value]
    _strip = (
        alt.Chart(_poses)
        .mark_circle(size=60, opacity=0.7)
        .encode(x=alt.X("true_rmsd:Q", title="RMSD to the crystal pose (Å)"), y=alt.Y("engine:N", title=None), color="compound:N", tooltip=["compound", "engine", "sample", "true_rmsd"])
        .properties(width=620, height=140)
    )
    _rule = alt.Chart({"values": [{"x": line.value}]}).mark_rule(color="#c44", strokeDash=[5, 4]).encode(x="x:Q")
    _share = _poses.assign(good=_poses["true_rmsd"] <= line.value).groupby("engine")["good"].agg(["mean", "count"]).rename(columns={"mean": "share of poses within the line", "count": "poses"}).round(3).reset_index()
    _by_rule = _rules.groupby("rule")["rmsd"].agg(["mean", "median", "count"]).round(3).reset_index()
    mo.vstack([_strip + _rule, mo.md("**Poses within the line, by engine**"), mo.ui.table(_share, selection=None), mo.md("**RMSD of the pose each rule submits**"), mo.ui.table(_by_rule, selection=None)])


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    *What this page is.* A reader of finished results: it holds no model and starts no run. A result that you find here and want to use belongs in the notebook or the repository as
    documented code that has passed the same gates as every other idea. The tables are the public part of the repository (`explorer/public`, with a manifest of SHA-256 sums); the
    results for the blinded compounds are not in it.
    """)


if __name__ == "__main__":
    app.run()
