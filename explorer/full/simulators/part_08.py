# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part VIII, the final TDI model: the grid of candidates (8.1), a solver that stops where it happens to be (8.3), the calibration of a score (8.5) and the checks of a file of calls (8.6)."""

import io

import altair as alt
import numpy as np
import pandas as pd

from simulators import store, validation


def c_8_1(mo):
    blocks = list(store.table("blocks")["block"])
    return {"blocks": mo.ui.multiselect(options=blocks, value=[], label="Show the candidates that use every one of these blocks"), "family": mo.ui.text(value="", label="Words in the family")}


def v_8_1(mo, v):
    candidates = store.table("candidates").fillna({"blocks": "", "family": ""})
    used = candidates["blocks"].map(lambda s: set(s.split("+")) - {""})
    shown = candidates[used.map(lambda u: set(v["blocks"]) <= u) & candidates["family"].str.contains(v["family"], case=False, regex=False)]
    counts = candidates.groupby("family").size().rename("candidates").reset_index().sort_values("candidates", ascending=False)
    return mo.vstack([mo.md(f"The grid of the final stack has **{len(candidates)}** candidates, each a learner on a choice of feature blocks; **{len(shown)}** match. The nested validation fits every one of them in every fold."), mo.ui.table(shown, selection=None, page_size=10),
                      alt.Chart(counts).mark_bar().encode(y=alt.Y("family:N", sort="-x"), x="candidates:Q").properties(width=520, height=max(120, 18 * len(counts)))])


def c_8_3(mo):
    return {"tol": mo.ui.slider(start=-12, stop=0, step=1, value=-1, label="log10 of the tolerance at which the solver stops", show_value=True), "members": mo.ui.slider(start=3, stop=20, step=1, value=8, label="candidates in the blend", show_value=True)}


def v_8_3(mo, v):
    oof = store.table("oof_CYP3A4")
    columns = [c for c in oof.columns if c.endswith("|0") and not c.startswith("fold")][: int(v["members"])]
    r = validation.stack_coefficients(oof[columns].to_numpy(), oof["y"].to_numpy().astype(int), 10.0 ** v["tol"])
    frame = pd.DataFrame({"candidate": [c.split("|")[0] for c in columns], "stopped early": r["early"], "converged": r["converged"]})
    return mo.vstack([mo.md(f"A logistic blend of {len(columns)} candidates. Allowed to stop at a tolerance of 1e{int(v['tol'])} the solver ran {r['iterations']} iteration(s) and the coefficients differ from the converged ones (which need {r['iterations to converge']} iterations) by up to **{np.abs(r['early'] - r['converged']).max():.3f}**. "
                            "A blend that is not converged depends on where the solver happened to stop, which is the bug of 8.3: the same data gave different stacks until the tolerance was fixed."),
                      alt.Chart(frame.melt("candidate", ["stopped early", "converged"], "solver", "coefficient")).mark_bar().encode(x="candidate:N", y="coefficient:Q", color="solver:N", xOffset="solver:N").properties(width=560, height=240)])


def c_8_5(mo):
    return {"isoform": mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Enzyme", inline=True), "method": mo.ui.radio(options=["raw", "logistic", "isotonic"], value="raw", label="Calibration", inline=True), "bins": mo.ui.slider(start=4, stop=20, step=1, value=10, label="bins", show_value=True)}


def v_8_5(mo, v):
    oof = store.table(f"oof_{v['isoform']}")
    candidate = next(c.split("|")[0] for c in oof.columns if c.endswith("|0") and not c.startswith("fold"))
    y, s = validation.oof_scores(oof, candidate)
    r = validation.calibration(y, s, v["method"], int(v["bins"]))
    diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 3], color="gray").encode(x="x:Q", y="y:Q")
    points = alt.Chart(r["table"]).mark_line(point=True).encode(x=alt.X("predicted:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("observed:Q", scale=alt.Scale(domain=[0, 1])), size="n:Q")
    return mo.vstack([mo.md(f"The out-of-fold score of `{candidate}` ({v['isoform']}), {'as it comes' if v['method'] == 'raw' else v['method'] + ' calibration fitted on one half of the compounds and applied to the other'}: expected calibration error **{r['ece']:.3f}**, Brier score **{r['brier']:.3f}**. On the diagonal, a score of 0.3 would be right three times in ten."), (diagonal + points).properties(width=380, height=380)])


def c_8_6(mo):
    return {"file": mo.ui.file(kind="area", filetypes=[".csv"], label="A TDI file of calls (CSV), to check as the audit does")}


def v_8_6(mo, v):
    test = store.table("test_names")
    if v["file"]:
        frame, source = pd.read_csv(io.BytesIO(v["file"][0].contents)), v["file"][0].name
    else:
        frame, source = test.assign(CYP2D6_is_TDI=False, CYP3A4_is_TDI=[i % 3 == 0 for i in range(len(test))]), "(a demonstration file: calls made up for the check)"
    checks = pd.DataFrame([{"check": k, "passes": ok} for k, ok in validation.check_calls(frame, test).items()])
    return mo.vstack([mo.md(f"The audit makes six checks on every TDI file before it can be uploaded: the number of rows, the columns, no missing value, boolean calls and the names and the SMILES in the order of the test file. File: `{source}`. The calls themselves are not looked at."), mo.ui.table(checks, selection=None, pagination=False)])


SIMS = {"8.1": (c_8_1, v_8_1), "8.3": (c_8_3, v_8_3), "8.5": (c_8_5, v_8_5), "8.6": (c_8_6, v_8_6)}
