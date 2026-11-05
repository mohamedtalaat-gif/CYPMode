# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part XIII, closing: the claims and their sections (13.1), the limitations (13.2), the open questions (13.3) and the final results with the state of the submission files (13.4)."""

import altair as alt

from simulators import store


def c_13_1(mo):
    return {"word": mo.ui.text(value="", label="Words in the claim")}


def v_13_1(mo, v):
    claims = store.table("claims")
    shown = claims[claims["claim"].str.contains(v["word"], case=False, regex=False)]
    return mo.vstack([mo.md(f"What the project shows is what it can point to: **{claims['claim'].nunique()}** claims, each with the sections that explain it and compute its number. What it does not show is in the limitations (13.2). Every figure of the project is a cross-validated estimate on the training data, never a score on the hidden labels."), mo.ui.table(shown, selection=None, pagination=False)])


def c_13_2(mo):
    kinds = ["(all)", *sorted(store.table("limitations")["kind"].unique())]
    return {"kind": mo.ui.dropdown(options=kinds, value="(all)", label="Kind of limitation"), "word": mo.ui.text(value="", label="Words in the limitation")}


def v_13_2(mo, v):
    limits = store.table("limitations")
    shown = limits[limits["text"].str.contains(v["word"], case=False, regex=False)]
    if v["kind"] != "(all)":
        shown = shown[shown["kind"] == v["kind"]]
    return mo.vstack([mo.md(f"**{len(limits)}** limitations are logged in the method report, in {limits['kind'].nunique()} kinds; **{len(shown)}** match. Each is a thing the project does not know or cannot claim, with the numbers computed where the report is built."), mo.ui.table(shown, selection=None, page_size=6)])


def c_13_3(mo):
    return {"word": mo.ui.text(value="", label="Words in the question")}


def v_13_3(mo, v):
    questions = store.table("open_questions")
    shown = questions[questions.apply(lambda r: v["word"].lower() in " ".join(map(str, r)).lower(), axis=1)]
    return mo.vstack([mo.md("The questions whose answers would change what the project does next, with what would settle each."), mo.ui.table(shown, selection=None, page_size=5)])


def c_13_4(mo):
    return {"cut": mo.ui.radio(options=["AUC at z0 = 1.96", "AUC at z0 = 1.0"], value="AUC at z0 = 1.96", label="Confidence cut of the labels", inline=True)}


def v_13_4(mo, v):
    stacks = store.table("final_stacks")
    files = store.table("submission_files").fillna("")
    chart = alt.Chart(stacks).mark_bar().encode(x="isoform:N", y=alt.Y(f"{v['cut']}:Q", scale=alt.Scale(domain=[0.5, 1.0])), color="isoform:N")
    return mo.vstack([mo.md("The final stack of each enzyme, with its nested cross-validated estimate (an estimate on the training data, not a score on the hidden labels), and the files that were submitted with their SHA-256."), mo.ui.table(stacks, selection=None, pagination=False), chart.properties(width=320, height=220),
                      mo.ui.table(files, selection=None, pagination=False)])


SIMS = {"13.1": (c_13_1, v_13_1), "13.2": (c_13_2, v_13_2), "13.3": (c_13_3, v_13_3), "13.4": (c_13_4, v_13_4)}
