# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The page of a section that is told by experiments of the ledger: a control for the level of the gate and the enzyme, the verdicts, the paired differences and the log lines of the families named for the section."""

import altair as alt

from simulators import experiments


def controls(mo):
    return {"enzyme": mo.ui.radio(options=["both", "CYP3A4", "CYP2D6"], value="both", label="Enzyme", inline=True), "alpha": mo.ui.slider(start=0.001, stop=0.2, step=0.001, value=0.05, label="level of the gate (the corrected p must be below it)", show_value=True)}


def page(mo, v, families: list[str], intro: str):
    ledger = experiments.verdicts(families)
    paired = experiments.passes(experiments.comparisons(families, v["enzyme"]), v["alpha"])
    if ledger.empty and paired.empty:
        return mo.md(f"{intro}\n\nThe ledger has no gate for this section: the idea is taught in the notebook without a pre-registered verdict.")
    summary = experiments.gain_summary(paired)
    chart = alt.Chart(paired).mark_circle(size=70).encode(x=alt.X("auc_gain:Q", title="paired AUC gain"), y=alt.Y("p:Q", scale=alt.Scale(type="log"), title="corrected p"), color=alt.Color("passes:N", scale=alt.Scale(domain=[True, False], range=["#2e7d32", "#9e9e9e"])),
                                              shape="enzyme:N", tooltip=["log", "comparison", "auc_gain", "p", "z0"]) if len(paired) else None
    rule = alt.Chart(alt.Data(values=[{"p": v["alpha"]}])).mark_rule(strokeDash=[4, 3], color="#c62828").encode(y="p:Q") if len(paired) else None
    parts = [mo.md(intro), mo.md(f"A replay of what the project ran (nothing is fitted here). Families: `{', '.join(families)}`. **{len(ledger)}** pre-registered verdicts, **{experiments.share_met(ledger):.0%}** of them MET. "
                                f"Of **{summary['comparisons']}** paired comparisons, **{int(paired['passes'].sum()) if len(paired) else 0}** pass at a level of {v['alpha']:.3f}; the best gain is {summary['best AUC gain']:+.4f} AUC and the smallest p {summary['smallest p']:.3g}."),
             mo.ui.table(ledger, selection=None, page_size=8)]
    if chart is not None:
        parts.append((chart + rule).properties(width=560, height=260))
        parts.append(mo.accordion({"The lines of the logs": mo.ui.table(experiments.lines(families), selection=None, page_size=10)}))
    return mo.vstack(parts)
