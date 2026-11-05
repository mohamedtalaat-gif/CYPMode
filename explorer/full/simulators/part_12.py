# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part XII, engineering for trust: the order of a sum (12.1), the checks of the audit (12.2), the tests and the workflow (12.3), the runner of the queue (12.4), the generated documents (12.5), the rules (12.6) and the bundle of the explorer itself (12.7)."""

import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

from simulators import store

HERE = Path(__file__).resolve().parents[1]


def c_12_1(mo):
    return {"n": mo.ui.slider(start=1000, stop=1000000, step=1000, value=100000, label="numbers to add", show_value=True), "dtype": mo.ui.radio(options=["float32", "float64"], value="float32", label="precision", inline=True),
            "orders": mo.ui.slider(start=5, stop=50, step=5, value=20, label="orders of the same numbers", show_value=True)}


def v_12_1(mo, v):
    rng = np.random.default_rng(0)
    numbers = (rng.normal(0, 1, int(v["n"])) * 1e3 + 1e5).astype(v["dtype"])
    sums = np.array([numbers[np.random.default_rng(k).permutation(len(numbers))].sum(dtype=v["dtype"]) for k in range(int(v["orders"]))], dtype=float)
    spread = float(sums.max() - sums.min())
    return mo.vstack([mo.md(f"Adding the same {int(v['n']):,} numbers in {int(v['orders'])} different orders in {v['dtype']}: the sums differ by up to **{spread:.4g}** (a relative {spread / abs(sums.mean()):.2e}). A computer adds numbers one after another and rounds each time, so the order changes the result; "
                            "threads that add in a different order from run to run give a different model from the same data, which is why the project fixes seeds, threads and solvers and checks that two runs of the audit are byte for byte the same."),
                      alt.Chart(pd.DataFrame({"sum": sums})).mark_bar().encode(x=alt.X("sum:Q", bin=alt.Bin(maxbins=20)), y="count()").properties(width=480, height=200)])


def c_12_2(mo):
    sections = ["(all)", *sorted(store.table("audit_checks")["section"].unique())]
    return {"section": mo.ui.dropdown(options=sections, value="(all)", label="Section of the audit"), "status": mo.ui.multiselect(options=["PASS", "FAIL", "INFO", "WARN"], value=["PASS", "FAIL", "INFO", "WARN"], label="Status"), "word": mo.ui.text(value="", label="Words in the check")}


def v_12_2(mo, v):
    checks = store.table("audit_checks").fillna("")
    shown = checks[checks["status"].isin(v["status"]) & checks["check"].str.contains(v["word"], case=False, regex=False)]
    if v["section"] != "(all)":
        shown = shown[shown["section"] == v["section"]]
    counts = checks.groupby(["section", "status"]).size().rename("checks").reset_index()
    return mo.vstack([mo.md(f"The audit report holds **{len(checks)}** checks in {checks['section'].nunique()} sections: **{int((checks['status'] == 'PASS').sum())}** pass, **{int((checks['status'] == 'FAIL').sum())}** fail. The audit is generated twice and the two reports must be byte for byte the same. **{len(shown)}** checks match."),
                      mo.ui.table(shown, selection=None, page_size=10), alt.Chart(counts).mark_bar().encode(y=alt.Y("section:N", sort="-x", axis=alt.Axis(labelLimit=260)), x="checks:Q", color="status:N").properties(width=520, height=max(200, 16 * counts["section"].nunique()))])


def c_12_3(mo):
    return {"word": mo.ui.text(value="", label="Words in the name of the test file")}


def v_12_3(mo, v):
    tests = store.table("tests")
    shown = tests[tests["file"].str.contains(v["word"], case=False, regex=False)].sort_values("tests", ascending=False)
    workflow = HERE.parents[1] / ".github/workflows/tests.yml"
    steps = re.findall(r"^\s+- name: (.+)$", workflow.read_text(), re.MULTILINE) if workflow.exists() else []
    return mo.vstack([mo.md(f"The suite has **{len(tests)}** test files and **{int(tests['tests'].sum()):,}** tests; **{len(shown)}** files match. The workflow of continuous integration runs them on three systems and has the steps: {'; '.join(steps) or '(workflow not found)'}."), mo.ui.table(shown, selection=None, page_size=10)])


def c_12_4(mo):
    jobs = store.table("jobs")
    numbers = sorted(set(jobs["number"]))
    return {"done": mo.ui.multiselect(options=[str(n) for n in numbers], value=[str(n) for n in numbers if n < 900], label="Jobs that are done"), "failed": mo.ui.multiselect(options=[str(n) for n in numbers], value=[], label="Jobs that failed twice")}


def v_12_4(mo, v):
    jobs = store.table("jobs").sort_values("number")
    finished = {int(n) for n in v["done"]} | {int(n) for n in v["failed"]}
    waiting = jobs[~jobs["number"].isin(finished)]
    nxt = waiting.head(5)
    return mo.vstack([mo.md(f"The runner takes the lowest-numbered job that is neither done nor failed. With {len(v['done'])} done and {len(v['failed'])} failed, **{len(waiting)}** are waiting and the next is **{nxt.iloc[0]['job'] if len(nxt) else 'none'}**. "
                            "(A job also checks its own conditions, such as a clean working tree, and answers 'not ready' until they hold; that part is not modelled here. A failed job is tried once more and then skipped, so that the chain never stops.)"), mo.ui.table(nxt, selection=None, pagination=False)])


def c_12_5(mo):
    return {"kind": mo.ui.radio(options=["built (UTC)", "files written"], value="files written", label="Show", inline=True)}


def v_12_5(mo, v):
    artifacts = store.table("artifacts")
    return mo.vstack([mo.md("The documents and pictures that present the results are generated, and each generator records the time, the files it wrote and a fingerprint of the numbers it read. The audit refuses to call the project ready while one of them is older than the last output."),
                      mo.ui.table(artifacts, selection=None, pagination=False), alt.Chart(artifacts).mark_bar().encode(x="artifact:N", y="files written:Q").properties(width=420, height=200)])


def c_12_6(mo):
    return {"word": mo.ui.text(value="", label="Words in the rule")}


def v_12_6(mo, v):
    rules = store.table("rules")
    return mo.vstack([mo.md("The rules that no result or deadline changes, and the mechanism that enforces each: a rule that a program can check is checked, and one that cannot is called a rule of procedure."), mo.ui.table(rules[rules.apply(lambda r: v["word"].lower() in " ".join(map(str, r)).lower(), axis=1)], selection=None, pagination=False)])


def c_12_7(mo):
    tables = sorted(p.name for p in (HERE / "data/public").iterdir() if p.suffix in {".csv", ".gz"})
    return {"table": mo.ui.dropdown(options=tables, value=tables[0], label="A table of the public bundle")}


def v_12_7(mo, v):
    folder = HERE / "data/public"
    frame = pd.read_csv(folder / v["table"])
    sizes = pd.DataFrame([{"table": p.name, "kB": round(p.stat().st_size / 1000, 1)} for p in sorted(folder.iterdir()) if p.suffix in {".csv", ".gz"}])
    return mo.vstack([mo.md(f"The explorer reads a small bundle of the finished results: **{len(sizes)}** tables, **{sizes['kB'].sum() / 1000:.1f} MB**. `{v['table']}` has {len(frame):,} rows and {frame.shape[1]} columns; the first rows:"), mo.ui.table(frame.head(50), selection=None, page_size=8), mo.accordion({"All the tables of the bundle": mo.ui.table(sizes, selection=None, pagination=False)})])


SIMS = {"12.1": (c_12_1, v_12_1), "12.2": (c_12_2, v_12_2), "12.3": (c_12_3, v_12_3), "12.4": (c_12_4, v_12_4), "12.5": (c_12_5, v_12_5), "12.6": (c_12_6, v_12_6), "12.7": (c_12_7, v_12_7)}
