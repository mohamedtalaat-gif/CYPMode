# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part XIII, closing: the claims and their sections (13.1), the limitations (13.2), the open questions (13.3) and the final results with the state of the submission files (13.4); and the analysis after the deadline:
the gate that reproduces the board (13.5), the call rate and the threshold on the released labels (13.6), the iron contact and the pool of poses on the released structures (13.7), the error bar of 20 ligands (13.8) and the
exploratory comparison with the first entry of the structure board (13.9). The tables of 13.5 to 13.9 are written by ``build_data.py`` from ``data/post_deadline/outputs``; the 20 ligands are numbered and not named."""

import altair as alt
import numpy as np
import pandas as pd

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


def paired_interval(differences, resamples: int = 10_000, seed: int = 42):
    """The mean of paired differences and the 2.5th and 97.5th percentiles of its paired bootstrap, as ``cypmode.post_deadline.structure.paired_difference`` forms them (the same seed and draw)."""
    d = np.asarray(differences, dtype=float)
    means = d[np.random.default_rng(seed).integers(0, len(d), size=(resamples, len(d)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(low), float(high)


def verdict(low: float, high: float) -> str:
    return "the first is better" if low > 0 else "the second is better" if high < 0 else "no difference shown"


TDI_TOLERANCES = ["0.01 (registered)", "0.001", "0.0001", "0.000001", "0.000000001"]
STRUCTURE_TOLERANCES = ["0.005 (registered)", "0.001", "0.0001"]


def c_13_5(mo):
    return {
        "tdi": mo.ui.dropdown(options=TDI_TOLERANCES, value=TDI_TOLERANCES[0], label="Tolerance on the macro MCC"),
        "structure": mo.ui.dropdown(options=STRUCTURE_TOLERANCES, value=STRUCTURE_TOLERANCES[0], label="Tolerance on the LDDT-PLI"),
    }


def v_13_5(mo, v):
    gate = store.table("pd_gate").copy()
    tolerance = {"macro MCC": float(v["tdi"].split()[0]), "LDDT-PLI": float(v["structure"].split()[0]), "BiSyRMSD": 0.1}
    gate["difference"] = gate["scored"] - gate["board"]
    gate["tolerance"] = gate["metric"].map(tolerance)
    gate["verdict"] = np.where(gate["difference"].abs() <= gate["tolerance"], "within", "outside")
    tdi = gate[gate["track"] == "TDI"]
    inside = list(tdi.loc[tdi["verdict"] == "within", "entry"])
    shown = gate.assign(scored=gate["scored"].map("{:.7f}".format), board=gate["board"].map("{:.7f}".format), difference=gate["difference"].map("{:+.2e}".format), tolerance=gate["tolerance"].map("{:g}".format))
    both = len(inside) == 2
    reading = (
        "Both candidates lie within it, so the gate cannot tell the file that was submitted from the other one: this is how the registered tolerance failed, and why the exact match of the mean and the standard deviation of the bootstrap was used instead."
        if both
        else ("Only " + inside[0] + " lies within it, so the board names the file that was submitted." if inside else "Neither candidate lies within it.")
    )
    return mo.vstack([
        mo.md(f"The plan fixed a tolerance before any label was read: the score of the submitted file, run through the scoring of the organisers, must lie within it of the value on the final board. At a tolerance of **{tolerance['macro MCC']:g}** on the macro MCC, **{len(inside)}** of the two TDI candidates lie within it ({', '.join(inside) or 'none'}). {reading}"),
        mo.ui.table(shown[["track", "entry", "metric", "scored", "board", "difference", "tolerance", "verdict"]], selection=None, pagination=False),
    ])


def c_13_6(mo):
    return {
        "enzyme": mo.ui.radio(options=["CYP3A4", "CYP2D6"], value="CYP3A4", label="Enzyme", inline=True),
        "rate": mo.ui.slider(start=0.02, stop=0.60, step=0.01, value=0.40, label="Share of compounds called positive"),
    }


def v_13_6(mo, v):
    table = store.table("pd_threshold")
    part = table[table["enzyme"] == v["enzyme"]]
    curve, marks = part[part["kind"] == "curve"].sort_values("call_rate"), part[part["kind"] != "curve"]
    at = float(np.interp(v["rate"], curve["call_rate"], curve["mcc"]))
    submitted, best = (marks[marks["kind"] == kind].iloc[0] for kind in ("submitted calls", "best threshold"))
    wide = table[table["kind"] != "curve"].pivot(index="enzyme", columns="kind", values="mcc")
    gain = float((wide["best threshold"] - wide["submitted calls"]).mean())
    line = alt.Chart(curve).mark_line().encode(x=alt.X("call_rate:Q", title="Share of compounds called positive", scale=alt.Scale(domain=[0, 0.62])), y=alt.Y("mcc:Q", title="MCC on the released labels"))
    dots = alt.Chart(marks).mark_point(filled=True, size=120).encode(x="call_rate:Q", y="mcc:Q", color=alt.Color("kind:N", title=None), tooltip=["kind", "call_rate", "mcc"])
    here = alt.Chart(pd.DataFrame({"call_rate": [v["rate"]], "mcc": [at]})).mark_point(shape="diamond", size=200, color="#e5773d", filled=True).encode(x="call_rate:Q", y="mcc:Q")
    return mo.vstack([
        mo.md(f"With **{v['rate']:.0%}** of the {v['enzyme']} compounds called positive, the MCC of the submitted scores on the released labels is **{at:.3f}**. The submitted calls ({submitted['call_rate']:.1%}) gave {submitted['mcc']:.3f}; the best threshold, which needs the labels, would have given {best['mcc']:.3f} at {best['call_rate']:.1%}. Averaged over the two enzymes the best threshold would have added **{gain:.3f}** to the macro MCC, against a gap of 0.126 to the first place: the ceiling uses the answer, and it is small next to the gap."),
        line + dots + here,
    ])


def c_13_7(mo):
    return {
        "cut": mo.ui.slider(start=2.0, stop=3.6, step=0.1, value=2.6, label="Iron-nitrogen cut-off (angstroms)"),
        "ligand": mo.ui.dropdown(options=[f"ligand {i}" for i in range(1, 21)], value="ligand 1", label="Pool of poses of"),
    }


def v_13_7(mo, v):
    ligands, pool = store.table("pd_ligands"), store.table("pd_pool")
    truth, submitted = ligands["truth_distance"] <= v["cut"], ligands["submitted_distance"] <= v["cut"]
    counts = pd.crosstab(truth, submitted).reindex(index=[False, True], columns=[False, True], fill_value=0)
    counts.index, counts.columns = ["not coordinated in the released structure", "coordinated in the released structure"], ["not coordinated in the submitted pose", "coordinated in the submitted pose"]
    number = int(v["ligand"].split()[1])
    row, poses = ligands[ligands["ligand"] == number].iloc[0], pool[pool["ligand"] == number].reset_index(drop=True)
    poses = poses.assign(pose=range(1, len(poses) + 1))
    marks = pd.DataFrame({"what": ["submitted pose", "published baseline", "best pose of the pool"], "lddt_pli": [row["submitted_pli"], row["baseline_pli"], row["pool_best"]]})
    dots = alt.Chart(poses).mark_point(filled=True, size=90, color="#6c97ff").encode(x=alt.X("pose:O", title="Pose of the pool"), y=alt.Y("lddt_pli:Q", title="LDDT-PLI", scale=alt.Scale(domain=[0, 1])), tooltip=["pose", "lddt_pli"])
    rules = alt.Chart(marks).mark_rule(strokeDash=[4, 3]).encode(y="lddt_pli:Q", color=alt.Color("what:N", title=None))
    called, missed, extra = int(counts.iloc[1, 1]), int(counts.iloc[1, 0]), int(counts.iloc[0, 1])
    return mo.vstack([
        mo.md(f"A ligand counts as coordinated when a nitrogen of the ligand lies within the cut-off of an iron of the heme. At **{v['cut']:.1f}** angstroms the released structures have **{int(truth.sum())}** coordinated ligands; the submitted poses find **{called}** of them, miss **{missed}**, and call **{extra}** more that the structures do not show (a distance near the cut-off, in maps of 3.0 to 3.6 angstroms resolution, does not exclude a contact). {v['ligand'].capitalize()}: the pool holds {len(poses)} poses, the best scores {row['pool_best']:.3f}, the submitted pose {row['submitted_pli']:.3f}."),
        mo.ui.table(counts.reset_index(names="released structure"), selection=None, pagination=False),
        dots + rules,
    ])


def c_13_8(mo):
    return {
        "n": mo.ui.slider(start=4, stop=20, step=1, value=20, label="Number of ligands"),
        "resamples": mo.ui.dropdown(options=["200", "1000", "5000"], value="1000", label="Bootstrap resamples"),
        "seed": mo.ui.number(start=0, stop=9999, value=42, label="Seed of the draw"),
    }


def v_13_8(mo, v):
    ligands = store.table("pd_ligands")
    difference = (ligands["submitted_pli"] - ligands["baseline_pli"]).to_numpy()
    seed, resamples = int(v["seed"]), int(v["resamples"])
    order = np.random.default_rng(seed).permutation(len(difference))
    frame = pd.DataFrame([(n, *paired_interval(difference[order[:n]], resamples, seed)) for n in range(4, len(difference) + 1)], columns=["ligands", "mean", "low", "high"])
    chosen = frame[frame["ligands"] == int(v["n"])].iloc[0]
    bars = alt.Chart(frame).mark_errorbar(thickness=2).encode(x=alt.X("ligands:O", title="Ligands in the comparison"), y=alt.Y("low:Q", title="Submitted minus baseline, LDDT-PLI"), y2="high:Q")
    means = alt.Chart(frame).mark_point(filled=True, size=50).encode(x="ligands:O", y="mean:Q")
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="#e8575d").encode(y="y:Q")
    return mo.vstack([
        mo.md(f"The mean difference between the submitted poses and the published baseline on **{int(v['n'])}** ligands is {chosen['mean']:+.4f}, and its 95% interval runs from {chosen['low']:+.4f} to {chosen['high']:+.4f}: {verdict(chosen['low'], chosen['high'])}. The interval is {chosen['high'] - chosen['low']:.3f} wide. With few ligands the bars reach far on both sides of zero, and with all 20 they still do, so a difference below about 0.04 is reported as not shown, whatever its sign."),
        bars + means + zero,
    ])


def c_13_9(mo):
    return {
        "metric": mo.ui.radio(options=["LDDT-PLI", "BiSyRMSD", "LDDT-LP"], value="LDDT-PLI", label="Metric", inline=True),
        "against": mo.ui.radio(options=["submitted poses", "published baseline"], value="submitted poses", label="Compared with", inline=True),
        "leave": mo.ui.slider(start=0, stop=6, step=1, value=0, label="Ligands left out, the largest differences first"),
        "resamples": mo.ui.dropdown(options=["1000", "10000"], value="10000", label="Bootstrap resamples"),
    }


def v_13_9(mo, v):
    ligands = store.table("pd_ligands")
    key, other = {"LDDT-PLI": "pli", "BiSyRMSD": "bisy", "LDDT-LP": "lp"}[v["metric"]], ("submitted" if v["against"] == "submitted poses" else "baseline")
    difference = ligands[f"external_{key}"] - ligands[f"{other}_{key}"]
    if v["metric"] == "BiSyRMSD":
        difference = -difference  # a lower BiSyRMSD is better, so the sign is reversed
    ordered = difference.sort_values(ascending=False)
    kept = ordered.iloc[int(v["leave"]):]
    mean, low, high = paired_interval(kept.to_numpy(), int(v["resamples"]), 42)
    share = float(difference.nlargest(3).sum() / difference.sum()) if difference.sum() > 0 else float("nan")
    above = int((ligands["external_pli"] > ligands["pool_best"]).sum())
    frame = pd.DataFrame({"ligand": ordered.index.map(lambda i: f"ligand {int(ligands.loc[i, 'ligand'])}"), "difference": ordered.to_numpy(), "state": ["left out"] * int(v["leave"]) + ["kept"] * len(kept)})
    bars = alt.Chart(frame).mark_bar().encode(x=alt.X("ligand:N", sort=None, title=None), y=alt.Y("difference:Q", title=f"Published poses minus {other}, {v['metric']}"), color=alt.Color("state:N", title=None, scale=alt.Scale(domain=["kept", "left out"], range=["#6c97ff", "#93a3ba"])), tooltip=["ligand", "difference"])
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="#e8575d").encode(y="y:Q")
    return mo.vstack([
        mo.md(f"The poses that the first entry of the structure board published, scored with the same function, against the {v['against']} (for the BiSyRMSD the sign is reversed, so that a positive value is better for all three metrics). On **{len(kept)}** of the 20 ligands the mean difference is {mean:+.4f}, with a 95% interval of {low:+.4f} to {high:+.4f}: {verdict(low, high)}. The three ligands with the largest differences give {share:.0%} of the mean difference over all 20, and on {above} ligands the published pose scores higher than the best pose of the pool of the project, which no choice among those poses could have given. The comparison is exploratory: the labels were known, and the six comparisons are not corrected for their number."),
        bars + zero,
    ])


SIMS = {"13.1": (c_13_1, v_13_1), "13.2": (c_13_2, v_13_2), "13.3": (c_13_3, v_13_3), "13.4": (c_13_4, v_13_4), "13.5": (c_13_5, v_13_5), "13.6": (c_13_6, v_13_6), "13.7": (c_13_7, v_13_7), "13.8": (c_13_8, v_13_8), "13.9": (c_13_9, v_13_9)}
