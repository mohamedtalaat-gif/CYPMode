"""Figures added when the project grew past the first plan: what the leak guard checked, how the test prevalence was recovered from the public boards, why a small gain cannot be certified,
the neural pair from discovery to replicate, the reliability of the final stack by similarity, the choice of the call rate, and the whole set of submitted structures."""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from catalog import figure
from style import apply, tidy

ROOT = Path(__file__).resolve().parents[3]
LOGS = "data/structure_track/outputs/cyp3a4/logs"
FINAL_TDI = "data/tdi_track/outputs/final"
BOARDS = "data/tdi_track/inputs/leaderboard_snapshots"
FINAL_PDB = "data/structure_track/outputs/cyp3a4/final/_pdb_files"
ISOFORMS = ("CYP3A4", "CYP2D6")
STACKED = re.compile(r"z0=([\d.]+)\s+stacked base\+added vs base: AUC ([+-][\d.]+) \(p=([\d.e-]+), better (\d+)/(\d+)\)")
N_TEST_OVER_N_TRAIN = {5: 0.25, 10: 1 / 9}  # the test fraction of an outer fold over the training fraction, for 5 and 10 outer folds


def stacked_lines(name: str) -> dict[float, tuple[float, float, int, int]]:
    """{z0: (mean AUC difference, corrected p, folds better, folds)} of the stacked comparison in one experiment log; empty when the log is not there (yet)."""
    path = ROOT / LOGS / name
    if not path.exists():
        return {}
    return {float(z0): (float(d), float(p), int(b), int(n)) for z0, d, p, b, n in STACKED.findall(path.read_text())}


# -- what the leak guard checked ---------------------------------------------------------------------------------------------------------------

def scout_data() -> dict:
    sys.path[:0] = [str(ROOT / "docs/video")]
    import facts

    return facts._scout()


def leak_guard_caption() -> str:
    data = scout_data()
    taken = ", ".join(data["excluded"])
    return (f"The files of the organisers' releases that a model could read, and what each shares with the blinded test set. Only the identifier column of each file was read to make this "
            f"picture: no measurement of a test compound is looked at, not even for a figure. {len(data['excluded'])} compounds of the blinded set turn up in three of the {len(data['sources'])} files "
            f"({taken}); they are removed from every use of those files, and the organisers confirmed the overlap of the Octant release. The other files hold no test compound.")


@figure("data_leak_guard", "data", "The files released by the organisers, checked against the blinded test set", leak_guard_caption,
        data=("data/tdi_track/inputs/official_extra", "data/tdi_track/inputs/octant_cache/inhibition.parquet", "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"), section="2.8")
def leak_guard(theme_name):
    th = apply(theme_name)
    sources = scout_data()["sources"]
    fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(sources) + 1.6))
    y = np.arange(len(sources))
    ax.barh(y, [s["compounds"] for s in sources], color=th["blue"], alpha=0.85, label="compounds in the file")
    ax.barh(y, [s["in_training"] for s in sources], color=th["teal"], alpha=0.9, height=0.4, label="of them in the training set")
    for i, s in enumerate(sources):
        marker = f"{s['in_test']} in the blinded test set: removed" if s["in_test"] else "none in the blinded test set"
        ax.text(s["compounds"] * 1.01 + 80, i, marker, va="center", fontsize=9, color=th["red"] if s["in_test"] else th["green"], fontweight="bold")
    ax.set_yticks(y, [s["label"] for s in sources])
    ax.invert_yaxis()
    ax.set_xlim(0, max(s["compounds"] for s in sources) * 1.55)
    ax.set_xlabel("compounds (rows of the file counted by compound)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    tidy(ax, th)
    fig.tight_layout()
    return fig


# -- the test prevalence recovered from the public boards ---------------------------------------------------------------------------------------

def board_prevalence(iso: str) -> tuple[np.ndarray, dict]:
    from cypmode.challenge.leaderboard_prior import estimate_from_csv, implied_prevalence

    board = pd.read_csv(ROOT / BOARDS / f"interim_{iso}_2026-10-04.csv").dropna(subset=["Username"])
    values = implied_prevalence(board["Accuracy"], board["Precision"], board["Recall"])
    return values[np.isfinite(values)], estimate_from_csv(ROOT / BOARDS / f"interim_{iso}_2026-10-04.csv")


def prevalence_caption() -> str:
    parts = []
    for iso in ISOFORMS:
        values, estimate = board_prevalence(iso)
        parts.append(f"{iso}: {len(values)} entries, median {estimate['prevalence']:.3f}, spread (standard deviation) {estimate['spread']:.4f}")
    return ("Every row of the public interim board publishes the accuracy, the precision and the recall of one prediction file; the three numbers fix the share of positives among the scored "
            "compounds, p = (1 - accuracy) / ((1 - recall) + (recall / precision - recall)). All rows score the same hidden labels, so the implied p must agree across rows, and it does: "
            + "; ".join(parts) + ". This aggregate is the only information about the hidden labels that the project uses (to weight its yardstick and to set the call rate); it was available "
            "because the boards publish it, no probing submission was made, and it is disclosed as limitation L13 of the method report.")


@figure("data_prevalence_recovery", "data", "The positive rate of the hidden test set, recovered from the public board", prevalence_caption,
        data=(f"{BOARDS}/interim_<isoform>_2026-10-04.csv",), section="2.10")
def prevalence_recovery(theme_name):
    th = apply(theme_name)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax, iso, color in zip(axes, ISOFORMS, (th["blue"], th["orange"]), strict=True):
        values, estimate = board_prevalence(iso)
        ax.scatter(np.arange(len(values)), values, s=9, color=color, alpha=0.8)
        ax.axhline(estimate["prevalence"], color=th["text"], lw=1)
        ax.text(len(values) * 0.98, estimate["prevalence"], f"median {estimate['prevalence']:.3f}", ha="right", va="bottom", fontsize=9)
        ax.set(title=iso, xlabel="board entries (one point per prediction file)", ylim=(0, max(0.5, float(np.nanmax(values)) * 1.15)))
        tidy(ax, th)
    axes[0].set_ylabel("implied share of positives")
    fig.tight_layout()
    return fig


# -- why a small gain cannot be certified ---------------------------------------------------------------------------------------------------------

def logged_comparisons() -> pd.DataFrame:
    """Every stacked comparison of the 5-fold design (15 folds) in the experiment logs, with the standard error implied by its p value."""
    rows = []
    for path in sorted((ROOT / LOGS).glob("*_CYP*.log")):
        if "_f10_" in path.name or path.name.startswith(("replicate", "committee", "selection")):
            continue
        for z0, d, p, better, folds in STACKED.findall(path.read_text()):
            d, p, folds = float(d), float(p), int(folds)
            if folds != 15 or not 0 < p < 1:
                continue
            rows.append({"log": path.name, "diff": d, "p": p, "se": abs(d) / stats.t.isf(p / 2, folds - 1)})
    return pd.DataFrame(rows)


def floor_caption() -> str:
    frame = logged_comparisons()
    sd = float(np.median(frame["se"])) / np.sqrt(1 / 15 + 0.25)
    floors = {j: stats.t.isf(0.025, j - 1) * sd * np.sqrt(1 / j + r) for j, r in ((5, 0.25), (15, 0.25), (30, 1 / 9))}
    return (f"The smallest gain in AUC that the corrected resampled t can certify (two-sided p < 0.05), as a function of the number of folds J, for the standard deviation of the fold differences "
            f"that the logged comparisons imply ({sd:.4f}, the median of {len(frame)} comparisons). With 5 outer folds and one repeat (J = 5) it is {floors[5]:.4f}; with 5 outer folds and 3 repeats "
            f"(J = 15, the design of the gates) {floors[15]:.4f}; with 10 outer folds and 3 repeats (J = 30, the replicate) {floors[30]:.4f}. The variance factor 1/J + n_test/n_train never falls below the "
            f"test fraction, so more repeats shrink the floor only so far: a gain of a few thousandths of AUC cannot be certified by this design, whatever its truth. Right: every logged "
            f"comparison as a point (its gain against the standard error implied by its p value); the lines are the certification limits of the two designs.")


@figure("ml_design_floor", "ml", "The smallest gain a gate can certify", floor_caption, data=(f"{LOGS}/<experiment>_<isoform>.log",), section="6.5")
def design_floor(theme_name):
    th = apply(theme_name)
    frame = logged_comparisons()
    sd = float(np.median(frame["se"])) / np.sqrt(1 / 15 + 0.25)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.3))
    js = np.arange(3, 61)
    for r, label, color in ((0.25, "5 outer folds: n_test/n_train = 1/4", th["blue"]), (1 / 9, "10 outer folds: n_test/n_train = 1/9", th["orange"])):
        left.plot(js, [stats.t.isf(0.025, j - 1) * sd * np.sqrt(1 / j + r) for j in js], color=color, label=label)
        left.plot(js, [(stats.t.isf(0.05, j - 1) + stats.t.ppf(0.8, j - 1)) * sd * np.sqrt(1 / j + r) for j in js], color=color, ls="--", lw=1)
    for j, r, name in ((5, 0.25, "5 x 1"), (15, 0.25, "5 x 3"), (30, 1 / 9, "10 x 3")):
        value = stats.t.isf(0.025, j - 1) * sd * np.sqrt(1 / j + r)
        left.plot(j, value, "o", color=th["text"])
        left.annotate(f"{name}\n{value:.4f}", (j, value), textcoords="offset points", xytext=(6, 8), fontsize=8)
    left.set(xlabel="J, the number of folds in the comparison", ylabel="smallest certifiable gain in AUC", ylim=(0, None))
    left.legend(frameon=False, fontsize=8)
    left.text(0.98, 0.04, "solid: p < 0.05 at the margin; dashed: gain found with 80% power", transform=left.transAxes, ha="right", fontsize=7.5, color=th["muted"])
    tidy(left, th)
    se = np.linspace(0, float(frame["se"].max()) * 1.05, 50)
    right.fill_between(se, -stats.t.isf(0.025, 14) * se, stats.t.isf(0.025, 14) * se, color=th["grid"], alpha=0.6, label="cannot be certified at J = 15")
    right.scatter(frame["se"], frame["diff"], s=14, color=np.where(frame["p"] < 0.05, th["green"], th["muted"]), alpha=0.9)
    right.axhline(0, color=th["text"], lw=0.8)
    right.set(xlabel="standard error of the comparison (implied by its p value)", ylabel="gain in AUC when added to the stack")
    right.legend(frameon=False, fontsize=8, loc="lower right")
    tidy(right, th)
    fig.tight_layout()
    return fig


# -- the neural pair: from near-misses to a discovery and its replicate -------------------------------------------------------------------------------

NEURAL_ROWS = (("multitask network", "aux_mlp_{iso}.log"), ("plain Chemprop", "aux_chemprop_{iso}.log"), ("neural pair (15 folds)", "aux_neural_{iso}.log"), ("neural pair (10-fold replicate)", "aux_neural_f10_{iso}.log"))


def neural_caption() -> str:
    pair, replicate = stacked_lines("aux_neural_CYP3A4.log"), stacked_lines("aux_neural_f10_CYP3A4.log")
    first = f"{pair[1.0][0]:+.4f} AUC (p {pair[1.0][1]:.3f}) and {pair[1.96][0]:+.4f} (p {pair[1.96][1]:.3f})" if pair else "not yet logged"
    second = (f"the replicate on the 10-outer-fold plan gives {replicate[1.0][0]:+.4f} (p {replicate[1.0][1]:.3f}) and {replicate[1.96][0]:+.4f} (p {replicate[1.96][1]:.3f})" if replicate
              else "the replicate on the 10-outer-fold plan has not finished")
    return ("The change in outer AUC when a candidate is added to the default stack on CYP3A4, at the two confidence cuts (circle z0 = 1.0, square 1.96). The multitask network and plain Chemprop "
            f"each came close to the gate and missed it; their pair, a group formed after seeing that, gives {first}. A group found after looking is a discovery, not a verdict: the pre-declared "
            f"confirmation is the 10-fold replicate, and {second}. Filled green markers pass (p < 0.05 and a positive gain at the same cut).")


@figure("ml_neural_pair", "ml", "Two near-misses and the pair that joined them", neural_caption, data=tuple(f"{LOGS}/{n.format(iso='CYP3A4')}" for _, n in NEURAL_ROWS), section="9.9")
def neural_pair(theme_name):
    th = apply(theme_name)
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    for i, (label, pattern) in enumerate(NEURAL_ROWS):
        lines = stacked_lines(pattern.format(iso="CYP3A4"))
        if not lines:
            ax.text(0.0005, i, "not run yet", va="center", fontsize=9, color=th["muted"], style="italic")
            continue
        for z0, dy, marker in ((1.0, -0.14, "o"), (1.96, 0.14, "s")):
            if z0 not in lines:
                continue
            diff, p, better, folds = lines[z0]
            passed = p < 0.05 and diff > 0
            color = th["green"] if passed else th["muted"]
            ax.plot(diff, i + dy, marker, color=color, mfc=color if passed else "none", ms=7)
            ax.text(diff + 0.0003, i + dy, f"p {p:.3f}, better in {better}/{folds}", va="center", fontsize=7.5, color=th["muted"])
    ax.axvline(0, color=th["text"], lw=1)
    ax.set_yticks(range(len(NEURAL_ROWS)), [r[0] for r in NEURAL_ROWS])
    ax.set_ylim(len(NEURAL_ROWS) - 0.4, -0.6)
    ax.set_xlim(-0.003, 0.0185)
    ax.set_xlabel("change in outer AUC when added to the default stack (CYP3A4)")
    tidy(ax, th)
    fig.tight_layout()
    return fig


# -- the reliability of the final stack by similarity to the training data ------------------------------------------------------------------------

def bins_frame() -> pd.DataFrame:
    return pd.read_csv(ROOT / FINAL_TDI / "applicability_domain_bins.csv")


def ad_caption() -> str:
    frame = bins_frame()
    compounds = pd.read_csv(ROOT / FINAL_TDI / "applicability_domain_compounds.csv")
    tiers = {iso: compounds[f"{iso}_domain_tier"].value_counts().to_dict() for iso in ISOFORMS}
    text = "; ".join(f"{iso}: " + ", ".join(f"{n} {t}" for t, n in tiers[iso].items()) for iso in tiers)
    return ("How well the final stack ranks, by the similarity of each held-out compound to the rows its fold was trained on (ECFP4 Tanimoto), measured on the outer folds of the nested validation "
            "at the first confidence cut: AUC with its 95% interval from a bootstrap over chemical clusters. A bin is supported when the lower end of its interval is above 0.5. "
            f"The blinded compounds fall in these tiers: {text}. CYP3A4 is supported everywhere; CYP2D6 ranks usefully only where the nearest training compound is at least moderately similar.")


@figure("ml_applicability_domain", "ml", "Where the final stack can be trusted: reliability by similarity", ad_caption,
        data=(f"{FINAL_TDI}/applicability_domain_bins.csv", f"{FINAL_TDI}/applicability_domain_compounds.csv"), section="8.7")
def applicability_domain(theme_name):
    th = apply(theme_name)
    frame = bins_frame()
    frame = frame[frame["z0"] == 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for ax, iso, color in zip(axes, ISOFORMS, (th["blue"], th["orange"]), strict=True):
        part = frame[frame["isoform"] == iso].reset_index(drop=True)
        x = np.arange(len(part))
        ok = part["auc"].notna()
        ax.errorbar(x[ok], part.loc[ok, "auc"], yerr=[part.loc[ok, "auc"] - part.loc[ok, "auc_low"], part.loc[ok, "auc_high"] - part.loc[ok, "auc"]], fmt="o", color=color, capsize=3)
        ax.axhline(0.5, color=th["muted"], ls="--", lw=1)
        for i, row in part.iterrows():
            ax.text(i, 0.455, f"n = {int(row['compounds'])}" + ("" if ok[i] else "\n(too few)"), ha="center", fontsize=7.5, color=th["muted"])
        ax.set_xticks(x, part["bin"], fontsize=8)
        ax.set(title=iso, xlabel="nearest-training similarity of the held-out compound", ylim=(0.42, 1.0))
        tidy(ax, th)
    axes[0].set_ylabel("AUC (95% interval over chemical clusters)")
    fig.tight_layout()
    return fig


# -- the choice of the call rate -------------------------------------------------------------------------------------------------------------------

def call_caption() -> str:
    parts = []
    for iso in ISOFORMS:
        table = pd.read_csv(ROOT / FINAL_TDI / f"{iso}_call_rate_choice.csv")
        best = table.groupby("multiplier")["mean_outer_mcc"].mean().idxmax()
        parts.append(f"{iso} {best:.1f} times the prevalence")
    return ("The weighted MCC on the outer folds as a function of the number of positive calls, expressed as a multiple of the scored population's prevalence, at the two confidence cuts. "
            f"The call rate is chosen by the highest mean over the two cuts: {'; '.join(parts)}. The rule transfers a rate to the test file and never a score; the prevalence it multiplies is the "
            "aggregate recovered from the public board (see the previous figure and limitation L13).")


@figure("ml_call_rate", "ml", "How many compounds to call positive", call_caption, data=(f"{FINAL_TDI}/<isoform>_call_rate_choice.csv",), section="8.4")
def call_rate(theme_name):
    th = apply(theme_name)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, iso in zip(axes, ISOFORMS, strict=True):
        table = pd.read_csv(ROOT / FINAL_TDI / f"{iso}_call_rate_choice.csv")
        for z0, color, marker in ((1.0, th["blue"], "o"), (1.96, th["orange"], "s")):
            part = table[table["z0"] == z0]
            ax.plot(part["multiplier"], part["mean_outer_mcc"], marker + "-", color=color, label=f"z0 = {z0}")
        best = table.groupby("multiplier")["mean_outer_mcc"].mean().idxmax()
        ax.axvline(best, color=th["muted"], ls=":", lw=1)
        ax.set(title=iso, xlabel="positive calls as a multiple of the prevalence")
        tidy(ax, th)
    axes[0].set_ylabel("weighted MCC on the outer folds")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# -- all twenty submitted structures ---------------------------------------------------------------------------------------------------------------

def gallery_rows() -> list[dict]:
    import json

    sys.path[:0] = [str(ROOT / "docs/card")]
    import build_card

    recipe = json.loads((ROOT / "data/structure_track/outputs/cyp3a4/final/recipe.json").read_text())
    return [{"name": r["ligand"], "donor": r["type_ii_donor"], "distance": build_card.fe_n_distance(ROOT / FINAL_PDB / f"{r['ligand']}.pdb")} for r in sorted(recipe["ligands"], key=lambda r: r["ligand"])]


def gallery_caption() -> str:
    rows = gallery_rows()
    donors = [r for r in rows if r["donor"]]
    return (f"All {len(rows)} structures of the submitted zip, the pocket of each predicted complex with the heme (orange) and the ligand (teal). {len(donors)} ligands have a Type II donor nitrogen: the iron-nitrogen "
            f"restraint was used for them and the nearest nitrogen is {min(r['distance'] for r in donors):.2f} to {max(r['distance'] for r in donors):.2f} angstrom from the iron. The other {len(rows) - len(donors)} "
            "ligands have no such donor and were run without the restraint; their nearest nitrogen is far from the iron, as it should be for ligands that do not bind it. All twenty pass the 22 PoseBusters checks "
            "under the organisers' own call. Nothing was left out of this picture.")


@figure("struct_all20_gallery", "structure", "Every submitted structure", gallery_caption, data=(FINAL_PDB, "data/structure_track/outputs/cyp3a4/final/recipe.json"), section="11.10")
def all20_gallery(theme_name):
    th = apply(theme_name)
    from gen.pymol_scene import predicted_pocket

    rows = gallery_rows()
    fig, axes = plt.subplots(4, 5, figsize=(13, 9.6))
    for ax, row in zip(axes.flat, rows, strict=True):
        scene = predicted_pocket(ROOT / FINAL_PDB / f"{row['name']}.pdb", theme_name, (576, 400), key=f"card_tile_{row['name']}")
        ax.imshow(plt.imread(scene.png))
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(th["teal"] if row["donor"] else th["muted"])
            side.set_linewidth(2 if row["donor"] else 1)
        ax.set_title(f"{row['name']}\n{'Type II donor' if row['donor'] else 'no donor'}, nearest N to iron {row['distance']:.2f} A", fontsize=8)
    fig.tight_layout()
    return fig


# -- the ledger as a map of what was explored ------------------------------------------------------------------------------------------------------

KINDS = {  # the family of a ledger row (the stem of its log) -> the kind of idea it tested
    "data usage": ("aux_screen", "aux_emax", "aux_pxr", "aux_ext_cont", "aux_ext_tox", "aux_ext_adme", "aux_nodirect", "test_likeness"),
    "representation and physics": ("embeddings", "monroe_embeddings", "physics_experiment", "physics_compact", "physics_tierb", "population_physics", "aux_basic_centre", "aux_atomjepa", "aux_atomjepa_head"),
    "learners and stacking": ("aux_tabicl", "aux_tabpfn", "aux_tabicl_monroe", "aux_tabicl_chemeleon", "aux_tabpfn_chemeleon", "aux_mlp", "aux_mlp_aux", "aux_chemprop", "aux_chemeleon", "aux_neural", "nonneg_stack"),
    "label process and selection": ("selection_aware_v2", "hedged_blend", "population_screen", "arms_experiment", "committee_label_model", "aux_gate_shift", "xiso_transfer", "pair_transfer",
                                    "pair_transfer_physics", "mmp_transfer"),
    "structure levers": ("structure_gates", "selection_validation", "pose_awareness", "series_consensus", "heme_frame_prior", "ei_sampling", "restraint_expansion"),
    "replicates": ("replicate_verdict",),
}


def ledger_by_kind() -> pd.DataFrame:
    from cypmode.audit.method_report import gate_rows

    kind_of = {stem: kind for kind, stems in KINDS.items() for stem in stems}
    rows = []
    for log, name, verdict in gate_rows():
        if "exploratory" in name or log.startswith("permutation_control"):
            continue
        stem = re.sub(r"(_f10)?\.log$", "", re.sub(r"_(CYP3A4|CYP2D6)(?=(_f10)?\.log)", "", log))
        rows.append({"stem": stem, "kind": kind_of.get(stem, "other"), "verdict": verdict})
    return pd.DataFrame(rows)


def map_caption() -> str:
    frame = ledger_by_kind()
    kinds = frame.groupby("kind").agg(verdicts=("verdict", "size"), met=("verdict", lambda v: int((v == "MET").sum())), families=("stem", "nunique"))
    parts = "; ".join(f"{k}: {int(r.verdicts)} verdicts in {int(r.families)} {'family' if int(r.families) == 1 else 'families'}, {int(r.met)} met" for k, r in kinds.iterrows())
    return (f"Every pre-registered verdict of the ledger, grouped by the kind of idea it tested: {len(frame)} verdicts in {frame['stem'].nunique()} families ({parts}). The search was wide before it was deep: "
            "new data (the screen, the Emax, the induction readings, public panels), new representations (pretrained embeddings, physics at four levels), new learners, a model of how the labels came to be, "
            "and levers of the structure recipe, each judged by the rule written before its result. Ideas that passed are green; the grey ones stay in the ledger as results. "
            "No reinforcement-learning or evolutionary search is used: the exploration is a designed, auditable programme of experiments, not an optimiser.")


@figure("ml_exploration_map", "ml", "What was explored: the ledger by kind of idea", map_caption, data=(f"{LOGS}/<experiment>_<isoform>.log",), section="9.9")
def exploration_map(theme_name):
    th = apply(theme_name)
    frame = ledger_by_kind()
    order = [k for k in list(KINDS) + ["other"] if k in set(frame["kind"])]
    fig, ax = plt.subplots(figsize=(9.5, 0.7 * len(order) + 1.6))
    for i, kind in enumerate(order):
        part = frame[frame["kind"] == kind]
        met, not_met = int((part["verdict"] == "MET").sum()), int((part["verdict"] != "MET").sum())
        ax.barh(i, not_met, color=th["muted"], alpha=0.75)
        ax.barh(i, met, left=not_met, color=th["green"])
        ax.text(met + not_met + 0.6, i, f"{part['stem'].nunique()} {'family' if part['stem'].nunique() == 1 else 'families'}, {met} met", va="center", fontsize=9)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlim(0, max(len(frame[frame["kind"] == k]) for k in order) * 1.35)
    ax.set_xlabel("pre-registered verdicts (grey: not met, green: met)")
    tidy(ax, th)
    fig.tight_layout()
    return fig


# -- the exploration-exploitation dial -------------------------------------------------------------------------------------------------------------

def softmax_curve(iso: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/tdi_track/outputs/comparisons" / f"softmax_curve_{iso}.csv")


def dial_caption() -> str:
    parts = []
    for iso in ISOFORMS:
        frame = softmax_curve(iso)
        fixed = frame[(frame["z0"] == 1.0) & frame["selector"].str.startswith("softmax_T")].copy()
        fixed["T"] = fixed["selector"].str.removeprefix("softmax_T").astype(float)
        fixed = fixed.sort_values("T")
        stack = float(frame[(frame["z0"] == 1.0) & (frame["selector"] == "stack_logit")]["outer_auc"].iloc[0])
        parts.append(f"{iso}: {fixed['outer_auc'].iloc[0]:.3f} at the best-model end, {fixed['outer_auc'].iloc[-1]:.3f} at the committee end, {stack:.3f} for the logistic blend (z0 = 1.0)")
    return ("The outer-fold AUC of a softmax blend of the cached candidates of the default stack as the temperature T moves from the best single model (T small, exploitation) to an equal-weight "
            "committee (T large, exploration); the dashed line is the logistic blend that the final stack uses, the open marker the blend whose T is chosen by an inner cross-validation. " + "; ".join(parts) + ". "
            "On CYP3A4 the curve is flat and the learned weights are worth a little more than any point of it; on CYP2D6 it rises towards the committee, and the inner choice, which favours small T, "
            "over-trusts the best candidates. The pre-registered test of the softmax selector against the logistic blend is not met on either enzyme.")


@figure("ml_exploration_exploitation", "ml", "From the best model to a committee: the exploration-exploitation dial of the blend", dial_caption, data=("data/tdi_track/outputs/comparisons/softmax_curve_<isoform>.csv",), section="8.2")
def exploration_exploitation(theme_name):
    th = apply(theme_name)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, iso in zip(axes, ISOFORMS, strict=True):
        frame = softmax_curve(iso)
        fixed = frame[frame["selector"].str.startswith("softmax_T")].copy()
        fixed["T"] = fixed["selector"].str.removeprefix("softmax_T").astype(float)
        temperatures = sorted(fixed["T"].unique())
        x = np.arange(len(temperatures))
        for z0, color, marker in ((1.0, th["blue"], "o"), (1.96, th["orange"], "s")):
            part = fixed[fixed["z0"] == z0].set_index("T").loc[temperatures]
            ax.errorbar(x, part["outer_auc"], yerr=part["outer_auc_sd"] / np.sqrt(part["folds"]), fmt=marker + "-", color=color, capsize=2, label=f"z0 = {z0}")
            stack = float(frame[(frame["z0"] == z0) & (frame["selector"] == "stack_logit")]["outer_auc"].iloc[0])
            chosen = float(frame[(frame["z0"] == z0) & (frame["selector"] == "softmax")]["outer_auc"].iloc[0])
            ax.axhline(stack, color=color, ls="--", lw=1)
            ax.plot(len(temperatures) - 0.5, chosen, marker, color=color, mfc="none", ms=9)
        ax.set_xticks(x, ["inf" if np.isinf(t) else f"{t:g}" for t in temperatures], fontsize=8)
        ax.set_title(iso)
        ax.set_xlabel("temperature T: best model (left) to committee (inf)", fontsize=9)
        tidy(ax, th)
    axes[0].set_ylabel("outer-fold AUC (mean, bar = standard error)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig
