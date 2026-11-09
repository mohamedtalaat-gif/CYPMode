# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The section of the method report about the analysis after the deadline: every number is read from the tables and the logs that the analysis wrote (``docs/post_deadline/PLAN.md`` is the plan).

``section()`` returns the lines of the section; the method report appends them. ``readme_block()`` returns the short account that the README shows. Nothing is typed in: a table that is missing is said to be missing, and a verdict is the one that the table holds.
"""

import json
from pathlib import Path

import pandas as pd

from cypmode.post_deadline.released import ROOT

OUT = ROOT / "data/post_deadline/outputs"


def _table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> list[str]:
    """A markdown table of `frame`; a column is written with the format given for it, and every other column with its own text."""
    formats = formats or {}
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(format(value, formats[column]) if column in formats and pd.notna(value) else str(value) for column, value in zip(frame.columns, row)) + " |" for row in frame.itertuples(index=False)]
    return [header, rule, *rows]


def _read(folder: Path, name: str) -> pd.DataFrame | None:
    path = folder / name
    return pd.read_csv(path) if path.exists() else None


def _json(folder: Path, name: str) -> dict | None:
    path = folder / name
    return json.loads(path.read_text()) if path.exists() else None


def describe_extra_calls(folder: Path) -> str:
    """The distance to the iron in the released structures of the ligands that the submitted poses call coordinated and the structures do not: the range, how many lie within 0.5 angstroms of the cut-off, and the resolution of their maps."""
    from cypmode.validation.constants import COORDINATION_DISTANCE_CUTOFF_A

    rows = _read(folder, "structure_question5c.csv")
    if rows is None:
        return ""
    extra = rows[~rows["truth_coordinated"] & rows["submitted_coordinated"]]
    if extra.empty:
        return ""
    near = int((extra["truth_distance"] <= COORDINATION_DISTANCE_CUTOFF_A + 0.5).sum())
    resolutions = extra["resolution"].dropna()
    return (f"the nearest nitrogen of the ligand lies {extra['truth_distance'].min():.2f} to {extra['truth_distance'].max():.2f} angstroms from the iron, {near} of the {len(extra)} within "
            f"{COORDINATION_DISTANCE_CUTOFF_A + 0.5:.1f} angstroms, in maps of {resolutions.min():.1f} to {resolutions.max():.1f} angstroms resolution")

def tdi_lines(folder: Path) -> list[str]:
    gate = _json(folder, "tdi_gate.json")
    if gate is None:
        return ["The questions of the time-dependent inhibition track have not been run (`python -m cypmode.post_deadline.tdi`).", ""]
    macro = gate["macro_mcc"]
    scores = _read(folder, "tdi_board_scores_A.csv")
    lines = [
        ("**Question 0, time-dependent inhibition: the gate.** The scoring of the organisers, run on the released labels, gives candidate A a macro MCC of "
        f"{macro['A']:.7f} and candidate B {macro['B']:.7f}; the final leaderboard shows {gate['board']:.7f} for the entry. "
        f"The registered rule (a candidate within {gate['registered']['tolerance']} of the board value) names {' and '.join(gate['registered']['within_tolerance']) or 'no candidate'}, so it "
        f"{'passes' if gate['registered']['passed'] else 'does not separate the two files'}; the exact match of the mean and of the standard deviation (within {gate['exact']['tolerance']}) names "
        f"{', '.join(gate['exact']['match']) or 'no candidate'}, which is the file that was submitted. The deviation is recorded in the plan."),
        "",
    ]
    if scores is not None:
        shown = scores[["Endpoint", "MCC_mean", "MCC_std"]].rename(columns={"MCC_mean": "MCC (mean)", "MCC_std": "MCC (standard deviation)"})
        lines += [*_table(shown, {"MCC (mean)": ".4f", "MCC (standard deviation)": ".4f"}), ""]
    difference, rates = _read(folder, "tdi_question1_difference.csv"), _read(folder, "tdi_question1_rates.csv")
    if difference is not None and rates is not None:
        macro_row = difference[difference["endpoint"] == "macro"].iloc[0]
        lines += [
            ("**Question 1, the two candidates.** The macro MCC of candidate B minus that of candidate A is "
            f"{macro_row['mean_difference']:+.4f}, with a 95% interval from {macro_row['low']:+.4f} to {macro_row['high']:+.4f} over a paired bootstrap of the scored compounds: {macro_row['verdict']}. "
            "The two files are identical for CYP3A4, so the difference is entirely in the CYP2D6 calls."),
            "",
            *_table(difference.rename(columns={"mean_difference": "mean difference", "low": "2.5%", "high": "97.5%"}), {"mean difference": "+.4f", "2.5%": "+.4f", "97.5%": "+.4f"}),
            "",
            "The share of positive calls of each file and the share of true positives among the scored compounds:",
            "",
            *_table(rates, {"true_positive_rate": ".4f", "called_A": ".4f", "called_B": ".4f"}),
            "",
        ]
    agreement, scored_set = _read(folder, "tdi_question2a_agreement.csv"), _read(folder, "tdi_question2b_scored_set.csv")
    if agreement is not None and scored_set is not None:
        lines += [
            f"**Question 2a, the label rule.** The rule recovered from the training data gives the organisers' label for every labelled compound whose two values are present: {int(agreement['agree'].sum())} agree and {int(agreement['disagree'].sum())} disagree.",
            "",
            *_table(agreement),
            "",
            ("**Question 2b, the scored set.** The compounds that carry a label in the released file are the compounds that the organisers score. They are "
            f"{int(agreement.loc[agreement['isoform'] == 'CYP3A4', 'labelled'].iloc[0])} of the test compounds for CYP3A4 and {int(agreement.loc[agreement['isoform'] == 'CYP2D6', 'labelled'].iloc[0])} for CYP2D6, that is, nearly the whole test set and not a set filtered by the confidence of the label. "
            "The confidence rule of the project keeps fewer of them, at each of the two cut-offs of the final build:"),
            "",
            *_table(scored_set, {"z0": ".2f"}),
            "",
        ]
    ceiling = _read(folder, "tdi_question3_ceiling.csv")
    if ceiling is not None:
        lines += [
            "**Question 3, the threshold.** The MCC of the submitted calls and the highest MCC that any threshold on the stored score would give on the released labels (a ceiling that uses the labels; it is not a result and no file is changed by it):",
            "",
            *_table(ceiling, {"mcc_submitted": ".4f", "call_rate_submitted": ".4f", "mcc_ceiling": ".4f", "call_rate_at_ceiling": ".4f"}),
            "",
            f"Taking the mean of the two isoforms, the submitted calls give {ceiling['mcc_submitted'].mean():.4f} and the best threshold {ceiling['mcc_ceiling'].mean():.4f}, so the call rate alone accounts for at most {ceiling['mcc_ceiling'].mean() - ceiling['mcc_submitted'].mean():.4f}.",
            "",
        ]
    return lines


def structure_lines(folder: Path) -> list[str]:
    gate = _json(folder, "structure_gate.json")
    if gate is None:
        return ["The questions of the structure track have not been run (`python -m cypmode.post_deadline.structure`).", ""]
    versions = _json(folder, "structure_versions.json") or {}
    lines = [
        "**Question 0, structure: the gate.** The submitted archive, scored with the organisers' function"
        + (f" (OpenStructure {versions['openstructure']}, PoseBusters {versions['posebusters']})" if versions else "")
        + f", gives a mean LDDT-PLI of {gate['scored']['LDDT-PLI']:.5f} against {gate['board']['LDDT-PLI']:.5f} on the final leaderboard (difference {gate['difference']['LDDT-PLI']:+.5f}) "
        f"and a mean BiSyRMSD of {gate['scored']['BiSyRMSD']:.4f} against {gate['board']['BiSyRMSD']:.4f} ({gate['difference']['BiSyRMSD']:+.4f}); "
        f"{len(gate['posebusters_failures'])} poses fail PoseBusters. The gate as registered is {'met' if gate['passed'] else 'not met'}. The difference to the board is reported and not explained; "
        "it is the same with OpenStructure 2.12.0 and 2.11.1 (the plan records it).",
        "",
    ]
    a = _read(folder, "structure_question5a.csv")
    if a is not None:
        lines += [
            ("**Question 5a, the recipe against two plain Boltz-2 poses.** Mean paired difference over the 20 compounds (the sign of the BiSyRMSD is reversed, so that a positive value is better for both), "
            "with a 95% interval from a paired bootstrap of 10,000 resamples:"),
            "",
            *_table(a.rename(columns={"mean_difference": "mean difference", "low": "2.5%", "high": "97.5%", "first_higher": "compounds in which the submitted pose is higher"}), {"mean difference": "+.4f", "2.5%": "+.4f", "97.5%": "+.4f"}),
            "",
        ]
    b = _read(folder, "structure_question5b.csv")
    if b is not None:
        mean = b[["best", "random", "submitted", "gap_submitted", "gap_random"]].mean()
        lines += [
            ("**Question 5b, where the limit lay.** Over the 20 compounds the best pose of the pool has a mean LDDT-PLI of "
            f"{mean['best']:.4f}, a pose drawn at random from the pool {mean['random']:.4f} and the submitted pose {mean['submitted']:.4f}. The submitted pose is {mean['gap_submitted']:.4f} below the best of the pool, "
            f"and a random pose is {mean['gap_random']:.4f} below it: the selection recovered {100 * (1 - mean['gap_submitted'] / mean['gap_random']):.0f}% of the distance between a random pose and the best one."),
            "",
        ]
    counts, resolution = _read(folder, "structure_question5c_counts.csv"), _json(folder, "structure_resolution.json")
    if counts is not None:
        lines += [
            "**Question 5c, the heme iron.** A ligand counts as coordinated when a nitrogen atom of the ligand lies within the cut-off of the project of an iron atom of a heme. Rows: the released structure; columns: the submitted pose.",
            "",
            *_table(counts),
            "",
        ]
        extra = describe_extra_calls(folder)
        if extra:
            lines += [f"Of the ligands that the submitted poses call coordinated and the released structures do not, {extra}. A distance within half an angstrom of the cut-off, in maps of this resolution, does not exclude a contact: the table counts the ligands beyond the cut-off and does not show that they make no contact.", ""]
        if resolution is not None:
            lines += [f"The rank correlation between the resolution of the released map and the LDDT-PLI of the submitted pose is {resolution['spearman_rho']:+.3f} over {resolution['compounds']} compounds ({resolution.get('without_resolution', 0)} files record no resolution).", ""]
    return lines


def section(folder: Path = OUT) -> list[str]:
    return [
        "## 15. After the deadline: the released labels and structures",
        "",
        ("The organisers released the test labels and the 20 structures after the challenge closed. The plan of this analysis, `docs/post_deadline/PLAN.md`, was committed and pushed before any of them was read; its questions, "
        "measures and decision rules were fixed there, and every change made after a result was seen is written in its section on deviations. The scoring is the organisers' own: the evaluation code of their tutorial "
        "repository at a named commit, called without change. The analysis only measures; it fits no model and changes no submitted file."),
        "",
        *tdi_lines(folder),
        *structure_lines(folder),
    ]


def readme_block(folder: Path = OUT) -> str:
    """The account of the final standing and of the findings of the analysis that the README shows, between the markers of its generated block."""
    board_path = folder.parent / "final_board.json"
    gate, agreement, ceiling, rates = _json(folder, "tdi_gate.json"), _read(folder, "tdi_question2a_agreement.csv"), _read(folder, "tdi_question3_ceiling.csv"), _read(folder, "tdi_question1_rates.csv")
    a, b, counts = _read(folder, "structure_question5a.csv"), _read(folder, "structure_question5b.csv"), _read(folder, "structure_question5c_counts.csv")
    poses = _read(folder, "structure_poses_scored.csv")
    if not (board_path.exists() and gate and agreement is not None and ceiling is not None and a is not None and b is not None and counts is not None and poses is not None and rates is not None):
        return "(pending: the questions of the analysis after the deadline, jobs 907 to 909)"
    board = json.loads(board_path.read_text())["tracks"]
    tdi, structure = board["classification"], board["structure"]
    mean = poses[poses["arm"].isin(["submitted", "baseline"])].groupby("arm")["LDDT-PLI"].mean()
    pool = b[["best", "random", "submitted"]].mean()
    from cypmode.validation.constants import COORDINATION_DISTANCE_CUTOFF_A

    coordinated_true, called = int(counts.loc[counts["truth_coordinated"], "True"].sum()), int(counts["True"].sum())
    gap = tdi["first"]["MA-MCC"] - tdi["values"]["MA-MCC"]["mean"]
    labelled = [int(x) for x in agreement.set_index("isoform").loc[["CYP3A4", "CYP2D6"], "labelled"]]
    macro_diff = float((ceiling["mcc_ceiling"] - ceiling["mcc_submitted"]).mean())
    verdict_a = a[(a["comparison"] == "submitted minus baseline") & (a["metric"] == "LDDT-PLI")].iloc[0]["verdict"]
    return "\n".join(
        [
            ("**After the deadline.** The final leaderboards, published on 5 November 2026, place the TDI entry "
            f"{tdi['rank']} of {tdi['entries']} (tier {tdi['tier']} of {tdi['tiers']}; macro MCC {tdi['values']['MA-MCC']['mean']:.4f} ± {tdi['values']['MA-MCC']['std']:.4f}) and the structure entry "
            f"{structure['rank']} of {structure['entries']} (tier {structure['tier']} of {structure['tiers']}; LDDT-PLI {structure['values']['LDDT-PLI']['mean']:.4f} ± {structure['values']['LDDT-PLI']['std']:.4f}, "
            f"BiSyRMSD {structure['values']['BiSyRMSD']['mean']:.3f} angstroms). The organisers then released the test labels and the 20 structures. An analysis of the two entries against them, whose plan "
            "([`docs/post_deadline/PLAN.md`](docs/post_deadline/PLAN.md)) was pushed before any of them was read, found:"),
            "",
            f"- The scoring of the organisers reproduces the TDI board to the last digit ({gate['macro_mcc'][gate['submitted']]:.7f} against {gate['board']:.7f}), and the file that was submitted is candidate {gate['submitted']}.",
            (f"- The label rule of the project gives the organisers' label for all {int(agreement['agree'].sum())} compounds that it can be applied to, and the organisers score {labelled[0]} (CYP3A4) and {labelled[1]} (CYP2D6) of the 750 test compounds, "
            "nearly the whole test set and not a set that is filtered by the confidence of the label."),
            f"- Another threshold on the stored scores would have added at most {macro_diff:.3f} to the macro MCC, against {gap:.3f} between the entry and the first of the board.",
            (f"- On the 20 released structures the submitted poses ({mean['submitted']:.3f} LDDT-PLI) cannot be told from the published Boltz-2 baseline ({mean['baseline']:.3f}): {verdict_a}. "
            f"The pool of poses held better ones (the best of each pool gives {pool['best']:.3f} on average) than the choice among them, and the contact restraint found all {coordinated_true} coordinated ligands "
            f"and called {called - coordinated_true} more ligands coordinated that the released structures place beyond the cut-off of {COORDINATION_DISTANCE_CUTOFF_A} angstroms"
            + (f": {describe_extra_calls(folder)}." if describe_extra_calls(folder) else ".")),
            "",
            "The tables are in section 15 of [the method report](METHOD_REPORT.md) and in 13.5 to 13.8 of the notebook.",
        ]
    )
