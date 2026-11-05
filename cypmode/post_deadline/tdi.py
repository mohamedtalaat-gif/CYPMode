# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The questions of ``docs/post_deadline/PLAN.md`` about the time-dependent inhibition track: the scoring gate (question 0), the two candidates (1), the label rule and the scored set (2) and the threshold (3).

The scoring is the organisers' own: ``score_activity_predictions``, ``add_macro_endpoint`` and ``average_bootstrap_results_by_endpoint`` of the evaluation code that ``released.py`` fetched at a fixed
commit, called without change. Their bootstrap has a fixed seed, so the board value of a file is reproduced to the digit when the labels and the predictions are the same. Each function takes the
frames it works on, so the tests run it on small invented tables; ``python -m cypmode.post_deadline.tdi`` reads the released label file and the submitted candidates, answers the questions in the order of
the plan, writes the tables to ``data/post_deadline/outputs`` and stops after question 0 when its gate fails.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from cypmode.challenge.label_confidence import confidence_z
from cypmode.challenge.tdi_targets import (
    TDI_ISOFORMS,
    direct_column,
    label_column,
    label_from_shift,
    shift_target,
    tdi_column,
)
from cypmode.post_deadline.released import DEST, ROOT

OUT = ROOT / "data/post_deadline/outputs"
TRUTH = DEST / "dataset/cyp-challenge-TEST-UNBLINDED_TDI.csv"
CANDIDATES = {"A": ROOT / "submissions/tdi_track/tdi_track_candidate_A_hits_only.csv", "B": ROOT / "submissions/tdi_track/tdi_track_candidate_B_selection_aware.csv"}
FINAL = ROOT / "data/tdi_track/outputs/final"
ENDPOINTS = [label_column(iso) for iso in ("CYP2D6", "CYP3A4")]

BOARD_MACRO_MCC = 0.3722973127856141  # the final leaderboard of the organisers, mean over 1,000 bootstrap samples
BOARD_MACRO_MCC_STD = 0.037791222850103  # and the standard deviation over the same samples
BOARD_TOLERANCE = 0.01  # the gate of question 0 as registered
EXACT_TOLERANCE = 1e-6  # the identification of the submitted file after the first run (see the deviations of the plan)
SEED = 42
RESAMPLES = 1000
Z_CUTS = (1.0, 1.96)  # the two cut-offs of the confidence rule in the final build


def organisers():
    """The evaluation code of the organisers (the module of the scoring and its configuration), imported from the files that were fetched."""
    path = str(DEST / "organisers")
    if path not in sys.path:
        sys.path.insert(0, path)
    from loguru import logger

    logger.remove()
    import evaluation.evaluate_predictions as evaluate
    from evaluation import config

    return evaluate, config


def board_scores(predictions: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """The mean and the standard deviation over the bootstrap samples of every metric, per isoform and for the macro average, as the leaderboard of the organisers forms them."""
    evaluate, config = organisers()
    results = evaluate.score_activity_predictions(predictions, truth, ENDPOINTS)
    with_macro = evaluate.add_macro_endpoint(results, ENDPOINTS, config.CLASSIFICATION_METRICS)
    return evaluate.average_bootstrap_results_by_endpoint(with_macro)


def gate(scores: dict[str, pd.DataFrame], board: float = BOARD_MACRO_MCC, tolerance: float = BOARD_TOLERANCE, board_std: float = BOARD_MACRO_MCC_STD) -> dict:
    """Question 0: which candidate is the file that the board scored.

    The rule that the plan registered takes the candidate whose macro MCC lies within the tolerance of the board value and passes when exactly one does. The fixed seed of the organisers' bootstrap means that
    the submitted file reproduces the mean and the standard deviation of the board to the last digits, and a candidate that differs in a few calls does not. The identification that the analysis uses is the
    exact match of both numbers; the registered rule is kept in the result next to it.
    """
    macro = {name: float(frame.loc["MA", "MCC_mean"]) for name, frame in scores.items()}
    near = sorted(name for name, value in macro.items() if abs(value - board) <= tolerance)
    exact = sorted(name for name, frame in scores.items() if abs(frame.loc["MA", "MCC_mean"] - board) <= EXACT_TOLERANCE and abs(frame.loc["MA", "MCC_std"] - board_std) <= EXACT_TOLERANCE)
    return {
        "board": board,
        "macro_mcc": macro,
        "registered": {"tolerance": tolerance, "within_tolerance": near, "passed": len(near) == 1},
        "exact": {"tolerance": EXACT_TOLERANCE, "match": exact, "difference_of_the_mean": {name: macro[name] - board for name in macro}},
        "passed": len(exact) == 1,
        "submitted": exact[0] if len(exact) == 1 else None,
    }


def _scored(truth: pd.DataFrame, endpoint: str) -> np.ndarray:
    return ~pd.isna(truth[endpoint]).to_numpy()


def paired_difference(first: pd.DataFrame, second: pd.DataFrame, truth: pd.DataFrame, resamples: int = RESAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Question 1: the MCC of `second` minus the MCC of `first` per isoform and for the macro average, over a paired bootstrap of the scored compounds.

    Both files are scored on the same resampled compounds. The macro value of a sample is the mean of the two isoform values of that sample, as in the organisers' scoring. The interval is the 2.5 and 97.5
    percentiles of the differences; the verdict is that the second file is better when the whole interval lies above zero, the first when it lies below zero, and no difference is shown otherwise.
    """
    rng = np.random.default_rng(seed)
    merged = truth[["Molecule_Name", *ENDPOINTS]].merge(first[["Molecule_Name", *ENDPOINTS]], on="Molecule_Name", suffixes=("", "_a")).merge(second[["Molecule_Name", *ENDPOINTS]].rename(columns={e: f"{e}_b" for e in ENDPOINTS}), on="Molecule_Name")
    per_sample = {}
    for endpoint in ENDPOINTS:
        keep = _scored(merged, endpoint)
        y, a, b = merged.loc[keep, endpoint].astype(bool).to_numpy(), merged.loc[keep, f"{endpoint}_a"].astype(bool).to_numpy(), merged.loc[keep, f"{endpoint}_b"].astype(bool).to_numpy()
        indices = rng.integers(0, len(y), size=(resamples, len(y)))
        per_sample[endpoint] = np.array([matthews_corrcoef(y[i], b[i]) - matthews_corrcoef(y[i], a[i]) for i in indices])
    per_sample["macro"] = np.mean([per_sample[e] for e in ENDPOINTS], axis=0)
    rows = []
    for name, differences in per_sample.items():
        low, high = np.percentile(differences, [2.5, 97.5])
        verdict = "second better" if low > 0 else "first better" if high < 0 else "no difference shown"
        rows.append({"endpoint": name, "mean_difference": float(differences.mean()), "low": float(low), "high": float(high), "verdict": verdict})
    return pd.DataFrame(rows)


def call_rates(files: dict[str, pd.DataFrame], truth: pd.DataFrame) -> pd.DataFrame:
    """Question 1, the rates: the share of positive calls of each file and the share of true positives, per isoform, among the scored compounds."""
    rows = []
    for endpoint in ENDPOINTS:
        keep = _scored(truth, endpoint)
        names = truth.loc[keep, "Molecule_Name"]
        row = {"endpoint": endpoint, "scored": int(keep.sum()), "true_positive_rate": float(truth.loc[keep, endpoint].astype(bool).mean())}
        for name, frame in files.items():
            called = frame.set_index("Molecule_Name").loc[names, endpoint].astype(bool)
            row[f"called_{name}"] = float(called.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def rule_agreement(truth: pd.DataFrame) -> pd.DataFrame:
    """Question 2a: the label that the rule of the project gives from the two measured pIC50 values against the organisers' label, per isoform.

    ``disagree`` counts the labelled compounds for which the rule gives the other label; ``not_computable`` the labelled compounds for which a value of the rule is missing; ``computable_unlabelled`` the compounds
    that have both values but no label.
    """
    rows = []
    for iso in TDI_ISOFORMS:
        label = truth[label_column(iso)]
        rule = label_from_shift(shift_target(truth[direct_column(iso)], truth[tdi_column(iso)]))
        labelled, computable = ~pd.isna(label).to_numpy(), ~np.isnan(rule)
        both = labelled & computable
        rows.append(
            {
                "isoform": iso,
                "labelled": int(labelled.sum()),
                "agree": int((label[both].astype(float).to_numpy() == rule[both]).sum()),
                "disagree": int((label[both].astype(float).to_numpy() != rule[both]).sum()),
                "not_computable": int((labelled & ~computable).sum()),
                "computable_unlabelled": int((~labelled & computable).sum()),
            }
        )
    return pd.DataFrame(rows)


def rule_disagreements(truth: pd.DataFrame) -> pd.DataFrame:
    """Question 2a, the list: every labelled compound for which the rule gives the other label."""
    parts = []
    for iso in TDI_ISOFORMS:
        rule = label_from_shift(shift_target(truth[direct_column(iso)], truth[tdi_column(iso)]))
        label = truth[label_column(iso)]
        wrong = (~pd.isna(label).to_numpy()) & ~np.isnan(rule) & (label.astype(float).to_numpy() != rule)
        part = truth.loc[wrong, ["Molecule_Name", direct_column(iso), tdi_column(iso)]].rename(columns={direct_column(iso): "direct_pIC50", tdi_column(iso): "tdi_pIC50"})
        part["isoform"], part["organisers_label"], part["rule_label"] = iso, label[wrong].astype(bool).to_numpy(), rule[wrong].astype(bool)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def scored_set_agreement(truth: pd.DataFrame, z_cuts: tuple[float, ...] = Z_CUTS) -> pd.DataFrame:
    """Question 2b: the compounds that carry a label in the released file against the compounds that the confidence rule of the project keeps, per isoform and per cut-off."""
    rows = []
    for iso in TDI_ISOFORMS:
        organisers_set = ~pd.isna(truth[label_column(iso)]).to_numpy()
        z = confidence_z(truth, iso)
        for z0 in z_cuts:
            ours = np.nan_to_num(z, nan=0.0) >= z0
            rows.append({"isoform": iso, "z0": z0, "both": int((organisers_set & ours).sum()), "only_organisers": int((organisers_set & ~ours).sum()), "only_project": int((~organisers_set & ours).sum()), "neither": int((~organisers_set & ~ours).sum())})
    return pd.DataFrame(rows)


def threshold_ceiling(scores: pd.DataFrame, truth: pd.DataFrame, endpoint: str) -> dict:
    """Question 3: the MCC of a file at the threshold that was submitted and the highest MCC that any threshold on its score would give on the released labels (a ceiling that uses the labels).

    `scores` holds ``Molecule_Name``, ``score`` and ``call``; a compound is called positive when its score is above the threshold, and the submitted threshold is read from the calls.
    """
    keep = _scored(truth, endpoint)
    joined = truth.loc[keep, ["Molecule_Name", endpoint]].merge(scores, on="Molecule_Name")
    y, score, call = joined[endpoint].astype(bool).to_numpy(), joined["score"].to_numpy(), joined["call"].astype(bool).to_numpy()
    best = max(((matthews_corrcoef(y, score > cut), float(np.mean(score > cut))) for cut in np.unique(score)), key=lambda pair: pair[0])
    return {"endpoint": endpoint, "scored": len(y), "mcc_submitted": float(matthews_corrcoef(y, call)), "call_rate_submitted": float(call.mean()), "mcc_ceiling": float(best[0]), "call_rate_at_ceiling": best[1]}


def _read_candidates() -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(path) for name, path in CANDIDATES.items()}


def run(out: Path = OUT) -> int:
    out.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(TRUTH)
    files = _read_candidates()
    scores = {name: board_scores(frame[["Molecule_Name", *ENDPOINTS]], truth) for name, frame in files.items()}
    for name, frame in scores.items():
        frame.to_csv(out / f"tdi_board_scores_{name}.csv")
    verdict = gate(scores)
    (out / "tdi_gate.json").write_text(json.dumps(verdict, indent=1) + "\n")
    registered = verdict["registered"]
    print(f"QUESTION 0 (TDI): macro MCC of A {verdict['macro_mcc']['A']:.7f}, of B {verdict['macro_mcc']['B']:.7f}; board {verdict['board']:.7f}")
    print(f"  as registered (within {registered['tolerance']}): {registered['within_tolerance']}, {'MET' if registered['passed'] else 'NOT MET'}")
    print(f"  exact match of the mean and the standard deviation (within {verdict['exact']['tolerance']}): {verdict['exact']['match']}, GATE: {'MET' if verdict['passed'] else 'NOT MET'}")
    if not verdict["passed"]:
        return 1
    print(f"the submitted file is candidate {verdict['submitted']}")
    difference = paired_difference(files["A"], files["B"], truth)
    difference.to_csv(out / "tdi_question1_difference.csv", index=False)
    call_rates(files, truth).to_csv(out / "tdi_question1_rates.csv", index=False)
    print("QUESTION 1: B minus A\n" + difference.to_string(index=False))
    rule_agreement(truth).to_csv(out / "tdi_question2a_agreement.csv", index=False)
    rule_disagreements(truth).to_csv(out / "tdi_question2a_disagreements.csv", index=False)
    scored_set_agreement(truth).to_csv(out / "tdi_question2b_scored_set.csv", index=False)
    submitted = verdict["submitted"]
    sources = {"CYP3A4_is_TDI": FINAL / "CYP3A4_hits_only_scores.csv", "CYP2D6_is_TDI": FINAL / ("CYP2D6_hits_only_scores.csv" if submitted == "A" else "CYP2D6_selection_aware_scores.csv")}
    ceilings = pd.DataFrame([threshold_ceiling(pd.read_csv(path)[["Molecule_Name", "score", "call"]], truth, endpoint) for endpoint, path in sources.items()])
    ceilings.to_csv(out / "tdi_question3_ceiling.csv", index=False)
    print("QUESTION 3\n" + ceilings.to_string(index=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    return run()


if __name__ == "__main__":
    sys.exit(main())
