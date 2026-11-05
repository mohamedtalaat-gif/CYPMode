# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The questions of ``docs/post_deadline/PLAN.md`` about the structure track: the scoring gate (question 0) and the compound by compound comparison (question 5).

The poses are scored with the organisers' functions, which need OpenStructure and PoseBusters in one process; ``structure_worker.py`` runs them in the environment that holds both, and this module, which
runs in the main environment, builds the jobs and does the arithmetic on the scores. The organisers' bootstrap and their averaging of the bootstrap samples are imported from the same evaluation code, so the
value of the submitted archive is formed as the board forms it.

``python -m cypmode.post_deadline.structure`` scores the submitted archive first and stops when the gate of question 0 is not met; it then scores the other poses (the published baseline of the organisers, the
pose that unguided Boltz-2 ranks first and every pose of the pool of each compound) and answers questions 5a, 5b and 5c.
"""

import argparse
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cypmode.post_deadline.released import DEST, ROOT
from cypmode.validation.constants import COORDINATION_DISTANCE_CUTOFF_A

OST_PYTHON = Path.home() / "miniforge3/envs/ost/bin/python"
WORKER = Path(__file__).with_name("structure_worker.py")
OUT = ROOT / "data/post_deadline/outputs"
WORK = ROOT / "data/post_deadline/work"
SUBMISSION = ROOT / "submissions/structure_track/structure_track_submission.zip"
REFERENCES = DEST / "dataset/structures_unblinded"
BASELINE = DEST / "organisers/inputs/boltz-2_cyp3a4_predictions"
OFFICIAL = ROOT / "data/structure_track/inputs/structure_track_test_blinded.csv"
POOLS = ROOT / "data/structure_track/outputs/cyp3a4/final/official_posebusters_pools.csv"
TRAINING = ROOT / "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"

BOARD = {"LDDT-PLI": 0.4133490815684199, "BiSyRMSD": 3.969566656285525, "LDDT-LP": 0.8261987148956736}  # the final leaderboard of the organisers, mean over 1,000 bootstrap samples
TOLERANCE = {"LDDT-PLI": 0.005, "BiSyRMSD": 0.1}  # the gate of question 0
SEED = 42
RESAMPLES = 10_000
METRICS = ("LDDT-PLI", "BiSyRMSD", "LDDT-LP")


def organisers():
    """The evaluation code of the organisers, imported from the files that were fetched (the functions that do not need OpenStructure)."""
    path = str(DEST / "organisers")
    if path not in sys.path:
        sys.path.insert(0, path)
    from loguru import logger

    logger.remove()
    import evaluation.evaluate_predictions as evaluate

    return evaluate


def board_value(per_compound: pd.DataFrame) -> pd.DataFrame:
    """The mean and the standard deviation over 1,000 bootstrap samples of the three metrics of a set of per-compound scores, formed by the organisers' bootstrap and averaging."""
    evaluate = organisers()
    bootstrap = evaluate.bootstrap_structure_metrics(per_compound[list(METRICS)], 1000)
    return evaluate.average_bootstrap_results_by_endpoint(bootstrap)


def gate(summary: pd.DataFrame, board: dict[str, float] = BOARD, tolerance: dict[str, float] = TOLERANCE) -> dict:
    """Question 0: the mean LDDT-PLI and the mean BiSyRMSD of the submitted archive lie within the tolerances of the board values."""
    row = summary.loc["Structure"]
    difference = {metric: float(row[f"{metric}_mean"] - board[metric]) for metric in METRICS}
    return {"board": board, "scored": {metric: float(row[f"{metric}_mean"]) for metric in METRICS}, "difference": difference, "tolerance": tolerance, "passed": all(abs(difference[m]) <= tolerance[m] for m in tolerance)}


def paired_difference(first: pd.Series, second: pd.Series, resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    """Question 5a: the mean of `first` minus `second` over the compounds, with the 2.5 and 97.5 percentiles of a paired bootstrap, the verdict and the number of compounds in which `first` is higher."""
    a, b = first.to_numpy(dtype=float), second.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(resamples, len(a)))
    differences = (a[indices] - b[indices]).mean(axis=1)
    low, high = np.percentile(differences, [2.5, 97.5])
    verdict = "first better" if low > 0 else "second better" if high < 0 else "no difference shown"
    return {"mean_difference": float((a - b).mean()), "low": float(low), "high": float(high), "verdict": verdict, "first_higher": int((a > b).sum()), "compounds": len(a)}


def oriented(values: pd.Series, metric: str) -> pd.Series:
    """The metric with its sign chosen so that a larger value is better: BiSyRMSD, which is lower when better, is negated."""
    return -values if metric == "BiSyRMSD" else values


def pool_gaps(pool: pd.DataFrame, submitted: pd.Series) -> pd.DataFrame:
    """Question 5b: per compound, the LDDT-PLI of the best pose of the pool, of a random pose (the mean of the pool) and of the submitted pose, and the gaps of the two to the best.

    `pool` holds ``ligand`` and ``LDDT-PLI`` for every pose of every pool; `submitted` is indexed by ligand.
    """
    grouped = pool.groupby("ligand")["LDDT-PLI"]
    table = pd.DataFrame({"best": grouped.max(), "random": grouped.mean(), "poses": grouped.size()})
    table["submitted"] = submitted
    table["gap_submitted"] = table["best"] - table["submitted"]
    table["gap_random"] = table["best"] - table["random"]
    return table


_ATOM = re.compile(r"^(?:ATOM  |HETATM)")


def iron_nitrogen_distance(pdb_text: str) -> float | None:
    """The shortest distance in a PDB file from an iron atom of a heme to a nitrogen atom of the ligand (residue name LIG), in angstroms; None when either is absent."""
    iron, nitrogen = [], []
    for line in pdb_text.splitlines():
        if not _ATOM.match(line):
            continue
        residue, name = line[17:20].strip(), line[12:16].strip()
        element = (line[76:78].strip() or name[:1]).upper()
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        if residue == "HEM" and (element == "FE" or name.upper() == "FE"):
            iron.append(xyz)
        elif residue == "LIG" and element == "N":
            nitrogen.append(xyz)
    if not iron or not nitrogen:
        return None
    return float(np.linalg.norm(np.array(iron)[:, None, :] - np.array(nitrogen)[None, :, :], axis=2).min())


def coordination_table(truth: dict[str, float | None], predicted: dict[str, float | None]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Question 5c: per compound, the iron-nitrogen distance in the released structure and in the submitted pose and whether each is coordinated, and the two-by-two table of the calls."""
    rows = []
    for name in sorted(truth):
        t, p = truth[name], predicted.get(name)
        rows.append({"ligand": name, "truth_distance": t, "truth_coordinated": t is not None and t <= COORDINATION_DISTANCE_CUTOFF_A, "submitted_distance": p, "submitted_coordinated": p is not None and p <= COORDINATION_DISTANCE_CUTOFF_A})
    table = pd.DataFrame(rows)
    counts = pd.crosstab(table["truth_coordinated"], table["submitted_coordinated"]).reindex(index=[False, True], columns=[False, True], fill_value=0)
    counts.index.name, counts.columns.name = "truth_coordinated", "submitted_coordinated"
    return table, counts


def resolution(pdb_text: str) -> float | None:
    """The resolution of the map that a released structure records in its header (angstroms)."""
    match = re.search(r"RESOLUTION:\s*([0-9.]+)", pdb_text)
    return float(match.group(1)) if match else None


def _run_worker(job: dict) -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    job_path, out_path = WORK / "job.json", WORK / "result.json"
    job_path.write_text(json.dumps(job))
    done = subprocess.run([str(OST_PYTHON), str(WORKER), str(job_path), str(out_path), str(DEST / "organisers")], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"the scoring worker failed:\n{done.stderr[-2000:]}")
    return json.loads(out_path.read_text())


def _smiles() -> dict[str, str]:
    return dict(pd.read_csv(OFFICIAL)[["Molecule_Name", "SMILES"]].itertuples(index=False))


def _references() -> dict[str, str]:
    return {path.stem: str(path) for path in sorted(REFERENCES.glob("*.pdb"))}


def _submitted() -> dict[str, str]:
    folder = WORK / "submitted"
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SUBMISSION) as archive:
        archive.extractall(folder)
    return {path.stem: str(path) for path in sorted(folder.glob("*.pdb"))}


def score_submission() -> tuple[pd.DataFrame, dict]:
    """The per-compound scores of the submitted archive as the organisers' function gives them, and the compounds whose pose failed PoseBusters."""
    result = _run_worker({"mode": "submission", "predicted": _submitted(), "reference": _references(), "smiles": _smiles()})
    return pd.DataFrame(result["per_compound"]), result["pb_failures"]


def _convert(cif: Path, name: str) -> str:
    from cypmode.challenge.structure_submission import convert_cif_to_pdb

    target = WORK / "converted" / f"{name}.pdb"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        convert_cif_to_pdb(cif, target)
    return str(target)


def pose_items() -> pd.DataFrame:
    """Every pose that question 5 scores: the submitted ones, the published baseline of the organisers, the pose that unguided Boltz-2 ranks first, and every pose of the pool of each compound."""
    references, smiles, rows = _references(), _smiles(), []
    for ligand, path in _submitted().items():
        rows.append({"ligand": ligand, "arm": "submitted", "pose": "submitted", "model": path})
    for ligand in references:
        cif = BASELINE / f"cyp3a4_{ligand.lower()}_model_0.cif"
        rows.append({"ligand": ligand, "arm": "baseline", "pose": "baseline", "model": _convert(cif, f"{ligand}_baseline")})
    pools = pd.read_csv(POOLS)
    for _, row in pools.iterrows():
        pose = f"{row['run']}_{row['sample']}"
        model = _convert(ROOT / row["cif"], f"{row['ligand']}_{pose}")
        rows.append({"ligand": row["ligand"], "arm": "pool", "pose": pose, "model": model})
        if row["run"] == row["ligand"] and int(row["sample"]) == 0:
            rows.append({"ligand": row["ligand"], "arm": "unguided_first", "pose": pose, "model": model})
    table = pd.DataFrame(rows)
    table["reference"] = table["ligand"].map(references)
    table["smiles"] = table["ligand"].map(smiles)
    table["key"] = table["arm"] + "|" + table["ligand"] + "|" + table["pose"]
    return table


def score_poses(items: pd.DataFrame, cache: Path) -> pd.DataFrame:
    """The organisers' scores of every pose, computed once and kept in `cache`."""
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["key"])
    todo = items[~items["key"].isin(done["key"])]
    if len(todo):
        job = {"mode": "poses", "items": [{"key": r.key, "model": r.model, "reference": r.reference, "smiles": r.smiles} for r in todo.itertuples()]}
        scored = pd.DataFrame(_run_worker(job)["rows"])
        scored["pb_failed"] = scored["pb_failed"].map(lambda failed: ";".join(failed))
        done = pd.concat([done, scored], ignore_index=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        done.to_csv(cache, index=False)
    merged = items.merge(done, on="key", how="left")
    for column in ("LDDT-PLI", "LDDT-LP"):
        merged[column] = merged[column].fillna(0.0)  # a pose that could not be matched scores zero, as in the organisers' function
    merged["BiSyRMSD"] = merged["BiSyRMSD"].fillna(20.0)
    failed = merged["pb_failed"].fillna("") != ""
    merged.loc[failed, ["LDDT-PLI", "LDDT-LP"]], merged.loc[failed, "BiSyRMSD"] = 0.0, 20.0  # a pose that fails a PoseBusters check is zeroed
    return merged


def run(out: Path = OUT) -> int:
    out.mkdir(parents=True, exist_ok=True)
    per_compound, failures = score_submission()
    per_compound.to_csv(out / "structure_submitted_per_compound.csv", index=False)
    summary = board_value(per_compound)
    summary.to_csv(out / "structure_board_scores.csv")
    verdict = gate(summary)
    (out / "structure_gate.json").write_text(json.dumps({**verdict, "posebusters_failures": failures}, indent=1) + "\n")
    print(f"QUESTION 0 (structure): LDDT-PLI {verdict['scored']['LDDT-PLI']:.7f} (board {BOARD['LDDT-PLI']:.7f}, difference {verdict['difference']['LDDT-PLI']:+.2e}); BiSyRMSD {verdict['scored']['BiSyRMSD']:.7f} (board {BOARD['BiSyRMSD']:.7f}, difference {verdict['difference']['BiSyRMSD']:+.2e}); GATE: {'MET' if verdict['passed'] else 'NOT MET'}")
    if not verdict["passed"]:
        return 1
    scored = score_poses(pose_items(), out / "structure_poses.csv")
    by_arm = {arm: frame.set_index("ligand") for arm, frame in scored.groupby("arm")}
    rows = [
        {"comparison": f"submitted minus {arm}", "metric": metric, **paired_difference(oriented(by_arm["submitted"][metric], metric), oriented(by_arm[arm][metric], metric))}
        for arm in ("baseline", "unguided_first")
        for metric in ("LDDT-PLI", "BiSyRMSD")
    ]
    pd.DataFrame(rows).to_csv(out / "structure_question5a.csv", index=False)
    print("QUESTION 5a (BiSyRMSD is negated, so that a positive difference is better for both metrics)\n" + pd.DataFrame(rows).to_string(index=False))
    gaps = pool_gaps(scored[scored["arm"] == "pool"], by_arm["submitted"]["LDDT-PLI"])
    gaps.to_csv(out / "structure_question5b.csv")
    print("QUESTION 5b\n" + gaps[["best", "random", "submitted", "gap_submitted", "gap_random"]].mean().to_string())
    references = _references()
    truth = {name: iron_nitrogen_distance(Path(path).read_text()) for name, path in references.items()}
    predicted = {name: iron_nitrogen_distance(Path(path).read_text()) for name, path in _submitted().items()}
    table, counts = coordination_table(truth, predicted)
    table["resolution"] = [resolution(Path(references[name]).read_text()) for name in table["ligand"]]
    table["in_training"] = table["ligand"].isin(set(pd.read_csv(TRAINING)["Molecule_Name"]))
    table = table.merge(by_arm["submitted"][["LDDT-PLI", "BiSyRMSD"]].rename(columns={"LDDT-PLI": "submitted_lddt_pli", "BiSyRMSD": "submitted_bisyrmsd"}), left_on="ligand", right_index=True)
    table.to_csv(out / "structure_question5c.csv", index=False)
    counts.to_csv(out / "structure_question5c_counts.csv")
    rho = spearmanr(table["resolution"], table["submitted_lddt_pli"])
    (out / "structure_resolution.json").write_text(json.dumps({"spearman_rho": float(rho.statistic), "p_value": float(rho.pvalue), "compounds": len(table)}, indent=1) + "\n")
    print("QUESTION 5c: two-by-two table of the calls\n" + counts.to_string())
    print(f"resolution against the LDDT-PLI of the submitted pose: rho {rho.statistic:+.3f} (n = {len(table)})")
    return 0


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    return run()


if __name__ == "__main__":
    sys.exit(main())
