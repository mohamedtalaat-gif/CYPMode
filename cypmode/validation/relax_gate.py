"""Gate of the force-field relaxation: does it help on poses whose answer is known?

``relax`` repairs a pose in a frozen pocket. The challenge scorer gives a pose that fails its PoseBusters checks an LDDT-PLI of 0, so a relaxation that turns invalid poses into valid
ones without moving the rest away from the crystal raises the official score. This module tests exactly that on every Boltz-2 pose of the ligands whose crystal pose is public
(the restraint-expansion set ``exp-*`` and the benchmark set ``bench-*``), before any use on the 20 official ligands:

  1. every pose is converted to the submission format and relaxed in the conda environment ``openmm`` (resumable);
  2. every pose, before and after, goes through the scorer's own PoseBusters call (``official_posebusters``: pass or fail, and the smallest relative distance to the receptor) and
     through OpenStructure's LDDT-PLI against the crystal copies of the ligand (``ost_official``);
  3. the official-style score of a pose is its LDDT-PLI, or 0 when it fails PoseBusters.

``python -m cypmode.validation.relax_gate`` writes ``data/structure_track/outputs/cyp3a4/relaxation_gate_poses.csv`` (one row per pose, resumable) and prints the verdict.

GATE (registered before the run; the unit is the ligand, the mean over its poses; one-sided paired Wilcoxon tests, ``relaxed`` against ``original``):
  validity:    the share of poses that pass the 22 checks does not fall and the official-style score rises (p < 0.05);
  margin:      among the poses that pass before and after, the smallest relative distance to the receptor rises (p < 0.05);
  fidelity:    among the poses that pass before and after, LDDT-PLI does not fall by more than 0.01 on average (the mean paired change is above -0.01 and its one-sided test against
               -0.01 has p < 0.05).
The gate is MET when all three hold. Otherwise the relaxation is not used.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/structure_track/outputs/cyp3a4")
WORK = OUT / "relaxed"
TABLE = OUT / "relaxation_gate_poses.csv"
MAMBA = Path.home() / "miniforge3" / "bin" / "mamba"
ENV = "openmm"
ALPHA = 0.05
FIDELITY_MARGIN = 0.01
INPUTS = Path("data/structure_track/inputs")


def crystal_poses() -> list[dict]:
    """Every Boltz-2 pose of the ligands with a public crystal pose: compound, ligand SMILES, sample index and CIF path."""
    from cypmode.validation.pose_pools import pool_paths

    ligands = pd.read_csv(INPUTS / "restraint_expansion_ligands.csv")
    smiles_of = {f"exp-{c}": s for c, s in zip(ligands.comp_id, ligands.smiles, strict=True)}
    bench = pd.read_csv(INPUTS / "structure_benchmark.csv")
    smiles_of |= dict(zip(bench.Molecule_Name, bench.SMILES, strict=True))
    compounds = sorted(set(pd.read_csv(OUT / "restraint_expansion_poses.csv").compound) | set(pd.read_csv(OUT / "pose_lddt_to_crystal.csv").compound))
    rows = []
    for compound in compounds:
        for engine, sample, path in pool_paths(compound, ("boltz",)):
            if compound in smiles_of:
                rows.append({"compound": compound, "smiles": smiles_of[compound], "engine": engine, "sample": sample, "cif": str(path)})
    return rows


def pose_name(row: dict) -> str:
    return f"{row['compound']}_{row['engine']}{row['sample']}"


def prepare(rows: list[dict]) -> list[dict]:
    """Write the original poses in the submission format and the manifest of the relaxation; returns the manifest entries."""
    from cypmode.challenge.structure_submission import convert_cif_to_pdb

    jobs = []
    for row in rows:
        original = WORK / "original" / f"{pose_name(row)}.pdb"
        if not original.exists():
            original.parent.mkdir(parents=True, exist_ok=True)
            convert_cif_to_pdb(Path(row["cif"]), original)
        jobs.append({"pdb": str(original), "smiles": row["smiles"], "out": str(WORK / "relaxed" / f"{pose_name(row)}.pdb")})
    return jobs


def relax_all(jobs: list[dict]) -> None:
    """Relax every pose that has no result yet, in one process of the ``openmm`` environment."""
    missing = [j for j in jobs if not (Path(j["out"]).exists() and Path(j["out"] + ".json").exists())]
    if not missing:
        return
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "jobs.json"
        manifest.write_text(json.dumps(missing))
        command = [str(MAMBA), "run", "-n", ENV, "python", "-m", "cypmode.validation.relax", "--manifest", str(manifest)]
        subprocess.run(command, check=True, env={**__import__("os").environ, "PYTHONPATH": "."})


def score_pose(pdb: Path, smiles: str, records: list[dict], workdir: Path) -> dict:
    from cypmode.validation.official_posebusters import check_pdb
    from cypmode.validation.ost_official import score_against_copies

    report = check_pdb(pdb, smiles)
    lddt = score_against_copies(pdb, smiles, records, workdir).get("lddt_pli", float("nan")) if report.passed else float("nan")
    return {"passed": report.passed, "failed": ";".join(report.failed) or report.error, "margin": report.distance_margin, "lddt_pli": lddt}


def score_all(rows: list[dict], jobs: list[dict]) -> pd.DataFrame:
    from cypmode.validation.ost_official import reference_records

    done = pd.read_csv(TABLE) if TABLE.exists() else pd.DataFrame()
    finished = set(done.pose) if len(done) else set()
    table = done.to_dict("records")
    for row, job in zip(rows, jobs, strict=True):
        name = pose_name(row)
        relaxed = Path(job["out"])
        info_path = Path(job["out"] + ".json")
        if name in finished or not relaxed.exists() or not info_path.exists():
            continue
        info = json.loads(info_path.read_text())
        records = reference_records(row["smiles"])
        with tempfile.TemporaryDirectory() as tmp:
            before = score_pose(Path(job["pdb"]), row["smiles"], records, Path(tmp))
            after = score_pose(relaxed, row["smiles"], records, Path(tmp))
        table.append({"pose": name, "compound": row["compound"], "sample": row["sample"], **{f"{k}_before": v for k, v in before.items()}, **{f"{k}_after": v for k, v in after.items()},
                      "rmsd": info.get("rmsd", np.nan), "energy_before": info.get("energy_before", np.nan), "energy_after": info.get("energy_after", np.nan), "relax_error": info.get("error", "")})
        pd.DataFrame(table).to_csv(TABLE, index=False)
        print(f"{name}: pass {before['passed']} -> {after['passed']}, margin {before['margin']:.3f} -> {after['margin']:.3f}, LDDT-PLI {before['lddt_pli']:.3f} -> {after['lddt_pli']:.3f}", flush=True)
    return pd.DataFrame(table)


def paired_greater(delta: np.ndarray, shift: float = 0.0) -> float:
    """One-sided Wilcoxon p-value of ``delta - shift > 0`` (1.0 when every difference is zero)."""
    from scipy.stats import wilcoxon

    centred = np.asarray(delta, dtype=float) - shift
    if len(centred) == 0 or np.all(centred == 0):
        return 1.0
    return float(wilcoxon(centred, alternative="greater", zero_method="wilcox").pvalue)


def verdict(table: pd.DataFrame) -> dict:
    """The three parts of the gate from the per-pose table (the unit of every test is the ligand)."""
    usable = table[table.relax_error.fillna("") == ""].copy()
    usable["official_before"] = np.where(usable.passed_before, usable.lddt_pli_before.fillna(0.0), 0.0)
    usable["official_after"] = np.where(usable.passed_after, usable.lddt_pli_after.fillna(0.0), 0.0)
    by_ligand = usable.groupby("compound").agg(pass_before=("passed_before", "mean"), pass_after=("passed_after", "mean"), official_before=("official_before", "mean"), official_after=("official_after", "mean"))
    both = usable[usable.passed_before & usable.passed_after]
    kept = both.groupby("compound").agg(margin_before=("margin_before", "mean"), margin_after=("margin_after", "mean"), lddt_before=("lddt_pli_before", "mean"), lddt_after=("lddt_pli_after", "mean")).dropna()
    result = {
        "poses": len(usable), "failed_to_relax": int(len(table) - len(usable)), "ligands": len(by_ligand), "ligands_with_poses_passing_both": len(kept),
        "pass_rate_before": float(usable.passed_before.mean()), "pass_rate_after": float(usable.passed_after.mean()),
        "official_before": float(by_ligand.official_before.mean()), "official_after": float(by_ligand.official_after.mean()),
        "p_official": paired_greater((by_ligand.official_after - by_ligand.official_before).to_numpy()),
        "margin_before": float(kept.margin_before.mean()) if len(kept) else float("nan"), "margin_after": float(kept.margin_after.mean()) if len(kept) else float("nan"),
        "p_margin": paired_greater((kept.margin_after - kept.margin_before).to_numpy()),
        "lddt_change": float((kept.lddt_after - kept.lddt_before).mean()) if len(kept) else float("nan"),
        "p_fidelity": paired_greater((kept.lddt_after - kept.lddt_before).to_numpy(), -FIDELITY_MARGIN),
        "median_movement_A": float(usable.rmsd.median()),
    }
    result["validity_met"] = bool(result["pass_rate_after"] >= result["pass_rate_before"] and result["p_official"] < ALPHA)
    result["margin_met"] = bool(result["p_margin"] < ALPHA)
    result["fidelity_met"] = bool(result["lddt_change"] > -FIDELITY_MARGIN and result["p_fidelity"] < ALPHA)
    result["met"] = result["validity_met"] and result["margin_met"] and result["fidelity_met"]
    return result


def main() -> int:
    rows = crystal_poses()
    jobs = prepare(rows)
    print(f"{len(rows)} poses of {len({r['compound'] for r in rows})} crystal-anchored ligands", flush=True)
    relax_all(jobs)
    table = score_all(rows, jobs)
    if table.empty:
        print("no pose could be scored")
        return 1
    v = verdict(table)
    print(f"\n{v['poses']} poses of {v['ligands']} ligands ({v['failed_to_relax']} could not be relaxed); the ligand moved {v['median_movement_A']:.2f} A (median heavy-atom RMSD)")
    print(f"poses passing the 22 checks: {v['pass_rate_before']:.1%} -> {v['pass_rate_after']:.1%}; official-style score {v['official_before']:.3f} -> {v['official_after']:.3f} (p = {v['p_official']:.4f})")
    print(f"margin (poses passing before and after, {v['ligands_with_poses_passing_both']} ligands): {v['margin_before']:.3f} -> {v['margin_after']:.3f} (p = {v['p_margin']:.4f})")
    print(f"LDDT-PLI change on those poses: {v['lddt_change']:+.4f} (non-inferiority margin -{FIDELITY_MARGIN}, p = {v['p_fidelity']:.4f})")
    print(f"STRUCTURE GATE force_field_relaxation (validity, margin and fidelity at p < {ALPHA}): {'MET' if v['met'] else 'NOT MET'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
