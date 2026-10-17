"""Pilot: which lever makes Boltz-2's pose pool differ, and does the difference reach the right pose?

Unguided Boltz-2 returns five samples from one trunk pass and they agree with each other (on the diverse
crystal benchmark the best of five is only ~0.7 A better than the first pick). Three levers could break
that, each cheap to try on a few ligands before any large run:

  seeds      the same input with different random seeds (a fresh trunk pass per seed, as AF3 does)
  conformers a different crystal conformation of the pocket as protein template
  stepscale  a lower sampling step scale (boltz predict --step_scale), more spread between samples

A lever "creates diversity" when its pool spreads more than the unguided pool does, and "reaches the
right pose" when the best pose in the pool is clearly closer to the crystal pose than the unguided
pool's best. Both thresholds are fixed here, before any result exists.
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.validation.conformers import (
    entry_pockets,
    excluded_entries,
    pick_conformers,
)
from cypmode.validation.pose_rescoring_validation import NOVEL_SIMILARITY
from cypmode.validation.reference_ligands import load_reference_set

LEVERS = ("seeds", "conformers", "stepscale", "cryoem")
DIVERSITY_FACTOR = 1.5  # pool spread must exceed the unguided pool's by this factor
GAIN_A = 0.75  # best-in-pool true RMSD must beat the unguided pool's best by this much (Angstrom)...
GAIN_LIGAND_FRACTION = 0.5  # ...for at least this fraction of the ligands
STEP_SCALE = 0.8
LEVER_OF_SUFFIX = {"seed": "seeds", "cf": "conformers", "ss": "stepscale", "em": "cryoem"}  # run-name suffix -> lever (em: cryoem_templates.py)
SAMPLES = 3


def build_pilot_table(ligands: pd.DataFrame, references: list[dict], n_runs: int = 3) -> pd.DataFrame:
    """One row per run: ``<ligand>-seed<k>``, ``<ligand>-cf<k>`` (with its template) and ``<ligand>-ss``."""
    pockets = entry_pockets(references)
    rows = []
    for _, ligand in ligands.iterrows():
        name, smiles = ligand.Molecule_Name, ligand.SMILES
        for k in range(1, n_runs + 1):
            rows.append({"Molecule_Name": f"{name}-seed{k}", "SMILES": smiles, "Seed": 100 + k, "Diffusion_Samples": 1})
        banned = excluded_entries(smiles, references, NOVEL_SIMILARITY)
        for k, pdb_id in enumerate(pick_conformers(pockets, n_runs, banned), start=1):
            rows.append({"Molecule_Name": f"{name}-cf{k}", "SMILES": smiles, "Template_PDB": pdb_id, "Diffusion_Samples": 1})
        rows.append({"Molecule_Name": f"{name}-ss", "SMILES": smiles, "Step_Scale": STEP_SCALE, "Diffusion_Samples": SAMPLES})
    return pd.DataFrame(rows).fillna("")


def lever_of(name: str) -> str:
    return LEVER_OF_SUFFIX[name.rsplit("-", 1)[1].rstrip("0123456789")]


def ligand_of(name: str) -> str:
    return name.rsplit("-", 1)[0]


def pool_spread(cif_paths: list[Path], smiles: str) -> float | None:
    """Mean pairwise pocket-aligned, symmetry-aware ligand RMSD (A) within a pool of predicted complexes."""
    from cypmode.validation.pose_consensus import aligned_ligand_rmsd
    from cypmode.validation.structures import load_structure

    if len(cif_paths) < 2:
        return None
    structures = [load_structure(p) for p in cif_paths]
    return float(np.mean([aligned_ligand_rmsd(a, b, smiles)["ligand_rmsd"] for a, b in combinations(structures, 2)]))


def verdicts(pools: pd.DataFrame) -> pd.DataFrame:
    """``pools``: one row per (ligand, lever) with spread, best_rmsd and the unguided pool's spread and best_rmsd."""
    rows = []
    for lever, group in pools.groupby("lever"):
        wider = float((group["spread"] >= DIVERSITY_FACTOR * group["unguided_spread"]).mean())
        closer = float((group["unguided_best"] - group["best_rmsd"] >= GAIN_A).mean())
        rows.append(
            {
                "lever": lever,
                "ligands": len(group),
                "ligands_with_wider_pool": wider,
                "ligands_with_better_best_pose": closer,
                "creates_diversity": wider >= GAIN_LIGAND_FRACTION,
                "reaches_right_pose": closer >= GAIN_LIGAND_FRACTION,
            }
        )
    return pd.DataFrame(rows)


def _report(pose_table: pd.DataFrame, smiles_by_ligand: dict[str, str], unguided: pd.DataFrame) -> pd.DataFrame:
    from cypmode.validation.pose_consensus import boltz_cif_paths

    rows = []
    for ligand, smiles in smiles_by_ligand.items():
        base = unguided[unguided["compound"] == ligand]
        if base.empty:
            continue
        base_spread = pool_spread(boltz_cif_paths(ligand), smiles)
        for lever in LEVERS:
            names = sorted({c for c in pose_table["compound"] if ligand_of(c) == ligand and lever_of(c) == lever})
            paths = [p for n in names for p in boltz_cif_paths(n)]
            truth = pose_table[pose_table["compound"].isin(names)]["true_rmsd"].dropna()
            if len(paths) < 2 or truth.empty:
                continue
            rows.append(
                {
                    "ligand": ligand,
                    "lever": lever,
                    "poses": len(paths),
                    "spread": pool_spread(paths, smiles),
                    "unguided_spread": base_spread,
                    "best_rmsd": float(truth.min()),
                    "unguided_best": float(base["true_rmsd"].min()),
                }
            )
    return pd.DataFrame(rows)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="write the batch CSV for the pilot runs")
    build.add_argument("--benchmark", default="data/structure_track/inputs/structure_benchmark.csv")
    build.add_argument("--only", nargs="+", required=True)
    build.add_argument("--out", required=True)
    report = sub.add_parser("report", help="compare the pools once the runs are scored")
    report.add_argument("--benchmark", default="data/structure_track/inputs/structure_benchmark.csv")
    report.add_argument("--pilot-poses", required=True, help="pose_rescoring_validation --engine-poses table of the pilot runs")
    report.add_argument("--unguided-poses", required=True, help="the same table for the unguided benchmark runs")
    args = parser.parse_args()
    ligands = pd.read_csv(args.benchmark)
    if args.command == "build":
        table = build_pilot_table(ligands[ligands.Molecule_Name.isin(args.only)], load_reference_set())
        table.to_csv(args.out, index=False)
        print(table.to_string(index=False))
        return
    smiles = dict(zip(ligands.Molecule_Name, ligands.SMILES))
    pools = _report(pd.read_csv(args.pilot_poses), smiles, pd.read_csv(args.unguided_poses))
    print(pools.round(2).to_string(index=False))
    print()
    print(verdicts(pools).round(2).to_string(index=False))


if __name__ == "__main__":
    _main()
