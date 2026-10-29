# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Check of the conformational bias of the ligand ensembles (job 273_conformer_bias, declared on 2026-10-23 before any code ran).

The ensembles of the pocket block and of the site model hold up to 12 conformers within 5 kcal/mol of the lowest (MMFF), low-energy states by construction; a ligand bound in a pocket is often not in
one. For each crystal ligand copy of the CYP3A4 reference set (``reference_ligands``; no blinded compound is involved): ``N_CONFORMERS`` ETKDG conformers (seed 42, MMFF-relaxed), the heavy-atom RMSD
(symmetry aware) of the nearest one to the crystal pose and its energy above the lowest, the lowest energy among the conformers within ``COVER_A`` of the crystal pose (the window that the bound state needs),
and the same nearest RMSD for the ensemble that the block uses (``pocket_pharmacophore.conformers``: the coverage of the bound state by the registered ensemble).

Consequence, fixed in advance: if fewer than ``COVERAGE_MIN`` of the copies are covered by the ensemble of the block, every place that reports the verdicts of the pocket block and of the site model says that
they were tested with an ensemble that misses the bound conformation of that share of the crystal ligands.

Usage: python -m cypmode.validation.conformer_bias      (job 273)
"""

import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

N_CONFORMERS = 300
COVER_A = 1.0
WINDOWS = (5.0, 10.0, 20.0)
COVERAGE_MIN = 0.8
MAX_MATCHES = 5000  # symmetry matches tried per alignment: enough for a ligand with several symmetric groups, bounded for a very symmetric one
OUT = Path("data/structure_track/outputs/cyp3a4/conformer_bias.csv")


def _nearest(heavy_conformers, crystal, energies: np.ndarray) -> tuple[float, float, float]:
    """(RMSD of the nearest conformer to the crystal pose, its relative energy, the lowest relative energy among the conformers within COVER_A of it or NaN); the alignment is symmetry aware."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdMolAlign

    rmsds = np.full(len(energies), np.nan)
    with rdBase.BlockLogs():
        for k, conformer in enumerate(heavy_conformers):
            probe = Chem.Mol(conformer)
            try:
                rmsds[k] = float(rdMolAlign.GetBestRMS(probe, Chem.Mol(crystal), maxMatches=MAX_MATCHES))
            except Exception:
                continue
    if np.isnan(rmsds).all():
        return float("nan"), float("nan"), float("nan")
    nearest = int(np.nanargmin(rmsds))
    covered = np.flatnonzero(rmsds <= COVER_A)
    return float(rmsds[nearest]), float(energies[nearest]), float(energies[covered].min()) if len(covered) else float("nan")


def _split(mol, ids: list[int]):
    """The conformers of a molecule as separate single-conformer molecules, hydrogens removed."""
    from rdkit import Chem

    return [Chem.RemoveHs(Chem.Mol(mol, False, cid)) for cid in ids]


def copy_row(record: dict) -> dict:
    """The row of one crystal ligand copy: sizes, the nearest of the many conformers and the nearest of the ensemble of the block; NaN where the ligand could not be treated."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem, Descriptors

    from cypmode.challenge.pocket_pharmacophore import (
        CONFORMER_SEED,
        conformers,
        crystal_pose,
    )

    row = {"pdb_id": record["pdb_id"], "comp_id": record["comp_id"], "n_heavy": record["n_heavy"], "rotatable": np.nan, "nearest_rmsd": np.nan, "nearest_energy": np.nan, "needed_energy": np.nan, "ensemble_rmsd": np.nan}
    with rdBase.BlockLogs():
        crystal = crystal_pose(record)
        template = Chem.MolFromSmiles(record["smiles"]) if record.get("smiles") else None
        if crystal is None or template is None:
            return row
        row["rotatable"] = Descriptors.NumRotatableBonds(template)
        mol = Chem.AddHs(template)
        ids = list(AllChem.EmbedMultipleConfs(mol, N_CONFORMERS, randomSeed=CONFORMER_SEED, numThreads=1))
        if not ids:
            return row
        energies = np.array([e for _, e in AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500, numThreads=1)])
        relative = energies - energies.min()
        row["nearest_rmsd"], row["nearest_energy"], row["needed_energy"] = _nearest(_split(mol, ids), crystal, relative)
        block_mol, block_ids = conformers(template)
        if block_mol is not None and block_ids:
            row["ensemble_rmsd"], _, _ = _nearest([Chem.Mol(block_mol, False, cid) for cid in block_ids], crystal, np.zeros(len(block_ids)))
    return row


def summarize(table: pd.DataFrame) -> list[str]:
    """The lines of the log: coverage by the many conformers, the window the bound state needs, coverage by the ensemble of the block and the consequence rule."""
    done = table[table.nearest_rmsd.notna()]
    covered = done[done.nearest_rmsd <= COVER_A]
    lines = [f"crystal ligand copies evaluated: {len(done)} of {len(table)} ({done.comp_id.nunique()} distinct ligands)",
             f"{N_CONFORMERS} conformers: a conformer within {COVER_A} A of the crystal pose for {len(covered) / max(len(done), 1):.0%} of the copies; median nearest RMSD {done.nearest_rmsd.median():.2f} A, median energy of the nearest conformer {done.nearest_energy.median():.1f} kcal/mol"]
    needed = covered.needed_energy.dropna()
    for window in WINDOWS:
        lines.append(f"  the bound state lies within {window:g} kcal/mol of the lowest conformer for {(needed <= window).sum()} of {len(covered)} covered copies ({(needed <= window).sum() / max(len(covered), 1):.0%}), {(needed <= window).sum() / max(len(done), 1):.0%} of all evaluated")
    lines.append(f"  median energy that the bound state needs {needed.median():.1f} kcal/mol, 90th percentile {needed.quantile(0.9):.1f} kcal/mol")
    block = done[done.ensemble_rmsd.notna()]
    share = float((block.ensemble_rmsd <= COVER_A).sum() / max(len(done), 1))
    lines.append(f"the ensemble of the block (12 conformers within 5 kcal/mol): a member within {COVER_A} A of the crystal pose for {share:.0%} of the {len(done)} copies; median nearest RMSD {block.ensemble_rmsd.median():.2f} A")
    lines.append(f"CONFORMER CHECK block_ensemble_covers_the_bound_state (at least {COVERAGE_MIN:.0%} of the copies within {COVER_A} A): {'YES' if share >= COVERAGE_MIN else 'NO'}")
    if share < COVERAGE_MIN:
        lines.append(f"consequence: the verdicts of the pocket block and of the site model are reported as tested with an ensemble that misses the bound conformation of {1 - share:.0%} of the crystal ligands")
    return lines


def pose_strain_lines(path: Path = Path("data/structure_track/outputs/cyp3a4/pose_awareness_poses.csv")) -> list[str]:
    """Information: how far the predicted benchmark poses of the pools of `pose_awareness` lie from the low-energy conformers. The conformer ensemble of the pose terms holds only conformers within 8 kcal/mol, so the relative energy of the nearest one cannot
    exceed that; a pose that resembles no low-energy conformer shows as a large RMSD to the nearest one instead."""
    if not path.exists():
        return []
    table = pd.read_csv(path)[["conf_rmsd", "conf_energy"]].dropna()
    return [(f"the {len(table)} predicted benchmark poses against the low-energy conformers of the pose terms (within 8 kcal/mol): median RMSD to the nearest conformer {table.conf_rmsd.median():.2f} A; farther than 1.0 A for {(table.conf_rmsd > 1.0).mean():.0%}, farther than 2.0 A for {(table.conf_rmsd > 2.0).mean():.0%} "
            f"(these poses resemble no low-energy conformer); median relative energy of the nearest conformer {table.conf_energy.median():.1f} kcal/mol")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--limit", type=int, default=0, help="only the first N copies (a test run)")
    args = parser.parse_args()
    import json

    from cypmode.challenge.pocket_pharmacophore import POCKETS

    records = json.loads(POCKETS["cyp3a4"].read_text())["ligands"]
    if args.limit:
        records = records[: args.limit]
    with multiprocessing.Pool(args.workers) as pool:
        rows = pool.map(copy_row, records, chunksize=2)
    table = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print("\n".join([*summarize(table), *pose_strain_lines()]))


if __name__ == "__main__":
    from cypmode import provenance

    provenance.announce()
    main()
