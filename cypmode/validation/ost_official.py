# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Does the LDDT-PLI of the project agree with the one the challenge computes with OpenStructure?

Every structure gate of the project (the restraint, the selection rules, the series and pose-awareness terms) rests on ``lddt_to_crystal``, a re-implementation of the
official LDDT-PLI. The scorer itself is OpenStructure (``LDDTPLIScorer``, ``SCRMSDScorer``), which is available only through conda; ``ost_worker`` runs it in the ``ost``
environment exactly as the organisers' evaluation code does. This module writes the crystal complexes as reference PDBs (protein chain A, ligand ``LIG`` as chain B, heme as
chain C: the layout of a submitted file), scores every pose of the confirmation ligands against them with OpenStructure, and compares the result with the local values pose by pose.

``python -m cypmode.validation.ost_official`` writes ``data/structure_track/outputs/cyp3a4/ost_vs_ours.csv`` (resumable: finished poses are kept) and prints the agreement.
GATE (registered before the run): the rank correlation between the two LDDT-PLI values over all poses is at least 0.9 and the mean absolute difference is at most 0.05; the
ordering of the poses of a ligand (the quantity every selection rule uses) agrees in at least 90% of the pairs. Otherwise the gates that rest on the local measure are repeated with
OpenStructure's.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

MAMBA = Path.home() / "miniforge3" / "bin" / "mamba"
ENV = "ost"
WORKER = Path(__file__).resolve().parent / "ost_worker.py"
OUT = Path("data/structure_track/outputs/cyp3a4")
TABLE = OUT / "ost_vs_ours.csv"
MIN_RANK_CORRELATION, MAX_MEAN_ABSOLUTE_DIFFERENCE, MIN_PAIR_AGREEMENT = 0.9, 0.05, 0.9


def write_reference_pdb(record: dict, path: Path) -> None:
    """The crystal complex of a reference record as a submission-format PDB: protein chain A, the ligand as residue LIG of chain B, the nearest heme as chain C."""
    from Bio.PDB import PDBIO, MMCIFParser, Select
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Structure import Structure

    from cypmode.validation.reference_ligands import CACHE_DIR
    from cypmode.validation.structures import _STANDARD_RESIDUES

    source = MMCIFParser(QUIET=True).get_structure(record["pdb_id"], str(Path(CACHE_DIR) / f"{record['pdb_id']}.cif"))[0]
    anchor = np.asarray(record["coords"][0], dtype=float)
    holder = ligand = None
    for chain in source:
        for residue in chain:
            if any(np.linalg.norm(atom.coord - anchor) < 0.02 for atom in residue):
                holder, ligand = chain, residue
    if holder is None:
        raise ValueError(f"{record['pdb_id']}: the reference ligand is not in the entry")
    protein, ligand_chain, heme_chain = Chain("A"), Chain("B"), Chain("C")
    for residue in holder:
        if residue.resname in _STANDARD_RESIDUES:
            protein.add(residue.copy())
    copy = ligand.copy()
    copy.resname = "LIG"
    copy.id = (" ", 1, " ")
    ligand_chain.add(copy)
    hemes = [r for r in holder if r.resname == "HEM"]
    if hemes:
        nearest = min(hemes, key=lambda r: min(np.linalg.norm(a.coord - anchor) for a in r))
        heme = nearest.copy()
        heme.id = (" ", 1, " ")
        heme_chain.add(heme)
    model = Model(0)
    for chain in (protein, ligand_chain, heme_chain):
        if len(chain):
            model.add(chain)
    structure = Structure(record["pdb_id"])
    structure.add(model)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path), Select())


def ost_score(model_pdb: Path, reference_pdb: Path, smiles: str) -> dict:
    command = [str(MAMBA), "run", "-n", ENV, "python", str(WORKER), "--model", str(model_pdb), "--reference", str(reference_pdb), "--smiles", smiles]
    done = subprocess.run(command, capture_output=True, text=True, check=False, timeout=900)
    for line in done.stdout.splitlines():
        if line.startswith("OST_RESULT "):
            return json.loads(line.removeprefix("OST_RESULT "))
    return {"error": (done.stderr or done.stdout)[-300:]}


def reference_records(smiles: str) -> list[dict]:
    """The crystal copies of a ligand that ``lddt_to_crystal`` compares a pose with (same filter), as reference records."""
    from rdkit import Chem

    from cypmode.validation.pose_rescoring import connectivity_key
    from cypmode.validation.pose_rescoring_validation import crystal_mol
    from cypmode.validation.reference_ligands import load_reference_set

    template, key = Chem.MolFromSmiles(smiles), connectivity_key(smiles)
    out = []
    for record in load_reference_set():
        if not key or connectivity_key(record["smiles"]) != key:
            continue
        mol = crystal_mol(smiles, record["elements"], np.asarray(record["coords"], dtype=float))
        if mol is not None and len(mol.GetSubstructMatch(template)) == template.GetNumAtoms():
            out.append(record)
    return out


def score_against_copies(model_pdb: Path, smiles: str, records: list[dict], workdir: Path) -> dict:
    """OpenStructure's best LDDT-PLI over the crystal copies (as ``lddt_to_crystal`` takes the best copy)."""
    best: dict = {}
    for k, record in enumerate(records):
        reference = workdir / f"ref_{k}.pdb"
        try:
            write_reference_pdb(record, reference)
        except Exception as exc:
            best.setdefault("error", f"reference {record['pdb_id']}: {exc}")
            continue
        result = ost_score(model_pdb, reference, smiles)
        if "lddt_pli" in result and result["lddt_pli"] > best.get("lddt_pli", -1.0):
            best = result
    return best or {"error": "no reference copy could be scored"}


def pairs_agreement(table: pd.DataFrame) -> float:
    """Share of the pose pairs of the same ligand that both measures order the same way (ties ignored)."""
    agree = total = 0
    for _, group in table.groupby("ligand"):
        values = group[["ours", "ost"]].to_numpy()
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                a, b = values[i, 0] - values[j, 0], values[i, 1] - values[j, 1]
                if a != 0 and b != 0:
                    total += 1
                    agree += int(a * b > 0)
    return agree / total if total else float("nan")


def finished_keys(done: pd.DataFrame) -> set[tuple]:
    """(compound, engine, sample) of the poses already scored; the columns are read by name because ``DataFrame.sample`` is a method, not the column."""
    return set(zip(done["compound"], done["engine"], done["sample"], strict=True))


def main() -> int:
    from cypmode.challenge.structure_submission import convert_cif_to_pdb
    from cypmode.validation.pose_pools import pool_paths
    from cypmode.validation.restraint_expansion import INPUTS, RESTRAINED_SUFFIX

    ours = pd.read_csv(OUT / "restraint_expansion_poses.csv").dropna(subset=["lddt_pli"])
    table = pd.read_csv(INPUTS / "restraint_expansion_ligands.csv")
    smiles_of = {f"exp-{c}": s for c, s in zip(table.comp_id, table.smiles, strict=True)}
    done = pd.read_csv(TABLE) if TABLE.exists() else pd.DataFrame(columns=["compound", "engine", "sample"])
    finished = finished_keys(done)
    rows = done.to_dict("records")
    paths = {(c, e, s): p for c in ours.compound.unique() for e, s, p in pool_paths(c, ("boltz",))}
    for r in ours.itertuples(index=False):
        if (r.compound, r.engine, r.sample) in finished or (r.compound, r.engine, r.sample) not in paths:
            continue
        smiles = smiles_of[r.compound.removesuffix(RESTRAINED_SUFFIX)]
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            model = workdir / "model.pdb"
            convert_cif_to_pdb(paths[(r.compound, r.engine, r.sample)], model)
            result = score_against_copies(model, smiles, reference_records(smiles), workdir)
        rows.append({"compound": r.compound, "engine": r.engine, "sample": r.sample, "ligand": r.compound.removesuffix(RESTRAINED_SUFFIX), "ours": r.lddt_pli, "ost": result.get("lddt_pli", np.nan), "ost_bisyrmsd": result.get("bisyrmsd", np.nan), "error": result.get("error", "")})
        pd.DataFrame(rows).to_csv(TABLE, index=False)
        print(f"{r.compound} sample {r.sample}: ours {r.lddt_pli:.3f}  OpenStructure {result.get('lddt_pli', float('nan')):.3f}", flush=True)
    result = pd.DataFrame(rows)
    scored = result.dropna(subset=["ost"])
    rho = scored.ours.corr(scored.ost, method="spearman")
    mad = float((scored.ours - scored.ost).abs().mean())
    agreement = pairs_agreement(scored)
    print(f"\n{len(scored)} of {len(result)} poses scored by OpenStructure; Spearman {rho:.3f}; mean absolute difference {mad:.3f} (largest {float((scored.ours - scored.ost).abs().max()):.3f}); pose pairs ordered the same way {agreement:.1%}")
    met = rho >= MIN_RANK_CORRELATION and mad <= MAX_MEAN_ABSOLUTE_DIFFERENCE and agreement >= MIN_PAIR_AGREEMENT
    print(f"STRUCTURE GATE lddt_pli_agrees_with_openstructure (rank correlation >= {MIN_RANK_CORRELATION}, mean absolute difference <= {MAX_MEAN_ABSOLUTE_DIFFERENCE}, ordering agreement >= {MIN_PAIR_AGREEMENT:.0%}): {'MET' if met else 'NOT MET'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
