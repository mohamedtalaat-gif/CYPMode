# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Cryo-EM protein templates for the pose pool (job 135).

The PDB holds five cryo-EM structures of human CYP3A4 (UniProt P08684): 38FU (apo), 38FV (ritonavir), 38FW (ketoconazole), 38FX (azamulin) and
38FY (vardenafil), released 2026-09-30 at 3.2-3.7 A. They are trimers; chain A is used as a one-chain protein template, like a crystal monomer.
The crystal conformations used by the "conformers" lever of ``diversity_pilot`` come from X-ray models, so the cryo-EM entries add pocket
shapes that the crystal set may not contain.

The lever is judged exactly like the other pilot levers (``diversity_pilot.verdicts``): it must widen the pose pool and reach a better pose.
As for crystal templates, an entry whose ligand is the benchmark ligand or a close analog of it is never offered for that ligand.

Usage:
    python -m cypmode.validation.cryoem_templates fetch                # download the five mmCIF files from RCSB into data/structure_track/inputs/cryoem_templates
    python -m cypmode.validation.cryoem_templates prepare              # ligand records and pocket shapes -> data/structure_track/inputs/cyp3a4_cryoem_templates.json
    python -m cypmode.validation.cryoem_templates build --only NAME... --out CSV   # batch rows <ligand>-em1.. for the named ligands
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.validation.conformers import (
    CRYOEM_DIR,
    CRYOEM_TEMPLATE_CHAIN,
    excluded_entries,
    pick_conformers,
)
from cypmode.validation.pose_rescoring_validation import NOVEL_SIMILARITY
from cypmode.validation.reference_ligands import POCKET_CA_RADIUS_A

ENTRIES = ("38FU", "38FV", "38FW", "38FX", "38FY")
RECORDS = Path("data/structure_track/inputs/cyp3a4_cryoem_templates.json")


def chain_pocket(structure, chain_id: str = CRYOEM_TEMPLATE_CHAIN) -> dict[str, list[float]]:
    """{residue number: C-alpha} of the pocket residues of one protein chain: within POCKET_CA_RADIUS_A of that chain's heme iron.

    The heme of the chain is the one whose iron is nearest to the chain's own residues (the trimer has three).
    """
    model = structure[0]
    chain = model[chain_id]
    ca = {r.id[1]: r["CA"].coord for r in chain if r.id[0] == " " and r.id[2] == " " and "CA" in r}
    irons = [a.coord for r in model.get_residues() if r.resname == "HEM" for a in r if a.get_name().strip().upper() == "FE"]
    if not irons or not ca:
        return {}
    centre = np.mean(list(ca.values()), axis=0)
    iron = min(irons, key=lambda fe: float(np.linalg.norm(fe - centre)))
    return {str(k): [round(float(x), 2) for x in v] for k, v in ca.items() if np.linalg.norm(v - iron) <= POCKET_CA_RADIUS_A}


def build_records(directory: Path = CRYOEM_DIR, out: Path = RECORDS) -> dict:
    """Ligand records (same fields as the crystal reference set) and the chain-A pocket of every entry; writes the compact JSON."""
    from Bio.PDB.MMCIFParser import MMCIFParser

    from cypmode.validation.reference_ligands import (
        extract_active_site_ligands,
        fetch_ligand_inventory,
    )

    inventory = fetch_ligand_inventory(list(ENTRIES))
    parser = MMCIFParser(QUIET=True)
    ligands, pockets = [], {}
    for pdb_id in ENTRIES:
        structure = parser.get_structure(pdb_id, str(Path(directory) / f"{pdb_id}.cif"))
        pockets[pdb_id] = chain_pocket(structure)
        if inventory.get(pdb_id, {}).get("ligands"):
            ligands.extend(extract_active_site_ligands(structure, pdb_id, inventory[pdb_id]))
    record = {"source": "RCSB cryo-EM entries of UniProt P08684", "template_chain": CRYOEM_TEMPLATE_CHAIN, "entries": list(ENTRIES), "ligands": ligands, "pockets": pockets}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record) + "\n")
    return record


def load_records(path: Path = RECORDS) -> dict:
    return json.loads(Path(path).read_text())


def pockets_of(record: dict) -> dict[str, dict[int, np.ndarray]]:
    return {pdb_id: {int(k): np.asarray(v, dtype=float) for k, v in pocket.items()} for pdb_id, pocket in record["pockets"].items() if pocket}


def build_cryoem_table(ligands: pd.DataFrame, record: dict, n_runs: int = 3, directory: Path = CRYOEM_DIR) -> pd.DataFrame:
    """One row per run: ``<ligand>-em<k>`` with a cryo-EM entry as template (the entries differing most in pocket shape, never one
    whose ligand is the compound or a close analog of it)."""
    pockets = pockets_of(record)
    rows = []
    for _, ligand in ligands.iterrows():
        banned = excluded_entries(ligand.SMILES, record["ligands"], NOVEL_SIMILARITY)
        for k, pdb_id in enumerate(pick_conformers(pockets, n_runs, banned, cache_dir=directory), start=1):
            rows.append({"Molecule_Name": f"{ligand.Molecule_Name}-em{k}", "SMILES": ligand.SMILES, "Template_PDB": pdb_id, "Diffusion_Samples": 1})
    return pd.DataFrame(rows, columns=["Molecule_Name", "SMILES", "Template_PDB", "Diffusion_Samples"])


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download the entries that are not on disk yet (public PDB files)")
    sub.add_parser("prepare", help="extract ligand records and pocket shapes from the downloaded entries")
    build = sub.add_parser("build", help="write the batch CSV of the cryo-EM template runs")
    build.add_argument("--benchmark", default="data/structure_track/inputs/structure_benchmark.csv")
    build.add_argument("--only", nargs="+", required=True)
    build.add_argument("--runs", type=int, default=3)
    build.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "fetch":
        from cypmode.validation.reference_ligands import download_mmcif

        for pdb_id in ENTRIES:
            print(download_mmcif(pdb_id, CRYOEM_DIR))
        return
    if args.command == "prepare":
        record = build_records()
        print(f"{len(record['entries'])} entries, {len(record['ligands'])} ligand copies; pocket residues per entry: " + ", ".join(f"{k}={len(v)}" for k, v in record["pockets"].items()))
        return
    ligands = pd.read_csv(args.benchmark)
    table = build_cryoem_table(ligands[ligands.Molecule_Name.isin(args.only)], load_records(), args.runs)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    _main()
