"""Batch input for the conformer-conditioning experiment on the crystal benchmark.

One row per (ligand, conformation): the same SMILES run once per template conformation, each template
chosen by cypmode.validation.conformers with the ligand's own entries and close analogs left out. The
rows are named ``<ligand>-cf<k>`` and carry the template's PDB id in ``Template_PDB``, which
``cypmode.validation.batch`` turns into a Boltz template for that run.
"""

import argparse
import re

import pandas as pd

from cypmode.validation.conformers import entry_pockets, excluded_entries, pick_conformers
from cypmode.validation.pose_rescoring_validation import NOVEL_SIMILARITY
from cypmode.validation.reference_ligands import load_reference_set

CONFORMER_SUFFIX = re.compile(r"-cf\d+$")


def build_conformer_table(ligands: pd.DataFrame, references: list[dict], n_conformers: int = 3) -> pd.DataFrame:
    pockets = entry_pockets(references)
    rows = []
    for _, ligand in ligands.iterrows():
        banned = excluded_entries(ligand.SMILES, references, NOVEL_SIMILARITY)
        for k, pdb_id in enumerate(pick_conformers(pockets, n_conformers, banned), start=1):
            rows.append({"Molecule_Name": f"{ligand.Molecule_Name}-cf{k}", "SMILES": ligand.SMILES, "Template_PDB": pdb_id})
    return pd.DataFrame(rows, columns=["Molecule_Name", "SMILES", "Template_PDB"])


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/structure_track/inputs/structure_benchmark.csv")
    parser.add_argument("--only", nargs="*", help="Molecule_Names to include (default: all)")
    parser.add_argument("--n", type=int, default=3, help="conformations per ligand")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    ligands = pd.read_csv(args.benchmark)
    if args.only:
        ligands = ligands[ligands.Molecule_Name.isin(args.only)]
    table = build_conformer_table(ligands, load_reference_set(), args.n)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    _main()
