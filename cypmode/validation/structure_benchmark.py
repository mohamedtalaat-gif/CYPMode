"""A crystal-anchored benchmark for tuning the structure engine.

Five panel compounds are too few to learn which engine, sampling setup or selection rule to
trust, and the official compounds have no public answer. The PDB does hold co-crystal poses for
~100 distinct CYP3A4 ligands (cypmode.validation.reference_ligands), so this module picks the
most chemically DIVERSE of them -- maximin on Morgan fingerprints, seeded with the compounds we
already predicted so every new pick is far from those and from each other -- and writes them as a
batch input. The pick is deterministic and criteria-driven (drug-sized, well resolved, not
organometallic), not a hand-written list.

Caveat kept visible: Boltz-2 / OpenFold3 may have seen some of these PDB entries in training, so
absolute accuracy here is optimistic; the benchmark is for ranking engines and selection rules,
not for forecasting leaderboard scores.
"""

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.SimDivFilters import MaxMinPicker

from cypmode.validation.reference_ligands import load_reference_set

RDLogger.DisableLog("rdApp.*")

MIN_HEAVY, MAX_HEAVY, MAX_RESOLUTION = 20, 60, 3.0
ORGANIC = set("CNOSFClBrIPH")


def candidates(references: list[dict]) -> pd.DataFrame:
    best: dict[str, dict] = {}
    for ref in references:
        if not ref.get("smiles") or not MIN_HEAVY <= ref["n_heavy"] <= MAX_HEAVY or (ref["resolution"] or 99) > MAX_RESOLUTION:
            continue
        mol = Chem.MolFromSmiles(ref["smiles"])
        if mol is None or {a.GetSymbol() for a in mol.GetAtoms()} - ORGANIC or len(Chem.GetMolFrags(mol)) > 1:
            continue
        if ref["comp_id"] not in best or ref["resolution"] < best[ref["comp_id"]]["resolution"]:
            best[ref["comp_id"]] = {**ref, "mw": Descriptors.MolWt(mol)}
    return pd.DataFrame(best.values())


def pick_diverse(table: pd.DataFrame, seed_smiles: list[str], n: int, random_seed: int = 0) -> pd.DataFrame:
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in [*seed_smiles, *table["smiles"]]]
    picker = MaxMinPicker()
    n_seed = len(seed_smiles)
    chosen = picker.LazyBitVectorPick(fingerprints, len(fingerprints), n_seed + n, firstPicks=list(range(n_seed)), seed=random_seed)
    return table.iloc[[i - n_seed for i in chosen if i >= n_seed]].reset_index(drop=True)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--out", default="data/structure_track/inputs/structure_benchmark.csv")
    args = parser.parse_args()
    references = load_reference_set()
    panel = [p.stem.removeprefix("cyp3a4_") for p in Path("data/structure_track/inputs").glob("cyp3a4_*.yaml") if not p.stem.startswith("cyp3a4_OCNT")]
    import yaml

    seeds = []
    for name in panel:
        spec = yaml.safe_load(Path(f"data/structure_track/inputs/cyp3a4_{name}.yaml").read_text())
        seeds += [e["ligand"]["smiles"] for e in spec["sequences"] if "ligand" in e and "smiles" in e["ligand"]]
    table = candidates(references)
    picked = pick_diverse(table, seeds, args.n)
    out = pd.DataFrame({"Molecule_Name": [f"bench-{c}" for c in picked["comp_id"]], "SMILES": picked["smiles"]})
    out.to_csv(args.out, index=False)
    print(picked[["comp_id", "name", "n_heavy", "mw", "resolution", "pdb_id"]].to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    _main()
