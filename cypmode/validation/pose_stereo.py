# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Whether a predicted pose carries the stereochemistry of the official SMILES (registered on 2026-10-24).

The scorer's PoseBusters call (the ``dock`` configuration) has no chirality check, so a pose of the wrong enantiomer passes all 22 checks with the same margin as the right one. Three of the 20 official ligands
were first published without stereochemistry, and a co-folding model that is given such a SMILES is free to draw either enantiomer, so the stereochemistry of a pose has to be read from its coordinates.

``reproduces_stereo`` does that for one ligand: the chirality is perceived from the 3D coordinates and the official SMILES, with its stereocentres and stereo double bonds, is used as a substructure query with
chirality switched on. A SMILES without stereo elements is reproduced by any pose. ``pdb_reproduces_stereo``, ``cif_reproduces_stereo`` and ``zip_reproduces_stereo`` read the ligand the way the scorer does
(the residue ``LIG``, bond orders from the SMILES, ``official_posebusters.ligand_copies``); a pose from which no ligand copy can be built gives ``None`` and is never eligible.

``python -m cypmode.validation.pose_stereo [--zip A.zip ...] [--pools]`` checks the poses of submission zips and the share of every pool that reproduces the official stereochemistry (the table of
``pose_margins``) and prints ``STEREO CHECK <what>: PASS|FAIL`` lines; the exit code is 1 when a check fails.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

OFFICIAL = Path("data/structure_track/inputs/structure_track_test_blinded.csv")


def unspecified_stereo(smiles: str) -> list[str]:
    """The stereo elements of a SMILES that it leaves open: ``"atom <index>"`` for a stereocentre without a label and ``"bond <index>"`` for a stereo double bond without one."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ["unparsable SMILES"]
    open_elements = [f"atom {idx}" for idx, label in Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False) if label == "?"]
    Chem.FindPotentialStereoBonds(mol)
    open_elements += [f"bond {b.GetIdx()}" for b in mol.GetBonds() if b.GetStereo() == Chem.BondStereo.STEREOANY]
    return open_elements


def has_stereo(smiles: str) -> bool:
    """Whether the SMILES carries any specified stereocentre or stereo double bond."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    specified_bonds = (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS)
    return bool(Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)) or any(b.GetStereo() in specified_bonds for b in mol.GetBonds())


def reproduces_stereo(copy: Chem.Mol, smiles: str) -> bool:
    """Whether the 3D ligand ``copy`` (heavy atoms or with hydrogens, bond orders set) has the stereochemistry of ``smiles``."""
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        return False
    if not has_stereo(smiles):
        return copy.HasSubstructMatch(template)
    mol = Chem.Mol(copy)
    Chem.AssignStereochemistryFrom3D(mol)
    mol = Chem.RemoveHs(mol, sanitize=False)
    return mol.HasSubstructMatch(template, useChirality=True)


def pdb_reproduces_stereo(pdb_path: str | Path, smiles: str) -> bool | None:
    """The verdict for the ligand residue ``LIG`` of a submission-format PDB (every copy must reproduce it); ``None`` when no ligand copy can be built from the SMILES."""
    from cypmode.validation.official_posebusters import ligand_copies

    try:
        copies = ligand_copies(pdb_path, smiles)
    except Exception:
        return None
    if not copies:
        return None
    return all(reproduces_stereo(copy, smiles) for copy in copies)


def cif_reproduces_stereo(cif_path: str | Path, smiles: str) -> bool | None:
    """The same for a predicted CIF, converted to the submission PDB exactly as the zip builder does."""
    from cypmode.challenge.structure_submission import convert_cif_to_pdb

    with tempfile.TemporaryDirectory() as tmp:
        pdb = Path(tmp) / "pose.pdb"
        try:
            convert_cif_to_pdb(Path(cif_path), pdb)
        except Exception:
            return None
        return pdb_reproduces_stereo(pdb, smiles)


def zip_reproduces_stereo(zip_path: str | Path, smiles_of: dict[str, str]) -> dict[str, bool | None]:
    """The verdict for every ``<compound>.pdb`` of a submission zip; a compound that is not in ``smiles_of`` gets ``None``."""
    out: dict[str, bool | None] = {}
    with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as tmp:
        for member in sorted(zf.namelist()):
            name = Path(member).stem
            path = Path(tmp) / Path(member).name
            path.write_bytes(zf.read(member))
            out[name] = pdb_reproduces_stereo(path, smiles_of[name]) if name in smiles_of else None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", nargs="*", default=[], help="submission zips whose poses must reproduce the official stereochemistry")
    parser.add_argument("--pools", action="store_true", help="the share of every ligand's pool that reproduces it (table of pose_margins)")
    parser.add_argument("--official", default=str(OFFICIAL))
    args = parser.parse_args()
    official = pd.read_csv(args.official)
    smiles_of = dict(zip(official.Molecule_Name, official.SMILES, strict=True))
    chiral = [n for n, s in smiles_of.items() if has_stereo(s)]
    open_ligands = {n: unspecified_stereo(s) for n, s in smiles_of.items() if unspecified_stereo(s)}
    print(f"ligands with specified stereochemistry: {len(chiral)} of {len(smiles_of)} ({', '.join(chiral)})")
    failed = False
    verdict = "PASS" if not open_ligands else "FAIL"
    print(f"STEREO CHECK official file has no unspecified stereo element: {verdict}" + (f" ({open_ligands})" if open_ligands else ""))
    failed |= bool(open_ligands)
    for path in args.zip:
        result = zip_reproduces_stereo(path, smiles_of)
        bad = sorted(n for n, ok in result.items() if ok is not True)
        print(f"STEREO CHECK {Path(path).name}: {'PASS' if not bad and len(result) == len(smiles_of) else 'FAIL'} ({len(result) - len(bad)} of {len(smiles_of)} poses reproduce the official stereochemistry"
              + (f"; not: {bad}" if bad else "") + ")")
        failed |= bool(bad) or len(result) != len(smiles_of)
    if args.pools:
        from cypmode.validation.pose_margins import TABLE

        pools = pd.read_csv(TABLE)
        if "stereo_ok" not in pools:
            print("STEREO CHECK pools: FAIL (the table of pose_margins has no stereo_ok column; run pose_margins again)")
            return 1
        for ligand in chiral:
            sub = pools[pools.ligand == ligand]
            ok = int(sub.stereo_ok.astype(bool).sum())
            print(f"  {ligand}: {ok} of {len(sub)} pool poses reproduce the official stereochemistry")
        total = pools[pools.ligand.isin(chiral)]
        print(f"STEREO POOLS share of the pool poses of the {len(chiral)} chiral ligands that reproduce the official stereochemistry: {total.stereo_ok.astype(bool).mean():.3f} ({int(total.stereo_ok.astype(bool).sum())} of {len(total)})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
