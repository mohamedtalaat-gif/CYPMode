"""PoseBusters exactly as the challenge's structure scorer applies it, with the numeric margin of every check.

The organisers' scorer (the evaluation code published in OpenADMET/CYP-Challenge-Tutorial, Apache-2.0) validates a submitted complex like this:

* the ligand copies are read from the residue named ``LIG`` of the submitted PDB and every disconnected copy gets its bond orders from the compound's
  SMILES (a PDB carries none);
* the receptor handed to PoseBusters is the whole predicted complex minus ``LIG`` (protein, heme, ions, waters);
* ``PoseBusters(config="dock").bust(mol_pred=ligand, mol_true=template, mol_cond=receptor)`` yields 22 yes/no checks;
* a pose that fails more than ``MAX_FAILURES`` (0) of them scores LDDT-PLI 0, LDDT-LP 0 and the BiSyRMSD penalty.

The project's ``pose_validity`` module runs the same checks but not through this path (it hands PoseBusters a reference-free ligand and converts the
structure itself). This module follows the scorer's path step by step, so a margin read here is the margin the scorer will see, and it reports how far each
check is from its threshold. A pose that passes by 0.003 on the minimum protein distance passes today and fails the day a library version moves an atomic
radius by a hundredth; ``MARGIN_FLOOR`` is the distance margin a pose must clear to be called safe.

The bond-order assignment is written from the behaviour the organisers document (every topological match of the template is tried, hydrogens are matched
by heavy-atom identity, a distance-guessed bond that the template does not have is dropped); ``tests/test_official_posebusters.py`` pins the result on a
real submitted pose.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

CONFIG = "dock"
MAX_FAILURES = 0
CLASH_CUTOFF = 0.75  # the threshold of the relative-distance checks in PoseBusters' dock configuration
MARGIN_FLOOR = 0.80  # a pose is "safe" when every relative distance to the receptor is at least this (0.05 above the threshold)

DISTANCE_KEYS = {
    "protein": "most_extreme_relative_distance_protein",
    "organic_cofactors": "most_extreme_relative_distance_organic_cofactors",
    "inorganic_cofactors": "most_extreme_relative_distance_inorganic_cofactors",
}
# numeric column -> (threshold, direction): "min" means the value must be at least the threshold, "max" at most
OTHER_LIMITS = {
    "volume_overlap_protein": (0.075, "max"),
    "volume_overlap_organic_cofactors": (0.075, "max"),
    "volume_overlap_inorganic_cofactors": (0.075, "max"),
    "aromatic_ring_maximum_distance_from_plane": (0.25, "max"),
    "double_bond_maximum_distance_from_plane": (0.25, "max"),
    "energy_ratio": (100.0, "max"),
}


@dataclass
class OfficialReport:
    """The scorer's verdict on one submitted pose and how far each numeric check is from failing."""

    passed: bool
    failed: list[str] = field(default_factory=list)
    n_checks: int = 0
    n_copies: int = 0
    margins: dict[str, float] = field(default_factory=dict)  # slack to the threshold in the unit of the check (positive = inside)
    distances: dict[str, float] = field(default_factory=dict)  # relative distance to protein / organic cofactors / inorganic cofactors
    error: str = ""

    @property
    def distance_margin(self) -> float:
        """The smallest relative distance to the receptor over its parts (NaN when none could be computed)."""
        values = [v for v in self.distances.values() if np.isfinite(v)]
        return min(values) if values else float("nan")

    @property
    def safe(self) -> bool:
        return self.passed and bool(self.distance_margin >= MARGIN_FLOOR)


def bond_orders_from_template(template: Chem.Mol, mol: Chem.Mol) -> Chem.Mol:
    """``mol`` (coordinates, no reliable bond orders) with the bond orders, aromaticity and charges of ``template`` (from SMILES).

    Hydrogens are ignored while matching and carried back by atom identity; every topological match is tried until one sanitises; a bond between
    two matched atoms that the template does not have (guessed from distance) is removed. Raises ``ValueError`` when no match sanitises.
    """
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetIntProp("_origin", atom.GetIdx())
    heavy = Chem.RemoveHs(mol, sanitize=False)
    origin = [a.GetIntProp("_origin") for a in heavy.GetAtoms()]
    if not heavy.GetSubstructMatch(template):
        flat_template, flat_heavy = Chem.Mol(template), Chem.Mol(heavy)
        for m in (flat_template, flat_heavy):
            for bond in m.GetBonds():
                bond.SetBondType(Chem.BondType.SINGLE)
                bond.SetIsAromatic(False)
            for atom in m.GetAtoms():
                atom.SetFormalCharge(0)
        matches = flat_heavy.GetSubstructMatches(flat_template, uniquify=False)
        if not matches:
            raise ValueError("the template does not match the ligand's connectivity")
        built, errors = None, []
        for match in matches:
            rw = Chem.RWMol(heavy)
            wanted = {frozenset((match[b.GetBeginAtomIdx()], match[b.GetEndAtomIdx()])) for b in template.GetBonds()}
            for bond in list(rw.GetBonds()):
                pair = frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                if pair <= set(match) and pair not in wanted:
                    rw.RemoveBond(*pair)
            for bond in template.GetBonds():
                a, b = match[bond.GetBeginAtomIdx()], match[bond.GetEndAtomIdx()]
                if rw.GetBondBetweenAtoms(a, b) is None:
                    rw.AddBond(a, b, bond.GetBondType())
                new = rw.GetBondBetweenAtoms(a, b)
                new.SetBondType(bond.GetBondType())
                new.SetIsAromatic(bond.GetIsAromatic())
            for atom in template.GetAtoms():
                target = rw.GetAtomWithIdx(match[atom.GetIdx()])
                target.SetIsAromatic(atom.GetIsAromatic())
                target.SetNumExplicitHs(atom.GetNumExplicitHs())
                target.SetFormalCharge(atom.GetFormalCharge())
            candidate = rw.GetMol()
            try:
                Chem.SanitizeMol(candidate)
            except Exception as exc:  # noqa: BLE001 - the next match may sanitise
                errors.append(str(exc))
                continue
            built = candidate
            break
        if built is None:
            raise ValueError(f"no topological match sanitised ({len(matches)} tried): {'; '.join(errors[:2])}")
        heavy = built
    if mol.GetNumAtoms() == heavy.GetNumAtoms():
        Chem.SanitizeMol(heavy)
        return heavy
    full = Chem.RWMol(mol)
    kept = set(origin)
    heavy_pairs = {frozenset((origin[b.GetBeginAtomIdx()], origin[b.GetEndAtomIdx()])) for b in heavy.GetBonds()}
    for bond in list(full.GetBonds()):
        pair = frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        if pair <= kept and pair not in heavy_pairs:
            full.RemoveBond(*pair)
    for bond in heavy.GetBonds():
        a, b = origin[bond.GetBeginAtomIdx()], origin[bond.GetEndAtomIdx()]
        if full.GetBondBetweenAtoms(a, b) is None:
            full.AddBond(a, b, bond.GetBondType())
        new = full.GetBondBetweenAtoms(a, b)
        new.SetBondType(bond.GetBondType())
        new.SetIsAromatic(bond.GetIsAromatic())
    for atom in heavy.GetAtoms():
        target = full.GetAtomWithIdx(origin[atom.GetIdx()])
        target.SetIsAromatic(atom.GetIsAromatic())
        target.SetFormalCharge(atom.GetFormalCharge())
    result = full.GetMol()
    Chem.SanitizeMol(result)
    return result


def ligand_copies(pdb_path: str | Path, smiles: str, resname: str = "LIG") -> list[Chem.Mol]:
    """Every copy of the ligand residue of a submitted PDB, each with bond orders from ``smiles``; a copy that cannot be assigned is skipped."""
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
    template = Chem.MolFromSmiles(smiles)
    if mol is None or template is None:
        return []
    residue = Chem.SplitMolByPDBResidues(mol).get(resname)
    if residue is None:
        return []
    copies = []
    for part in Chem.GetMolFrags(residue, asMols=True, sanitizeFrags=False):
        try:
            copies.append(bond_orders_from_template(template, part))
        except Exception:  # noqa: BLE001 - one copy that does not match is skipped, as in the scorer
            continue
    return copies


def receptor(pdb_path: str | Path, resname: str = "LIG") -> Chem.Mol | None:
    """The whole predicted complex minus the ligand residue: what the scorer passes to PoseBusters as the conditioning molecule."""
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
    if mol is None:
        return None
    parts = Chem.SplitMolByPDBResidues(mol)
    parts.pop(resname, None)
    out = None
    for part in parts.values():
        out = part if out is None else Chem.CombineMols(out, part)
    return out


def _report_for(copy: Chem.Mol, template: Chem.Mol, context: Chem.Mol) -> OfficialReport:
    from posebusters import PoseBusters

    verdict = PoseBusters(config=CONFIG).bust(mol_pred=copy, mol_true=template, mol_cond=context)
    boolean = [c for c in verdict.columns if verdict[c].dtype == bool]
    row = verdict[boolean].iloc[0]
    failed = sorted(row.index[~row].tolist())
    if len(failed) > MAX_FAILURES:  # a pose that is already rejected has no margin to report
        return OfficialReport(passed=False, failed=failed, n_checks=len(boolean), n_copies=1)
    full = PoseBusters(config=CONFIG).bust(mol_pred=copy, mol_true=template, mol_cond=context, full_report=True).iloc[0]
    distances = {part: float(full[col]) for part, col in DISTANCE_KEYS.items() if col in full.index and not np.isnan(float(full[col]))}
    margins = {f"distance_{part}": value - CLASH_CUTOFF for part, value in distances.items()}
    for col, (limit, direction) in OTHER_LIMITS.items():
        if col in full.index and not np.isnan(float(full[col])):
            value = float(full[col])
            margins[col] = limit - value if direction == "max" else value - limit
    return OfficialReport(passed=len(failed) <= MAX_FAILURES, failed=failed, n_checks=len(boolean), n_copies=1, margins=margins, distances=distances)


def check_pdb(pdb_path: str | Path, smiles: str) -> OfficialReport:
    """The scorer's PoseBusters verdict on a submission-format PDB (ligand residue LIG). With several ligand copies the worst one decides."""
    try:
        copies = ligand_copies(pdb_path, smiles)
        context = receptor(pdb_path)
        template = Chem.MolFromSmiles(smiles)
        if not copies or context is None or template is None:
            return OfficialReport(passed=False, error="no ligand copy could be built from the SMILES, or no receptor")
        reports = [_report_for(copy, template, context) for copy in copies]
    except Exception as exc:  # noqa: BLE001 - an unprocessable pose counts as failed
        return OfficialReport(passed=False, error=f"{type(exc).__name__}: {exc}")
    worst = min(reports, key=lambda r: (r.passed, r.distance_margin if np.isfinite(r.distance_margin) else -1.0))
    worst.n_copies = len(reports)
    return worst


def check_cif(cif_path: str | Path, smiles: str) -> OfficialReport:
    """Convert a predicted CIF to the submission PDB exactly as the zip builder does, then check that file."""
    from cypmode.challenge.structure_submission import convert_cif_to_pdb

    with tempfile.TemporaryDirectory() as tmp:
        pdb = Path(tmp) / "pose.pdb"
        try:
            convert_cif_to_pdb(Path(cif_path), pdb)
        except Exception as exc:  # noqa: BLE001
            return OfficialReport(passed=False, error=f"conversion failed: {type(exc).__name__}: {exc}")
        return check_pdb(pdb, smiles)


def check_zip(zip_path: str | Path, smiles_of: dict[str, str]) -> dict[str, OfficialReport]:
    """The scorer's verdict on every ``<compound>.pdb`` of a submission zip."""
    import zipfile

    out = {}
    with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as tmp:
        for member in sorted(zf.namelist()):
            name = Path(member).stem
            path = Path(tmp) / Path(member).name
            path.write_bytes(zf.read(member))
            out[name] = check_pdb(path, smiles_of[name]) if name in smiles_of else OfficialReport(passed=False, error="compound not in the official list")
    return out


def main() -> int:
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser(description="Check a structure-track zip the way the organisers' scorer does.")
    parser.add_argument("zip", help="submission zip with one <Molecule_Name>.pdb per official ligand")
    parser.add_argument("--official", default="data/structure_track/inputs/structure_track_test_blinded.csv")
    args = parser.parse_args()
    official = pd.read_csv(args.official)
    reports = check_zip(args.zip, dict(zip(official.Molecule_Name, official.SMILES, strict=True)))
    rows = [{"compound": k, "passes": r.passed, "checks": r.n_checks, "min_distance": round(r.distance_margin, 3), "safe": r.safe, "failed": ",".join(r.failed) or r.error or "-"} for k, r in reports.items()]
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    missing = sorted(set(official.Molecule_Name) - set(reports))
    unsafe = table[~table.safe]
    print(f"\n{len(table)} structures; {int(table.passes.sum())} pass all {int(table.checks.max())} checks; {len(unsafe)} below the margin floor {MARGIN_FLOOR}; official ligands without a structure: {missing or 'none'}")
    return 0 if table.passes.all() and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
