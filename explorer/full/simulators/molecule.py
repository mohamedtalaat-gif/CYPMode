# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Section 2.1: a molecule as text and as a graph. A SMILES string becomes a graph of atoms and bonds, a drawing, the bits of a Morgan fingerprint and, with the project's own function, the nitrogen
that could bind the heme iron."""

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from cypmode.metrics.motifs import TYPE_II_MOTIFS
from cypmode.validation.build_input import heme_donor_atom

RADIUS, N_BITS = 2, 2048  # the fingerprint of the project (section 3.2)


def parse(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles.strip()) if smiles and smiles.strip() else None


def drawing(mol: Chem.Mol, atoms: tuple[int, ...] = (), bonds: tuple[int, ...] = (), size: tuple[int, int] = (460, 340), indices: bool = False, atom_colors: dict | None = None) -> str:
    """The SVG of the molecule with the given atoms and bonds highlighted, each atom in its own colour when ``atom_colors`` ({atom: (r, g, b)}) is given."""
    canvas = rdMolDraw2D.MolDraw2DSVG(*size)
    canvas.drawOptions().addAtomIndices = indices
    if atom_colors:
        canvas.DrawMolecule(mol, highlightAtoms=list(atom_colors), highlightAtomColors=atom_colors, highlightBonds=[])
    else:
        canvas.DrawMolecule(mol, highlightAtoms=list(atoms), highlightBonds=list(bonds))
    canvas.FinishDrawing()
    return canvas.GetDrawingText()


def atom_rows(mol: Chem.Mol) -> list[dict]:
    return [{"index": a.GetIdx(), "element": a.GetSymbol(), "aromatic": a.GetIsAromatic(), "heavy neighbours": a.GetDegree(), "hydrogens": a.GetTotalNumHs(), "charge": a.GetFormalCharge(), "in a ring": a.IsInRing()}
            for a in mol.GetAtoms()]


def bond_rows(mol: Chem.Mol) -> list[dict]:
    return [{"from": b.GetBeginAtomIdx(), "to": b.GetEndAtomIdx(), "order": str(b.GetBondType()).lower(), "aromatic": b.GetIsAromatic(), "in a ring": b.IsInRing()} for b in mol.GetBonds()]


def summary(mol: Chem.Mol) -> dict:
    return {"canonical SMILES": Chem.MolToSmiles(mol), "formula": rdMolDescriptors.CalcMolFormula(mol), "heavy atoms": mol.GetNumHeavyAtoms(), "bonds": mol.GetNumBonds(), "rings": rdMolDescriptors.CalcNumRings(mol),
            "molecular weight": round(Descriptors.MolWt(mol), 2)}


def morgan_bits(mol: Chem.Mol) -> dict[int, list[tuple[int, int]]]:
    """{bit: [(centre atom, radius)]} of the Morgan fingerprint (radius 2, 2,048 bits) of the molecule."""
    info: dict = {}
    rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS, bitInfo=info)
    return {bit: [(int(center), int(radius)) for center, radius in environments] for bit, environments in sorted(info.items())}


def environment(mol: Chem.Mol, center: int, radius: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The atoms and the bonds of the circular neighbourhood that a bit stands for."""
    if radius == 0:
        return (center,), ()
    bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center)
    atoms = {center}
    for index in bonds:
        bond = mol.GetBondWithIdx(index)
        atoms |= {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
    return tuple(sorted(atoms)), tuple(bonds)


def type_ii_motifs(mol: Chem.Mol) -> list[dict]:
    """The Type II motifs (a nitrogen that can bind the iron) found in the molecule, with the atoms of each match."""
    found = []
    for name, smarts in TYPE_II_MOTIFS.items():
        for match in mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)):
            found.append({"motif": name, "atoms": tuple(match)})
    return found


def donor(smiles: str) -> dict | None:
    """The nitrogen that the restraint of the structure track would put on the iron: the project's own function ``heme_donor_atom``."""
    result = heme_donor_atom(smiles)
    return None if result is None else {"atom index": result[0], "Boltz-2 atom name": result[1]}
