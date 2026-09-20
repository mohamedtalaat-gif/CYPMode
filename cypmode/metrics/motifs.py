"""Structural-alert screen for CYP450 Type II (heme-coordinating) inhibition.

Type II CYP inhibition occurs when a ligand nitrogen lone pair coordinates
directly to the heme iron, displacing the axial water and producing a much
tighter complex than ordinary Type I (substrate-pocket) binding. The classic
coordinating groups are azole (imidazole, triazole), pyridine, and thiazole
nitrogens (de Groot, Drug Discov. Today, 2006, 11:601-606; Correia & Ortiz de
Montellano, ch. 7 in de Montellano (ed.), Cytochrome P450: Structure,
Mechanism, and Biochemistry, 3rd ed., 2005).

This is a fast substructure screen in the QSPR/structural-alert tradition
(Leach, Molecular Modelling: Principles and Applications, 2001, ch. 12,
section 12.2), not a mechanism predictor: a motif match flags a compound as
*capable* of heme coordination, not confirmed to coordinate in a specific
CYP isoform's active site. Ring systems can match a pattern by coincidence
of atom connectivity without behaving like the discrete azole the rule was
derived from — see cypmode/validation/structures.py, which checks motif
calls against physics-based (Boltz-2) structure prediction restricted to
compounds with independently solved reference structures.
"""

from rdkit import Chem

TYPE_II_MOTIFS = {
    "imidazole": "c1cnc[nH]1",
    "imidazole_Nsub": "c1cncn1",
    "triazole_1_2_4": "c1ncnn1",
    "triazole_1_2_3": "c1cnnn1",
    "pyridine": "c1ccncc1",
    "thiazole": "c1cscn1",
}

_COMPILED_PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in TYPE_II_MOTIFS.items()}


def detect_motifs(smiles: str) -> list[str]:
    """Return the names of all Type II heme-coordinating motifs found in smiles."""
    if not smiles or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles!r}")
    return [name for name, pattern in _COMPILED_PATTERNS.items() if mol.HasSubstructMatch(pattern)]


def classify_mechanism(smiles: str) -> str:
    """Coarse Type I / Type II call based on presence of a coordinating motif."""
    motifs = detect_motifs(smiles)
    if motifs:
        return "Type II (heme-coordinating motif present)"
    return "No coordinating motif detected (Type I or non-classical)"
