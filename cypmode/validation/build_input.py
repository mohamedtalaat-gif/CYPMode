# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Builds Boltz-2 YAML inputs (protein + heme, covalently bonded) for any ligand.

The four data/structure_track/inputs/cyp3a4_*.yaml files were originally hand-written,
one at a time, with the coordinating cysteine's position located once,
inline, and typed into each file. That doesn't scale past four known
compounds and can't be re-verified against a different protein construct.
This module makes both steps real, callable code.
"""

from collections.abc import Sequence
from pathlib import Path

CYS_MOTIF_PATTERN = r"F..G...C.G"  # conserved P450 heme-binding signature (Leach, ch. 12)

HEME_CCD_CODE = "HEM"
HEME_IRON_ATOM = "FE"
CYSTEINE_SULFUR_ATOM = "SG"


def find_coordinating_cysteine(protein_sequence: str) -> int:
    """Locate the axial heme-ligating cysteine via the conserved FxxGxxxCxG motif.

    Returns the cysteine's 1-indexed residue position. Raises ValueError if
    the motif isn't found exactly once -- this is a real structural feature
    to verify per protein/construct, not something to guess or reuse across
    a different numbering scheme.
    """
    import re

    matches = list(re.finditer(CYS_MOTIF_PATTERN, protein_sequence))
    if not matches:
        raise ValueError(
            "no FxxGxxxCxG P450 heme-binding signature found in this sequence -- "
            "is this actually a cytochrome P450, and is the sequence complete?"
        )
    if len(matches) > 1:
        positions = [m.start() + 8 for m in matches]
        raise ValueError(
            f"found {len(matches)} candidate FxxGxxxCxG matches (Cys at residues "
            f"{positions}) -- ambiguous, resolve by hand which one is the real "
            "axial ligand rather than guessing"
        )
    return matches[0].start() + 8  # Cys is the 8th character of the 10-char motif, 1-indexed


DONOR_RESTRAINTS = ("bond", "contact")
CONTACT_MAX_DISTANCE_A = 4.0  # Boltz accepts 4-20 A; 4 A is its tightest contact


def boltz_atom_name(smiles: str, atom_index: int) -> str:
    """The name Boltz-2 gives atom ``atom_index`` (RDKit order of the plain SMILES) of a SMILES ligand.

    Boltz adds hydrogens, ranks the atoms canonically and names each one element + (rank + 1); a
    constraint has to quote that name, so it is reproduced here rather than typed by hand.
    """
    from rdkit import Chem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ranks = list(Chem.CanonicalRankAtoms(mol))
    return mol.GetAtomWithIdx(atom_index).GetSymbol().upper() + str(ranks[atom_index] + 1)


def heme_donor_atom(smiles: str) -> tuple[int, str] | None:
    """(RDKit atom index, Boltz atom name) of the nitrogen most likely to coordinate the heme iron, or None.

    Candidates are the unsubstituted aromatic nitrogens (two heavy neighbours, no hydrogen) inside a Type II
    motif (imidazole, triazole, pyridine, thiazole). With several candidates the most accessible wins:
    fewest heavy atoms within two bonds, so a ring nitrogen flanked by substituents loses to a bare one.
    """
    from rdkit import Chem

    from cypmode.metrics.motifs import TYPE_II_MOTIFS

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles!r}")
    candidates = set()
    for smarts in TYPE_II_MOTIFS.values():
        for match in mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)):
            for idx in match:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == "N" and atom.GetIsAromatic() and atom.GetDegree() == 2 and atom.GetTotalNumHs() == 0:
                    candidates.add(idx)
    if not candidates:
        return None

    def crowding(idx: int) -> tuple[int, int]:
        within_two = {n.GetIdx() for a in mol.GetAtomWithIdx(idx).GetNeighbors() for n in [a, *a.GetNeighbors()]} - {idx}
        return len(within_two), idx

    best = min(candidates, key=crowding)
    return best, boltz_atom_name(smiles, best)


def _template_yaml(template: "str | Path | tuple[str | Path, str]", protein_chain: str) -> str:
    path, template_chain = template if isinstance(template, tuple) else (template, None)
    text = f"  - cif: {Path(path).resolve()}\n    chain_id: {protein_chain}\n"
    return text + (f"    template_id: {template_chain}\n" if template_chain else "")


def build_boltz_yaml(
    protein_sequence: str,
    ligand_smiles: str,
    coordinating_cys: int,
    protein_chain: str = "A",
    ligand_chain: str = "B",
    heme_chain: str = "C",
    predict_affinity: bool = True,
    donor_restraint: str | None = None,
    templates: "Sequence[str | Path | tuple[str | Path, str]] | None" = None,
    protein_msa: "str | Path | None" = None,
) -> str:
    """Return Boltz-2 YAML text: protein + ligand + heme, heme covalently bonded
    to coordinating_cys's sulfur, and by default no constraint on the ligand's own position.

    donor_restraint steers the ligand onto the iron through its predicted donor nitrogen (see
    heme_donor_atom; nothing is added when the ligand has none): "bond" adds a bond constraint to the
    heme FE, "contact" a contact constraint at Boltz's tightest distance. It needs predict_affinity=False
    because Boltz standardizes the SMILES before naming atoms when the affinity head is on.

    protein_msa is a precomputed MSA file (Boltz .csv/.a3m) for the protein chain: the same sequence needs the MSA only once, and
    reusing it removes the dependency on the public MSA server (a DNS outage failed four runs in a row).

    templates are protein structure files (a crystal conformation of the pocket) given to Boltz as soft
    structural templates for the protein chain, steering which conformation the prediction starts from.
    An entry may be a (file, chain) pair to use one chain of a multi-chain file (the cryo-EM trimers).

    predict_affinity=False omits the properties/affinity block entirely.
    Confirmed directly (2026-09-25) that Boltz-2's affinity head runs its
    own diffusion sampling 5x by default (diffusion_samples_affinity=5 vs
    1 for the base structure) -- real, observed cost of 55min-10+ hours
    per compound on this hardware for that step alone. Callers that only
    need heme-coordination geometry (the Fe-ligand-N distance; see
    cypmode/validation/structures.py) don't need the affinity head at all.
    """
    if coordinating_cys < 1 or coordinating_cys > len(protein_sequence):
        raise ValueError(
            f"coordinating_cys={coordinating_cys} is out of range for a "
            f"{len(protein_sequence)}-residue sequence"
        )
    residue = protein_sequence[coordinating_cys - 1]
    if residue != "C":
        raise ValueError(
            f"residue {coordinating_cys} in the given sequence is {residue!r}, not "
            "cysteine ('C') -- wrong position or wrong sequence/numbering"
        )
    if donor_restraint is not None:
        if donor_restraint not in DONOR_RESTRAINTS:
            raise ValueError(f"donor_restraint must be one of {DONOR_RESTRAINTS}, got {donor_restraint!r}")
        if predict_affinity:
            raise ValueError("donor_restraint needs predict_affinity=False (Boltz renames atoms after standardizing the SMILES)")

    yaml_text = (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        f"      id: {protein_chain}\n"
        f"      sequence: {protein_sequence}\n"
        + (f"      msa: {Path(protein_msa).resolve()}\n" if protein_msa else "")
        + "  - ligand:\n"
        f"      id: {ligand_chain}\n"
        f"      smiles: '{ligand_smiles}'\n"
        "  - ligand:\n"
        f"      id: {heme_chain}\n"
        f"      ccd: {HEME_CCD_CODE}\n"
        "constraints:\n"
        "  - bond:\n"
        f"      atom1: [{protein_chain}, {coordinating_cys}, {CYSTEINE_SULFUR_ATOM}]\n"
        f"      atom2: [{heme_chain}, 1, {HEME_IRON_ATOM}]\n"
    )
    donor = heme_donor_atom(ligand_smiles) if donor_restraint else None
    if donor is not None:
        name = donor[1]
        if donor_restraint == "bond":
            yaml_text += f"  - bond:\n      atom1: [{ligand_chain}, 1, {name}]\n      atom2: [{heme_chain}, 1, {HEME_IRON_ATOM}]\n"
        else:
            yaml_text += (
                "  - contact:\n"
                f"      token1: [{ligand_chain}, {name}]\n"
                f"      token2: [{heme_chain}, {HEME_IRON_ATOM}]\n"
                f"      max_distance: {CONTACT_MAX_DISTANCE_A:g}\n"
                "      force: false\n"
            )
    if templates:
        yaml_text += "templates:\n" + "".join(_template_yaml(t, protein_chain) for t in templates)
    if predict_affinity:
        yaml_text += "properties:\n" "    - affinity:\n" f"        binder: {ligand_chain}\n"
    return yaml_text
