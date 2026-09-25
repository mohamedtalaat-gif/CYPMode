"""Builds Boltz-2 YAML inputs (protein + heme, covalently bonded) for any ligand.

The four data/boltz_test/cyp3a4_*.yaml files were originally hand-written,
one at a time, with the coordinating cysteine's position located once,
inline, and typed into each file. That doesn't scale past four known
compounds and can't be re-verified against a different protein construct.
This module makes both steps real, callable code.
"""

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


def build_boltz_yaml(
    protein_sequence: str,
    ligand_smiles: str,
    coordinating_cys: int,
    protein_chain: str = "A",
    ligand_chain: str = "B",
    heme_chain: str = "C",
    predict_affinity: bool = True,
) -> str:
    """Return Boltz-2 YAML text: protein + ligand + heme, heme covalently bonded
    to coordinating_cys's sulfur, no constraint on the ligand's own position.

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

    yaml_text = (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        f"      id: {protein_chain}\n"
        f"      sequence: {protein_sequence}\n"
        "  - ligand:\n"
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
    if predict_affinity:
        yaml_text += "properties:\n" "    - affinity:\n" f"        binder: {ligand_chain}\n"
    return yaml_text
