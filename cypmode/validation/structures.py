# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Check the motif screen against Boltz-2 structure predictions.

cypmode/metrics/motifs.py flags a compound as a candidate Type II inhibitor
from 2D substructure alone. This module checks that call against a 3D
prediction: it parses a Boltz-2-predicted complex where the heme cofactor
is explicitly present (as a `HEM` ligand chain) and covalently bonded to the
real axial cysteine, and measures the distance from the heme iron to the
ligand nitrogen(s) that actually match a Type II motif -- not just whichever
ligand nitrogen happens to sit closest to the iron. The input places no
constraint on the inhibitor's position relative to the heme, so those
distances are an unforced structural prediction, not something built into
the input.

Earlier versions of this module used "closest ligand nitrogen, regardless of
which one" as a proxy for "the coordinating nitrogen." That is wrong in
general -- confirmed directly on vardenafil, where the geometrically closest
ligand nitrogen (4.84 A) is not part of any matched Type II motif at all; the
nitrogens the motif screen actually flagged sit at 10.5-11.8 A. The fix
reconstructs the ligand's real bonds and aromaticity from its SMILES
(RDKit's AssignBondOrdersFromTemplate, the same technique
github.com/PatWalters/cyp-heatmap uses) so the same RDKit atom indices used
for SMARTS matching also carry the predicted 3D coordinates -- no separate,
fragile mapping between a 2D match and a 3D atom name.
"""

import tempfile
from pathlib import Path

import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from rdkit import Chem
from rdkit.Chem import AllChem

from cypmode.metrics.motifs import TYPE_II_MOTIFS
from cypmode.validation.constants import COORDINATION_DISTANCE_CUTOFF_A

HEME_RESNAME = "HEM"
HEME_IRON_ATOM = "FE"

__all__ = [
    "COORDINATION_DISTANCE_CUTOFF_A",
    "all_ligand_nitrogen_distances",
    "find_heme_iron",
    "load_affinity",
    "load_confidence",
    "load_structure",
    "motif_nitrogen_distances",
    "summarize_compound",
]

_STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

_COMPILED_MOTIFS = {name: Chem.MolFromSmarts(smarts) for name, smarts in TYPE_II_MOTIFS.items()}


def _add_missing_occupancy(cif_text: str) -> str:
    """OpenFold3's mmCIF output omits the optional `_atom_site.occupancy`
    column (its B-factor column carries per-atom pLDDT instead), which
    Biopython's MMCIFParser requires. Adds the column with a constant 1.0 --
    valid for a predicted model, which has no partial occupancy -- relying on
    OpenFold3 writing one atom per line, so a row's trailing token lines up
    with the new header appended last. Returns the text unchanged if there is
    no `_atom_site` loop to patch.
    """
    lines = cif_text.splitlines()
    header_ends = [i for i, line in enumerate(lines) if line.startswith("_atom_site.")]
    if not header_ends:
        return cif_text
    last_header = header_ends[-1]
    out = lines[: last_header + 1] + ["_atom_site.occupancy"]
    for line in lines[last_header + 1 :]:
        if line.startswith(("ATOM", "HETATM")):
            line = f"{line.rstrip()} 1.0"
        out.append(line)
    return "\n".join(out) + "\n"


def load_structure(cif_path: str | Path):
    parser = MMCIFParser(QUIET=True)
    try:
        return parser.get_structure("complex", str(cif_path))
    except KeyError as exc:
        if exc.args != ("_atom_site.occupancy",):
            raise
    patched = _add_missing_occupancy(Path(cif_path).read_text())
    with tempfile.TemporaryDirectory() as tmp_dir:
        patched_path = Path(tmp_dir) / "patched.cif"
        patched_path.write_text(patched)
        return MMCIFParser(QUIET=True).get_structure("complex", str(patched_path))


def find_heme_iron(structure):
    for atom in structure.get_atoms():
        if atom.get_parent().get_resname() == HEME_RESNAME and atom.get_name() == HEME_IRON_ATOM:
            return atom
    raise ValueError(
        f"no {HEME_IRON_ATOM} atom found in a {HEME_RESNAME} residue -- "
        "was heme included in this prediction's input YAML?"
    )


def _ligand_atoms(structure):
    """Non-protein, non-heme atoms -- the predicted inhibitor, whatever it is."""
    return [
        atom
        for atom in structure.get_atoms()
        if atom.get_parent().get_resname() not in _STANDARD_RESIDUES
        and atom.get_parent().get_resname() != HEME_RESNAME
    ]


_BOND_LENGTH_RANGE_A = (0.8, 2.0)  # generous heavy-atom single/double/triple-bond range


def _mol_from_direct_atom_correspondence(template, ligand_atoms):
    """Fallback for when proximity-bonding + AssignBondOrdersFromTemplate
    fails: builds the molecule from the template's own (known-correct)
    bond graph, with 3D coordinates assigned directly by atom index rather
    than re-derived from interatomic distances at all.

    Confirmed directly (OCNT-2311782, a real compound whose predicted
    conformation makes two non-bonded atoms sit closer together than two
    actually-bonded atoms elsewhere in the same ligand): Boltz-2 does not
    always preserve the input SMILES's atom order in its CIF output --
    spot-checked against itraconazole/ketoconazole, where it does not -- so
    this cannot replace proximity bonding as the default method. But when it
    does happen to hold, it sidesteps proximity bonding's real failure mode
    entirely (a spurious close contact being misread as a bond, which is
    what actually broke OCNT-2311782 -- proximity bonding found 29 bonds
    against the template's real 27, and RDKit's own DetermineConnectivity,
    tested as an alternative, made the identical mistake, since both are
    fundamentally the same kind of distance-based heuristic).

    Returns None (not raised) if the atom order does not correspond
    (different element sequence) or the coordinates do not actually support
    it (a template bond's length would be chemically unreasonable under
    this direct correspondence) -- callers should fall back further, not
    trust an unverified index match.
    """
    if [a.element for a in ligand_atoms] != [a.GetSymbol() for a in template.GetAtoms()]:
        return None

    coords = np.array([a.get_coord() for a in ligand_atoms])
    low, high = _BOND_LENGTH_RANGE_A
    for bond in template.GetBonds():
        length = np.linalg.norm(coords[bond.GetBeginAtomIdx()] - coords[bond.GetEndAtomIdx()])
        if not (low < length < high):
            return None

    mol = Chem.Mol(template)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    return mol


def _ligand_mol_3d(structure, ligand_smiles: str):
    """Rebuild the ligand as an RDKit mol with correct bonds/aromaticity and a
    3D conformer from the predicted coordinates, so SMARTS-matched atom
    indices and 3D positions refer to the same atoms.

    Returns (mol, pdb_atom_names) -- pdb_atom_names[i] is the CIF atom name
    for mol atom i, for reporting.
    """
    ligand_atoms = _ligand_atoms(structure)
    if not ligand_atoms:
        raise ValueError("no ligand atoms found -- is the inhibitor chain present?")

    lines = []
    for i, atom in enumerate(ligand_atoms, start=1):
        x, y, z = atom.get_coord()
        lines.append(
            f"HETATM{i:>5} {atom.get_name():<4} LIG A   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom.element:>2}"
        )
    lines.append("END")
    raw_mol = Chem.MolFromPDBBlock("\n".join(lines), sanitize=False, proximityBonding=True)
    if raw_mol is None:
        raise ValueError("could not parse predicted ligand geometry as a molecule")

    template = Chem.MolFromSmiles(ligand_smiles)
    if template is None:
        raise ValueError(f"could not parse ligand_smiles: {ligand_smiles!r}")

    try:
        mol = AllChem.AssignBondOrdersFromTemplate(template, raw_mol)
    except Exception:
        mol = _mol_from_direct_atom_correspondence(template, ligand_atoms)
        if mol is None:
            raise
    pdb_atom_names = [atom.get_name() for atom in ligand_atoms]
    return mol, pdb_atom_names


def motif_nitrogen_distances(structure, ligand_smiles: str, fe_atom) -> list[dict]:
    """Distance (A) from fe_atom to every ligand nitrogen that matches a Type
    II motif, tagged with which motif matched. Sorted closest first.
    """
    mol, pdb_atom_names = _ligand_mol_3d(structure, ligand_smiles)
    # The distance is read from the coordinates of the file, as in all_ligand_nitrogen_distances: the conformer of
    # `mol` was written through PDB text and can differ from them by a few 1e-4 A, enough to round the same atom
    # to two different values in the two lists.
    atoms_by_name = {atom.get_name(): atom for atom in _ligand_atoms(structure)}

    results = []
    for motif_name, pattern in _COMPILED_MOTIFS.items():
        for match in mol.GetSubstructMatches(pattern):
            for atom_idx in match:
                if mol.GetAtomWithIdx(atom_idx).GetSymbol() != "N":
                    continue
                name = pdb_atom_names[atom_idx]
                results.append({"atom": name, "motif": motif_name, "distance_angstrom": round(float(atoms_by_name[name] - fe_atom), 2)})
    return sorted(results, key=lambda r: r["distance_angstrom"])


def all_ligand_nitrogen_distances(structure, fe_atom) -> list[dict]:
    """Distance (A) from fe_atom to every ligand nitrogen, motif-matched or
    not -- for manual review, so nothing is silently hidden behind the
    motif-based filter above.
    """
    results = []
    for atom in _ligand_atoms(structure):
        if atom.element != "N":
            continue
        results.append({"atom": atom.get_name(), "distance_angstrom": round(float(atom - fe_atom), 2)})
    return sorted(results, key=lambda r: r["distance_angstrom"])


def load_affinity(predictions_dir: str | Path) -> dict:
    files = list(Path(predictions_dir).glob("affinity_*.json"))
    if not files:
        raise FileNotFoundError(f"no affinity_*.json in {predictions_dir}")
    import json

    return json.loads(files[0].read_text())


def load_confidence(predictions_dir: str | Path) -> dict:
    files = list(Path(predictions_dir).glob("confidence_*model_0.json"))
    if not files:
        raise FileNotFoundError(f"no confidence_*model_0.json in {predictions_dir}")
    import json

    return json.loads(files[0].read_text())


def load_trunk_embeddings(predictions_dir: str | Path) -> dict:
    """Loads Boltz-2's own pooled single (s) and pair (z) trunk
    representations from embeddings_*.npz -- only present when the
    `boltz predict` run was made with the real, first-class --write_embeddings
    CLI flag (confirmed directly in third_party/boltz-main's own source:
    data/write/writer.py writes prediction["s"]/["z"] to this file when
    that flag is set; model/models/boltz2.py's predict_step returns them
    unconditionally, so the flag only controls whether they are saved, not
    whether they are computed).
    """
    files = list(Path(predictions_dir).glob("embeddings_*.npz"))
    if not files:
        raise FileNotFoundError(f"no embeddings_*.npz in {predictions_dir} -- was --write_embeddings passed?")
    data = np.load(files[0])
    return {"s": data["s"], "z": data["z"]}


def pooled_trunk_embedding(embeddings: dict) -> np.ndarray:
    """Mean-pools s (per-token) and z (per-token-pair) down to a fixed-size
    vector each, then concatenates -- fixed size regardless of a given
    complex's token count, so compounds of different sizes are directly
    comparable/usable as one ML feature vector. Every axis except the last
    (the model's own feature/channel dimension) is pooled over -- s and z
    have different rank (token vs. token-pair), so this is written
    generically rather than assuming a specific number of axes.
    """
    s_pooled = embeddings["s"].reshape(-1, embeddings["s"].shape[-1]).mean(axis=0)
    z_pooled = embeddings["z"].reshape(-1, embeddings["z"].shape[-1]).mean(axis=0)
    return np.concatenate([s_pooled, z_pooled])


def summarize_compound(
    result_dir: str | Path, ligand_smiles: str, expect_affinity: bool = True, expect_trunk_embedding: bool = False
) -> dict:
    """Summarize one Boltz-2 result directory, e.g. .../boltz_results_cyp3a4_ketoconazole.

    ligand_smiles is required -- motif matching needs the 2D structure, not
    just the predicted geometry, and this function does not assume or look up
    a fixed set of known compounds.

    expect_affinity=False skips reading the affinity JSON entirely (for a
    run made with predict_affinity=False in build_boltz_yaml, which never
    produces one) -- the affinity_* fields are simply absent from the
    result rather than raising FileNotFoundError.
    """
    result_dir = Path(result_dir)
    name = result_dir.name.removeprefix("boltz_results_")
    predictions_dir = result_dir / "predictions" / name
    cif_path = predictions_dir / f"{name}_model_0.cif"

    structure = load_structure(cif_path)
    fe_atom = find_heme_iron(structure)
    motif_distances = motif_nitrogen_distances(structure, ligand_smiles, fe_atom)
    all_distances = all_ligand_nitrogen_distances(structure, fe_atom)

    closest_motif_distance = motif_distances[0]["distance_angstrom"] if motif_distances else None
    confidence = load_confidence(predictions_dir)

    result = {
        "compound": name,
        "motif_nitrogen_distances": motif_distances,
        "all_ligand_nitrogen_distances": all_distances,
        "closest_motif_nitrogen_distance_angstrom": closest_motif_distance,
        "coordinated": closest_motif_distance is not None and closest_motif_distance <= COORDINATION_DISTANCE_CUTOFF_A,
        "confidence_score": confidence["confidence_score"],
    }
    if expect_affinity:
        affinity = load_affinity(predictions_dir)
        result["affinity_pred_value"] = affinity["affinity_pred_value"]
        result["affinity_probability_binary"] = affinity["affinity_probability_binary"]
    if expect_trunk_embedding:
        embeddings = load_trunk_embeddings(predictions_dir)
        # list, not ndarray -- this result dict is JSON-serialized (batch.py's
        # checkpointing), and json.dumps cannot handle a raw numpy array.
        result["pooled_trunk_embedding"] = pooled_trunk_embedding(embeddings).tolist()
    return result
