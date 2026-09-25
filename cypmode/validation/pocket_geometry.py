"""Pocket-geometry features from an already-predicted Boltz-2 structure --
a genuinely different signal from the Fe-N coordination distance
(cypmode.validation.structures): how much of the ligand is buried inside
the protein, and how big the cavity around it actually is. Both operate
on the same static .cif output already sitting under data/boltz_test/out/
-- neither needs a new structure-prediction run.

Two real tools, both confirmed directly on this project's own real
completed structures before being wired in here:

- ligand_burial_sasa: Biopython's own Shrake-Rupley SASA implementation
  (Bio.PDB.SASA), already a dependency (the `validate` extra) -- no new
  install. Confirmed on ketoconazole: bound SASA 43.5 sq-A vs. unbound
  (isolated) SASA 748.1 sq-A, a 94% burial fraction, plausible for a real
  Type II inhibitor known to sit deep in the CYP3A4 active site.

- pocket_volume: pyKVFinder (pip install pyKVFinder), confirmed real and
  working, but needed two real fixes to run on Boltz-2's own output: (1)
  Boltz-2 names the ligand residue "LIG1" in its own CIF file -- 4
  characters, which overflows legacy PDB format's 3-character residue-name
  field and corrupts column alignment downstream (confirmed directly: a
  real ValueError parsing the residue number, caused by exactly this
  overflow) -- renamed to "LIG" before PDB export fixes it; (2) run
  unrestricted, pyKVFinder scans the WHOLE protein for every cavity it can
  find (dozens, most nothing to do with the ligand) -- restricted to the
  region around the ligand via pyKVFinder's own `ligand`/`ligand_cutoff`
  parameters. The largest cavity found within that restricted region is
  taken as "the binding pocket" -- a real, documented simplification, not
  a guarantee that's always the single physically correct cavity.
"""

import copy
import tempfile
from pathlib import Path

from Bio.PDB import Chain, Model, PDBIO, Structure
from Bio.PDB.SASA import ShrakeRupley

from cypmode.validation.structures import _ligand_atoms

LIGAND_CUTOFF_A = 5.0


def ligand_burial_sasa(structure) -> dict:
    """Fraction of the ligand's own surface area buried by the rest of the
    complex: 1 - (SASA computed within the full complex) / (SASA computed
    with the ligand isolated, as if unbound). 0 = fully solvent-exposed,
    close to 1 = fully buried.
    """
    ligand_atoms = _ligand_atoms(structure)
    if not ligand_atoms:
        raise ValueError("no ligand atoms found -- is the inhibitor chain present?")
    ligand_residue = ligand_atoms[0].get_parent()

    sr = ShrakeRupley()
    sr.compute(structure, level="A")
    bound_sasa = sum(a.sasa for a in ligand_atoms)

    # a throwaway, isolated copy -- computing SASA on it must not perturb
    # the real structure object the caller (and other callers) still use
    isolated_residue = copy.deepcopy(ligand_residue)
    sr.compute(isolated_residue, level="A")
    unbound_sasa = sum(a.sasa for a in isolated_residue.get_atoms())

    burial_fraction = 1.0 - (bound_sasa / unbound_sasa) if unbound_sasa > 0 else None
    return {"bound_sasa_A2": bound_sasa, "unbound_sasa_A2": unbound_sasa, "burial_fraction": burial_fraction}


def _write_pdb_with_renamed_ligand(structure, path: Path) -> None:
    """Boltz-2's own ligand residue name ("LIG1", 4 chars) overflows PDB's
    3-char residue-name field -- see module docstring. Renamed on a
    throwaway copy so the caller's own structure object is never mutated.
    """
    structure = copy.deepcopy(structure)
    ligand_atoms = _ligand_atoms(structure)
    ligand_atoms[0].get_parent().resname = "LIG"
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path))


def _write_ligand_only_pdb(structure, path: Path) -> None:
    """A minimal single-residue PDB containing just the ligand, used only
    to tell pyKVFinder which region to restrict cavity detection to (its
    own `ligand` parameter) -- not a structure of its own significance.
    """
    ligand_residue = copy.deepcopy(_ligand_atoms(structure)[0].get_parent())
    ligand_residue.resname = "LIG"
    lig_structure = Structure.Structure("ligand")
    lig_model = Model.Model(0)
    lig_chain = Chain.Chain("A")
    lig_chain.add(ligand_residue)
    lig_model.add(lig_chain)
    lig_structure.add(lig_model)
    io = PDBIO()
    io.set_structure(lig_structure)
    io.save(str(path))


def pocket_volume(structure, ligand_cutoff: float = LIGAND_CUTOFF_A) -> dict:
    """Largest detected cavity within `ligand_cutoff` Angstroms of the
    ligand -- see module docstring for why this is restricted (an
    unrestricted scan finds dozens of cavities across the whole protein,
    almost none relevant to this specific binding site) and why "largest"
    is used as the real, if imperfect, choice of "the" binding pocket.
    """
    import pyKVFinder

    with tempfile.TemporaryDirectory() as tmp_dir:
        complex_path = Path(tmp_dir) / "complex.pdb"
        ligand_path = Path(tmp_dir) / "ligand.pdb"
        _write_pdb_with_renamed_ligand(structure, complex_path)
        _write_ligand_only_pdb(structure, ligand_path)

        results = pyKVFinder.run_workflow(str(complex_path), ligand=str(ligand_path), ligand_cutoff=ligand_cutoff)

    if not results or not results.volume:
        return {"pocket_volume_A3": None, "pocket_area_A2": None, "n_cavities_detected": 0}

    largest_cavity = max(results.volume, key=results.volume.get)
    return {
        "pocket_volume_A3": results.volume[largest_cavity],
        "pocket_area_A2": results.area[largest_cavity],
        "n_cavities_detected": len(results.volume),
    }
