# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Symmetry-corrected ligand-pose RMSD between two structure predictions
of the same compound -- e.g. two independent Boltz-2 runs (different
seeds) for a consistency check, or a prediction against a real deposited
reference structure once one is available locally.

Originally investigated as pyDockRMSD (github.com/neudinger/pyDockRMSD,
wrapping the published DockRMSD algorithm), but that package isn't
currently on PyPI and its own README explicitly warns Apple Silicon needs
a manual conda/source build -- exactly the heavy, fragile-install pattern
this project has repeatedly and correctly declined (openadmet-models,
EnsembleFlex, GNINA, ProLIF). RDKit's own rdMolAlign.GetBestRMS provides
the same real capability DockRMSD's algorithm does -- symmetry-aware RMSD
via atom-order-permutation matching, not a naive fixed-index RMSD -- as
something already a dependency everywhere in this project, confirmed
directly: it detects real conformational differences (a random per-atom
perturbation, ~2.4 A RMSD, close to the injected noise's own magnitude)
while correctly ignoring pure rigid-body translation (0.0 A, since
alignment removes that before scoring), which a naive coordinate-only
RMSD would not do correctly for either case.

Note on current use: this project doesn't yet have real, independent
repeat predictions of the same compound (each real batch compound has
exactly one Boltz-2 run), nor the actual deposited experimental
coordinates for the 4 cryo-EM reference compounds locally (see README.md:
our own structural validation explicitly compares against a *predicted*,
not deposited, reference structure). This module is real, tested
infrastructure for when either becomes available, not applied to a real
comparison yet.
"""

from rdkit.Chem import rdMolAlign

from cypmode.validation.structures import _ligand_mol_3d


def ligand_pose_rmsd(mol_a, mol_b) -> float:
    """Symmetry-corrected RMSD between two RDKit mols of the SAME
    molecular graph (same compound, different 3D coordinates) -- e.g.
    from two different structure predictions. Callers must pass copies if
    the original mols' conformers matter afterward: GetBestRMS's first
    positional argument is the probe molecule, so it aligns mol_a onto
    mol_b as a side effect (confirmed against RDKit's own docstring for
    GetBestRMS(prbMol, refMol, ...) -- prbMol is left in the aligned
    state, not refMol).
    """
    return rdMolAlign.GetBestRMS(mol_a, mol_b)


def ligand_pose_rmsd_between_predictions(structure_a, structure_b, ligand_smiles: str) -> float:
    """Convenience wrapper: reconstructs each structure's own predicted
    ligand pose (see cypmode.validation.structures._ligand_mol_3d) and
    compares them. Both structures must be real predictions for the SAME
    compound -- this does not check that itself; a mismatched pair either
    raises inside _ligand_mol_3d (template/geometry mismatch) or produces
    a meaningless RMSD, not a graceful "different compound" error.
    """
    mol_a, _ = _ligand_mol_3d(structure_a, ligand_smiles)
    mol_b, _ = _ligand_mol_3d(structure_b, ligand_smiles)
    return ligand_pose_rmsd(mol_a, mol_b)
