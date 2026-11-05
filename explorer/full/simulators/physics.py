# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Sections 10.4 and 10.5: the physics block, live. The project's own ``compute_molecule`` runs a GFN2-xTB calculation on a conformer ensemble of the molecule and returns the 133 quantities that the models
could use and, for every heavy atom, its charge, its Fukui functions, its valence and its exposed surface."""

import numpy as np

from cypmode.challenge.physics_block import compute_molecule, largest_fragment

HEADLINE = ("phys_homo_ev", "phys_lumo_ev", "phys_gap_ev", "phys_dipole_d")


def run(smiles: str) -> dict | None:
    """The molecule-level quantities and the per-atom table for a SMILES, or None when the SMILES cannot be read."""
    mol = largest_fragment(smiles)
    if mol is None or mol.GetNumHeavyAtoms() > 40:
        return None
    features, atoms = compute_molecule(smiles)
    return {"molecule": mol, "features": {k: float(v) for k, v in features.items() if v is not None and np.isfinite(v)}, "n_quantities": len(features), "atoms": atoms}


def charge_colors(atoms: list[dict]) -> dict:
    """A colour for every heavy atom: blue for a negative partial charge, red for a positive one, stronger with the size of the charge."""
    q = np.array([float(a["q"]) for a in atoms])
    scale = max(float(np.abs(q).max()), 1e-6)
    colors = {}
    for index, value in enumerate(q):
        strength = min(1.0, abs(value) / scale)
        colors[index] = (1.0, 1.0 - 0.7 * strength, 1.0 - 0.7 * strength) if value > 0 else (1.0 - 0.7 * strength, 1.0 - 0.7 * strength, 1.0)
    return colors


def atom_rows(atoms: list[dict]) -> list[dict]:
    return [{"atom": a["atom"], "element": a["symbol"], "role": a["ch_role"] or a["binder_role"] or "", "partial charge": round(float(a["q"]), 3), "Fukui f-": round(float(a["f_minus"]), 3), "Fukui f+": round(float(a["f_plus"]), 3),
             "valence": round(float(a["valence"]), 2), "exposed surface (A^2)": round(float(a["sasa"]), 1)} for a in atoms]


def feature_rows(features: dict) -> list[dict]:
    return [{"quantity": k, "value": round(v, 4)} for k, v in sorted(features.items())]
