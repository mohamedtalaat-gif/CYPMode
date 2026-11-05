# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part III: molecules as numbers. The Tanimoto similarity of two Morgan fingerprints (3.1, 3.2) and a molecule as a few physical quantities (3.3)."""

from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

RADIUS, N_BITS = 2, 2048


def fingerprint(mol: Chem.Mol):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)


def tanimoto(mol_a: Chem.Mol, mol_b: Chem.Mol) -> dict:
    """|A and B| / |A or B| of the on-bits of the two fingerprints, with the counts and the shared bits."""
    a, b = fingerprint(mol_a), fingerprint(mol_b)
    on_a, on_b = set(a.GetOnBits()), set(b.GetOnBits())
    shared = sorted(on_a & on_b)
    return {"similarity": len(shared) / len(on_a | on_b) if on_a | on_b else 0.0, "bits of A": len(on_a), "bits of B": len(on_b), "shared": len(shared), "union": len(on_a | on_b),
            "rdkit": DataStructs.TanimotoSimilarity(a, b), "shared bits": shared}


def descriptor_rows(mol: Chem.Mol) -> list[dict]:
    values = {
        "molecular weight": Descriptors.MolWt(mol), "heavy atoms": mol.GetNumHeavyAtoms(), "logP (Crippen)": Crippen.MolLogP(mol), "polar surface area (A^2)": rdMolDescriptors.CalcTPSA(mol),
        "hydrogen-bond donors": Lipinski.NumHDonors(mol), "hydrogen-bond acceptors": Lipinski.NumHAcceptors(mol), "rotatable bonds": Lipinski.NumRotatableBonds(mol),
        "rings": rdMolDescriptors.CalcNumRings(mol), "aromatic rings": rdMolDescriptors.CalcNumAromaticRings(mol), "fraction of sp3 carbons": rdMolDescriptors.CalcFractionCSP3(mol),
    }
    return [{"descriptor": k, "value": round(float(v), 3)} for k, v in values.items()]
