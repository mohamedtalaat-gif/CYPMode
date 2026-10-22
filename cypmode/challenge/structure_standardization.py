# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Structure standardization for cross-dataset compound matching, via
ChEMBL's own structure pipeline (chembl_structure_pipeline, pip-installable,
rdkit-only dependency -- the same tool ChEMBL uses internally to curate its
own structures).

Motivation, confirmed directly: plain RDKit canonical SMILES (Chem.MolToSmiles)
treats a salt form as a different compound from its free base -- e.g.
ketoconazole vs. ketoconazole.Cl canonicalize to two different strings, even
though get_parent_mol correctly identifies them as the same real compound.
Any exact-SMILES overlap/leakage check built on plain canonicalization can
silently miss this kind of duplicate. Used by pubchem_auxiliary.py and
bindingdb_auxiliary.py's exclude_exact_overlap, and split_validity.py's
train/test similarity check.
"""

from chembl_structure_pipeline import standardizer
from rdkit import Chem


def standardized_parent_smiles(smiles: str) -> str | None:
    """The canonical SMILES of a compound's standardized parent structure
    (salt-stripped, neutralized) -- None for an unparseable SMILES, not a
    raised exception, since callers use this for bulk dataset matching
    where a handful of malformed rows shouldn't crash the whole pass.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        parent, _ = standardizer.get_parent_mol(mol)
    except Exception:
        return None
    if parent is None:
        return None
    return Chem.MolToSmiles(parent)
