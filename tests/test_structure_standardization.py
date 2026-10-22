# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import pytest

pytest.importorskip("chembl_structure_pipeline")

from cypmode.challenge.structure_standardization import standardized_parent_smiles

KETOCONAZOLE = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"


def test_salt_form_matches_its_free_base_after_standardization():
    free_base_parent = standardized_parent_smiles(KETOCONAZOLE)
    salt_form_parent = standardized_parent_smiles(KETOCONAZOLE + ".Cl")
    assert free_base_parent == salt_form_parent


def test_plain_canonical_smiles_would_have_missed_this_match():
    # the real problem this module exists to fix -- confirms the premise,
    # not just the fix
    from rdkit import Chem

    plain_free_base = Chem.MolToSmiles(Chem.MolFromSmiles(KETOCONAZOLE))
    plain_salt = Chem.MolToSmiles(Chem.MolFromSmiles(KETOCONAZOLE + ".Cl"))
    assert plain_free_base != plain_salt


def test_returns_none_for_unparseable_smiles():
    assert standardized_parent_smiles("not-a-smiles") is None
