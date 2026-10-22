# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import pytest

from cypmode.metrics.amines import has_trialkylamine
from cypmode.validation.tdi_reference import REFERENCE_SMILES


def test_matches_the_real_trialkylamine_tdi_drugs():
    # verapamil, diltiazem, troleandomycin: real TDI-positive, motif-negative
    # drugs whose actual mechanism is trialkylamine N-dealkylation.
    for name in ["verapamil", "diltiazem", "troleandomycin"]:
        assert has_trialkylamine(REFERENCE_SMILES[name]) is True


def test_excludes_ketoconazoles_n_aryl_piperazine():
    # Ketoconazole is real, documented *reversible* (non-MBI) Type II
    # inhibition -- its piperazine nitrogen is N-aryl, not a trialkylamine,
    # and an unrefined "any tertiary amine" SMARTS false-positives on it.
    assert has_trialkylamine(REFERENCE_SMILES["ketoconazole"]) is False


def test_excludes_non_amine_compounds():
    assert has_trialkylamine(REFERENCE_SMILES["azamulin"]) is False
    assert has_trialkylamine(REFERENCE_SMILES["clotrimazole"]) is False


def test_raises_on_unparseable_smiles():
    with pytest.raises(ValueError):
        has_trialkylamine("not a smiles")


def test_raises_on_empty_smiles():
    with pytest.raises(ValueError):
        has_trialkylamine("")


@pytest.mark.parametrize(
    "smiles",
    [
        "CN(C)S(=O)(=O)c1ccccc1",  # sulfonamide nitrogen
        "CC=CN(C)C",  # enamine nitrogen
        "CN(C)C(=N)N",  # guanidine nitrogen
        "CN(C)C(=N)C",  # amidine nitrogen
    ],
)
def test_excludes_delocalised_nitrogens_that_are_not_trialkylamines(smiles):
    assert has_trialkylamine(smiles) is False


def test_still_matches_plain_secondary_and_tertiary_alkylamines():
    assert has_trialkylamine("CCN(CC)CC") is True
    assert has_trialkylamine("CCNCC") is True
    assert has_trialkylamine("CN(C)CCc1ccccc1") is True  # an aromatic ring elsewhere in the molecule does not matter
