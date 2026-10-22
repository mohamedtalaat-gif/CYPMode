# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import re

from cypmode.validation.constants import (
    COORDINATION_DISTANCE_CUTOFF_A,
    CYP2D6_SEQUENCE,
    CYP3A4_SEQUENCE,
)

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def test_sequences_are_plain_amino_acid_strings_of_the_published_length():
    assert set(CYP3A4_SEQUENCE) <= AMINO_ACIDS and len(CYP3A4_SEQUENCE) == 503  # UniProt P08684
    assert set(CYP2D6_SEQUENCE) <= AMINO_ACIDS and len(CYP2D6_SEQUENCE) == 497  # UniProt P10635


def test_known_residues_sit_where_the_published_numbering_puts_them():
    assert CYP3A4_SEQUENCE[441] == "C"  # Cys442, the axial heme thiolate ligand
    assert re.search(r"F.{2}G.{3}C.G", CYP3A4_SEQUENCE).start() + 7 == 441  # the FxxGxxxCxG signature motif
    assert CYP3A4_SEQUENCE.startswith("MALIPDLAMETWLLL") and CYP2D6_SEQUENCE.startswith("MGLEALVPLAVIVAIFLLLVDLM")


def test_coordination_cutoff_is_a_little_above_the_dative_bond_range():
    assert 2.3 < COORDINATION_DISTANCE_CUTOFF_A <= 3.0
