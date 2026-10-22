# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import pytest

from cypmode.challenge import multiplicity as mp


def test_holm_matches_the_textbook_step_down():
    assert mp.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert mp.holm([0.5]) == [0.5]
    assert mp.holm([0.3, 0.3]) == pytest.approx([0.6, 0.6])
    assert max(mp.holm([0.9, 0.8, 0.7])) <= 1.0


def test_the_logs_of_the_added_tests_are_parsed(tmp_path):
    (tmp_path / "aux_screen_CYP3A4.log").write_text(
        "  z0=1.0  stacked base+added vs base: AUC +0.0015 (p=0.535, better 10/15) AP +0.0002 (p=0.976, better 7/15)\n"
        "  z0=1.96 stacked base+added vs base: AUC -0.0018 (p=0.588, better 10/15) AP +0.0023 (p=0.628, better 12/15)\n"
        "CYP3A4 GATE (stacked, corrected p < 0.05 at both z0): NOT MET\n")
    (tmp_path / "nonneg_stack_CYP2D6.log").write_text("  z0=1.0  outer AUC: logistic blend 0.7312, non-negative blend 0.7353 (+0.0041, p=0.688, better 8/15) | MCC\n"
                                                      "  z0=1.96 outer AUC: logistic blend 0.7536, non-negative blend 0.7577 (+0.0041, p=0.698, better 7/15) | MCC\n")
    table = mp.collect(tmp_path)
    assert len(table) == 2
    screen = table[table.log == "aux_screen_CYP3A4.log"].iloc[0]
    assert screen.p_both == pytest.approx(0.588) and not screen.gain_positive_at_both and screen.isoform == "CYP3A4"
    assert table.holm_adjusted.between(0, 1).all() and (table.holm_adjusted >= table.p_both - 1e-12).all()
