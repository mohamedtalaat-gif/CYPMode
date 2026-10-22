# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

from cypmode.challenge.chemeleon_stage import decide


def write(tmp_path, line):
    path = tmp_path / "stage1.txt"
    path.write_text("CYP3A4: 5 outer folds; base stack 23 candidates; added candidates cached: ['chemeleon_mt_shift']\n" + line + "\n")
    return path


def test_stage_two_runs_only_for_a_gain_of_at_least_005_in_at_least_four_of_five_folds(tmp_path):
    assert decide(write(tmp_path, "  z0=1.0  stacked base+added vs base: AUC +0.0061 (p=0.140, better 4/5) AP +0.0100 (p=0.2, better 4/5)"))
    assert not decide(write(tmp_path, "  z0=1.0  stacked base+added vs base: AUC +0.0049 (p=0.140, better 5/5) AP +0.0100 (p=0.2, better 4/5)"))
    assert not decide(write(tmp_path, "  z0=1.0  stacked base+added vs base: AUC +0.0090 (p=0.140, better 3/5) AP +0.0100 (p=0.2, better 4/5)"))


def test_a_missing_or_unreadable_stage_one_result_is_a_stop(tmp_path):
    assert not decide(tmp_path / "missing.txt")
    assert not decide(write(tmp_path, "no comparison was printed"))
