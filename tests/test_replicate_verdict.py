# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

from cypmode.challenge import replicate_verdict as rv

LINE = "  z0={z} stacked base+added vs base: AUC {d} (p={p}, better 13/15) AP +0.0052 (p=0.569, better 10/15)\n"


def write(path, rows):
    path.write_text("noise\n" + "".join(LINE.format(z=z, d=d, p=p) for z, d, p in rows))


def test_replicated_only_with_significance_in_the_10_fold_design_and_the_same_sign_in_the_5_fold(tmp_path):
    write(tmp_path / "physics_experiment_CYP3A4.log", [("1.0", "+0.0050", "0.078"), ("1.96", "+0.0055", "0.119")])
    write(tmp_path / "physics_experiment_f10_CYP3A4.log", [("1.0", "+0.0052", "0.021"), ("1.96", "+0.0057", "0.034")])
    five, ten = rv.read_stacked(tmp_path / "physics_experiment_CYP3A4.log"), rv.read_stacked(tmp_path / "physics_experiment_f10_CYP3A4.log")
    assert rv.verdict(five, ten) is True
    write(tmp_path / "physics_experiment_f10_CYP3A4.log", [("1.0", "+0.0052", "0.021"), ("1.96", "+0.0057", "0.090")])
    assert rv.verdict(five, rv.read_stacked(tmp_path / "physics_experiment_f10_CYP3A4.log")) is False  # one z0 not significant
    write(tmp_path / "physics_experiment_CYP3A4.log", [("1.0", "-0.0010", "0.9"), ("1.96", "+0.0055", "0.119")])
    write(tmp_path / "physics_experiment_f10_CYP3A4.log", [("1.0", "+0.0052", "0.021"), ("1.96", "+0.0057", "0.034")])
    assert rv.verdict(rv.read_stacked(tmp_path / "physics_experiment_CYP3A4.log"), rv.read_stacked(tmp_path / "physics_experiment_f10_CYP3A4.log")) is False  # sign disagrees


def test_missing_logs_are_not_a_pass(tmp_path):
    assert rv.read_stacked(tmp_path / "none.log") == {} and rv.verdict({}, {}) is False
