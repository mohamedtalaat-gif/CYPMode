# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import sys

import pandas as pd

from cypmode.validation import ost_pool_gates as gates
from cypmode.validation import ost_pools


def test_verdicts_are_read_from_the_gate_lines_and_the_last_line_of_a_gate_wins():
    text = "x\nSTRUCTURE GATE mbr_selection (paired, p < 0.1): NOT MET\n  details\nSTRUCTURE GATE restraint_bond: MET\nSTRUCTURE GATE mbr_selection (again): MET\n"
    assert gates.verdicts(text) == {"mbr_selection": True, "restraint_bond": True}
    assert gates.verdicts("STRUCTURE POOL GATES repeated: 3; verdict changes: none") == {}


def test_the_tables_get_openstructures_rmsd_and_the_identity_copy_is_unchanged(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    table = pd.DataFrame({"compound": ["a", "a", "b"], "engine": ["boltz", "openfold3", "boltz"], "sample": [0, 0, 1], "true_rmsd": [1.0, 2.0, 3.0], "template_score": [0.1, 0.2, 0.3]})
    table.to_csv(source / ost_pools.YARDSTICK_TABLES[0], index=False)
    ost = pd.DataFrame({"compound": ["a", "a"], "engine": ["boltz", "openfold3"], "sample": [0, 0], "ost_bisyrmsd": [1.5, 2.5]})
    identity = gates.write_tables(tmp_path / "identity", None, source)
    changed = gates.write_tables(tmp_path / "ost", ost, source)
    assert identity == {} and changed == {ost_pools.YARDSTICK_TABLES[0]: 1}
    assert pd.read_csv(tmp_path / "identity" / ost_pools.YARDSTICK_TABLES[0]).equals(table)
    result = pd.read_csv(tmp_path / "ost" / ost_pools.YARDSTICK_TABLES[0])
    assert result["true_rmsd"].iloc[:2].tolist() == [1.5, 2.5] and result["true_rmsd"].isna().tolist() == [False, False, True]  # a pose OpenStructure did not score has no RMSD
    assert result["template_score"].tolist() == [0.1, 0.2, 0.3] and "ost_bisyrmsd" not in result


def test_the_poses_to_score_are_those_with_an_rmsd_plus_those_of_the_lddt_table(tmp_path):
    pd.DataFrame({"compound": ["a", "b"], "engine": ["boltz", "boltz"], "sample": [0, 1], "true_rmsd": [1.0, float("nan")]}).to_csv(tmp_path / ost_pools.YARDSTICK_TABLES[0], index=False)
    pd.DataFrame({"compound": ["b"], "engine": ["openfold3"], "sample": [2], "lddt_pli": [0.5]}).to_csv(tmp_path / "pose_lddt_to_crystal.csv", index=False)
    assert ost_pools.pose_keys(tmp_path) == [("a", "boltz", 0), ("b", "openfold3", 2)]  # b sample 1 has no RMSD and is not in the LDDT table


def test_capture_returns_what_a_gate_prints_and_restores_the_arguments():
    def gate():
        print("STRUCTURE GATE demo: MET", sys.argv[1:])
        sys.exit(75)

    saved = list(sys.argv)
    text = gates.capture(gate, ["--table", "x.csv"])
    assert "STRUCTURE GATE demo: MET ['--table', 'x.csv']" in text and "exit status 75" in text
    assert sys.argv == saved
