# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import json

import pandas as pd

from cypmode.post_deadline import report


def _structure_folder(tmp_path):
    (tmp_path / "structure_gate.json").write_text(
        json.dumps({"board": {"LDDT-PLI": 0.41, "BiSyRMSD": 4.0, "LDDT-LP": 0.8}, "scored": {"LDDT-PLI": 0.4, "BiSyRMSD": 4.1, "LDDT-LP": 0.8}, "difference": {"LDDT-PLI": -0.01, "BiSyRMSD": 0.1, "LDDT-LP": 0.0}, "passed": True, "posebusters_failures": {}})
    )
    (tmp_path / "structure_versions.json").write_text(json.dumps({"openstructure": "2.11.1", "posebusters": "0.6.5", "rdkit": "x"}))
    pd.DataFrame({"comparison": ["submitted minus baseline"], "metric": ["LDDT-PLI"], "mean_difference": [-0.01], "low": [-0.07], "high": [0.05], "verdict": ["no difference shown"], "first_higher": [11], "compounds": [20]}).to_csv(tmp_path / "structure_question5a.csv", index=False)
    pd.DataFrame({"best": [0.6, 0.4], "random": [0.4, 0.3], "submitted": [0.5, 0.3], "gap_submitted": [0.1, 0.1], "gap_random": [0.2, 0.1]}, index=pd.Index(["a", "b"], name="ligand")).to_csv(tmp_path / "structure_question5b.csv")
    pd.DataFrame({"truth_coordinated": [False, True], "False": [8, 0], "True": [8, 4]}).to_csv(tmp_path / "structure_question5c_counts.csv", index=False)
    (tmp_path / "structure_resolution.json").write_text(json.dumps({"spearman_rho": -0.2, "p_value": 0.4, "compounds": 17, "without_resolution": 3}))


def test_a_folder_without_results_says_that_the_questions_have_not_been_run(tmp_path):
    text = "\n".join(report.section(tmp_path))
    assert "## 15." in text and text.count("have not been run") == 2


def test_the_structure_lines_state_the_numbers_of_the_tables_and_the_share_of_the_range_that_the_selection_recovered(tmp_path):
    _structure_folder(tmp_path)
    text = "\n".join(report.structure_lines(tmp_path))
    assert "OpenStructure 2.11.1" in text and "0.40000 against 0.41000" in text and "-0.01000" in text and "0 poses fail PoseBusters" in text
    assert "the selection recovered 33% of the distance" in text  # 1 - 0.1 / 0.15 of the mean gaps
    assert "over 17 compounds (3 files record no resolution)" in text and "| False | 8 | 8 |" in text


def test_the_tdi_lines_name_the_file_that_the_exact_match_identifies_and_read_the_tables(tmp_path):
    (tmp_path / "tdi_gate.json").write_text(
        json.dumps({"board": 0.3723, "macro_mcc": {"A": 0.3723, "B": 0.3725}, "registered": {"tolerance": 0.01, "within_tolerance": ["A", "B"], "passed": False}, "exact": {"tolerance": 1e-06, "match": ["A"]}, "passed": True, "submitted": "A"})
    )
    pd.DataFrame({"Endpoint": ["MA"], "MCC_mean": [0.3723], "MCC_std": [0.0378]}).to_csv(tmp_path / "tdi_board_scores_A.csv", index=False)
    pd.DataFrame({"endpoint": ["macro"], "mean_difference": [-0.0018], "low": [-0.049], "high": [0.043], "verdict": ["no difference shown"]}).to_csv(tmp_path / "tdi_question1_difference.csv", index=False)
    pd.DataFrame({"endpoint": ["CYP3A4_is_TDI"], "scored": [750], "true_positive_rate": [0.31], "called_A": [0.40], "called_B": [0.40]}).to_csv(tmp_path / "tdi_question1_rates.csv", index=False)
    pd.DataFrame({"isoform": ["CYP3A4", "CYP2D6"], "labelled": [750, 748], "agree": [747, 741], "disagree": [0, 0], "not_computable": [3, 7], "computable_unlabelled": [0, 0]}).to_csv(tmp_path / "tdi_question2a_agreement.csv", index=False)
    pd.DataFrame({"isoform": ["CYP3A4"], "z0": [1.0], "both": [652], "only_organisers": [98], "only_project": [0], "neither": [0]}).to_csv(tmp_path / "tdi_question2b_scored_set.csv", index=False)
    pd.DataFrame({"endpoint": ["CYP3A4_is_TDI", "CYP2D6_is_TDI"], "scored": [750, 748], "mcc_submitted": [0.46, 0.27], "call_rate_submitted": [0.4, 0.06], "mcc_ceiling": [0.52, 0.29], "call_rate_at_ceiling": [0.35, 0.07]}).to_csv(tmp_path / "tdi_question3_ceiling.csv", index=False)
    text = "\n".join(report.tdi_lines(tmp_path))
    assert "names A, which is the file that was submitted" in text and "does not separate the two files" in text
    assert "1488 agree and 0 disagree" in text  # 747 + 741 labelled compounds whose two values are present
    assert "at most 0.0400" in text  # (0.52 + 0.29) / 2 - (0.46 + 0.27) / 2


def test_the_readme_block_says_that_the_questions_are_pending_until_the_tables_exist_and_states_the_standing_and_the_findings_after(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    assert report.readme_block(outputs).startswith("(pending")
    (tmp_path / "final_board.json").write_text(
        json.dumps(
            {
                "tracks": {
                    "classification": {"rank": 14, "entries": 62, "tier": 2, "tiers": 5, "values": {"MA-MCC": {"mean": 0.3723, "std": 0.0378}}, "first": {"MA-MCC": 0.498}},
                    "structure": {"rank": 15, "entries": 38, "tier": 2, "tiers": 4, "values": {"LDDT-PLI": {"mean": 0.4133, "std": 0.0301}, "BiSyRMSD": {"mean": 3.97, "std": 0.28}}},
                }
            }
        )
    )
    (outputs / "tdi_gate.json").write_text(json.dumps({"board": 0.3723, "macro_mcc": {"A": 0.3723, "B": 0.3725}, "submitted": "A"}))
    pd.DataFrame({"isoform": ["CYP3A4", "CYP2D6"], "labelled": [750, 748], "agree": [747, 741]}).to_csv(outputs / "tdi_question2a_agreement.csv", index=False)
    pd.DataFrame({"endpoint": ["a", "b"], "mcc_submitted": [0.46, 0.27], "mcc_ceiling": [0.52, 0.29]}).to_csv(outputs / "tdi_question3_ceiling.csv", index=False)
    pd.DataFrame({"endpoint": ["a"]}).to_csv(outputs / "tdi_question1_rates.csv", index=False)
    pd.DataFrame({"comparison": ["submitted minus baseline"], "metric": ["LDDT-PLI"], "verdict": ["no difference shown"]}).to_csv(outputs / "structure_question5a.csv", index=False)
    pd.DataFrame({"best": [0.6], "random": [0.4], "submitted": [0.5]}).to_csv(outputs / "structure_question5b.csv", index=False)
    pd.DataFrame({"truth_coordinated": [False, True], "False": [8, 0], "True": [8, 4]}).to_csv(outputs / "structure_question5c_counts.csv", index=False)
    pd.DataFrame({"arm": ["submitted", "baseline"], "LDDT-PLI": [0.41, 0.42]}).to_csv(outputs / "structure_poses_scored.csv", index=False)
    text = report.readme_block(outputs)
    assert "place the TDI entry 14 of 62 (tier 2 of 5" in text and "the structure entry 15 of 38" in text and "candidate A" in text
    assert "for all 1488 compounds" in text and "at most 0.040 to the macro MCC, against 0.126 between" in text
    assert "all 4 coordinated ligands and called 8 more ligands coordinated although" in text and "no difference shown" in text
