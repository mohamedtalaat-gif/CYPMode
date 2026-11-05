# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import json

import pandas as pd
import pytest

from cypmode.post_deadline import board


def _board(primary, extra, own_rank, links):
    rows = [{"Rank": i + 1, "Username": f"user{i}", "Model Report Link": link, "Significance (tiers)": f"Tier {1 if i < 2 else 2}", "Open Code": "Yes", **{k: v - 0.01 * i for k, v in primary.items()}, **{f"{k} std": 0.03 for k in primary}} for i, link in enumerate(links)]
    return pd.DataFrame(rows)


def test_the_entry_is_found_by_its_link_and_its_position_on_the_board_is_read(tmp_path):
    links = ["https://a.example/x", "https://b.example/y", "https://github.com/mohamedtalaat-gif/CYPMode/blob/v0.2.0/METHOD_REPORT.md", "https://c.example/z"]
    frame = _board({"LDDT-PLI": 0.5, "BiSyRMSD": 3.0, "LDDT-LP": 0.8}, None, 3, links)
    entry = board.track_entry(frame, "structure")
    assert entry["rank"] == 3 and entry["entries"] == 4 and entry["tier"] == 2 and entry["tiers"] == 2 and entry["first_tier_entries"] == 2
    assert entry["values"]["LDDT-PLI"]["mean"] == pytest.approx(0.48) and entry["values"]["LDDT-PLI"]["std"] == 0.03
    assert entry["first"]["LDDT-PLI"] == 0.5 and entry["last_of_first_tier"]["LDDT-PLI"] == pytest.approx(0.49)


def test_a_board_without_exactly_one_row_of_the_repository_is_refused():
    frame = _board({"LDDT-PLI": 0.5, "BiSyRMSD": 3.0, "LDDT-LP": 0.8}, None, 1, ["https://a.example", "https://b.example"])
    with pytest.raises(SystemExit):
        board.track_entry(frame, "structure")
    twice = _board({"LDDT-PLI": 0.5, "BiSyRMSD": 3.0, "LDDT-LP": 0.8}, None, 1, ["https://github.com/mohamedtalaat-gif/CYPMode/a", "https://github.com/mohamedtalaat-gif/CYPMode/b"])
    with pytest.raises(SystemExit):
        board.track_entry(twice, "structure")


def test_the_two_files_are_copied_recorded_by_checksum_and_the_result_names_no_alias(tmp_path):
    link = "https://github.com/mohamedtalaat-gif/CYPMode/blob/v0.2.0/METHOD_REPORT.md"
    classification = _board({"MA-MCC": 0.4, "MA-Accuracy": 0.8, "MA-Precision": 0.5, "MA-Recall": 0.5, "MA-F1 Score": 0.5}, None, 1, [link, "https://b.example"])
    structure = _board({"LDDT-PLI": 0.5, "BiSyRMSD": 3.0, "LDDT-LP": 0.8}, None, 1, ["https://b.example", link])
    classification.to_csv(tmp_path / "c.csv", index=False)
    structure.to_csv(tmp_path / "s.csv", index=False)
    out = board.build(tmp_path / "c.csv", tmp_path / "s.csv", tmp_path / "kept", tmp_path / "board.json")
    assert out["tracks"]["classification"]["rank"] == 1 and out["tracks"]["structure"]["rank"] == 2
    assert len(out["tracks"]["structure"]["file_sha256"]) == 64 and (tmp_path / "kept/structure_leaderboard.csv").exists()
    assert json.loads((tmp_path / "board.json").read_text()) == out and "user0" not in (tmp_path / "board.json").read_text()
