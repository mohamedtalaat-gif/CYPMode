# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pandas as pd

from cypmode.challenge.nested import OUTER_REFIT, NestedPlan, PredictionCache
from cypmode.explorer import data

LOG = """\
noise before
  z0=1.0  stacked base+added vs base: AUC +0.0117 (p=0.105, better 8/15) AP -0.0003 (p=0.945, better 8/15)
  z0=1.96  all-in-one vs final: AUC -0.0021 (p=0.866)
CYP3A4 GATE (corrected p < 0.05 at both z0): NOT MET
"""


def test_paired_rows_read_the_gain_the_p_and_the_folds(tmp_path):
    (tmp_path / "aux_x_CYP3A4.log").write_text(LOG)
    rows = data.paired_rows(tmp_path)
    assert list(rows["enzyme"]) == ["CYP3A4", "CYP3A4"]
    assert list(rows["z0"]) == [1.0, 1.96]
    assert rows["auc_gain"].tolist() == [0.0117, -0.0021]
    assert rows["p"].tolist() == [0.105, 0.866]
    assert rows.loc[0, "better"] == 8 and rows.loc[0, "folds"] == 15
    assert pd.isna(rows.loc[1, "better"])


def test_excerpts_keep_only_the_verdict_and_the_paired_lines(tmp_path):
    (tmp_path / "aux_x_CYP3A4.log").write_text(LOG)
    rows = data.excerpt_rows(tmp_path)
    assert rows["line"].tolist() == [2, 3, 4]
    assert "noise before" not in " ".join(rows["text"])


def test_a_bundle_that_names_a_blinded_compound_or_has_a_blinded_size_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "held_back_patterns", list)
    pd.DataFrame({"a": range(5)}).to_csv(tmp_path / "ok.csv", index=False)
    assert data.check_public(tmp_path) == []
    (tmp_path / "named.csv").write_text("compound\nOCNT-2395436\n")
    pd.DataFrame({"a": range(750)}).to_csv(tmp_path / "size.csv", index=False)
    problems = " ".join(data.check_public(tmp_path))
    assert "named.csv names a blinded compound" in problems
    assert "size.csv has 750 rows" in problems


def test_a_bundle_that_mentions_a_held_back_path_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "held_back_patterns", lambda: ["submissions/tdi_track/*.csv", "data/tdi_track/outputs/final/"])
    (tmp_path / "leak.csv").write_text("path\ndata/tdi_track/outputs/final/x.csv\n")
    assert any("held-back path data/tdi_track/outputs/final" in problem for problem in data.check_public(tmp_path))


def test_the_held_back_block_is_read_from_the_gitignore(tmp_path):
    path = tmp_path / ".gitignore"
    path.write_text("a/\n# >>> HELD BACK UNTIL X\n# comment\nsecret/*.csv\nmore/\n# <<< HELD BACK\nb/\n")
    assert data.held_back_patterns(path) == ["secret/*.csv", "more/"]


def test_the_out_of_fold_predictions_fill_every_compound_once_per_repeat_and_skip_incomplete_candidates(tmp_path):
    pool = np.arange(100, 140)
    plan = NestedPlan(pool, np.tile([0, 1], 20), repeats=2, outer_splits=2, inner_splits=2, seed=0, external=np.arange(500, 510))
    root = tmp_path / "CYP3A4"
    root.mkdir()
    plan.save(root / "plan.npz")
    cache = PredictionCache(root / "cache", plan)
    folds = [plan.task(r, k, OUTER_REFIT) for r, k in plan.outer_folds()]
    for task in folds:
        cache.put("whole", task, task.evaluate / 1000.0)
    for task in folds[:-1]:
        cache.put("partial", task, np.zeros(len(task.evaluate)))
    names, y, scores = data.outer_predictions("CYP3A4", tmp_path)
    assert names == ["whole"]
    assert scores.shape == (1, 2, 40) and y.shape == (40,)
    assert np.allclose(scores[0, 0], pool / 1000.0) and np.allclose(scores[0, 1], pool / 1000.0)


def test_the_outer_fold_of_every_compound_is_recorded_once_per_repeat(tmp_path):
    pool = np.arange(100, 160)
    plan = NestedPlan(pool, np.tile([0, 1], 30), repeats=2, outer_splits=3, inner_splits=2, seed=0, external=np.arange(500, 510))
    root = tmp_path / "CYP3A4"
    root.mkdir()
    plan.save(root / "plan.npz")
    folds = data.outer_fold_assignment("CYP3A4", tmp_path)
    assert folds.shape == (2, 60)
    assert set(np.unique(folds)) == {0, 1, 2}  # every compound has a fold in every repeat, and there is none left over
    assert all(np.bincount(row).min() > 0 for row in folds)
    for repeat, fold in plan.outer_folds():
        task = plan.task(repeat, fold, OUTER_REFIT)
        positions = [int(np.flatnonzero(pool == row)[0]) for row in task.evaluate]
        assert (folds[task.repeat, positions] == fold).all()


def test_every_candidate_of_the_files_has_a_family_and_the_blocks_that_it_uses():
    table = data.candidate_rows(["lgbm_A_deep", "lgbm_A+fps+physB_deep", "knn_tanimoto_k5", "mlp_multitask_label", "pxrblk_A_deep", "pair_desc+phys", "atomjepa_head", "chemprop_mt_shift"]).set_index("candidate")
    assert table.loc["lgbm_A_deep", "blocks"] == "morgan_c2+rdkit_desc"
    assert {"physics_b", "maccs", "erg"} <= set(table.loc["lgbm_A+fps+physB_deep", "blocks"].split("+"))
    assert table.loc["knn_tanimoto_k5", "blocks"] == "" and table.loc["mlp_multitask_label", "blocks"] == ""
    assert table.loc["pxrblk_A_deep", "blocks"] == "morgan_c2+rdkit_desc" and table.loc["pair_desc+phys", "blocks"] == "rdkit_desc+physics"
    assert table.loc["atomjepa_head", "blocks"] == "atomjepa" and table.loc["chemprop_mt_shift", "family"] == "Chemprop, multitask"
    assert table["family"].ne("").all()


def test_every_block_that_a_candidate_uses_has_a_description():
    used = {block for _, blocks in data.candidate_registry().values() for block in blocks}
    assert used and used <= set(data.BLOCK_LABELS)
    rows = data.block_rows(data.candidate_rows(["lgbm_A_deep", "lgbm_A+phys_deep"]))
    assert rows.set_index("block").loc["physics", "candidates"] == 1 and rows.set_index("block").loc["rdkit_desc", "candidates"] == 2


def test_a_table_with_the_fold_columns_is_still_a_public_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "held_back_patterns", list)
    frame = pd.DataFrame({"y": [0, 1, 0, 1, 0], "fold|0": [0, 1, 0, 1, 0], "fold|1": [1, 0, 1, 0, 1], "a|0": [0.1] * 5, "a|1": [0.2] * 5})
    frame.to_csv(tmp_path / "oof_CYP3A4.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    assert data.check_public(tmp_path) == []


def test_the_bundle_of_the_repository_has_the_outer_folds_and_the_two_new_tables():
    from pathlib import Path

    folder = Path("explorer/public")
    for enzyme in data.ENZYMES:
        frame = pd.read_csv(folder / f"oof_{enzyme}.csv.gz", nrows=5)
        assert list(frame.columns[:4]) == ["y", "fold|0", "fold|1", "fold|2"]
    candidates = pd.read_csv(folder / "candidates.csv").fillna("")
    names = {column.rsplit("|", 1)[0] for column in pd.read_csv(folder / "oof_CYP3A4.csv.gz", nrows=1).columns[4:]}
    assert names <= set(candidates["candidate"]) and (folder / "blocks.csv").exists()
    assert not candidates["family"].eq("other").any(), "a candidate that no registry names would be listed without a family: " + ", ".join(candidates.loc[candidates["family"].eq("other"), "candidate"])
