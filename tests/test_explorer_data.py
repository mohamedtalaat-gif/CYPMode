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
