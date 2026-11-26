# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The analysis of the ratio of columns to compounds: the spectrum, the width of a candidate, the extra columns, and the rule that judges them."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cypmode.challenge import descriptor_ratio as dr
from cypmode.challenge.nested import NestedPlan, PredictionCache
from cypmode.challenge.nested_runners import Candidate


def test_the_spectrum_of_a_matrix_of_known_rank_finds_that_rank():
    rng = np.random.default_rng(0)
    low = rng.standard_normal((200, 4)) @ rng.standard_normal((4, 30))
    out = dr.spectrum(low + 1e-9 * rng.standard_normal((200, 30)))
    assert out["components_for_95"] <= 4 and out["components_for_90"] <= 4
    assert 1 <= out["participation_ratio"] <= 4.0001
    assert out["columns_used"] == 30


def test_constant_columns_are_left_out_of_the_spectrum():
    x = np.hstack([np.random.default_rng(1).standard_normal((60, 5)), np.ones((60, 3))])
    out = dr.spectrum(x)
    assert out["columns_used"] == 5 and out["rank"] == 5


def test_the_width_of_a_candidate_is_the_sum_of_its_blocks_and_the_fingerprint_for_a_kernel():
    store = SimpleNamespace(blocks={"morgan_bits": np.zeros((3, 2048)), "rdkit_desc": np.zeros((3, 217)), "motif": np.zeros((3, 7))})
    assert dr.candidate_width(Candidate("a", "lgbm_clf", ("rdkit_desc", "motif")), store) == 224
    assert dr.candidate_width(Candidate("b", "lgbm_clf", ("rdkit_desc",), knn_label_features=True), store) == 221
    assert dr.candidate_width(Candidate("c", "svm_tanimoto"), store) == 2048
    table = dr.width_table(store, [Candidate("a", "lgbm_clf", ("rdkit_desc", "motif"))], {"CYP3A4": 112})
    assert table.n_columns.iloc[0] == 224 and table["ratio_CYP3A4"].iloc[0] == pytest.approx(2.0)


def test_shuffled_columns_keep_every_distribution_and_break_every_link_to_the_row():
    rng = np.random.default_rng(2)
    base = np.hstack([rng.integers(0, 2, (300, 6)).astype(float), rng.exponential(size=(300, 4))])
    extra = dr.extra_columns("shuffled", base, 2, 7)
    assert extra.shape == (300, 20)
    for j in range(base.shape[1]):
        assert np.allclose(np.sort(extra[:, j]), np.sort(base[:, j]), atol=1e-6)
    assert not np.allclose(extra[:, : base.shape[1]], base, atol=1e-6)
    assert not np.allclose(extra[:, : base.shape[1]], extra[:, base.shape[1] :], atol=1e-6)
    gaussian = dr.extra_columns("gaussian", base, 3, 7)
    assert gaussian.shape == (300, 30) and abs(float(gaussian.mean())) < 0.1
    with pytest.raises(ValueError):
        dr.extra_columns("uniform", base, 1, 7)


def test_the_non_inferiority_rule_accepts_a_small_loss_and_refuses_a_loss_at_the_margin():
    small = np.array([-0.002, -0.001, -0.003, 0.0, -0.002, -0.001, -0.002, -0.003, -0.001, -0.002])
    mean, p = dr.noninferior(small, 800, 200)
    assert mean == pytest.approx(-0.0017) and p < 0.05
    large = np.array([-0.02, -0.03, -0.015, -0.025, -0.02, -0.03, -0.01, -0.02, -0.025, -0.02])
    mean, p = dr.noninferior(large, 800, 200)
    assert mean < -0.01 and p > 0.5


def _stub(n: int = 200):
    plan = NestedPlan(np.arange(n), np.random.default_rng(0).integers(0, 2, n), 1, 5, 4, 0)
    return SimpleNamespace(plan=plan, iso="CYP3A4")


def test_the_stack_summary_reports_not_harmful_when_the_stack_keeps_its_auc():
    f = _stub()
    rows = []
    for z0 in dr.Z_CUTS:
        for k in range(5):
            base = 0.80 + 0.001 * k
            rows.append({"isoform": "CYP3A4", "variant": "plain", "z0": z0, "repeat": 0, "outer": k, "outer_auc": base, "inner_auc": base})
            rows.append({"isoform": "CYP3A4", "variant": "extra columns", "z0": z0, "repeat": 0, "outer": k, "outer_auc": base - 0.001 * (k % 2), "inner_auc": base})
    lines = dr.summarise_stack(pd.DataFrame(rows), f, 10)
    assert lines[-1].endswith("NOT HARMFUL") and lines[-1].startswith("CHECK CYP3A4 stack")
    rows = [{**r, "outer_auc": r["outer_auc"] - 0.05} if r["variant"] == "extra columns" else r for r in rows]
    assert dr.summarise_stack(pd.DataFrame(rows), f, 10)[-1].endswith("HARMFUL OR NOT RESOLVED")


def test_the_noise_summary_has_a_line_for_every_kind_and_no_gate_line_for_the_ledger():
    f = _stub()
    rows = []
    for name, kinds in dr.KINDS_OF.items():
        for kind in kinds:
            for multiple in dr.NOISE_MULTIPLES:
                for k in range(5):
                    rows.append({"isoform": "CYP3A4", "candidate": name, "kind": kind if multiple else "none", "multiple": multiple, "extra_columns": 10 * multiple, "repeat": 0, "outer": k,
                                 "auc_z1.0": 0.8 - 0.05 * multiple * (kind == "gaussian"), "auc_z1.96": 0.8 - 0.05 * multiple * (kind == "gaussian")})
    lines = dr.summarise_noise(pd.DataFrame(rows).drop_duplicates(["candidate", "kind", "multiple", "outer"]), f)
    checks = [line for line in lines if line.startswith("CHECK")]
    assert len(checks) == sum(len(k) for k in dr.KINDS_OF.values())
    assert any("(gaussian): HARMFUL" in line for line in checks)
    assert all("GATE" not in line for line in lines)


def test_a_routed_cache_borrows_the_predictions_it_does_not_hold(tmp_path):
    plan = NestedPlan(np.arange(40), np.tile([0, 1], 20), 1, 5, 4, 0)
    other = PredictionCache(tmp_path / "other", plan)
    task = plan.task(0, 0, -1)
    other.put("borrowed", task, np.arange(len(task.evaluate), dtype=float))
    routed = dr.RoutedCache(tmp_path / "own", plan, {"borrowed": other})
    routed.put("own", task, np.ones(len(task.evaluate)))
    assert np.array_equal(routed.get("borrowed", task), np.arange(len(task.evaluate)))
    assert np.array_equal(routed.get("own", task), np.ones(len(task.evaluate)))
