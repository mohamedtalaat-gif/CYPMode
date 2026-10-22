# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from cypmode.challenge.hedged_blend import BlendOfStacks, SubsetStack, decide


def _matrix(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    informative = y + rng.normal(0, 0.8, n)
    other = y * 0.5 + rng.normal(0, 1.0, n)
    noise = rng.normal(0, 1, n)
    return y, np.column_stack([informative, other, noise]), ["good", "ok", "noise"]


def test_subset_stack_uses_only_its_columns():
    y, m, names = _matrix()
    sel = SubsetStack(["good"], "x").fit(names, m, y, None, "auc")
    shuffled = m.copy()
    shuffled[:, 1:] = shuffled[::-1, 1:]  # the other columns must not matter
    assert np.allclose(sel.predict(m), sel.predict(shuffled))


def test_blend_is_a_rank_average_between_its_two_parts():
    y, m, names = _matrix()
    blend = BlendOfStacks(["good"], ["ok"]).fit(names, m, y, None, "auc")
    a = SubsetStack(["good"], "a").fit(names, m, y, None, "auc").predict(m)
    b = SubsetStack(["ok"], "b").fit(names, m, y, None, "auc").predict(m)
    s = blend.predict(m)
    from scipy.stats import spearmanr

    assert ((s >= 0) & (s <= 1)).all() and spearmanr(s, a).statistic > 0.4 and spearmanr(s, b).statistic > 0.2
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y, s) >= min(roc_auc_score(y, a), roc_auc_score(y, b)) - 0.02


def test_minimax_rule_adopts_blend_only_when_its_worst_case_is_at_least_as_good():
    def table(blend_by_weight):
        rows = []
        for model, vals in (("hits_only", {0.0: .68, 0.1: .64, 0.25: .6, 0.4: .59, 0.5: .58, 0.68: .57, 1.0: .55}),
                            ("selection_aware", {0.0: .68, 0.1: .58, 0.25: .54, 0.4: .6, 0.5: .62, 0.68: .67, 1.0: .73}),
                            ("blend", blend_by_weight)):
            for z0 in (1.0, 1.96):
                for w, v in vals.items():
                    rows.append({"model": model, "z0": z0, "weight": w, "auc": v, "mcc": 0.0})
        return pd.DataFrame(rows)

    good = {0.0: .68, 0.1: .63, 0.25: .6, 0.4: .62, 0.5: .64, 0.68: .68, 1.0: .70}
    bad = {w: v - 0.05 for w, v in good.items()}
    assert decide(table(good))["adopt_blend"] is True
    assert decide(table(bad))["adopt_blend"] is False
