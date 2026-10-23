# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np

from cypmode.challenge.addition_experiment import cached_candidates
from cypmode.challenge.nested import DEPLOY, NestedPlan, PredictionCache


def _plan_and_cache(tmp_path):
    plan = NestedPlan(np.arange(60), np.tile([0, 1], 30), repeats=1, outer_splits=2, inner_splits=2, seed=0, external=np.arange(100, 110))
    return plan, PredictionCache(tmp_path / "cache", plan)


def _fill(cache, plan, name, only_repeat_zero):
    for task in plan.tasks():
        if task.inner == DEPLOY:
            continue
        if only_repeat_zero and task.repeat < 0:
            continue
        cache.put(name, task, np.zeros(len(task.evaluate)))


def test_a_screen_counts_a_candidate_with_complete_outer_fold_tasks_but_a_full_experiment_does_not(tmp_path):
    plan, cache = _plan_and_cache(tmp_path)
    _fill(cache, plan, "screened", only_repeat_zero=True)
    _fill(cache, plan, "complete", only_repeat_zero=False)
    assert cached_candidates(cache, ["screened", "complete", "absent"], deploy_tasks=False) == ["screened", "complete"]
    assert cached_candidates(cache, ["screened", "complete", "absent"]) == ["complete"]
