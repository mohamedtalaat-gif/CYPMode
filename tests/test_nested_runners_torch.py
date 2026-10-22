# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Leakage guards for the torch-side nested runners. Torch-marked: run in its own pytest
invocation, never alongside LightGBM tests (see the `torch` marker in pyproject.toml)."""

import numpy as np
import pytest

pytest.importorskip("torch")

pytestmark = pytest.mark.torch

from cypmode.challenge.nested import NestedPlan
from cypmode.challenge.nested_runners_torch import (
    TorchContext,
    fit_predict_multitask,
    training_rows,
)


def _setup(n_pool=80, n_outside=30, n_external=15, seed=0):
    rng = np.random.default_rng(seed)
    n = n_pool + n_outside + n_external
    y = (rng.random(n_pool) < 0.4).astype(int)
    plan = NestedPlan(np.arange(n_pool), y, 1, 4, 3, seed, external=np.arange(n_pool + n_outside, n))
    labels = np.full(n, -1)
    labels[:n_pool] = y
    targets = rng.normal(size=(n, 8)).astype(np.float32)
    targets[n_pool + n_outside :] = np.nan  # external rows carry no targets
    ctx = TorchContext(
        log_counts=rng.random((n, 20)).astype(np.float32),
        descriptors=rng.normal(size=(n, 6)).astype(np.float32),
        targets=targets,
        labels=labels,
        isoform_position=3,
        outside=np.arange(n_pool, n_pool + n_outside),
    )
    return ctx, plan


def test_a_task_trains_only_on_its_own_rows_plus_compounds_outside_the_pool():
    ctx, plan = _setup()
    for task in plan.inner_tasks(0, 0)[:2]:
        rows = set(training_rows(ctx, task).tolist())
        outer_val = set(plan.task(0, 0, -1).evaluate.tolist())
        assert not rows & set(task.evaluate.tolist())
        assert not rows & outer_val  # an inner task never sees the outer-validation rows
        assert not rows & set(plan.external.tolist())
        assert set(task.train.tolist()) <= rows and set(ctx.outside.tolist()) <= rows


@pytest.mark.parametrize("output", ["label", "shift", None])
def test_multitask_predictions_do_not_depend_on_evaluation_or_outer_validation_rows(output):
    ctx, plan = _setup()
    task = plan.inner_tasks(0, 1)[0]
    outer_val = plan.task(0, 1, -1).evaluate
    before = fit_predict_multitask(ctx, task, output, epochs=2, n_seeds=1)
    corrupted = TorchContext(ctx.log_counts, ctx.descriptors, ctx.targets.copy(), ctx.labels.copy(), ctx.isoform_position, ctx.outside)
    hidden = np.concatenate([task.evaluate, outer_val])
    corrupted.targets[hidden] = 50.0
    corrupted.labels[hidden] = 1 - corrupted.labels[hidden]
    after = fit_predict_multitask(corrupted, task, output, epochs=2, n_seeds=1)
    if output is None:
        assert set(before) == {"label", "shift"}
        before, after = np.concatenate(list(before.values())), np.concatenate(list(after.values()))
    else:
        assert before.shape == (len(task.evaluate),)
    assert np.allclose(before, after, atol=1e-5)
