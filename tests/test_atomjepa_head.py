"""The attention readout on cached per-atom features trains, separates a planted signal and ignores padding (it needs torch, so it is skipped in the lightweight environment)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cypmode.challenge.nested import Task
from cypmode.challenge.nested_runners_torch import (
    TorchContext,
    fit_predict_atomjepa_head,
)


def planted_context(n=900, seed=0):
    """Molecules of 5 to 14 atoms; a positive molecule has one atom (carbon-like marker in channel 0) whose feature is raised, wherever it sits."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(5, 15, n)
    labels = rng.integers(0, 2, n)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    feats = rng.normal(size=(offsets[-1], 3, 256)).astype(np.float16)
    for i in np.flatnonzero(labels):
        feats[offsets[i] + rng.integers(0, counts[i]), :, 0] += 4.0
    numbers = rng.integers(1, 20, offsets[-1]).astype(np.int16)
    atoms = {"rows": offsets[:-1], "count": counts, "features": feats, "numbers": numbers}
    ctx = TorchContext(log_counts=np.zeros((n, 1), np.float32), descriptors=np.zeros((n, 1), np.float32), targets=np.zeros((n, 8), np.float32), labels=labels, isoform_position=0,
                       outside=np.array([], dtype=np.int64), atoms=atoms)
    return ctx, labels


def test_the_readout_finds_a_planted_atom_wherever_it_sits():
    from sklearn.metrics import roc_auc_score

    ctx, labels = planted_context()
    train, evaluate = np.arange(0, 720), np.arange(720, 900)  # a readout of a few hundred molecules is too small to learn a rare atom: the real tasks have about 3,500
    task = Task(repeat=0, outer=0, inner=0, train=train, evaluate=evaluate)
    scores = fit_predict_atomjepa_head(ctx, task, epochs=20, n_seeds=1)
    assert scores.shape == (180,) and np.all((scores >= 0) & (scores <= 1))
    assert roc_auc_score(labels[evaluate], scores) > 0.85


def test_a_missing_per_atom_cache_is_an_error_not_a_silent_zero():
    ctx, _ = planted_context(20)
    ctx.atoms = None
    with pytest.raises(ValueError, match="per-atom cache"):
        fit_predict_atomjepa_head(ctx, Task(repeat=0, outer=0, inner=0, train=np.arange(10), evaluate=np.arange(10, 20)))
