"""The Chemprop command runs the same two variants as before when no variant is named; the foundation-weight variants are reached only by name (jobs 224 and 225)."""

from cypmode.challenge.nested_runners_chemprop import DEFAULT_VARIANTS, VARIANTS


def test_the_default_variants_are_the_original_two_and_the_foundation_variants_use_a_weight_file():
    assert DEFAULT_VARIANTS == ("chemprop_mt_shift", "chemeleon_mt_shift")
    for name in ("chemeleon_minimol_mt_shift", "chemeleon_surrogate_mt_shift"):
        assert name in VARIANTS and name not in DEFAULT_VARIANTS
        assert VARIANTS[name][0] == "--from-foundation" and VARIANTS[name][1].endswith("__s42_mpnn.pt")


def test_a_staged_run_fits_only_the_cross_validation_tasks_of_the_first_repeats(tmp_path, monkeypatch):
    """max_repeat 0 fits the 5 x (inner + refit) tasks of repeat 0 and nothing else (no other repeat, no deployment fit); the count it returns covers those tasks only."""
    import numpy as np

    from cypmode.challenge import nested_runners_chemprop as runner
    from cypmode.challenge.nested import NestedPlan, PredictionCache

    n = 200
    y = (np.arange(n) % 3 == 0).astype(int)
    plan = NestedPlan(np.arange(n), y, 2, 3, 2, 0, external=np.arange(n, n + 10))
    cache = PredictionCache(tmp_path / "cache", plan)
    fitted = []

    def fake_fit(task, smiles, targets, outside, isoform, extra, epochs, binary, seed=0, accelerator="cpu"):
        fitted.append(task)
        return np.zeros(len(task.evaluate))

    monkeypatch.setattr(runner, "fit_predict", fake_fit)
    targets = np.zeros((n + 10, len(runner.TARGETS)), dtype=np.float32)
    left = runner.run(plan, cache, ["C"] * (n + 10), targets, "CYP3A4", ["chemprop_mt_shift"], 1, "chemprop", max_repeat=0)
    assert left == 0 and {t.repeat for t in fitted} == {0} and len(fitted) == 3 * (2 + 1)
    assert len(cache.missing("chemprop_mt_shift")) > 0  # repeat 1 and the deployment fits are still to do
