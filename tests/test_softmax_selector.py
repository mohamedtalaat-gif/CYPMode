import numpy as np

from cypmode.challenge.softmax_selector import FixedTemperature, SoftmaxBlend, blend, softmax_weights


def test_the_temperature_runs_from_the_best_model_to_the_equal_committee():
    scores = np.array([0.80, 0.78, 0.70])
    assert np.allclose(softmax_weights(scores, 1e-9), [1, 0, 0])
    assert np.allclose(softmax_weights(scores, np.inf), [1 / 3] * 3)
    middle = softmax_weights(scores, 0.02)
    assert middle[0] > middle[1] > middle[2] and abs(middle.sum() - 1) < 1e-12


def planted(n=600, good=1, noise=6, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    columns = [y + rng.normal(0, 1.0, n) for _ in range(good)] + [rng.normal(0, 1.0, n) for _ in range(noise)]
    return y, np.column_stack(columns)


def test_with_one_informative_candidate_and_noise_the_inner_choice_exploits():
    y, oof = planted()
    selection = SoftmaxBlend().fit([f"c{j}" for j in range(oof.shape[1])], oof, y, None, "auc")
    assert selection.temperature <= 0.02
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y, selection.predict(oof)) > roc_auc_score(y, oof[:, 0]) - 0.03


def test_with_equally_informative_candidates_the_inner_choice_explores():
    y, oof = planted(good=7, noise=0, seed=1)
    selection = SoftmaxBlend().fit([f"c{j}" for j in range(oof.shape[1])], oof, y, None, "auc")
    assert selection.temperature >= 0.02 and len(selection.chosen) >= 4


def test_a_fixed_temperature_is_a_selector_of_its_own_and_the_blend_ignores_zero_weights():
    assert FixedTemperature(0.01).name == "softmax_T0.01" and FixedTemperature(np.inf).name == "softmax_Tinf"
    ref = np.arange(10, dtype=float).reshape(10, 1).repeat(2, axis=1)
    assert np.allclose(blend(ref, ref, np.array([1.0, 0.0])), blend(ref[:, :1], ref[:, :1], np.array([1.0])))
