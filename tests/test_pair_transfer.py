import numpy as np
import pytest

pytest.importorskip("lightgbm")

from cypmode.challenge.pair_transfer import fit_predict_pairs, neighbour_table, pair_matrix


def planted(n=240, seed=0):
    """Compounds in 6 close series; the label is decided by one feature, so the SAR is learnable from pair deltas."""
    rng = np.random.default_rng(seed)
    series = rng.integers(0, 6, n)
    x = rng.normal(0, 0.05, (n, 6))
    x[:, 0] = series
    x[:, 1] = rng.normal(0, 1, n)  # the feature that decides the label inside every series
    y = (x[:, 1] > 0).astype(int)
    sim = 1.0 / (1.0 + np.abs(x[:, [0]] - x[:, [0]].T) * 5 + 0.02 * np.abs(x[:, [2]] - x[:, [2]].T))
    return x, y, sim


def test_neighbour_table_excludes_self_and_orders_best_first():
    sim = np.array([[1.0, 0.2, 0.9], [0.2, 1.0, 0.1], [0.9, 0.1, 1.0]])
    top, sims = neighbour_table(sim, np.arange(3), np.arange(3), 2, True)
    assert top[0].tolist() == [2, 1] and sims[0].tolist() == [0.9, 0.2]
    assert all(i not in row for i, row in enumerate(top))


def test_pair_matrix_layout():
    x = np.arange(6, dtype=float).reshape(3, 2)
    y = np.array([1, 0, 1])
    top, sims = np.array([[1, 2], [0, 2]]), np.array([[0.9, 0.5], [0.8, 0.4]])
    pairs = pair_matrix(x[:2], x, y, top, sims)
    assert pairs.shape == (4, 2 + 2 + 3)
    assert pairs[0].tolist() == [0, 1, 0 - 2, 1 - 3, pytest.approx(0.9), 0, 0]  # query 0, neighbour 1 (label 0), rank 0
    assert pairs[3].tolist() == [2, 3, 2 - 4, 3 - 5, pytest.approx(0.4), 1, 1]  # query 1, neighbour 2 (label 1), rank 1


def test_pair_model_learns_series_sar_and_beats_plain_neighbour_voting():
    x, y, sim = planted()
    train, test = np.arange(0, 180), np.arange(180, 240)
    p = fit_predict_pairs(x[train], y[train], x[test], sim[np.ix_(train, train)], sim[np.ix_(test, train)], 5, {"n_estimators": 80, "num_leaves": 7, "min_child_samples": 5, "learning_rate": 0.1}, 1, 0)
    knn_top, knn_sim = neighbour_table(sim, test, train, 5, False)
    knn = y[train][knn_top].mean(axis=1)
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y[test], p) > 0.9 > roc_auc_score(y[test], knn)  # neighbours alone are uninformative here, deltas are not


def test_evaluation_labels_and_rows_never_influence_their_own_prediction():
    x, y, sim = planted()
    train, test = np.arange(0, 180), np.arange(180, 240)
    args = (x[train], y[train], x[test], sim[np.ix_(train, train)], sim[np.ix_(test, train)], 5, {"n_estimators": 30, "num_leaves": 7, "min_child_samples": 5}, 1, 0)
    first = fit_predict_pairs(*args)
    x_shuffled = x.copy()
    x_shuffled[test[0]] = x[test[1]]  # changing another evaluation row's features must not change the others' predictions
    second = fit_predict_pairs(x[train], y[train], x_shuffled[test], sim[np.ix_(train, train)], sim[np.ix_(test, train)], 5, args[6], 1, 0)
    assert np.allclose(first[1:], second[1:])
