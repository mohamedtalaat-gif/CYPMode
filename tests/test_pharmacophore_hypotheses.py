# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score

from cypmode.challenge import pharmacophore_hypotheses as ph


def _planted(seed=0, n=600, width=300):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    x = (rng.random((n, width)) < 0.25).astype(np.uint8)
    x[:, 5] = np.where(y == 1, rng.random(n) < 0.9, rng.random(n) < 0.1)  # a hypothesis of the positives
    x[:, 7] = np.where(y == 1, rng.random(n) < 0.1, rng.random(n) < 0.9)  # an exclusion hypothesis
    return csr_matrix(x), y


def test_the_model_keeps_the_planted_hypotheses_on_the_side_of_their_ratio_and_ranks_them_first():
    x, y = _planted()
    model = ph.fit_model(x, y)
    positive, negative = model.index[: model.n_positive], model.index[model.n_positive :]
    assert positive[0] == 5 and negative[0] == 7
    assert (model.ratio[: model.n_positive] > 0).all() and (model.ratio[model.n_positive :] < 0).all()
    assert len(positive) <= ph.KEEP_PER_SIDE and len(negative) <= ph.KEEP_PER_SIDE


def test_the_columns_are_the_sum_of_the_ratios_the_counts_the_largest_ratio_and_the_share_of_the_strongest_positives():
    x, y = _planted()
    model = ph.fit_model(x, y)
    columns = ph.model_columns(model, x)
    assert columns.shape == (x.shape[0], len(ph.COLUMNS))
    dense = np.asarray(x[:, model.index].todense(), dtype=float)
    assert np.allclose(columns[:, 0], dense @ model.ratio)
    assert np.allclose(columns[:, 1], dense[:, : model.n_positive].sum(axis=1)) and np.allclose(columns[:, 2], dense[:, model.n_positive :].sum(axis=1))
    none = np.flatnonzero(dense.sum(axis=1) == 0)
    if len(none):
        assert (columns[none, 3] == 0).all()
    assert ((0 <= columns[:, 4]) & (columns[:, 4] <= 1)).all()
    assert columns[y == 1, 0].mean() > columns[y == 0, 0].mean()


def test_the_out_of_fold_columns_carry_no_selection_bias_while_the_in_sample_columns_do():
    rng = np.random.default_rng(1)
    n, width = 500, 3000
    x = csr_matrix((rng.random((n, width)) < 0.2).astype(np.uint8))
    y = rng.integers(0, 2, n)  # no signal at all
    rows = np.arange(n)
    oof, _ = ph.hypothesis_columns(x, rows, y, rows[:5], seed=3)
    in_sample = ph.model_columns(ph.fit_model(x, y), x)
    assert roc_auc_score(y, oof[:, 0]) < 0.58
    assert roc_auc_score(y, in_sample[:, 0]) > 0.75


def test_the_evaluation_columns_depend_on_the_training_labels_only_and_are_repeatable():
    x, y = _planted(2)
    train, evaluate = np.arange(0, 450), np.arange(450, 600)
    a_train, a_eval = ph.hypothesis_columns(x, train, y[train], evaluate, seed=11)
    b_train, b_eval = ph.hypothesis_columns(x, train, y[train], evaluate, seed=11)
    assert np.array_equal(a_train, b_train) and np.array_equal(a_eval, b_eval)
    corrupted = y.copy()
    corrupted[evaluate] = 1 - corrupted[evaluate]
    c_train, c_eval = ph.hypothesis_columns(x, train, corrupted[train], evaluate, seed=11)
    assert np.array_equal(a_eval, c_eval) and np.array_equal(a_train, c_train)
    assert roc_auc_score(y[evaluate], a_eval[:, 0]) > 0.8


def test_the_permuted_control_keeps_the_shape_but_carries_no_label_information():
    x, y = _planted(3)
    rows = np.arange(x.shape[0])
    real_train, _ = ph.hypothesis_columns(x, rows[:450], y[:450], rows[450:], seed=5)
    fake_train, fake_eval = ph.hypothesis_columns(x, rows[:450], y[:450], rows[450:], seed=5, permute=True)
    assert fake_train.shape == real_train.shape and fake_eval.shape == (150, len(ph.COLUMNS))
    assert roc_auc_score(y[:450], real_train[:, 0]) > 0.85
    assert roc_auc_score(y[:450], fake_train[:, 0]) < 0.6
    assert roc_auc_score(y[450:], fake_eval[:, 0]) < 0.65


def test_a_model_without_a_supported_hypothesis_gives_zero_columns():
    x = csr_matrix(np.zeros((20, 8), dtype=np.uint8))
    model = ph.fit_model(x, np.array([0, 1] * 10))
    assert len(model.index) == 0 and not ph.model_columns(model, x).any()


def test_the_cache_is_read_back_in_the_order_asked_for_and_an_unknown_compound_is_an_error(tmp_path):
    cache = tmp_path / "hyp.npz"
    np.savez_compressed(cache, smiles=np.array(["CC", "CCO", "CCN"]), vocabulary=np.array([3, 9, 12], dtype=np.uint64), indptr=np.array([0, 2, 2, 3], dtype=np.int64), indices=np.array([0, 2, 1], dtype=np.int32))
    matrix, vocabulary = ph.load(("CCN", "CC"), str(cache))
    assert matrix.shape == (2, 3) and matrix.toarray().tolist() == [[0, 1, 0], [1, 0, 1]] and vocabulary.tolist() == [3, 9, 12]
    with pytest.raises(KeyError):
        ph.load(("CCC",), str(cache))


def test_a_compound_realises_sorted_unique_hypotheses_and_each_can_be_described():
    pytest.importorskip("rdkit")
    pytest.importorskip("pmapper")
    signatures = ph.realised("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    assert signatures.dtype == np.uint64 and len(signatures) > 20 and (np.diff(signatures.astype(float)) > 0).all()
    text = ph.describe("CC(C)Cc1ccc(cc1)C(C)C(=O)O", int(signatures[len(signatures) // 2]))
    assert text and " + " in text
    assert len(ph.realised("not a molecule")) == 0
    assert ph.describe("CC(C)Cc1ccc(cc1)C(C)C(=O)O", 12345) is None


def test_the_hypotheses_of_a_compound_do_not_depend_on_how_its_smiles_is_written():
    pytest.importorskip("rdkit")
    pytest.importorskip("pmapper")
    assert np.array_equal(ph.realised("CC(C)Cc1ccc(cc1)C(C)C(=O)O"), ph.realised("OC(=O)C(C)c1ccc(CC(C)C)cc1"))


def test_the_conformers_behind_the_hypotheses_are_distinct():
    pytest.importorskip("rdkit")
    conformers = ph._conformer_molecules("O=C(C1CCN(CC1)CCc1ccccc1)Nc1ccc(F)cc1")
    positions = [c.GetConformer().GetPositions() for c in conformers]
    assert len(conformers) >= 3 and sum(not np.allclose(positions[0], p, atol=0.05) for p in positions[1:]) >= 2
