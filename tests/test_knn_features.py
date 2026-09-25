import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from cypmode.challenge.knn_features import (
    _morgan_fp,
    add_knn_label_features_for_fold,
    cross_validate_with_knn_features,
    knn_label_features,
)

KETOCONAZOLE = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"
CLOTRIMAZOLE = "Clc1ccccc1C(c1ccccc1)(c1ccccc1)n1ccnc1"
ETHANOL = "CCO"
BENZENE = "c1ccccc1"


def test_knn_label_features_recovers_a_clean_majority_vote():
    query_fp = _morgan_fp(KETOCONAZOLE)
    # 3 near-identical azole-like refs labeled positive, 1 very different ref labeled negative
    reference_fps = [_morgan_fp(KETOCONAZOLE), _morgan_fp(CLOTRIMAZOLE), _morgan_fp(KETOCONAZOLE), _morgan_fp(ETHANOL)]
    reference_labels = np.array([1, 1, 1, 0])
    out = knn_label_features([query_fp], reference_fps, reference_labels, k=4)
    frac_pos, sim_weighted_frac_pos, mean_sim, n_neighbors = out[0]
    assert frac_pos == pytest.approx(0.75)
    assert n_neighbors == 4
    # similarity-weighted vote should push even higher toward 1.0 since the
    # near-identical positives have much higher similarity than the negative
    assert sim_weighted_frac_pos > frac_pos


def test_knn_label_features_handles_empty_reference_pool():
    query_fp = _morgan_fp(KETOCONAZOLE)
    out = knn_label_features([query_fp], [], np.array([]), k=5)
    assert out[0][3] == 0  # n_neighbors_used


def test_add_knn_label_features_for_fold_excludes_self_for_training_rows():
    # a training compound identical to itself must not see itself as a
    # neighbor -- construct a case where, if self-inclusion leaked, the
    # feature would trivially equal the compound's own label
    smiles = [KETOCONAZOLE, KETOCONAZOLE, ETHANOL, ETHANOL, CLOTRIMAZOLE]
    labels = np.array([1, 0, 0, 0, 1])  # two identical-structure compounds with DIFFERENT labels
    fps = [_morgan_fp(s) for s in smiles]

    train_idx = np.array([0, 1, 2, 3])
    val_idx = np.array([4])
    out = add_knn_label_features_for_fold(fps, labels, train_idx, val_idx, k=1)

    # compound 0's single nearest neighbor (excluding itself) among the rest
    # of training is compound 1 (identical structure), whose label is 0 --
    # if self-exclusion were broken, compound 0's own label (1) would leak in
    assert out[0][0] == pytest.approx(0.0)


def test_add_knn_label_features_for_fold_val_rows_use_full_training_pool():
    smiles = [KETOCONAZOLE, KETOCONAZOLE, ETHANOL, BENZENE]
    labels = np.array([1, 1, 0, 0])
    fps = [_morgan_fp(s) for s in smiles]

    train_idx = np.array([0, 1, 2])
    val_idx = np.array([3])
    out = add_knn_label_features_for_fold(fps, labels, train_idx, val_idx, k=3)
    # val row uses the FULL training portion (3 compounds), not leave-one-out
    assert out[3][3] == 3


def _synthetic_df(n: int = 150, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    real_smiles = [KETOCONAZOLE, CLOTRIMAZOLE, ETHANOL, BENZENE]
    rows = []
    for i in range(n):
        rows.append(
            {
                "SMILES": real_smiles[i % 4],
                "descriptors": rng.normal(size=5),
                "label": bool(rng.random() > 0.5),
            }
        )
    return pd.DataFrame(rows)


def test_cross_validate_with_knn_features_covers_every_compound_once():
    df = _synthetic_df()
    results = cross_validate_with_knn_features(df, label_col="label", feature_col="descriptors", n_splits=5)
    assert results["n_val"].sum() == len(df)
    assert len(results) == 5
    assert {"mcc_m3", "mcc_m3_plus_knn"} <= set(results.columns)
