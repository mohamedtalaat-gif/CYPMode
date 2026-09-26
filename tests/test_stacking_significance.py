import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("statsmodels")

from cypmode.challenge.stacking_significance import (
    repeated_cv_stacking_comparison,
    tukey_hsd_summary,
)

KETOCONAZOLE = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"
CLOTRIMAZOLE = "Clc1ccccc1C(c1ccccc1)(c1ccccc1)n1ccnc1"
ETHANOL = "CCO"
BENZENE = "c1ccccc1"


def test_repeated_cv_stacking_comparison_covers_every_repeat_and_fold():
    rng = np.random.default_rng(0)
    n = 120
    real_smiles = [KETOCONAZOLE, CLOTRIMAZOLE, ETHANOL, BENZENE]
    smiles = [real_smiles[i % 4] for i in range(n)]
    m3 = rng.normal(size=(n, 5))
    y = (rng.random(n) > 0.5).astype(int)
    from cypmode.challenge.knn_features import _morgan_fp

    fps = [_morgan_fp(s) for s in smiles]
    aux_features = {"aux1": rng.uniform(size=(n, 1))}

    results = repeated_cv_stacking_comparison(m3, y, fps, aux_features, n_repeats=2, n_splits=3, k=5)
    assert set(results["repeat"]) == {0, 1}
    assert set(results["fold"]) == {0, 1, 2}
    assert set(results["variant"]) == {"M3", "M3+KNN", "M3+aux1", "M3+KNN+aux1"}
    # each (repeat, fold) should contribute one row per variant
    assert len(results) == 2 * 3 * 4


def test_repeated_cv_stacking_comparison_without_aux_features_still_runs():
    rng = np.random.default_rng(0)
    n = 100
    real_smiles = [KETOCONAZOLE, CLOTRIMAZOLE, ETHANOL, BENZENE]
    smiles = [real_smiles[i % 4] for i in range(n)]
    m3 = rng.normal(size=(n, 5))
    y = (rng.random(n) > 0.5).astype(int)
    from cypmode.challenge.knn_features import _morgan_fp

    fps = [_morgan_fp(s) for s in smiles]

    results = repeated_cv_stacking_comparison(m3, y, fps, aux_features={}, n_repeats=1, n_splits=3, k=5)
    assert set(results["variant"]) == {"M3", "M3+KNN"}


def test_tukey_hsd_summary_detects_a_real_difference():
    rng = np.random.default_rng(0)
    # variant A scores consistently higher than variant B, no overlap
    rows = []
    for repeat in range(5):
        for fold in range(5):
            rows.append({"repeat": repeat, "fold": fold, "variant": "A", "mcc": 0.5 + rng.normal(scale=0.01)})
            rows.append({"repeat": repeat, "fold": fold, "variant": "B", "mcc": 0.2 + rng.normal(scale=0.01)})
    results = pd.DataFrame(rows)

    summary = tukey_hsd_summary(results)
    row = summary[(summary["group1"] == "A") & (summary["group2"] == "B")].iloc[0]
    assert row["reject"] == True


def test_tukey_hsd_summary_does_not_falsely_flag_identical_distributions():
    rng = np.random.default_rng(0)
    rows = []
    for repeat in range(5):
        for fold in range(5):
            rows.append({"repeat": repeat, "fold": fold, "variant": "A", "mcc": 0.5 + rng.normal(scale=0.05)})
            rows.append({"repeat": repeat, "fold": fold, "variant": "B", "mcc": 0.5 + rng.normal(scale=0.05)})
    results = pd.DataFrame(rows)

    summary = tukey_hsd_summary(results)
    row = summary[(summary["group1"] == "A") & (summary["group2"] == "B")].iloc[0]
    assert row["reject"] == False
