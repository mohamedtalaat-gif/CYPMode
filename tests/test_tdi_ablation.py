# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from cypmode.challenge.tdi_ablation import (
    MODELS,
    add_ablation_features,
    run_ablation,
    summarize_ablation,
)

KETOCONAZOLE = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"  # motif+, amine-
VERAPAMIL = "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC"  # motif-, amine+
ETHANOL = "CCO"  # motif-, amine-


def test_add_ablation_features_grows_the_feature_vector_each_tier():
    df = pd.DataFrame(
        {
            "SMILES": [KETOCONAZOLE, VERAPAMIL, ETHANOL],
            "descriptors": [np.zeros(5), np.zeros(5), np.zeros(5)],
        }
    )
    out = add_ablation_features(df)

    assert len(out.loc[0, "m0_features"]) == 5
    assert len(out.loc[0, "m1_features"]) == 6  # + motif flag
    assert len(out.loc[0, "m2_features"]) == 7  # + amine flag
    assert len(out.loc[0, "m3_features"]) == 7 + 2048  # + Morgan fingerprint
    assert len(out.loc[0, "m3_no_amine_features"]) == 6 + 2048  # descriptors + motif + Morgan, no amine

    # ketoconazole: motif+, amine-
    assert out.loc[0, "motif_flag"][0] == 1.0
    assert out.loc[0, "amine_flag"][0] == 0.0
    # verapamil: motif-, amine+
    assert out.loc[1, "motif_flag"][0] == 0.0
    assert out.loc[1, "amine_flag"][0] == 1.0


def test_m3_no_amine_features_never_contains_the_amine_flag_value():
    """Real regression: m3_features (used everywhere downstream by default)
    silently kept the amine flag despite the documented, settled decision
    to drop it (CYPMode_Architecture_Specification.md's Decision Log).
    m3_no_amine_features is the corrected production feature set --
    confirm it is descriptors+motif+Morgan, with the amine
    flag's value nowhere in it (not just the same length by coincidence).
    """
    df = pd.DataFrame({"SMILES": [VERAPAMIL], "descriptors": [np.zeros(5)]})  # verapamil: amine flag = 1.0
    out = add_ablation_features(df)

    no_amine = out.loc[0, "m3_no_amine_features"]
    with_amine = out.loc[0, "m3_features"]
    assert len(no_amine) == len(with_amine) - 1
    # m3_features = [descriptors(5), motif(1), amine(1)=1.0, morgan(2048)];
    # m3_no_amine_features must equal that with the amine slot (index 6) removed.
    np.testing.assert_array_equal(no_amine, np.delete(with_amine, 6))


def _synthetic_ablation_df(n_per_class: int = 20, n_features: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    smiles_pool = [KETOCONAZOLE, VERAPAMIL, ETHANOL]
    rows = []
    for cyp in ["CYP3A4", "CYP2D6"]:
        for label in (0, 1):
            for _ in range(n_per_class):
                rows.append(
                    {
                        "SMILES": smiles_pool[len(rows) % len(smiles_pool)],
                        f"{cyp}_is_TDI": bool(label),
                        "descriptors": rng.normal(size=n_features),
                    }
                )
    return pd.DataFrame(rows)


def test_run_ablation_covers_all_four_models_and_both_isoforms():
    df = add_ablation_features(_synthetic_ablation_df())
    results = run_ablation(df, n_splits=3)

    assert set(results["model"]) == set(MODELS)
    assert set(results["isoform"]) == {"CYP3A4", "CYP2D6"}
    # 4 models x 2 isoforms x 3 folds
    assert len(results) == 4 * 2 * 3


def test_run_ablation_uses_paired_folds_across_models():
    """Same random_state -> same row order -> identical fold membership for
    every model, so MCC differences reflect the features, not split luck."""
    df = add_ablation_features(_synthetic_ablation_df())
    results = run_ablation(df, n_splits=3, random_state=7)

    m0 = results[(results["model"] == "M0") & (results["isoform"] == "CYP3A4")]["n_test"].tolist()
    m3 = results[(results["model"] == "M3") & (results["isoform"] == "CYP3A4")]["n_test"].tolist()
    assert m0 == m3


def test_summarize_ablation_groups_by_model_and_isoform():
    df = add_ablation_features(_synthetic_ablation_df())
    results = run_ablation(df, n_splits=3)
    summary = summarize_ablation(results)
    assert set(summary.index.get_level_values("model")) == set(MODELS)
    assert {"accuracy", "f1", "mcc", "roc_auc"} <= set(summary.columns)
