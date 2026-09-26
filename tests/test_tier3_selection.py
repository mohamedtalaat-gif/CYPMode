import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from cypmode.challenge.tier3_selection import (
    add_motif_selection_flag,
    compute_selection,
    select_fixed_budget_oof,
    uncertainty_selection_mask,
)

KETOCONAZOLE = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"  # motif+
ETHANOL = "CCO"  # motif-


def test_add_motif_selection_flag():
    df = pd.DataFrame({"SMILES": [KETOCONAZOLE, ETHANOL]})
    out = add_motif_selection_flag(df)
    assert out["motif_selected"].tolist() == [True, False]


def _synthetic_df(n: int = 80, n_features: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    smiles_pool = [ETHANOL] * n  # motif-negative for all, isolates the uncertainty path
    descriptors = [rng.normal(size=n_features) for _ in range(n)]
    labels = [int(d.sum() > 0) for d in descriptors]
    return pd.DataFrame({"SMILES": smiles_pool, "descriptors": descriptors, "label": labels})


def test_uncertainty_selection_mask_threshold_is_a_training_fold_property():
    df = _synthetic_df()
    features = np.stack(df["descriptors"])
    labels = df["label"].to_numpy()

    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[:60] = True

    _selected_a, threshold_a = uncertainty_selection_mask(
        features[train_mask], labels[train_mask], features, quantile=0.3
    )
    # Changing only the held-out portion's features must not change the
    # threshold -- it's derived from the training portion alone.
    features_perturbed = features.copy()
    features_perturbed[60:] = features_perturbed[60:] * 100  # wildly different held-out features
    _selected_b, threshold_b = uncertainty_selection_mask(
        features[train_mask], labels[train_mask], features_perturbed, quantile=0.3
    )
    assert threshold_a == threshold_b


def test_compute_selection_combines_motif_and_uncertainty():
    df = _synthetic_df()
    df.loc[0, "SMILES"] = KETOCONAZOLE  # force one real motif-positive row
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[:60] = True

    result = compute_selection(df, train_mask, label_col="label", uncertainty_quantile=0.3)

    assert result.loc[0, "motif_selected"] == True
    assert result.loc[0, "tier3_selected"]  # motif alone guarantees selection
    assert result["tier3_selected"].dtype == bool
    assert (result["tier3_selected"] == (result["motif_selected"] | result["uncertainty_selected"])).all()


def test_compute_selection_validation_labels_never_touch_the_policy():
    """The core leakage guard: flipping the validation portion's labels
    must not change which validation compounds get selected."""
    df = _synthetic_df()
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[:60] = True

    result_a = compute_selection(df, train_mask, label_col="label", uncertainty_quantile=0.4)

    df_flipped = df.copy()
    df_flipped.loc[~train_mask, "label"] = 1 - df_flipped.loc[~train_mask, "label"]
    result_b = compute_selection(df_flipped, train_mask, label_col="label", uncertainty_quantile=0.4)

    assert (result_a["tier3_selected"] == result_b["tier3_selected"]).all()


def test_select_fixed_budget_oof_returns_exactly_the_budget():
    df = _synthetic_df(n=100)
    selected = select_fixed_budget_oof(df, label_col="label", budget=15, n_splits=5)
    assert len(selected) == 15
    assert "oof_uncertainty" in selected.columns
    # every compound's OOF prediction came from a model that didn't train on it
    assert selected["oof_proba"].notna().all()


def test_select_fixed_budget_oof_picks_the_most_uncertain_rows():
    df = _synthetic_df(n=100)
    full_ranked = select_fixed_budget_oof(df, label_col="label", budget=100, n_splits=5)
    top_15 = select_fixed_budget_oof(df, label_col="label", budget=15, n_splits=5, random_state=42)
    # same random_state -> same OOF folds -> the small budget is a strict
    # prefix of the full ranking by uncertainty
    assert top_15["oof_uncertainty"].max() <= full_ranked["oof_uncertainty"].iloc[14] + 1e-9
