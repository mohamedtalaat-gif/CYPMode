"""Unit tests for app.py's pure logic functions, independent of Streamlit's runtime."""

import pandas as pd

from app import load_challenge_compounds, merge_challenge_results, screen_compounds, validate_smiles

KETOCONAZOLE = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"


def test_validate_smiles_splits_valid_invalid():
    valid, invalid = validate_smiles(["CCO", "not a smiles!!", KETOCONAZOLE, ""])
    assert valid == ["CCO", KETOCONAZOLE]
    assert invalid == ["not a smiles!!", ""]


def test_screen_compounds_returns_expected_columns():
    df = screen_compounds([KETOCONAZOLE, "CCO"])
    assert list(df.columns) == ["SMILES", "Motifs matched", "Mechanism call"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df.loc[df["SMILES"] == "CCO", "Motifs matched"].iloc[0] == "none"
    assert df.loc[df["SMILES"] == KETOCONAZOLE, "Mechanism call"].iloc[0].startswith("Type II")


def test_load_challenge_compounds_reads_the_real_blinded_test_set():
    df = load_challenge_compounds()
    assert len(df) == 20
    assert {"Molecule_Name", "SMILES", "Motifs matched", "Mechanism call"} <= set(df.columns)
    assert df["Motifs matched"].ne("none").sum() == 12  # matches the reported 12/20 motif-positive


def test_merge_challenge_results_fills_only_completed_compounds():
    compounds_df = pd.DataFrame({"Molecule_Name": ["a", "b"], "SMILES": ["CCO", "CCN"]})
    results_df = pd.DataFrame(
        [{"compound": "a", "smiles": "CCO", "coordinated": True, "confidence_score": 0.9}]
    )
    merged = merge_challenge_results(compounds_df, results_df)
    assert merged.loc[merged["Molecule_Name"] == "a", "coordinated"].iloc[0] == True  # noqa: E712
    assert pd.isna(merged.loc[merged["Molecule_Name"] == "b", "coordinated"].iloc[0])
