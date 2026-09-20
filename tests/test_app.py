"""Unit tests for app.py's pure logic functions, independent of Streamlit's runtime."""

import pandas as pd

from app import screen_compounds, validate_smiles

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
