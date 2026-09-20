import pandas as pd
import pytest

pytest.importorskip("tdc")

from cypmode.data.panel import annotate_mechanism, load_isoform, summarize_panel


def test_load_isoform_shape():
    df = load_isoform("CYP3A4")
    assert {"drug_id", "smiles", "inhibitor", "isoform"} <= set(df.columns)
    assert len(df) > 1000


def test_annotate_mechanism_adds_columns():
    df = load_isoform("CYP3A4").head(20)
    annotated = annotate_mechanism(df)
    assert "motifs" in annotated.columns
    assert "mechanism" in annotated.columns


def test_unknown_isoform_raises():
    with pytest.raises(ValueError):
        load_isoform("CYP9Z9")


def test_summarize_panel_shape():
    df = pd.DataFrame(
        {
            "isoform": ["CYP3A4", "CYP3A4", "CYP3A4"],
            "inhibitor": [1, 1, 0],
            "motifs": [["thiazole"], [], []],
        }
    )
    summary = summarize_panel(df)
    row = summary.iloc[0]
    assert row["n_labeled_inhibitors"] == 2
    assert row["n_with_type_ii_motif"] == 1
    assert row["frac_with_type_ii_motif"] == 0.5
