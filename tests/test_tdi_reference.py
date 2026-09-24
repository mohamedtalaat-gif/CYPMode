from cypmode.validation.tdi_reference import (
    REFERENCE_SMILES,
    load_tdi_reference,
    with_structural_coordination,
)


def test_load_tdi_reference_has_all_six_named_drugs():
    df = load_tdi_reference()
    assert set(df["drug"]) == set(REFERENCE_SMILES)
    assert len(df) == 6


def test_tdi_positive_uses_the_challenge_threshold():
    df = load_tdi_reference()
    # azamulin and troleandomycin are real, strong, documented MBI inhibitors
    assert df.loc[df["drug"] == "azamulin", "tdi_positive"].iloc[0] == True  # noqa: E712
    assert df.loc[df["drug"] == "troleandomycin", "tdi_positive"].iloc[0] == True  # noqa: E712
    # ketoconazole is documented reversible (non-MBI) Type II inhibition
    assert df.loc[df["drug"] == "ketoconazole", "tdi_positive"].iloc[0] == False  # noqa: E712


def test_motif_screen_misses_the_non_azole_tdi_inhibitors():
    """Real, honest finding: 3 of the 6 real TDI-positive drugs here (via
    tertiary-amine/macrolide mechanisms) carry no Type II motif at all --
    this screen was only ever built to catch azole/pyridine/thiazole heme
    coordination, not every MBI route. See README's Scope and limitations.
    """
    df = load_tdi_reference()
    motif_negative_tdi_positive = df[~df["has_motif"] & df["tdi_positive"]]["drug"].tolist()
    assert set(motif_negative_tdi_positive) == {"diltiazem", "verapamil", "troleandomycin"}


def test_with_structural_coordination_fills_only_the_known_two():
    df = with_structural_coordination(load_tdi_reference())
    assert df.loc[df["drug"] == "ketoconazole", "coordinated"].iloc[0] == True  # noqa: E712
    assert df.loc[df["drug"] == "azamulin", "coordinated"].iloc[0] == False  # noqa: E712
    assert df.loc[df["drug"] == "verapamil", "coordinated"].isna().iloc[0]
