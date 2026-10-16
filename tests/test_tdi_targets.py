from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cypmode.challenge.tdi_targets import (
    ASSAY_FLOOR,
    BOTH_ARMS,
    NO_DIRECT,
    NO_LABEL,
    SHIFT_THRESHOLD,
    label_from_arms,
    label_from_shift,
    shift_target,
    training_view,
)

TRAIN_PATH = Path("data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv")


def test_shift_is_measured_against_the_floor_clipped_direct_value():
    assert shift_target(5.0, 5.5) == pytest.approx(0.5)  # d above the floor: plain t - d
    assert shift_target(3.0, 4.5) == pytest.approx(0.5)  # d below the floor: t - 4
    assert np.isnan(shift_target(np.nan, 5.0))
    assert np.isnan(shift_target(5.0, np.nan))


@pytest.mark.parametrize(
    ("direct", "tdi", "expected"),
    [
        (5.0, 5.31, True),  # measured shift just over 2-fold
        (5.0, 5.29, False),  # just under
        (4.5, 4.9, True),
        (3.0, 4.31, True),  # inferred positive: direct below floor, TDI arm clears 4.301
        (3.0, 4.29, False),  # assigned negative
        (2.5, 3.0, False),
        (6.0, 5.0, False),  # TDI arm weaker than direct
    ],
)
def test_the_faq_rule_and_the_single_shift_threshold_agree(direct, tdi, expected):
    assert label_from_arms(direct, tdi) is expected
    assert bool(label_from_shift(shift_target(direct, tdi))) is expected


def test_both_formulations_agree_on_a_dense_grid_away_from_the_floor():
    for direct in np.arange(1.5, 8.0, 0.173):
        if abs(direct - ASSAY_FLOOR) < 1e-9:
            continue
        for tdi in np.arange(1.5, 8.5, 0.157):
            assert label_from_arms(direct, tdi) == bool(label_from_shift(shift_target(direct, tdi))), (direct, tdi)


def test_label_from_shift_keeps_missing_values_missing():
    out = label_from_shift([0.5, 0.1, np.nan])
    assert out[0] == 1.0 and out[1] == 0.0 and np.isnan(out[2])


def _toy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CYP3A4_is_TDI": [True, False, False, np.nan],
            "CYP3A4_pIC50_direct_inhibition": [5.0, 5.0, np.nan, np.nan],
            "CYP3A4_pIC50_TDI_condition": [5.5, 5.1, 5.4, 6.0],
        }
    )


def test_training_view_tags_populations_and_never_mixes_up_imputed_with_measured_shifts():
    view = training_view(_toy(), "CYP3A4")
    assert list(view["population"]) == [BOTH_ARMS, BOTH_ARMS, NO_DIRECT, NO_LABEL]
    assert view["shift"].iloc[0] == pytest.approx(0.5)
    assert np.isnan(view["shift"].iloc[2])  # a missing direct arm is not a measured shift
    assert view["shift_imputed"].iloc[2] == pytest.approx(1.4)  # t - floor, flagged separately
    assert np.isnan(view["shift_imputed"].iloc[3])  # unlabeled rows are never imputed


@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="real training file not present")
@pytest.mark.parametrize("isoform", ["CYP3A4", "CYP2D6"])
def test_the_single_threshold_reproduces_every_given_training_label(isoform):
    df = pd.read_csv(TRAIN_PATH)
    view = training_view(df, isoform)
    both = view[view["population"] == BOTH_ARMS]
    assert len(both) > 1000
    assert (label_from_shift(both["shift"]) == both["label"].astype(float)).all()


@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="real training file not present")
def test_cyp3a4_rows_without_a_direct_measurement_are_all_negative_despite_a_potent_tdi_arm():
    view = training_view(pd.read_csv(TRAIN_PATH), "CYP3A4")
    no_direct = view[view["population"] == NO_DIRECT]
    assert len(no_direct) > 1000
    assert no_direct["label"].astype(float).sum() == 0
    assert no_direct["tdi"].median() > 5.0  # the labels there are by convention, not by measured shift
    assert SHIFT_THRESHOLD == pytest.approx(0.30103, abs=1e-4)
