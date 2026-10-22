# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pandas as pd
import pytest

from cypmode.challenge.label_confidence import (
    confidence_z,
    confident_weights,
    training_weights,
)
from cypmode.challenge.tdi_targets import SHIFT_THRESHOLD, direct_column, tdi_column

ISO = "CYP3A4"


def _frame(rows):
    t, d, st, sd = zip(*rows)
    return pd.DataFrame(
        {tdi_column(ISO): t, direct_column(ISO): d, tdi_column(ISO) + "_std": st, direct_column(ISO) + "_std": sd}
    )


def test_z_is_the_distance_of_the_shift_from_the_threshold_in_standard_deviations():
    sd_total = np.hypot(0.3, 0.4)
    z = confidence_z(_frame([(6.0, 5.0, 0.3, 0.4)]), ISO)[0]  # shift 1.0
    assert z == pytest.approx((1.0 - SHIFT_THRESHOLD) / sd_total)


def test_below_the_assay_floor_the_direct_value_is_clipped_and_carries_no_noise():
    z = confidence_z(_frame([(4.6, 3.0, 0.2, 0.9)]), ISO)[0]  # d < 4: shift = t - 4, only the TDI arm is uncertain
    assert z == pytest.approx((0.6 - SHIFT_THRESHOLD) / 0.2)


def test_a_label_on_the_threshold_has_zero_confidence_and_missing_arms_are_nan():
    frame = _frame([(5.0 + SHIFT_THRESHOLD, 5.0, 0.3, 0.3), (np.nan, 5.0, 0.3, 0.3)])
    z = confidence_z(frame, ISO)
    assert z[0] == pytest.approx(0.0) and np.isnan(z[1])


def test_confident_weights_zero_out_uncertain_and_unknown_rows():
    z = np.array([0.2, 1.5, 3.0, np.nan])
    wrapped = confident_weights(lambda rows, ref: np.full(len(rows), 2.0), z, 1.0)
    assert list(wrapped(np.arange(4), np.arange(4))) == [0.0, 2.0, 2.0, 0.0]
    assert list(confident_weights(None, z, 1.0)(np.arange(4), np.arange(4))) == [0.0, 1.0, 1.0, 0.0]


def test_training_weights_modes():
    z = np.array([0.2, 1.5, 3.0, np.nan])
    keep, w = training_weights(z, "")
    assert keep.all() and (w == 1).all()
    keep, w = training_weights(z, "drop1.0")
    assert list(keep) == [False, True, True, False] and (w == 1).all()
    keep, w = training_weights(z, "soft")
    assert keep.all() and w[0] == pytest.approx(0.1) and w[1] == pytest.approx(0.75) and w[2] == 1.0
    with pytest.raises(ValueError):
        training_weights(z, "sharpen")
