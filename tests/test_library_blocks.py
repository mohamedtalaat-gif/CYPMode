# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pandas as pd

from cypmode.challenge.library_blocks import compound_numbers, panel_block_mask

COLS = ("CYP1A2_pIC50_TDI_condition", "CYP2C9_pIC50_TDI_condition", "CYP2D6_pIC50_TDI_condition")


def _frame(n_low=300, n_panel=800, n_high=300, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.concatenate([np.arange(n_low), 1_000 + np.arange(n_panel), 10_000 + np.arange(n_high)])
    other = np.concatenate([rng.random(n_low) < 0.03, rng.random(n_panel) < 0.6, rng.random(n_high) < 0.03])
    frame = pd.DataFrame({"Molecule_Name": [f"OCNT-{i}" for i in ids]})
    for c in COLS:
        frame[c] = np.where(other, 5.0, np.nan)
    return frame.sample(frac=1.0, random_state=1).reset_index(drop=True)  # row order must not matter


def test_compound_numbers_take_the_trailing_digits():
    assert list(compound_numbers(pd.Series(["OCNT-2312554", "OCNT-0000422"]))) == [2312554.0, 422.0]


def test_panel_block_is_the_id_range_screened_on_the_other_isoforms_whatever_the_row_order():
    frame = _frame()
    mask = panel_block_mask(frame)
    ids = compound_numbers(frame["Molecule_Name"])
    inside = (ids >= 1_000) & (ids < 1_800)
    assert (mask & inside).sum() > 0.95 * inside.sum()
    assert (mask & ~inside).sum() < 0.05 * len(frame)


def test_no_panel_when_nothing_was_measured_outside_cyp3a4():
    frame = _frame()
    for c in COLS:
        frame[c] = np.nan
    assert not panel_block_mask(frame).any()
