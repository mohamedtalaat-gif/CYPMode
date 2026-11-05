# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Section 2.3: the label rule. Two potencies, the direct pIC50 d and the TDI-arm pIC50 t, give the label; the literal rule has two branches and the project writes it as one inequality on s = t - max(d, 4).
Both forms are the project's own functions, so the simulator cannot disagree with the notebook."""

from cypmode.challenge.tdi_targets import (
    ASSAY_FLOOR,
    SHIFT_THRESHOLD,
    label_from_arms,
    label_from_shift,
)


def evaluate(d: float, t: float) -> dict:
    """The shift, the branch of the literal rule that applies and the label by the literal rule and by the single threshold."""
    shift = t - max(d, ASSAY_FLOOR)
    literal = bool(label_from_arms(d, t))
    single = bool(label_from_shift(shift) == 1)
    branch = "d above the floor: the shift is t - d" if d > ASSAY_FLOOR else "d below the floor: the shift is t - 4 (the inferred positive)" if d < ASSAY_FLOOR else "d at the floor: both forms agree"
    return {"shift": shift, "threshold": SHIFT_THRESHOLD, "floor": ASSAY_FLOOR, "branch": branch, "literal rule": literal, "single threshold": single, "agree": literal == single}


def boundary(d_values) -> list[float]:
    """The t that sits on the decision boundary for each d: t = max(d, 4) + log10(2)."""
    return [max(d, ASSAY_FLOOR) + SHIFT_THRESHOLD for d in d_values]
