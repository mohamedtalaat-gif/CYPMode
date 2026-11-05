# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part II: what the data file hides. The rows that have a label and no direct arm (2.4), the block of compounds that were measured on all the enzymes (2.6) and the estimates of a positive rate from a
classifier alone (2.11); the project's own functions do the work."""

import numpy as np
import pandas as pd

from cypmode.challenge.library_blocks import compound_numbers, panel_block_mask
from cypmode.challenge.tdi_targets import ASSAY_FLOOR, label_from_shift


def positives_if_direct_is(t, d_assumed: float) -> int:
    """How many compounds with this TDI-arm potency the written rule would call positive if their missing direct potency were d_assumed."""
    return int(np.sum(label_from_shift(np.asarray(t, dtype=float) - max(d_assumed, ASSAY_FLOOR))))


def block_of(numbering: pd.DataFrame, window: int, min_share: float) -> tuple[pd.DataFrame, np.ndarray]:
    """The share of compounds with a measurement on another enzyme along the numbering, and the mask of the block that the project's finder returns for this window and this minimum share."""
    mask = panel_block_mask(numbering, window, min_share)
    other = numbering.drop(columns="Molecule_Name").notna().any(axis=1).to_numpy().astype(float)
    order = np.argsort(compound_numbers(numbering["Molecule_Name"]), kind="stable")
    share = pd.Series(other[order]).rolling(window, center=True, min_periods=window // 2).mean().to_numpy()
    return pd.DataFrame({"position": np.arange(len(order)), "share": share, "in the block": mask[order]}), mask


def estimate_positive_rate(scores, labels, true_rate: float, n: int, repeats: int, seed: int) -> tuple[np.ndarray, float, float]:
    """pi = (mean score of the new file - m0) / (m1 - m0), with m1 and m0 the mean scores of the known positives and negatives; the new file is drawn from the known scores at the true rate. Returns the estimates, m0 and m1."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels).astype(int)
    m0, m1 = float(scores[labels == 0].mean()), float(scores[labels == 1].mean())
    rng = np.random.default_rng(seed)
    positives, negatives = scores[labels == 1], scores[labels == 0]
    estimates = []
    for _ in range(repeats):
        k = rng.binomial(n, true_rate)
        sample = np.concatenate([rng.choice(positives, k), rng.choice(negatives, n - k)])
        estimates.append((sample.mean() - m0) / (m1 - m0))
    return np.array(estimates), m0, m1
