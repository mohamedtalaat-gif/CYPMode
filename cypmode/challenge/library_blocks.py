"""The training set is not one library: find the block whose compounds went through the all-isoform panel.

Compound ids fall into blocks that were screened differently. Two small blocks (the lowest ids and the top of the
range) are almost entirely measured on CYP3A4 alone; the large middle block, a diversity-library panel, was run on
all four isoforms, and within it "not measured on CYP2D6" means "screened but not promoted to dose-response". Outside
the panel "not measured" only means "never tested", so those compounds are not non-hits and must not be treated as
such in any selection-aware model. The block is found from the data (a rolling share of compounds measured on any
isoform other than CYP3A4, taken in id order) rather than from hand-typed id limits.
"""

import numpy as np
import pandas as pd

WINDOW = 300
MIN_SHARE = 0.3
OTHER_ISOFORM_COLUMNS = ("CYP1A2_pIC50_TDI_condition", "CYP2C9_pIC50_TDI_condition", "CYP2D6_pIC50_TDI_condition")


def compound_numbers(names: pd.Series) -> np.ndarray:
    return names.str.extract(r"(\d+)$")[0].astype(float).to_numpy()


def panel_block_mask(train: pd.DataFrame, window: int = WINDOW, min_share: float = MIN_SHARE) -> np.ndarray:
    """True for the rows of the largest contiguous (in id order) run where at least ``min_share`` of the compounds
    carry a measurement on an isoform other than CYP3A4.
    """
    ids = compound_numbers(train["Molecule_Name"])
    other = train[list(OTHER_ISOFORM_COLUMNS)].notna().any(axis=1).to_numpy().astype(float)
    order = np.argsort(ids, kind="stable")
    share = pd.Series(other[order]).rolling(window, center=True, min_periods=window // 2).mean().to_numpy()
    keep = (share >= min_share).astype(int)
    edges = np.diff(np.concatenate([[0], keep, [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    if len(starts) == 0:
        return np.zeros(len(train), dtype=bool)
    k = int(np.argmax(ends - starts))
    mask = np.zeros(len(train), dtype=bool)
    mask[order[starts[k] : ends[k]]] = True
    return mask
