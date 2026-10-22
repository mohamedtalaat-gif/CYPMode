# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Predicted inhibition of the five major CYP isoforms by models trained ONLY on external data.

The TDC/Veith qHTS panel (~12-13k compounds per isoform, binary inhibitor labels) knows far more
about what is *inactive* than the challenge's training pool does: the pool consists of compounds
promoted to dose-response, so for CYP2D6 only ~9% of its compounds fall below the assay floor,
while the blinded test set is full of them. A model trained on the external panel scores every
challenge compound out-of-sample by construction (no challenge label is involved, and any
external compound that is also a challenge training or test compound is dropped from the external
training set), so the five probabilities are an ordinary label-free feature block -- they cannot
leak, and need no nesting.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

PANEL = ("cyp1a2", "cyp2c9", "cyp2c19", "cyp2d6", "cyp3a4")
TDC_DIR = Path("data/tdi_track/inputs/tdc_cache")
CHALLENGE_FILES = (Path("data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"), Path("data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"))


def connectivity_key(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return Chem.MolToInchiKey(mol)[:14] if mol is not None else None


def _external_training_set(isoform: str, exclude_keys: set[str]) -> tuple[list[str], np.ndarray]:
    table = pd.read_csv(TDC_DIR / f"{isoform}_veith.tab", sep="\t")
    smiles_column = next(c for c in table.columns if c.lower() in ("drug", "smiles"))
    keys = [connectivity_key(s) for s in table[smiles_column]]
    keep = [i for i, k in enumerate(keys) if k is not None and k not in exclude_keys]
    return table[smiles_column].iloc[keep].tolist(), table["Y"].iloc[keep].astype(int).to_numpy()


def external_activity_block(mols) -> np.ndarray:
    """(n_compounds x 5) out-of-sample inhibitor probabilities, columns in PANEL order."""
    from lightgbm import LGBMClassifier

    from cypmode.challenge.feature_store import BLOCKS

    exclude = set()
    for path in CHALLENGE_FILES:
        exclude |= {k for k in (connectivity_key(s) for s in pd.read_csv(path)["SMILES"]) if k}
    target_x = np.hstack([BLOCKS["morgan_c2"](mols), BLOCKS["maccs"](mols)])
    out = np.zeros((len(mols), len(PANEL)), dtype=np.float32)
    for j, isoform in enumerate(PANEL):
        smiles, y = _external_training_set(isoform, exclude)
        train_mols = [Chem.MolFromSmiles(s) for s in smiles]
        train_x = np.hstack([BLOCKS["morgan_c2"](train_mols), BLOCKS["maccs"](train_mols)])
        model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, verbose=-1, n_jobs=4, random_state=0)
        out[:, j] = model.fit(train_x, y).predict_proba(target_x)[:, 1]
    return out
