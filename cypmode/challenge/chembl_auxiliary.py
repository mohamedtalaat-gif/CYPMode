# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Auxiliary direct-CYP-activity signal from ChEMBL (ebi.ac.uk/chembl) --
targeted REST pulls for CYP3A4 (target CHEMBL340) and CYP2D6 (target
CHEMBL289) specifically, not a bulk database download (same narrowly-
scoped pattern as cypmode.challenge.pubchem_auxiliary's AID 1851 pull and
cypmode.challenge.bindingdb_auxiliary's targeted BindingDB pull).

Motivation and scope: a third, independent, real external activity
source, distinct from the two already in this project. AID 1851/AID891
are PubChem qHTS screening-library data -- broad, chemically diverse,
and, per pubchem_auxiliary's own module docstring, "largely disjoint"
from this project's own curated, medicinal-chemistry-series-heavy TDI
training pool. ChEMBL is the opposite kind of source: literature-curated
from medicinal chemistry papers, the same general chemical space this
project's own TDI compounds were drawn from (SAR series, not diverse
screening libraries) -- plausibly MORE transferable as an auxiliary
signal for that reason, not just another independent data point. Like
AID 1851 and BindingDB, ChEMBL's activity records are direct
inhibition/binding potency (IC50, Ki, Kd, EC50 -- standard pChEMBL-scale
measurements), not a preincubation-vs-direct TDI comparison, so this has
no TDI labels of its own either -- same "plausible correlate, not a TDI
signal itself" scope as the other two auxiliary modules.

Real data quirk, confirmed directly: ChEMBL aggregates many independent
papers, so the same compound can appear multiple times (different
assays, different activity types) against the same target -- like
BindingDB, these are independent real measurements, combined via median
pChEMBL value per compound rather than deduplicated by dropping rows.

No fold-internal OOF construction needed -- same reasoning as
pubchem_auxiliary/bindingdb_auxiliary's own docstrings: this is a fully
external dataset, safe to fit once and apply identically to every TDI
compound (train and blind test alike), not a leak.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMClassifier
from rdkit import Chem
from rdkit.Chem import AllChem

from cypmode.challenge.structure_standardization import standardized_parent_smiles
from cypmode.challenge.tdi_baseline import compute_descriptors

CHEMBL_ISOFORM_TARGETS = {"CYP3A4": "CHEMBL340", "CYP2D6": "CHEMBL289"}
CHEMBL_RAW_DIR = Path("data/tdi_track/inputs/chembl_cache")
ACTIVE_PCHEMBL_THRESHOLD = 5.0  # pChEMBL 5 = 10 uM -- same cutoff BindingDB's ACTIVE_THRESHOLD_UM uses
API_BASE = "https://www.ebi.ac.uk/chembl/api/data/activity.json"


def fetch_chembl_isoform(
    isoform: str, target_chembl_id: str, out_dir: str | Path = CHEMBL_RAW_DIR, timeout: int = 60, page_limit: int = 1000
) -> Path:
    """Real, targeted, paginated REST pull for one isoform's own real
    activity records with a real pChEMBL value (standard_relation='=',
    a standard unit -- see ChEMBL's own pChEMBL definition) -- cached to
    disk, not re-fetched if already present.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{isoform}_{target_chembl_id}.json"
    if path.exists():
        return path

    records = []
    offset = 0
    while True:
        params = {
            "target_chembl_id": target_chembl_id,
            "pchembl_value__isnull": "false",
            "limit": page_limit,
            "offset": offset,
        }
        resp = requests.get(API_BASE, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        records.extend(payload["activities"])
        total = payload["page_meta"]["total_count"]
        offset += page_limit
        if offset >= total:
            break
        time.sleep(0.2)  # polite pacing on a real public API, not a mandated rate limit

    path.write_text(json.dumps(records))
    return path


def load_chembl_isoform(path: str | Path) -> pd.DataFrame:
    """Parses one isoform's real, cached ChEMBL activity records into one
    row per unique compound, with the median real pChEMBL value across
    every real measurement found for that compound (see module docstring
    for why median-combining, not row-deduplication, is the right call --
    same reasoning as bindingdb_auxiliary's own load function). A record
    whose canonical_smiles does not parse in RDKit is dropped, not passed
    through (same real-world-data discipline as bindingdb_auxiliary's own
    loader).
    """
    records = json.loads(Path(path).read_text())
    rows = []
    for r in records:
        smiles = r.get("canonical_smiles")
        pchembl = r.get("pchembl_value")
        if not smiles or pchembl is None:
            continue
        if Chem.MolFromSmiles(smiles) is None:
            continue
        rows.append({"SMILES": smiles, "pchembl_value": float(pchembl)})
    if not rows:
        return pd.DataFrame(columns=["SMILES", "pchembl_value"])
    df = pd.DataFrame(rows)
    return df.groupby("SMILES", as_index=False)["pchembl_value"].median()


def exclude_exact_overlap(chembl_df: pd.DataFrame, challenge_smiles: pd.Series) -> pd.DataFrame:
    """Same real safety check as pubchem_auxiliary/bindingdb_auxiliary's
    own -- standardized-parent matching, not plain canonical SMILES."""
    challenge_canonical = set(challenge_smiles.map(standardized_parent_smiles).dropna())
    chembl_canonical = chembl_df["SMILES"].map(standardized_parent_smiles)
    return chembl_df[~chembl_canonical.isin(challenge_canonical)].reset_index(drop=True)


def _morgan_fp_bits(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def _featurize(df: pd.DataFrame) -> np.ndarray:
    described = compute_descriptors(df)
    descriptors = np.stack(described["descriptors"])
    morgan = np.stack([_morgan_fp_bits(s) for s in df["SMILES"]])
    return np.hstack([descriptors, morgan])


def fit_auxiliary_model(
    chembl_df: pd.DataFrame, active_threshold: float = ACTIVE_PCHEMBL_THRESHOLD, random_state: int = 42
) -> LGBMClassifier:
    """Fits one LightGBM classifier on ALL of chembl_df's real median
    pChEMBL values, thresholded to active/inactive at pChEMBL 5 (10 uM,
    the same cutoff bindingdb_auxiliary uses -- kept consistent across
    both potency-based auxiliary sources, not independently tuned). No
    fold-internal OOF treatment needed -- see module docstring.
    """
    y = (chembl_df["pchembl_value"] >= active_threshold).astype(int).to_numpy()
    X = _featurize(chembl_df)
    model = LGBMClassifier(verbose=-1, random_state=random_state)
    model.fit(X, y)
    return model


def predict_auxiliary_activity(model: LGBMClassifier, target_df: pd.DataFrame) -> np.ndarray:
    X = _featurize(target_df)
    return model.predict_proba(X)[:, 1]


def _main() -> None:
    from sklearn.metrics import matthews_corrcoef
    from sklearn.model_selection import StratifiedKFold

    from cypmode.challenge.tdi_ablation import add_ablation_features
    from cypmode.challenge.tdi_baseline import TDI_ISOFORMS, load_tdi_training_data

    tdi_df = add_ablation_features(compute_descriptors(load_tdi_training_data()))

    all_results = []
    for cyp in TDI_ISOFORMS:
        target_id = CHEMBL_ISOFORM_TARGETS[cyp]
        raw_path = fetch_chembl_isoform(cyp, target_id)
        chembl_df = load_chembl_isoform(raw_path)
        n_before = len(chembl_df)
        chembl_df = exclude_exact_overlap(chembl_df, tdi_df["SMILES"])
        print(f"{cyp} (ChEMBL {target_id}): {n_before} compounds, {n_before - len(chembl_df)} exact-SMILES overlap excluded")

        aux_model = fit_auxiliary_model(chembl_df)
        aux_proba = predict_auxiliary_activity(aux_model, tdi_df)
        tdi_df[f"chembl_{cyp}_activity_proba"] = aux_proba

        cyp_df = tdi_df.dropna(subset=[f"{cyp}_is_TDI"]).reset_index(drop=True)
        y = cyp_df[f"{cyp}_is_TDI"].astype(int).to_numpy()
        m3_features = np.stack(cyp_df["m3_no_amine_features"])
        m5_features = np.hstack([m3_features, cyp_df[[f"chembl_{cyp}_activity_proba"]].to_numpy()])

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rows = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(m3_features, y)):
            m3_model = LGBMClassifier(verbose=-1, random_state=42)
            m3_model.fit(m3_features[train_idx], y[train_idx])
            m3_pred = m3_model.predict(m3_features[val_idx])

            m5_model = LGBMClassifier(verbose=-1, random_state=42)
            m5_model.fit(m5_features[train_idx], y[train_idx])
            m5_pred = m5_model.predict(m5_features[val_idx])

            rows.append(
                {
                    "isoform": cyp,
                    "fold": fold,
                    "n_val": len(val_idx),
                    "mcc_m3": matthews_corrcoef(y[val_idx], m3_pred),
                    "mcc_m3_plus_aux": matthews_corrcoef(y[val_idx], m5_pred),
                }
            )
        results = pd.DataFrame(rows)
        all_results.append(results)
        print(f"{cyp}: mean MCC M3={results['mcc_m3'].mean():.4f}, M3+aux={results['mcc_m3_plus_aux'].mean():.4f}")

    combined = pd.concat(all_results, ignore_index=True)
    out_path = Path("data/tdi_track/outputs/comparisons/chembl_auxiliary_ablation.csv")
    combined.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    _main()
