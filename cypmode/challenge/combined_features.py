"""Combines the three real, independently-validated feature additions --
KNN read-across (cypmode.challenge.knn_features), PubChem AID 1851
auxiliary activity (cypmode.challenge.pubchem_auxiliary), and BindingDB
auxiliary activity (cypmode.challenge.bindingdb_auxiliary) -- into one
feature set and tests them together.

Each was validated as a SEPARATE addition to plain M3 (all three real
results real and positive: CYP3A4 MCC 0.300->0.308-0.312, CYP2D6 MCC
0.043->0.071-0.089 depending on which one). Whether their gains stack,
partially overlap (all three ultimately draw on similar chemical/activity
signal), or interact was real, unmeasured work until this module -- not
something safe to assume in either direction.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

from cypmode.challenge.bindingdb_auxiliary import BINDINGDB_ISOFORM_UNIPROTS
from cypmode.challenge.bindingdb_auxiliary import exclude_exact_overlap as exclude_bindingdb_overlap
from cypmode.challenge.bindingdb_auxiliary import fetch_bindingdb_isoform
from cypmode.challenge.bindingdb_auxiliary import fit_auxiliary_model as fit_bindingdb_model
from cypmode.challenge.bindingdb_auxiliary import load_bindingdb_isoform
from cypmode.challenge.bindingdb_auxiliary import predict_auxiliary_activity as predict_bindingdb_activity
from cypmode.challenge.knn_features import _morgan_fp, add_knn_label_features_for_fold
from cypmode.challenge.pubchem_auxiliary import exclude_exact_overlap as exclude_aid1851_overlap
from cypmode.challenge.pubchem_auxiliary import fit_auxiliary_model as fit_aid1851_model
from cypmode.challenge.pubchem_auxiliary import load_aid1851
from cypmode.challenge.pubchem_auxiliary import predict_auxiliary_activity as predict_aid1851_activity


def add_external_auxiliary_features(tdi_df: pd.DataFrame, isoform: str) -> pd.DataFrame:
    """Fits both external auxiliary models (AID1851, BindingDB) once on
    their own real external data and adds their predicted-activity
    probabilities as two new columns. Safe to do globally, not
    fold-internally -- same reasoning as each module's own docstring:
    both are fully external datasets, not derived from this project's own
    TDI labels, unlike the KNN feature (which IS built fold-internally in
    cross_validate_combined below, since it reads this project's own
    labels directly).
    """
    aid_df = load_aid1851()
    aid_df = exclude_aid1851_overlap(aid_df, tdi_df["SMILES"])
    aid_model = fit_aid1851_model(aid_df, isoform)
    aid_proba = predict_aid1851_activity(aid_model, tdi_df)

    bdb_path = fetch_bindingdb_isoform(isoform, BINDINGDB_ISOFORM_UNIPROTS[isoform])
    bdb_df = load_bindingdb_isoform(bdb_path)
    bdb_df = exclude_bindingdb_overlap(bdb_df, tdi_df["SMILES"])
    bdb_model = fit_bindingdb_model(bdb_df)
    bdb_proba = predict_bindingdb_activity(bdb_model, tdi_df)

    out = tdi_df.copy()
    out[f"aux_aid1851_{isoform}"] = aid_proba
    out[f"aux_bindingdb_{isoform}"] = bdb_proba
    return out


def cross_validate_combined(
    df: pd.DataFrame,
    label_col: str,
    isoform: str,
    feature_col: str = "m3_features",
    k: int = 10,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fold-internal comparison: plain M3 vs. M3 + KNN (fold-internal,
    leakage-guarded) + AID1851 + BindingDB (both external/global, safe --
    see add_external_auxiliary_features), on the same outer folds so the
    combined effect is directly visible. `df` must already carry
    aux_aid1851_{isoform} and aux_bindingdb_{isoform} columns (see
    add_external_auxiliary_features).
    """
    cyp_df = df.dropna(subset=[label_col]).reset_index(drop=True)
    y = cyp_df[label_col].astype(int).to_numpy()
    m3 = np.stack(cyp_df[feature_col])
    aux = cyp_df[[f"aux_aid1851_{isoform}", f"aux_bindingdb_{isoform}"]].to_numpy()
    fps = [_morgan_fp(s) for s in cyp_df["SMILES"]]

    rows = []
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(m3, y)):
        knn_feats = add_knn_label_features_for_fold(fps, y, train_idx, val_idx, k=k)
        combined = np.hstack([m3, knn_feats, aux])

        m3_model = LGBMClassifier(verbose=-1, random_state=random_state)
        m3_model.fit(m3[train_idx], y[train_idx])
        m3_pred = m3_model.predict(m3[val_idx])

        combined_model = LGBMClassifier(verbose=-1, random_state=random_state)
        combined_model.fit(combined[train_idx], y[train_idx])
        combined_pred = combined_model.predict(combined[val_idx])

        rows.append(
            {
                "fold": fold,
                "n_val": len(val_idx),
                "mcc_m3": matthews_corrcoef(y[val_idx], m3_pred),
                "mcc_combined": matthews_corrcoef(y[val_idx], combined_pred),
            }
        )
    return pd.DataFrame(rows)


def _main() -> None:
    from cypmode.challenge.tdi_ablation import add_ablation_features
    from cypmode.challenge.tdi_baseline import TDI_ISOFORMS, compute_descriptors, load_tdi_training_data

    tdi_df = add_ablation_features(compute_descriptors(load_tdi_training_data()))

    all_results = []
    for cyp in TDI_ISOFORMS:
        tdi_df = add_external_auxiliary_features(tdi_df, cyp)
        results = cross_validate_combined(tdi_df, label_col=f"{cyp}_is_TDI", isoform=cyp)
        results.insert(0, "isoform", cyp)
        all_results.append(results)
        print(f"{cyp}: mean MCC M3={results['mcc_m3'].mean():.4f}, M3+KNN+AID1851+BindingDB={results['mcc_combined'].mean():.4f}")

    combined = pd.concat(all_results, ignore_index=True)
    out_path = Path("data/cyp_challenge/tdi_track/combined_features_ablation.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    _main()
