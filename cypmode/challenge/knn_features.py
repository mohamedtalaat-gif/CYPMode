# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""KNN-label (read-across) features: for each compound, the known TDI
status of its K nearest chemical neighbors (by Tanimoto similarity on
ECFP4), aggregated into a small feature set. Captures local
structure-activity signal a global tree/linear model doesn't automatically
exploit, on top of M3's own features.

Leakage-critical, in two ways: (1) a compound's own label must never
contribute to its own feature (trivial to get wrong -- it would just let
the feature recover the label); (2) less obviously, the neighbor
reference pool for ANY compound must be restricted to the current CV
fold's training portion, not the full dataset. Precomputing this feature
once globally, before splitting into folds, would let a training
compound's feature encode fragments of the validation fold's own labels
(via shared neighbors), inflating validation MCC without the model ever
having "seen" that label directly. See add_knn_label_features_for_fold and
cross_validate_with_knn_features, which build it fold-internally instead.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

KNN_FEATURE_NAMES = ["knn_frac_positive", "knn_sim_weighted_frac_positive", "knn_mean_topk_similarity", "knn_n_neighbors"]


def _morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def knn_label_features(query_fps: list, reference_fps: list, reference_labels: np.ndarray, k: int = 10) -> np.ndarray:
    """Per query fingerprint: (fraction of its k nearest reference
    neighbors that are positive, the same fraction but Tanimoto-similarity
    weighted, the mean similarity to those k neighbors, and how many
    neighbors were actually available). reference_fps/reference_labels
    must already exclude the query itself if it's part of the reference
    pool -- this function has no notion of compound identity, so it
    can't enforce that; see add_knn_label_features_for_fold.
    """
    reference_labels = np.asarray(reference_labels, dtype=float)
    n_ref = len(reference_fps)
    out = np.zeros((len(query_fps), 4))
    if n_ref == 0:
        return out
    k_eff = min(k, n_ref)
    for i, qfp in enumerate(query_fps):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(qfp, reference_fps))
        top_idx = np.argsort(-sims)[:k_eff]
        top_sims = sims[top_idx]
        top_labels = reference_labels[top_idx]
        frac_pos = top_labels.mean()
        weight_sum = top_sims.sum()
        sim_weighted_frac_pos = float((top_sims * top_labels).sum() / weight_sum) if weight_sum > 0 else frac_pos
        out[i] = [frac_pos, sim_weighted_frac_pos, top_sims.mean(), k_eff]
    return out


def add_knn_label_features_for_fold(
    fps: list, labels: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, k: int = 10
) -> np.ndarray:
    """Fold-internal KNN feature construction -- see module docstring.
    Returns a (len(fps), 4) array aligned to the full fps/labels arrays:
    train_idx rows use "rest of the training portion" as reference (self
    excluded, leave-one-out), val_idx rows use the "full training portion"
    as reference (no exclusion needed -- validation compounds are never
    in the reference pool at all).
    """
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    train_fps = [fps[i] for i in train_idx]
    train_labels = np.asarray(labels)[train_idx]

    out = np.zeros((len(fps), 4))
    for pos, idx in enumerate(train_idx):
        rest_fps = train_fps[:pos] + train_fps[pos + 1 :]
        rest_labels = np.concatenate([train_labels[:pos], train_labels[pos + 1 :]])
        out[idx] = knn_label_features([fps[idx]], rest_fps, rest_labels, k=k)[0]

    val_fps = [fps[i] for i in val_idx]
    out[val_idx] = knn_label_features(val_fps, train_fps, train_labels, k=k)
    return out


def cross_validate_with_knn_features(
    df: pd.DataFrame, label_col: str, feature_col: str = "m3_no_amine_features", k: int = 10, n_splits: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Same fold-internal protocol as the rest of cypmode.challenge:
    outer StratifiedKFold, KNN features built fold-internally (see above),
    concatenated with the existing M3 features, one LightGBM per fold.
    Reports both the plain-M3 MCC and the M3+KNN MCC on the same fold so
    the effect of adding KNN features is directly visible.
    """
    cyp_df = df.dropna(subset=[label_col]).reset_index(drop=True)
    y = cyp_df[label_col].astype(int).to_numpy()
    features = np.stack(cyp_df[feature_col])
    fps = [_morgan_fp(s) for s in cyp_df["SMILES"]]

    rows = []
    outer_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(outer_kfold.split(features, y)):
        knn_feats = add_knn_label_features_for_fold(fps, y, train_idx, val_idx, k=k)
        combined = np.hstack([features, knn_feats])

        model_m3 = LGBMClassifier(verbose=-1, random_state=random_state)
        model_m3.fit(features[train_idx], y[train_idx])
        pred_m3 = model_m3.predict(features[val_idx])

        model_m4 = LGBMClassifier(verbose=-1, random_state=random_state)
        model_m4.fit(combined[train_idx], y[train_idx])
        pred_m4 = model_m4.predict(combined[val_idx])

        rows.append(
            {
                "fold": fold,
                "n_val": len(val_idx),
                "mcc_m3": matthews_corrcoef(y[val_idx], pred_m3),
                "mcc_m3_plus_knn": matthews_corrcoef(y[val_idx], pred_m4),
            }
        )
    return pd.DataFrame(rows)


def _main() -> None:
    from cypmode.challenge.tdi_ablation import add_ablation_features
    from cypmode.challenge.tdi_baseline import (
        TDI_ISOFORMS,
        compute_descriptors,
        load_tdi_training_data,
    )

    df = add_ablation_features(compute_descriptors(load_tdi_training_data()))

    all_results = []
    for cyp in TDI_ISOFORMS:
        results = cross_validate_with_knn_features(df, label_col=f"{cyp}_is_TDI")
        results.insert(0, "isoform", cyp)
        all_results.append(results)
        print(f"{cyp}: mean MCC M3={results['mcc_m3'].mean():.4f}, M3+KNN={results['mcc_m3_plus_knn'].mean():.4f}")

    combined = pd.concat(all_results, ignore_index=True)
    out_path = Path("data/tdi_track/outputs/comparisons/knn_features_ablation.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    _main()
