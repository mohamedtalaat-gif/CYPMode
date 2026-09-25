"""Cross-cutting reliability score: ensemble disagreement + chemical OOD
distance (+ Boltz-2 structural confidence, wired in once Tier 3 results
exist). Deliberately kept outside the prediction path -- see
CYPMode-TDI_Architecture_Specification.md. Not called "uncertainty
quantification" anywhere until validate_reliability_correlation shows it
actually tracks real error; until then it's just three computed numbers.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.stats import pointbiserialr, spearmanr


def ensemble_disagreement(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    apply_features: np.ndarray,
    n_models: int = 5,
    base_seed: int = 42,
) -> np.ndarray:
    """Std deviation of predicted P(TDI) across n_models LGBMClassifiers,
    same training data, different random seeds only. High disagreement =
    the training signal doesn't pin down a confident answer for that
    compound. All n_models are fit on train_features/train_labels only --
    apply_features never influences training.
    """
    predictions = np.zeros((n_models, len(apply_features)))
    for i in range(n_models):
        model = LGBMClassifier(verbose=-1, random_state=base_seed + i)
        model.fit(train_features, train_labels)
        predictions[i] = model.predict_proba(apply_features)[:, 1]
    return predictions.std(axis=0)


def _morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto_ood_distance(train_smiles: pd.Series, apply_smiles: pd.Series) -> np.ndarray:
    """1 - max(Tanimoto similarity to any training compound), per compound
    in apply_smiles -- 0 means chemically identical to something in
    training, close to 1 means genuinely novel chemistry. Same ECFP4
    (radius 2, 2048 bits) the official tutorial itself uses for its own
    train/test similarity check, so this is directly comparable to that,
    not a different, incompatible metric.
    """
    train_fps = [_morgan_fp(s) for s in train_smiles]
    distances = np.empty(len(apply_smiles))
    for i, smiles in enumerate(apply_smiles):
        fp = _morgan_fp(smiles)
        max_sim = max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
        distances[i] = 1.0 - max_sim
    return distances


def one_class_svm_ood_score(
    train_features: np.ndarray, apply_features: np.ndarray, nu: float = 0.1, random_state: int = 42
) -> np.ndarray:
    """One-class SVM anomaly score for how out-of-distribution each
    apply_features compound is, relative to train_features -- fit ONLY on
    the training portion (never apply_features), same fold-internal rule
    as ensemble_disagreement. Higher = more anomalous, matching
    tanimoto_ood_distance's own "higher = more OOD" convention (sklearn's
    own decision_function returns the opposite sign -- negated here for
    consistency across this module's components, not a different metric).

    A generic, textbook anomaly-detection pattern (scale features, fit a
    one-class SVM on the "normal" population, score new points by
    distance from the learned boundary) -- not from a specific tool, and
    a genuinely different signal from tanimoto_ood_distance: this operates
    in RDKit-descriptor space (continuous, correlated physicochemical
    properties), not ECFP4 fingerprint space (discrete substructure
    presence/absence), so the two can disagree on which compounds count
    as novel.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    apply_scaled = scaler.transform(apply_features)

    ocsvm = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
    ocsvm.fit(train_scaled)
    return -ocsvm.decision_function(apply_scaled)


def compute_reliability_score(
    ensemble_disagreement_values: np.ndarray,
    ood_distance_values: np.ndarray,
    svm_ood_score_values: np.ndarray | None = None,
    structural_confidence_values: np.ndarray | None = None,
) -> pd.DataFrame:
    """Combines the components into one table (not one number by fiat --
    z-scoring and summing would itself be an unvalidated modeling choice).
    svm_ood_score_values and structural_confidence_values are both
    optional; structural_confidence_values is NaN-filled where Tier 3
    wasn't run for a compound. validate_reliability_correlation handles
    NaNs by dropping them per-component, not by imputing.
    """
    out = pd.DataFrame(
        {
            "ensemble_disagreement": ensemble_disagreement_values,
            "ood_distance": ood_distance_values,
        }
    )
    if svm_ood_score_values is not None:
        out["svm_ood_score"] = svm_ood_score_values
    if structural_confidence_values is not None:
        out["structural_confidence"] = structural_confidence_values
    return out


def validate_reliability_correlation(reliability_df: pd.DataFrame, prediction_error: np.ndarray) -> pd.DataFrame:
    """The empirical test this whole module exists for: does each
    reliability component actually correlate with real prediction error
    on held-out data? Reports both Pearson-style point-biserial (error is
    typically binary: correct/incorrect) and Spearman rank correlation,
    per component, dropping NaNs pairwise rather than imputing them.
    Nothing here is called "validated" or "uncertainty quantification"
    until this has been run and reported.
    """
    rows = []
    for column in reliability_df.columns:
        values = reliability_df[column]
        mask = values.notna()
        if mask.sum() < 3:
            rows.append({"component": column, "n": int(mask.sum()), "point_biserial_r": np.nan, "spearman_r": np.nan})
            continue
        pb_r, _ = pointbiserialr(prediction_error[mask], values[mask])
        sp_r, _ = spearmanr(prediction_error[mask], values[mask])
        rows.append({"component": column, "n": int(mask.sum()), "point_biserial_r": pb_r, "spearman_r": sp_r})
    return pd.DataFrame(rows)


def _main() -> None:
    from sklearn.model_selection import StratifiedKFold

    from cypmode.challenge.tdi_baseline import compute_descriptors, load_tdi_training_data

    df = compute_descriptors(load_tdi_training_data())
    cyp_df = df.dropna(subset=["CYP3A4_is_TDI"]).reset_index(drop=True)
    y = cyp_df["CYP3A4_is_TDI"].astype(int).to_numpy()
    features = np.stack(cyp_df["descriptors"])

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(kfold.split(features, y))

    disagreement = ensemble_disagreement(features[train_idx], y[train_idx], features[val_idx])
    ood = tanimoto_ood_distance(cyp_df.loc[train_idx, "SMILES"], cyp_df.loc[val_idx, "SMILES"])
    svm_ood = one_class_svm_ood_score(features[train_idx], features[val_idx])

    model = LGBMClassifier(verbose=-1, random_state=42)
    model.fit(features[train_idx], y[train_idx])
    proba = model.predict_proba(features[val_idx])[:, 1]
    pred = (proba >= 0.5).astype(int)
    error = (pred != y[val_idx]).astype(int)

    reliability_df = compute_reliability_score(disagreement, ood, svm_ood)
    correlation = validate_reliability_correlation(reliability_df, error)
    print(correlation.to_string(index=False))

    out_path = Path("data/cyp_challenge/tdi_track/reliability_correlation.csv")
    correlation.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    _main()
