"""M0-M3 ablation: does CYPMode's Tier 2 evidence (motif + trialkylamine
flags) actually help a real multivariate TDI classifier, beyond the plain
RDKit-descriptor baseline?

CYPMode_DATA_AUDIT.md found that the motif+amine categorization has no
useful *marginal* (univariate) association with real TDI status across the
full training population -- the "neither" category had the highest
TDI-positive rate for both isoforms, the opposite of what the small 6-drug
reference panel suggested. That result doesn't settle the question on its
own: a marginal cross-tab can miss a real effect a multivariate model
picks up (e.g. the amine flag might matter only in combination with
specific descriptor ranges), and it can also just confirm the null. This
module runs the actual, honest test -- four models, same CV folds
(cypmode.challenge.tdi_baseline.cross_validate_fixed_seed's feature_col
pairing), same everything else, one feature block added at a time:

M0 = RDKit descriptors only (the reproduced official baseline)
M1 = M0 + motif flag
M2 = M1 + trialkylamine flag
M3 = M2 + Morgan fingerprint (ECFP4, radius 2, 2048 bits)

If M2 doesn't beat M1, Tier 2's amine flag is dropped, not kept out of
attachment to CYPMode's own prior work -- see README.md's Scope and
limitations for how this project has handled negative results before.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from cypmode.challenge.tdi_baseline import (
    compute_descriptors,
    cross_validate_fixed_seed,
    load_tdi_training_data,
)
from cypmode.metrics.amines import has_trialkylamine
from cypmode.metrics.motifs import detect_motifs

MODELS = ["M0", "M1", "M2", "M3"]


def _morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float64)


def add_ablation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds motif_flag, amine_flag, and morgan_fp columns, plus one combined
    feature-vector column per model (m0_features .. m3_features), so
    cross_validate_fixed_seed can score each with feature_col=.

    m3_features keeps the amine flag on purpose: it's the last stage of
    the M0-M3 ablation staircase (M2 = M1 + amine, M3 = M2 + Morgan), and
    changing what it contains would retroactively redefine every already-
    computed M0-M3 ablation result (data/tdi_track/outputs/comparisons/ablation_m0_m3.csv).
    See m3_no_amine_features below for the actual production feature set.
    """
    out = df.copy()
    out["motif_flag"] = out["SMILES"].apply(lambda s: np.array([float(bool(detect_motifs(s)))]))
    out["amine_flag"] = out["SMILES"].apply(lambda s: np.array([float(has_trialkylamine(s))]))
    out["morgan_fp"] = out["SMILES"].apply(_morgan_fp)

    out["m0_features"] = out["descriptors"]
    out["m1_features"] = [
        np.concatenate([d, m]) for d, m in zip(out["descriptors"], out["motif_flag"], strict=True)
    ]
    out["m2_features"] = [
        np.concatenate([d, m, a])
        for d, m, a in zip(out["descriptors"], out["motif_flag"], out["amine_flag"], strict=True)
    ]
    out["m3_features"] = [
        np.concatenate([d, m, a, fp])
        for d, m, a, fp in zip(
            out["descriptors"], out["motif_flag"], out["amine_flag"], out["morgan_fp"], strict=True
        )
    ]
    # The actual settled decision (CYPMode_Architecture_Specification.md's Decision Log:
    # "Tier 2: motif only, amine dropped") was never implemented in m3_features above --
    # every downstream module (threshold_tuning, multitask, combined_features, knn_features)
    # defaults feature_col to "m3_features" and so silently kept the amine flag the project
    # decided to drop. Confirmed directly (same fixed 5-fold protocol, naive threshold):
    # CYP2D6 MCC 0.043 (with amine) -> 0.072 (without, +68% relative); CYP3A4 flat
    # (0.300 -> 0.295). This is the real, decision-matching production feature set --
    # cypmode.challenge.final_assembly and any future comparison meant to reflect the
    # settled architecture should use this, not m3_features.
    out["m3_no_amine_features"] = [
        np.concatenate([d, m, fp])
        for d, m, fp in zip(out["descriptors"], out["motif_flag"], out["morgan_fp"], strict=True)
    ]
    return out


def run_ablation(df: pd.DataFrame, n_splits: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Runs cross_validate_fixed_seed once per model (M0-M3), same folds
    each time (same df, same random_state), and tags each result row with
    which model produced it.
    """
    all_results = []
    for model_name in MODELS:
        results = cross_validate_fixed_seed(
            df, n_splits=n_splits, random_state=random_state, feature_col=f"{model_name.lower()}_features"
        )
        results.insert(0, "model", model_name)
        all_results.append(results)
    return pd.concat(all_results, ignore_index=True)


def summarize_ablation(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby(["model", "isoform"])[["accuracy", "f1", "mcc", "roc_auc"]].mean()


def _main() -> None:
    df = add_ablation_features(compute_descriptors(load_tdi_training_data()))
    results = run_ablation(df)
    summary = summarize_ablation(results)
    print(summary.to_string())

    out_path = Path("data/tdi_track/outputs/comparisons/ablation_m0_m3.csv")
    results.to_csv(out_path, index=False)
    print(f"\nwrote per-fold results to {out_path}")


if __name__ == "__main__":
    _main()
