"""Tests whether Tier 3 structural features and KNN read-across, combined,
move classification performance -- specifically for CYP2D6, where each
was already validated SEPARATELY and each came back null-to-negative on
its own (see cypmode.challenge.tier3_features and
cypmode.challenge.knn_features's own CYP2D6 results). Whether the two
interact positively despite each being null alone was real, unmeasured
work until this module -- not something safe to assume in either
direction (see cypmode.challenge.combined_features and
cypmode.challenge.stacking_significance's own docstrings for the same
reasoning applied to other feature combinations in this project).

Scored ONLY on Tier3-covered validation rows, same restricted-scoring
rationale as cypmode.challenge.tier3_features: Tier 3 covers a fixed
50-compound budget, so a full-population MCC delta would dilute any real
effect almost to invisibility by construction, and this keeps the
M3/M3+Tier3 numbers here directly comparable to tier3_features.py's own
already-reported result for the same population. KNN is still built
fold-internally over the FULL training population each fold (not
restricted to Tier3-covered rows) -- it's a real project-wide feature
that just happens to be scored on this same restricted subset here, not
a Tier3-only feature itself.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

from cypmode.challenge.knn_features import _morgan_fp, add_knn_label_features_for_fold
from cypmode.challenge.stacking_significance import paired_comparison_summary
from cypmode.challenge.tier3_features import (
    CYP2D6_RESULTS_PATH,
    TIER3_FEATURE_COLUMNS,
    add_tier3_features,
)


def repeated_cv_tier3_knn_comparison(
    m3_features: np.ndarray,
    tier3_features: np.ndarray,
    fps: list,
    y: np.ndarray,
    has_tier3: np.ndarray,
    n_repeats: int = 5,
    n_splits: int = 5,
    k: int = 10,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Repeated-CV comparison of M3, M3+Tier3, M3+KNN, and
    M3+Tier3+KNN, scored only on the Tier3-covered validation rows each
    fold (see module docstring). KNN features are recomputed
    fold-internally every fold of every repeat -- the same leakage
    discipline used everywhere else KNN appears in this project.

    Long-format output (one row per repeat x fold x variant), directly
    usable by cypmode.challenge.stacking_significance.paired_comparison_summary.
    A repeat/fold with zero Tier3-covered validation rows contributes no
    rows; one with a single-class Tier3 validation subset contributes a
    NaN mcc rather than a fabricated number, matching
    cypmode.challenge.tier3_features's own handling.
    """
    rows = []
    for repeat in range(n_repeats):
        seed = base_seed + repeat
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(m3_features, y)):
            val_tier3_idx = val_idx[has_tier3[val_idx]]
            if len(val_tier3_idx) == 0:
                continue

            knn_feats = add_knn_label_features_for_fold(fps, y, train_idx, val_idx, k=k)

            variants = {
                "M3": m3_features,
                "M3+Tier3": np.hstack([m3_features, tier3_features]),
                "M3+KNN": np.hstack([m3_features, knn_feats]),
                "M3+Tier3+KNN": np.hstack([m3_features, tier3_features, knn_feats]),
            }

            y_val = y[val_tier3_idx]
            can_score = len(set(y_val)) > 1
            for name, X in variants.items():
                model = LGBMClassifier(verbose=-1, random_state=seed)
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[val_tier3_idx])
                rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "variant": name,
                        "n_val_tier3": len(val_tier3_idx),
                        "mcc": matthews_corrcoef(y_val, pred) if can_score else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def _main() -> None:
    import argparse

    from cypmode.challenge.tdi_ablation import add_ablation_features
    from cypmode.challenge.tdi_baseline import (
        compute_descriptors,
        load_tdi_training_data,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", default="CYP2D6", choices=["CYP3A4", "CYP2D6"])
    parser.add_argument("--results-path", type=Path, default=None)
    args = parser.parse_args()
    isoform = args.isoform
    results_path = args.results_path or (
        CYP2D6_RESULTS_PATH if isoform == "CYP2D6" else Path("data/tdi_track/outputs/cyp3a4/tier3_structural_results.json")
    )

    df = add_ablation_features(compute_descriptors(load_tdi_training_data()))
    df = add_tier3_features(df, results_path=results_path)

    label_col = f"{isoform}_is_TDI"
    cyp_df = df.dropna(subset=[label_col]).reset_index(drop=True)
    y = cyp_df[label_col].astype(int).to_numpy()
    # m3_no_amine_features -- the real production feature set, not m3_features
    # (amine flag still included there for the historical M0-M3 staircase only;
    # see cypmode.challenge.tdi_ablation's own docstring).
    m3_features = np.stack(cyp_df["m3_no_amine_features"])
    tier3_features = cyp_df[TIER3_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    has_tier3 = cyp_df["has_tier3_data"].to_numpy()
    fps = [_morgan_fp(s) for s in cyp_df["SMILES"]]

    n_covered = int(has_tier3.sum())
    print(f"{isoform}: {n_covered}/{len(cyp_df)} labeled compounds have real Tier 3 structural data")

    results = repeated_cv_tier3_knn_comparison(m3_features, tier3_features, fps, y, has_tier3)
    print(f"\n{results['n_val_tier3'].groupby(results['repeat']).sum().to_dict()}")
    print("\n-- mean MCC per variant, scored only on Tier3-covered validation rows --")
    print(results.groupby("variant")["mcc"].agg(["mean", "std", "count"]).to_string())

    isoform_dir = Path(f"data/tdi_track/outputs/{isoform.lower()}")
    isoform_dir.mkdir(parents=True, exist_ok=True)
    out_path = isoform_dir / "tier3_knn_significance_raw.csv"
    results.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")

    scoreable = results.dropna(subset=["mcc"])
    n_dropped = len(results) - len(scoreable)
    if n_dropped:
        print(f"({n_dropped} repeat/fold rows had an undefined MCC -- single-class Tier3 validation subset -- excluded from the Tukey test)")

    summary = paired_comparison_summary(scoreable)
    print("\n-- Tukey HSD pairwise comparison (all variant pairs) --")
    print(summary.to_string(index=False))
    summary_path = isoform_dir / "tier3_knn_significance_tukey.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    _main()
