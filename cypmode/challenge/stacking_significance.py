"""Statistical significance testing for feature-stacking comparisons --
repeated CV + Tukey HSD, not a single point-estimate MCC delta.

Motivated directly by a real methodology from Pat Walters'
model-validation-central (patwalters.github.io/model-validation-central):
comparing several methods on one split risks reading real fold-to-fold
noise as a genuine effect ("pooling more methods turns outright wins into
ties" without correction). Applied here to CYPMode's own real,
surprising finding: combining KNN + BindingDB auxiliary features together
scored WORSE for CYP2D6 (single-split MCC 0.054) than either alone (KNN
0.075, BindingDB 0.089) -- a real result that needs to survive repeated,
corrected testing before being trusted as a genuine negative interaction
rather than single-split noise.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from cypmode.challenge.knn_features import add_knn_label_features_for_fold


def repeated_cv_stacking_comparison(
    m3_features: np.ndarray,
    y: np.ndarray,
    fps: list,
    aux_features: dict[str, np.ndarray],
    n_repeats: int = 5,
    n_splits: int = 5,
    k: int = 10,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Repeated-CV comparison of M3 alone, M3+KNN, M3+each named aux
    feature alone, and M3+KNN+all aux features together (the real
    variant set CYPMode's own stacking ablations compare). KNN is
    recomputed fold-internally on every fold of every repeat, not
    reused across repeats or precomputed globally -- the same leakage
    discipline used everywhere else KNN features appear in this project,
    not shortcut just because this is a statistics pass.

    Long-format output (one row per repeat x fold x variant) so
    tukey_hsd_summary can treat each (repeat, fold) as one paired
    observation per variant.
    """
    rows = []
    for repeat in range(n_repeats):
        seed = base_seed + repeat
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(m3_features, y)):
            knn_feats = add_knn_label_features_for_fold(fps, y, train_idx, val_idx, k=k)

            variants = {"M3": m3_features, "M3+KNN": np.hstack([m3_features, knn_feats])}
            for aux_name, aux_values in aux_features.items():
                variants[f"M3+{aux_name}"] = np.hstack([m3_features, aux_values])
            if aux_features:
                combined_name = "M3+KNN+" + "+".join(aux_features.keys())
                variants[combined_name] = np.hstack([m3_features, knn_feats, *aux_features.values()])

            for name, X in variants.items():
                model = LGBMClassifier(verbose=-1, random_state=seed)
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[val_idx])
                rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "variant": name,
                        "mcc": matthews_corrcoef(y[val_idx], pred),
                    }
                )
    return pd.DataFrame(rows)


def tukey_hsd_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Tukey HSD across every variant in results_df's `variant`
    column, correcting for comparing more than two groups at once --
    Pat Walters' own point about pooling multiple methods without
    correction turning real differences into apparent ties (or, just as
    real a risk, apparent differences into noise). One row per pairwise
    comparison: mean MCC difference, whether it's significant at
    alpha=0.05 after correction, and the confidence interval.
    """
    result = pairwise_tukeyhsd(results_df["mcc"].to_numpy(), results_df["variant"].to_numpy(), alpha=0.05)
    return pd.DataFrame(data=result._results_table.data[1:], columns=result._results_table.data[0])


def _main() -> None:
    from cypmode.challenge.bindingdb_auxiliary import BINDINGDB_ISOFORM_UNIPROTS
    from cypmode.challenge.bindingdb_auxiliary import exclude_exact_overlap as exclude_bdb
    from cypmode.challenge.bindingdb_auxiliary import fetch_bindingdb_isoform
    from cypmode.challenge.bindingdb_auxiliary import fit_auxiliary_model as fit_bdb
    from cypmode.challenge.bindingdb_auxiliary import load_bindingdb_isoform
    from cypmode.challenge.bindingdb_auxiliary import predict_auxiliary_activity as predict_bdb
    from cypmode.challenge.knn_features import _morgan_fp
    from cypmode.challenge.tdi_ablation import add_ablation_features
    from cypmode.challenge.tdi_baseline import TDI_ISOFORMS, compute_descriptors, load_tdi_training_data

    tdi_df = add_ablation_features(compute_descriptors(load_tdi_training_data()))

    all_results = []
    all_summaries = []
    # both isoforms now, not just CYP2D6 -- CYP2D6's own 25-fold result
    # (M3=0.067, M3+KNN=0.092, M3+BDB=0.075, M3+KNN+BDB=0.086, all
    # pairwise reject=False) already exists from an earlier run; this
    # applies the identical rigor to CYP3A4, whose only real result so
    # far was a single-split combined-features number (0.321) that
    # hasn't been checked against fold-to-fold noise the same way.
    for cyp in TDI_ISOFORMS:
        bdb_path = fetch_bindingdb_isoform(cyp, BINDINGDB_ISOFORM_UNIPROTS[cyp])
        bdb_df = load_bindingdb_isoform(bdb_path)
        bdb_df = exclude_bdb(bdb_df, tdi_df["SMILES"])
        bdb_model = fit_bdb(bdb_df)
        tdi_df[f"bdb_proba_{cyp}"] = predict_bdb(bdb_model, tdi_df)

        cyp_df = tdi_df.dropna(subset=[f"{cyp}_is_TDI"]).reset_index(drop=True)
        y = cyp_df[f"{cyp}_is_TDI"].astype(int).to_numpy()
        m3 = np.stack(cyp_df["m3_features"])
        fps = [_morgan_fp(s) for s in cyp_df["SMILES"]]
        aux_features = {"BindingDB": cyp_df[[f"bdb_proba_{cyp}"]].to_numpy()}

        print(f"Running 5x5=25-fold repeated comparison for {cyp}...")
        results = repeated_cv_stacking_comparison(m3, y, fps, aux_features, n_repeats=5, n_splits=5)
        results.insert(0, "isoform", cyp)
        all_results.append(results)

        print(f"\n-- {cyp}: mean MCC per variant (25 folds each) --")
        print(results.groupby("variant")["mcc"].agg(["mean", "std"]).to_string())

        summary = tukey_hsd_summary(results)
        summary.insert(0, "isoform", cyp)
        all_summaries.append(summary)
        print(f"\n-- {cyp}: Tukey HSD pairwise comparison --")
        print(summary.to_string(index=False))

    combined_results = pd.concat(all_results, ignore_index=True)
    out_path = Path("data/tdi_track/outputs/stacking_significance_raw.csv")
    combined_results.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    summary_path = Path("data/tdi_track/outputs/stacking_significance_tukey.csv")
    combined_summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    _main()
