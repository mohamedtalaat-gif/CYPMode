"""Statistical significance testing for feature-stacking comparisons --
repeated CV with the corrected resampled t of Nadeau and Bengio (Holm-adjusted over the pairs),
not a single point-estimate MCC delta and not an unpaired test on correlated folds.

Motivated directly by a real methodology from Pat Walters'
model-validation-central (patwalters.github.io/model-validation-central):
comparing several methods on one split risks reading real fold-to-fold
noise as a genuine effect ("pooling more methods turns outright wins into
ties" without correction). Applied here to CYPMode's own real,
surprising finding: combining KNN + BindingDB auxiliary features together
scored WORSE for CYP2D6 (single-split MCC 0.054) than either alone (KNN
0.075, BindingDB 0.089) -- a real result that needed to survive repeated,
corrected testing before being trusted as a genuine negative interaction
rather than single-split noise.

Resolved, and it didn't survive -- but not for the reason it was first
checked. This module's own _main() was itself silently using m3_features
(amine flag still included) instead of m3_no_amine_features, the same
bug fixed everywhere else in this project (see
cypmode.challenge.tdi_ablation's docstring), so its first "confirmed"
25-fold CYP2D6 result (M3=0.067, M3+KNN=0.092, M3+BDB=0.075,
M3+KNN+BDB=0.086) was itself computed on the wrong feature set. Re-run
with the corrected features: M3=0.082, M3+KNN=0.066, M3+BDB=0.057,
M3+KNN+BDB=0.089 -- KNN and BindingDB alone no longer even look
positive, and the combined variant is no longer the worst of the four,
it's the best (all four still pairwise non-significant, reject=False
throughout). The original "negative interaction" story does not
replicate; it was an artifact of the amine-flag bug, not a real
CYP2D6-specific effect.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

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
    paired_comparison_summary can treat each (repeat, fold) as one paired
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


def paired_comparison_summary(results_df: pd.DataFrame, value: str = "mcc", alpha: float = 0.05) -> pd.DataFrame:
    """Pairwise comparison of every variant in ``results_df`` (columns ``variant``, ``repeat``, ``fold`` and ``value``).

    Each pair is compared on the per-(repeat, fold) score differences with the corrected resampled t of Nadeau and Bengio (variance factor
    1/J + n_test/n_train, with n_test/n_train = 1/(k - 1) for k folds per repeat), and the p-values are Holm-adjusted over all pairs.
    One row per pair: ``meandiff`` (group2 minus group1), ``p-adj``, a 95% interval from the corrected standard error and ``reject``.
    """
    from itertools import combinations

    from scipy import stats

    from cypmode.challenge.model_comparison import corrected_resampled_t

    frame = results_df.sort_values(["repeat", "fold"]) if {"repeat", "fold"} <= set(results_df.columns) else results_df
    folds = int(frame["fold"].nunique()) if "fold" in frame.columns else 5
    scores = {name: group[value].to_numpy() for name, group in frame.groupby("variant", sort=True)}
    rows = []
    for a, b in combinations(scores, 2):
        diff = scores[b] - scores[a]
        result = corrected_resampled_t(diff, n_train=folds - 1, n_test=1)
        half = float(stats.t.ppf(1 - alpha / 2, df=len(diff) - 1) * result.standard_error)
        rows.append({"group1": a, "group2": b, "meandiff": result.mean_difference, "p_raw": result.p_value, "lower": result.mean_difference - half, "upper": result.mean_difference + half})
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_raw"])
    adjusted, running = [0.0] * len(rows), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * rows[i]["p_raw"]))
        adjusted[i] = running
    out = pd.DataFrame(rows).drop(columns="p_raw") if rows else pd.DataFrame(columns=["group1", "group2", "meandiff", "lower", "upper"])
    out.insert(3, "p-adj", adjusted)
    out["reject"] = out["p-adj"] < alpha
    return out[["group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject"]]


def _main() -> None:
    from cypmode.challenge.bindingdb_auxiliary import (
        BINDINGDB_ISOFORM_UNIPROTS,
        fetch_bindingdb_isoform,
        load_bindingdb_isoform,
    )
    from cypmode.challenge.bindingdb_auxiliary import (
        exclude_exact_overlap as exclude_bdb,
    )
    from cypmode.challenge.bindingdb_auxiliary import fit_auxiliary_model as fit_bdb
    from cypmode.challenge.bindingdb_auxiliary import (
        predict_auxiliary_activity as predict_bdb,
    )
    from cypmode.challenge.knn_features import _morgan_fp
    from cypmode.challenge.tdi_ablation import add_ablation_features
    from cypmode.challenge.tdi_baseline import (
        TDI_ISOFORMS,
        compute_descriptors,
        load_tdi_training_data,
    )

    tdi_df = add_ablation_features(compute_descriptors(load_tdi_training_data()))

    all_results = []
    all_summaries = []
    # both isoforms, run with m3_no_amine_features -- the real production
    # feature set, not m3_features (amine flag still included there for the
    # historical M0-M3 staircase only; see cypmode.challenge.tdi_ablation's
    # own docstring). This module was missed when threshold_tuning,
    # multitask, combined_features, and knn_features were all fixed to use
    # m3_no_amine_features instead -- its previously-reported CYP2D6 25-fold
    # result (M3=0.067, M3+KNN=0.092, M3+BDB=0.075, M3+KNN+BDB=0.086, all
    # pairwise reject=False) was computed on the wrong feature set and is
    # superseded by this run.
    for cyp in TDI_ISOFORMS:
        bdb_path = fetch_bindingdb_isoform(cyp, BINDINGDB_ISOFORM_UNIPROTS[cyp])
        bdb_df = load_bindingdb_isoform(bdb_path)
        bdb_df = exclude_bdb(bdb_df, tdi_df["SMILES"])
        bdb_model = fit_bdb(bdb_df)
        tdi_df[f"bdb_proba_{cyp}"] = predict_bdb(bdb_model, tdi_df)

        cyp_df = tdi_df.dropna(subset=[f"{cyp}_is_TDI"]).reset_index(drop=True)
        y = cyp_df[f"{cyp}_is_TDI"].astype(int).to_numpy()
        m3 = np.stack(cyp_df["m3_no_amine_features"])
        fps = [_morgan_fp(s) for s in cyp_df["SMILES"]]
        aux_features = {"BindingDB": cyp_df[[f"bdb_proba_{cyp}"]].to_numpy()}

        print(f"Running 5x5=25-fold repeated comparison for {cyp}...")
        results = repeated_cv_stacking_comparison(m3, y, fps, aux_features, n_repeats=5, n_splits=5)
        results.insert(0, "isoform", cyp)
        all_results.append(results)

        print(f"\n-- {cyp}: mean MCC per variant (25 folds each) --")
        print(results.groupby("variant")["mcc"].agg(["mean", "std"]).to_string())

        summary = paired_comparison_summary(results)
        summary.insert(0, "isoform", cyp)
        all_summaries.append(summary)
        print(f"\n-- {cyp}: paired corrected-t comparison (Holm) --")
        print(summary.to_string(index=False))

    combined_results = pd.concat(all_results, ignore_index=True)
    out_path = Path("data/tdi_track/outputs/comparisons/stacking_significance_raw.csv")
    combined_results.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    summary_path = Path("data/tdi_track/outputs/comparisons/stacking_significance_tukey.csv")
    combined_summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    _main()
