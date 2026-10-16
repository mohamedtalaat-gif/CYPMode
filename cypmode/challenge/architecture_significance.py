"""Tests whether CYP2D6 genuinely does better under an independent-head
model (cypmode.challenge.tdi_baseline/threshold_tuning's own approach,
same family as CYP3A4's deployed model -- a standalone LightGBM trained
on CYP2D6 data alone) or the shared-trunk multitask model
(cypmode.challenge.multitask -- trained jointly with CYP3A4's data via a
shared representation), on CYP2D6's own m3_no_amine_features.

Motivated by a real, striking single-split number: the existing
(pre-dating this module) comparisons data/tdi_track/outputs/comparisons/
shared_trunk_threshold_tuning_no_amine.csv (mean tuned MCC ~0.163) vs
cyp2d6/threshold_tuning_cyp2d6_no_amine.csv (mean tuned MCC ~0.088) --
nearly 2x apart. But those two files are NOT a fair paired comparison:
the shared-trunk one splits with plain (non-stratified) KFold over the
FULL multi-isoform dataframe, while the independent-head one splits
with StratifiedKFold over CYP2D6-only rows -- different populations,
different fold compositions (n_val 302/319/286/281/309 vs
300/300/299/299/299), confounding the architecture comparison with a
splitting-strategy difference. This module fixes that: both models are
scored on the EXACT SAME held-out CYP2D6 compounds each fold (same
seeded StratifiedKFold over CYP2D6-labeled rows only), so the only
real difference between variants is the model itself -- the independent
-head only ever sees CYP2D6 training rows, the shared-trunk model is
trained on all CYP3A4 rows plus that fold's own CYP2D6 training rows
(its real designed advantage: extra signal from CYP3A4's larger,
better-populated label set), consistent with how each is actually meant
to be used.

Process-isolated by design, same real constraint as everywhere else in
this project (torch+lightgbm segfault in the same process -- see
pyproject.toml's `torch` marker): run as two separate invocations,
--model lgbm and --model shared_trunk, each appending its own
repeat/fold rows to one shared CSV; --combine then reads that CSV and
runs the Tukey HSD comparison in a third, model-library-free step.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

RESULTS_PATH = Path("data/tdi_track/outputs/cyp2d6/architecture_significance_raw.csv")
SUMMARY_PATH = Path("data/tdi_track/outputs/cyp2d6/architecture_significance_tukey.csv")

N_REPEATS = 5
N_SPLITS = 5
BASE_SEED = 42
INNER_SPLITS = 3
EPOCHS = 200


def _load_full_df():
    from cypmode.challenge.tdi_ablation import add_ablation_features
    from cypmode.challenge.tdi_baseline import (
        compute_descriptors,
        load_tdi_training_data,
    )

    return add_ablation_features(compute_descriptors(load_tdi_training_data()))


def _cyp2d6_names_and_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cyp_df = df.dropna(subset=["CYP2D6_is_TDI"]).reset_index(drop=True)
    names = cyp_df["Molecule_Name"].to_numpy()
    y = cyp_df["CYP2D6_is_TDI"].astype(int).to_numpy()
    return names, y


def repeated_splits(y: np.ndarray, n_repeats: int = N_REPEATS, n_splits: int = N_SPLITS, base_seed: int = BASE_SEED):
    """Deterministic given the same y -- both the lgbm and shared_trunk
    runs reconstruct `names, y` via the identical pipeline above, so
    calling this with each run's own `y` yields the exact same
    (repeat, fold) -> (train_idx, val_idx) splits in both processes.
    """
    for repeat in range(n_repeats):
        seed = base_seed + repeat
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(y)), y)):
            yield repeat, fold, train_idx, val_idx


def run_independent_head() -> pd.DataFrame:
    from lightgbm import LGBMClassifier
    from sklearn.metrics import matthews_corrcoef

    from cypmode.challenge.threshold_tuning import _inner_oof_threshold

    df = _load_full_df()
    cyp_df = df.dropna(subset=["CYP2D6_is_TDI"]).reset_index(drop=True)
    _, y = _cyp2d6_names_and_labels(df)
    features = np.stack(cyp_df["m3_no_amine_features"])

    rows = []
    for repeat, fold, train_idx, val_idx in repeated_splits(y):
        seed = BASE_SEED + repeat
        tuned_threshold = _inner_oof_threshold(features[train_idx], y[train_idx], INNER_SPLITS, seed)
        model = LGBMClassifier(verbose=-1, random_state=seed)
        model.fit(features[train_idx], y[train_idx])
        val_proba = model.predict_proba(features[val_idx])[:, 1]
        pred = (val_proba >= tuned_threshold).astype(int)
        y_val = y[val_idx]
        rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "variant": "Independent-head",
                "n_val": len(val_idx),
                "mcc": matthews_corrcoef(y_val, pred) if len(set(y_val)) > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def run_shared_trunk() -> pd.DataFrame:
    from sklearn.metrics import matthews_corrcoef

    from cypmode.challenge.multitask import (
        ISOFORMS,
        _inner_oof_threshold_shared_trunk,
        _standardize,
        predict_proba,
        train_shared_trunk,
    )

    df = _load_full_df()
    names, y = _cyp2d6_names_and_labels(df)
    name_to_row = {n: i for i, n in enumerate(df["Molecule_Name"])}

    X_raw_full = np.stack(df["m3_no_amine_features"]).astype(np.float32)
    y_full = {cyp: df[f"{cyp}_is_TDI"].fillna(0).astype(int).to_numpy().astype(np.float32) for cyp in ISOFORMS}
    mask_full = {cyp: df[f"{cyp}_is_TDI"].notna().to_numpy().astype(np.float32) for cyp in ISOFORMS}

    rows = []
    for repeat, fold, train_idx, val_idx in repeated_splits(y):
        seed = BASE_SEED + repeat
        val_full_idx = np.array([name_to_row[n] for n in names[val_idx]])

        # Training set = every full-df row except this fold's held-out CYP2D6
        # compounds -- includes ALL CYP3A4 rows (the shared-trunk model's real
        # designed advantage) plus this fold's own CYP2D6 training rows.
        train_full_mask = np.ones(len(df), dtype=bool)
        train_full_mask[val_full_idx] = False
        train_full_idx = np.where(train_full_mask)[0]

        y_train_full = {cyp: y_full[cyp][train_full_idx] for cyp in ISOFORMS}
        mask_train_full = {cyp: mask_full[cyp][train_full_idx] for cyp in ISOFORMS}

        tuned_thresholds = _inner_oof_threshold_shared_trunk(
            X_raw_full[train_full_idx], y_train_full, mask_train_full, INNER_SPLITS, EPOCHS, seed
        )

        mean, std = X_raw_full[train_full_idx].mean(axis=0), X_raw_full[train_full_idx].std(axis=0) + 1e-8
        X_train = _standardize(X_raw_full[train_full_idx], mean, std)
        X_val = _standardize(X_raw_full[val_full_idx], mean, std)

        model = train_shared_trunk(X_train, y_train_full, mask_train_full, epochs=EPOCHS, seed=seed)
        proba = predict_proba(model, X_val)["CYP2D6"]
        pred = (proba >= tuned_thresholds["CYP2D6"]).astype(int)
        y_val = y[val_idx]
        rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "variant": "Shared-trunk",
                "n_val": len(val_idx),
                "mcc": matthews_corrcoef(y_val, pred) if len(set(y_val)) > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["lgbm", "shared_trunk"], help="which variant to run and append")
    parser.add_argument("--combine", action="store_true", help="combine both variants' rows and run Tukey HSD")
    args = parser.parse_args()

    if args.model:
        results = run_independent_head() if args.model == "lgbm" else run_shared_trunk()
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = pd.read_csv(RESULTS_PATH) if RESULTS_PATH.exists() else pd.DataFrame()
        existing = existing[existing["variant"] != results["variant"].iloc[0]] if len(existing) else existing
        combined = pd.concat([existing, results], ignore_index=True)
        combined.to_csv(RESULTS_PATH, index=False)
        print(f"{args.model}: mean MCC = {results['mcc'].mean():.4f}, std = {results['mcc'].std():.4f}")
        print(f"wrote {RESULTS_PATH} ({len(combined)} total rows)")

    if args.combine:
        from cypmode.challenge.stacking_significance import paired_comparison_summary

        results = pd.read_csv(RESULTS_PATH)
        variants_present = set(results["variant"].unique())
        if variants_present != {"Independent-head", "Shared-trunk"}:
            raise SystemExit(f"need both variants in {RESULTS_PATH} before combining -- have {variants_present}")

        print("\n-- mean MCC per variant (25 folds each) --")
        print(results.groupby("variant")["mcc"].agg(["mean", "std", "count"]).to_string())

        scoreable = results.dropna(subset=["mcc"])
        n_dropped = len(results) - len(scoreable)
        if n_dropped:
            print(f"({n_dropped} repeat/fold rows had an undefined MCC -- excluded from the Tukey test)")

        summary = paired_comparison_summary(scoreable)
        print("\n-- Tukey HSD pairwise comparison (Independent-head vs Shared-trunk) --")
        print(summary.to_string(index=False))
        summary.to_csv(SUMMARY_PATH, index=False)
        print(f"wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    _main()
