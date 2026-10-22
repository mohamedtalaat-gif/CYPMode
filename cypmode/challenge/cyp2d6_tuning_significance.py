# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Tests three untried levers for CYP2D6's deployed shared-trunk model --
hyperparameter tuning, class-weighted loss, and seed ensembling -- after
8 independent feature additions (KNN, Tier3 structural, ChemBERTa,
quantum pharmacophore, pharmacophore fingerprint, BindingDB, AID1851,
ChEMBL) all came back null against repeated CV + Tukey HSD. These are
training-procedure changes, not another feature, so they get their own
test rather than being folded into an existing one.

Real, checked facts motivating the specific variants below: CYP2D6's
positive rate (21.6%, 324/1497) is nearly identical to CYP3A4's (21.3%,
764/3584) -- this isn't an unusually imbalanced isoform, so class
weighting is tested because it's cheap and untried, not because the
imbalance itself looks unusual. CYP2D6 does have under half CYP3A4's
labeled compound count, the more likely real driver of every repeated-CV
test's large fold-to-fold variance (std ~0.05-0.08, routinely dwarfing
the effect sizes under test) -- the smaller, more-regularized
architecture variant targets that directly: a smaller model should
overfit a small, noisy dataset less.

Same paired-fold infrastructure as cypmode.challenge.architecture_
significance (same held-out CYP2D6 compounds every fold, shared-trunk
training population = all CYP3A4 rows + that fold's own CYP2D6 training
rows). Inner-OOF threshold tuning always uses a single seed per variant's
own hyperparameters/class-weight (same compute cost as the existing
baseline regardless of variant); only the Seed-ensembled variant's OUTER
prediction averages across multiple seeds -- ensembling the inner
threshold search too would multiply an already real compute cost for a
refinement unlikely to matter much on top of final-prediction averaging.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold

from cypmode.challenge.architecture_significance import (
    BASE_SEED,
    EPOCHS,
    INNER_SPLITS,
    _cyp2d6_names_and_labels,
    _load_full_df,
    repeated_splits,
)
from cypmode.challenge.architecture_significance import (
    RESULTS_PATH as ARCH_RESULTS_PATH,
)

RESULTS_PATH = ARCH_RESULTS_PATH.parent / "tuning_significance_raw.csv"
SUMMARY_PATH = ARCH_RESULTS_PATH.parent / "tuning_significance_tukey.csv"

N_ENSEMBLE_SEEDS = 3

VARIANTS = {
    "Baseline": {"trunk_dim": 64, "head_dim": 32, "dropout": 0.2, "pos_weight": None, "n_outer_seeds": 1},
    "Class-weighted": {"trunk_dim": 64, "head_dim": 32, "dropout": 0.2, "pos_weight": {"CYP2D6": 1173 / 324}, "n_outer_seeds": 1},
    "Smaller+Dropout": {"trunk_dim": 32, "head_dim": 16, "dropout": 0.4, "pos_weight": None, "n_outer_seeds": 1},
    "Seed-ensembled": {"trunk_dim": 64, "head_dim": 32, "dropout": 0.2, "pos_weight": None, "n_outer_seeds": N_ENSEMBLE_SEEDS},
}


def _inner_oof_threshold(
    X_train_raw: np.ndarray,
    y_train: dict,
    mask_train: dict,
    inner_splits: int,
    epochs: int,
    seed: int,
    trunk_dim: int,
    head_dim: int,
    dropout: float,
    pos_weight: dict | None,
) -> dict:
    from cypmode.challenge.multitask import (
        ISOFORMS,
        _find_optimal_threshold,
        _standardize,
        predict_proba,
        train_shared_trunk,
    )

    n = X_train_raw.shape[0]
    inner_oof_proba = {cyp: np.full(n, np.nan, dtype=np.float32) for cyp in ISOFORMS}
    inner_kfold = KFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    for inner_train_idx, inner_val_idx in inner_kfold.split(X_train_raw):
        mean, std = X_train_raw[inner_train_idx].mean(axis=0), X_train_raw[inner_train_idx].std(axis=0) + 1e-8
        X_inner_train = _standardize(X_train_raw[inner_train_idx], mean, std)
        X_inner_val = _standardize(X_train_raw[inner_val_idx], mean, std)

        model = train_shared_trunk(
            X_inner_train,
            {cyp: y_train[cyp][inner_train_idx] for cyp in ISOFORMS},
            {cyp: mask_train[cyp][inner_train_idx] for cyp in ISOFORMS},
            epochs=epochs,
            seed=seed,
            trunk_dim=trunk_dim,
            head_dim=head_dim,
            dropout=dropout,
            pos_weight=pos_weight,
        )
        proba = predict_proba(model, X_inner_val)
        for cyp in ISOFORMS:
            inner_oof_proba[cyp][inner_val_idx] = proba[cyp]

    thresholds = {}
    for cyp in ISOFORMS:
        valid = mask_train[cyp].astype(bool) & ~np.isnan(inner_oof_proba[cyp])
        threshold, _ = _find_optimal_threshold(y_train[cyp][valid].astype(int), inner_oof_proba[cyp][valid])
        thresholds[cyp] = threshold
    return thresholds


def run_variant(variant_name: str) -> pd.DataFrame:
    from cypmode.challenge.multitask import (
        ISOFORMS,
        _standardize,
        predict_proba,
        train_shared_trunk,
    )

    config = VARIANTS[variant_name]
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

        train_full_mask = np.ones(len(df), dtype=bool)
        train_full_mask[val_full_idx] = False
        train_full_idx = np.where(train_full_mask)[0]

        y_train_full = {cyp: y_full[cyp][train_full_idx] for cyp in ISOFORMS}
        mask_train_full = {cyp: mask_full[cyp][train_full_idx] for cyp in ISOFORMS}

        tuned_thresholds = _inner_oof_threshold(
            X_raw_full[train_full_idx], y_train_full, mask_train_full, INNER_SPLITS, EPOCHS, seed,
            config["trunk_dim"], config["head_dim"], config["dropout"], config["pos_weight"],
        )

        mean, std = X_raw_full[train_full_idx].mean(axis=0), X_raw_full[train_full_idx].std(axis=0) + 1e-8
        X_train = _standardize(X_raw_full[train_full_idx], mean, std)
        X_val = _standardize(X_raw_full[val_full_idx], mean, std)

        proba_sum = np.zeros(len(val_full_idx), dtype=np.float64)
        for s in range(config["n_outer_seeds"]):
            outer_seed = seed * 1000 + s
            model = train_shared_trunk(
                X_train, y_train_full, mask_train_full, epochs=EPOCHS, seed=outer_seed,
                trunk_dim=config["trunk_dim"], head_dim=config["head_dim"], dropout=config["dropout"],
                pos_weight=config["pos_weight"],
            )
            proba_sum += predict_proba(model, X_val)["CYP2D6"]
        proba = proba_sum / config["n_outer_seeds"]

        pred = (proba >= tuned_thresholds["CYP2D6"]).astype(int)
        y_val = y[val_idx]
        rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "variant": variant_name,
                "n_val": len(val_idx),
                "mcc": matthews_corrcoef(y_val, pred) if len(set(y_val)) > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=list(VARIANTS), help="which single variant to run and append")
    parser.add_argument("--combine", action="store_true", help="combine all variants' rows and run Tukey HSD")
    args = parser.parse_args()

    if args.variant:
        results = run_variant(args.variant)
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = pd.read_csv(RESULTS_PATH) if RESULTS_PATH.exists() else pd.DataFrame()
        existing = existing[existing["variant"] != args.variant] if len(existing) else existing
        combined = pd.concat([existing, results], ignore_index=True)
        combined.to_csv(RESULTS_PATH, index=False)
        print(f"{args.variant}: mean MCC = {results['mcc'].mean():.4f}, std = {results['mcc'].std():.4f}")
        print(f"wrote {RESULTS_PATH} ({len(combined)} total rows)")

    if args.combine:
        from cypmode.challenge.stacking_significance import paired_comparison_summary

        results = pd.read_csv(RESULTS_PATH)
        missing = set(VARIANTS) - set(results["variant"].unique())
        if missing:
            raise SystemExit(f"need all variants in {RESULTS_PATH} before combining -- missing {missing}")

        print("\n-- mean MCC per variant (25 folds each) --")
        print(results.groupby("variant")["mcc"].agg(["mean", "std", "count"]).to_string())

        scoreable = results.dropna(subset=["mcc"])
        n_dropped = len(results) - len(scoreable)
        if n_dropped:
            print(f"({n_dropped} repeat/fold rows had an undefined MCC -- excluded from the Tukey test)")

        summary = paired_comparison_summary(scoreable)
        print("\n-- Tukey HSD pairwise comparison (all variant pairs) --")
        print(summary.to_string(index=False))
        summary.to_csv(SUMMARY_PATH, index=False)
        print(f"wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    _main()
