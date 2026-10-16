"""Joins a Tier 3 (real Boltz-2 structural evidence) batch into the TDI
feature matrix, and tests whether it moves that isoform's classification
performance. Originally built for CYP3A4 only, per
CYPMode_Architecture_Specification.md's Decision Log ("[shared trunk] Not
treated as final until Tier 3's structural features are wired in for
CYP3A4"); generalized to any isoform (CYP2D6's own due-diligence batch,
same OOF-uncertainty budgeted-selection methodology, see
cypmode.challenge.tier3_selection) via --isoform, since the join/compare
logic below never actually depended on which isoform it was.

Tier 3 covers a fixed budget of 50 compounds only (see
cypmode.challenge.tier3_selection), so on the full labeled population
this feature is missing (NaN) for all but a small fraction of rows --
LightGBM's native NaN handling (already relied on elsewhere in this
project, e.g. the OOF Direct-Inhibition Bridge's own missingness) means
the model can use it where present without needing to impute or drop
rows. But a full-population MCC delta would dilute any real effect
almost to invisibility by construction: the only fair test is the effect
specifically on compounds Tier 3 actually covers, which is what the
repeated-CV comparison here measures (matching this project's own
stated "stratified evaluation by chemical familiarity" design intent in
CYPMode_Architecture_Specification.md's Final architecture diagram).

Real, honest caveat stated up front, not buried: 50 compounds is a small
evaluation population (each fold's validation split covers roughly 10 of
them). The repeated 5x5 CV + Tukey HSD rigor already standard in this
project is applied here for exactly that reason -- a promising-looking
single-split number on 50 rows is even less trustworthy than the same
finding on the full ~3,500-row population, not more.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

TIER3_RESULTS_PATH = Path("data/tdi_track/outputs/cyp3a4/tier3_structural_results.json")
TIER3_FEATURE_COLUMNS = ["tier3_coordinated", "tier3_fe_n_distance", "tier3_confidence_score"]

# Real observed Fe-N distances run ~2-14 A (see CYPMode_Architecture_Specification.md's
# validation table); this sentinel is deliberately far outside that range so a
# tree can split on "a Type II motif nitrogen was present to measure and it
# just wasn't coordinating" as a distinct, informative case from "not in
# Tier 3 at all" (NaN) -- collapsing both into NaN would throw away a real,
# available fact (no motif nitrogen existed here at all) that costs nothing
# to keep.
NO_MOTIF_DISTANCE_SENTINEL = 99.0


def load_tier3_structural_features(results_path: str | Path = TIER3_RESULTS_PATH) -> pd.DataFrame:
    """Real per-compound structural features from Tier 3's completed
    Boltz-2 runs -- one row per compound with a real result, success
    only (a compound that errored out carries no real structural
    evidence and is correctly absent here, not zero-filled).
    """
    results = json.loads(Path(results_path).read_text())
    rows = []
    for name, result in results.items():
        if "error" in result:
            continue
        distance = result.get("closest_motif_nitrogen_distance_angstrom")
        rows.append(
            {
                "Molecule_Name": name,
                "tier3_coordinated": float(result["coordinated"]),
                "tier3_fe_n_distance": float(distance) if distance is not None else NO_MOTIF_DISTANCE_SENTINEL,
                "tier3_confidence_score": float(result["confidence_score"]),
            }
        )
    return pd.DataFrame(rows)


def add_tier3_features(df: pd.DataFrame, results_path: str | Path = TIER3_RESULTS_PATH) -> pd.DataFrame:
    """Left-joins Tier 3's real structural features onto df by
    Molecule_Name. Compounds Tier 3 never ran on -- the overwhelming
    majority, see module docstring -- get NaN in all three tier3_*
    columns and has_tier3_data=False, not a silently-wrong zero (a real
    zero would mean "coordinated=no, 0 A away," which is a false claim,
    not a missing-data marker).
    """
    tier3 = load_tier3_structural_features(results_path)
    out = df.merge(tier3, on="Molecule_Name", how="left")
    out["has_tier3_data"] = out["tier3_coordinated"].notna()
    return out


def repeated_cv_tier3_comparison(
    m3_features: np.ndarray,
    tier3_features: np.ndarray,
    y: np.ndarray,
    has_tier3: np.ndarray,
    n_repeats: int = 5,
    n_splits: int = 5,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Compares M3-alone vs. M3+Tier3 held-out MCC, scored ONLY on the
    validation-fold rows that actually carry real Tier 3 data -- the
    only rows this feature can possibly move, given its fixed
    50-compound budget (see module docstring for why a full-population
    number would be misleadingly diluted). Both models are trained on
    the FULL training-fold population each fold (not just the
    Tier3-covered rows) -- only the *scoring* is restricted to the
    compounds the feature actually covers, matching how every other
    feature addition in this project is trained (on everything
    available) but this one specifically needs its evaluation
    restricted to stay honest about where it can plausibly help.

    Long-format output (one row per repeat x fold x variant) -- directly
    usable by cypmode.challenge.stacking_significance.paired_comparison_summary.
    A repeat/fold with zero Tier3-covered validation rows contributes no
    row at all (nothing to score); one with a single-class Tier3 subset
    (rare at n~10, but real at this small a scale) contributes a NaN mcc
    rather than a fabricated number, matching cypmode.challenge.tdi_ablation's
    own mcc-undefined handling.
    """
    rows = []
    for repeat in range(n_repeats):
        seed = base_seed + repeat
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(m3_features, y)):
            val_tier3_idx = val_idx[has_tier3[val_idx]]
            if len(val_tier3_idx) == 0:
                continue

            m3_model = LGBMClassifier(verbose=-1, random_state=seed)
            m3_model.fit(m3_features[train_idx], y[train_idx])

            combined_train = np.hstack([m3_features[train_idx], tier3_features[train_idx]])
            combined_model = LGBMClassifier(verbose=-1, random_state=seed)
            combined_model.fit(combined_train, y[train_idx])

            y_val = y[val_tier3_idx]
            m3_pred = m3_model.predict(m3_features[val_tier3_idx])
            combined_val = np.hstack([m3_features[val_tier3_idx], tier3_features[val_tier3_idx]])
            combined_pred = combined_model.predict(combined_val)

            can_score = len(set(y_val)) > 1
            for variant, pred in [("M3", m3_pred), ("M3+Tier3", combined_pred)]:
                rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "variant": variant,
                        "n_val_tier3": len(val_tier3_idx),
                        "mcc": matthews_corrcoef(y_val, pred) if can_score else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


CYP2D6_RESULTS_PATH = Path("data/tdi_track/outputs/cyp2d6/tier3_structural_results.json")


def _main() -> None:
    import argparse

    from cypmode.challenge.stacking_significance import paired_comparison_summary
    from cypmode.challenge.tdi_ablation import add_ablation_features
    from cypmode.challenge.tdi_baseline import (
        compute_descriptors,
        load_tdi_training_data,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isoform", default="CYP3A4", choices=["CYP3A4", "CYP2D6"], help="which isoform's TDI label + Tier3 batch to test"
    )
    parser.add_argument(
        "--results-path", type=Path, default=None, help="defaults to that isoform's own real Tier3 results file"
    )
    args = parser.parse_args()
    isoform = args.isoform
    is_default_isoform = isoform == "CYP3A4"

    results_path = args.results_path or (TIER3_RESULTS_PATH if is_default_isoform else CYP2D6_RESULTS_PATH)

    df = add_ablation_features(compute_descriptors(load_tdi_training_data()))
    df = add_tier3_features(df, results_path=results_path)

    label_col = f"{isoform}_is_TDI"
    cyp_df = df.dropna(subset=[label_col]).reset_index(drop=True)
    y = cyp_df[label_col].astype(int).to_numpy()
    # m3_no_amine_features, not m3_features -- see cypmode.challenge.tdi_ablation's
    # own docstring for why m3_features (amine flag still included) is the wrong
    # production base; a real, previously-undiscovered bug, already fixed and
    # re-verified for every other module in this project.
    m3_features = np.stack(cyp_df["m3_no_amine_features"])
    tier3_features = cyp_df[TIER3_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    has_tier3 = cyp_df["has_tier3_data"].to_numpy()

    n_covered = int(has_tier3.sum())
    print(f"{isoform}: {n_covered}/{len(cyp_df)} labeled compounds have real Tier 3 structural data")

    results = repeated_cv_tier3_comparison(m3_features, tier3_features, y, has_tier3)
    print(f"\n{results['n_val_tier3'].groupby(results['repeat']).sum().to_dict()}")
    print("\n-- mean MCC per variant, scored only on Tier3-covered validation rows --")
    print(results.groupby("variant")["mcc"].agg(["mean", "std", "count"]).to_string())

    isoform_dir = Path(f"data/tdi_track/outputs/{isoform.lower()}")
    isoform_dir.mkdir(parents=True, exist_ok=True)
    out_path = isoform_dir / "tier3_significance_raw.csv"
    results.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")

    scoreable = results.dropna(subset=["mcc"])
    n_dropped = len(results) - len(scoreable)
    if n_dropped:
        print(f"({n_dropped} repeat/fold rows had an undefined MCC -- single-class Tier3 validation subset -- excluded from the Tukey test)")

    summary = paired_comparison_summary(scoreable)
    print("\n-- Tukey HSD pairwise comparison (M3 vs M3+Tier3) --")
    print(summary.to_string(index=False))
    summary_path = isoform_dir / "tier3_significance_tukey.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    _main()
