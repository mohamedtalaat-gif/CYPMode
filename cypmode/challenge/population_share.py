"""How many never-measured compounds does the scored CYP2D6 test set contain? Inferred from public baselines.

The selection-aware model only helps if the scored set holds many compounds that the screen would never have
promoted on CYP2D6 ("non-hits", scored as inactive). Cross-validation cannot say how many, but the official
baseline (descriptor LightGBM trained on hits) has public leaderboard accuracy, precision and recall, and its
behaviour on a population depends on that mix. Using the baseline's out-of-fold predictions for EVERY
training compound (hits and non-hits), the training population is re-weighted by two numbers,

    w   weight of every non-hit compound (0 = hits only, 1 = the training population),
    c   weight of the positive hits relative to the negative hits (the test's TDI rate among hits may differ),

and the weighted accuracy, precision and recall at the baseline's 0.5 threshold are compared with its
leaderboard values. The (w, c) pairs within a chi-square 95% region of the best fit are the plausible
population mixes; the models are then judged across that whole band, not at a single guessed share.

Usage: python -m cypmode.challenge.population_share [--board live|interim]
"""

import argparse

import numpy as np
import pandas as pd

TRAIN = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
SNAP = "data/tdi_track/inputs/leaderboard_snapshots"
ISO = "CYP2D6"


def baseline_oof(n_jobs: int = 3) -> pd.DataFrame:
    """Out-of-fold baseline probabilities for every training compound; the model only ever sees hits."""
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold

    from cypmode.challenge.tdi_baseline import compute_descriptors

    from cypmode.challenge.feature_store import FeatureStore
    from cypmode.challenge.validation_matching import SimilarityMatcher

    raw = pd.read_csv(TRAIN)
    test = pd.read_csv("data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv")
    bits = FeatureStore.load_or_build("data/tdi_track/outputs/caches/feature_store.npz", list(raw.SMILES) + list(test.SMILES), ["morgan_bits"]).blocks["morgan_bits"]
    matcher = SimilarityMatcher(bits, np.arange(len(raw), len(raw) + len(test)))
    desc = compute_descriptors(raw[["Molecule_Name", "SMILES"]])
    x = np.stack(desc["descriptors"])
    hit = raw[f"{ISO}_is_TDI"].notna().to_numpy()
    y = raw[f"{ISO}_is_TDI"].fillna(0).astype(int).to_numpy()
    p = np.zeros(len(raw))
    sim = np.ones(len(raw))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(x, hit * 2 + y):
        h = tr[hit[tr]]
        p[te] = LGBMClassifier(verbose=-1, random_state=42, n_jobs=n_jobs).fit(x[h], y[h]).predict_proba(x[te])[:, 1]
        sim[te] = matcher(te, tr)  # how test-like each held-out compound is, given the fold's training reference
    return pd.DataFrame({"hit": hit, "y": y, "p": p, "sim": sim})


def weighted_metrics(frame: pd.DataFrame, w: float, c: float) -> dict:
    weight = np.where(frame["hit"], np.where(frame["y"] == 1, c, 1.0), w) * frame["sim"].to_numpy()
    y, pred = frame["y"].to_numpy(), (frame["p"].to_numpy() >= 0.5).astype(int)
    tp = float((weight * (y == 1) * (pred == 1)).sum())
    fp = float((weight * (y == 0) * (pred == 1)).sum())
    fn = float((weight * (y == 1) * (pred == 0)).sum())
    tn = float((weight * (y == 0) * (pred == 0)).sum())
    total = tp + fp + fn + tn
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "prevalence": (tp + fn) / total,
        "accuracy": (tp + tn) / total,
        "precision": tp / max(tp + fp, 1e-12),
        "recall": tp / max(tp + fn, 1e-12),
        "mcc": (tp * tn - fp * fn) / denom if denom > 0 else 0.0,
        "called_rate": (tp + fp) / total,
    }


def fit_band(frame: pd.DataFrame, board: pd.Series, w_grid, c_grid) -> pd.DataFrame:
    rows = []
    for w in w_grid:
        for c in c_grid:
            m = weighted_metrics(frame, w, c)
            loss = 0.0
            for key, col in (("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall")):
                sd = max(float(board.get(col + " std", 0.0)), 0.01)
                loss += ((m[key] - float(board[col])) / sd) ** 2
            rows.append({"w": w, "c": c, **m, "chi2": loss})
    out = pd.DataFrame(rows)
    out["plausible"] = out["chi2"] <= out["chi2"].min() + 7.81  # 95% region for 3 degrees of freedom
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", default="live", choices=["live", "interim"])
    args = parser.parse_args()
    frame = baseline_oof()
    date = "2026-10-07" if args.board == "live" else "2026-10-04"
    f = f"{SNAP}/{args.board}_{ISO}_{date}.csv"
    board_all = pd.read_csv(f)
    row = board_all[board_all["Model Report Link"].astype(str).str.contains("Official LightGBM baseline", na=False)].iloc[0]
    print(f"{ISO} {args.board} board, official baseline: accuracy {row['Accuracy']:.4f}, precision {row['Precision']:.4f}, recall {row['Recall']:.4f}, MCC {row['MCC']:.4f}")
    grid = fit_band(frame, row, np.round(np.linspace(0, 1.5, 31), 3), np.round(np.geomspace(0.3, 3.0, 25), 3))
    best = grid.loc[grid["chi2"].idxmin()]
    band = grid[grid["plausible"]]
    print(f"best fit: w = {best['w']:.2f}, c = {best['c']:.2f}, chi2 = {best['chi2']:.2f}; predicted prevalence {best['prevalence']:.3f}, accuracy {best['accuracy']:.3f}, precision {best['precision']:.3f}, recall {best['recall']:.3f}, MCC {best['mcc']:.3f}")
    print(f"plausible band ({len(band)} of {len(grid)} grid points): w from {band['w'].min():.2f} to {band['w'].max():.2f}, c from {band['c'].min():.2f} to {band['c'].max():.2f}")
    for w in sorted(band["w"].unique()):
        sub = band[band["w"] == w]
        print(f"   w = {w:.2f}: c in [{sub['c'].min():.2f}, {sub['c'].max():.2f}], predicted prevalence {sub['prevalence'].min():.3f}-{sub['prevalence'].max():.3f}, MCC {sub['mcc'].min():.3f}-{sub['mcc'].max():.3f}")
    grid.to_csv("data/tdi_track/outputs/comparisons/population_share_band.csv", index=False)


if __name__ == "__main__":
    main()
