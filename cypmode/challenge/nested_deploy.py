"""Turn the nested-CV prediction caches into a TDI submission candidate.

    python -m cypmode.challenge.nested_deploy --out data/tdi_track/outputs/nested/submission_candidate.csv

For each isoform it runs the PRE-REGISTERED selection procedure (an L1/L2-penalised logistic
stack of every complete candidate, nested.StackedLogit) on the whole pool's inner out-of-fold
predictions -- the same procedure nested_report evaluates -- applies the frozen choice to the
blinded test compounds, and calls a compound positive when its stack score reaches the
operating threshold chosen from those inner predictions under the similarity- and
prior-matched weighting. Writes the schema-checked submission plus per-isoform scores; it never
touches submissions/ and never uploads anything.
"""

import argparse
from pathlib import Path

import pandas as pd

from cypmode.challenge.feature_store import FeatureStore
from cypmode.challenge.leaderboard_prior import estimate_from_csv
from cypmode.challenge.nested import NestedPlan, PredictionCache, StackedLogit, deploy
from cypmode.challenge.nested_runners import default_candidates
from cypmode.challenge.submission import (
    build_submission,
    load_test_set,
    validate_submission,
)
from cypmode.challenge.validation_matching import SimilarityMatcher

BOARDS = {
    "CYP3A4": "data/tdi_track/inputs/leaderboard_snapshots/interim_CYP3A4_2026-10-04.csv",
    "CYP2D6": "data/tdi_track/inputs/leaderboard_snapshots/interim_CYP2D6_2026-10-04.csv",
}


def deploy_isoform(isoform: str, root: str | Path, store: FeatureStore, board: str) -> tuple[pd.DataFrame, dict]:
    root = Path(root) / isoform
    plan = NestedPlan.load(root / "plan.npz")
    cache = PredictionCache(root / "cache", plan)
    order = [c.name for c in default_candidates()]
    names = [n for n in order if not cache.missing(n)] + [n for n in cache.candidates() if n not in order and not cache.missing(n)]
    prior = estimate_from_csv(board)
    result = deploy(cache, names, StackedLogit(), SimilarityMatcher(store.blocks["morgan_bits"], plan.external), prior["prevalence"])
    table = pd.DataFrame({"row": result.rows, "score": result.scores, "call": result.calls})
    info = {
        "isoform": isoform,
        "n_candidates": len(names),
        "target_prevalence": prior["prevalence"],
        "predicted_positive_rate": float(result.calls.mean()),
        "threshold": result.threshold,
        "stack_weights": {k: round(v, 3) for k, v in result.selection.chosen.items()},
    }
    return table, info


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/tdi_track/outputs/nested")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    train = pd.read_csv("data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv")
    test = load_test_set()
    store = FeatureStore.load_or_build("data/tdi_track/outputs/caches/feature_store.npz", list(train["SMILES"]) + list(test["SMILES"]))
    calls = {}
    for isoform, board in BOARDS.items():
        table, info = deploy_isoform(isoform, args.root, store, board)
        table["row"] -= len(train)  # external row ids index the concatenated train+test list
        assert (table["row"].to_numpy() == range(len(test))).all(), "external rows must be the test set in file order"
        calls[isoform] = table["call"].to_numpy()
        table.to_csv(Path(args.root) / isoform / "deploy_scores.csv", index=False)
        print(info)
    submission = build_submission(test, calls["CYP2D6"], calls["CYP3A4"])
    ok, errors = validate_submission(submission, set(test["Molecule_Name"]))
    assert ok, errors  # the tuple was ignored before, so an invalid file could have been written silently
    submission.to_csv(args.out, index=False)
    print(f"wrote {args.out}: CYP3A4 {submission.CYP3A4_is_TDI.mean():.3f} positive, CYP2D6 {submission.CYP2D6_is_TDI.mean():.3f} positive")


if __name__ == "__main__":
    _main()
