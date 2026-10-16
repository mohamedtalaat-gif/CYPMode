"""Query-by-committee uncertainty, judged against real held-out errors -- and
against the simplest possible baseline.

Query by committee measures uncertainty as DISAGREEMENT inside a diverse
ensemble of distinct models, as opposed to the stochastic passes of one model
(MC dropout) or the seed-to-seed spread of one algorithm. The nested harness
(cypmode.challenge.nested) already holds the out-of-fold predictions of 25+
heterogeneous candidates, so a committee costs nothing extra to build.

Whether it is *useful* is an empirical question, and the bar is not "is it
correlated with error" but "does it tell us anything the distance of the
stack's score to its own decision threshold (the margin) does not". Every
component is therefore evaluated on compounds no candidate or selector ever
saw (the outer-validation rows of every nested fold) with three questions:

1. does it flag the stack's mistakes better than chance (AUROC, bootstrap CI)?
2. does it add information beyond the margin (paired bootstrap of the gain in
   a cross-validated error model's AUROC)?
3. does acting on it improve the decisions (fall back to the majority class
   on the most uncertain compounds and re-score MCC)?

``verdict`` turns those answers into one of ``adds_information`` /
``redundant_with_margin`` / ``no_signal``. The outcome, not a design choice,
decides how a component is described downstream: it is kept in the
reliability table either way, but a component that is not ``adds_information``
is documented as a measured negative result and is never used to alter a
prediction.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from cypmode.challenge.nested import (
    DEPLOY,
    OUTER_REFIT,
    NestedPlan,
    PredictionCache,
    Selector,
    StackedLogit,
    _EmpiricalCdf,
    call_top_fraction,
    inner_oof,
)
from cypmode.challenge.validation_matching import (
    nearest_neighbor_similarity,
    weighted_mcc,
)

BASELINE = "boundary_margin"
CONTROLS = ("boundary_margin", "novelty")
COMMITTEE_SIGNALS = ("qbc_vote_entropy", "qbc_spread", "qbc_vote_entropy_top5", "qbc_spread_family")


def vote_entropy(votes: np.ndarray) -> np.ndarray:
    """Binary entropy of the committee's vote share per compound (rows = compounds, columns = members, 0/1)."""
    share = np.clip(np.asarray(votes, dtype=float).mean(axis=1), 1e-9, 1 - 1e-9)
    return -(share * np.log(share) + (1 - share) * np.log(1 - share))


def committee_spread(scores: np.ndarray) -> np.ndarray:
    """Standard deviation across members of scores already on a common [0, 1] scale."""
    return np.asarray(scores, dtype=float).std(axis=1)


def _family(name: str) -> str:
    return name.split("_")[0]


def member_signals(oof: np.ndarray, y_in: np.ndarray, outer: np.ndarray, names: Sequence[str], rate: float, family: str = "lgbm", top_k: int = 5) -> dict[str, np.ndarray]:
    """Committee signals for ``outer`` (rows x members matrix of raw scores). Members are put on a
    common scale with their own inner out-of-fold empirical CDFs; each member votes the top
    ``rate`` fraction of the compounds in front of it positive.
    """
    cdfs = [_EmpiricalCdf(oof[:, j]) for j in range(oof.shape[1])]
    unit = np.column_stack([cdfs[j](outer[:, j]) for j in range(outer.shape[1])])
    votes = np.column_stack([call_top_fraction(outer[:, j], rate) for j in range(outer.shape[1])]).astype(float)
    inner_auc = np.array([roc_auc_score(y_in, oof[:, j]) for j in range(oof.shape[1])])
    best = np.argsort(-inner_auc)[:top_k]
    same_family = [j for j, n in enumerate(names) if _family(n) == family] or list(range(len(names)))
    return {
        "qbc_vote_entropy": vote_entropy(votes),
        "qbc_spread": committee_spread(unit),
        "qbc_vote_entropy_top5": vote_entropy(votes[:, best]),
        "qbc_spread_family": committee_spread(unit[:, same_family]),
    }


def collect_signals(cache: PredictionCache, names: Sequence[str], fingerprints: np.ndarray, selector: Selector | None = None) -> pd.DataFrame:
    """One row per outer-validation compound across all nested folds: its label, the stack's call and
    whether that call is wrong, the two controls and every committee signal.
    """
    plan, frames = cache.plan, []
    selector = selector or StackedLogit()
    for repeat, k in plan.outer_folds():
        rows_in, oof = inner_oof(cache, names, repeat, k)
        refit = plan.task(repeat, k, OUTER_REFIT)
        y_in, y_out = plan.labels(rows_in), plan.labels(refit.evaluate)
        outer = np.column_stack([cache.get(n, refit) for n in names])
        chosen = selector.fit(list(names), oof, y_in, None, "auc")
        score = chosen.predict(outer)
        call = call_top_fraction(score, chosen.rate())
        frame = pd.DataFrame(
            {
                "fold": f"{repeat}-{k}",
                "row": refit.evaluate,
                "y": y_out,
                "call": call.astype(int),
                "wrong": (call != y_out).astype(int),
                "stack_score": score,
                BASELINE: -np.abs(score - chosen.threshold),
                "novelty": 1.0 - nearest_neighbor_similarity(fingerprints[refit.evaluate], fingerprints[refit.train]),
            }
        )
        for name, values in member_signals(oof, y_in, outer, names, chosen.rate()).items():
            frame[name] = values
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _bootstrap(values_fn, n: int, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        draws.append(values_fn(rng.integers(0, n, n)))
    draws = [d for d in draws if np.isfinite(d)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _error_model(frame: pd.DataFrame, columns: Sequence[str], seed: int = 0) -> np.ndarray:
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(LogisticRegression(max_iter=500), frame[list(columns)], frame["wrong"], cv=folds, method="predict_proba")[:, 1]


def evaluate_components(
    frame: pd.DataFrame,
    components: Sequence[str] = (*CONTROLS, *COMMITTEE_SIGNALS),
    n_boot: int = 300,
    n_perm: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Per component: AUROC for "the stack is wrong" with a bootstrap CI, Spearman with the error,
    the gain in error-detection AUROC when added to the margin and a verdict.

    The gain is judged against its own null: the component's values are permuted across compounds
    (keeping the margin), the cross-validated error model is refit and the gain recomputed
    ``n_perm`` times; ``gain_p`` is the share of null gains at least as large as the observed one.
    Bootstrapping fixed predictions does not do this -- it produced a "discovery" from pure-noise
    committees in the unit tests. ``alpha`` is Bonferroni-corrected over the tested components.
    """
    wrong = frame["wrong"].to_numpy()
    baseline_prediction = _error_model(frame, [BASELINE], seed)
    baseline_auc = roc_auc_score(wrong, baseline_prediction)
    tested = [c for c in components if c != BASELINE]
    corrected = alpha / max(len(tested), 1)
    rng = np.random.default_rng(seed)
    rows = []
    for component in components:
        values = frame[component].to_numpy()
        auc = roc_auc_score(wrong, values)
        low, high = _bootstrap(lambda i, v=values: roc_auc_score(wrong[i], v[i]), len(frame), n_boot, seed)
        gain = gain_p = np.nan
        if component != BASELINE:
            gain = roc_auc_score(wrong, _error_model(frame, [BASELINE, component], seed)) - baseline_auc
            null = []
            for _ in range(n_perm):
                shuffled = frame[[BASELINE]].assign(**{component: rng.permutation(values)})
                shuffled["wrong"] = wrong
                null.append(roc_auc_score(wrong, _error_model(shuffled, [BASELINE, component], seed)) - baseline_auc)
            gain_p = (1 + sum(g >= gain for g in null)) / (1 + n_perm)
        rows.append(
            {
                "component": component,
                "auroc": auc,
                "auroc_lo": low,
                "auroc_hi": high,
                "spearman_with_error": float(spearmanr(values, wrong)[0]),
                "gain_over_margin": gain,
                "gain_p": gain_p,
                "verdict": _verdict(component, low, gain, gain_p, corrected),
            }
        )
    return pd.DataFrame(rows)


def _verdict(component: str, auroc_lo: float, gain: float, gain_p: float, alpha: float = 0.05) -> str:
    if component == BASELINE:
        return "baseline"
    if np.isfinite(gain_p) and gain > 0 and gain_p < alpha:
        return "adds_information"
    return "redundant_with_margin" if auroc_lo > 0.5 else "no_signal"


def selective_fallback(frame: pd.DataFrame, signal: str, fractions: Sequence[float] = (0.05, 0.1, 0.2, 0.3)) -> pd.DataFrame:
    """MCC over ALL compounds when the most uncertain fraction falls back to a negative call, against
    no fallback. "Most uncertain" follows the direction the data show (the sign of the Spearman
    correlation with the error): a signal that is high where the stack is RIGHT is flipped, because
    acting on it the other way round would be acting on noise.
    """
    direction = 1.0 if spearmanr(frame[signal], frame["wrong"])[0] >= 0 else -1.0
    values = direction * frame[signal].to_numpy()
    rows = [{"signal": signal, "direction": direction, "fallback_fraction": 0.0, "mcc": weighted_mcc(frame["y"], frame["call"])}]
    for fraction in fractions:
        call = np.where(values >= np.quantile(values, 1 - fraction), 0, frame["call"])
        rows.append({"signal": signal, "direction": direction, "fallback_fraction": fraction, "mcc": weighted_mcc(frame["y"], call)})
    return pd.DataFrame(rows)


def deploy_signals(cache: PredictionCache, names: Sequence[str], fingerprints: np.ndarray, selector: Selector | None = None) -> pd.DataFrame:
    """The same signals for the external (blinded test) compounds, from the whole-pool inner
    out-of-fold predictions -- informational only: a boolean submission cannot carry them.
    """
    plan = cache.plan
    rows_in, oof = inner_oof(cache, names, -1, 0)
    task = plan.task(-1, 0, DEPLOY)
    y_in = plan.labels(rows_in)
    outer = np.column_stack([cache.get(n, task) for n in names])
    chosen = (selector or StackedLogit()).fit(list(names), oof, y_in, None, "auc")
    score = chosen.predict(outer)
    frame = pd.DataFrame({"row": task.evaluate, "stack_score": score, "call": call_top_fraction(score, chosen.rate()).astype(int), BASELINE: -np.abs(score - chosen.threshold)})
    frame["novelty"] = 1.0 - nearest_neighbor_similarity(fingerprints[task.evaluate], fingerprints[task.train])
    for name, values in member_signals(oof, y_in, outer, names, chosen.rate()).items():
        frame[name] = values
    return frame


MIN_MCC_GAIN = 0.01


def decision_record(isoform: str, evaluation: pd.DataFrame, fallback: pd.DataFrame) -> dict:
    """What the evidence says about each component, in a form the Method Report can quote.

    A component may alter a prediction only if BOTH hold: it adds information beyond the margin
    (verdict ``adds_information``) AND acting on it improves MCC by at least ``MIN_MCC_GAIN`` in the
    selective-fallback check. Detecting errors is not the same as being able to fix them.
    """
    by_signal = fallback.groupby("signal")["mcc"].agg(["first", "max"])
    improves = {signal: bool(row["max"] > row["first"] + MIN_MCC_GAIN) for signal, row in by_signal.iterrows()}
    components = {}
    for r in evaluation.itertuples():
        components[r.component] = {
            "auroc": round(r.auroc, 3),
            "gain_over_margin": None if not np.isfinite(r.gain_over_margin) else round(r.gain_over_margin, 3),
            "gain_p": None if not np.isfinite(r.gain_p) else round(r.gain_p, 4),
            "verdict": r.verdict,
            "acting_on_it_improves_mcc": improves.get(r.component),
            "used_to_alter_predictions": bool(r.verdict == "adds_information" and improves.get(r.component, False)),
        }
    return {
        "isoform": isoform,
        "components": components,
        "policy": "a component alters a prediction only if it adds information beyond the margin AND acting on it improves MCC; "
        "every other component is reported as a measured negative result and never used to alter a prediction",
    }


def _main() -> None:
    from cypmode.challenge.feature_store import FeatureStore
    from cypmode.challenge.nested_runners import default_candidates

    parser = argparse.ArgumentParser()
    parser.add_argument("--isoform", required=True)
    parser.add_argument("--root", default="data/tdi_track/outputs/nested")
    parser.add_argument("--out-dir", default="data/tdi_track/outputs/comparisons")
    args = parser.parse_args()

    root = Path(args.root) / args.isoform
    plan = NestedPlan.load(root / "plan.npz")
    cache = PredictionCache(root / "cache", plan)
    train = pd.read_csv("data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv")
    test = pd.read_csv("data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv")
    bits = FeatureStore.load_or_build("data/tdi_track/outputs/caches/feature_store.npz", list(train["SMILES"]) + list(test["SMILES"])).blocks["morgan_bits"]
    order = [c.name for c in default_candidates()]
    names = [n for n in order if not cache.missing(n)] + [n for n in cache.candidates() if n not in order and not cache.missing(n)]

    frame = collect_signals(cache, names, bits)
    evaluation = evaluate_components(frame)
    fallback = pd.concat([selective_fallback(frame, s) for s in ("qbc_vote_entropy", "qbc_spread", "qbc_vote_entropy_top5", "qbc_spread_family", BASELINE)], ignore_index=True)
    out = Path(args.out_dir)
    evaluation.to_csv(out / f"committee_uncertainty_{args.isoform}.csv", index=False)
    fallback.to_csv(out / f"committee_uncertainty_fallback_{args.isoform}.csv", index=False)
    record = decision_record(args.isoform, evaluation, fallback)
    (out / f"committee_uncertainty_decision_{args.isoform}.json").write_text(json.dumps(record, indent=2) + "\n")
    deploy_signals(cache, names, bits).to_csv(root / "reliability_deploy.csv", index=False)
    print(f"{args.isoform}: {len(names)}-member committee, {len(frame)} held-out compounds, stack error {frame['wrong'].mean():.3f}")
    print(evaluation.round(3).to_string(index=False))
    print(fallback.round(3).to_string(index=False))
    print(json.dumps({k: v["used_to_alter_predictions"] for k, v in record["components"].items()}))


if __name__ == "__main__":
    _main()
