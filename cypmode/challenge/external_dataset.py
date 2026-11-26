# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The selection-corrected stack on a second, published data set: the direct-inhibition panel of Veith et al. (2009) in the benchmark of the Therapeutics Data Commons.

Why. Jorgensen (J. Chem. Inf. Model. 2006, 46, 937) asks that a new modelling procedure be validated on at least one other common data set of some size, for which a study with another method
has been published, and on compounds that are not in the training set. The time-dependent inhibition (TDI) data of the challenge have no such partner, so the procedure is run unchanged on the CYP3A4 and CYP2D6 inhibition
sets of the Therapeutics Data Commons (TDC; Huang et al., NeurIPS 2021 Datasets and Benchmarks), whose benchmark fixes a scaffold-split test set, a metric (the area under the precision-recall curve,
AUPRC, which is scikit-learn's average precision) and five train/validation splits, and whose public leaderboard lists the AUPRC of 19 published methods for each of the two sets.

What is run. The label-free candidates of the default grid (the two regressors of the assay arms and every set that reads a block trained on the TDC panel itself are left out, 16 candidates), the
stack of the final build (``StackedLogit``: a logistic blend of the inner out-of-fold scores, penalty and strength chosen on the inner folds) and nothing else. No part of the label process, the
selection-aware model, the confidence weights, or the call-rate multiple of the TDI track applies to a binary inhibition label, so this validates the generic part of the procedure (the features, the
candidates, the stack, and the nested protocol), not the label model. The label here is direct inhibition, not time-dependent inhibition.

Four analyses, all on structures and the labels of the training part only (the labels of validation and test compounds are never put in the context that the candidates read):

1. Official protocol. For each seed 1 to 5 of the benchmark the training part is the ``train`` split of ``train_val`` (scaffold split, fractions 0.875 and 0.125, as ``get_train_valid_split``), the
   inner folds keep every Bemis-Murcko scaffold whole, the stack is selected on the inner out-of-fold scores, is refitted on ``train``, and scores the ``valid`` split and the official ``test`` set. The
   result is the mean and the standard deviation (population form, as the leaderboard) of the test AUPRC over the five seeds, placed among the published entries.
2. Nested design of the project. The same candidates on 3 repeats of 5 outer and 4 inner folds over the whole ``train_val`` (scaffolds whole at both levels), and the deployment on the official test
   set. The nested estimate (selection inside every outer fold) is set against the score on the test set.
3. Negative control. The labels of ``train`` are permuted three times and the official protocol is run with seed 1; the test area under the receiver operating characteristic curve (AUC) must be 0.5.
4. Reliability. For the deployment on the official test set: the AUC in bins of the nearest-training similarity with intervals that resample whole Butina clusters (the bins and the interval of the
   applicability domain of the TDI track), and the calibration of isotonic probabilities fitted on the inner out-of-fold scores (the expected calibration error on the test set). The labels of that fit
   are taken in the row order of the inner out-of-fold scores, which is the order of the inner folds and not the order of the pool. In the order of the pool the isotonic map is almost constant (a slope
   of 74.8 and 51.3 on these two sets, and a Brier score equal to that of a constant prediction).

Rule, fixed before the first run (2026-11-26):

1. A placement is stated as it comes out: the mean test AUPRC of the five seeds is ABOVE THE MEDIAN when it is at least the median of the published entries read on 2026-11-26, and NOT ABOVE THE MEDIAN
   otherwise. No superiority over any entry is claimed whatever the rank.
2. The negative control is CHANCE when the mean permuted test AUC is within 0.04 of 0.5 (the tolerance of the control of the TDI track) and the real test AUC exceeds it by at least 0.1.
3. The nested estimate, the test score, the AUC by similarity bin and the calibration are reported and carry no verdict. Nothing is tuned after the first run and no verdict line matches the lines that
   the ledger of gates reads.

Usage: python -m cypmode.challenge.external_dataset [--n-jobs 8] [--isoform CYP3A4]
Outputs (data/tdi_track/outputs/external_dataset): official_protocol.csv, members_test.csv, nested_design.csv, permutation_control.csv, similarity_bins.csv, calibration.csv, placement.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from cypmode.challenge.applicability_domain import BIN_LABELS, auc_with_interval, bin_of
from cypmode.challenge.calibration_experiment import calibration_metrics, fit_calibrator
from cypmode.challenge.feature_store import OPT_IN_BLOCKS, FeatureStore
from cypmode.challenge.grouping import butina_groups, scaffold_groups
from cypmode.challenge.nested import (
    DEPLOY,
    BestSingle,
    NestedPlan,
    PredictionCache,
    StackedLogit,
    deploy,
    inner_oof,
    nested_evaluate,
)
from cypmode.challenge.nested_runners import Candidate, Context, default_candidates, run
from cypmode.challenge.validation_matching import (
    nearest_neighbor_similarity,
    tanimoto_matrix,
    weighted_mcc,
)

logger = logging.getLogger(__name__)

BENCH = Path("data/tdi_track/inputs/tdc_cache/admet_group")
DATASETS = {"CYP3A4": "cyp3a4_veith", "CYP2D6": "cyp2d6_veith"}
LEADERBOARD = BENCH / "leaderboard_veith.csv"
CHECKSUMS = BENCH / "SHA256SUMS.json"
STORE_TEMPLATE = "data/tdi_track/outputs/caches/tdc_veith_{iso}_feature_store.npz"
OUT = Path("data/tdi_track/outputs/external_dataset")
SEEDS = (1, 2, 3, 4, 5)
SPLIT_FRACTIONS = [0.875, 0.125, 0.0]  # the benchmark's own train/validation fractions (``get_train_valid_split``)
INNER = 4
NESTED_REPEATS, NESTED_OUTER, NESTED_SEED = 3, 5, 0
PERMUTATIONS = 3
CHANCE_TOLERANCE, CHANCE_GAP = 0.04, 0.1
CLUSTER_SIMILARITY = 0.6
MIN_BIN_ROWS = 100


def external_candidates() -> list[Candidate]:
    """The default grid without the candidates that read the assay arms (a binary label has none) and without every candidate that reads an opt-in block
    (``ext_activity`` is itself trained on the TDC panel, so it would carry the evaluation labels into the features)."""
    return [c for c in default_candidates() if c.family not in ("lgbm_reg_shift", "lgbm_reg_arms") and not set(c.blocks) & set(OPT_IN_BLOCKS)]


# -- data ---------------------------------------------------------------------------------------


def verify_files() -> None:
    """The four benchmark files are the ones whose SHA-256 is stored beside them."""
    expected = json.loads(CHECKSUMS.read_text())
    for name, digest in expected.items():
        actual = hashlib.sha256((BENCH / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"{name} differs from the checksum stored in {CHECKSUMS}")


def load_benchmark(iso: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = BENCH / DATASETS[iso]
    train_val, test = pd.read_csv(folder / "train_val.csv"), pd.read_csv(folder / "test.csv")
    both = pd.concat([train_val.Drug, test.Drug])
    if both.duplicated().any():
        raise ValueError(f"{iso}: a SMILES string occurs twice in the benchmark files")
    return train_val, test


def official_split(train_val: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The train and validation parts of ``train_val`` for a seed: the function and the fractions of ``BenchmarkGroup.get_train_valid_split`` for a scaffold-split benchmark."""
    from tdc.utils import create_scaffold_split

    out = create_scaffold_split(train_val, seed, frac=SPLIT_FRACTIONS, entity="Drug")
    return out["train"], out["valid"]


class Benchmark:
    """One isoform of the benchmark with its features: row ``i`` is compound ``i`` of ``train_val`` followed by the test compounds."""

    def __init__(self, iso: str, n_jobs: int = 4, store_path: str | Path | None = None, limit: int | None = None):
        train_val, test = load_benchmark(iso)
        if limit:
            train_val, test = train_val.iloc[:limit], test.iloc[: max(limit // 4, 20)]
        self.iso, self.train_val, self.test = iso, train_val.reset_index(drop=True), test.reset_index(drop=True)
        self.smiles = list(self.train_val.Drug) + list(self.test.Drug)
        self.row_of = {s: i for i, s in enumerate(self.smiles)}
        self.n_train_val = len(self.train_val)
        self.y = np.full(len(self.smiles), -1, dtype=np.int64)
        self.y[: self.n_train_val] = self.train_val.Y.astype(int).to_numpy()
        self.y_test = self.test.Y.astype(int).to_numpy()
        blocks = tuple(sorted({b for c in external_candidates() for b in c.blocks} | {"morgan_bits"}))
        path = store_path if store_path is not None else STORE_TEMPLATE.format(iso=iso.lower())
        self.store = FeatureStore.load_or_build(path, self.smiles, blocks) if path else FeatureStore.build(self.smiles, blocks)
        self.bits = self.store.blocks["morgan_bits"]
        self.similarity = tanimoto_matrix(self.bits)
        self.scaffold = scaffold_groups(self.smiles)
        self.n_jobs = n_jobs
        self.test_rows = np.arange(self.n_train_val, len(self.smiles))

    def rows_of(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([self.row_of[s] for s in frame.Drug], dtype=np.int64)

    def context(self, pool_rows: np.ndarray, labels: np.ndarray | None = None) -> Context:
        """A context whose label vector holds the labels of the pool rows only: validation and test labels are not in it, so no candidate can read them."""
        y = np.full(len(self.smiles), -1, dtype=np.int64)
        y[pool_rows] = (self.y if labels is None else labels)[pool_rows]
        return Context(self.store, y, {}, self.similarity, self.n_jobs)


# -- metrics ------------------------------------------------------------------------------------


def score_all(y: np.ndarray, score: np.ndarray, calls: np.ndarray | None = None) -> dict[str, float]:
    out = {"auprc": float(average_precision_score(y, score)), "auc": float(roc_auc_score(y, score)), "prevalence": float(np.mean(y))}
    if calls is not None:
        out["mcc"] = weighted_mcc(y, calls.astype(int))
    return out


def run_plan(bench: Benchmark, plan: NestedPlan, candidates: list[Candidate], root: str | Path, pool_rows: np.ndarray, labels: np.ndarray | None = None) -> PredictionCache:
    cache = PredictionCache(root, plan)
    run(candidates, bench.context(pool_rows, labels), plan, cache)
    return cache


def official_plan(bench: Benchmark, seed: int):
    """(plan, train rows, validation rows): the pool is the ``train`` split of the seed, the external rows are the validation split followed by the official test set."""
    train, valid = official_split(bench.train_val, seed)
    train_rows, valid_rows = bench.rows_of(train), bench.rows_of(valid)
    plan = NestedPlan(train_rows, bench.y[train_rows], 0, NESTED_OUTER, INNER, seed, external=np.concatenate([valid_rows, bench.test_rows]), groups=bench.scaffold[train_rows])
    return plan, train_rows, valid_rows


def official_protocol(bench: Benchmark, candidates: list[Candidate], work: Path) -> tuple[list[dict], list[dict]]:
    names = [c.name for c in candidates]
    rows, members = [], []
    for seed in SEEDS:
        plan, train_rows, valid_rows = official_plan(bench, seed)
        cache = run_plan(bench, plan, candidates, work / f"official_seed{seed}", train_rows)
        n_valid = len(valid_rows)
        for label, selector in (("stack", StackedLogit()), ("best_single", BestSingle())):
            dep = deploy(cache, names, selector)
            calls = dep.calls
            for part, sl, y in (("valid", slice(0, n_valid), bench.y[valid_rows]), ("test", slice(n_valid, None), bench.y_test)):
                rows.append({"isoform": bench.iso, "seed": seed, "selector": label, "part": part, "n": len(y), **score_all(y, dep.scores[sl], calls[sl]), "inner_score": dep.selection.inner_score})
        refit = plan.task(-1, 0, DEPLOY)
        for name in names:
            score = cache.get(name, refit)[n_valid:]
            members.append({"isoform": bench.iso, "seed": seed, "candidate": name, **score_all(bench.y_test, score)})
        logger.info("%s seed %d done", bench.iso, seed)
    return rows, members


def nested_design(bench: Benchmark, candidates: list[Candidate], work: Path) -> tuple[pd.DataFrame, dict]:
    """Repeats of outer folds over the whole ``train_val`` with the selection inside every fold, and the deployment on the official test set."""
    pool = np.arange(bench.n_train_val)
    plan = NestedPlan(pool, bench.y[pool], NESTED_REPEATS, NESTED_OUTER, INNER, NESTED_SEED, external=bench.test_rows, groups=bench.scaffold[pool])
    cache = run_plan(bench, plan, candidates, work / "nested", pool)
    names = [c.name for c in candidates]
    frames = []
    for metric in ("auc", "ap"):
        table = nested_evaluate(cache, names, [StackedLogit(), BestSingle()], None, None, metric)
        table["metric"] = metric
        frames.append(table)
    dep = deploy(cache, names, StackedLogit())
    return pd.concat(frames, ignore_index=True), {"plan": plan, "cache": cache, "deployment": dep, "names": names}


def permutation_control(bench: Benchmark, candidates: list[Candidate]) -> list[dict]:
    plan, train_rows, valid_rows = official_plan(bench, SEEDS[0])
    names = [c.name for c in candidates]
    out = []
    for k in range(PERMUTATIONS):
        permuted = bench.y.copy()
        permuted[train_rows] = np.random.default_rng(1000 + k).permutation(bench.y[train_rows])
        shuffled_plan = NestedPlan(train_rows, permuted[train_rows], 0, NESTED_OUTER, INNER, SEEDS[0], external=plan.external, groups=bench.scaffold[train_rows])
        with tempfile.TemporaryDirectory() as tmp:
            cache = run_plan(bench, shuffled_plan, candidates, tmp, train_rows, permuted)
            dep = deploy(cache, names, StackedLogit())
        test_scores = dep.scores[len(valid_rows):]
        out.append({"isoform": bench.iso, "permutation": k, **score_all(bench.y_test, test_scores, dep.calls[len(valid_rows):])})
    return out


def similarity_bins(bench: Benchmark, run_state: dict) -> pd.DataFrame:
    """AUC of the deployed stack on the official test compounds by nearest-training similarity, with the interval of the applicability domain of the TDI track (whole Butina clusters resampled)."""
    plan, dep = run_state["plan"], run_state["deployment"]
    nn = nearest_neighbor_similarity(bench.bits[bench.test_rows], bench.bits[plan.pool])
    cluster = butina_groups(bench.bits[bench.test_rows], CLUSTER_SIMILARITY)
    frame = pd.DataFrame({"row": bench.test_rows, "y": bench.y_test, "score": dep.scores, "nn": nn, "cluster": cluster})
    frame["bin"] = bin_of(frame.nn.to_numpy())
    rng = np.random.default_rng(0)
    rows = []
    for b, label in enumerate(BIN_LABELS):
        part = frame[frame.bin == b]
        point, low, high = auc_with_interval(part, rng) if len(part) >= 20 else (float("nan"),) * 3
        rows.append({"isoform": bench.iso, "bin": label, "compounds": len(part), "positive_share": float(part.y.mean()) if len(part) else float("nan"),
                     "auc": point, "auc_low": low, "auc_high": high, "enough_compounds": len(part) >= MIN_BIN_ROWS})
    rows.append({"isoform": bench.iso, "bin": "all", "compounds": len(frame), "positive_share": float(frame.y.mean()), "auc": float(roc_auc_score(frame.y, frame.score)),
                 "auc_low": float("nan"), "auc_high": float("nan"), "enough_compounds": True})
    return pd.DataFrame(rows)


def inner_labels(plan: NestedPlan, cache: PredictionCache, names: list[str]) -> np.ndarray:
    """The labels of the rows of the inner out-of-fold scores of the deployment, in the order of those scores (the rows of ``inner_oof``: inner fold after inner fold, not the order of the pool)."""
    rows, _ = inner_oof(cache, names, -1, 0)
    return plan.labels(rows)


def calibration_on_test(bench: Benchmark, run_state: dict) -> pd.DataFrame:
    """Isotonic probabilities fitted on the inner out-of-fold scores of the whole ``train_val`` and judged on the official test set (expected calibration error, slope, Brier score).

    The scores and the labels of the fit must describe the same rows: ``selection.inner_scores`` follow the inner folds, so the labels are read in that order by ``inner_labels``.
    """
    dep, plan = run_state["deployment"], run_state["plan"]
    y_inner = inner_labels(plan, run_state["cache"], run_state["names"])
    if len(y_inner) != len(dep.selection.inner_scores):
        raise AssertionError("the labels and the inner out-of-fold scores of the calibration are not of the same rows")
    rows = []
    for method in ("raw", "isotonic"):
        calibrate = fit_calibrator(method, dep.selection.inner_scores, y_inner, None)
        p = np.clip(calibrate(dep.scores), 1e-6, 1 - 1e-6)
        rows.append({"isoform": bench.iso, "method": method, **calibration_metrics(bench.y_test, p, np.ones(len(p)))})
    return pd.DataFrame(rows)


# -- published entries --------------------------------------------------------------------------


def placement(iso: str, mean_auprc: float) -> dict:
    board = pd.read_csv(LEADERBOARD)
    board = board[board.dataset == f"{iso}_Veith"]
    values = board.auprc_mean.to_numpy()
    median = float(np.median(values))
    return {"isoform": iso, "entries": len(values), "median_published": median, "best_published": float(values.max()), "mean_test_auprc": mean_auprc,
            "rank_among_entries": int(1 + np.sum(values > mean_auprc)), "verdict": "ABOVE THE MEDIAN" if mean_auprc >= median else "NOT ABOVE THE MEDIAN"}


# -- driver -------------------------------------------------------------------------------------


def run_isoform(iso: str, n_jobs: int, work: Path, out: Path) -> dict[str, pd.DataFrame]:
    bench = Benchmark(iso, n_jobs)
    candidates = external_candidates()
    print(f"\n== {iso}: {bench.n_train_val} compounds in train_val ({bench.y[:bench.n_train_val].mean():.3f} positive), {len(bench.test)} in the official test set ({bench.y_test.mean():.3f} positive), {len(candidates)} candidates")
    official, members = official_protocol(bench, candidates, work / iso)
    nested, state = nested_design(bench, candidates, work / iso)
    permuted = permutation_control(bench, candidates)
    bins, calibration = similarity_bins(bench, state), calibration_on_test(bench, state)
    dep = state["deployment"]
    deployed = {"isoform": iso, **score_all(bench.y_test, dep.scores, dep.calls), "inner_score": dep.selection.inner_score}
    return {"official": pd.DataFrame(official), "members": pd.DataFrame(members), "nested": nested.assign(isoform=iso), "permutation": pd.DataFrame(permuted), "bins": bins, "calibration": calibration, "deployed": pd.DataFrame([deployed])}


def report(results: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    placements = []
    for iso, r in results.items():
        official = r["official"]
        stack_test = official[(official.selector == "stack") & (official.part == "test")]
        stack_valid = official[(official.selector == "stack") & (official.part == "valid")]
        single_test = official[(official.selector == "best_single") & (official.part == "test")]
        mean, sd = stack_test.auprc.mean(), stack_test.auprc.std(ddof=0)
        print(f"\n{iso}: official protocol, five seeds (population standard deviation, as the leaderboard)")
        print(f"  stack        test AUPRC {mean:.3f} +/- {sd:.3f}  AUC {stack_test.auc.mean():.3f}  MCC at the transferred call rate {stack_test.mcc.mean():+.3f}  | validation AUPRC {stack_valid.auprc.mean():.3f} AUC {stack_valid.auc.mean():.3f}")
        print(f"  best single  test AUPRC {single_test.auprc.mean():.3f} +/- {single_test.auprc.std(ddof=0):.3f}  AUC {single_test.auc.mean():.3f}")
        members = r["members"].groupby("candidate")[["auprc", "auc"]].mean().sort_values("auprc", ascending=False)
        print("  members on the test set (mean over seeds): " + ", ".join(f"{name} {row.auprc:.3f}" for name, row in members.head(5).iterrows()) + f" ... lowest {members.auprc.min():.3f} ({members.auprc.idxmin()})")
        place = placement(iso, float(mean))
        placements.append(place)
        print(f"  published entries: {place['entries']}, median {place['median_published']:.3f}, best {place['best_published']:.3f}; the stack would rank {place['rank_among_entries']} of {place['entries'] + 1}")
        print(f"CHECK {iso} placement among the published entries: {place['verdict']}")
        nested = r["nested"]
        for metric, name in (("auc", "AUC"), ("ap", "AUPRC")):
            part = nested[(nested.metric == metric) & (nested.selector == "stack_logit")]
            print(f"  nested design ({len(part)} outer folds, scaffolds whole): stack {name} outer {part.outer_score.mean():.3f} (inner {part.inner_score.mean():.3f}, optimism {part.optimism.mean():+.3f}), MCC at the transferred rate {part.outer_mcc_rate.mean():+.3f}")
        deployed = r["deployed"].iloc[0]
        nested_auc = nested[(nested.metric == "auc") & (nested.selector == "stack_logit")].outer_score.mean()
        nested_ap = nested[(nested.metric == "ap") & (nested.selector == "stack_logit")].outer_score.mean()
        print(f"  deployment on the official test set: AUC {deployed.auc:.3f}, AUPRC {deployed.auprc:.3f}; nested estimate minus test: AUC {nested_auc - deployed.auc:+.3f}, AUPRC {nested_ap - deployed.auprc:+.3f}")
        perm = r["permutation"]
        chance = abs(perm.auc.mean() - 0.5) <= CHANCE_TOLERANCE and stack_test[stack_test.seed == SEEDS[0]].auc.iloc[0] - perm.auc.mean() >= CHANCE_GAP
        print(f"  negative control, labels of train permuted {len(perm)} times (seed {SEEDS[0]}): test AUC {perm.auc.mean():.3f} (range {perm.auc.min():.3f} to {perm.auc.max():.3f}), test AUPRC {perm.auprc.mean():.3f} against a prevalence of {perm.prevalence.iloc[0]:.3f}")
        print(f"CHECK {iso} negative control: {'CHANCE' if chance else 'NOT CHANCE'}")
        print("  AUC of the deployed stack by nearest-training similarity on the official test set (compounds, AUC, 95% interval over whole clusters):")
        for row in r["bins"].itertuples():
            interval = "n/a" if np.isnan(row.auc_low) else f"{row.auc_low:.3f} to {row.auc_high:.3f}"
            auc = "n/a" if np.isnan(row.auc) else f"{row.auc:.3f}"
            print(f"    {row.bin:>10s}: {row.compounds:5d}  {auc}  {interval}")
        for row in r["calibration"].itertuples():
            print(f"  calibration on the test set, {row.method}: ECE {row.ece:.4f}, slope {row.slope:.3f}, Brier {row.brier:.4f}, mean predicted {row.mean_pred:.3f} against observed {row.mean_obs:.3f}")
    return pd.DataFrame(placements)


def main() -> None:
    from cypmode import provenance

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", choices=sorted(DATASETS), help="one isoform (both by default)")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    provenance.announce()
    verify_files()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    results = {iso: run_isoform(iso, args.n_jobs, work, out) for iso in ([args.isoform] if args.isoform else sorted(DATASETS))}
    placements = report(results)
    for key, name in (("official", "official_protocol"), ("members", "members_test"), ("nested", "nested_design"), ("permutation", "permutation_control"), ("bins", "similarity_bins"), ("calibration", "calibration")):
        pd.concat([r[key] for r in results.values()], ignore_index=True).to_csv(out / f"{name}.csv", index=False, float_format="%.6f")
    placements.to_csv(out / "placement.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
