# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Applicability domain of the 750 blinded compounds, generated from the final build.

A model is most reliable where it interpolates between compounds it has seen (Leach 2001, section 12.12.4). This module measures that for the final TDI stack of each isoform and writes the
result as files that the method report and the notebook read at build time; nothing in them is typed by hand.

Three steps, all from cached predictions and fingerprints (no model is refitted):

  1. The final stack of the isoform is rebuilt exactly as ``final_tdi_build`` does (same surviving groups, same ``choose_stack``), and for every repeat and outer fold of the nested plan the
     stack is fitted on the inner out-of-fold predictions and applied to the held-out outer rows, with the same similarity-matching and prevalence weights that the deployment uses.
  2. Every held-out row gets its nearest-training similarity: the highest ECFP4 Tanimoto similarity to the rows the outer fold was trained on. The held-out rows are cut into similarity bins,
     and each bin gets its AUC with a 95% bootstrap interval for the two confidence cuts z0 of the nested design. Compounds of one chemical series are not independent observations, so the
     bootstrap resamples whole Butina clusters (Tanimoto 0.6, the clusters of the series-level validation) and not single compounds; a compound-level bootstrap would give intervals that are too narrow
     (hierarchical data: Janczyk and Pfister, Understanding Inferential Statistics, section 11.4.3). Only confidently labelled rows count, as in the challenge's scoring.
  3. The 750 blinded compounds get their nearest-training similarity to the whole labelled pool. Each is placed in its bin and assigned a tier from the bin's measured reliability: ``supported``
     when the lower end of the bin's interval is at least ``SUPPORTED_AUC``, ``weak`` when the bin has enough compounds but a lower end below it, ``unsupported`` when the bin has fewer than
     ``MIN_BIN_ROWS`` held-out compounds. The share of the test set in every tier is the applicability-domain statement.

Absolute cross-validation numbers are not forecasts of the leaderboard (the official baseline's cross-validation score sits below its board score); the table is read by its shape across bins.

Usage: python -m cypmode.challenge.applicability_domain [--out-dir data/tdi_track/outputs/final]
Outputs: applicability_domain_bins.csv, applicability_domain_compounds.csv, applicability_domain.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cypmode.challenge import final_tdi_build as final

logger = logging.getLogger(__name__)
from cypmode.challenge.feature_store import FeatureStore
from cypmode.challenge.leaderboard_prior import estimate_from_csv
from cypmode.challenge.nested import (
    OUTER_REFIT,
    NestedPlan,
    PredictionCache,
    StackedLogit,
    _weights,
    inner_oof,
)
from cypmode.challenge.submission import load_test_set
from cypmode.challenge.validation_matching import (
    SimilarityMatcher,
    nearest_neighbor_similarity,
    tanimoto_matrix,
)

BIN_EDGES = np.array([0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01])
BIN_LABELS = ["< 0.3", "0.3 - 0.4", "0.4 - 0.5", "0.5 - 0.6", "0.6 - 0.7", ">= 0.7"]
SUPPORTED_AUC = 0.60  # the lower end of a bin's 95% interval must reach this for the bin to be called supported
MIN_BIN_ROWS = 100  # held-out compounds a bin needs before its AUC is trusted
N_BOOT = 200
SEED = 0


def bin_of(similarity: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(similarity, BIN_EDGES) - 1, 0, len(BIN_LABELS) - 1)


def final_stack(iso: str, gates: dict, store: FeatureStore):
    """The plan, cache, context, matcher, prevalence and candidate names of the final stack of ``iso``, rebuilt as ``final_tdi_build.build_isoform`` does."""
    root = Path(final.DESIGN["root"]) / iso
    live = final.survivors(gates, iso)
    by_group = {g: [c for c in final.GROUPS[g][0]() if c.name] for g in live}
    blocks = tuple(sorted({b for cs in by_group.values() for c in cs for b in c.blocks}))
    ctx, plan = final.build_context_and_plan(final.TRAIN_CSV, final.TEST_CSV, iso, final.STORE, final.REPEATS, final.DESIGN["outer"], 4, 0, 4, blocks or None)
    saved = NestedPlan.load(root / "plan.npz")
    assert saved.plan_id == plan.plan_id, "the plan changed: the report must share the experiment caches"
    cache = PredictionCache(root / "cache", saved)
    matcher = SimilarityMatcher(store.blocks["morgan_bits"], plan.external)
    prevalence = float(estimate_from_csv(final.BOARDS[iso])["prevalence"])
    defaults = [c.name for c in final.default_candidates() if not cache.missing(c.name)]
    groups = {g: [c.name for c in cs if not cache.missing(c.name)] for g, cs in by_group.items()}
    label, names, _ = final.choose_stack(cache, ctx, matcher, prevalence, defaults, {g: n for g, n in groups.items() if n}, (0, 0))
    return plan, cache, ctx, matcher, prevalence, label, names


def heldout_table(plan: NestedPlan, cache: PredictionCache, ctx, matcher, prevalence: float, names: list[str], bits: np.ndarray, clusters: dict[int, int] | None = None) -> pd.DataFrame:
    """One row per (repeat, outer fold, held-out compound): label, stack score, nearest-training similarity, the confidence z of the compound and its cluster."""
    frames = []
    for repeat, k in plan.outer_folds():
        rows, oof = inner_oof(cache, names, repeat, k)
        refit = plan.task(repeat, k, OUTER_REFIT)
        outer_scores = np.column_stack([cache.get(c, refit) for c in names])
        y_in, y_out = plan.labels(rows), plan.labels(refit.evaluate)
        chosen = StackedLogit().fit(names, oof, y_in, _weights(y_in, rows, refit.train, matcher, prevalence), "auc")
        frames.append(pd.DataFrame({
            "row": refit.evaluate, "repeat": repeat, "outer": k, "y": y_out, "score": chosen.predict(outer_scores),
            "nn": nearest_neighbor_similarity(bits[refit.evaluate], bits[refit.train]), "z": np.nan_to_num(ctx.confidence[refit.evaluate], nan=0.0),
            **({} if clusters is None else {"cluster": [clusters[int(r)] for r in refit.evaluate]}),
        }))
    return pd.concat(frames, ignore_index=True)


def cluster_ids(iso: str, plan: NestedPlan, bits: np.ndarray) -> dict[int, int]:
    """Butina cluster (Tanimoto 0.6) of every pool compound, read from the series-level plan of the isoform when it exists and computed otherwise (the same function builds both)."""
    saved = final.NESTED / f"{iso}_group0.6" / "plan.npz"
    if saved.exists():
        grouped = NestedPlan.load(saved)
        if np.array_equal(grouped.pool, plan.pool) and grouped.groups is not None:
            return dict(zip(grouped.pool.tolist(), grouped.groups.tolist(), strict=True))
    from cypmode.challenge.grouping import butina_groups

    return dict(zip(plan.pool.tolist(), butina_groups(bits[plan.pool], 0.6).tolist(), strict=True))


def auc_with_interval(frame: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float, float]:
    """AUC of a group of held-out predictions and a 95% bootstrap interval that resamples whole clusters (the column ``cluster``; one compound per cluster when it is absent), so that all
    repeats of a compound and all members of a series travel together."""
    if frame.y.nunique() < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(roc_auc_score(frame.y, frame.score))
    by_row = {r: g for r, g in frame.groupby("cluster" if "cluster" in frame else "row")}
    ids = np.array(list(by_row))
    values = []
    for _ in range(N_BOOT):
        pick = rng.choice(ids, size=len(ids), replace=True)
        sample = pd.concat([by_row[r] for r in pick])
        if sample.y.nunique() == 2:
            values.append(roc_auc_score(sample.y, sample.score))
    return (point, float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if values else (point, float("nan"), float("nan"))


def bin_table(held: pd.DataFrame, iso: str) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for z0 in final.Z_CUTS:
        scored = held[held.z >= z0].copy()
        scored["bin"] = bin_of(scored.nn.to_numpy())
        for b, label in enumerate(BIN_LABELS):
            part = scored[scored.bin == b]
            n_rows = int(part.row.nunique())
            point, low, high = auc_with_interval(part, rng) if n_rows >= 20 else (float("nan"),) * 3
            tier = "unsupported" if n_rows < MIN_BIN_ROWS else ("supported" if low >= SUPPORTED_AUC else "weak")
            rows.append({"isoform": iso, "z0": z0, "bin": label, "compounds": n_rows, "positive_share": float(part.drop_duplicates("row").y.mean()) if n_rows else float("nan"),
                         "auc": point, "auc_low": low, "auc_high": high, "tier": tier})
    return pd.DataFrame(rows)


def test_compounds(plan: NestedPlan, bits: np.ndarray, test: pd.DataFrame, iso: str, bins: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """The blinded compounds with their nearest-training similarity, the compound it is nearest to, its bin and the bin's tier (from the primary confidence cut)."""
    matrix = tanimoto_matrix(bits[plan.external], bits[plan.pool])
    similarity, nearest = matrix.max(axis=1), matrix.argmax(axis=1)
    train = pd.read_csv(final.TRAIN_CSV)
    tier_of = dict(zip(bins[bins.z0 == final.Z_CUTS[0]].bin, bins[bins.z0 == final.Z_CUTS[0]].tier, strict=True))
    scores = pd.read_csv(out_dir / f"{iso}_hits_only_scores.csv").set_index("Molecule_Name")
    names = test["Molecule_Name"].to_numpy()
    pool_names = train["Molecule_Name"].to_numpy()[plan.pool] if len(train) > int(plan.pool.max()) else np.array([""] * len(plan.pool))
    bin_label = np.array(BIN_LABELS)[bin_of(similarity)]
    return pd.DataFrame({
        "Molecule_Name": names, f"{iso}_nearest_training_similarity": similarity, f"{iso}_nearest_training_compound": pool_names[nearest], f"{iso}_similarity_bin": bin_label,
        f"{iso}_domain_tier": [tier_of[b] for b in bin_label], f"{iso}_committee_disagreement": scores.loc[names, "committee_disagreement"].to_numpy(), f"{iso}_call": scores.loc[names, "call"].to_numpy(),
    })


def markdown(bins: pd.DataFrame, compounds: pd.DataFrame, labels: dict[str, str]) -> str:
    lines = ["# Applicability domain of the blinded compounds (generated by `cypmode/challenge/applicability_domain.py`)", "",
             (
                 "The reliability of the final stack is measured on held-out outer folds of the nested validation, cut by the nearest-training similarity (ECFP4 Tanimoto) of each held-out compound to the "
                 f"rows its fold was trained on. The intervals resample whole chemical series (Butina clusters at Tanimoto 0.6). A bin is *supported* when the lower end of its 95% bootstrap interval is at least {SUPPORTED_AUC:.2f}, *weak* when it holds at least {MIN_BIN_ROWS} "
                 f"compounds but a lower end below that, and *unsupported* when it holds fewer than {MIN_BIN_ROWS}. Cross-validation numbers are read by their shape across bins; they are not forecasts of "
                 "the leaderboard."
             ), ""]
    for iso in sorted(bins.isoform.unique()):
        lines += [f"## {iso}", "", f"Final stack: {labels[iso]}.", ""]
        test_bins = compounds[f"{iso}_similarity_bin"].value_counts()
        for z0 in final.Z_CUTS:
            part = bins[(bins.isoform == iso) & (bins.z0 == z0)]
            lines += [f"Confidence cut z0 = {z0}:", "", "| similarity bin | held-out compounds | positive share | AUC | 95% interval | tier | blinded compounds in the bin |", "| --- | ---: | ---: | ---: | --- | --- | ---: |"]
            for r in part.itertuples():
                interval = "n/a" if np.isnan(r.auc_low) else f"{r.auc_low:.3f} to {r.auc_high:.3f}"
                auc = "n/a" if np.isnan(r.auc) else f"{r.auc:.3f}"
                lines.append(f"| {r.bin} | {r.compounds} | {r.positive_share:.2f} | {auc} | {interval} | {r.tier} | {int(test_bins.get(r.bin, 0))} |")
            lines.append("")
        tiers = compounds[f"{iso}_domain_tier"].value_counts()
        total = len(compounds)
        similarity = compounds[f"{iso}_nearest_training_similarity"]
        lines += [f"Blinded compounds by tier (primary cut z0 = {final.Z_CUTS[0]}): " + ", ".join(f"{tiers.get(t, 0)} {t} ({tiers.get(t, 0) / total:.0%})" for t in ("supported", "weak", "unsupported")) + ".",
                  (
                      f"Median nearest-training similarity of the blinded compounds: {similarity.median():.2f}; {int((similarity >= 0.5).sum())} of {total} ({(similarity >= 0.5).mean():.0%}) are at or above 0.5 and "
                      f"{int((similarity < 0.3).sum())} ({(similarity < 0.3).mean():.0%}) are below 0.3."
                  ), ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/tdi_track/outputs/final")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out_dir = Path(args.out_dir)
    train, test = pd.read_csv(final.TRAIN_CSV), load_test_set(final.TEST_CSV)
    store = FeatureStore.load_or_build(final.STORE, list(train["SMILES"]) + list(test["SMILES"]))
    bits = np.asarray(store.blocks["morgan_bits"], dtype=np.float32)
    gates = final.parse_gates()
    all_bins, merged, labels = [], test[["Molecule_Name"]].copy(), {}
    for iso in ("CYP3A4", "CYP2D6"):
        plan, cache, ctx, matcher, prevalence, label, names = final_stack(iso, gates, store)
        labels[iso] = f"{label} ({len(names)} candidates)"
        held = heldout_table(plan, cache, ctx, matcher, prevalence, names, bits, cluster_ids(iso, plan, bits))
        bins = bin_table(held, iso)
        all_bins.append(bins)
        merged = merged.merge(test_compounds(plan, bits, test, iso, bins, out_dir), on="Molecule_Name")
        logger.info("%s: %d held-out rows, %d bins", iso, len(held), len(bins))
    bins = pd.concat(all_bins, ignore_index=True)
    bins.to_csv(out_dir / "applicability_domain_bins.csv", index=False)
    merged.to_csv(out_dir / "applicability_domain_compounds.csv", index=False)
    (out_dir / "applicability_domain.md").write_text(markdown(bins, merged, labels))
    print((out_dir / "applicability_domain.md").read_text())


if __name__ == "__main__":
    main()
