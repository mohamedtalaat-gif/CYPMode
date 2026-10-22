# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The all-in-one comparison: the final stack against one stack that holds every candidate that was ever cached (declared 2026-10-22, before any result).

The final stack of each isoform is the default stack plus the groups whose registered gate is MET (``final_tdi_build.choose_stack``). A gate asks that an idea helps on its own and inside the default stack, so an
idea that is weak alone and useful only together with others could be refused by every gate and still help a stack that holds all of them. This module measures that, on the plan of the experiments (3 repeats of
5 outer folds), with every choice inside the loop:

  * the final stack: the default candidates plus the candidates of the groups that passed (``final_tdi_build.parse_gates``);
  * the all-in-one stack: the default candidates plus every candidate of every group of the final build whose predictions are complete on the cross-validation tasks of the plan (the deployment fits are not needed for the comparison), whatever the verdict of its gate;
  * the comparison: the nested evaluation of both, one logistic blend each, penalty chosen on the inner folds (``StackedLogit``), at the two confidence cuts z0 = 1.0 and 1.96, with the paired corrected resampled t of
    Nadeau and Bengio (``model_comparison.compare_candidate_sets``).

Registered outcome: the all-in-one stack is better when its mean outer AUC exceeds the final stack's with corrected p < 0.05 at both cuts and by at least the least difference the design recognises
(``FLOOR``, 0.0099). That is reported either way. It does not replace the final stack by itself: there is no replicate (the groups have no predictions on the 10-outer-fold plan), so a replacement is the owner's decision,
taken with the numbers in front of it. The log is written under a name that no gate parser reads (``exploratory_all_in_one_<isoform>.log``) and is listed in the ledger under that name.

Usage: PYTHONPATH=. python -m cypmode.challenge.all_in_one --isoform CYP3A4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from cypmode.challenge import final_tdi_build as build
from cypmode.challenge.feature_store import FeatureStore
from cypmode.challenge.label_confidence import confident_weights
from cypmode.challenge.leaderboard_prior import estimate_from_csv
from cypmode.challenge.model_comparison import compare_candidate_sets
from cypmode.challenge.nested import (
    NestedPlan,
    PredictionCache,
    StackedLogit,
    nested_evaluate,
)
from cypmode.challenge.nested_runners import build_context_and_plan, default_candidates
from cypmode.challenge.submission import load_test_set
from cypmode.challenge.validation_matching import SimilarityMatcher

FLOOR = 0.0099  # the least difference in AUC that the main design recognises 8 times in 10 (6.7)


def stack_names(default_names: list[str], groups: dict[str, list[str]], chosen: list[str] | None = None) -> list[str]:
    """The default candidates followed by the candidates of ``chosen`` groups (all groups when None), each name once and in the order of the groups."""
    names = list(default_names)
    for group, members in groups.items():
        if chosen is not None and group not in chosen:
            continue
        names += [m for m in members if m not in names]
    return names


def cv_complete(cache: PredictionCache, name: str) -> bool:
    """True when the candidate has a prediction for every cross-validation task of the plan. The deployment task (repeat -1) is not needed to compare two stacks on the outer folds, and the experiments cached cross-validation
    tasks only, so a candidate whose deployment fit was never made (every group that did not join a final stack) must not be left out for that reason."""
    return not [task for task in cache.missing(name) if task.repeat >= 0]


def verdict_line(isoform: str, rows: list[dict]) -> str:
    """The line of the log: MET only when both cuts show a gain with p < 0.05 and the smaller gain is at least FLOOR."""
    met = all(r["p"] < 0.05 and r["diff"] >= FLOOR for r in rows)
    return f"{isoform} ALL-IN-ONE (stack of every cached candidate against the final stack, corrected p < 0.05 at both z0 and a gain of at least {FLOOR}): {'MET' if met else 'NOT MET'}"


def run(isoform: str) -> list[str]:
    train, test = pd.read_csv(build.TRAIN_CSV), load_test_set(build.TEST_CSV)
    store = FeatureStore.load_or_build(build.STORE, list(train["SMILES"]) + list(test["SMILES"]))
    root = Path(build.NESTED) / isoform
    gates = build.parse_gates()
    passed = build.survivors(gates, isoform)
    candidates_by_group = {g: [c for c in builder() if c.name] for g, (builder, _) in build.GROUPS.items()}
    ctx, plan = build_context_and_plan(build.TRAIN_CSV, build.TEST_CSV, isoform, build.STORE, build.REPEATS, build.DESIGN["outer"], 4, 0, 4, None)
    saved = NestedPlan.load(root / "plan.npz")
    assert saved.plan_id == plan.plan_id, "the plan changed: the comparison must share the experiment caches"
    cache = PredictionCache(root / "cache", saved)
    matcher = SimilarityMatcher(store.blocks["morgan_bits"], plan.external)
    prevalence = float(estimate_from_csv(build.BOARDS[isoform])["prevalence"])
    default_names = [c.name for c in default_candidates() if cv_complete(cache, c.name)]
    groups = {g: [c.name for c in cs if cv_complete(cache, c.name)] for g, cs in candidates_by_group.items()}
    groups = {g: names for g, names in groups.items() if names}
    final_names = stack_names(default_names, groups, passed)
    everything = stack_names(default_names, groups)
    lines = [f"{isoform}: final stack of {len(final_names)} candidates (groups that passed: {', '.join(passed) or 'none'}); all-in-one stack of {len(everything)} candidates from {len(groups)} groups with complete predictions"]
    rows = []
    for z0 in build.Z_CUTS:
        weight_fn = confident_weights(matcher, ctx.confidence, z0)
        auc_final = float(nested_evaluate(cache, final_names, [StackedLogit()], weight_fn, prevalence, "auc")["outer_score"].mean())
        auc_all = float(nested_evaluate(cache, everything, [StackedLogit()], weight_fn, prevalence, "auc")["outer_score"].mean())
        comparison = compare_candidate_sets(cache, everything, final_names, StackedLogit(), weight_fn, prevalence, "auc")
        rows.append({"z0": z0, "diff": comparison.mean_difference, "p": comparison.p_value})
        lines.append(f"z0={z0}  outer AUC final {auc_final:.4f}  all-in-one {auc_all:.4f}  all-in-one vs final: AUC {comparison.mean_difference:+.4f} (p={comparison.p_value:.3f})")
    lines.append(verdict_line(isoform, rows))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", required=True, choices=["CYP3A4", "CYP2D6"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print("\n".join(run(args.isoform)))
    return 0


if __name__ == "__main__":
    from cypmode import provenance

    provenance.announce()
    raise SystemExit(main())
