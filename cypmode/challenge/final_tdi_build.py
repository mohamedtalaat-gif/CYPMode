"""Gate-driven final TDI build: from the experiment verdicts to validated candidate submission files.

Nothing here decides by hand which model joins the final stack. Every experiment group (physics block, SAR pair
model, matched-pair transformations, frozen embeddings, test-likeness weighting, floor-clipped arms, label-process
committee) prints a pre-registered line such as ``CYP3A4 GATE (stacked, corrected p < 0.05 at both z0): MET`` into its
log. This module parses those lines and applies one rule per isoform:

  1. candidate sets = the default stack, the default stack + each group whose STACKED gate is MET, and the default stack
     + all surviving groups together (a gain alone is not enough: the combination is evaluated as one stack);
  2. each set is evaluated by the same nested procedure as everywhere else (similarity x prevalence x confident rows,
     z0 in {1.0, 1.96}); a set is eligible if its outer AUC beats the default stack with corrected p < 0.05 at both z0;
  3. the eligible set with the highest mean AUC over the two z0 wins (ties: fewer candidates); with no eligible set the
     default stack is the final model.

Call rate: the number of positive calls is a multiple of the scored population's prevalence in {1.0, 1.1, 1.2, 1.3}, chosen
by the highest mean outer-fold weighted MCC over both z0 (the stack is refitted inside every outer fold, as in the
nested evaluation); the top-scoring compounds of the test file are called, so the rule transfers a RATE, never a score.

CYP2D6 gets two candidate files for the A/B plan: A = the hits-only stack above, B = the selection-aware stack on the panel
pool (fit weight chosen by ``hedged_blend``, default 1.0) when its machinery is cached. CYP3A4 is identical in both files.
The hedged rank blend is written as a third file only if its own gate (hedged_blend) was MET.

Outputs under --out-dir: one CSV per candidate (schema validated, reloaded and re-validated), per-isoform score tables with
isotonic-calibrated probabilities and a committee-disagreement column, ``manifest.json`` (file hashes, environment, seeds,
gate verdicts, nested numbers, call rates, leakage audit) and ``final_report.md``. It never uploads or submits anything.

Usage: python -m cypmode.challenge.final_tdi_build [--out-dir data/tdi_track/outputs/final] [--dry-run]
"""

import argparse
import hashlib
import json
import logging
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.challenge.aux_blocks import aux_candidates, gate_shift_candidates, nodirect_candidates, screen_selection_candidates
from cypmode.challenge.cached_candidates import GROUPS as CACHED_GROUPS
from cypmode.challenge.calibration_experiment import fit_calibrator
from cypmode.challenge.feature_store import FeatureStore
from cypmode.challenge.label_confidence import confident_weights
from cypmode.challenge.leaderboard_prior import estimate_from_csv
from cypmode.challenge.model_comparison import compare_candidate_sets
from cypmode.challenge.nested import (
    DEPLOY,
    OUTER_REFIT,
    NestedPlan,
    PredictionCache,
    StackedLogit,
    _weights,
    call_top_fraction,
    deploy,
    inner_oof,
    nested_evaluate,
)
from cypmode.challenge.nested_runners import (
    arm_candidates,
    basic_centre_candidates,
    ext_cont_candidates,
    atomjepa_candidates,
    ext_adme_candidates,
    ext_tox_candidates,
    build_context_and_plan,
    committee_candidates,
    compact_physics_candidates,
    default_candidates,
    embedding_candidates,
    mmp_candidates,
    monroe_candidates,
    pair_candidates,
    physics_b_candidates,
    physics_candidates,
    population_candidates,
    population_physics_candidates,
    run,
    testlike_candidates,
    xiso_candidates,
)
from cypmode.challenge.submission import build_submission, load_test_set, portal_validate_classification, validate_submission
from cypmode.challenge.validation_matching import SimilarityMatcher, weighted_mcc

LOGS = Path("data/structure_track/outputs/cyp3a4/logs")
TRAIN_CSV = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST_CSV = "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
STORE = "data/tdi_track/outputs/caches/feature_store.npz"
NESTED = Path("data/tdi_track/outputs/nested")
BOARDS = {
    "CYP3A4": "data/tdi_track/inputs/leaderboard_snapshots/interim_CYP3A4_2026-10-04.csv",
    "CYP2D6": "data/tdi_track/inputs/leaderboard_snapshots/interim_CYP2D6_2026-10-04.csv",
}
Z_CUTS = (1.0, 1.96)
REPEATS = 3
RATE_MULTIPLIERS = (1.0, 1.1, 1.2, 1.3)  # called rate as a multiple of the scored population's prevalence
# group -> (candidate builder, log file pattern); the log carries the pre-registered GATE lines
GROUPS = {
    "physics": (physics_candidates, "physics_experiment_{iso}.log"),
    "pair_desc": (lambda: pair_candidates(False), "pair_transfer_{iso}.log"),
    "pair_physics": (lambda: [c for c in pair_candidates(True) if c.name == "pair_desc+phys"], "pair_transfer_physics_{iso}.log"),
    "mmp": (mmp_candidates, "mmp_transfer_{iso}.log"),
    "embeddings": (embedding_candidates, "embeddings_{iso}.log"),
    "testlike": (testlike_candidates, "test_likeness_{iso}.log"),
    "arms_clip": (arm_candidates, "arms_experiment_{iso}.log"),
    "committee": (committee_candidates, "committee_label_model_{iso}.log"),
    "physics_compact": (compact_physics_candidates, "physics_compact_{iso}.log"),
    "physics_b": (physics_b_candidates, "physics_tierb_{iso}.log"),
    "xiso": (xiso_candidates, "xiso_transfer_{iso}.log"),
    "screen": (lambda: aux_candidates(("screen",)), "aux_screen_{iso}.log"),
    "emax": (lambda: aux_candidates(("emax",)), "aux_emax_{iso}.log"),
    "pxr": (lambda: aux_candidates(("pxr",)), "aux_pxr_{iso}.log"),
    "gate_shift": (gate_shift_candidates, "aux_gate_shift_{iso}.log"),
    "nodirect": (nodirect_candidates, "aux_nodirect_{iso}.log"),
    "basic_centre": (basic_centre_candidates, "aux_basic_centre_{iso}.log"),
    "ext_cont": (ext_cont_candidates, "aux_ext_cont_{iso}.log"),
    "ext_tox": (ext_tox_candidates, "aux_ext_tox_{iso}.log"),
    "ext_adme": (ext_adme_candidates, "aux_ext_adme_{iso}.log"),
    "atomjepa": (atomjepa_candidates, "aux_atomjepa_{iso}.log"),
    "monroe": (monroe_candidates, "monroe_embeddings_{iso}.log"),
    **{name: (builder, f"aux_{name}_{{iso}}.log") for name, builder in CACHED_GROUPS.items()},
}
REPLICATE_LINE = re.compile(r"^(CYP\w+) GATE \(replicate (\w+)[^)]*\): (MET|NOT MET)", re.M)
DESIGN = {"outer": 5, "root": NESTED}  # design 10 switches to the 10-outer-fold replicate (see replicate_verdict.py)
GATE_LINE = re.compile(r"^(CYP\w+) GATE \((alone|stacked)[^)]*\): (MET|NOT MET)", re.M)


CONFIRMED_GROUPS = ("neural",)  # groups found after looking at the 15-fold results: their stacked pass counts only when the 10-outer-fold replicate (replicate_verdict.py) is MET as well


def parse_gates(logs: Path = LOGS) -> dict[tuple[str, str], dict[str, bool]]:
    """{(group, isoform): {"alone": bool, "stacked": bool}} from the pre-registered GATE lines; missing log -> absent.

    A group of ``CONFIRMED_GROUPS`` passes only if the replicate verdict of the same group and isoform is MET too (a missing replicate line is a fail).
    """
    out: dict[tuple[str, str], dict[str, bool]] = {}
    for group, (_, pattern) in GROUPS.items():
        for iso in ("CYP3A4", "CYP2D6"):
            path = logs / pattern.format(iso=iso)
            if not path.exists():
                continue
            verdicts = {kind: verdict == "MET" for found_iso, kind, verdict in GATE_LINE.findall(path.read_text()) if found_iso == iso}
            if verdicts:
                out[(group, iso)] = verdicts
    replicated = parse_replicate_gates(logs)
    for group in CONFIRMED_GROUPS:
        for iso in ("CYP3A4", "CYP2D6"):
            if (group, iso) in out:
                out[(group, iso)]["stacked"] = bool(out[(group, iso)].get("stacked")) and replicated.get((group, iso), {}).get("stacked", False)
    return out


def parse_replicate_gates(logs: Path = LOGS) -> dict[tuple[str, str], dict[str, bool]]:
    """Verdicts of the 10-fold replicate, written by replicate_verdict.py; a group replicated counts as a stacked pass."""
    out: dict[tuple[str, str], dict[str, bool]] = {}
    for iso in ("CYP3A4", "CYP2D6"):
        path = logs / f"replicate_verdict_{iso}.log"
        if path.exists():
            for found_iso, group, verdict in REPLICATE_LINE.findall(path.read_text()):
                if found_iso == iso:
                    out[(group, iso)] = {"stacked": verdict == "MET"}
    return out


def survivors(gates: dict, iso: str) -> list[str]:
    return [g for (g, i), v in gates.items() if i == iso and v.get("stacked")]


def choose_stack(cache, ctx, matcher, prevalence, base: list[str], groups: dict[str, list[str]], n_sizes: tuple[int, int]) -> tuple[str, list[str], pd.DataFrame]:
    """The eligibility rule of the module docstring; returns (label, candidate names, evidence table)."""
    options = {"default": base}
    for g, names in groups.items():
        options[f"default+{g}"] = base + names
    if len(groups) > 1:
        options["default+all_survivors"] = base + [n for names in groups.values() for n in names]
    rows = []
    for label, names in options.items():
        row = {"stack": label, "n_candidates": len(names)}
        eligible = label == "default"
        wins = []
        for z0 in Z_CUTS:
            weight_fn = confident_weights(matcher, ctx.confidence, z0)
            scored = nested_evaluate(cache, names, [StackedLogit()], weight_fn, prevalence, "auc")
            row[f"auc_z{z0}"] = float(scored["outer_score"].mean())
            row[f"mcc_z{z0}"] = float(scored["outer_mcc_rate"].mean())
            if label != "default":
                c = compare_candidate_sets(cache, names, base, StackedLogit(), weight_fn, prevalence, "auc")
                row[f"diff_z{z0}"], row[f"p_z{z0}"] = c.mean_difference, c.p_value
                wins.append(c.p_value < 0.05 and c.mean_difference > 0)
        row["eligible"] = eligible or (len(wins) == len(Z_CUTS) and all(wins))
        rows.append(row)
    table = pd.DataFrame(rows)
    eligible = table[table.eligible].copy()
    eligible["mean_auc"] = eligible[[f"auc_z{z}" for z in Z_CUTS]].mean(axis=1)
    best = eligible.sort_values(["mean_auc", "n_candidates"], ascending=[False, True]).iloc[0]["stack"]
    return best, options[best], table


def choose_call_rate(cache, names, ctx, weight_fn_for, prevalence) -> tuple[float, pd.DataFrame]:
    """Pre-declared rule for the call rate: the multiple of the prevalence in {1.0, 1.1, 1.2, 1.3} with the highest mean
    outer-fold weighted MCC over both z0 (the stack is refitted inside every outer fold exactly as in nested_evaluate)."""
    plan = cache.plan
    mcc: dict[tuple[float, float], list[float]] = {(z, m): [] for z in Z_CUTS for m in RATE_MULTIPLIERS}
    for repeat, k in plan.outer_folds():
        rows, oof = inner_oof(cache, names, repeat, k)
        refit = plan.task(repeat, k, OUTER_REFIT)
        outer = np.column_stack([cache.get(c, refit) for c in names])
        y_in, y_out = plan.labels(rows), plan.labels(refit.evaluate)
        chosen = StackedLogit().fit(list(names), oof, y_in, _weights(y_in, rows, refit.train, weight_fn_for(1.0), prevalence), "auc")
        scores = chosen.predict(outer)
        for z0 in Z_CUTS:
            w_out = _weights(y_out, refit.evaluate, refit.train, weight_fn_for(z0), prevalence)
            for m in RATE_MULTIPLIERS:
                mcc[(z0, m)].append(weighted_mcc(y_out, call_top_fraction(scores, m * prevalence).astype(int), w_out))
    table = pd.DataFrame([{"z0": z, "multiplier": m, "mean_outer_mcc": float(np.mean(v))} for (z, m), v in mcc.items()])
    by_multiplier = table.groupby("multiplier").mean_outer_mcc.mean()
    return float(by_multiplier.idxmax()), table


def calibrated_probabilities(cache, names, deployment, plan) -> np.ndarray:
    """Isotonic map fitted on the inner out-of-fold stack scores (checklist item I9), applied to the test scores."""
    rows, _ = inner_oof(cache, names, -1, 0)
    y = plan.labels(rows)
    calibrator = fit_calibrator("isotonic", np.asarray(deployment.selection.inner_scores), y, deployment.selection.inner_weights)
    return np.clip(calibrator(deployment.scores), 1e-4, 1 - 1e-4)


def committee_disagreement(cache, names, plan) -> np.ndarray:
    """Spread (standard deviation) of the member candidates' rank-normalised deployment scores per test compound."""
    task = plan.task(-1, 0, DEPLOY)
    matrix = np.column_stack([cache.get(c, task) for c in names])
    ranks = matrix.argsort(axis=0).argsort(axis=0) / max(len(matrix) - 1, 1)
    return ranks.std(axis=1)


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join(" --- " for _ in frame.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment() -> dict:
    import lightgbm
    import numpy
    import rdkit
    import scipy
    import sklearn

    return {"python": platform.python_version(), "platform": platform.platform(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__, "rdkit": rdkit.__version__, "pandas": pd.__version__}


def leakage_audit(plan, ctx, n_train: int, n_all: int) -> dict:
    """Test structures are used unlabeled only: no test row in the training pool, no label stored for any test row."""
    external = np.asarray(plan.external)
    checks = {
        "external rows are exactly the test file rows": bool((external == np.arange(n_train, n_all)).all()),
        "no test row in the labelled pool": bool(np.intersect1d(external, plan.pool).size == 0),
        "no label stored for any test row": bool((ctx.labels[external] == -1).all()),
    }
    assert all(checks.values()), f"LEAKAGE AUDIT FAILED: {checks}"
    return checks


def build_isoform(iso: str, gates: dict, test: pd.DataFrame, store: FeatureStore, out_dir: Path, fill_deploy: bool) -> dict:
    root = Path(DESIGN["root"]) / iso
    live = survivors(gates, iso)
    candidates_by_group = {g: [c for c in GROUPS[g][0]() if c.name] for g in live}
    wanted_blocks = tuple(sorted({b for cs in candidates_by_group.values() for c in cs for b in c.blocks}))
    ctx, plan = build_context_and_plan(TRAIN_CSV, TEST_CSV, iso, STORE, REPEATS, DESIGN["outer"], 4, 0, 4, wanted_blocks or None)
    saved = NestedPlan.load(root / "plan.npz")
    assert saved.plan_id == plan.plan_id, "the plan changed: the final build must share the experiment caches"
    cache = PredictionCache(root / "cache", saved)
    n_train, n_all = len(pd.read_csv(TRAIN_CSV)), len(ctx.labels)
    audit = leakage_audit(plan, ctx, n_train, n_all)
    matcher = SimilarityMatcher(store.blocks["morgan_bits"], plan.external)
    prevalence = float(estimate_from_csv(BOARDS[iso])["prevalence"])

    if fill_deploy:  # the experiments cached cross-validation tasks only; the winners also need their deployment fits
        for g, cs in candidates_by_group.items():
            run(cs, ctx, plan, cache, include_deploy=True)
    default_names = [c.name for c in default_candidates() if not cache.missing(c.name)]
    groups = {g: [c.name for c in cs if not cache.missing(c.name)] for g, cs in candidates_by_group.items()}
    groups = {g: names for g, names in groups.items() if names}
    label, names, evidence = choose_stack(cache, ctx, matcher, prevalence, default_names, groups, (0, 0))
    deployment = deploy(cache, names, StackedLogit(), matcher, prevalence)
    assert np.isfinite(deployment.scores).all() and len(deployment.scores) == len(test)
    multiplier, rate_table = choose_call_rate(cache, names, ctx, lambda z0: confident_weights(matcher, ctx.confidence, z0), prevalence)
    calls = call_top_fraction(deployment.scores, multiplier * prevalence)
    rate_table.to_csv(out_dir / f"{iso}_call_rate_choice.csv", index=False)
    scores = pd.DataFrame({
        "Molecule_Name": test["Molecule_Name"].to_numpy(),
        "score": deployment.scores,
        "call": calls,
        "probability_isotonic": calibrated_probabilities(cache, names, deployment, plan),
        "committee_disagreement": committee_disagreement(cache, names, plan),
    })
    scores.to_csv(out_dir / f"{iso}_hits_only_scores.csv", index=False)
    evidence.to_csv(out_dir / f"{iso}_stack_choice.csv", index=False)
    return {
        "isoform": iso, "stack": label, "n_candidates": len(names), "surviving_groups": live, "target_prevalence": prevalence,
        "called_rate": float(calls.mean()), "call_rate_multiplier": multiplier, "rate_before_cap_rule": float(deployment.calls.mean()), "threshold": float(deployment.threshold),
        "stack_weights": {k: round(v, 3) for k, v in deployment.selection.chosen.items()},
        "leakage_audit": audit, "calls": calls, "scores": deployment.scores, "evidence": evidence,
    }


def build_selection_aware(test: pd.DataFrame, store: FeatureStore, out_dir: Path, fit_weight: float, fill_deploy: bool) -> dict | None:
    """CYP2D6 selection-aware stack on the panel pool (hedged_blend picks the fit weight; 1.0 = the training share)."""
    iso, root = "CYP2D6", NESTED / "CYP2D6_pop_panel"
    if not (root / "plan.npz").exists():
        return None
    candidates = population_candidates()
    physics_log = LOGS / "population_physics_CYP2D6.log"
    with_physics = physics_log.exists() and "GATE (population physics stacked, corrected p < 0.05 at both z0): MET" in physics_log.read_text()
    if with_physics:
        candidates = candidates + population_physics_candidates()
    screen_log = LOGS / "population_screen_CYP2D6.log"
    with_screen = screen_log.exists() and "GATE (population screen stacked, corrected p < 0.05 at both z0): MET" in screen_log.read_text()
    if with_screen:
        candidates = candidates + screen_selection_candidates()
    blocks = tuple(sorted({b for c in candidates for b in c.blocks}))
    ctx, plan = build_context_and_plan(TRAIN_CSV, TEST_CSV, iso, STORE, REPEATS, 5, 4, 0, 4, blocks, None, True)
    saved = NestedPlan.load(root / "plan.npz")
    assert saved.plan_id == plan.plan_id
    cache = PredictionCache(root / "cache", saved)
    if fill_deploy:
        run(candidates, ctx, plan, cache, include_deploy=True)
    names = [c.name for c in candidates if not cache.missing(c.name)]
    if not names:
        return None
    from cypmode.challenge.population_experiment import scaled_nonhits

    matcher = SimilarityMatcher(store.blocks["morgan_bits"], plan.external)
    prevalence = float(estimate_from_csv(BOARDS[iso])["prevalence"])
    weight_fn = scaled_nonhits(confident_weights(matcher, ctx.confidence, 1.0), ctx.hit, fit_weight)
    deployment = deploy(cache, names, StackedLogit(), weight_fn, prevalence)
    assert np.isfinite(deployment.scores).all() and len(deployment.scores) == len(test)
    multiplier, rate_table = choose_call_rate(cache, names, ctx, lambda z0: scaled_nonhits(confident_weights(matcher, ctx.confidence, z0), ctx.hit, fit_weight), prevalence)
    calls = call_top_fraction(deployment.scores, multiplier * prevalence)
    rate_table.to_csv(out_dir / "CYP2D6_selection_aware_call_rate_choice.csv", index=False)
    pd.DataFrame({"Molecule_Name": test["Molecule_Name"].to_numpy(), "score": deployment.scores, "call": calls}).to_csv(out_dir / "CYP2D6_selection_aware_scores.csv", index=False)
    return {"n_candidates": len(names), "with_physics": with_physics, "with_screen": with_screen, "fit_weight": fit_weight, "called_rate": float(calls.mean()), "call_rate_multiplier": multiplier, "calls": calls, "scores": deployment.scores,
            "stack_weights": {k: round(v, 3) for k, v in deployment.selection.chosen.items()}}


def write_validated(test: pd.DataFrame, cyp2d6: np.ndarray, cyp3a4: np.ndarray, path: Path) -> dict:
    submission = build_submission(test, cyp2d6, cyp3a4)
    ok, errors = validate_submission(submission, set(test["Molecule_Name"]))
    assert ok, f"{path.name} fails validation: {errors}"
    submission.to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    for col in ("CYP2D6_is_TDI", "CYP3A4_is_TDI"):
        reloaded[col] = reloaded[col].astype(bool)
    ok, errors = validate_submission(reloaded, set(test["Molecule_Name"]))
    assert ok, f"{path.name} fails validation after reload: {errors}"
    portal_ok, portal_message = portal_validate_classification(path)
    assert portal_ok, f"{path.name} would be rejected by the portal: {portal_message}"
    assert (reloaded["Molecule_Name"].to_numpy() == test["Molecule_Name"].to_numpy()).all() and (reloaded["SMILES"].to_numpy() == test["SMILES"].to_numpy()).all()
    return {"file": path.name, "sha256": file_hash(path), "rows": len(reloaded), "CYP3A4_positive_rate": float(reloaded.CYP3A4_is_TDI.mean()), "CYP2D6_positive_rate": float(reloaded.CYP2D6_is_TDI.mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/tdi_track/outputs/final")
    parser.add_argument("--fit-weight", type=float, default=1.0, help="non-hit weight the selection-aware stack is fitted under (hedged_blend picks it)")
    parser.add_argument("--design", type=int, choices=(5, 10), default=5, help="5: the 5-outer-fold experiments; 10: the 10-outer-fold replicate (replicate_verdict.py)")
    parser.add_argument("--dry-run", action="store_true", help="read verdicts and cached predictions only; fit no missing deployment tasks")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, test = pd.read_csv(TRAIN_CSV), load_test_set(TEST_CSV)
    store = FeatureStore.load_or_build(STORE, list(train["SMILES"]) + list(test["SMILES"]))
    if args.design == 10:
        DESIGN.update(outer=10, root=Path("data/tdi_track/outputs/nested_f10"))
    gates = parse_replicate_gates() if args.design == 10 else parse_gates()
    results = {iso: build_isoform(iso, gates, test, store, out_dir, not args.dry_run) for iso in ("CYP3A4", "CYP2D6")}
    selection_aware = build_selection_aware(test, store, out_dir, args.fit_weight, not args.dry_run)

    files = [write_validated(test, results["CYP2D6"]["calls"], results["CYP3A4"]["calls"], out_dir / "candidate_A_hits_only.csv")]
    if selection_aware is not None:
        files.append(write_validated(test, selection_aware["calls"], results["CYP3A4"]["calls"], out_dir / "candidate_B_selection_aware.csv"))
    blend_log = LOGS / "hedged_blend_CYP2D6.log"
    blend_met = blend_log.exists() and "GATE (hedged blend adopted by the minimax rule, fit weight %s): MET" % args.fit_weight in blend_log.read_text()
    if blend_met and selection_aware is not None:
        from cypmode.challenge.nested import call_top_fraction

        rank = lambda s: pd.Series(s).rank(pct=True).to_numpy()  # noqa: E731
        blend = 0.5 * (rank(results["CYP2D6"]["scores"]) + rank(selection_aware["scores"]))
        files.append(write_validated(test, call_top_fraction(blend, results["CYP2D6"]["called_rate"]), results["CYP3A4"]["calls"], out_dir / "candidate_C_hedged_blend.csv"))

    manifest = {
        "environment": environment(),
        "seeds": {"nested plan seed": 0, "repeats": REPEATS, "outer": DESIGN["outer"], "inner": 4},
        "gate_verdicts": {f"{g}|{i}": v for (g, i), v in gates.items()},
        "isoforms": {iso: {k: v for k, v in r.items() if k not in ("calls", "scores", "evidence")} for iso, r in results.items()},
        "selection_aware": None if selection_aware is None else {k: v for k, v in selection_aware.items() if k not in ("calls", "scores")},
        "files": files,
        "hedged_blend_adopted": bool(blend_met),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    lines = ["# Final TDI build (generated; every number is computed by `final_tdi_build.py`)", ""]
    for iso, r in results.items():
        lines += [f"## {iso}", f"Final stack: **{r['stack']}** ({r['n_candidates']} candidates); surviving groups: {r['surviving_groups'] or 'none'}; called rate {r['called_rate']:.3f} against target prevalence {r['target_prevalence']:.3f}.", "", markdown_table(r["evidence"].round(4)), ""]
    lines += ["## Files", "", markdown_table(pd.DataFrame(files)), ""]
    (out_dir / "final_report.md").write_text("\n".join(lines))
    for f in files:
        print(f"{f['file']}: sha256 {f['sha256'][:16]}..., CYP3A4 {f['CYP3A4_positive_rate']:.3f}, CYP2D6 {f['CYP2D6_positive_rate']:.3f} positive")


if __name__ == "__main__":
    main()
