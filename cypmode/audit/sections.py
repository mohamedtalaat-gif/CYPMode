"""The audit sections. Each returns a ``Section``; every number is computed here from the live state."""

import importlib.metadata as md
import json
import re
import sys
import zipfile
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.audit.core import (
    FAIL,
    INFO,
    ISOFORMS,
    LOGS,
    NESTED_ROOT,
    PASS,
    ROOT,
    SNAPSHOTS,
    STORE_NPZ,
    TEST_CSV,
    TRAIN_CSV,
    Context,
    Section,
    fmt,
    md_table,
    run,
    sha256,
)

Z_CUTS = (1.0, 1.96)


# -- 1. provenance -------------------------------------------------------------------------------


def provenance(ctx: Context) -> Section:
    s = Section("1. Provenance and environment")
    pkgs = ["lightgbm", "scikit-learn", "numpy", "pandas", "scipy", "rdkit", "PyYAML"]
    versions = []
    for p in pkgs:
        try:
            versions.append((p, md.version(p)))
        except md.PackageNotFoundError:
            versions.append((p, "not installed"))
    versions.insert(0, ("python", ".".join(map(str, sys.version_info[:3]))))
    s.body.append(md_table(pd.DataFrame(versions, columns=["package", "version"])))
    head = run(["git", "rev-parse", "--short", "HEAD"]).strip()
    status = [line for line in run(["git", "status", "--porcelain"]).splitlines() if line]
    modified = sum(1 for line in status if line[:2].strip() in ("M", "MM", "AM", "RM"))
    untracked = sum(1 for line in status if line.startswith("??"))
    deleted = sum(1 for line in status if line[:2].strip() == "D")
    s.body.append(f"\nGit HEAD `{head}`; working tree: {modified} modified, {deleted} deleted, {untracked} untracked entries.")
    ctx.metrics.update(git_modified=modified, git_untracked=untracked)
    files = [TRAIN_CSV, TEST_CSV, *sorted(SNAPSHOTS.glob("*.csv"))]
    rows = [(str(p.relative_to(ROOT)), p.stat().st_size, sha256(p)) for p in files if p.exists()]
    s.body.append("\nInput files (size in bytes, sha256 prefix):\n\n" + md_table(pd.DataFrame(rows, columns=["file", "bytes", "sha256"])))
    s.add("training and test files present", TRAIN_CSV.exists() and TEST_CSV.exists())
    s.add("working tree clean (everything committed)", modified + untracked + deleted == 0, f"{modified} modified, {untracked} untracked")
    return s


# -- 2. tests ------------------------------------------------------------------------------------


def tests(ctx: Context, enabled: bool = True) -> Section:
    s = Section("2. Test suite (executed now)")
    if not enabled:
        s.body.append("Skipped (`--skip-tests`).")
        s.add("test suite executed", None, "skipped")
        ctx.metrics.update(tests_passed="not run", tests_failed="not run")
        return s
    out = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header", "tests"], timeout=3600)  # a hung test is a failed check, not a stalled audit
    last = [line for line in out.strip().splitlines() if re.search(r"\d+ (passed|failed|error)", line)]
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|deselected|errors?)", last[-1] if last else "")}
    failed = counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0)
    s.body.append("pytest summary counts: " + (", ".join(f"{k} = {v}" for k, v in sorted(counts.items())) or "no summary line found"))
    s.add("all tests pass", failed == 0 and counts.get("passed", 0) > 0, f"{counts.get('passed', 0)} passed, {failed} failed")
    ctx.metrics.update(tests_passed=counts.get("passed", 0), tests_failed=failed)
    return s


# -- 3. determinism ------------------------------------------------------------------------------


def determinism(ctx: Context) -> Section:
    from lightgbm import LGBMClassifier
    from rdkit import Chem

    from cypmode.challenge.feature_store import BLOCKS
    from cypmode.challenge.nested import NestedPlan
    from cypmode.challenge.nested_runners import (
        Candidate,
        default_candidates,
        fit_predict,
    )

    s = Section("3. Determinism and cache consistency")
    rows = []

    smiles = list(ctx.test["SMILES"][:150])
    mols = [Chem.MolFromSmiles(x) for x in smiles]
    for block in ("rdkit_desc", "morgan_c2", "tdi_alerts", "tdi_alert_context"):
        a, b = BLOCKS[block](mols), BLOCKS[block](mols)
        rows.append((f"feature block `{block}` computed twice", bool(np.array_equal(a, b)), f"max |diff| {fmt(float(np.max(np.abs(a - b))))}"))

    for iso in ISOFORMS:
        rctx, plan = ctx.runner(iso)
        saved, _ = ctx.cache(iso)
        rows.append((f"{iso} plan id rebuilt from raw files equals the saved plan", plan.plan_id == saved.plan_id, f"saved {saved.plan_id[:10]}, rebuilt {plan.plan_id[:10]}"))
        again = NestedPlan(plan.pool, plan.labels(plan.pool), plan.repeats, plan.outer_splits, plan.inner_splits, plan.seed, external=plan.external, groups=plan.groups)
        rows.append((f"{iso} plan splits identical when rebuilt in-process", all(np.array_equal(plan.task(r, k, -1).evaluate, again.task(r, k, -1).evaluate) for r, k in plan.outer_folds()), f"{len(plan.outer_folds())} outer folds"))

    rctx, plan = ctx.runner("CYP2D6")
    pool = plan.pool[:1500]
    x = rctx.store.matrix(("morgan_c2", "rdkit_desc"), pool)
    y = rctx.labels[pool]
    params = dict(n_estimators=80, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, random_state=0, verbose=-1)
    p1 = LGBMClassifier(n_jobs=1, **params).fit(x, y).predict_proba(x)[:, 1]
    p1b = LGBMClassifier(n_jobs=1, **params).fit(x, y).predict_proba(x)[:, 1]
    p4 = LGBMClassifier(n_jobs=4, **params).fit(x, y).predict_proba(x)[:, 1]
    rows.append(("LightGBM, same seed and 1 thread, fitted twice", bool(np.array_equal(p1, p1b)), f"max |diff| {fmt(float(np.max(np.abs(p1 - p1b))), 8)}"))
    rows.append(("LightGBM, same seed, 1 thread vs 4 threads", None, f"max |diff| {fmt(float(np.max(np.abs(p1 - p4))), 8)} (reported, not required to be zero)"))

    names = {c.name: c for c in default_candidates()}
    for iso in ISOFORMS:
        rctx, plan = ctx.runner(iso)
        _, cache = ctx.cache(iso)
        task = plan.task(0, 0, -1)
        for name in ("knn_tanimoto_k15", "svm_tanimoto_C1"):
            fresh = fit_predict(names[name], rctx, task)
            stored = cache.get(name, task)
            rows.append((f"{iso} cached `{name}` equals a fresh fit of the current code", bool(np.allclose(fresh, stored, atol=1e-9)), f"max |diff| {fmt(float(np.max(np.abs(fresh - stored))), 10)}"))

    lg = Candidate("probe", "lgbm_clf", ("morgan_c2", "rdkit_desc"), params={"n_estimators": 40, "num_leaves": 7, "learning_rate": 0.1})
    rctx, plan = ctx.runner("CYP2D6")
    t = plan.task(0, 0, 0)
    a, b = fit_predict(lg, rctx, t), fit_predict(lg, rctx, t)
    rows.append(("a plan task fitted twice with the runner (4 threads)", bool(np.allclose(a, b, atol=1e-6)), f"max |diff| {fmt(float(np.max(np.abs(a - b))), 8)}"))

    s.body.append(md_table(pd.DataFrame([(n, "PASS" if ok else "INFO" if ok is None else "FAIL", d) for n, ok, d in rows], columns=["check", "result", "detail"])))
    for n, ok, d in rows:
        s.add(n, ok, d)
    return s


# -- 4. data integrity ---------------------------------------------------------------------------


def data_integrity(ctx: Context) -> Section:
    from rdkit import Chem, RDLogger

    import cypmode.challenge.tdi_targets as T

    RDLogger.DisableLog("rdApp.*")
    s = Section("4. Data integrity")
    tr, te = ctx.train, ctx.test
    canon = lambda smi: (lambda m: Chem.MolToSmiles(m) if m is not None else None)(Chem.MolFromSmiles(smi))
    ctr, cte = [canon(x) for x in tr["SMILES"]], [canon(x) for x in te["SMILES"]]
    rows = [
        ("training rows", len(tr)),
        ("test rows", len(te)),
        ("invalid SMILES (train)", sum(c is None for c in ctr)),
        ("invalid SMILES (test)", sum(c is None for c in cte)),
        ("duplicate Molecule_Name (train)", int(tr["Molecule_Name"].duplicated().sum())),
        ("duplicate canonical SMILES (train)", len(ctr) - len(set(ctr))),
        ("test compounds whose canonical SMILES is also in train", len(set(cte) & set(ctr))),
        ("test compounds whose Molecule_Name is also in train", len(set(te["Molecule_Name"]) & set(tr["Molecule_Name"]))),
        ("label columns present in the test file", sum(c.endswith("_is_TDI") for c in te.columns)),
    ]
    s.body.append(md_table(pd.DataFrame(rows, columns=["item", "value"])))
    s.add("no invalid SMILES", rows[2][1] + rows[3][1] == 0, f"{rows[2][1] + rows[3][1]} invalid")
    s.add("no train/test structure overlap", rows[6][1] == 0, f"{rows[6][1]} shared structures")
    s.add("test file carries no labels", rows[8][1] == 0, f"{rows[8][1]} label columns")

    rule = []
    for iso in ISOFORMS:
        view = T.training_view(tr, iso)
        both = view[view["population"] == T.BOTH_ARMS]
        recomputed = ((both["tdi"] - np.maximum(both["direct"], T.ASSAY_FLOOR)) > T.SHIFT_THRESHOLD).astype(int)
        mismatch = int((recomputed.to_numpy() != both["label"].astype(int).to_numpy()).sum())
        pops = view["population"].value_counts().sort_index()
        conf = []
        for arm, col in (("TDI arm", f"{iso}_pIC50_TDI_condition"), ("direct arm", f"{iso}_pIC50_direct_inhibition")):
            v, lo, hi = tr[col], tr[col + "_conf_low"], tr[col + "_conf_high"]
            m = v.notna()
            conf.append(float(((lo[m] <= v[m]) & (v[m] <= hi[m])).mean()))
        rule.append(
            {
                "isoform": iso,
                "both-arms rows": len(both),
                "label-rule mismatches": mismatch,
                "no-direct rows": int(pops.get(T.NO_DIRECT, 0)),
                "positive rate (both arms)": float(both["label"].astype(int).mean()),
                "value inside its own interval, TDI arm": conf[0],
                "value inside its own interval, direct arm": conf[1],
            }
        )
        s.add(f"{iso}: label rule reproduces every both-arms label", mismatch == 0, f"{mismatch} mismatches of {len(both)}")
        ctx.metrics[f"{iso}_label_rule_mismatches"] = mismatch
    s.body.append("\nLabel rule `t - max(d, 4) > log10 2` against the given labels:\n\n" + md_table(pd.DataFrame(rule)))

    plan_rows = []
    for iso in ISOFORMS:
        plan, _ = ctx.cache(iso)
        disjoint = all(not set(plan.task(r, k, -1).train) & set(plan.task(r, k, -1).evaluate) for r, k in plan.outer_folds())
        covers = all(set(plan.task(r, k, -1).train) | set(plan.task(r, k, -1).evaluate) == set(plan.pool) for r, k in plan.outer_folds())
        ext = bool(set(plan.external) & set(plan.pool))
        inner = all(set(plan.task(r, k, i).evaluate) <= set(plan.task(r, k, -1).train) for r, k in plan.outer_folds() for i in range(plan.inner_splits))
        plan_rows.append((iso, len(plan.pool), len(plan.outer_folds()), disjoint, covers, not ext, inner))
        s.add(f"{iso}: outer train/evaluation disjoint and covering the pool in every fold", disjoint and covers)
        s.add(f"{iso}: inner folds stay inside the outer training rows", inner)
        s.add(f"{iso}: test (external) compounds never in the pool", not ext)
    s.body.append("\nNested plan integrity:\n\n" + md_table(pd.DataFrame(plan_rows, columns=["isoform", "pool", "outer folds", "train/eval disjoint", "train+eval = pool", "external not in pool", "inner inside outer train"])))
    return s


# -- 5. baselines --------------------------------------------------------------------------------


def baselines(ctx: Context) -> Section:
    from lightgbm import LGBMClassifier
    from sklearn.metrics import matthews_corrcoef, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    from cypmode.challenge.label_confidence import confidence_z
    from cypmode.challenge.nested import BestSingle, StackedLogit, nested_evaluate
    from cypmode.challenge.tdi_baseline import (
        compute_descriptors,
        load_tdi_training_data,
    )

    s = Section("5. Baselines against the current model")
    raw = ctx.train
    base = ctx.once("official_desc", lambda: compute_descriptors(load_tdi_training_data()).merge(raw, on=["Molecule_Name", "SMILES"], how="left", suffixes=("", "_raw")))
    rows = []
    for iso in ISOFORMS:
        col = f"{iso}_is_TDI"
        df = base.dropna(subset=[col]).reset_index(drop=True)
        y = df[col].astype(int).to_numpy()
        x = np.stack(df["descriptors"])
        oof = np.zeros(len(df))
        for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=42).split(x, y):
            oof[te_i] = LGBMClassifier(verbose=-1, random_state=42, n_jobs=ctx.n_jobs).fit(x[tr_i], y[tr_i]).predict_proba(x[te_i])[:, 1]
        pred = (oof >= 0.5).astype(int)
        z = confidence_z(df, iso)
        both = ~np.isnan(df[f"{iso}_pIC50_direct_inhibition"].to_numpy())
        for label, mask in [("all labeled rows", np.ones(len(df), bool)), ("both-arms rows", both)] + [(f"both arms, |z| >= {z0}", both & (z >= z0)) for z0 in Z_CUTS]:
            rows.append((iso, "official baseline (descriptor LightGBM, 5-fold OOF)", label, int(mask.sum()), roc_auc_score(y[mask], oof[mask]), matthews_corrcoef(y[mask], pred[mask])))
        board = []
        for kind in ("live", "interim"):
            f = SNAPSHOTS / f"{kind}_{iso}_2026-10-07.csv"
            if not f.exists():
                f = SNAPSHOTS / f"{kind}_{iso}_2026-10-04.csv"
            if f.exists():
                b = pd.read_csv(f)
                hit = b[b["Model Report Link"].astype(str).str.contains("Official LightGBM baseline", na=False)]
                if len(hit):
                    board.append((kind, float(hit["MCC"].iloc[0]), int(hit["Rank"].iloc[0]), len(b)))
        for kind, mcc, rank, n in board:
            rows.append((iso, f"official baseline on the {kind} board", f"rank {rank} of {n}", n, np.nan, mcc))
            ctx.metrics[f"{iso}_baseline_{kind}_board_mcc"] = mcc

        plan, cache = ctx.cache(iso)
        names = ctx.lgbm_family(iso)
        prev = ctx.prevalence(iso)
        for z0 in (None, *Z_CUTS):
            wf = ctx.weight_fn(iso, z0)
            tag = "similarity x prevalence" + ("" if z0 is None else f" x confident (z0 = {z0})")
            for sel, label in ((BestSingle(), "best single candidate"), (StackedLogit(), "stack (StackedLogit)")):
                table = nested_evaluate(cache, names, [sel], wf, prev, "auc")
                rows.append((iso, f"our nested {label}", tag, len(plan.pool), float(table["outer_score"].mean()), float(table["outer_mcc_rate"].mean())))
                if sel.name == "stack_logit":
                    ctx.metrics[f"{iso}_stack_auc_z{str(z0).replace('.', '_')}"] = float(table["outer_score"].mean())
    df = pd.DataFrame(rows, columns=["isoform", "model", "evaluation population", "n", "AUC", "MCC"])
    s.body.append("AUC and MCC are not comparable across rows with different evaluation populations; compare within a population.\n\n" + md_table(df))
    s.add("official baseline reproduced from the raw file", True, f"{len(df)} rows computed")
    return s


# -- 6. committee backtest -----------------------------------------------------------------------


def committee(ctx: Context) -> Section:
    from cypmode.challenge.committee_uncertainty import (
        BASELINE,
        collect_signals,
        decision_record,
        evaluate_components,
        selective_fallback,
    )
    from cypmode.challenge.nested_runners import default_candidates

    s = Section("6. Committee (query-by-committee) backtest against the margin baseline")
    s.body.append("Question: does committee disagreement tell us where the stack is wrong beyond what the stack's own decision margin already says? Judged by a permutation null with Bonferroni correction, never by bootstrapping fixed predictions.")
    for iso in ISOFORMS:
        plan, cache = ctx.cache(iso)
        rctx, _ = ctx.runner(iso)
        order = [c.name for c in default_candidates()]
        names = [n for n in order if not cache.missing(n)] + [n for n in cache.candidates() if n not in order and not cache.missing(n)]
        frame = collect_signals(cache, names, rctx.store.blocks["morgan_bits"])
        ev = evaluate_components(frame)
        fb = pd.concat([selective_fallback(frame, sig) for sig in ("qbc_vote_entropy", "qbc_spread", BASELINE)], ignore_index=True)
        record = decision_record(iso, ev, fb)
        s.body.append(f"\n### {iso}: {len(names)}-member committee, {len(frame)} held-out compounds, stack error rate {fmt(float(frame['wrong'].mean()))}\n")
        s.body.append(md_table(ev[["component", "auroc", "auroc_lo", "auroc_hi", "gain_over_margin", "gain_p", "verdict"]]))
        s.body.append("\nSelective prediction (abstain on the least reliable fraction):\n\n" + md_table(fb))
        used = {k: bool(v["used_to_alter_predictions"]) for k, v in record["components"].items()}
        s.body.append("\nComponents allowed to alter a prediction: " + (", ".join(k for k, v in sorted(used.items()) if v) or "none"))
        s.add(f"{iso}: committee backtest executed", True, f"{len(ev)} components")
        s.add(f"{iso}: no component alters predictions without adding information AND improving MCC", not any(used.values()), "policy check")
        ctx.metrics[f"{iso}_committee_components_adding_info"] = int((ev["verdict"] == "adds_information").sum())
    return s


# -- 7. calibration ------------------------------------------------------------------------------


def calibration(ctx: Context) -> Section:
    from scipy.special import expit

    from cypmode.challenge.calibration_experiment import (
        ECE_GATE,
        SLOPE_HI,
        SLOPE_LO,
        calibration_metrics,
        fit_calibrator,
    )
    from cypmode.challenge.nested import OUTER_REFIT, StackedLogit, _weights, inner_oof

    s = Section("7. Calibration of the stack on held-out compounds")
    s.body.append(
        "Outer-fold probabilities, weighted by similarity to the test set and by the test prevalence, 10 equal-width bins. "
        "Two outputs are measured: the sigmoid of the blend's decision value exactly as the regularised logistic blend gives it "
        "(it ranks; it is not published as a probability), and the isotonic-calibrated probability that the final build writes "
        "(``probability_isotonic``), whose calibrator is fitted on the inner out-of-fold blend scores of each outer fold and never on outer rows. "
        f"The pass bar, fixed in advance in ``calibration_experiment.py``, is weighted ECE <= {ECE_GATE} and slope in [{SLOPE_LO}, {SLOPE_HI}] for the published output."
    )
    for iso in ISOFORMS:
        plan, cache = ctx.cache(iso)
        names = ctx.lgbm_family(iso)
        wf, prev = ctx.weight_fn(iso), ctx.prevalence(iso)
        ys, raws, deployed, ws = [], [], [], []
        for repeat, k in plan.outer_folds():
            rows, oof = inner_oof(cache, names, repeat, k)
            refit = plan.task(repeat, k, OUTER_REFIT)
            outer = np.column_stack([cache.get(c, refit) for c in names])
            y_in, y_out = plan.labels(rows), plan.labels(refit.evaluate)
            w_in = _weights(y_in, rows, refit.train, wf, prev)
            chosen = StackedLogit().fit(names, oof, y_in, w_in, "auc")
            calibrator = fit_calibrator("isotonic", np.asarray(chosen.inner_scores), y_in, w_in)
            ys.append(y_out)
            raws.append(expit(chosen.predict(outer)))
            deployed.append(np.clip(calibrator(chosen.predict(outer)), 1e-4, 1 - 1e-4))
            ws.append(_weights(y_out, refit.evaluate, refit.train, wf, prev))
        y, w = np.concatenate(ys), np.concatenate(ws)
        w = w / w.sum()
        raw, final = np.concatenate(raws), np.concatenate(deployed)
        m_raw, m_final = calibration_metrics(y, raw, w), calibration_metrics(y, final, w)
        bins = np.minimum((final * 10).astype(int), 9)
        table = []
        for b in range(10):
            sel = bins == b
            if sel.any():
                table.append((f"{b / 10:.1f}-{(b + 1) / 10:.1f}", float(w[sel].sum()), float(np.average(final[sel], weights=w[sel])), float(np.average(y[sel], weights=w[sel]))))
        s.body.append(
            f"\n### {iso}\n\nPublished (isotonic): weighted ECE {fmt(m_final['ece'])}, Brier {fmt(m_final['brier'])}, calibration slope {fmt(m_final['slope'])} (1 = perfect), "
            f"intercept {fmt(m_final['intercept'])} (0 = perfect); mean predicted {fmt(m_final['mean_pred'])} vs observed {fmt(m_final['mean_obs'])}.\n\n"
            f"Raw blend sigmoid (not published as a probability): ECE {fmt(m_raw['ece'])}, Brier {fmt(m_raw['brier'])}, slope {fmt(m_raw['slope'])}, "
            f"mean predicted {fmt(m_raw['mean_pred'])}.\n"
        )
        s.body.append(md_table(pd.DataFrame(table, columns=["bin (published)", "weight share", "mean predicted", "observed rate"])))
        gap = abs(m_final["mean_pred"] - m_final["mean_obs"])
        s.add(f"{iso}: calibration computed", True, f"published ECE {fmt(m_final['ece'])}, slope {fmt(m_final['slope'])}; raw ECE {fmt(m_raw['ece'])}, slope {fmt(m_raw['slope'])}")
        s.add(f"{iso}: published mean probability within 0.05 of the observed rate", gap < 0.05, f"gap {fmt(gap)}")
        s.add(f"{iso}: published probabilities usable as probabilities (slope between {SLOPE_LO} and {SLOPE_HI})", SLOPE_LO <= m_final["slope"] <= SLOPE_HI, f"slope {fmt(m_final['slope'])}")
        s.add(f"{iso}: raw blend sigmoid is a ranking score, not a probability (reported only)", None, f"ECE {fmt(m_raw['ece'])}, slope {fmt(m_raw['slope'])}")
        ctx.metrics.update({f"{iso}_ece": m_final["ece"], f"{iso}_calibration_slope": m_final["slope"], f"{iso}_raw_ece": m_raw["ece"], f"{iso}_raw_calibration_slope": m_raw["slope"]})
    return s


# -- 8. chemistry evidence -----------------------------------------------------------------------


def _bit_fragment(bit: int, mols: list) -> str:
    from collections import Counter

    from rdkit import Chem
    from rdkit.Chem import AllChem

    seen = Counter()
    for m in mols:
        info: dict = {}
        AllChem.GetHashedMorganFingerprint(m, 2, nBits=2048, bitInfo=info)
        for atom, radius in info.get(bit, ()):
            if radius == 0:
                seen[m.GetAtomWithIdx(atom).GetSymbol()] += 1
                continue
            env = Chem.FindAtomEnvironmentOfRadiusN(m, radius, atom)
            amap: dict = {}
            sub = Chem.PathToSubmol(m, env, atomMap=amap)
            seen[Chem.MolToSmiles(sub)] += 1
    return seen.most_common(1)[0][0] if seen else "?"


def chemistry(ctx: Context) -> Section:
    from lightgbm import LGBMClassifier
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from scipy.stats import spearmanr
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from cypmode.challenge.feature_store import _ALERT_PATTERNS
    from cypmode.challenge.nested_runners import LGBM_PRESETS

    s = Section("8. Chemistry evidence: exact TreeSHAP, architecture agreement, alert enrichment")
    names = [f"morgan_count_bit_{i}" for i in range(2048)] + [n for n, _ in Descriptors.descList]
    smiles = ctx.train["SMILES"].tolist()
    mols_all = [Chem.MolFromSmiles(x) for x in smiles]
    for iso in ISOFORMS:
        rctx, plan = ctx.runner(iso)
        pool = plan.pool
        x = rctx.store.matrix(("morgan_c2", "rdkit_desc"), pool)
        y = rctx.labels[pool]
        lg = LGBMClassifier(verbose=-1, n_jobs=ctx.n_jobs, random_state=0, **LGBM_PRESETS["deep"]).fit(x, y)
        contrib = lg.booster_.predict(x, pred_contrib=True)[:, :-1]  # exact TreeSHAP, last column is the bias
        shap_imp = np.abs(contrib).mean(axis=0)
        order = np.argsort(-shap_imp, kind="stable")[:15]
        pool_mols = [mols_all[i] for i in pool[:600]]
        top = []
        for j in order:
            what = _bit_fragment(j, pool_mols) if j < 2048 else "descriptor"
            direction = float(np.corrcoef(x[:, j], contrib[:, j])[0, 1]) if x[:, j].std() > 0 and contrib[:, j].std() > 0 else 0.0
            top.append((names[j], what, shap_imp[j], direction))
        s.body.append(f"\n### {iso}: top 15 features by mean |SHAP| (LightGBM, {len(pool)} pool compounds)\n\nThe last column is the correlation between the feature value and its SHAP value (positive: more of it pushes toward TDI-positive).\n")
        s.body.append(md_table(pd.DataFrame(top, columns=["feature", "environment / kind", "mean |SHAP|", "direction"])))

        et = ExtraTreesClassifier(n_estimators=300, max_features=0.1, min_samples_leaf=2, n_jobs=ctx.n_jobs, random_state=0).fit(x, y)
        lr = LogisticRegression(C=0.05, max_iter=2000).fit(StandardScaler().fit_transform(x), y)
        agree = []
        for label, imp in (("ExtraTrees impurity importance", et.feature_importances_), ("standardized L2-logistic |coef|", np.abs(lr.coef_[0]))):
            rho = float(spearmanr(shap_imp, imp)[0])
            k = 30
            overlap = len(set(np.argsort(-shap_imp, kind="stable")[:k]) & set(np.argsort(-imp, kind="stable")[:k]))
            agree.append((label, rho, f"{overlap} of {k}"))
            ctx.metrics[f"{iso}_shap_agreement_{label.split()[0]}"] = rho
        s.body.append(f"\nAgreement of the SHAP importance with other architectures ({iso}):\n\n" + md_table(pd.DataFrame(agree, columns=["architecture", "Spearman rho", "top-30 overlap"])))
        s.add(f"{iso}: SHAP importance agrees (rho > 0.2) with a second architecture", agree[0][1] > 0.2, f"rho {fmt(agree[0][1])}")

        pm = [mols_all[i] for i in pool]
        rows = []
        for name, patt in sorted(_ALERT_PATTERNS.items()):
            has = np.array([m.HasSubstructMatch(patt) for m in pm])
            if has.sum() < 20 or (~has).sum() < 20:
                continue
            a, b, c, d = float(y[has].sum()) + 0.5, float((1 - y[has]).sum()) + 0.5, float(y[~has].sum()) + 0.5, float((1 - y[~has]).sum()) + 0.5
            lor = np.log(a * d / (b * c))
            se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
            rows.append((name, int(has.sum()), float(y[has].mean()), float(y[~has].mean()), float(np.exp(lor)), float(np.exp(lor - 1.96 * se)), float(np.exp(lor + 1.96 * se))))
        s.body.append(f"\nStructural-alert enrichment in the {iso} pool (alerts with at least 20 compounds on each side; odds ratio with 95% interval):\n\n" + md_table(pd.DataFrame(rows, columns=["alert", "n with", "positive rate with", "without", "odds ratio", "lo", "hi"])))
    return s


# -- 9. experiment ledger (parsed from logs) -----------------------------------------------------


def ledger(ctx: Context) -> Section:
    s = Section("9. Experiment ledger (parsed from the run logs)")
    patterns = re.compile(r"^\s*(z0=|S[0-3] |evaluation:|\w+: (population|\d+ outer folds))")
    found = 0
    for f in sorted(LOGS.glob("*.log")):
        if not re.match(r"(confidence|context|population)_", f.name):
            continue
        lines = [ln.rstrip() for ln in f.read_text(errors="ignore").splitlines() if patterns.match(ln) or "outer folds" in ln]
        if lines:
            found += 1
            s.body.append(f"\n`{f.name}`\n\n```\n" + "\n".join(lines) + "\n```")
    if not found:
        s.body.append("No experiment logs found.")
    s.add("experiment logs parsed", found > 0, f"{found} log files with results")
    ctx.metrics["ledger_logs"] = found
    return s


# -- 10. structure track -------------------------------------------------------------------------


def structure(ctx: Context) -> Section:
    s = Section("10. Structure-track benchmark (from the scored crystal poses)")
    out = ROOT / "data/structure_track/outputs/cyp3a4"
    found = False
    for name in ("benchmark_engine_pose_validation.csv", "benchmark_engine_pose_validation_unguided9.csv"):
        f = out / name
        if f.exists():
            t = pd.read_csv(f)
            found = True
            g = t[t["engine"] == "boltz"].sort_values(["compound", "sample"])
            per = g.groupby("compound").apply(lambda d: pd.Series({"poses": len(d), "first pick RMSD": d["true_rmsd"].iloc[0], "best pose RMSD": d["true_rmsd"].min(), "template-pick RMSD": d.loc[d["novel_template_score"].idxmax(), "true_rmsd"]}), include_groups=False).reset_index()
            s.body.append(f"`{name}`: {len(t)} scored poses, {per.shape[0]} ligands\n\n" + md_table(per, 2))
            means = per[["first pick RMSD", "best pose RMSD", "template-pick RMSD"]].mean()
            s.body.append("\nMean RMSD to the crystal pose: " + ", ".join(f"{k} = {fmt(v, 2)}" for k, v in means.items()) + f"; first picks under 2 A: {int((per['first pick RMSD'] < 2).sum())} of {len(per)}")
            ctx.metrics["structure_first_pick_rmsd"] = float(means["first pick RMSD"])
            break
    for log in ("bench_restraint_comparison.log", "pilot_report.log"):
        f = LOGS / log
        if f.exists() and f.stat().st_size:
            s.body.append(f"\n`{log}`\n\n```\n{f.read_text(errors='ignore').strip()}\n```")
    s.add("structure benchmark scores present", found, "no scored benchmark table found" if not found else "")
    return s


# -- 11. submission readiness --------------------------------------------------------------------


def readiness(ctx: Context) -> Section:
    s = Section("11. Submission-file readiness")
    f = ROOT / "submissions/tdi_track/tdi_track_submission.csv"
    if f.exists():
        d = pd.read_csv(f)
        test = ctx.test
        cols = ["SMILES", "Molecule_Name", "CYP2D6_is_TDI", "CYP3A4_is_TDI"]
        checks = {
            "750 rows": len(d) == 750,
            "required columns present": all(c in d.columns for c in cols),
            "no missing values": bool(d[cols].notna().all().all()) if all(c in d.columns for c in cols) else False,
            "calls are boolean": all(set(d[c].unique()) <= {True, False} for c in ("CYP2D6_is_TDI", "CYP3A4_is_TDI") if c in d.columns),
            "Molecule_Name order equals the test file": d["Molecule_Name"].tolist() == test["Molecule_Name"].tolist() if "Molecule_Name" in d.columns else False,
            "SMILES equal the test file": d["SMILES"].tolist() == test["SMILES"].tolist() if "SMILES" in d.columns else False,
        }
        for name, ok in checks.items():
            s.add(f"TDI file: {name}", ok)
        rates = {c: float(d[c].astype(int).mean()) for c in ("CYP2D6_is_TDI", "CYP3A4_is_TDI") if c in d.columns}
        s.body.append(f"`{f.relative_to(ROOT)}` (sha256 {sha256(f)}): " + ", ".join(f"called rate {c} = {fmt(v)}" for c, v in rates.items()) + " (test prevalence estimates: " + ", ".join(f"{iso} {fmt(ctx.prevalence(iso))}" for iso in ISOFORMS) + ")")
        s.body.append("\nThis is the file currently on disk; it is a candidate, not evidence that anything was submitted.")
    else:
        s.body.append("No TDI submission file on disk.")
        s.add("TDI submission file present", None, "none")
    z = ROOT / "submissions/structure_track/structure_track_submission.zip"
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            names = sorted(zf.namelist())
        cifs = [n for n in names if n.lower().endswith((".cif", ".pdb"))]
        s.body.append(f"\n`{z.relative_to(ROOT)}` (sha256 {sha256(z)}): {len(names)} entries, {len(cifs)} structure files")
        s.add("structure zip holds 20 structure files", len(cifs) == 20, f"{len(cifs)} structure files")
    else:
        s.add("structure submission zip present", None, "none")
    return s


# -- 11b. selection-aware CYP2D6 evidence and the share inference ---------------------------------


def population(ctx: Context) -> Section:
    s = Section("11b. Selection-aware CYP2D6 evidence (population shift)")
    band_file = ROOT / "data/tdi_track/outputs/comparisons/population_share_band.csv"
    for name in ("population_CYP2D6.log", "population_sensitivity_CYP2D6.log"):
        f = LOGS / name
        if f.exists():
            keep = [ln.rstrip() for ln in f.read_text(errors="ignore").splitlines() if re.match(r"^\s*(z0=|S[0-3] |evaluation:|CYP2D6: )", ln)]
            s.body.append(f"`{name}`\n\n```\n" + "\n".join(keep) + "\n```\n")
    if band_file.exists():
        g = pd.read_csv(band_file)
        best = g.loc[g["chi2"].idxmin()]
        board = pd.read_csv(SNAPSHOTS / "live_CYP2D6_2026-10-07.csv")
        row = board[board["Model Report Link"].astype(str).str.contains("Official LightGBM baseline", na=False)].iloc[0]
        band = g[g["plausible"]]
        rows = [
            ("leaderboard precision / recall of the official baseline", f"{fmt(float(row['Precision']))} / {fmt(float(row['Recall']))}"),
            ("same baseline, best re-weighted training population: precision / recall", f"{fmt(float(best['precision']))} / {fmt(float(best['recall']))}"),
            ("non-hit weight range inside the 95% region", f"{fmt(float(band['w'].min()), 2)} to {fmt(float(band['w'].max()), 2)}"),
            ("grid points in the region", f"{len(band)} of {len(g)}"),
        ]
        s.body.append("Inference of the scored set's non-hit share from the baseline's public metrics:\n\n" + md_table(pd.DataFrame(rows, columns=["quantity", "value"])))
        unidentified = float(band["w"].max() - band["w"].min()) > 1.0
        s.add(
            "non-hit share of the scored set versus the baseline's public metrics (a property of the public data, reported as limitation L9)",
            None,
            f"NOT identified: {len(band)} of {len(g)} grid points are consistent with the baseline's metrics; the hedged blend and the sensitivity table cover the whole range" if unidentified else "identified within the grid",
        )
        ctx.metrics["population_region_points"] = len(band)
    else:
        s.body.append("No population-share fit found (run `python -m cypmode.challenge.population_share`).")
        s.add("population-share fit present", None, "not run")
    return s


# -- 12. credit ------------------------------------------------------------------


def local_terms() -> list[str]:
    """Patterns read from a local list (one regular expression per line), a file that is not part of the repository."""
    path = ROOT / ".git" / "info" / "local_patterns"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def attribution(ctx: Context) -> Section:
    s = Section("12. Credit hygiene")
    terms = local_terms()
    forbidden = {
        "leaderboard alias used as a credit": re.compile(r"\b450nm\b"),
        **({"words of the private list": re.compile("|".join(terms), re.IGNORECASE)} if terms else {}),
    }
    tracked = run(["git", "ls-files"]).splitlines() + run(["git", "ls-files", "--others", "--exclude-standard"]).splitlines()
    text_ext = (".py", ".md", ".sh", ".txt", ".yaml", ".yml", ".toml", ".ipynb", ".cfg", ".html")
    counts = {k: 0 for k in forbidden}
    for rel in sorted(set(tracked)):
        p = ROOT / rel
        if not (rel.endswith(text_ext) or rel == ".gitignore") or rel.startswith(("cypmode/audit/", "third_party/", ".venv")) or not p.exists() or p.stat().st_size > 2_000_000:
            continue
        text = p.read_text(errors="ignore")
        for k, rx in forbidden.items():
            counts[k] += len(rx.findall(text))
    log = run(["git", "log", "--all", "--format=%B"])
    commit_hits = sum(len(forbidden[k].findall(log)) for k in forbidden if k not in ("leaderboard alias used as a credit",))
    authors = {a for a in run(["git", "log", "--all", "--format=%an <%ae>"]).splitlines() if a.strip()}  # the contributors graph is built from the authors
    rows = [(k, v) for k, v in counts.items()] + [
        ("words of the private list in commit messages (all branches)", commit_hits),
        ("commit authors besides the project author (all branches)", max(0, len(authors) - 1)),
    ]
    s.body.append(md_table(pd.DataFrame(rows, columns=["pattern", "occurrences in repository text"])))
    for k, v in rows:
        s.add(f"no occurrences of: {k}", v == 0, f"{v} found")
    if not terms:
        s.add("words of the private list", None, "list not present in this checkout: skipped")
    return s


# -- 13. limitations log -------------------------------------------------------------------------


def limitations(ctx: Context) -> Section:
    import yaml

    s = Section("13. Limitations log")
    entries = yaml.safe_load((Path(__file__).parent / "limitations.yaml").read_text())
    missing = []
    for e in entries:
        values = {}
        for key in e.get("uses", []):
            if key in ctx.metrics:
                values[key] = fmt(ctx.metrics[key]) if not isinstance(ctx.metrics[key], str) else ctx.metrics[key]
            else:
                missing.append(f"{e['id']}:{key}")
                values[key] = "n/a (metric not computed in this run)"
        s.body.append(f"- **{e['id']}** ({e['area']}): " + e["text"].format(**values))
    s.add("every limitation's evidence values were computed in this run", not missing, ", ".join(missing))
    return s


from cypmode.audit.sections_extra import (
    EXTRA,
)

ALL = [provenance, tests, determinism, data_integrity, baselines, committee, calibration, chemistry, ledger, structure, readiness, population, attribution, *EXTRA, limitations]
_ = (islice, json, FAIL, INFO, PASS, NESTED_ROOT, STORE_NPZ)  # names kept importable for callers
