# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The five principles of the Organisation for Economic Co-operation and Development (OECD) for the validation of (quantitative) structure–activity relationship ((Q)SAR) models, answered from the files of the project, together with the curation of the data.

The guidance document of the OECD on the validation of (quantitative) structure–activity relationship models (ENV/JM/MONO(2007)2) asks that a model for regulatory use have (1) a defined endpoint, (2) an
unambiguous algorithm, (3) a defined domain of applicability, (4) appropriate measures of goodness of fit, robustness, and predictivity, and (5) a mechanistic interpretation, if possible. The editors
of this field's journals use the same list as a test of a modelling paper (Merz et al., Journal of Chemical Information and Modeling, 2015, 10.1021/acs.jcim.5b00180) and ask for a section on data
curation. For the final time-dependent inhibition (TDI) stacks, this module collects the evidence that already exists, computes three things that did not exist, and writes one table in which every
cell is read from a file or computed here. Nothing in the table is typed by hand.

Computed here for the first time:

  * Goodness of fit. Every candidate of the default grid is refitted on the whole pool of its isoform and scored on that pool, on the yardstick of the nested design (the weights for similarity,
    prevalence, and confident rows), next to its outer-fold score. The difference is the optimism that the nested design removes. The stack itself is not refitted on its own training rows, because
    the blend is fitted on out-of-fold scores, and applying it to in-sample scores of its members would measure a different object. The candidates that are networks trained in another environment
    are not refitted either. For the neighbour models and the read-across columns, a compound that is scored on its own pool is its own nearest neighbour, so their fit is the fit of a model that has
    seen the answer. That is what goodness of fit measures, and it is the reason why the fit is never read as a forecast.
  * Interpretation. The enrichment of each structural alert and each heme-binding motif among the TDI-positive compounds of the pool (odds ratio with a 95% interval, Fisher's exact test, and the Holm
    adjustment over the alerts of the isoform), and the share of the gain of a LightGBM model on the widest fingerprint set that each block of columns carries. The alerts are those of the feature
    store, taken from the literature that ``feature_store.py`` cites (Orr et al., J. Med. Chem. 2012, 55, 4896; Kalgutkar et al., Curr. Drug Metab. 2007, 8, 407). The enrichment tests whether the
    alerts separate the positives of this data; it is not a claim about the mechanism of any one compound.
  * Curation. The counts of the checks on the files: parsing, duplicates, overlap of the blinded set with the training set, the rows that follow another labelling convention, and the rule of the label.

Read, not recomputed: the machine-readable algorithm (the candidates of each final stack with their families, columns, and hyperparameters; the blend; the call rule; the seeds; and the environment),
the applicability domain, the nested and permuted-label results, and the scores on the released labels. When their jobs have run, the table also reads the validation on a second data set and the
analysis of the ratio of columns to compounds. A value whose source does not exist yet reads ``(pending: <job>)``.

Usage: python -m cypmode.challenge.oecd_principles [--n-jobs 8] [--isoform CYP3A4]
Outputs (data/tdi_track/outputs/final): oecd_principles.md, oecd_principles.csv, algorithm_manifest.json, data_curation.csv, goodness_of_fit.csv, alert_enrichment.csv, block_gain.csv
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from scipy import stats
from statsmodels.stats.multitest import multipletests

from cypmode.challenge import final_tdi_build as final
from cypmode.challenge.descriptor_ratio import Final
from cypmode.challenge.feature_store import _ALERT_SMARTS, FeatureStore
from cypmode.challenge.nested import (
    META_MAX_ITER,
    META_TOL,
    StackedLogit,
    Task,
    _weights,
    score_metric,
)
from cypmode.challenge.nested_runners import (
    FEATURE_SETS,
    LGBM_PRESETS,
    default_candidates,
    fit_predict,
)
from cypmode.challenge.tdi_targets import (
    BOTH_ARMS,
    NO_DIRECT,
    label_from_shift,
    training_view,
)
from cypmode.metrics.motifs import TYPE_II_MOTIFS

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

OUT = Path("data/tdi_track/outputs/final")
COMPARISONS = Path("data/tdi_track/outputs/comparisons")
EXTERNAL = Path("data/tdi_track/outputs/external_dataset")
POST = Path("data/post_deadline/outputs")
LOGS = Path("data/structure_track/outputs/cyp3a4/logs")
RUNNERS = ("cypmode/challenge/nested_runners_torch.py", "cypmode/challenge/nested_runners_chemprop.py", "cypmode/challenge/multitask.py")
ISOFORMS = ("CYP3A4", "CYP2D6")
ALERT_COLUMNS = [*_ALERT_SMARTS, "macrocycle_ring_of_12_or_more", "largest_ring_size"]
MOTIF_COLUMNS = [*TYPE_II_MOTIFS, "any_motif"]
TOP_FEATURES = 12
ALPHA = 0.05


def pending(job: str) -> str:
    return f"(pending: {job})"


def module_constants(path: str | Path) -> dict[str, object]:
    """The upper-case constants of a module that are plain literals, read from its source without importing it (the runners of the networks import torch)."""
    out: dict[str, object] = {}
    for node in ast.parse(Path(path).read_text()).body:
        targets = [node.target] if isinstance(node, ast.AnnAssign) else getattr(node, "targets", [])
        if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None:
            for target in targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    try:
                        out[target.id] = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        continue
    return out


# -- principle 1 and the curation -----------------------------------------------------------------


def curation_table() -> pd.DataFrame:
    train, test = pd.read_csv(final.TRAIN_CSV), pd.read_csv(final.TEST_CSV)
    canonical = {}
    invalid = {}
    for name, frame in (("training", train), ("blinded", test)):
        mols = [Chem.MolFromSmiles(s) for s in frame.SMILES]
        invalid[name] = int(sum(m is None for m in mols))
        canonical[name] = [Chem.MolToSmiles(m) if m is not None else "" for m in mols]
    rows = [
        ("compounds in the training file", len(train)),
        ("compounds in the blinded file", len(test)),
        ("training SMILES that RDKit cannot parse", invalid["training"]),
        ("blinded SMILES that RDKit cannot parse", invalid["blinded"]),
        ("duplicated compound names in the training file", int(train.Molecule_Name.duplicated().sum())),
        ("duplicated canonical SMILES in the training file", int(pd.Series(canonical["training"]).duplicated().sum())),
        ("blinded compounds whose canonical SMILES is in the training file", len(set(canonical["blinded"]) & set(canonical["training"]))),
        ("blinded compounds whose name is in the training file", len(set(test.Molecule_Name) & set(train.Molecule_Name))),
        ("training compounds written as more than one fragment (salts, solvates)", int(sum("." in s for s in train.SMILES))),
        ("blinded compounds written as more than one fragment", int(sum("." in s for s in test.SMILES))),
        ("label columns in the blinded file", int(sum(c.endswith("_is_TDI") for c in test.columns))),
    ]
    for iso in ISOFORMS:
        view = training_view(train, iso)
        both = view[view.population == BOTH_ARMS]
        rule = label_from_shift(both["shift"])
        rows += [
            (f"{iso}: labelled compounds with both arms measured (the pool of every model)", len(both)),
            (f"{iso}: labelled compounds without a direct measurement (another labelling convention, left out of the pool)", int((view.population == NO_DIRECT).sum())),
            (f"{iso}: compounds without a label", int((view.population == "unlabeled").sum())),
            (f"{iso}: rows of the pool where the rule s = t - max(d, 4) > log10 2 differs from the given label", int((rule != both["label"].to_numpy(dtype=float)).sum())),
            (f"{iso}: positive rate of the pool", round(float(both["label"].mean()), 4)),
        ]
    return pd.DataFrame({"check": [r[0] for r in rows], "value": pd.Series([r[1] for r in rows], dtype=object)})


# -- principle 2 ----------------------------------------------------------------------------------


def algorithm_manifest(manifest: dict) -> dict:
    """The algorithm of each final stack in machine-readable form: the members with their families, the columns they read, and their hyperparameters; the blend; the calibration; the call rule; and the seeds."""
    out: dict[str, object] = {
        "environment": manifest["environment"],
        "nested_design": manifest["seeds"],
        "blend": {"selector": "StackedLogit", "c_grid": list(StackedLogit().c_grid), "penalties": list(StackedLogit().penalties), "meta_folds": StackedLogit().meta_folds, "solver": "liblinear", "tolerance": META_TOL, "max_iter": META_MAX_ITER, "input": "logit of the empirical cdf of the inner out-of-fold score of every member"},
        "calibration": {"method": "isotonic regression on the inner out-of-fold scores of the stack", "bounds": [1e-4, 1 - 1e-4]},
        "call_rule": "the top share of the test file, a multiple of the prevalence of the scored population, is called positive",
        "label_rule": "positive when s = t - max(d, 4) > log10(2), t the TDI-arm pIC50 and d the direct-inhibition pIC50",
        "feature_sets": {name: list(blocks) for name, blocks in FEATURE_SETS.items()},
        "lightgbm_presets": LGBM_PRESETS,
        "network_runners": {path: module_constants(path) for path in RUNNERS if Path(path).exists()},
        "isoforms": {},
    }
    for iso in ISOFORMS:
        entry = manifest["isoforms"][iso]
        members = [(c, "default") for c in default_candidates()]
        for group in entry["surviving_groups"]:
            members += [(c, group) for c in final.GROUPS[group][0]()]
        out["isoforms"][iso] = {
            "stack": entry["stack"],
            "members": [{**{k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(c).items()}, "group": group} for c, group in members],
            "target_prevalence": entry["target_prevalence"],
            "call_rate_multiplier": entry["call_rate_multiplier"],
            "called_rate": entry["called_rate"],
        }
    return out


# -- principle 3 ----------------------------------------------------------------------------------


def domain_text(iso: str) -> tuple[str, str]:
    bins = pd.read_csv(OUT / "applicability_domain_bins.csv")
    compounds = pd.read_csv(OUT / "applicability_domain_compounds.csv")
    part = bins[(bins.isoform == iso) & (bins.z0 == final.Z_CUTS[0])]
    tiers = compounds[f"{iso}_domain_tier"].value_counts()
    total = len(compounds)
    shares = ", ".join(f"{int(tiers.get(t, 0))} {t}" for t in ("supported", "weak", "unsupported"))
    rows = "; ".join(f"{r.bin.replace(' - ', ' to ')}: n {r.compounds}, AUC {f'{r.auc:.3f}' if r.auc == r.auc else 'not estimated'}, {r.tier}" for r in part.itertuples() if r.compounds)
    return f"{shares} of {total} blinded compounds", rows


# -- principle 4 ----------------------------------------------------------------------------------


def goodness_of_fit(f: Final) -> pd.DataFrame:
    """Every candidate refitted on the whole pool and scored on it, on the yardstick of the nested design, next to its outer-fold score."""
    pool = f.plan.pool
    y = f.plan.labels(pool)
    task = Task(-1, 0, -3, pool, pool)
    rows = []
    for c in default_candidates():
        if f.cache.missing(c.name, False):
            continue
        fit = fit_predict(c, f.ctx, task)
        outer = f.outer_auc(f.cached_scores(c.name))
        row = {"isoform": f.iso, "candidate": c.name, "family": c.family}
        for z0 in final.Z_CUTS:
            w = _weights(y, pool, pool, f.weight_fns[z0], f.prevalence)
            row[f"fit_auc_z{z0}"] = score_metric(y, fit, w, "auc")
            row[f"outer_auc_z{z0}"] = float(np.mean(outer[z0]))
            row[f"optimism_z{z0}"] = row[f"fit_auc_z{z0}"] - row[f"outer_auc_z{z0}"]
        rows.append(row)
        logger.info("%s %s fitted on the pool", f.iso, c.name)
    return pd.DataFrame(rows)


def robustness_text(iso: str) -> str:
    choice = pd.read_csv(OUT / f"{iso}_stack_choice.csv")
    manifest = json.loads((OUT / "manifest.json").read_text())
    row = choice[choice["stack"] == manifest["isoforms"][iso]["stack"]].iloc[0]
    path = LOGS / f"permutation_control_{iso}.log"
    permuted = [line.strip() for line in path.read_text().splitlines() if "permuted:" in line] if path.exists() else []
    text = f"nested outer AUC {row['auc_z1.0']:.4f} (z0 1.0) and {row['auc_z1.96']:.4f} (z0 1.96), MCC at the transferred rate {row['mcc_z1.0']:.3f} and {row['mcc_z1.96']:.3f}"
    return text + ("; permuted labels: " + " / ".join(re.sub(r"^.*\| permuted: ", "", line) for line in permuted) if permuted else "")


def predictivity_text(iso: str) -> str:
    board = pd.read_csv(POST / "tdi_board_scores_A.csv").set_index("Endpoint")
    row = board.loc[f"{iso}_is_TDI"]
    macro = board.loc["MA"]
    text = f"on the labels released after the deadline, scored by the organisers' own code: MCC {row.MCC_mean:.3f} (standard deviation {row.MCC_std:.3f}) for {iso}; macro MCC {macro.MCC_mean:.4f}"
    placement = EXTERNAL / "placement.csv"
    if placement.exists():
        p = pd.read_csv(placement)
        p = p[p.isoform == iso]
        if len(p):
            r = p.iloc[0]
            text += f"; on the second data set (TDC {iso}_Veith, official protocol) the test AUPRC is {r.mean_test_auprc:.3f} against a published median of {r.median_published:.3f} ({r.entries} entries): {r.verdict}"
    else:
        text += f"; second data set: {pending('287_external_dataset')}"
    return text


# -- principle 5 ----------------------------------------------------------------------------------


def enrichment_table(iso: str, columns: dict[str, np.ndarray], y: np.ndarray) -> pd.DataFrame:
    """For each column that counts an alert or a motif: the TDI-positive share among the compounds that carry it (count above 0) and among those that do not, the odds ratio with a Woolf interval
    (0.5 added to each cell), Fisher's exact test, and the Holm adjustment over the columns tested. A column that fewer than 10 compounds carry, or lack, is not tested."""
    rows = []
    for name, values in columns.items():
        has = np.asarray(values) > 0
        a, b, c, d = int((has & (y == 1)).sum()), int((has & (y == 0)).sum()), int((~has & (y == 1)).sum()), int((~has & (y == 0)).sum())
        if has.sum() < 10 or (~has).sum() < 10:
            continue
        a_, b_, c_, d_ = (x + 0.5 for x in (a, b, c, d))
        log_or = np.log(a_ * d_ / (b_ * c_))
        se = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
        rows.append({"isoform": iso, "feature": name, "compounds_with": int(has.sum()), "positive_share_with": a / has.sum(), "positive_share_without": c / (~has).sum(),
                     "odds_ratio": float(np.exp(log_or)), "or_low": float(np.exp(log_or - 1.96 * se)), "or_high": float(np.exp(log_or + 1.96 * se)), "p_fisher": float(stats.fisher_exact([[a, b], [c, d]])[1])})
    table = pd.DataFrame(rows)
    table["p_holm"] = multipletests(table.p_fisher, method="holm")[1]
    return table.sort_values("odds_ratio", ascending=False).reset_index(drop=True)


def alert_enrichment(f: Final) -> pd.DataFrame:
    """The enrichment of every structural alert, every heme-binding motif, and the trialkylamine flag among the TDI-positive compounds of the pool of the isoform."""
    smiles = list(pd.read_csv(final.TRAIN_CSV).SMILES) + list(pd.read_csv(final.TEST_CSV).SMILES)
    store = FeatureStore.load_or_build(final.STORE, smiles)
    pool = f.plan.pool
    columns = {**dict(zip(ALERT_COLUMNS, store.blocks["tdi_alerts"][pool].T, strict=True)), **dict(zip(MOTIF_COLUMNS, store.blocks["motif"][pool].T, strict=True)), "trialkylamine": store.blocks["amine"][pool, 0]}
    columns.pop("largest_ring_size")
    return enrichment_table(f.iso, columns, f.plan.labels(pool))


def column_names(blocks: tuple[str, ...], store: FeatureStore) -> list[str]:
    names = []
    for b in blocks:
        width = store.blocks[b].shape[1]
        if b == "rdkit_desc":
            names += [d for d, _ in Descriptors.descList]
        elif b == "motif":
            names += [f"motif_{m}" for m in MOTIF_COLUMNS]
        else:
            names += [f"{b}_{i}" for i in range(width)]
    return names


def block_gain(f: Final) -> pd.DataFrame:
    """The share of the total gain of ``lgbm_A+fps_deep`` fitted on the pool that each block of columns carries, and the named columns with the largest gain."""
    from lightgbm import LGBMClassifier

    blocks = FEATURE_SETS["A+fps"]
    x = f.ctx.store.matrix(blocks, f.plan.pool)
    y = f.plan.labels(f.plan.pool)
    model = LGBMClassifier(verbose=-1, n_jobs=f.ctx.n_jobs, random_state=0, importance_type="gain", **LGBM_PRESETS["deep"]).fit(x, y)
    gain = model.feature_importances_.astype(float)
    names = column_names(blocks, f.ctx.store)
    block_of = np.concatenate([[b] * f.ctx.store.blocks[b].shape[1] for b in blocks])
    rows = [{"isoform": f.iso, "kind": "block", "name": b, "columns": int((block_of == b).sum()), "gain_share": float(gain[block_of == b].sum() / gain.sum())} for b in blocks]
    for i in np.argsort(-gain)[:TOP_FEATURES]:
        rows.append({"isoform": f.iso, "kind": "column", "name": names[i], "columns": 1, "gain_share": float(gain[i] / gain.sum())})
    return pd.DataFrame(rows)


def interpretation_text(iso: str, enrichment: pd.DataFrame, gain: pd.DataFrame) -> str:
    e = enrichment[enrichment.isoform == iso]
    significant = e[(e.p_holm < ALPHA) & (e.odds_ratio > 1)]
    depleted = e[(e.p_holm < ALPHA) & (e.odds_ratio < 1)]
    blocks = gain[(gain.isoform == iso) & (gain.kind == "block")].sort_values("gain_share", ascending=False)
    top = ", ".join(f"{r.feature} ({r.odds_ratio:.2f})" for r in significant.head(5).itertuples()) or "none"
    low = ", ".join(f"{r.feature} ({r.odds_ratio:.2f})" for r in depleted.head(3).itertuples()) or "none"
    return (
        f"{len(significant)} of {len(e)} alerts and motifs {'is' if len(significant) == 1 else 'are'} enriched among the positives and {len(depleted)} {'is' if len(depleted) == 1 else 'are'} depleted (Fisher, Holm p < {ALPHA}); the largest odds ratios: {top}; depleted: {low}; "
        f"share of the gain of one LightGBM model by block: " + ", ".join(f"{r.name} {r.gain_share:.0%}" for r in blocks.head(4).itertuples())
    )


def external_note() -> str:
    path = COMPARISONS / "descriptor_ratio_widths.csv"
    if not path.exists():
        return pending("288_descriptor_ratio")
    w = pd.read_csv(path)
    return f"the widest candidate reads {int(w.n_columns.max())} columns for {int(w.rows.min())} to {int(w.rows.max())} compounds; the extra-column, effective-dimension and fewer-column analyses are in descriptor_ratio.log"


# -- the table ------------------------------------------------------------------------------------


def principle_rows(manifest: dict, curation: pd.DataFrame, fit: pd.DataFrame, enrichment: pd.DataFrame, gain: pd.DataFrame) -> list[dict]:
    agreement = pd.read_csv(POST / "tdi_question2a_agreement.csv").set_index("isoform")
    value = dict(zip(curation.check, curation.value, strict=True))
    rows = []
    for iso in ISOFORMS:
        mismatches = value[f"{iso}: rows of the pool where the rule s = t - max(d, 4) > log10 2 differs from the given label"]
        pool = value[f"{iso}: labelled compounds with both arms measured (the pool of every model)"]
        a = agreement.loc[iso]
        rows.append({"principle": "1. A defined endpoint", "isoform": iso, "evidence": (
            f"time-dependent inhibition of {iso}: positive when the pIC50 after pre-incubation exceeds the direct-inhibition pIC50 (floored at 4) by more than log10 2; the rule gives the given label for {pool - mismatches} of {pool} compounds of the pool "
            f"and for all {int(a.agree)} of the {int(a.labelled)} released test labels for which both potencies are given ({int(a.disagree)} disagree, {int(a.not_computable)} cannot be computed)"), "source": "audit section 4; tdi_targets.py; post_deadline tdi_question2a_agreement.csv"})
        entry = manifest["isoforms"][iso]
        rows.append({"principle": "2. An unambiguous algorithm", "isoform": iso, "evidence": (
            f"{entry['n_candidates']} members ({entry['stack']}), a logistic blend of the inner out-of-fold scores with the penalty and its strength chosen on the inner folds, isotonic calibration, a call rate of {entry['call_rate_multiplier']} times the prevalence; "
            "every member, column set, hyperparameter, seed, and package version is in algorithm_manifest.json; the code is open, and each submitted file has a SHA-256 hash"), "source": "algorithm_manifest.json; manifest.json; nested_runners.py; final_tdi_build.py"})
        tiers, bins = domain_text(iso)
        rows.append({"principle": "3. A defined domain of applicability", "isoform": iso, "evidence": f"reliability by the nearest-training similarity of a compound (Tanimoto similarity of extended-connectivity fingerprints of diameter 4, ECFP4), bins supported when the lower end of a 95% interval over whole Butina clusters reaches AUC 0.60; {tiers}; {bins}", "source": "applicability_domain.md; applicability_domain.py"})
        sub = fit[fit.isoform == iso]
        fit_text = (
            f"goodness of fit: the {len(sub)} candidates refitted on the pool score a mean AUC of {sub['fit_auc_z1.0'].mean():.3f} on the pool against {sub['outer_auc_z1.0'].mean():.3f} on held-out folds (z0 1.0), an optimism of {sub['optimism_z1.0'].mean():.3f} that the nested design removes"
            if len(sub) else f"goodness of fit {pending('289_oecd_principles')}"
        )
        rows.append({"principle": "4. Goodness of fit, robustness, and predictivity", "isoform": iso, "evidence": f"{fit_text}; robustness: {robustness_text(iso)}; ratio of columns to compounds: {external_note()}; predictivity: {predictivity_text(iso)}", "source": "goodness_of_fit.csv; nested results of the final build; permutation_control logs; post_deadline outputs; external_dataset.log; descriptor_ratio.log"})
        rows.append({"principle": "5. A mechanistic interpretation, if possible", "isoform": iso, "evidence": interpretation_text(iso, enrichment, gain), "source": "alert_enrichment.csv; block_gain.csv; feature_store.py (the alerts)"})
    return rows


def markdown(rows: list[dict], curation: pd.DataFrame) -> str:
    lines = ["# The five OECD principles and the curation of the data (generated by `cypmode/challenge/oecd_principles.py`)", "",
             "Each cell is read from a file of the project or computed by the module; the sources are named in the last column. The cross-validated numbers are estimates on the training data and not scores on the hidden labels, except where a row says that it reads the released labels.", "",
             "Abbreviations: AUC, area under the receiver operating characteristic curve; AUPRC, area under the precision–recall curve; ECFP4, extended-connectivity fingerprints of diameter 4; MCC, Matthews correlation coefficient; TDI, time-dependent inhibition; z0, the cut on the confidence of the label (1.0 or 1.96).", "",
             "| principle | isoform | evidence | source |", "| --- | --- | --- | --- |"]
    lines += [f"| {r['principle']} | {r['isoform']} | {r['evidence'].rstrip('.')}. | {r['source']} |" for r in rows]
    lines += ["", "## Curation of the data", "", "| check | value |", "| --- | --- |"] + [f"| {r.check} | {r.value} |" for r in curation.itertuples()] + [""]
    return "\n".join(lines)


def main() -> None:
    from cypmode import provenance

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", choices=ISOFORMS, help="one isoform (both by default)")
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    provenance.announce()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((OUT / "manifest.json").read_text())
    curation = curation_table()
    curation.to_csv(OUT / "data_curation.csv", index=False)
    (OUT / "algorithm_manifest.json").write_text(json.dumps(algorithm_manifest(manifest), indent=2, sort_keys=True) + "\n")
    fits, enrichments, gains = [], [], []
    for iso in [args.isoform] if args.isoform else ISOFORMS:
        f = Final(iso, args.n_jobs)
        fits.append(goodness_of_fit(f))
        enrichments.append(alert_enrichment(f))
        gains.append(block_gain(f))
    fit, enrichment, gain = (pd.concat(x, ignore_index=True) for x in (fits, enrichments, gains))
    fit.to_csv(OUT / "goodness_of_fit.csv", index=False, float_format="%.6f")
    enrichment.to_csv(OUT / "alert_enrichment.csv", index=False, float_format="%.6g")
    gain.to_csv(OUT / "block_gain.csv", index=False, float_format="%.6f")
    rows = principle_rows(manifest, curation, fit, enrichment, gain)
    pd.DataFrame(rows).to_csv(OUT / "oecd_principles.csv", index=False)
    (OUT / "oecd_principles.md").write_text(markdown(rows, curation))
    print(markdown(rows, curation))
    for iso in ISOFORMS:
        e = enrichment[enrichment.isoform == iso]
        print(f"\n{iso}: alerts and motifs by odds ratio (compounds with, positive share with and without, odds ratio, 95% interval, Holm p)")
        for r in e.head(8).itertuples():
            print(f"  {r.feature:34s} {r.compounds_with:5d}  {r.positive_share_with:.2f} / {r.positive_share_without:.2f}  {r.odds_ratio:5.2f} ({r.or_low:.2f} to {r.or_high:.2f})  p {r.p_holm:.3g}")


if __name__ == "__main__":
    main()
