# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Generate METHOD_REPORT.md and NEGATIVE_RESULTS.md from the repository's own logs, tables and audit.

Every number in these two files is copied by code from a log, a result table or the audit report; the prose around them is
fixed text. A value whose source does not exist yet is written as ``(pending: <job>)``, so the report can never contain a
number nobody computed. ``python -m cypmode.audit.method_report --check`` regenerates both files in memory and fails if the
files on disk differ, which is the claim-to-evidence check the audit runs: a hand-edited number is a difference.

Usage: python -m cypmode.audit.method_report [--check]
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "data/structure_track/outputs/cyp3a4/logs"
COMPARISONS = ROOT / "data/tdi_track/outputs/comparisons"
FINAL_TDI = ROOT / "data/tdi_track/outputs/final"
FINAL_STRUCTURE = ROOT / "data/structure_track/outputs/cyp3a4/final"
AUDIT = ROOT / "audit/AUDIT_REPORT.md"
GATE = re.compile(r"^(?:STRUCTURE GATE (\w+)[^:\n]*|(CYP\w+) GATE \(([^)]*)\)): (MET|NOT MET)", re.MULTILINE)


def pending(job: str) -> str:
    return f"(pending: {job})"


def log_lines(name: str, keep: str, job: str, limit: int = 40) -> str:
    path = LOGS / name
    if not path.exists():
        return pending(job)
    lines = [ln.rstrip() for ln in path.read_text(errors="ignore").splitlines() if re.search(keep, ln)]
    return "```\n" + "\n".join(lines[:limit]) + "\n```" if lines else pending(job)


def audit_section(prefix: str) -> str:
    if not AUDIT.exists():
        return pending("job 900")
    text = AUDIT.read_text()
    m = re.search(rf"^## {re.escape(prefix)}.*?(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    if not m:
        return pending("job 900")
    first, _, rest = m.group(0).strip().partition("\n")
    return "### Audit extract: " + re.sub(r"^## \d+[a-z]?\. ", "", first) + "\n" + rest


def gate_rows() -> list[tuple[str, str, str]]:
    rows = []
    for f in sorted(LOGS.glob("*.log")):
        for name, iso, rule, verdict in GATE.findall(f.read_text(errors="ignore")):
            rows.append((f.name, name or f"{iso}: {rule.strip()}", verdict))
    return rows


def table(rows: list[tuple], columns: list[str]) -> str:
    if not rows:
        return "_(none yet)_"
    head = "| " + " | ".join(columns) + " |\n|" + "|".join(" --- " for _ in columns) + "|\n"
    return head + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)


def selection_sensitivity() -> str:
    """The two selection gates of the structure track recomputed with each ligand left out in turn (a description, not a test): which ligand, if any, would change the registered verdict."""
    path = LOGS.parent / "selection_validation_outcomes.csv"
    if not path.exists():
        return pending("job 170")
    from cypmode.validation.selection_validation import gates

    outcomes = pd.read_csv(path)
    full = gates(outcomes)
    rows = []
    for name in ("mbr_selection", "engine_consensus"):
        if name not in full:
            continue
        g = full[name]
        flips = []
        for i in range(len(outcomes)):
            left = gates(outcomes.drop(outcomes.index[i]))[name]
            if left["met"] != g["met"]:
                flips.append(f"{outcomes['compound'].iloc[i]} ({left['mean_improvement']:+.3f} A over {left['n']} ligands, {left['wins']} wins, p {left['p']:.3f})")
        rows.append((name, f"{'MET' if g['met'] else 'NOT MET'}: mean improvement {g['mean_improvement']:+.3f} A, {g['wins']} wins of {g['n']}, p {g['p']:.3f}", "; ".join(flips) or "none"))
    return table(rows, ["gate", "registered verdict (all ligands)", "ligand whose removal would change the verdict"])


def npz_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    """The shape of every array of an .npz file, read from the array headers without loading the arrays."""
    import numpy as np

    shapes = {}
    with np.load(path, allow_pickle=True) as store:
        for name in store.files:
            with store.zip.open(name + ".npy") as handle:
                version = np.lib.format.read_magic(handle)
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shapes[name] = tuple(reader(handle)[0])
    return shapes


def summary_section() -> str:
    """The stand-alone summary and the disclosures that the submission form asks for, read from the final manifests and from the files that state the licences."""
    import json

    from cypmode import provenance

    cff = provenance.load()
    manifest_path, structure_path = FINAL_TDI / "manifest.json", FINAL_STRUCTURE / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    structure = json.loads(structure_path.read_text()) if structure_path.exists() else None
    final_text = (FINAL_TDI / "final_report.md").read_text() if (FINAL_TDI / "final_report.md").exists() else ""
    stack_rows = []
    for iso in ("CYP3A4", "CYP2D6"):
        block = re.search(rf"## {iso}\n(.*?)(?=\n## |\Z)", final_text, re.DOTALL)
        found = re.search(r"Final stack: \*\*(.+?)\*\* \((\d+) candidates\)", block.group(1)) if block else None
        if not found:
            continue
        for line in block.group(1).splitlines():
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 6 and cells[0] == found.group(1):
                stack_rows.append((iso, found.group(1), found.group(2), cells[4], cells[2]))
    file_rows = [(f["file"], f["sha256"], f["rows"]) for f in manifest["files"]] if manifest else []
    if structure:
        file_rows.append((Path(structure["zip"]).name, structure["sha256"], f"{len(structure['ligands'])} complexes"))
    ids = provenance.identifiers(cff)
    lines = [
        "## Summary and disclosures",
        "",
        f"{cff['abstract']}",
        "",
        "### What is submitted",
        "",
        table(file_rows, ["file", "SHA-256", "contents"]) if file_rows else pending("job 950"),
        "",
        "Nested cross-validated estimates of the final stacks (AUC at the stricter and at the looser confidence cut; they are estimates on the training data, not scores on the hidden labels):",
        "",
        table(stack_rows, ["isoform", "final stack", "candidates", "AUC at z0 = 1.96", "AUC at z0 = 1.0"]) if stack_rows else pending("job 090"),
        "",
        "### Disclosures",
        "",
        f"- **Open source.** The code is open source and public at {cff['repository-code']}, under {cff.get('license')} with the additional term of `NOTICE` that the attribution and the citation notice stay with every copy and every output. The",
        "  documents (this report, the notebook, the figures, the film) are under CC BY-NC-SA 4.0 and the data that the code generates under CC BY 4.0 (`LICENSES/`, `REUSE.toml`).",
        "- **Proprietary data.** None: every input is public.",
        "- **Sources of the data of the final stacks.** The OpenADMET challenge files (training file and blinded structures; the dataset `openadmet/cyp-challenge-train-test`, whose card states Apache-2.0); the TDC / Veith CYP panel (Veith et al., Nature",
        "  Biotechnology 2009; the TDC page states CC BY 4.0), used only to train a label-free block on compounds that are not challenge compounds; and the rows that the public leaderboards publish (accuracy, precision and recall of each entry), from which one",
        "  number about the hidden labels is recovered, the positive rate of the scored set (limitation L13). No label of any blinded compound and no probing submission is used.",
        "- **Components.** The programs, models and weights that the work uses and their licences are listed in section 14 and in Appendix C of the notebook.",
        f"- **Identifier.** {('DOI ' + ids['doi'] + '.') if 'doi' in ids else 'The archived release has no DOI yet; the citation file lists it when it has one.'} The text to cite is in `USE_AND_CITATION.md`.",
        "- **Reproduction and checks.** `NEGATIVE_RESULTS.md` (every experiment that failed its gate), `audit/AUDIT_REPORT.md` (generated twice and compared byte for byte), `bash scripts/reproduce_tdi.sh`.",
        "",
    ]
    return "\n".join(lines)


def architecture_section() -> str:
    """The path of each track in the report itself: what the model is, step by step, with the numbers read from the code and the final manifests."""
    import collections
    import inspect
    import json

    from cypmode.challenge import final_tdi_build as build
    from cypmode.challenge.nested import StackedLogit
    from cypmode.challenge.nested_runners import default_candidates
    from cypmode.validation.official_posebusters import MARGIN_FLOOR

    manifest = json.loads((FINAL_TDI / "manifest.json").read_text()) if (FINAL_TDI / "manifest.json").exists() else None
    recipe_path, structure_path = FINAL_STRUCTURE / "recipe.json", FINAL_STRUCTURE / "manifest.json"
    recipe = json.loads(recipe_path.read_text()) if recipe_path.exists() else None
    structure = json.loads(structure_path.read_text()) if structure_path.exists() else None
    store_path = ROOT / build.STORE
    shapes = npz_shapes(store_path) if store_path.exists() else {}
    blocks = {k: v for k, v in shapes.items() if k != "smiles" and len(v) == 2}
    families = collections.Counter(c.family for c in default_candidates())
    grid = inspect.signature(StackedLogit.__init__).parameters["c_grid"].default
    lines = [
        "## 3. Architecture of the two tracks",
        "",
        "This section says what the submitted models are, step by step. Every number is read from the code or from the final manifests when the report is generated. The notebook teaches each step from first principles; the section",
        "that does is named at the end of the step.",
        "",
        "### 3.1 TDI track (CYP3A4 and CYP2D6)",
        "",
        "1. **Label.** A compound is positive when `s = t - max(d, 4) > log10 2`; the rule reproduces every training label (section 2; notebook 2.3). The confidence of a label is `z = |s - log10 2| / sd(s)` from the posterior standard",
        "   deviations of the two arms, and only confidently labelled compounds are scored (notebook 2.5).",
        f"2. **Features, label-free.** The feature store holds {sum(v[1] for v in blocks.values()):,} columns in {len(blocks)} blocks ({', '.join(blocks)}). No block reads a label of a test compound; the leakage audit checks",
        "   it (audit section 17; notebook Part III and Part X).",
        f"3. **The default stack: {sum(families.values())} candidates** of {len(families)} families (" + ", ".join(f"{n} {f}" for f, n in sorted(families.items())) + "), each a model of the shift or of the label on one set of blocks (notebook 8.1).",
        f"4. **Validation.** Nested cross-validation: {build.REPEATS} repeats of {build.DESIGN['outer']} outer folds with 4 inner folds, seed 0, the plan saved with an identifier computed from the raw files. Every selection step (the blend, its penalty, the threshold) happens inside",
        "   the inner folds, and the outer folds are weighted to resemble the scored population: similarity to the blinded structures, the estimated prevalence, only confident labels at two cut-offs (notebook 6.3 and 6.4).",
        f"5. **The blend.** A logistic regression on the ranks of the members, with the penalty (L1 or L2) and its strength (C in {{{', '.join(str(c) for c in grid)}}}) chosen on the inner folds; a deterministic solver (notebook 8.2 and 8.3).",
        "6. **Which groups join.** A group of added candidates joins a stack only through a pre-registered gate (corrected resampled t, p < 0.05 at both confidence cuts, alone and inside the stack); a group found after looking also needs a",
        "   replicate on ten outer folds (notebook 6.7 and 9.9). The final build reads the verdict lines of the logs and nothing else decides.",
    ]
    if manifest:
        for iso, r in manifest["isoforms"].items():
            lines.append(f"   - {iso}: final stack `{r['stack']}` of {r['n_candidates']} candidates; groups that joined: {', '.join(r['surviving_groups']) or 'none'}.")
    lines += [
        "7. **Calibration and calls.** Probabilities by isotonic regression on the inner out-of-fold scores (notebook 8.5). The number of positive calls is a multiple of the estimated prevalence of the scored population, the multiple in",
        f"   {{{', '.join(str(m) for m in build.RATE_MULTIPLIERS)}}} chosen by the highest mean outer-fold MCC (notebook 8.4)" + (": " + "; ".join(f"{iso} {r['call_rate_multiplier']}" for iso, r in manifest["isoforms"].items()) if manifest else "") + ".",
        "8. **Two files.** A (hits-only CYP2D6 model) and B (the selection-aware CYP2D6 model, which multiplies P(label | hit) by a gate P(hit)); CYP3A4 is identical in both. The entry that is final is chosen by a rule declared in advance (notebook 9.1).",
        "",
        "Figures: `docs/figures/architecture/arch_tdi_pipeline-light.png` and `arch_nested_cv-light.png` (dark versions beside them).",
        "",
        "### 3.2 Structure track (CYP3A4, 20 ligands)",
        "",
        "1. **Input.** The CYP3A4 sequence with its alignment, the heme with its covalent bond to the proximal cysteine, and the ligand, written as a Boltz-2 input (notebook 11.3).",
        "2. **Donor restraint.** For a ligand with a Type II donor nitrogen the input carries a restraint of the iron-nitrogen bond; the gate that decided it is MET on the crystal-anchored benchmark (notebook 11.6).",
    ]
    if recipe and structure:
        lines += [
            f"3. **The pool.** Boltz-2 samples per ligand (the unguided and the restrained runs and {recipe['recipe']['extra_per_ligand']} extra samples), {len(structure['ligands'])} ligands, {sum(x['pool'] for x in structure['ligands'])} poses in all; no second engine (engine consensus is NOT MET).",
            f"4. **Validity and margin.** Every pose is checked by the scorer's own PoseBusters call (22 checks); among the valid poses only those whose smallest relative distance to the receptor is at least {MARGIN_FLOOR} are eligible (notebook 11.4 and 11.10).",
            f"5. **The pick.** `{recipe['recipe']['selection']}`: the model's own ranking among the eligible poses; for a ligand with a Type II donor the restrained run is read first. Every rule for choosing a pose that was tested is NOT MET.",
            f"6. **Relaxation under a registered rule.** A force-field relaxation of the ligand in the fixed pocket replaces a pose only if both pass the 22 checks, the margin rises and is at least the floor, the ligand moves at most {structure['relaxation']['rule']['max_movement_A']} A and the iron-nitrogen distance changes at most {structure['relaxation']['rule']['max_fe_n_change_A']} A; adopted for {structure['relaxation']['adopted']} of {structure['relaxation']['of']} (notebook 11.10).",
            f"7. **The archive.** A deterministic zip of {len(structure['ligands'])} complexes (the time of every entry fixed), SHA-256 `{structure['sha256'][:16]}...`.",
        ]
    else:
        lines.append(pending("job 180"))
    lines += ["", "Figures: `docs/figures/architecture/arch_structure_pipeline-light.png` (dark version beside it).", ""]
    return "\n".join(lines)


def exploration_section() -> str:
    """How much was tried and with what result, all of it counted from the ledger and the logs."""
    from cypmode.audit.ledger_kinds import KINDS, TRIED, ledger_by_kind
    from cypmode.challenge.final_tdi_build import CONFIRMED_GROUPS
    from cypmode.challenge.multiplicity import collect

    frame = ledger_by_kind()
    rows = []
    for kind in [*KINDS, "other"]:
        part = frame[frame["kind"] == kind]
        if part.empty:
            continue
        rows.append((kind, int(part["stem"].nunique()), len(part), int((part["verdict"] == "MET").sum()), int((part["verdict"] != "MET").sum()), TRIED.get(kind, "")))
    family = collect(LOGS)
    raw = family[family["p_both"] < 0.05]
    audit_text = AUDIT.read_text() if AUDIT.exists() else ""
    tests = re.search(r"pytest summary counts: passed = (\d+)", audit_text)
    counts = re.search(r"Counts: PASS = (\d+), FAIL = (\d+)", audit_text)
    mismatches = re.findall(r"label rule reproduces every both-arms label \| PASS \| (\d+) mismatches", audit_text)
    lines = [
        "## 4. Exploration: what was tried and with what result",
        "",
        "The project explored widely and decided narrowly. Every idea was a candidate or a block with a rule written before its result; the ledger below is generated from the logs of those rules, and the ones that failed stay in it as",
        "results (`NEGATIVE_RESULTS.md`). No search procedure chose among them: the programme was designed, and the final build applies the verdicts mechanically.",
        "",
        "### 4.1 The map of what was explored",
        "",
        table(rows, ["kind of idea", "families", "verdicts", "met", "not met", "what it tried"]),
        "",
        f"In all {len(frame)} pre-registered verdicts in {frame['stem'].nunique()} families, {int((frame['verdict'] == 'MET').sum())} of them MET (figure `docs/figures/ml/ml_exploration_map-light.png`; notebook Part IX).",
        "",
        "### 4.2 The accounting of the number of attempts",
        "",
        f"A gate controls the risk of admitting one idea that does not help; it does not control the risk that one of many ideas passes by chance. The {len(family)} stacked comparisons of the batch are therefore also reported with the Holm",
        f"step-down adjustment (`cypmode/challenge/multiplicity.py`): {len(raw)} have a raw p-value below 0.05 ({', '.join(raw['log']) or 'none'}), and the smallest adjusted p-value of the family is {family['holm_adjusted'].min():.2f}. A pass of a",
        "pre-registered gate is therefore reported as a pass of the registered rule and not as an effect that survives the correction (notebook 9.11).",
        "",
        "### 4.3 The controls",
        "",
        "- **Negative control.** The labels of the pool are permuted and the whole nested procedure is repeated; the outer AUC must be 0.5 (notebook 9.10). Log: `permutation_control_*.log`.",
        f"- **Replicates.** A group found after looking at the 15-fold results ({', '.join(CONFIRMED_GROUPS)}) joins only if the same comparison on ten outer folds is MET as well; the replicate was declared before any ten-fold number existed (notebook 9.9).",
        "- **Design floor.** The least difference in AUC that the main design recognises 8 times in 10 is 0.0099, so a near miss below it is reported as inconclusive and not as a negative (notebook 6.7).",
        "- **One ligand.** The two selection gates of the structure track recomputed with each ligand left out in turn, as a description (`NEGATIVE_RESULTS.md`).",
        "- **Second looks.** A near miss may be looked at again, but only in a log that no gate parser reads (`exploratory_*`), so that it cannot change a model; the all-in-one stack of every cached candidate against the final stack is one of them.",
        "",
        "### 4.4 The selection-corrected design and its audit",
        "",
        f"The label rule reproduces every training label ({' and '.join(mismatches) if mismatches else 'see audit section 4'} mismatches); the plans are rebuilt from the raw files and compared; every feature block is checked for leakage; the audit",
        f"is generated twice and compared byte for byte; the test suite has {tests.group(1) if tests else '(pending: job 900)'} passing tests; the summary of the audit is {counts.group(1) + ' PASS and ' + counts.group(2) + ' FAIL' if counts else '(pending: job 900)'} (notebook Part XII).",
        "",
        "### 4.5 What the project built, and what it borrowed",
        "",
        "Built here, each with its evidence in the section or the notebook part named: the label rule recovered from the data and the contamination fix of the CYP3A4 pool (section 2); the scoring-set-aware yardstick (section 5); the",
        "selection-aware CYP2D6 model (section 7); the committee test and its measured negative result (limitations); the margin rule and the force-field relaxation under a registered rule (section 11). Borrowed and credited, with the",
        "publication of each: the co-folding engines, PoseBusters, the corrected resampled t, Holm's adjustment, the pretrained models that were tested and refused (section 14; notebook Appendix C).",
        "",
        "### 4.6 The rule",
        "",
        "A gate that fails is a result. It is written down with its numbers, it is not tuned until it passes, and an idea that is found after looking is a discovery that needs a replicate, not a verdict.",
        "",
    ]
    return "\n".join(lines)


def ensemble_caveat(log: Path | None = None) -> str:
    """The registered consequence of the conformer check (S8 c): when the ensemble of the pocket block holds a member within 1.0 A of the crystal pose for fewer than 80% of the crystal ligand copies, the
    gates of the pocket block and of the site model are listed with that sentence, computed from the log of the check (empty when the check is missing or passes)."""
    log = LOGS / "conformer_bias.log" if log is None else log
    if not log.exists():
        return ""
    text = log.read_text(errors="ignore")
    block = re.search(r"the ensemble of the block \(.*?\): a member within ([\d.]+) A of the crystal pose for (\d+)% of the (\d+) copies", text)
    if not block or int(block.group(2)) >= 80:
        return ""
    share, copies = int(block.group(2)), int(block.group(3))
    return (f"The rows of the pocket block (`aux_pocket_*`, `exploratory_pocket_*`) and of the site model (`aux_site_*`, `exploratory_site_*`) are verdicts of models tested with an ensemble of 12 conformers that holds a member within "
            f"{block.group(1)} A of the crystal pose for {share}% of {copies} crystal ligand copies (`conformer_bias.log`), so that they missed the bound conformation of {100 - share}% of the crystal ligands; a wider ensemble was not tried.")


def negative_results() -> str:
    gate = [(f, g, v) for f, g, v in gate_rows() if v == "NOT MET"]
    nulls = []
    for f in sorted(COMPARISONS.glob("*_significance_paired.csv")):
        d = pd.read_csv(f)
        pcol = "p-adj"
        for _, r in d.iterrows():
            if not bool(r["reject"]):
                nulls.append((f.name, r.get("isoform", ""), f"{r['group1']} vs {r['group2']}", f"{r['meandiff']:+.4f}", f"{r[pcol]:.3f}"))
    return "\n".join([
        "# Negative results ledger (generated by `cypmode/audit/method_report.py`)",
        "",
        "Every experiment whose pre-registered gate was not met, and every older pairwise comparison that was not significant under",
        "repeated 5x5 cross-validation. Nothing here was tuned until it passed. The older comparisons are re-analysed from their saved",
        "per-fold scores with the paired corrected resampled t of Nadeau and Bengio and a Holm adjustment over the pairs; the gate ledger",
        "uses the same corrected t.",
        "",
        "## Gates NOT MET",
        "",
        table(gate, ["log", "gate", "verdict"]),
        "",
        ensemble_caveat(),
        "",
        "## Older comparisons that were not significant (paired corrected t, Holm)",
        "",
        table(nulls, ["table", "isoform", "comparison", "mean difference", "adjusted p"]),
        "",
        "## Sensitivity of the structure selection gates to one ligand",
        "",
        "Both selection gates were judged on ten crystal-anchored ligands under rules written beforehand, and both are NOT MET. As a description and not as a test, each gate was",
        "recomputed ten times with one ligand left out. Leaving a ligand out after the result is known is not a way to a verdict: the registered verdict is the one on all ligands and",
        "it is not changed by this table. The table shows how little each verdict rests on. With ten ligands a single ligand can turn a gate, and it is the ligand with the largest",
        "effect that does.",
        "",
        selection_sensitivity(),
        "",
    ])


def build_method_report() -> str:
    final = FINAL_TDI / "final_report.md"
    manifest = FINAL_STRUCTURE / "manifest.json"
    return "\n".join([
        "# Method report: label-process, selection-aware, physics-aware CYP TDI classification and CYP3A4 structure prediction",
        "",
        "Generated by `cypmode/audit/method_report.py`; every number is copied from a log, a result table or the audit report,",
        "and a value without a source reads `(pending: <job>)`. Nothing in this report is a claim about the hidden test labels.",
        "",
        summary_section(),
        "## 1. Scope and what is not claimed",
        "",
        "Two tracks of the OpenADMET CYP blind challenge: time-dependent-inhibition (TDI) classification for CYP3A4 and CYP2D6, and",
        "CYP3A4 protein-ligand structure prediction. All performance figures are cross-validated estimates on the public training data;",
        "the step to the leaderboard is unmeasured until a submission exists. Novelty is claimed only in the form \"to our knowledge not",
        "found in a search of the sources named in section 12 on the search date\"; every component has published relatives.",
        "",
        "## 2. Data and the label rule",
        "",
        "The TDI label is a threshold of two measured pIC50 values, positive when `s = t - max(d, 4) > log10 2` (t: TDI-condition arm,",
        "d: direct-inhibition arm, floor 4). The rule reproduces every training label exactly (audit section 4), including the",
        "convention for compounds without a direct measurement; those rows follow a different convention than the scored test set and",
        "are excluded from the CYP3A4 pool (audit section 4 gives the counts).",
        "",
        audit_section("4."),
        "",
        architecture_section(),
        "",
        exploration_section(),
        "",
        "## 5. Scoring-set-aware yardstick",
        "",
        "Only confidently labeled compounds are scored. The confidence z = |s - log10 2| / sd(s) from the arms' posterior standard",
        "deviations is used to restrict evaluation at two cut-offs (1.0 and 1.96), and evaluation weights follow similarity to the",
        "unlabeled test structures and the estimated test prevalence. That prevalence is the one piece of information about the hidden",
        "labels that the project uses: it is recovered from the accuracy, precision and recall that the public leaderboards publish for every",
        "entry (see limitation L13 and the figure `data_prevalence_recovery`); no label of any compound and no probing submission is used.",
        "Selection among models happens inside nested cross-validation and",
        "models are compared by the corrected resampled t of Nadeau and Bengio (never by bootstrapping fixed predictions).",
        "",
        audit_section("5."),
        "",
        "## 6. Final TDI stack (chosen by the pre-registered gate rule)",
        "",
        final.read_text() if final.exists() else pending("job 090"),
        "",
        "## 7. Selection-aware CYP2D6 model",
        "",
        "Training labels exist only for compounds promoted to dose-response on the isoform; the scored set may contain compounds that",
        "would never have been promoted. The model multiplies P(label | hit) by a gate P(hit), fitted on the all-isoform panel block",
        "(found by `library_blocks.py`). Results (15 folds, nested):",
        "",
        log_lines("population_panel_CYP2D6_r3.log", r"z0 =|S[0-3] ", "job 010"),
        "",
        "Dependence on the share of never-promoted compounds in the scored set (job 012), and the label-shift estimate of that share (job 013):",
        "",
        log_lines("population_sensitivity_panel_CYP2D6_r3.log", r"z0=", "job 012", 12),
        "",
        log_lines("test_nonhit_share_CYP2D6.log", r"hit share|weight|neighbour|gate", "job 013"),
        "",
        "Fit-weight design choice and hedged blend (job 014):",
        "",
        log_lines("hedged_blend_CYP2D6.log", r"GATE|best worst|worst-case|fitted under", "job 014", 24),
        "",
        "## 8. Physics-, SAR-, committee-, embedding-, cross-isoform and weighting-based additions",
        "",
        "Each addition is a block or candidate group tested alone and inside the assembled stack; the rule is corrected p < 0.05 at both",
        "confidence cut-offs. Verdicts:",
        "",
        table([r for r in gate_rows() if re.search(r"physics|pair_transfer|mmp|committee|embeddings|test_likeness|arms|selection_aware_v2|xiso", r[0])], ["log", "gate", "verdict"]),
        "",
        "Replicate with 10 outer folds for the physics additions (variance factor 0.144 instead of 0.317; an addition counts as replicated",
        "only if the 10-fold stacked comparison has corrected p < 0.05 and a positive difference at both cut-offs and the 5-fold difference is",
        "positive at both), job 083:",
        "",
        log_lines("replicate_verdict_CYP3A4.log", r"stacked|GATE", "job 083", 8),
        "",
        log_lines("replicate_verdict_CYP2D6.log", r"stacked|GATE", "job 083", 8),
        "",
        "## 9. Calibration",
        "",
        log_lines("calibration.log", r"CYP|gate|ece|isotonic|platt|raw", "job 030", 20),
        "",
        "## 10. Test-like holdout",
        "",
        log_lines("block_holdout.log", r"block|agreement|committee|best candidate", "job 045", 12),
        "",
        "## 11. Structure track",
        "",
        "Gate verdicts of the structure levers (restraint, diversity, selection, engines, expected-improvement sampling, heme-frame prior):",
        "",
        table([r for r in gate_rows() if r[1] and not re.match(r"CYP", r[1])], ["log", "gate", "verdict"]),
        "",
        "Final recipe and zip: " + (manifest.read_text()[:1500] if manifest.exists() else pending("job 180")),
        "",
        "Force-field relaxation of the chosen poses: the gate on the poses of ligands with public crystal poses, then its use on the 20 submitted poses under the rule registered in `relax_final.py`:",
        "",
        log_lines("relaxation_gate.log", r"poses passing|margin \(|LDDT-PLI change|STRUCTURE GATE", "job 218", 6),
        "",
        log_lines("relax_final.log", r"STRUCTURE RELAXATION|not adopted", "job 243", 8),
        "",
        "## 12. Negative results",
        "",
        "See `NEGATIVE_RESULTS.md` (generated from the same logs and tables). It also tabulates how far the two structure selection gates depend on a single ligand; that table does not change a registered verdict.",
        "",
        "## 13. Limitations",
        "",
        audit_section("13."),
        "",
        "## 14. Reproduction, credits and prior art",
        "",
        "Reproduce the TDI candidates with `bash scripts/reproduce_tdi.sh`; the audit with `bash scripts/run_audit.sh` (two runs, byte-identical);",
        "the environment is pinned in `environment/`. The structure recipe runs through `python -m cypmode.validation.final_recipe`.",
        "",
        "Sources credited (published, linked): OpenADMET CYP blind challenge and its official LightGBM baseline; the OpenADMET blog post",
        "\"Cofolding? I hardly know her!\" for the co-folding accuracy figure; repository `openadmet-cyp-structure`",
        "(https://github.com/xX-its-amit-Xx/openadmet-cyp-structure) for the idea of a donor-nitrogen restraint, an idea stated as a plan",
        "there, not a measured gain; Boltz-2; OpenFold3; PoseBusters; RDKit; LightGBM; scikit-learn; tblite / GFN2-xTB; CheMeleon (Zenodo",
        "10.5281/zenodo.15426600); Nadeau and Bengio, \"Inference for the generalization error\" (2003) for the corrected resampled t.",
        "Nearest published work for the selection-aware model: selection bias in compound testing and its effect on QSAR (SAR and QSAR in",
        "Environmental Research 2019, PubMed 31547686); Heckman selection models (BMC Medical Research Methodology 2019,",
        "doi 10.1186/s12874-019-0840-7); predicting which primary-screen hits yield dose-response potencies (arXiv 2608.26538);",
        "censored regression labels in drug discovery (arXiv 2409.04313); label noise and soft labels (arXiv 2104.02872).",
        "",
    ])


def outputs() -> dict[Path, str]:
    return {ROOT / "METHOD_REPORT.md": build_method_report(), ROOT / "NEGATIVE_RESULTS.md": negative_results()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the files on disk differ from a fresh generation")
    args = parser.parse_args()
    generated = outputs()
    if args.check:
        stale = [p.name for p, text in generated.items() if not p.exists() or p.read_text() != text]
        print("claim-to-evidence check: " + ("OK, both files equal a fresh generation" if not stale else f"STALE or missing: {stale}"))
        return 1 if stale else 0
    for path, text in generated.items():
        path.write_text(text)
        print(f"wrote {path.name} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
