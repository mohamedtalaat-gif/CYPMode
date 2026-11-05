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
import json
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
LEADERBOARD = ROOT / "docs/leaderboard_scores.csv"
STRUCTURE_BOARD = ROOT / "docs/structure_track/structure_board_scores.csv"
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


def leaderboard_scores() -> str:
    """The scores that the portal gave the uploaded files, as recorded by hand in ``docs/leaderboard_scores.csv`` (one row per upload); nothing is printed while the file does not exist."""
    if not LEADERBOARD.exists():
        return ""
    frame = pd.read_csv(LEADERBOARD)
    rows = [tuple("" if pd.isna(v) else v for v in r) for r in frame.itertuples(index=False)]
    return "\n".join([
        "Scores that the portal gave the uploaded files (the live board scores a part of the test set; `docs/leaderboard_scores.csv`, copied from the board and the receipts of the portal):",
        "",
        table(rows, list(frame.columns)),
        "",
    ])


def recipe_summary(manifest_path: Path) -> str:
    """The final recipe of the structure track in numbers and words, without any entry per ligand: the report names no blinded compound."""
    if not manifest_path.exists():
        return pending("job 180")
    data = json.loads(manifest_path.read_text())
    decision = data["decision"]
    recipe = decision["recipe"]
    levers = ", ".join(recipe["levers"]) if recipe["levers"] else "none"
    return "\n".join([
        (
            f"restraint `{recipe['restraint']}`, selection `{recipe['selection']}`, {recipe['extra_per_ligand']} extra samples per ligand, levers: {levers}, OpenFold3 {'used' if recipe['openfold3'] else 'not used'}, "
            f"expected-improvement allocation {'used' if recipe['ei_allocation'] else 'not used'}; {decision['estimated_hours']:.1f} hours of engine time were estimated before the runs."
        ),
        f"The archive `{Path(data['zip']).name}` (SHA-256 `{data['sha256'][:12]}`) holds {len(data['ligands'])} complexes and {len(data['posebusters_failures'])} of them fail a PoseBusters check.",
        "",
        "The reasons recorded with the recipe:",
        "",
        *[f"- {reason}" for reason in recipe["reasons"]],
    ])


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
        "- **Proprietary data.** None of the data is proprietary: every input is public. One point is stated here so that nothing is left to be found: the CYP3A4 stack holds two Chemprop networks that start from the public weights of OpenADMET (Apache-2.0). The pretraining targets of one of them are",
        "  predictions of a surrogate model that Novartis trained on its internal corpus, as the OpenADMET post of 4 August 2026 explains (https://doi.org/10.5281/zenodo.21828040). No proprietary measurement is in the project, and the weights are available to every participant, but they carry what that surrogate learned.",
        "- **Sources of the data of the final stacks.** The OpenADMET challenge files (training file and blinded structures; the dataset `openadmet/cyp-challenge-train-test`, whose card states Apache-2.0); the TDC / Veith CYP panel (Veith et al., Nature",
        "  Biotechnology 2009; the TDC page states CC BY 4.0), used only to train a label-free block on compounds that are not challenge compounds; and the rows that the public leaderboards publish (accuracy, precision and recall of each entry), from which one",
        "  number about the hidden labels is recovered, the positive rate of the scored set (limitation L13). No label of any blinded compound and no probing submission is used.",
        "- **Components.** The programs, models and weights that the work uses and their licences are listed in section 14 and in Appendix C of the notebook.",
        f"- **Identifier.** {('DOI ' + ids['doi'] + '.') if 'doi' in ids else 'The archived release has no DOI yet; the citation file lists it when it has one.'}{(' Software Heritage [' + ids['swh'] + '](https://archive.softwareheritage.org/' + ids['swh'] + ').') if 'swh' in ids else ''} The text to cite is in `USE_AND_CITATION.md`.",
        *published_lines(),
        "- **Reproduction and checks.** `NEGATIVE_RESULTS.md` (every experiment that failed its gate), `audit/AUDIT_REPORT.md` (generated twice and compared byte for byte), `bash scripts/reproduce_tdi.sh`.",
        "",
    ]
    return "\n".join(lines)


GROUP_TEXT = {  # what each group that can join a stack is, in words; the members and every number next to it are read from the code and the logs
    "neural": "the multitask network (one network over the assay arms, fitted to the label and to the arm shift) and plain Chemprop (a message-passing network on the molecular graph, fitted to the shift); two near-misses that joined into one lead on CYP3A4 and were confirmed on a replicate with ten outer folds",
    "chemprop_foundation": "two Chemprop networks that start from the weights of pretrained OpenADMET models (one trained on MiniMol targets, one the Novartis ADMET surrogate) and are fine-tuned to the arm shift; run for CYP3A4 only, in two stages under a rule written before any result",
}


def stack_rows(final_text: str, iso: str) -> dict[str, list[str]]:
    """The rows of the evidence table of one isoform in the final report, keyed by the name of the stack."""
    block = re.search(rf"## {iso}\n(.*?)(?=\n## |\Z)", final_text, re.DOTALL)
    rows = {}
    for line in (block.group(1).splitlines() if block else []):
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 6 and cells[0] not in ("stack", "---"):
            rows[cells[0]] = cells
    return rows


def what_joined(manifest: dict | None, final_text: str) -> list[str]:
    """For every group that joined a final stack, as sub-items of the step that says which groups join: what it is, its members and the gain it gave when added to the default stack."""
    from cypmode.challenge import final_tdi_build as build
    from cypmode.challenge.multiplicity import collect

    if not manifest:
        return []
    lines = []
    for iso, r in manifest["isoforms"].items():
        rows = stack_rows(final_text, iso)
        if not r["surviving_groups"]:
            lines.append(f"   - {iso}: no group passed its gate, so the final stack is the default stack of its candidates.")
        for group in r["surviving_groups"]:
            members = ", ".join(f"`{c.name}`" for c in build.GROUPS[group][0]())
            row = rows.get(f"default+{group}")
            gain = f" Added to the default stack it raises the AUC by {row[7]} (corrected p = {row[8]}) at the cut z0 = 1.0 and by {row[9]} (p = {row[10]}) at z0 = 1.96." if row and len(row) >= 11 else ""
            lines.append(f"   - {iso}, `{group}`: {GROUP_TEXT.get(group, 'a group of added candidates')}. Members: {members}.{gain}")
    smallest = float(collect(LOGS)["holm_adjusted"].min())
    if any(r["surviving_groups"] for r in manifest["isoforms"].values()):
        lines.append(f"   - These are passes of the registered rule. After the Holm adjustment over all stacked comparisons the smallest adjusted p-value is {smallest:.2f} (section 4.2), so they are reported as passes of the rule and " + ("not as effects that survive the number of attempts." if smallest >= 0.05 else "as effects that survive it."))
    return lines


def first_number(text: str, pattern: str, group: int = 1) -> str | None:
    found = re.search(pattern, text, re.MULTILINE)
    return found.group(group) if found else None


def stack_lead(final_text: str, manifest: dict | None) -> str:
    """Section 6 in words: what each final stack is and how far it stands above the default stack, from the table of the final build."""
    if not manifest:
        return pending("job 090")
    parts = []
    for iso, r in manifest["isoforms"].items():
        rows = stack_rows(final_text, iso)
        default, final = rows.get("default"), rows.get(r["stack"])
        if not default or not final:
            continue
        if r["stack"] == "default":
            parts.append(f"{iso}: the final stack is the default stack of {r['n_candidates']} candidates (AUC {default[2]} at z0 = 1.0 and {default[4]} at z0 = 1.96, MCC {default[3]} and {default[5]}); no added group passed its gate.")
        else:
            parts.append(f"{iso}: the final stack has {r['n_candidates']} candidates (AUC {final[2]} at z0 = 1.0 and {final[4]} at z0 = 1.96, MCC {final[3]} and {final[5]}), above the default stack of {default[1]} candidates (AUC {default[2]} and {default[4]}) by {final[7]} (p = {final[8]}) and {final[9]} (p = {final[10]}).")
    return "Each isoform has the stack that the pre-registered rule selects, from the nested cross-validation on the population the challenge scores. " + " ".join(parts) + f" The positive calls are the top {manifest['isoforms']['CYP3A4']['called_rate']:.1%} of the test compounds for CYP3A4 and the top {manifest['isoforms']['CYP2D6']['called_rate']:.1%} for CYP2D6."


def selection_numbers() -> tuple[str | None, ...]:
    """The hits-only AUC, the selection-aware AUC, their difference, its corrected p-value and the file that the registered rule gives, read from the logs of jobs 010 and 016."""
    panel = (LOGS / "population_panel_CYP2D6_r3.log").read_text(errors="ignore") if (LOGS / "population_panel_CYP2D6_r3.log").exists() else ""
    decision = (LOGS / "selection_decision_CYP2D6.log").read_text(errors="ignore") if (LOGS / "selection_decision_CYP2D6.log").exists() else ""
    z1 = panel.split("z0 = 1.96")[0]
    s0, s3 = first_number(z1, r"S0 hits-only\s+outer AUC ([\d.]+)"), first_number(z1, r"S3 \+end-to-end\s+outer AUC ([\d.]+)")
    gain, p = first_number(z1, r"S3 \+end-to-end.*?vs S0 AUC (\+[\d.]+) \(p=([\d.]+)"), first_number(z1, r"S3 \+end-to-end.*?vs S0 AUC \+[\d.]+ \(p=([\d.]+)")
    entry = first_number(decision, r"FINAL ENTRY BY THE REGISTERED RULE: (file \w)")
    return s0, s3, gain, p, entry


def selection_lead() -> str:
    """Section 7 in words, from the lines of the log of the selection-aware model and from the registered decision."""
    s0, s3, gain, p, entry = selection_numbers()
    if not (s0 and s3 and gain and p and entry):
        return pending("jobs 010 and 016")
    return (f"CYP2D6 compounds were measured on this enzyme only if an earlier screen promoted them, so the training labels come from a selected subset and the scored set may hold compounds that would never have been promoted. The selection-aware model multiplies P(label | hit) by a gate P(hit). "
            f"Against the hits-only stack it raises the outer AUC from {s0} to {s3} ({gain}, corrected p = {p}) when the scored set looks like the training panel. The gain depends on how many never-promoted compounds the scored set holds: it disappears when there are none, and the loss at the call rate of the final build is larger than the gain where it wins. "
            f"The choice between the two files was therefore made by a procedure written before any upload; it gives **{entry}** (the hits-only file), and the other file is kept with its SHA-256 to be scored on the test labels if the organisers publish them.")


def calibration_lead() -> str:
    text = (LOGS / "calibration.log").read_text(errors="ignore") if (LOGS / "calibration.log").exists() else ""
    cells = {(iso, method): ece for iso, method, ece in re.findall(r"^\s*(CYP\w+)\s+(raw|platt|isotonic)\s+([\d.]+)", text, re.MULTILINE)}
    if not cells:
        return pending("job 030")
    return (f"The published probabilities are calibrated by isotonic regression fitted on the inner out-of-fold scores. The expected calibration error is {cells.get(('CYP3A4', 'isotonic'), '?')} for CYP3A4 and {cells.get(('CYP2D6', 'isotonic'), '?')} for CYP2D6, against {cells.get(('CYP3A4', 'raw'), '?')} and {cells.get(('CYP2D6', 'raw'), '?')} for the raw output of the blend. "
            "The raw CYP2D6 blend ranks the compounds and transfers a call rate but is not a probability (limitation L10).")


def holdout_lead() -> str:
    text = (LOGS / "block_holdout.log").read_text(errors="ignore") if (LOGS / "block_holdout.log").exists() else ""
    rows, rate = first_number(text, r"held-out block rows (\d+)"), first_number(text, r"positive rate ([\d.]+)")
    if not (rows and rate):
        return pending("job 045")
    return (f"The highest block of identifiers of the training file is an analog-expansion round built like the blinded test set. Holding the whole block out ({rows} CYP3A4 compounds, positive rate {rate}) and training on the rest measures the candidates in a regime that matches the test better than a random or a similarity-weighted fold. "
            "It exists for CYP3A4 only, and the candidates and the best of them are chosen from cross-validation, never from the block. The AUC of each candidate on the block is listed next to its cross-validated AUC, so that the agreement of the two yardsticks is itself a result.")


def additions_lead() -> str:
    from cypmode.audit.ledger_kinds import ledger_rows

    frame = ledger_rows()
    part = frame[frame["log"].str.contains(r"physics|pair_transfer|mmp|committee|embeddings|test_likeness|arms|selection_aware_v2|xiso")]
    return (f"The additions of the physics, SAR, committee, embedding, cross-isoform and weighting families gave {len(part)} pre-registered verdicts in {part['stem'].nunique()} families, {int((part['verdict'] == 'MET').sum())} of them MET. "
            "A near miss is not tuned until it passes: the verdicts below are those of the first test, and the ones that were repeated on ten outer folds are listed after them.")


def benchmark_ligands() -> pd.DataFrame:
    """The ligands of the crystal-anchored benchmark with the PDB entry, the size and the resolution of the copy that the benchmark uses (the notebook and the report print the same table)."""
    from cypmode.validation.reference_ligands import load_reference_set
    from cypmode.validation.structure_benchmark import candidates

    table = candidates(load_reference_set()).set_index("comp_id")
    names = pd.read_csv(ROOT / "data/structure_track/inputs/structure_benchmark.csv")["Molecule_Name"]
    return table.loc[[n.removeprefix("bench-") for n in names], ["pdb_id", "n_heavy", "resolution"]].reset_index()


def benchmark_paragraph() -> str:
    """How the crystal ligands of the structure gates were chosen and what they can and cannot show, from the constants of the picking module and the files of the plan."""
    from cypmode.validation import structure_benchmark as sb
    from cypmode.validation.reference_ligands import load_reference_set

    references = load_reference_set()
    distinct, eligible = len({r["comp_id"] for r in references}), len(sb.candidates(references))
    plan = json.loads((ROOT / "data/structure_track/inputs/restraint_expansion_plan.json").read_text())
    stratum = plan["eligibility"]["strata"]["primary"]
    relaxed = pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/relaxation_gate_poses.csv")["compound"].drop_duplicates()
    ligands = benchmark_ligands()
    listing = ", ".join(f"{row.comp_id} ({row.pdb_id}, {row.n_heavy} heavy atoms, {row.resolution:.2f} A)" for row in ligands.itertuples())
    return (
        f"**The crystal ligands of the gates.** The 20 official ligands have no public answer, so every gate of the structure track was measured on ligands whose crystal pose is public. The benchmark is {len(ligands)} ligands "
        f"picked from the {distinct} distinct CYP3A4 ligands of the PDB (`cypmode/validation/reference_ligands.py`) by a rule fixed in advance (`cypmode/validation/structure_benchmark.py`): {sb.MIN_HEAVY} to {sb.MAX_HEAVY} heavy atoms, "
        f"resolution at most {sb.MAX_RESOLUTION} A, organic atoms only and one fragment ({eligible} of the {distinct} pass), then the most mutually dissimilar by a maximin pick on Morgan fingerprints that starts from the panel "
        f"compounds already predicted ({', '.join(sb.PANEL_SEEDS)}). The pick is deterministic and the list is not written by hand; with those seeds the function returns exactly these ligands (`tests/test_structure_benchmark.py`). "
        f"They are, with the PDB entry of the copy used: {listing}. Two cautions apply. The models may have seen these entries in training, so the absolute accuracy on the benchmark is optimistic and it serves to compare "
        "settings and selection rules, not to forecast a score. And the benchmark was chosen for diversity and not for likeness to the official ligands, which are not near the reference set (notebook section 11.5, limitation L15). "
        f"The gate of the relaxation used {len(relaxed)} ligands: these {len(ligands)} and the {int(relaxed.str.startswith('exp-').sum())} ligands of the restraint-expansion plan registered on {plan['registered']} (every crystal ligand with a Type II donor, "
        f"organic, one fragment, {stratum['min_heavy_atoms']} to {stratum['max_heavy_atoms']} heavy atoms, resolution at most {plan['eligibility']['max_resolution_angstrom']} A and not in the benchmark; every eligible ligand was run and none was picked)."
    )


def map_paragraph() -> str:
    """Where the map of the project is and what it holds, with the counts that the map itself writes."""
    from cypmode.audit import project_map

    modules, jobs = len(project_map.module_paths()), len(project_map.job_facts())
    return (
        f"**Where everything is.** The folder `docs/map` is the map of the project: its four pages list all {modules} modules with what each does and what it imports, all {jobs} jobs of the queue with what each waits for "
        "and the chain that rebuilds and checks the project, the flow of the data from the inputs through the jobs and the logs to the reports and the submission files, and the kinds of idea that were tried. "
        "The pages are written by `cypmode/audit/project_map.py` from the tracked files and from nothing else, so a copy of the repository regenerates them, and `python -m cypmode.audit.project_map --check` "
        "and a test fail when a page differs from the code. The README carries a short loop of the interactive map, `docs/map/explorer.html`, which draws all four views as one page that can be explored, with a search box and, for every module and job, what it is connected to and which section of the notebook names it, and it links to the page on the Pages site and to the file."

        "\n\n"
        "The flow of the data, from the inputs to the submission files:\n\n"
        f"{project_map.dataflow_diagram(project_map.dataflow_facts())}\n\n"
        "The packages of the code; an arrow goes from a package to a package whose modules it imports, and its label counts those imports:\n\n"
        f"{project_map.code_diagram(project_map.code_facts())}"
    )


def structure_lead() -> str:
    from cypmode.audit.docs_state import STRUCTURE_GATES, TITLES
    from cypmode.audit.ledger_kinds import ledger_rows

    frame = ledger_rows()
    part = frame[frame["kind"] == "structure levers"]
    met = [STRUCTURE_GATES.get(n.split(" ")[0], TITLES.get(s, n)) for s, n, v in zip(part["stem"], part["name"], part["verdict"], strict=True) if v == "MET"]
    return (f"The structure levers gave {len(part)} pre-registered verdicts, {len(met)} of them MET ({', '.join(sorted(set(met))) or 'none'}). Every rule for choosing a pose among the predicted ones was NOT MET on the crystal-anchored benchmark, so the pick is the model's own ranking among the valid poses; "
            "what the recipe adds is a restraint of the iron-nitrogen contact, the scorer's own validity call with a margin rule, and a registered relaxation.")


def label_rows() -> dict[str, tuple[str, str, str]]:
    """Per isoform from the data section of the audit: rows with both arms, mismatches of the label rule, rows without a direct measurement."""
    text = AUDIT.read_text() if AUDIT.exists() else ""
    return {iso: (both, miss, nodirect) for iso, both, miss, nodirect in re.findall(r"^\| (CYP\w+) \| (\d+) \| (\d+) \| (\d+) \| [\d.]+ \|", text, re.MULTILINE)}


def tier_counts() -> dict[str, tuple[str, str, str]]:
    """Per isoform from the applicability-domain log: blinded compounds in the supported, weak and unsupported tier at the primary confidence cut."""
    path = LOGS / "applicability_domain.log"
    text = path.read_text(errors="ignore") if path.exists() else ""
    found = {}
    for iso, body in re.findall(r"^## (CYP\w+)\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL):
        m = re.search(r"Blinded compounds by tier \(primary cut z0 = 1\.0\): (\d+) supported \(\d+%\), (\d+) weak \(\d+%\), (\d+) unsupported", body)
        if m:
            found[iso] = m.groups()
    return found



def label_shift_summary() -> dict[str, dict[str, str]]:
    """Per isoform from the log of the label-shift check: the estimate of the positive rate of the scored set with its interval at the primary cut, the rate of the final build, the recovery error and the verdict."""
    path = LOGS / "label_shift.log"
    text = path.read_text(errors="ignore") if path.exists() else ""
    out = {}
    for iso, body in re.findall(r"^LABEL SHIFT (CYP\w+): (.*?)(?=^LABEL-SHIFT CHECK)", text, re.MULTILINE | re.DOTALL):
        estimate = re.search(r"z0 = 1\.0 lgbm_A\+fps_deep: .*?scored compounds by transfer ([\d.]+) \(([\d.]+) to ([\d.]+)\)", body)
        ratio = re.search(r"mean error over the rates, as a share of the rate: ([\d.]+)", body)
        board = re.search(r"positive rate used by the final build \(read for the comparison only\): ([\d.]+)", body)
        verdict = re.search(rf"^LABEL-SHIFT CHECK {iso}: ([A-Z ]+) \((.*)\)$", text, re.MULTILINE)
        if estimate and ratio and board and verdict:
            out[iso] = {"estimate": estimate.group(1), "lo": estimate.group(2), "hi": estimate.group(3), "board": board.group(1), "ratio": ratio.group(1), "verdict": verdict.group(1), "reason": verdict.group(2)}
    return out


def committee_row(iso: str) -> tuple[str, str, str] | None:
    """The stacked gain of the committee candidates, its p-value and the calibration error of the committee probability, both at z0 = 1.0, from the log of the committee test."""
    path = LOGS / f"committee_label_model_{iso}.log"
    text = path.read_text(errors="ignore") if path.exists() else ""
    gain = re.search(r"z0=1\.0\s+stacked base\+added vs base: AUC ([+-][\d.]+) \(p=([\d.]+)", text)
    ece = re.search(r"z0=1\.0\s+ECE ([\d.]+)", text)
    return (gain.group(1), gain.group(2), ece.group(1)) if gain and ece else None


def innovation_section() -> str:
    """What the work adds, claim by claim, with the evidence next to each claim; every number is read from a log, a manifest or the audit report."""
    from cypmode.audit.ledger_kinds import totals
    from cypmode.challenge import final_tdi_build as build

    manifest = json.loads((FINAL_TDI / "manifest.json").read_text()) if (FINAL_TDI / "manifest.json").exists() else None
    structure = json.loads((FINAL_STRUCTURE / "manifest.json").read_text()) if (FINAL_STRUCTURE / "manifest.json").exists() else None
    labels, tiers, shift = label_rows(), tier_counts(), label_shift_summary()
    s0, s3, gain, p, entry = selection_numbers()
    verdicts, families, met = totals()
    permuted = {iso: first_number((LOGS / f"permutation_control_{iso}.log").read_text(errors="ignore") if (LOGS / f"permutation_control_{iso}.log").exists() else "", r"z0=1\.0\s+real labels.*?permuted: AUC ([\d.]+)") for iso in ("CYP3A4", "CYP2D6")}
    calibration = (LOGS / "calibration.log").read_text(errors="ignore") if (LOGS / "calibration.log").exists() else ""
    ece = {(iso, method): value for iso, method, value in re.findall(r"^\s*(CYP\w+)\s+(raw|isotonic)\s+([\d.]+)", calibration, re.MULTILINE)}
    stereo = (LOGS / "stereo_check.log").read_text(errors="ignore") if (LOGS / "stereo_check.log").exists() else ""
    chiral, pooled, pool_total = first_number(stereo, r"specified stereochemistry: (\d+) of \d+"), first_number(stereo, r"\((\d+) of \d+\)\s*$"), first_number(stereo, r"\(\d+ of (\d+)\)\s*$")
    gates = {name: verdict for _, name, verdict in gate_rows()}
    lines = [
        "## Innovation summary",
        "",
        "This section lists what the work adds, one claim at a time, with the evidence that stands next to it. Each row is a claim that the logs can check; the cross-validated numbers are estimates on the training data and not scores on the hidden labels (limitations L1, L2 and L13). Where an idea was tried and failed its gate, that is stated with the idea. Each claim ends with the section of the notebook that explains it from first principles and computes its number.",
        "",
        "### Architecture: the label is modelled through its process",
        "",
        f"- **The label is a function of two measurements, so the models learn the measurements.** The TDI label is positive when `s = t - max(d, 4) > log10 2`. The rule reproduces every training label ({labels['CYP3A4'][1] if 'CYP3A4' in labels else '?'} mismatches for CYP3A4, {labels['CYP2D6'][1] if 'CYP2D6' in labels else '?'} for CYP2D6). The stack therefore fits the shift `s` and the arms, and the label and its confidence `z = |s - log10 2| / sd(s)` follow from them, instead of fitting a binary label that discards the size of the margin.",
        f"- **A selection-aware CYP2D6 model.** CYP2D6 was measured only for compounds that an earlier screen promoted, so a model of the label alone answers a question about a selected population. The second file multiplies P(label | hit) by a gate P(hit). When the scored set looks like the training panel it raises the outer AUC from {s0} to {s3} ({gain}, corrected p = {p}); the gain disappears when the scored set holds no never-promoted compounds, and at the call rate of the final build the loss there is larger than the gain where it wins (section 7). The choice between the two files was made by a rule fixed before any upload, and the rule gives **{entry}** (the hits-only file); the other file is kept with its SHA-256." if (s0 and s3 and gain and p and entry) else f"- **A selection-aware CYP2D6 model.** {pending('jobs 010 and 016')}",
        f"- **Groups join a stack only through a gate.** {verdicts} pre-registered verdicts in {families} families; {met} are MET and {verdicts - met} are NOT MET. The final stacks are {manifest['isoforms']['CYP3A4']['n_candidates']} (CYP3A4) and {manifest['isoforms']['CYP2D6']['n_candidates']} (CYP2D6) candidates, selected by the logs and by nothing else." if manifest else f"- **Groups join a stack only through a gate.** {pending('job 090')}",
        "",
        "### Data usage",
        "",
        f"- **A contaminated slice of the labels was found and removed from the CYP3A4 pool.** {labels['CYP3A4'][2] if 'CYP3A4' in labels else '?'} CYP3A4 rows have no direct-inhibition measurement and follow a different label convention than the blinded set; the pool keeps the {labels['CYP3A4'][0] if 'CYP3A4' in labels else '?'} rows with both arms, and every CYP3A4 number is computed on that population (section 2).",
        f"- **The evaluation reproduces the scored population.** The outer folds are weighted by similarity to the blinded structures and by the estimated prevalence, and only confidently labelled compounds are evaluated; {build.REPEATS} repeats of {build.DESIGN['outer']} outer folds ({build.REPEATS * build.DESIGN['outer']} outer evaluations) with every selection step inside the inner folds (section 5).",
        f"- **One aggregate number is recovered from the public leaderboards.** The accuracy, precision and recall that the boards publish for every entry fix the positive rate of the scored set: {manifest['isoforms']['CYP3A4']['target_prevalence']:.3f} for CYP3A4 and {manifest['isoforms']['CYP2D6']['target_prevalence']:.3f} for CYP2D6. It sets the call rate (a multiple of the prevalence: {manifest['isoforms']['CYP3A4']['call_rate_multiplier']} and {manifest['isoforms']['CYP2D6']['call_rate_multiplier']}); no label of any compound and no probing submission is used (limitation L13)." if manifest else f"- **One aggregate number is recovered from the public leaderboards.** {pending('job 090')}",
        (
            "- **A check of that rate that reads no board.** Black-box label-shift estimation (a classifier that scores held-out compounds and the blinded structures on one scale, folds that keep chemical series whole, "
            + "the held-out compounds weighted by structure similarity only) estimates the positive rate of the scored set from the training data. At z0 = 1.0 it gives "
            + "; ".join(f"{v['estimate']} ({v['lo']} to {v['hi']}) for {iso}, against {v['board']} in the final build" for iso, v in shift.items())
            + ". The rule fixed before the first run (`cypmode/challenge/label_shift.py`) gives "
            + "; ".join(f"{v['verdict']} for {iso} ({v['reason']})" for iso, v in shift.items())
            + ". The training data alone do not fix the rate to better than the width of these intervals, so the boards remain the source of the rate; "
            + " ".join(f"the {iso} interval {'contains' if float(v['lo']) <= float(v['board']) <= float(v['hi']) else 'does not contain'} the rate of the final build." for iso, v in shift.items())
            if len(shift) == 2
            else f"- **A check of that rate that reads no board.** {pending('job 285')}"
        ),
        f"- **The official structure set was checked for stereochemistry.** {chiral or '?'} of the 20 ligands carry specified stereochemistry; only {pooled or '?'} of the {pool_total or '?'} pool poses of those ligands reproduce it, so the pick is restricted to poses that do, and all 20 submitted poses reproduce the official stereochemistry (`stereo_check.log`).",
        "",
        "### Uncertainty",
        "",
        "- **Reliability by similarity to the training data, with intervals over chemical series.** Held-out compounds of the outer folds are cut by the nearest-training similarity; the interval of each bin resamples whole Butina clusters, so a bin is called supported only when the lower end of its interval reaches 0.60 (`applicability_domain.py`, `applicability_domain.log`). "
        + (f"For CYP3A4 the 750 blinded compounds fall into {tiers['CYP3A4'][0]} supported, {tiers['CYP3A4'][1]} weak and {tiers['CYP3A4'][2]} unsupported compounds; for CYP2D6 into {tiers['CYP2D6'][0]} supported, {tiers['CYP2D6'][1]} weak and {tiers['CYP2D6'][2]} unsupported compounds, so the CYP2D6 predictions carry the weaker reliability." if "CYP3A4" in tiers and "CYP2D6" in tiers else pending("job 270")),
        f"- **Probabilities calibrated on the inner folds, with the failure reported.** The expected calibration error after isotonic regression is {ece.get(('CYP3A4', 'isotonic'), '?')} (CYP3A4) and {ece.get(('CYP2D6', 'isotonic'), '?')} (CYP2D6); the raw CYP2D6 blend has {ece.get(('CYP2D6', 'raw'), '?')} and is a ranking, not a probability (limitation L10).",
        "- **A committee of models as an uncertainty signal was tested and did not help.** " + (
            f"Added to the stack it moves the AUC by {committee_row('CYP3A4')[0]} (p = {committee_row('CYP3A4')[1]}) for CYP3A4 and {committee_row('CYP2D6')[0]} (p = {committee_row('CYP2D6')[1]}) for CYP2D6 at z0 = 1.0; its probability is miscalibrated as it comes out (expected calibration error {committee_row('CYP3A4')[2]} and {committee_row('CYP2D6')[2]}). It is reported with the predictions and never alters one."
            if committee_row("CYP3A4") and committee_row("CYP2D6") else pending("committee test")
        ),
        "",
        "### Simple ideas that carry the structure track",
        "",
        f"- **A restraint of the iron-nitrogen bond for Type II ligands.** One extra constraint in the Boltz-2 input; the gates `restraint_bond` and `restraint_contact` are {gates.get('restraint_bond', '?')} and {gates.get('restraint_contact', '?')} on the crystal-anchored benchmark, and the same gates under the OpenStructure scorer give {gates.get('restraint_bond_ost', '?')} and {gates.get('restraint_contact_ost', '?')}.",
        f"- **The scorer's own validity call with a margin.** Every pose is checked by the 22 PoseBusters checks of the scorer, and a pose is eligible only when its smallest relative distance to the receptor is at least {structure_margin()}; a pose that fails a check scores zero, so validity is treated as a constraint and not as a feature.",
        (f"- **A relaxation under a rule written before it was run.** The force-field relaxation of the ligand in the fixed pocket replaces a pose only when both pass the checks, the margin rises, the ligand moves at most {structure['relaxation']['rule']['max_movement_A']} A and the iron-nitrogen distance changes at most {structure['relaxation']['rule']['max_fe_n_change_A']} A; it was adopted for {structure['relaxation']['adopted']} of {structure['relaxation']['of']} ligands and raised the smallest margin from {structure['relaxation']['smallest_margin_before']:.3f} to {structure['relaxation']['smallest_margin_after']:.3f}." if structure else f"- **A relaxation under a rule written before it was run.** {pending('job 243')}"),
        "",
        "### Rigor",
        "",
        dynamics_bullet(),
        f"- **Negative results are results.** `NEGATIVE_RESULTS.md` lists all {verdicts - met} verdicts that were NOT MET. The negative control, the whole nested procedure on permuted labels, gives an outer AUC of {permuted['CYP3A4'] or '?'} (CYP3A4) and {permuted['CYP2D6'] or '?'} (CYP2D6) at z0 = 1.0.",
        "- **Every number in the reports is copied by code from a log and checked by the audit,** which is generated twice and compared byte for byte (section 4.4).",
        "",
        "### What is not claimed",
        "",
        "- The cross-validated AUC and MCC are estimates on the training data. The portal scores hidden labels, its test set sits closer to the training data than a random fold does, and the live board scores a part of it.",
        "- The structure recipe does not claim a better pose choice: every rule for choosing among predicted poses is NOT MET. What it adds is the restraint, the validity margin, the stereochemistry check and the relaxation.",
        "- The positive rates come from the public boards. The attempt to estimate them without a board is reported above with the verdict of its rule; limitation L13 states what depends on them.",
        "",
    ]
    return "\n".join(f"{line}{notebook_pointer(line)}" for line in lines)


INNOVATION_COLUMNS = ["claim", "evidence", "source"]

NOTEBOOK_OF_CLAIM = (  # each claim of the innovation summary with the sections of the notebook that explain it and compute its number (the notebook checks that every section exists and holds a code cell)
    ("The label is a function of two measurements, so the models learn the measurements.", "2.3 2.5"),
    ("A selection-aware CYP2D6 model.", "9.1"),
    ("Groups join a stack only through a gate.", "6.7 9.11"),
    ("A contaminated slice of the labels was found and removed from the CYP3A4 pool.", "2.3"),
    ("The evaluation reproduces the scored population.", "6.4"),
    ("One aggregate number is recovered from the public leaderboards.", "2.10"),
    ("A check of that rate that reads no board.", "2.11"),
    ("The official structure set was checked for stereochemistry.", "11.12"),
    ("Reliability by similarity to the training data, with intervals over chemical series.", "8.7"),
    ("Probabilities calibrated on the inner folds, with the failure reported.", "8.5"),
    ("A committee of models as an uncertainty signal was tested and did not help.", "7.6"),
    ("A restraint of the iron-nitrogen bond for Type II ligands.", "11.6"),
    ("The scorer's own validity call with a margin.", "11.4 11.10"),
    ("A relaxation under a rule written before it was run.", "11.10"),
    ("The pharmacophore of a pose read over a short trajectory, tested under a rule fixed before any number.", "11.13"),
    ("Negative results are results.", "9.10 9.11"),
    ("Every number in the reports is copied by code from a log and checked by the audit,", "12.2"),
)


def notebook_pointer(line: str) -> str:
    """The sentence that names the sections of the notebook for a claim of the innovation summary; nothing for a line that is not a claim."""
    for claim, sections in NOTEBOOK_OF_CLAIM:
        if line.startswith(f"- **{claim}**"):
            found = sections.split()
            return f" Notebook {'section' if len(found) == 1 else 'sections'} {' and '.join(found)}."
    return ""


def innovation_rows() -> list[tuple[str, str, str]]:
    """The claims of the innovation section, one row each: the claim, the number that carries it and the log it comes from."""
    from cypmode.audit.ledger_kinds import totals

    manifest = json.loads((FINAL_TDI / "manifest.json").read_text()) if (FINAL_TDI / "manifest.json").exists() else None
    structure = json.loads((FINAL_STRUCTURE / "manifest.json").read_text()) if (FINAL_STRUCTURE / "manifest.json").exists() else None
    labels, tiers = label_rows(), tier_counts()
    s0, s3, gain, p, entry = selection_numbers()
    verdicts, families, met = totals()
    gates = {name: verdict for _, name, verdict in gate_rows()}
    rows = []
    if labels:
        rows.append(("The label is a function of two measurements; the models learn the measurements", f"{labels['CYP3A4'][1]} and {labels['CYP2D6'][1]} mismatches of the label rule over {labels['CYP3A4'][0]} and {labels['CYP2D6'][0]} rows with both arms", "audit section 4"))
        rows.append(("A slice of the CYP3A4 labels follows another convention and is left out", f"{labels['CYP3A4'][2]} rows without a direct measurement", "audit section 4"))
    if s0 and s3 and gain and p and entry:
        rows.append(("Selection-aware CYP2D6 model; the file is chosen by a rule fixed before any upload", f"AUC {s0} to {s3} ({gain}, corrected p = {p}) on a scored set that looks like the training panel; the rule gives {entry}", "`population_panel_CYP2D6_r3.log`, `selection_decision_CYP2D6.log`"))
    rows.append(("Every idea passes a gate written before its result", f"{verdicts} verdicts in {families} families, {met} MET", "`NEGATIVE_RESULTS.md`"))
    if tiers.get("CYP3A4") and tiers.get("CYP2D6"):
        rows.append(("Reliability tiers by similarity to the training data, intervals over chemical series", f"blinded compounds supported: {tiers['CYP3A4'][0]} of 750 (CYP3A4), {tiers['CYP2D6'][0]} of 750 (CYP2D6)", "`applicability_domain.log`"))
    if manifest:
        rows.append(("Positive rates of the scored set recovered from the public boards", f"{manifest['isoforms']['CYP3A4']['target_prevalence']:.3f} (CYP3A4), {manifest['isoforms']['CYP2D6']['target_prevalence']:.3f} (CYP2D6)", "limitation L13"))
    shift = label_shift_summary()
    rows.append((
        "The same rates estimated without a board, by label shift, under a rule fixed before the first run",
        "; ".join(f"{v['estimate']} ({v['lo']} to {v['hi']}) against {v['board']} for {iso}: {v['verdict']}" for iso, v in shift.items()) if len(shift) == 2 else pending("job 285"),
        "`label_shift.log`",
    ))
    rows.append(("Restraint of the iron-nitrogen bond for Type II ligands", f"gate {gates.get('restraint_bond', '?')} on the crystal benchmark, {gates.get('restraint_bond_ost', '?')} under the OpenStructure scorer", "`structure_gates.log`, `ost_pool_gates.log`"))
    dynamics = dynamics_row()
    if dynamics:
        rows.append(dynamics)
    if structure:
        rows.append(("Validity margin of the scorer's own call and a relaxation under a registered rule", f"relaxation adopted for {structure['relaxation']['adopted']} of {structure['relaxation']['of']}; smallest margin {structure['relaxation']['smallest_margin_before']:.3f} to {structure['relaxation']['smallest_margin_after']:.3f}", "`relax_final.log`"))
    return rows


def dynamics_bullet() -> str:
    """The bullet of the innovation summary for the ligand dynamics in the frozen pocket: what was run, the verdicts of its three scores and what the structure entry does with them."""
    from cypmode.validation import ligand_dynamics as ld

    path = LOGS / "ligand_dynamics.log"
    if not path.exists():
        return f"- **A dynophore that the project can afford was tested and did not help.** {pending('job 286')}"
    facts = ld.read_log(path.read_text(errors="ignore"))
    scores, met = facts["scores"], [n for n in ld.SCORES if facts["scores"][n]["gate"] == "MET"]
    used = [n for n in ld.SCORES if scores[n]["use"] == "MET"]
    rhos = ", ".join(f"{scores[n]['rho']:+.3f}" for n in ld.SCORES)
    verdict = (f"none of the three scores met it (mean rho {rhos} against the bar of 0.3; the static model of the same poses gives {scores['static_pharmacophore_model']['rho']:+.3f}), so the structure entry was not changed"
               if not met else f"the gate was met by {', '.join(met)}" + (f" and the paired test by {', '.join(used)}" if used else ", the paired test by none"))
    return (f"- **The pharmacophore of a pose read over a short trajectory, tested under a rule fixed before any number.** The ligand moves for {round(ld.STEPS * ld.TIMESTEP_PS)} ps in the frozen pocket (protein and heme at their predicted "
            f"positions, no water) for {facts['poses']} poses of {facts['ligands']} benchmark ligands; it is not a dynophore of the literature, which follows the whole complex in water (`docs/structure_track/structure_ligand_dynamics.md`, limitation L16). "
            f"Judged by the gate of the pose rules, {verdict}.")


def dynamics_row() -> tuple[str, str, str] | None:
    """The claim of the ligand dynamics in the frozen pocket, read from the log of its gate (job 286)."""
    from cypmode.validation import ligand_dynamics as ld

    path = LOGS / "ligand_dynamics.log"
    if not path.exists():
        return None
    facts = ld.read_log(path.read_text(errors="ignore"))
    scores = facts["scores"]
    met = [n for n in ld.SCORES if scores[n]["gate"] == "MET"]
    evidence = ", ".join(f"{n.replace('_', ' ')} {scores[n]['rho']:+.3f}" for n in ld.SCORES) + f" against {scores['static_pharmacophore_model']['rho']:+.3f} for the static model: " + (f"gate MET for {len(met)} of 3" if met else "gate NOT MET for all 3")
    return (f"The pharmacophore of a pose read over {round(ld.STEPS * ld.TIMESTEP_PS)} ps of ligand dynamics in the frozen pocket, under a rule fixed before any number", f"mean rho with minus the true RMSD over {facts['ligands']} ligands, {evidence}", "`ligand_dynamics.log`")


def innovation_table() -> str:
    """The rows of ``innovation_rows`` as a markdown table; the README and the specification embed it as a generated block."""
    return table(innovation_rows(), INNOVATION_COLUMNS)


def structure_margin() -> str:
    from cypmode.validation.official_posebusters import MARGIN_FLOOR

    return f"{MARGIN_FLOOR}"


def structure_board_lead() -> str:
    """The live scores of the structure uploads next to the organisers' baseline, from the file that records them by hand, with the result of the rule that decided between the two candidates."""
    if not STRUCTURE_BOARD.exists():
        return pending("docs/structure_track/structure_board_scores.csv")
    frame = pd.read_csv(STRUCTURE_BOARD)
    by = {row.entry: row for row in frame.itertuples(index=False)}
    s1, c, base = by["submitted recipe (S1)"], by["candidate C"], by["organisers' Boltz-2 baseline"]
    return (
        f"Live scores of the two structure uploads (the live board scores 10 of the 20 ligands; `docs/structure_track/structure_board_scores.csv`, copied from the board on {s1.read_on}): the submitted recipe scored an LDDT-PLI of {s1.lddt_pli:.4f} (standard deviation {s1.lddt_pli_std:.4f}) and a BiSyRMSD of {s1.bisy_rmsd:.2f} A, "
        f"rank {s1.rank} of {s1.entries} while it was the live entry; the organisers' Boltz-2 baseline scored {base.lddt_pli:.4f} and {base.bisy_rmsd:.2f} A. The recipe is therefore below the baseline on the primary metric, by {base.lddt_pli - s1.lddt_pli:.3f} against a board standard deviation of {base.lddt_pli_std:.3f}, and ahead of it on the BiSyRMSD. "
        f"A second candidate, the first eligible unguided pose of each ligand, scored {c.lddt_pli:.4f} and {c.bisy_rmsd:.2f} A; its gain of {c.lddt_pli - s1.lddt_pli:+.4f} is below the threshold that was registered before the upload (`docs/structure_track/structure_candidate_c.md`), so the submitted recipe stays the structure entry."
    )


def lessons_lead() -> list[str]:
    """Section 12: how many ideas of each kind failed, and the failures that taught the most (the story of each is in the notebook and the specification)."""
    from cypmode.audit.ledger_kinds import KINDS, ledger_by_kind, totals

    frame = ledger_by_kind()
    verdicts, families, met = totals()
    rows = [(kind, int(part["stem"].nunique()), len(part), int((part["verdict"] != "MET").sum())) for kind in KINDS if not (part := frame[frame["kind"] == kind]).empty]
    return [
        f"Of {verdicts} pre-registered verdicts in {families} families of ideas, {verdicts - met} are NOT MET. A failed gate is a result: it is written down with its numbers and it is not tuned until it passes.",
        "",
        table(rows, ["kind of idea", "families", "verdicts", "not met"]),
        "",
        "`NEGATIVE_RESULTS.md` lists every idea with its gain and its p-value next to the verdict. The failures that changed the project most are told in full in the notebook (Part IX) and in the specification:",
        "",
        "- *The direct-inhibition bridge.* A predicted direct-inhibition potency looked like the largest gain of the project and was retracted: it read the availability of a measurement, which the blinded test set never shows.",
        "- *The population of the CYP3A4 labels.* A third of the labelled rows follow a convention that the scored test set does not have; every earlier CYP3A4 number mixed the two populations, and the evaluation was rebuilt on the population the challenge scores.",
        "- *The amine flag and the structural features of Tier 3.* Two mechanistically motivated additions that the data neither supported nor refuted; one was left out and the other could never reach the test rows.",
        "- *Choosing a pose.* Every rule tried for choosing among predicted poses failed its gate, which is why the recipe keeps the model's own ranking and spends its effort on the restraint, the validity margin and the relaxation.",
        "- *Uncertainty by a committee of models.* It adds no information beyond the decision margin for the final stacks, so it is reported and never alters a prediction.",
        "",
    ]


def idea_tables() -> list[str]:
    """The ideas that were added to the TDI stack and not admitted, with the gain they gave when added and the p-value of that gain, and the structure levers that were not adopted."""
    from cypmode.audit.docs_state import STRUCTURE_GATES, TITLES
    from cypmode.audit.ledger_kinds import KINDS, ledger_rows, stem_of
    from cypmode.explorer.data import paired_rows

    ledger = ledger_rows()
    tdi_kinds = [k for k in KINDS if k not in ("structure levers", "replicates")]
    paired = paired_rows(LOGS)
    paired = paired[paired["comparison"].str.startswith("stacked base+added")]
    rows = []
    for kind in tdi_kinds:
        for stem in KINDS[kind]:
            for iso in ("CYP3A4", "CYP2D6"):
                part = paired[(paired["log"].map(stem_of) == stem) & (paired["enzyme"] == iso)]
                verdicts = ledger[(ledger["stem"] == stem) & ledger["log"].str.contains(iso)]
                if part.empty or (verdicts["verdict"] == "MET").any():
                    continue
                by = {z: r for z, r in zip(part["z0"], part.itertuples(), strict=True)}
                low, high = by.get(1.0), by.get(1.96)
                if not (low and high):
                    continue
                reading = "no gain at either cut" if low.auc_gain <= 0 and high.auc_gain <= 0 else "a gain that is within noise" if low.p >= 0.05 and high.p >= 0.05 else "significant at one cut only"
                rows.append((kind, TITLES.get(stem, stem.replace("_", " ")), iso, f"{low.auc_gain:+.4f} (p = {low.p:.3f})", f"{high.auc_gain:+.4f} (p = {high.p:.3f})", reading))
    structure = ledger[(ledger["kind"] == "structure levers") & (ledger["verdict"] != "MET")]
    levers = sorted({(TITLES.get(s, s.replace("_", " ")), STRUCTURE_GATES.get(n.split(" ")[0], n)) for s, n in zip(structure["stem"], structure["name"], strict=True)})
    return [
        "## Ideas that were added to the TDI stack and not admitted",
        "",
        "Each row is one idea on one enzyme: the change in the AUC of the stack when the idea is added to it (positive is better), at the two confidence cuts of the labels, with the corrected resampled p-value of that change. The gate asks for p < 0.05 at both cuts, alone and inside the stack.",
        "",
        table(rows, ["kind of idea", "idea", "enzyme", "change in AUC at z0 = 1.0", "change in AUC at z0 = 1.96", "reading"]),
        "",
        "## Structure-track levers that were not adopted",
        "",
        "Each lever was judged on the crystal-anchored benchmark by a rule written beforehand; none of the levers below met it.",
        "",
        table(levers, ["family", "gate"]),
        "",
    ]


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
        lines += what_joined(manifest, (FINAL_TDI / "final_report.md").read_text() if (FINAL_TDI / "final_report.md").exists() else "")
    lines += [
        "7. **Calibration and calls.** Probabilities by isotonic regression on the inner out-of-fold scores (notebook 8.5). The number of positive calls is a multiple of the estimated prevalence of the scored population, the multiple in",
        f"   {{{', '.join(str(m) for m in build.RATE_MULTIPLIERS)}}} chosen by the highest mean outer-fold MCC (notebook 8.4)" + (": " + "; ".join(f"{iso} {r['call_rate_multiplier']}" for iso, r in manifest["isoforms"].items()) if manifest else "") + ".",
        "8. **Two files.** A (hits-only CYP2D6 model) and B (the selection-aware CYP2D6 model, which multiplies P(label | hit) by a gate P(hit)); CYP3A4 is identical in both. The entry that is final is chosen by a rule declared in advance (`cypmode/challenge/selection_decision.py`, notebook 13.4); the rule gives file A, and file B is kept, with its SHA-256 in the manifest, to be scored on the test labels if the organisers publish them.",
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
        (
            f"In all {len(frame)} pre-registered verdicts in {frame['stem'].nunique()} families, {int((frame['verdict'] == 'MET').sum())} of them MET (figure `docs/figures/ml/ml_exploration_map-light.png`; notebook Part IX). "
            f"This is the one count of what was tried that every document of the project quotes. The logs print {len(gate_rows()) - len(frame)} more gate lines, which are not in it: the negative control, where a MET verdict means that the pipeline gives chance on permuted labels, and the exploratory second looks, which no gate parser reads."
        ),
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
        ungated_section(),
    ]
    return "\n".join(lines)



ANALYSES = re.compile(r"^(?:confidence_.*|context_.*|data_checker_.*|why_no_gain|population_sensitivity_.*|population_panel_.*|selection_decision_.*|test_nonhit_share_.*|block_holdout|label_shift|calibration|applicability_domain|conformer_bias|site_representation)\.log$")  # logs that hold a result and carry no gate line: each is named in section 4.7
INFRASTRUCTURE = re.compile(
    r"^(?:.*_(?:extract|build)|build_.*|lint_.*|audit_run|pre_submission_gate|figures_build|video_render|voices_.*|blender_clips|pymol_clips|explorer_bundle|interactive_card_.*|card_build|submissions_sync|"
    r"method_report|docs_state|public_readiness|public_readiness_after_deadline|release_after_deadline|clean_clone_check|bench_.*|cryoem_.*|pilot_.*|series_.*|restraint_expansion_.*|stereo_.*|affinity_.*|bioemu2_benchmark|openfold3_.*|official_posebusters_.*|ost_pools|"
    r"ligand_dynamics_prepare|ligand_dynamics_simulate_.*|final_recipe|final_recipe_.*|relax_final|pose_awareness_single|official_extra_audit|exploratory_.*|tabicl.*|tabpfn.*|atomjepa_head.*|chemprop_CYP.*|mlp_aux_.*|neural_f10_.*|grid_f10_.*|final_tdi_build_f10|recompute_paired_summaries|chemeleon_speed)\.log$"
)  # builds, extractions and runs that feed a gate or a document; the exploratory second looks are listed by name in section 4.7


def log_roles() -> dict[str, str]:
    """Every log of the experiment folder and its role: a gate (the log holds a pre-registered verdict), an analysis (a result without a gate line, named in section 4.7), infrastructure, or none."""
    from cypmode.audit.ledger_kinds import stem_of

    gated = {stem_of(f) for f, _, _ in gate_rows()}
    roles = {}
    for path in sorted(LOGS.glob("*.log")):
        if stem_of(path.name) in gated:
            roles[path.name] = "gate"
        elif ANALYSES.match(path.name):
            roles[path.name] = "analysis"
        elif INFRASTRUCTURE.match(path.name):
            roles[path.name] = "infrastructure"
        else:
            roles[path.name] = "none"
    return roles


def unclassified_logs() -> list[str]:
    """Logs that have no role: a new experiment whose result no document would mention."""
    return [name for name, role in log_roles().items() if role == "none"]


PAIR = re.compile(r"z0=([\d.]+)\s+(\S+)\s+vs plain: AUC ([\d.]+) -> ([\d.]+) \(([+-][\d.]+), p=([\d.]+)")
ANALYSIS_TEXT = [  # log pattern, what the log holds, and where its result is described ("paired", "checker" and "shift" mean that the result is read from the log itself)
    (r"confidence_.*", "training on label confidence: the rows with |z| < 1 dropped or down-weighted, two deep LightGBM feature sets (notebook 9.8)", "paired"),
    (r"context_.*", "the alert-in-context block added to two deep LightGBM feature sets, alone and with the rows of |z| < 1 dropped (notebook 9.8)", "paired"),
    (r"data_checker_.*", "check of the training file against the blinded file with the challenge data checker", "checker"),
    (r"why_no_gain", "diagnostics of why each idea that did not join added nothing; the table is `why_no_gain.csv`", "described in notebook 9.12"),
    (r"block_holdout", "the test-like hold-out of the analog-expansion block", "described in section 10"),
    (r"population_sensitivity_.*", "dependence of the selection-aware model on the share of never-promoted compounds", "described in section 7"),
    (r"population_panel_.*", "the panel experiments of the selection-aware model", "described in section 7"),
    (r"selection_decision_.*", "the registered choice of the CYP2D6 file", "described in section 7"),
    (r"test_nonhit_share_.*", "the estimate of the share of never-promoted compounds among the blinded compounds", "described in section 7"),
    (r"label_shift", "the estimate of the positive rate without a board", "shift"),
    (r"calibration", "calibration of the published probabilities", "described in section 9"),
    (r"applicability_domain", "reliability by similarity to the training data", "described in the innovation summary"),
    (r"conformer_bias", "how often the 12-conformer ensemble holds the crystal pose", "limitation L14"),
    (r"site_representation", "variability of the binding site and the support of the pocket fields", "described in notebook 11.11"),
]


def paired_summary(name: str) -> str:
    """The comparisons of one log of paired candidates: the number of outer folds, how many comparisons, the range of the AUC change and the smallest p-value (the corrected resampled t of the log)."""
    path = LOGS / name
    text = path.read_text(errors="ignore") if path.exists() else ""
    pairs, folds = PAIR.findall(text), re.search(r"(\d+) outer folds", text)
    if not pairs or not folds:
        return pending(name)
    changes = [float(c) for *_, c, _ in pairs]
    return f"{folds.group(1)} outer folds, {len(pairs)} comparisons, AUC change {min(changes):+.4f} to {max(changes):+.4f}, smallest p = {min(float(p) for *_, p in pairs):.2f}"


def checker_summary(name: str) -> str:
    """The result of one run of the challenge data checker, read from its report: the similarity threshold, the number of pairs it flags and their similarity (no compound is named)."""
    path = ROOT / "audit/data_checker" / (name.removeprefix("data_checker_").removesuffix(".log") + ".json")
    if not path.exists():
        return pending(name)
    data = json.loads(path.read_text())
    flagged = data.get("train_test_leakage", {}).get("tanimoto_similarity", [])
    threshold, n = data["config"]["max_train_test_similarity"], data["summary"]["total_tanimoto_leakage_pairs"]
    values = ", ".join(f"{pair['similarity']:.3f}" for pair in flagged)
    return f"similarity threshold {threshold}: {n} pair(s) flagged" + (f", of similarity {values}" if flagged else "")


def analysis_row(name: str) -> tuple[str, str, str]:
    for pattern, holds, how in ANALYSIS_TEXT:
        if re.fullmatch(pattern + r"\.log", name):
            if how == "paired":
                result = paired_summary(name)
            elif how == "checker":
                result = checker_summary(name)
            elif how == "shift":
                shift = label_shift_summary()
                result = "; ".join(f"{iso}: {v['verdict']}" for iso, v in shift.items()) if shift else pending("job 285")
            else:
                result = how
            return f"`{name}`", holds, result
    raise KeyError(f"no description for the analysis log {name}")


def ungated_section() -> str:
    """Section 4.7: the experiments that hold a result and no gate line, so that no document leaves out an experiment because the ledger does not count it."""
    roles = log_roles()
    rows = [analysis_row(name) for name, role in roles.items() if role == "analysis"]
    exploratory = ", ".join(f"`{n}`" for n in roles if n.startswith("exploratory_"))
    return "\n".join([
        "### 4.7 Experiments without a registered gate",
        "",
        (
            "A log that holds no gate line is not a row of the ledger, and the ledger is the one count that this report quotes. These experiments were run and are listed here by their logs, so that the count of what was tried is not read as the count of what was run. "
            "The change in AUC is the paired difference of a candidate with its plain counterpart, with the corrected resampled t of the log."
        ),
        "",
        table(rows, ["log", "what it holds", "result"]),
        "",
        f"The logs of the exploratory second looks of section 4.3 are {exploratory or pending('the exploratory jobs')}. Every other log of the folder is a build, an extraction or a run that feeds a gate or a document, and a test fails when a log has none of these roles (`log_roles`).",
        "",
    ])


def ensemble_caveat(log: Path | None = None) -> str:
    """The registered consequence of the conformer check (job 273_conformer_bias): when the ensemble of the pocket block holds a member within 1.0 A of the crystal pose for fewer than 80% of the crystal ligand copies, the
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
        "uses the same corrected t. The first two tables read idea by idea, with the numbers; the list of every verdict line follows.",
        "",
        *idea_tables(),
        ensemble_caveat(),
        "",
        "## Every gate that was not met (the verdict lines of the logs)",
        "",
        "<details><summary>All verdict lines, with the exploratory second looks and the controls that print the same lines</summary>",
        "",
        table(gate, ["log", "gate", "verdict"]),
        "",
        "</details>",
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
        "What the work adds is listed in the section *Innovation summary*, claim by claim; each claim names the section of the notebook that explains it from first principles and computes its number. The public edition of the notebook opens in a browser at https://mohamedtalaat-gif.github.io/CYPMode/notebook/ (the files are too large for GitHub to display).",
        "",
        summary_section(),
        innovation_section(),
        "## 1. Scope and what is not claimed",
        "",
        "Two tracks of the OpenADMET CYP blind challenge: time-dependent-inhibition (TDI) classification for CYP3A4 and CYP2D6, and",
        "CYP3A4 protein-ligand structure prediction. All performance figures are cross-validated estimates on the public training data;",
        "the score that the portal gives an uploaded entry is a separate measurement on the hidden labels (limitations L1 and L2). Novelty is claimed only in the form \"not",
        "found in a search of the sources named in section 12 on the search date\"; every component has published relatives.",
        "",
        "## 2. Data and the label rule",
        "",
        "The TDI label is a threshold of two measured pIC50 values, positive when `s = t - max(d, 4) > log10 2` (t: TDI-condition arm,",
        "d: direct-inhibition arm, floor 4). The rule reproduces every training label exactly (audit section 4), including the",
        "convention for compounds without a direct measurement; those rows follow a different convention than the scored test set and",
        "are excluded from the CYP3A4 pool (audit section 4 gives the counts). The organisers confirmed the convention on the challenge Discord on 2026-10-29: a compound with a TDI arm and no direct arm",
        "cannot show a shift, so it is labelled negative, and every test compound has both arms.",
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
        stack_lead(final.read_text() if final.exists() else "", json.loads((FINAL_TDI / "manifest.json").read_text()) if (FINAL_TDI / "manifest.json").exists() else None),
        "",
        re.sub(r"^(#+) ", lambda m: "#" * (len(m.group(1)) + 2) + " ", final.read_text(), flags=re.MULTILINE) if final.exists() else pending("job 090"),
        "",
        "## 7. Selection-aware CYP2D6 model",
        "",
        selection_lead(),
        "",
        "The model is fitted on the all-isoform panel block (found by `library_blocks.py`). Results (15 folds, nested):",
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
        "Choice of the final CYP2D6 file (job 016, `cypmode/challenge/selection_decision.py`, a procedure fixed before any upload): the deployment table when the top share of the scored population that the final build calls is called, the evidence on the share of never-promoted compounds, and the live board read as a detector of the regime:",
        "",
        log_lines("selection_decision_CYP2D6.log", r".", "job 016", 70),
        "",
        leaderboard_scores(),
        "## 8. Physics-, SAR-, committee-, embedding-, cross-isoform and weighting-based additions",
        "",
        "Each addition is a block or candidate group tested alone and inside the assembled stack; the rule is corrected p < 0.05 at both",
        "confidence cut-offs. " + additions_lead(),
        "",
        "Verdicts:",
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
        calibration_lead(),
        "",
        log_lines("calibration.log", r"CYP|gate|ece|isotonic|platt|raw", "job 030", 20),
        "",
        "## 10. Test-like holdout",
        "",
        holdout_lead(),
        "",
        log_lines("block_holdout.log", r"block|agreement|committee|best candidate", "job 045", 12),
        "",
        "## 11. Structure track",
        "",
        structure_lead(),
        "",
        benchmark_paragraph(),
        "",
        structure_board_lead(),
        "",
        "Gate verdicts of the structure levers (restraint, diversity, selection, engines, expected-improvement sampling, heme-frame prior, ligand dynamics):",
        "",
        table([r for r in gate_rows() if r[1] and not re.match(r"CYP", r[1])], ["log", "gate", "verdict"]),
        "",
        "Final recipe and zip: " + recipe_summary(manifest),
        "",
        "Force-field relaxation of the chosen poses: the gate on the poses of ligands with public crystal poses, then its use on the 20 submitted poses under the rule registered in `relax_final.py`:",
        "",
        log_lines("relaxation_gate.log", r"poses passing|margin \(|LDDT-PLI change|STRUCTURE GATE", "job 218", 6),
        "",
        log_lines("relax_final.log", r"STRUCTURE RELAXATION|not adopted", "job 243", 8),
        "",
        "## 12. Negative results",
        "",
        *lessons_lead(),
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
        map_paragraph(),
        "",
        "Sources credited (published, linked): OpenADMET CYP blind challenge and its official LightGBM baseline; the OpenADMET blog post",
        "\"Cofolding? I hardly know her!\" for the co-folding accuracy figure; repository `openadmet-cyp-structure`",
        "(commit 8161934f8bc9596ae4b820199c42bcc35890c659, https://github.com/xX-its-amit-Xx/openadmet-cyp-structure/tree/8161934f8bc9596ae4b820199c42bcc35890c659,",
        "read on 2026-11-05) for the idea of a donor-nitrogen restraint: its README calls a forced Fe-donor contact \"the lever\" and its plan of 2026-09-10 puts it first",
        "among the steps; a later note of the same repository reports that forcing the contact on 87 ligands was \"a null, slightly negative\" and drops it,",
        "and the measurements of the project are in section 11.6 of the notebook; Boltz-2; OpenFold3; PoseBusters; RDKit; LightGBM; scikit-learn; tblite / GFN2-xTB; CheMeleon (Zenodo",
        "10.5281/zenodo.15426600); Nadeau and Bengio, \"Inference for the generalization error\" (2003) for the corrected resampled t.",
        "Nearest published work for the selection-aware model: selection bias in compound testing and its effect on QSAR (SAR and QSAR in",
        "Environmental Research 2019, PubMed 31547686); Heckman selection models (BMC Medical Research Methodology 2019,",
        "doi 10.1186/s12874-019-0840-7); predicting which primary-screen hits yield dose-response potencies (arXiv 2608.26538);",
        "censored regression labels in drug discovery (arXiv 2409.04313); label noise and soft labels (arXiv 2104.02872).",
        "",
        *data_terms_lines(),
    ])


def published_lines() -> list[str]:
    """What the release holds, said as it is: before the end of the challenge the files that show a result for a blinded compound are held back, after it they are in the release (decided by the block of the ignore file that holds them back)."""
    if "HELD BACK UNTIL AFTER THE DEADLINE" in (ROOT / ".gitignore").read_text():
        return [
            "- **What is published after the challenge closes.** This release is complete: the code, the method, the reports and the checks are all in it, and the code regenerates every submitted file. The only files that are not in it are the ones that show a",
            "  result for a blinded compound: the submitted files, the predictions and the structures of the blinded compounds, and the figures, the film and the complete edition of the notebook that display them; the public edition of the notebook is in it. A full",
            "  release with those files is published after the challenge has closed (2026-11-03 23:59 UTC). The SHA-256 of each submitted file is in the section *What is submitted*, so that each file can be matched to this report when it is published.",
        ]
    return [
        "- **What is published.** This release is complete, the challenge having closed on 2026-11-03 at 23:59 UTC: the code, the method, the reports and the checks, and the files that show a result for a blinded compound (the submitted files, their predictions and structures, the figures, the film and the complete edition of the notebook).",
        "  The SHA-256 of each submitted file is in the section *What is submitted*, so that each file in the release can be matched to this report.",
    ]


def data_terms_lines() -> list[str]:
    """The terms of the sources and models behind the data record, as the policy of the record states them (section 14)."""
    from cypmode import data_record

    rows = data_record.terms_read()
    intro = (
        "**Terms behind the data record.** The files that the repository does not hold are packed by `scripts/fetch_data.py` (policy in `data_record/record.json`) and are under CC BY 4.0. The licence of every source and model behind them was read where it is published, "
        "and a statement that rests on the word of its authors and not on a licence file is said so; a file is left out of the record when the terms of what produced it do not allow it to be redistributed under that licence, or when none are stated."
    )
    lines = [intro, "", "| source | terms as read | where read | in the data record |", "| --- | --- | --- | --- |"]
    lines += [f"| {r['source']} | {r['terms']} | {r['read']} | {r['in the record']} |" for r in rows]
    return [*lines, ""]


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
