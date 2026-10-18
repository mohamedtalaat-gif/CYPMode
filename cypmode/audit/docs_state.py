"""The generated blocks of README.md and CYPMode_Architecture_Specification.md: the final stack, the verdicts of the gates and the structure recipe.

The text around the blocks is written by hand. Between a pair of markers

    <!-- generated:NAME begin (...) -->
    <!-- generated:NAME end -->

this module writes tables that are copied by code from the final build (``data/tdi_track/outputs/final``), from the logs of the gates and from the structure recipe, so that no number in them is typed. A
value whose source does not exist yet is written as ``(pending: <job>)``. ``python -m cypmode.audit.docs_state`` rewrites the blocks; ``--check`` regenerates them in memory and fails if a block on disk
differs, which is the claim-to-evidence check for these two files (a hand-edited number is a difference). Aggregate figures only: nothing here names a blinded compound or a call.

Usage: python -m cypmode.audit.docs_state [--check]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FINAL_TDI = ROOT / "data/tdi_track/outputs/final"
FINAL_STRUCTURE = ROOT / "data/structure_track/outputs/cyp3a4/final"
ISOFORMS = ("CYP3A4", "CYP2D6")
FAMILIES = {
    "knn_tanimoto": "k-nearest neighbours (Tanimoto)", "svm_tanimoto": "support vector machine (Tanimoto kernel)", "lgbm_clf": "LightGBM classifier", "extratrees": "extremely randomised trees",
    "rf": "random forest", "lgbm_reg_shift": "LightGBM regressor of the arm shift", "lgbm_reg_arms": "LightGBM regressors of the two arms",
}
TITLES = {
    "arms_experiment": "two-arm regressors, floor-clipped", "aux_basic_centre": "basic-centre block (protonation states, centre classes)", "aux_chemprop": "plain Chemprop in the stack",
    "aux_emax": "Emax of both arms as a predicted block", "aux_gate_shift": "gate-shift features", "aux_mlp": "multitask network in the stack",
    "aux_mlp_aux": "multitask network with the screen and Emax as auxiliary heads", "aux_nodirect": "compounds without a direct curve", "aux_pxr": "PXR-induction readings as a predicted block",
    "aux_screen": "single-concentration screen as a predicted block", "aux_tabicl": "TabICL in the stack", "aux_tabicl_monroe": "TabICL on Monroe embeddings", "aux_tabicl_chemeleon": "TabICL on frozen CheMeleon embeddings", "aux_tabpfn_chemeleon": "TabPFN on frozen CheMeleon embeddings", "aux_tabpfn": "TabPFN in the stack",
    "committee_label_model": "committee label-process model", "ei_sampling": "expected-improvement sampling of poses", "embeddings": "frozen CheMeleon embeddings",
    "hedged_blend": "hedged blend of the hits-only and the selection-aware stacks", "heme_frame_prior": "heme-frame prior over poses",
    "mbr_lddt_reevaluation": "minimum Bayes risk and restraint judged by LDDT-PLI (exploratory)", "mmp_transfer": "matched-molecular-pair transfer", "monroe_embeddings": "Monroe (GRIT) embeddings",
    "nonneg_stack": "non-negative stacking", "ost_pool_gates": "pool gates repeated with OpenStructure's yardstick", "ost_regate": "restraint confirmation repeated with OpenStructure's LDDT-PLI",
    "ost_vs_ours": "our LDDT-PLI against OpenStructure's", "pair_transfer": "SAR pair transfer", "pair_transfer_physics": "SAR pair transfer with physics", "permutation_control": "negative control (permuted labels)",
    "physics_compact": "compact physics block", "physics_compact_f10": "compact physics block, replicate with 10 outer folds", "physics_experiment": "physics block, tier A",
    "physics_experiment_f10": "physics block tier A, replicate with 10 outer folds", "physics_tierb": "physics block, tier B", "physics_tierb_f10": "physics block tier B, replicate with 10 outer folds",
    "population_physics": "selection-aware CYP2D6 model with physics", "population_screen": "selection-aware CYP2D6 model with the screen", "pose_awareness": "physics, pharmacophore and combined pose terms",
    "relaxation_gate": "force-field relaxation of poses", "replicate_verdict": "replicates with 10 outer folds", "restraint_expansion": "bond restraint confirmed on new ligands",
    "selection_aware_v2": "selection-aware model, second version", "selection_validation": "minimum Bayes risk selection and engine consensus", "series_consensus": "consensus of poses across a chemical series",
    "softmax_selector": "softmax-temperature selector", "structure_gates": "restraints and pose-diversity pilots", "test_likeness": "test-likeness weighting", "xiso_transfer": "cross-isoform transfer of the assay arms",
}


STRUCTURE_GATES = {"restraint_bond": "bond restraint", "restraint_contact": "contact restraint"}


def pending(job: str) -> str:
    return f"(pending: {job})"


def table(rows: list[list], columns: list[str]) -> str:
    head = "| " + " | ".join(columns) + " |\n|" + "|".join(" --- " for _ in columns) + "|\n"
    return head + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)


def members_line() -> str:
    from cypmode.challenge.nested_runners import default_candidates

    counts = Counter(c.family for c in default_candidates())
    return ", ".join(f"{FAMILIES.get(f, f)} ({n})" for f, n in counts.items()) + f"; {sum(counts.values())} members in all"


def final_stack_rows() -> list[list] | None:
    report = FINAL_TDI / "final_report.md"
    if not report.exists():
        return None
    text = report.read_text()
    rows = []
    for iso in ISOFORMS:
        section = re.search(rf"^## {iso}\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
        found = re.search(r"Final stack: \*\*(.+?)\*\* \((\d+) candidates\); surviving groups: ([^;]*);", section.group(1)) if section else None
        choice = FINAL_TDI / f"{iso}_stack_choice.csv"
        if not (found and choice.exists()):
            return None
        row = pd.read_csv(choice).set_index("stack").loc[found.group(1)]
        rows.append([iso, found.group(1), found.group(2), found.group(3).strip(), f"{row['auc_z1.0']:.3f}", f"{row['auc_z1.96']:.3f}", f"{row['mcc_z1.0']:.3f} / {row['mcc_z1.96']:.3f}"])
    return rows


def final_stack_block() -> str:
    rows = final_stack_rows()
    if rows is None:
        return pending("job 090")
    return "\n".join([
        "Final stack per isoform, from the nested cross-validation on the population the challenge scores (z0 is the confidence cut-off of the label weights; MCC is at the rate transferred from the scored prevalence):",
        "",
        table(rows, ["isoform", "stack", "members", "groups added by a met gate", "AUC (z0 1.0)", "AUC (z0 1.96)", "MCC (z0 1.0 / 1.96)"]),
        "",
        f"Members of the default stack: {members_line()}.",
    ])


def verdict_cell(entries: list[tuple[str, str]]) -> str:
    if not entries:
        return "-"
    verdicts = {v for _, v in entries}
    if len(verdicts) == 1:
        return verdicts.pop()
    return "; ".join(f"{name} {v}" for name, v in entries)


def verdict_rows() -> list[list]:
    from cypmode.audit.method_report import gate_rows

    groups: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for log, gate, verdict in gate_rows():
        stem = re.sub(r"(_CYP3A4|_CYP2D6|_addition)?\.log$", "", log)
        iso = re.match(r"(CYP\d\w+):", gate)
        column = iso.group(1) if iso else "structure"
        label = re.sub(r"^CYP\d\w+:\s*", "", gate).split(",")[0].split(" corrected")[0].strip() if iso else gate
        groups.setdefault(stem, {}).setdefault(column, []).append((label, verdict))
    rows = []
    for stem in sorted(groups, key=lambda s: TITLES.get(s, s)):
        by = groups[stem]
        rows.append([TITLES.get(stem, stem.replace("_", " ")), verdict_cell(by.get("CYP3A4", [])), verdict_cell(by.get("CYP2D6", [])), verdict_cell(by.get("structure", []))])
    return rows


def gate_verdicts_block() -> str:
    from cypmode.audit.method_report import gate_rows

    gates = gate_rows()
    if not gates:
        return pending("the logs of the gates")
    met = [(log, gate) for log, gate, verdict in gates if verdict == "MET"]
    names = []
    for log, gate in met:
        stem = re.sub(r"(_CYP3A4|_CYP2D6|_addition)?\.log$", "", log)
        names.append(TITLES.get(stem, stem) if stem not in ("structure_gates",) else STRUCTURE_GATES.get(gate.split(" ")[0], gate.split(" ")[0]))
    rows = verdict_rows()
    stack = final_stack_rows()
    if stack is None:
        added = pending("job 090")
    else:
        groups = {r[0]: r[3] for r in stack}
        added = "none was" if all(g == "none" for g in groups.values()) else "; ".join(f"{iso}: {g}" for iso, g in groups.items())
    return "\n".join([
        (
            f"Every addition was judged by a rule fixed before its result. {len(gates)} gate verdicts from {len(rows)} experiments: {len(met)} MET and {len(gates) - len(met)} NOT MET. "
            f"The gates that were met: {', '.join(sorted(set(names)))}. A MET verdict on a negative control means that the pipeline gives chance on permuted labels, as it must. "
            f"An addition to the TDI stack counts only when its gate is met alone and inside the stack; groups added to the final stack: {added}."
        ),
        "",
        "<details><summary>All verdicts</summary>",
        "",
        table(rows, ["experiment", "TDI CYP3A4", "TDI CYP2D6", "structure"]),
        "",
        "</details>",
    ])


def structure_block() -> str:
    recipe = FINAL_STRUCTURE / "recipe.json"
    if not recipe.exists():
        return pending("job 180")
    reasons = json.loads(recipe.read_text()).get("recipe", {}).get("reasons", [])
    lines = [f"- {reason}" for reason in reasons] or [pending("job 180")]
    stamp = FINAL_STRUCTURE / "relaxation" / "stamp.json"
    if stamp.exists():
        s = json.loads(stamp.read_text())
        rule = s["rule"]
        lines.append(f"- force-field relaxation of the chosen poses in the frozen pocket (gate met on 30 crystal-anchored ligands): {s['adopted']} of {s['of']} poses replaced under a rule fixed before the use "
                     f"(both poses pass the 22 checks, the smallest relative distance to the receptor rises and stays at or above {rule['margin_floor']}, the ligand moves by at most {rule['max_movement_A']} A, "
                     f"a coordinated iron-nitrogen distance changes by at most {rule['max_fe_n_change_A']} A); smallest relative distance {s['smallest_margin_before']:.3f} to {s['smallest_margin_after']:.3f}")
    else:
        lines.append(f"- force-field relaxation of the chosen poses: {pending('job 243')}")
    return "\n".join(["How the 20 poses of the submitted zip were chosen, from the verdicts of the gates and the recipe that reads them (`cypmode/validation/final_recipe.py`):", "", *lines])


BLOCKS = {"final-stack": final_stack_block, "gate-verdicts": gate_verdicts_block, "structure-recipe": structure_block}
TARGETS = {"README.md": ("final-stack", "gate-verdicts", "structure-recipe"), "CYPMode_Architecture_Specification.md": ("final-stack", "structure-recipe")}


def pattern(name: str) -> re.Pattern:
    return re.compile(rf"(<!-- generated:{re.escape(name)} begin[^>]*-->\n)(.*?)(\n<!-- generated:{re.escape(name)} end -->)", re.DOTALL)


def render(text: str, names: tuple[str, ...], blocks: dict | None = None) -> str:
    """The text with the content between the markers of every named block replaced; a missing marker pair is an error."""
    blocks = blocks or BLOCKS
    for name in names:
        match = pattern(name).search(text)
        if match is None:
            raise ValueError(f"the markers of the block '{name}' are missing")
        body = blocks[name]()
        text = text[:match.start()] + match.group(1) + body + match.group(3) + text[match.end():]
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if a block on disk differs from the regenerated one")
    args = parser.parse_args()
    bad = 0
    for filename, names in TARGETS.items():
        path = ROOT / filename
        current = path.read_text()
        new = render(current, names)
        if args.check:
            if new != current:
                bad += 1
                print(f"{filename}: a generated block differs from its source (run python -m cypmode.audit.docs_state)")
            else:
                print(f"{filename}: generated blocks match their sources")
        else:
            path.write_text(new)  # rewritten even when unchanged, so that the file is newer than the outputs it describes
            print(f"{filename}: rewrote {', '.join(names)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
