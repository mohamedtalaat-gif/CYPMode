# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The kind of idea each row of the gate ledger tested, shared by the exploration figure and by the exploration section of the method report.

A ledger row is a pre-registered verdict found in a run log (``method_report.gate_rows``); its family is the stem of the log, and the kind groups the families by what they tried.
Logs of exploratory second looks (``exploratory_*``) and the negative control are not pre-registered gates and are left out of the map.
"""

from __future__ import annotations

import re

import pandas as pd

KINDS = {  # the family of a ledger row (the stem of its log) -> the kind of idea it tested
    "data usage": ("aux_screen", "aux_emax", "aux_pxr", "aux_ext_cont", "aux_ext_tox", "aux_ext_adme", "aux_nodirect", "test_likeness"),
    "representation and physics": ("aux_pocket_cyp3a4", "aux_pocket_cyp2d6", "aux_site_cyp3a4", "aux_site_cyp2d6", "aux_hypothesis", "embeddings", "monroe_embeddings", "physics_experiment", "physics_compact", "physics_tierb", "population_physics", "aux_basic_centre", "aux_atomjepa", "aux_atomjepa_head"),
    "learners and stacking": ("aux_tabicl", "aux_tabpfn", "aux_tabicl_monroe", "aux_tabicl_chemeleon", "aux_tabpfn_chemeleon", "aux_mlp", "aux_mlp_aux", "aux_chemprop", "aux_chemeleon", "aux_chemprop_foundation", "aux_neural", "nonneg_stack", "softmax_selector"),
    "label process and selection": ("selection_aware_v2", "hedged_blend", "population_screen", "arms_experiment", "committee_label_model", "aux_gate_shift", "xiso_transfer", "pair_transfer",
                                    "pair_transfer_physics", "mmp_transfer"),
    "structure levers": ("structure_gates", "selection_validation", "selection_validation_bioemu2", "pose_awareness", "pose_pharmacophore_model", "ligand_dynamics", "series_consensus", "heme_frame_prior", "ei_sampling", "restraint_expansion", "ost_pool_gates", "ost_regate", "ost_vs_ours", "relaxation_gate"),
    "replicates": ("replicate_verdict",),
}

TRIED = {  # one line on what each kind tried (the numbers of the table next to it are generated)
    "data usage": "the organisers' single-concentration screen and Emax, the PXR induction readings, the NCATS and Tox21 panels, predicted logD and protein binding, the compounds without a direct curve, the unlabeled test structures as weights",
    "representation and physics": "frozen embeddings of pretrained networks (CheMeleon, Monroe, Atom-JEPA), semi-empirical quantum chemistry at four levels, the basic-centre block for CYP2D6",
    "learners and stacking": "tabular foundation models (TabICL, TabPFN, alone and on the embeddings), a multitask network, plain and fine-tuned Chemprop, the OpenADMET pretraining weights, a non-negative and a softmax blend",
    "label process and selection": "a model of how the labels came to be (the arms, the gate, the selection-aware CYP2D6 model and its blends), matched molecular pairs, cross-isoform transfer, committees of label models",
    "structure levers": "restraint of the iron-nitrogen contact, conformer, seed, template and step-scale levers, minimum-Bayes-risk and other rules for choosing a pose, a second co-folding engine, expected-improvement sampling, the heme-frame prior",
    "replicates": "the same comparison on ten outer folds for the physics additions and for the neural pair",
}


def stem_of(log: str) -> str:
    """The family of a log: its name without the isoform, the replicate suffix and the extension."""
    return re.sub(r"(_f10)?\.log$", "", re.sub(r"_(CYP3A4|CYP2D6)(?=(_f10)?\.log)", "", log))


def totals() -> tuple[int, int, int]:
    """(verdicts, families, met) of the pre-registered ledger: the one count of what was tried that every document quotes. The negative control and the exploratory second looks print GATE lines too and are not in it."""
    frame = ledger_rows()
    return len(frame), int(frame["stem"].nunique()), int((frame["verdict"] == "MET").sum())


def ledger_rows() -> pd.DataFrame:
    """One row per pre-registered verdict: the log it was found in, the name of the gate, the family (stem of the log), its kind and the verdict."""
    from cypmode.audit.method_report import gate_rows

    kind_of = {stem: kind for kind, stems in KINDS.items() for stem in stems}
    rows = []
    for log, name, verdict in gate_rows():
        if "exploratory" in name or log.startswith(("permutation_control", "exploratory")):
            continue
        stem = stem_of(log)
        rows.append({"log": log, "name": name, "stem": stem, "kind": kind_of.get(stem, "other"), "verdict": verdict})
    return pd.DataFrame(rows, columns=["log", "name", "stem", "kind", "verdict"])


def ledger_by_kind() -> pd.DataFrame:
    """One row per pre-registered verdict: the family (stem of the log), its kind and the verdict."""
    return ledger_rows()[["stem", "kind", "verdict"]]
