# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Every number shown in the animation, read from the project's own final files when the video is rendered."""

import hashlib
import importlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.audit.method_report import gate_rows

ROOT = Path(__file__).resolve().parents[2]
FINAL_TDI = ROOT / "data/tdi_track/outputs/final"
STRUCTURE = ROOT / "data/structure_track/outputs/cyp3a4"
TRAIN = ROOT / "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST = ROOT / "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
STORE = ROOT / "data/tdi_track/outputs/caches/feature_store.npz"


@dataclass(frozen=True)
class Facts:
    n_train: int
    n_test: int
    n_columns: int
    n_models: int
    gates: int
    gates_not_met: int
    auc: dict
    mcc: dict
    poses: int
    pose_failures: int
    audit_reproducible: bool
    zip_hash: str


def stack_row(isoform: str) -> pd.Series:
    return pd.read_csv(FINAL_TDI / f"{isoform}_stack_choice.csv").query("stack == 'default'").iloc[0]


CONTROLS = ("permutation_control",)  # checks of the pipeline itself, not ideas: they are not rows of the ledger


def gate_counts() -> tuple[int, int]:
    """(gates judged, gates not met), from the same ledger the method report prints; the exploratory re-scorings were not declared beforehand and are not counted."""
    verdicts = [verdict for log, name, verdict in gate_rows() if "exploratory" not in name and not log.startswith(CONTROLS)]  # a control checks the pipeline; it is not an idea
    return len(verdicts), verdicts.count("NOT MET")


def feature_columns() -> int:
    """Columns of the feature store: every block that describes a compound."""
    store = np.load(STORE, allow_pickle=True)
    return int(sum(store[name].shape[1] for name in store.files if name != "smiles"))


def to_javascript(facts: Facts, real: dict | None = None) -> str:
    """The facts, and the real data the scenes draw, as the ``facts.js`` the animation page loads."""
    return "window.FACTS = " + json.dumps(asdict(facts), indent=2) + ";\nwindow.REAL = " + json.dumps(real or {}, separators=(",", ":")) + ";\n"


def load() -> Facts:
    poses = pd.read_csv(STRUCTURE / "posebusters_validity_official20.csv")
    audit_log = (STRUCTURE / "logs/audit_run.log").read_text()
    gates, not_met = gate_counts()
    zip_path = STRUCTURE / "final/structures_final.zip"
    return Facts(
        n_train=len(pd.read_csv(TRAIN, usecols=["Molecule_Name"])),
        n_test=len(pd.read_csv(TEST, usecols=["Molecule_Name"])),
        n_columns=feature_columns(),
        n_models=int(stack_row("CYP3A4")["n_candidates"]),
        gates=gates,
        gates_not_met=not_met,
        auc={iso: float(stack_row(iso)["auc_z1.0"]) for iso in ("CYP3A4", "CYP2D6")},
        mcc={iso: float(stack_row(iso)["mcc_z1.0"]) for iso in ("CYP3A4", "CYP2D6")},
        poses=len(poses),
        pose_failures=int((~poses["all_passed"].astype(bool)).sum()),
        audit_reproducible="NOT REPRODUCIBLE" not in audit_log and "REPRODUCIBLE" in audit_log,
        zip_hash=hashlib.sha256(zip_path.read_bytes()).hexdigest()[:8],
    )


# -- real data drawn by the scenes ----------------------------------------------------------------------------------------------------------

EXAMPLE = "OCNT-2395436"  # the first official structure ligand: it has a Type II donor, a TDI measurement and a predicted pose, so one real compound can be followed through the film
GATE_LABELS = {
    "selection_aware_v2": "selection-aware CYP2D6 model", "hedged_blend": "hedged blend of two stacks", "pair_transfer": "SAR pair model", "mmp_transfer": "matched molecular pairs",
    "committee_label_model": "label-process committee", "embeddings": "foundation-model embeddings", "test_likeness": "test-like weighting", "arms_experiment": "floor-clipped assay arms",
    "xiso_transfer": "cross-enzyme transfer", "physics_experiment": "xTB physics features", "physics_compact": "compact physics set", "physics_tierb": "Boltzmann physics, tier B",
    "population_physics": "physics in the selection model", "pair_transfer_physics": "SAR pairs on physics", "replicate_verdict": "10-fold replicate", "structure_gates": "structure levers",
    "selection_validation": "pose choice by minimum Bayes risk", "pose_awareness": "physics terms in pose choice", "series_consensus": "SAR term for poses", "heme_frame_prior": "heme-frame pose prior",
    "ei_sampling": "extra poses by disagreement", "restraint_expansion": "restraint on 20 more ligands",
    "aux_gate_shift": "hit gate times test shift", "aux_nodirect": "labels without a direct curve", "aux_screen": "single-concentration screen", "nonneg_stack": "non-negative stacking",
    "aux_basic_centre": "protonation and basic-centre block", "aux_emax": "predicted Emax readings", "aux_pxr": "PXR induction readings", "aux_ext_cont": "continuous external panel", "aux_tabicl": "TabICL in-context model",
    "aux_tabicl_monroe": "Monroe embeddings in TabICL", "aux_tabicl_chemeleon": "CheMeleon embeddings in TabICL", "aux_tabpfn_chemeleon": "CheMeleon embeddings in TabPFN", "aux_mlp": "multitask network", "aux_mlp_aux": "multitask with screen heads", "aux_chemprop": "plain Chemprop", "aux_chemeleon": "CheMeleon fine-tuned", "aux_neural": "multitask network with Chemprop", "monroe_embeddings": "Monroe embeddings in LightGBM",
}


def _gate_families() -> list[dict]:
    families: dict[str, dict] = {}
    for log, name, verdict in gate_rows():
        if "exploratory" in name:
            continue
        stem = re.sub(r"(_f10)?\.log$", "", re.sub(r"_(CYP3A4|CYP2D6)(?=(_f10)?\.log)", "", log))
        if stem in CONTROLS:
            continue
        entry = families.setdefault(stem, {"label": GATE_LABELS.get(stem, stem.replace("_", " ")), "met": 0, "not_met": 0})
        entry["met" if verdict == "MET" else "not_met"] += 1
    return list(families.values())


def _molecule() -> dict:
    """The example compound as a flat drawing (atoms and bonds), its Morgan bits, a few descriptors and its two measured potencies."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors

    smiles = pd.read_csv(ROOT / "data/structure_track/inputs/structure_track_test_blinded.csv").set_index("Molecule_Name").loc[EXAMPLE, "SMILES"]
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])
    xy = (xy - xy.mean(axis=0)) / max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
    bits = sorted(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).GetOnBits())
    train = pd.read_csv(TRAIN).set_index("Molecule_Name")
    row = train.loc[EXAMPLE] if EXAMPLE in train.index else None
    return {
        "name": EXAMPLE, "smiles": smiles,
        "atoms": [[round(float(x), 4), round(float(-y), 4), a.GetSymbol()] for (x, y), a in zip(xy, mol.GetAtoms(), strict=True)],
        "bonds": [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondTypeAsDouble())] for b in mol.GetBonds()],
        "bits": [int(b) for b in bits],
        "descriptors": {"molecular weight": round(Descriptors.MolWt(mol), 1), "logP": round(Descriptors.MolLogP(mol), 2), "polar surface area": round(rdMolDescriptors.CalcTPSA(mol), 1),
                        "H-bond donors": Lipinski.NumHDonors(mol), "H-bond acceptors": Lipinski.NumHAcceptors(mol), "rotatable bonds": Lipinski.NumRotatableBonds(mol), "heavy atoms": mol.GetNumHeavyAtoms()},
        "direct": None if row is None else float(row["CYP3A4_pIC50_direct_inhibition"]), "tdi": None if row is None else float(row["CYP3A4_pIC50_TDI_condition"]),
    }


def _scatter(n: int = 420) -> dict:
    """A seeded sample of the CYP3A4 training compounds with both measurements: direct pIC50 (floored at 4), TDI-condition pIC50 and the label."""
    from cypmode.challenge.tdi_targets import training_view

    both = training_view(pd.read_csv(TRAIN), "CYP3A4").query("population == 'both_arms'")
    sample = both.sample(n=n, random_state=7)
    return {"points": [[round(float(max(d, 4.0)), 3), round(float(t), 3), int(bool(lab))] for d, t, lab in zip(sample["direct"], sample["tdi"], sample["label"], strict=True)],
            "n": len(both), "positives": int(both["label"].astype(bool).sum())}


def _curves() -> dict:
    """The mean ROC curve of the default stack on the outer folds (rebuilt from the cached predictions by the figure code), per enzyme, and its mean AUC."""
    from sklearn.metrics import roc_auc_score, roc_curve

    sys.path[:0] = [str(ROOT / "docs/figures")]
    figure_stack = importlib.import_module("gen._stack")

    out = {}
    grid = np.linspace(0, 1, 61)
    for iso in ("CYP3A4", "CYP2D6"):
        folds = figure_stack.evaluation(iso)[2]
        tprs = [np.interp(grid, *roc_curve(y, score, sample_weight=w)[:2]) for y, score, w, _ in folds]
        out[iso] = {"fpr": [round(float(g), 4) for g in grid], "tpr": [round(float(t), 4) for t in np.mean(tprs, axis=0)],
                    "auc": round(float(np.mean([roc_auc_score(y, score, sample_weight=w) for y, score, w, _ in folds])), 4)}
    return out


def _physics() -> list[dict]:
    sys.path[:0] = [str(ROOT / "docs/figures")]
    figure_physics = importlib.import_module("gen.physics")

    auc = figure_physics.single_column_auc("CYP3A4")
    best = (auc - 0.5).abs().sort_values(ascending=False).head(7).index
    return [{"column": c.replace("phys_", "").replace("_", " "), "auc": round(float(auc[c]), 3)} for c in best]


def _structure_checks() -> dict:
    table = pd.read_csv(STRUCTURE / "posebusters_validity_official20.csv")
    checks = [c for c in table.columns if c not in ("compound", "all_passed", "model_index")]
    recipe = json.loads((STRUCTURE / "final/recipe.json").read_text())
    return {"checks": [c.replace("_", " ") for c in checks], "passed": [int(table[c].astype(bool).sum()) for c in checks], "n": len(table), "donors": [r["ligand"] for r in recipe["ligands"] if r["type_ii_donor"]]}


def _families() -> dict:
    from collections import Counter

    from cypmode.challenge.nested_runners import default_candidates

    return dict(Counter(c.name.split("_")[0] for c in default_candidates()))


def _repo_and_audit() -> dict:
    tests = sorted((ROOT / "tests").glob("test_*.py"))
    report = ROOT / "audit/AUDIT_REPORT.md"
    return {"modules": len([p for p in (ROOT / "cypmode").rglob("*.py") if p.name != "__init__.py"]), "test_files": len(tests), "test_functions": sum(len(re.findall(r"^def test_", p.read_text(), re.MULTILINE)) for p in tests),
            "jobs": len(list((ROOT / "queue/jobs").glob("*.sh"))), "audit_hash": hashlib.sha256(report.read_bytes()).hexdigest()[:12]}


def _scout() -> dict:
    """The sources the data scout went through and what each shares with the training set and with the blinded test set.

    Only the identifier column of each file is read, never a measurement: the values of a blinded test compound are not looked at, not even for a picture.
    """
    from cypmode.challenge import official_extra as oe

    train = set(pd.read_csv(TRAIN, usecols=["Molecule_Name"]).Molecule_Name)
    test = set(pd.read_csv(TEST, usecols=["Molecule_Name"]).Molecule_Name)
    octant_ids = pd.read_parquet(ROOT / "data/tdi_track/inputs/octant_cache/inhibition.parquet", columns=["ocnt_batch"]).ocnt_batch
    sources = [
        ("Octant inhibition release", octant_ids.map(oe.compound_of_batch)),
        ("single-concentration screen", pd.read_csv(oe.SCREEN_FILE, usecols=["Molecule_Name"]).Molecule_Name),
        ("Emax of both arms", pd.read_csv(oe.EMAX_FILE, usecols=["Molecule_Name"]).Molecule_Name),
        ("direct inhibition", pd.read_csv(oe.INHIBITION_FILE, usecols=["Molecule_Name"]).Molecule_Name),
        ("reactivity phenotyping", pd.read_csv(oe.REACTIVITY_FILE, sep="\t", usecols=["ocnt_batch"]).ocnt_batch.map(oe.compound_of_batch)),
        ("PXR single concentration", pd.read_csv(oe.PXR_SINGLE_FILE, usecols=["OCNT_ID"]).OCNT_ID),
    ]
    rows = []
    for label, ids in sources:
        compounds = set(ids)
        rows.append({"label": label, "rows": len(ids), "compounds": len(compounds), "in_training": len(compounds & train), "in_test": sorted(compounds & test)})
    return {"sources": [{**r, "in_test": len(r["in_test"])} for r in rows], "excluded": sorted({c for r in rows for c in r["in_test"]})}


FOUNDATION = (  # (label of the pretrained model, stem of the log of its experiment, enzymes that are not part of its design); the experiment logs are written by the queue as the jobs finish
    ("CheMeleon embeddings", "embeddings", ()), ("TabICL", "aux_tabicl", ()), ("CheMeleon fine-tuned", "aux_chemeleon", ("CYP2D6",)), ("OpenADMET pretraining weights", "aux_chemprop_foundation", ("CYP2D6",)), ("TabPFN", "aux_tabpfn", ()),
    ("Atom-JEPA", "aux_atomjepa_head", ()), ("Monroe embeddings", "monroe_embeddings", ()), ("Monroe in TabICL", "aux_tabicl_monroe", ()), ("OpenADMET surrogates", "aux_ext_adme", ()),
)
STACKED = re.compile(r"z0=1\.0\s+stacked[^:]*:\s+AUC ([+-]\d\.\d+) \(p=([\d.]+)")
LOG_DIR = ROOT / "data/structure_track/outputs/cyp3a4/logs"


def _foundry() -> list[dict]:
    """Each pretrained model that was added to the stack: the change in AUC it brought at the first confidence cut (both enzymes), its p value and the verdict.

    "pending" while its job runs, "screened" when only the 5-fold first stage of a staged design exists (``stage1_<model>_<enzyme>.txt``), "not run" when the model was dropped for cost (the first stage
    wrote "not run" instead of a comparison) or when the enzyme is not part of the design of the model (CYP2D6 for the fine-tuned networks, dropped on 2026-10-16).
    """
    verdicts: dict[str, dict[str, str]] = {}
    for log, _, verdict in gate_rows():
        match = re.match(r"(.+?)_(CYP3A4|CYP2D6)(?:_f10)?\.log$", log)
        if match:
            verdicts.setdefault(match.group(1), {})[match.group(2)] = verdict
    out = []
    for label, stem, dropped in FOUNDATION:
        row = {"label": label, "enzymes": {}}
        for iso in ("CYP3A4", "CYP2D6"):
            if iso in dropped:
                row["enzymes"][iso] = {"verdict": "not run"}
                continue
            path, stage = LOG_DIR / f"{stem}_{iso}.log", LOG_DIR / f"stage1_{stem.removeprefix('aux_')}_{iso}.txt"
            found = STACKED.search(path.read_text(errors="ignore")) if path.exists() else None
            verdict = verdicts.get(stem, {}).get(iso)
            if found and verdict:
                row["enzymes"][iso] = {"delta_auc": float(found.group(1)), "p": float(found.group(2)), "verdict": verdict}
                continue
            stage_text = stage.read_text(errors="ignore") if stage.exists() else ""
            staged = STACKED.search(stage_text)
            if staged:
                row["enzymes"][iso] = {"delta_auc": float(staged.group(1)), "p": float(staged.group(2)), "verdict": "screened"}
            elif "not run" in stage_text:
                row["enzymes"][iso] = {"verdict": "not run"}
            else:
                row["enzymes"][iso] = {"verdict": "pending"}
        out.append(row)
    return out


def real_data() -> dict:
    """Everything the scenes draw from the project's files rather than from imagination."""
    store = np.load(STORE, allow_pickle=True)
    return {"molecule": _molecule(), "scatter": _scatter(), "curves": _curves(), "gates": _gate_families(), "physics": _physics(), "structure": _structure_checks(),
            "blocks": [{"name": n, "columns": int(store[n].shape[1])} for n in store.files if n != "smiles"], "families": _families(), "repo": _repo_and_audit(),
            "scout": _scout(), "foundry": _foundry()}
