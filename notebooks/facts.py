"""The numbers the notebook quotes in its running text, read from the project's own files when the notebook is built.

``load()`` returns one ``Facts`` object; Markdown cells refer to its fields as ``{{=F.n_train:,}}``, so a number in a sentence is never typed by hand.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.audit.method_report import gate_rows
from cypmode.challenge.tdi_targets import training_view

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST = ROOT / "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
STORE = ROOT / "data/tdi_track/outputs/caches/feature_store.npz"
FINAL_TDI = ROOT / "data/tdi_track/outputs/final"
STRUCTURE = ROOT / "data/structure_track/outputs/cyp3a4"
ISOFORMS = ("CYP3A4", "CYP2D6")


@dataclass(frozen=True)
class Facts:
    n_train: int
    n_test: int
    n_columns: int
    n_blocks: int
    n_models: int
    labelled: dict = field(default_factory=dict)
    positives: dict = field(default_factory=dict)
    unlabelled: dict = field(default_factory=dict)
    auc: dict = field(default_factory=dict)
    mcc: dict = field(default_factory=dict)
    gates: int = 0
    gates_met: int = 0
    gates_not_met: int = 0
    poses: int = 0
    pose_failures: int = 0
    modules: int = 0
    scripts: int = 0
    jobs: int = 0
    test_files: int = 0
    test_functions: int = 0


def gate_counts() -> tuple[int, int, int]:
    """(judged, met, not met) from the ledger the method report prints; the exploratory re-scorings were not declared beforehand and are not counted."""
    verdicts = [verdict for _, name, verdict in gate_rows() if "exploratory" not in name]
    return len(verdicts), verdicts.count("MET"), verdicts.count("NOT MET")


def load() -> Facts:
    train = pd.read_csv(TRAIN)
    store = np.load(STORE, allow_pickle=True)
    blocks = [name for name in store.files if name != "smiles"]
    labelled, positives, unlabelled = {}, {}, {}
    for iso in ISOFORMS:
        view = training_view(train, iso)
        both = view[view["population"] == "both_arms"]
        labelled[iso], positives[iso] = len(both), int(both["label"].astype(bool).sum())
        unlabelled[iso] = int((view["population"] == "unlabeled").sum())
    stacks = {iso: pd.read_csv(FINAL_TDI / f"{iso}_stack_choice.csv").query("stack == 'default'").iloc[0] for iso in ISOFORMS}
    poses = pd.read_csv(STRUCTURE / "posebusters_validity_official20.csv")
    judged, met, not_met = gate_counts()
    tests = sorted((ROOT / "tests").glob("test_*.py"))
    return Facts(
        n_train=len(train),
        n_test=len(pd.read_csv(TEST, usecols=["Molecule_Name"])),
        n_columns=int(sum(store[b].shape[1] for b in blocks)),
        n_blocks=len(blocks),
        n_models=int(stacks["CYP3A4"]["n_candidates"]),
        labelled=labelled,
        positives=positives,
        unlabelled=unlabelled,
        auc={iso: float(stacks[iso]["auc_z1.0"]) for iso in ISOFORMS},
        mcc={iso: float(stacks[iso]["mcc_z1.0"]) for iso in ISOFORMS},
        gates=judged,
        gates_met=met,
        gates_not_met=not_met,
        poses=len(poses),
        pose_failures=int((~poses["all_passed"].astype(bool)).sum()),
        modules=len([p for p in (ROOT / "cypmode").rglob("*.py") if p.name != "__init__.py"]),
        scripts=len([p for p in (ROOT / "scripts").iterdir() if p.is_file()]),
        jobs=len(list((ROOT / "queue/jobs").glob("*.sh"))),
        test_files=len(tests),
        test_functions=sum(len(re.findall(r"^def test_", p.read_text(), re.MULTILINE)) for p in tests),
    )
