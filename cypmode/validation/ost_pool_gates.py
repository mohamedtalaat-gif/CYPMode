"""The gates of the pools repeated with OpenStructure's yardstick (the registered consequence of the verdict of ``ost_official``).

The gates of the structure pools are judged by ``true_rmsd``, our pocket-aligned symmetry-aware RMSD to the crystal pose, and three exploratory gates by our LDDT-PLI. ``ost_pools`` scored every pose
of their tables with OpenStructure, the organisers' scorer. This module writes copies of the pose tables in which ``true_rmsd`` is OpenStructure's BiSyRMSD (the smallest over the crystal copies, as ours
is) and runs each gate again on them, with the same code and the same rule; the exploratory LDDT-PLI gates run on OpenStructure's LDDT-PLI. Nothing the gates wrote before is touched: the copies and the
outcome tables go to ``data/structure_track/outputs/cyp3a4/ost_yardstick/`` and the verdict lines carry the suffix ``_ost``.

Gates repeated: mbr_selection and engine_consensus (``selection_validation``), ei_sampling, heme_frame_prior, pose_physics / pose_pharmacophore / pose_combined (``pose_awareness``), series_consensus,
restraint_bond / restraint_contact and the pilots (``structure_gates``), and the three exploratory LDDT-PLI gates (``mbr_lddt_reevaluation``). The restraint gates of the confirmation ligands were repeated by
``ost_regate``.

``python -m cypmode.validation.ost_pool_gates`` first runs every gate on unchanged copies of the tables and stops if one of the verdicts differs from the one in its original log (a check of this module's
plumbing), then runs them on the OpenStructure copies and prints, for each gate, the verdict with our yardstick and with OpenStructure's, and whether any verdict changed.
"""

from __future__ import annotations

import contextlib
import io
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

from cypmode.validation import ost_pools

OUT = ost_pools.OUT
LOGS = OUT / "logs"
OST_LOG_NAMES = ("selection_validation.log", "heme_frame_prior.log", "ei_sampling.log", "structure_gates.log", "pose_awareness.log", "series_consensus.log", "mbr_lddt_reevaluation.log")
GATE_LINE = re.compile(r"^STRUCTURE GATE (\w+)[^:\n]*: (MET|NOT MET)", re.M)
RESTRAINED_SUFFIX = "-febond"


def verdicts(text: str) -> dict[str, bool]:
    """{gate: MET} of the ``STRUCTURE GATE`` lines of a text (the last line of a gate wins)."""
    return {name: verdict == "MET" for name, verdict in GATE_LINE.findall(text)}


def our_verdicts(logs: Path = LOGS) -> dict[str, bool]:
    """The verdicts of the gates in the logs of their original runs."""
    out: dict[str, bool] = {}
    for name in OST_LOG_NAMES:
        path = logs / name
        if path.exists():
            out |= verdicts(path.read_text(errors="ignore"))
    return out


def write_tables(directory: Path, ost: pd.DataFrame | None, source: Path = OUT) -> dict[str, int]:
    """Copies of the yardstick tables in ``directory``: unchanged, or with ``true_rmsd`` replaced by OpenStructure's BiSyRMSD; returns the number of poses without an OpenStructure value per table."""
    directory.mkdir(parents=True, exist_ok=True)
    missing = {}
    for name in ost_pools.YARDSTICK_TABLES:
        path = source / name
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if ost is not None:
            if "engine" not in table:
                table["engine"] = "boltz"
            joined = table.merge(ost[["compound", "engine", "sample", "ost_bisyrmsd"]], on=["compound", "engine", "sample"], how="left")
            joined["true_rmsd"] = joined["ost_bisyrmsd"]
            missing[name] = int(joined["true_rmsd"].isna().sum())
            table = joined.drop(columns="ost_bisyrmsd")
        table.to_csv(directory / name, index=False)
    return missing


def capture(main, argv: list[str] | None = None) -> str:
    """The text a gate's ``main`` prints (its ``sys.exit`` is turned into a line)."""
    buffer, saved = io.StringIO(), sys.argv
    try:
        if argv is not None:
            sys.argv = ["gate", *argv]
        with contextlib.redirect_stdout(buffer):
            try:
                main()
            except SystemExit as stop:
                print(f"(the gate stopped with exit status {stop.code})")
    finally:
        sys.argv = saved
    return buffer.getvalue()


def run_selection(directory: Path) -> str:
    from cypmode.validation import selection_validation

    return capture(selection_validation.main, ["--table", str(directory / "benchmark_engine_pose_validation.csv"), "--out", str(directory / "selection_validation_outcomes.csv")])


def run_ei_sampling(directory: Path) -> str:
    from cypmode.validation import ei_sampling

    return capture(ei_sampling.main, ["--table", str(directory / "benchmark_engine_pose_validation.csv"), "--out", str(directory / "ei_sampling_outcomes.csv")])


def run_heme_frame_prior(directory: Path) -> str:
    from cypmode.validation import heme_frame_prior

    return capture(heme_frame_prior.main, ["--table", str(directory / "benchmark_engine_pose_validation.csv"), "--out", str(directory / "heme_frame_prior_outcomes.csv")])


def run_pose_awareness(directory: Path) -> str:
    from cypmode.validation import pose_awareness

    return capture(pose_awareness.main, ["--mode", "pooled", "--directory", str(directory), "--out", str(directory / "pose_awareness_poses.csv")])


def run_series(directory: Path) -> str:
    from cypmode.validation import series_consensus

    saved = series_consensus.POSE_TABLE
    series_consensus.POSE_TABLE = directory / "series_pose_validation.csv"
    try:
        return capture(series_consensus.main)
    finally:
        series_consensus.POSE_TABLE = saved


def run_structure_gates(directory: Path) -> str:
    from cypmode.validation import structure_gates

    saved = structure_gates.TABLES, structure_gates.PILOT_POSES, structure_gates.CRYOEM_POSES
    structure_gates.TABLES = {name: directory / path.name for name, path in saved[0].items()}
    structure_gates.PILOT_POSES, structure_gates.CRYOEM_POSES = directory / saved[1].name, directory / saved[2].name
    try:
        return capture(structure_gates.main)
    finally:
        structure_gates.TABLES, structure_gates.PILOT_POSES, structure_gates.CRYOEM_POSES = saved


def run_lddt_gates(poses: pd.DataFrame) -> str:
    """The three exploratory gates of ``mbr_lddt_reevaluation`` on a table of LDDT-PLI per pose (ours, or OpenStructure's)."""
    from cypmode.validation import mbr_lddt_reevaluation as m
    from cypmode.validation.selection_validation import P_VALUE, paired_gate

    benchmark = pd.read_csv(m.BENCHMARK)
    base = dict(zip(benchmark["Molecule_Name"], benchmark["SMILES"], strict=True))
    smiles_of = {**base, **{f"{name}{RESTRAINED_SUFFIX}": smiles for name, smiles in base.items()}}
    lines = []
    selection = m.selection_outcomes(poses, smiles_of)
    stats = paired_gate(-selection["first_boltz"], -selection["mbr_boltz"])
    lines.append(m.gate_line("mbr_lddt_exploratory", stats, f"minimum Bayes risk over the Boltz samples against the first pick, scored by LDDT-PLI, thresholds of mbr_selection (p < {P_VALUE}), exploratory"))
    restraint = m.restraint_outcomes(poses, smiles_of)
    stats = paired_gate(-restraint["first_unguided"], -restraint["first_restrained"])
    lines.append(m.gate_line("restraint_bond_lddt_exploratory", stats, f"first sample with the bond restraint against without it, scored by LDDT-PLI, p < {P_VALUE}, exploratory"))
    stats = paired_gate(-restraint["first_restrained"], -restraint["mbr_restrained"])
    lines.append(m.gate_line("mbr_on_restrained_lddt_exploratory", stats, f"minimum Bayes risk inside the restrained pool against its first sample, scored by LDDT-PLI, p < {P_VALUE}, exploratory"))
    return "\n".join(lines) + "\n"


def lddt_table(ost: pd.DataFrame | None) -> pd.DataFrame:
    """The LDDT-PLI per pose of the exploratory gates: our table, or the same poses with OpenStructure's value."""
    ours = pd.read_csv(OUT / "pose_lddt_to_crystal.csv")
    if ost is None:
        return ours
    joined = ours.drop(columns="lddt_pli").merge(ost[["compound", "engine", "sample", "ost_lddt_pli"]], on=["compound", "engine", "sample"], how="left")
    return joined.rename(columns={"ost_lddt_pli": "lddt_pli"})


def run_all(directory: Path, ost: pd.DataFrame | None) -> dict[str, str]:
    """Every gate once; {label: the text it printed}."""
    write_tables(directory, ost)
    runners = {"selection_validation": lambda: run_selection(directory), "ei_sampling": lambda: run_ei_sampling(directory), "heme_frame_prior": lambda: run_heme_frame_prior(directory),
               "pose_awareness": lambda: run_pose_awareness(directory), "series_consensus": lambda: run_series(directory), "structure_gates": lambda: run_structure_gates(directory),
               "mbr_lddt_reevaluation": lambda: run_lddt_gates(lddt_table(ost))}
    texts = {}
    for label, run in runners.items():
        print(f"running {label} ...", flush=True)
        texts[label] = run()
    return texts


def main() -> int:
    table = pd.read_csv(ost_pools.TABLE) if ost_pools.TABLE.exists() else pd.DataFrame()
    keys = ost_pools.pose_keys()
    scored = table.dropna(subset=["ost_bisyrmsd"]) if len(table) else table
    if len(table) < len(ost_pools.inventory(keys)):
        print(f"OpenStructure has scored {len(table)} of {len(keys)} poses; run ost_pools first")
        return 75
    ours = our_verdicts()
    identity_dir, ost_dir = OUT / "ost_yardstick" / "identity", OUT / "ost_yardstick" / "openstructure"
    texts = run_all(identity_dir, None)
    again = verdicts("\n".join(texts.values()))
    differing = {g: (ours[g], v) for g, v in again.items() if g in ours and ours[g] != v}
    uncompared = sorted(set(again) - set(ours))
    print(f"CHECK of the plumbing: {len(again)} gate verdicts from unchanged copies of the tables, {len(again) - len(uncompared) - len(differing)} equal to the original logs, {len(differing)} different, {len(uncompared)} without an original to compare")
    if differing:
        print(f"CHECK FAILED: {differing}")
        return 1
    print(f"\nOpenStructure values for {len(scored)} of {len(table)} poses (BiSyRMSD)")
    texts = run_all(ost_dir, table)
    repeated = verdicts("\n".join(texts.values()))
    for label, text in texts.items():
        print(f"\n== {label}, yardstick: OpenStructure")
        print(GATE_LINE.sub(lambda m: m.group(0).replace(f"STRUCTURE GATE {m.group(1)}", f"STRUCTURE GATE {m.group(1)}_ost", 1), text).rstrip())
    changed = {g: (ours.get(g), v) for g, v in repeated.items() if g in ours and ours[g] != v}
    print("\nSUMMARY (verdict with our yardstick -> with OpenStructure's):")
    for gate, verdict in sorted(repeated.items()):
        before = ours.get(gate)
        print(f"  {gate}: {'MET' if before else 'NOT MET'} -> {'MET' if verdict else 'NOT MET'}" if before is not None else f"  {gate}: (no original verdict) -> {'MET' if verdict else 'NOT MET'}")
    print(f"STRUCTURE POOL GATES repeated with OpenStructure: {len(repeated)}; verdict changes: " + ("none" if not changed else ", ".join(f"{g} ({'MET' if a else 'NOT MET'} -> {'MET' if b else 'NOT MET'})" for g, (a, b) in changed.items())))
    shutil.rmtree(identity_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
