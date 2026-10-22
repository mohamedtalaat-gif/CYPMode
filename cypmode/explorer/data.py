# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""A snapshot of the public results for the explorer: five small tables in ``explorer/public`` and a manifest with the SHA-256 of each.

The explorer (``explorer/cypmode_explorer.py``, a marimo notebook) reads only this folder, so that it runs the same from a clone of the repository, in a browser without a server and
beside the full local tree, and so that what it shows is exactly what this module chose to publish. The tables are read from the run logs and result files that already exist; nothing is
computed again, no model is fitted and no queue job is started.

  ledger.csv      every pre-registered verdict of the gate ledger: the log, the gate, the kind of idea, MET or NOT MET
  paired.csv      every paired comparison that a run log prints: the experiment, the enzyme, the confidence cut z0, the gain in AUC, the corrected p, the folds in which the addition was better
  selectors.csv   the nested cross-validation of the selection rules, one row per repeat and outer fold, for each enzyme
  poses.csv       the poses of the crystal-anchored benchmark, one row per ligand, engine and sample, with the RMSD to the crystal pose
  rules.csv       the RMSD of the pose that each choosing rule submits, per benchmark ligand
  excerpts.csv    the lines of every run log that carry a registered verdict or a paired comparison, so that a gate can be read with its numbers
  oof_<enzyme>.csv.gz  the outer-fold predictions of every complete candidate for the training compounds: one row per compound of the pool, the label and one column per candidate and repeat; the rows of the blinded test set are never in it

Nothing here names a blinded compound: the sources are the logs and the tables of the training compounds and of the public crystal ligands, and ``check_public`` refuses a bundle that holds a
name of the blinded test sets, a file the HELD BACK block of ``.gitignore`` lists, or a table of 750 or 20 rows (the sizes of the two blinded sets).

Usage: python -m cypmode.explorer.data [--out explorer/public]
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

LOGS = Path("data/structure_track/outputs/cyp3a4/logs")
NESTED = Path("data/tdi_track/outputs/nested")
STRUCTURE = Path("data/structure_track/outputs/cyp3a4")
OUT = Path("explorer/public")
ENZYMES = ("CYP3A4", "CYP2D6")
BLINDED_SIZES = (750, 20)
PAIRED = re.compile(r"^\s*z0=([\d.]+)\s+(.+?):\s+AUC ([+-][\d.]+) \(p=([\d.]+)(?:, better (\d+)/(\d+))?\)", re.MULTILINE)
BLINDED_NAME = re.compile(r"OCNT-\d+")


def paired_rows(logs: Path = LOGS) -> pd.DataFrame:
    """Every paired AUC comparison printed by a run log, with the enzyme taken from the name of the log."""
    rows = []
    for path in sorted(Path(logs).glob("*.log")):
        enzyme = next((e for e in ENZYMES if e in path.name), "")
        for z0, comparison, gain, p, better, folds in PAIRED.findall(path.read_text(errors="ignore")):
            rows.append({"log": path.name, "enzyme": enzyme, "z0": float(z0), "comparison": comparison.strip(), "auc_gain": float(gain), "p": float(p),
                         "better": int(better) if better else None, "folds": int(folds) if folds else None})
    return pd.DataFrame(rows, columns=["log", "enzyme", "z0", "comparison", "auc_gain", "p", "better", "folds"])


def selector_rows(nested: Path = NESTED) -> pd.DataFrame:
    """The nested cross-validation of the selection rules of both enzymes, one row per repeat and outer fold."""
    frames = []
    for enzyme in ENZYMES:
        path = Path(nested) / enzyme / "nested_auc.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frames.append(frame.assign(enzyme=enzyme)[["enzyme", "selector", "repeat", "outer", "inner_score", "outer_score", "optimism", "outer_mcc"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["enzyme", "selector", "repeat", "outer", "inner_score", "outer_score", "optimism", "outer_mcc"])


def pose_rows(structure: Path = STRUCTURE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The poses of the benchmark (ligand, engine, sample, RMSD to the crystal pose) and the RMSD of the pose that each rule submits."""
    poses = pd.read_csv(Path(structure) / "benchmark_engine_pose_validation.csv")[["compound", "engine", "sample", "true_rmsd"]]
    rules = pd.read_csv(Path(structure) / "benchmark_selection_outcomes.csv")[["compound", "rule", "rmsd"]]
    return poses, rules


def excerpt_rows(logs: Path = LOGS) -> pd.DataFrame:
    """The lines of every run log that hold a registered verdict or a paired comparison, with the line number."""
    from cypmode.audit.method_report import GATE

    rows = []
    for path in sorted(Path(logs).glob("*.log")):
        for number, line in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
            if GATE.match(line) or PAIRED.match(line):
                rows.append({"log": path.name, "line": number, "text": line.strip()})
    return pd.DataFrame(rows, columns=["log", "line", "text"])


def outer_predictions(enzyme: str, nested: Path = NESTED) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(candidate names, labels of the pool, scores of shape candidates x repeats x pool) from the prediction cache of the nested plan.

    Only the outer-validation tasks are read, so every compound of the pool is scored once per repeat by models that never saw it; the deployment rows of the blinded test set are not read.
    A candidate that lacks one of those tasks is left out.
    """
    from cypmode.challenge.nested import OUTER_REFIT, NestedPlan, PredictionCache

    root = Path(nested) / enzyme
    plan = NestedPlan.load(root / "plan.npz")
    cache = PredictionCache(root / "cache", plan)
    position = {int(row): i for i, row in enumerate(plan.pool)}
    folds = [plan.task(r, k, OUTER_REFIT) for r, k in plan.outer_folds()]
    names, matrices = [], []
    for name in cache.candidates():
        if not all(cache.has(name, task) for task in folds):
            continue
        scores = np.full((plan.repeats, len(plan.pool)), np.nan, dtype=np.float32)
        for task in folds:
            scores[task.repeat, [position[int(row)] for row in task.evaluate]] = cache.get(name, task)
        if np.isfinite(scores).all():
            names.append(name)
            matrices.append(scores)
    return names, plan.y.astype(np.int8), np.stack(matrices) if matrices else np.zeros((0, plan.repeats, len(plan.pool)), dtype=np.float32)


def tables(logs: Path = LOGS, nested: Path = NESTED, structure: Path = STRUCTURE) -> dict[str, pd.DataFrame]:
    from cypmode.audit.ledger_kinds import ledger_rows

    poses, rules = pose_rows(structure)
    return {"ledger": ledger_rows(), "paired": paired_rows(logs), "selectors": selector_rows(nested), "poses": poses, "rules": rules, "excerpts": excerpt_rows(logs)}


def held_back_patterns(gitignore: Path = Path(".gitignore")) -> list[str]:
    """The patterns of the HELD BACK block of ``.gitignore``."""
    text = Path(gitignore).read_text()
    block = text[text.index(">>> HELD BACK"):text.index("<<< HELD BACK")]
    return [line.strip() for line in block.splitlines()[1:] if line.strip() and not line.startswith("#")]  # the first line of the slice is the rest of the opening marker


def check_public(folder: Path = OUT) -> list[str]:
    """What keeps a bundle from being public: a blinded compound name, a held-back path, a table of the size of a blinded set."""
    problems = []
    held = held_back_patterns()
    for path in sorted(Path(folder).glob("*.csv")):
        text = path.read_text()
        if BLINDED_NAME.search(text):
            problems.append(f"{path.name} names a blinded compound")
        for pattern in held:
            literal = pattern.rstrip("/").split("*")[0].rstrip("/")
            if literal and literal in text:
                problems.append(f"{path.name} mentions the held-back path {literal}")
        rows = len(text.strip().splitlines()) - 1
        if rows in BLINDED_SIZES:
            problems.append(f"{path.name} has {rows} rows, the size of a blinded set")
    for path in sorted(Path(folder).glob("oof_*.csv.gz")):
        frame = pd.read_csv(path)
        if len(frame) in BLINDED_SIZES:
            problems.append(f"{path.name} has {len(frame)} rows, the size of a blinded set")
        if frame.columns[0] != "y" or not all(re.fullmatch(r".+\|\d+", column) for column in frame.columns[1:]):
            problems.append(f"{path.name} holds columns beyond the label and the candidate and repeat predictions")
    return problems


def build(out: Path = OUT) -> dict:
    """Write the five tables and the manifest; return the manifest."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, frame in tables().items():
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        manifest[path.name] = {"rows": len(frame), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for enzyme in ENZYMES:
        if (NESTED / enzyme / "plan.npz").exists():
            names, y, scores = outer_predictions(enzyme)
            path = out / f"oof_{enzyme}.csv.gz"
            wide = {"y": y.astype(int)}
            for i, name in enumerate(names):
                for repeat in range(scores.shape[1]):
                    wide[f"{name}|{repeat}"] = scores[i, repeat].round(4)
            pd.DataFrame(wide).to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
            manifest[path.name] = {"candidates": len(names), "repeats": int(scores.shape[1]), "compounds": int(scores.shape[2]), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    manifest = build(Path(args.out))
    problems = check_public(Path(args.out))
    for name, entry in manifest.items():
        print(f"  {name}: " + (f"{entry['rows']} rows" if "rows" in entry else f"{entry['candidates']} candidates x {entry['repeats']} repeats x {entry['compounds']} compounds"))
    if problems:
        raise SystemExit("the bundle is not public: " + "; ".join(problems))
    print("the bundle names no blinded compound and no held-back path")


if __name__ == "__main__":
    from cypmode import provenance

    provenance.announce()
    main()
