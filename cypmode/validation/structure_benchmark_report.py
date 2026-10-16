"""Which engine and which pose-selection rule to trust, judged on the crystal-anchored benchmark.

Input is the table written by ``pose_rescoring_validation --engine-poses`` (one row per predicted
pose, with its true ligand RMSD to the crystal pose and the leave-the-compound-out rescoring
scores). Every rule answers the same question -- "which single pose would we submit for this
compound?" -- and is scored by the true RMSD of the pose it picks. The oracle rules (the best pose
that exists) are the ceiling a selection rule could reach, not a rule anyone can apply.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

SUCCESS_RMSD = 2.0  # Angstrom, the usual docking-success cut
ENGINES = ("boltz", "openfold3")
SCORE = "novel_template_score"  # computed with every reference as similar as the compound itself removed


def _picked(poses: pd.DataFrame, rule: str, engines: tuple[str, ...]) -> float | None:
    pool = poses[poses["engine"].isin(engines)]
    if pool.empty or set(engines) - set(pool["engine"]):
        return None
    if rule == "first":  # the engine's own ranking: Boltz model_0, OpenFold3 sample 1
        return float(pool.sort_values(["engine", "sample"]).groupby("engine").head(1)["true_rmsd"].mean())
    if rule == "oracle":
        return float(pool["true_rmsd"].min())
    if rule == "template":
        return float(pool.loc[pool[SCORE].idxmax(), "true_rmsd"])
    raise ValueError(rule)


def selection_outcomes(table: pd.DataFrame) -> pd.DataFrame:
    """One row per (compound, rule): the true RMSD of the pose the rule submits."""
    rules = {
        "boltz first pick": ("first", ("boltz",)),
        "openfold3 first sample": ("first", ("openfold3",)),
        "boltz best sample (oracle)": ("oracle", ("boltz",)),
        "openfold3 best sample (oracle)": ("oracle", ("openfold3",)),
        "either engine best sample (oracle)": ("oracle", ENGINES),
        "template score over Boltz samples": ("template", ("boltz",)),
        "template score over OpenFold3 samples": ("template", ("openfold3",)),
        "template score over both engines": ("template", ENGINES),
    }
    rows = []
    for compound, poses in table.dropna(subset=["true_rmsd", SCORE]).groupby("compound"):
        for label, (rule, engines) in rules.items():
            value = _picked(poses, rule, engines)
            if value is not None:
                rows.append({"compound": compound, "rule": label, "rmsd": value})
    return pd.DataFrame(rows, columns=["compound", "rule", "rmsd"])


def summarize(outcomes: pd.DataFrame) -> pd.DataFrame:
    grouped = outcomes.groupby("rule")["rmsd"]
    summary = pd.DataFrame(
        {
            "compounds": grouped.size(),
            "mean_rmsd": grouped.mean(),
            "median_rmsd": grouped.median(),
            f"success_under_{SUCCESS_RMSD:g}A": grouped.apply(lambda r: float(np.mean(r < SUCCESS_RMSD))),
        }
    )
    return summary.sort_values("mean_rmsd")


def variant_comparison(tables: dict[str, pd.DataFrame], samples: int = 3) -> pd.DataFrame:
    """Does steering (a restraint variant) beat the unguided run on the same ligands?

    ``tables`` maps a variant label to its pose table; a restrained run names its compounds with a
    ``-febond`` / ``-fecontact`` suffix, which is stripped so a ligand lines up across variants. Only
    Boltz poses among the first ``samples`` samples and only ligands every variant produced are compared,
    so the variants cover the same compounds with the same sampling budget. Returns one row per
    (variant, ligand): the RMSD of the model's first pick, of the best sample, and of the pose the
    template score picks.
    """
    prepared = {}
    for label, table in tables.items():
        t = table[(table["engine"] == "boltz") & (table["sample"] < samples)].dropna(subset=["true_rmsd", SCORE]).copy()
        t["compound"] = t["compound"].map(lambda c: re.sub(r"-fe(bond|contact)$", "", c))
        prepared[label] = t
    shared = set.intersection(*(set(t["compound"]) for t in prepared.values()))
    rows = []
    for label, t in prepared.items():
        for compound, poses in t[t["compound"].isin(shared)].groupby("compound"):
            rows.append(
                {
                    "variant": label,
                    "compound": compound,
                    "first_pick": float(poses.sort_values("sample")["true_rmsd"].iloc[0]),
                    "best_sample": float(poses["true_rmsd"].min()),
                    "template_pick": float(poses.loc[poses[SCORE].idxmax(), "true_rmsd"]),
                }
            )
    return pd.DataFrame(rows, columns=["variant", "compound", "first_pick", "best_sample", "template_pick"])


def summarize_variants(comparison: pd.DataFrame) -> pd.DataFrame:
    metrics = ["first_pick", "best_sample", "template_pick"]
    mean = comparison.groupby("variant")[metrics].mean().add_prefix("mean_rmsd_")
    success = comparison.groupby("variant")[metrics].agg(lambda r: float(np.mean(r < SUCCESS_RMSD))).add_prefix("under_2A_")
    return pd.concat([comparison.groupby("variant").size().rename("ligands"), mean, success], axis=1)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", nargs="?", help="engine_pose_validation-style CSV")
    parser.add_argument("--compare", nargs="+", metavar="LABEL=CSV", help="compare restraint variants on the same ligands instead")
    parser.add_argument("--samples", type=int, default=3, help="Boltz samples per ligand used by --compare")
    parser.add_argument("--out", default=None, help="optional CSV for the per-compound outcomes")
    args = parser.parse_args()
    if args.compare:
        tables = {item.split("=", 1)[0]: pd.read_csv(item.split("=", 1)[1]) for item in args.compare}
        print(summarize_variants(variant_comparison(tables, args.samples)).round(3).to_string())
        return
    outcomes = selection_outcomes(pd.read_csv(args.table))
    print(summarize(outcomes).round(3).to_string())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        outcomes.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    _main()
