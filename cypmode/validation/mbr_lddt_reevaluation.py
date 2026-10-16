"""Minimum-Bayes-risk pose selection and the iron-nitrogen restraint, re-scored with LDDT-PLI against the crystal (exploratory).

The Structure Track is scored by LDDT-PLI, and minimum Bayes risk optimises it by construction: it picks the pose with the highest mean LDDT-PLI against the other poses of the
pool. The pre-registered gate of ``selection_validation`` (job 170) scored the picks by RMSD to the crystal pose, because RMSD was the metric of the earlier benchmark tables,
so a re-scoring with the challenge's own metric is a different question. It is reported as exploratory: the rule and the thresholds are those of the registered gate, but the
metric was chosen after that gate had been judged.

The same poses give the restraint's effect in the same metric: the first sample of the unguided run against the first sample of the bond-restrained run, and minimum Bayes
risk inside the restrained pool.

Writes ``pose_lddt_to_crystal.csv`` (every pose), ``selection_lddt_outcomes.csv`` and ``restraint_lddt_outcomes.csv`` and prints one
``STRUCTURE GATE <name> ...: MET|NOT MET`` line per question (read by ``method_report``; the final recipe does not read them).

Usage: python -m cypmode.validation.mbr_lddt_reevaluation
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/structure_track/outputs/cyp3a4")
BENCHMARK = Path("data/structure_track/inputs/structure_benchmark.csv")
RESTRAINED_SUFFIX = "-febond"
ENGINES = ("boltz", "openfold3")


def pose_table(compounds: list[str], smiles_of: dict[str, str]) -> pd.DataFrame:
    """LDDT-PLI to the crystal of every pose in the pools of the given compounds (run names; the SMILES is that of the compound the run belongs to)."""
    from cypmode.validation.lddt_to_crystal import crystal_references, pose_lddt_to_crystal
    from cypmode.validation.pose_pools import pool_paths, template_automorphisms
    from cypmode.validation.structures import load_structure

    rows = []
    for compound in compounds:
        smiles = smiles_of[compound]
        references = crystal_references(smiles)
        permutations = template_automorphisms(smiles)
        for engine, sample, path in pool_paths(compound, ENGINES):
            if path.exists():
                score = pose_lddt_to_crystal(load_structure(path), smiles, references, permutations)
                rows.append({"compound": compound, "engine": engine, "sample": sample, "lddt_pli": score, "n_references": len(references)})
    return pd.DataFrame(rows)


def selection_outcomes(poses: pd.DataFrame, smiles_of: dict[str, str]) -> pd.DataFrame:
    """The rules of ``selection_validation`` scored by LDDT-PLI (higher is better): the scores are negated on the way in so that its code, which minimises, can be reused."""
    from cypmode.validation.pose_pools import pairwise_lddt, pool_paths
    from cypmode.validation.selection_validation import evaluate_rules

    unguided = poses[~poses["compound"].str.endswith(RESTRAINED_SUFFIX)]
    table = unguided.dropna(subset=["lddt_pli"]).assign(true_rmsd=lambda t: -t["lddt_pli"])
    matrices = {}
    for compound in sorted(table["compound"].unique()):
        pool = [(engine, sample, path) for engine, sample, path in pool_paths(compound, ENGINES) if path.exists()]
        if len(pool) >= 2:
            matrices[compound] = ([(engine, sample) for engine, sample, _ in pool], pairwise_lddt([path for _, _, path in pool], smiles=smiles_of.get(compound)))
    outcomes = evaluate_rules(table, matrices)
    return outcomes.assign(**{column: -outcomes[column] for column in outcomes.columns.drop("compound")})


def restraint_outcomes(poses: pd.DataFrame, smiles_of: dict[str, str]) -> pd.DataFrame:
    """Per ligand that was also run with the bond restraint: first sample of each run, minimum Bayes risk and the best sample inside the restrained pool."""
    from cypmode.validation.pose_pools import mbr_pick, pairwise_lddt, pool_paths

    boltz = poses[poses["engine"] == "boltz"].set_index(["compound", "sample"])["lddt_pli"]
    rows = []
    for restrained in sorted({c for c in poses["compound"] if c.endswith(RESTRAINED_SUFFIX)}):
        base = restrained.removesuffix(RESTRAINED_SUFFIX)
        pool = [(sample, path) for engine, sample, path in pool_paths(restrained, ("boltz",)) if path.exists()]
        scores = [boltz.get((restrained, sample), np.nan) for sample, _ in pool]
        if base not in set(poses["compound"]) or len(pool) < 2:
            continue
        matrix = pairwise_lddt([path for _, path in pool], smiles=smiles_of[restrained])
        rows.append({"compound": base, "first_unguided": boltz.get((base, 0), np.nan), "first_restrained": scores[0], "mbr_restrained": scores[mbr_pick(matrix)], "best_restrained": float(np.nanmax(scores))})
    return pd.DataFrame(rows)


def gate_line(name: str, stats: dict, description: str) -> str:
    verdict = "MET" if stats["met"] else "NOT MET"
    return (f"  {name}: mean LDDT-PLI gain {stats['mean_improvement']:+.3f} over {stats['n']} ligands, {stats['wins']} wins, Wilcoxon p {stats['p']:.3f}, worst drop-one mean {stats['min_drop_one']:+.3f}\n"
            f"STRUCTURE GATE {name} ({description}): {verdict}")


def main() -> int:
    from scipy.stats import spearmanr

    from cypmode.validation.selection_validation import P_VALUE, paired_gate

    benchmark = pd.read_csv(BENCHMARK)
    base_smiles = dict(zip(benchmark["Molecule_Name"], benchmark["SMILES"], strict=True))
    smiles_of = {**base_smiles, **{f"{name}{RESTRAINED_SUFFIX}": smiles for name, smiles in base_smiles.items()}}
    unguided_table = OUT / "benchmark_engine_pose_validation_unguided_boltz.csv"
    restrained_table = OUT / "benchmark_engine_pose_validation_febond.csv"
    if not unguided_table.exists() or not restrained_table.exists():
        print("the pose tables of the benchmark do not exist yet")
        return 75
    compounds = sorted(set(pd.read_csv(unguided_table)["compound"]) | set(pd.read_csv(restrained_table)["compound"]))
    poses = pose_table(compounds, smiles_of)
    poses.to_csv(OUT / "pose_lddt_to_crystal.csv", index=False)

    rmsd = pd.concat([pd.read_csv(unguided_table), pd.read_csv(restrained_table)])
    joined = poses.merge(rmsd[["compound", "engine", "sample", "true_rmsd"]], on=["compound", "engine", "sample"]).dropna(subset=["lddt_pli", "true_rmsd"])
    rho = float(spearmanr(joined["lddt_pli"], joined["true_rmsd"]).statistic)
    print(f"{len(poses)} poses of {poses['compound'].nunique()} runs scored; LDDT-PLI against the crystal and RMSD to the crystal agree: Spearman rho {rho:.2f} over {len(joined)} poses")

    selection = selection_outcomes(poses, smiles_of)
    selection.to_csv(OUT / "selection_lddt_outcomes.csv", index=False)
    print(f"selection rules scored by LDDT-PLI, {len(selection)} ligands with a pool (higher is better):")
    print(selection.drop(columns="compound").agg(["mean", "median"]).round(3).to_string())
    stats = paired_gate(-selection["first_boltz"], -selection["mbr_boltz"])
    print(gate_line("mbr_lddt_exploratory", stats, f"minimum Bayes risk over the Boltz samples against the first pick, scored by LDDT-PLI, thresholds of mbr_selection (p < {P_VALUE}), exploratory"))

    restraint = restraint_outcomes(poses, smiles_of)
    restraint.to_csv(OUT / "restraint_lddt_outcomes.csv", index=False)
    print(f"restraint, {len(restraint)} ligands run both ways (higher is better):")
    print(restraint.drop(columns="compound").agg(["mean", "median"]).round(3).to_string())
    stats = paired_gate(-restraint["first_unguided"], -restraint["first_restrained"])
    print(gate_line("restraint_bond_lddt_exploratory", stats, f"first sample with the bond restraint against without it, scored by LDDT-PLI, p < {P_VALUE}, exploratory"))
    stats = paired_gate(-restraint["first_restrained"], -restraint["mbr_restrained"])
    print(gate_line("mbr_on_restrained_lddt_exploratory", stats, f"minimum Bayes risk inside the restrained pool against its first sample, scored by LDDT-PLI, p < {P_VALUE}, exploratory"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
