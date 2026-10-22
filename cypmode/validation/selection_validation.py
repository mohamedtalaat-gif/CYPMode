# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Does minimum-Bayes-risk pose selection beat the model's own first pick, and does a second engine help? (jobs 170 / 150)

Every rule answers one question per ligand, "which single pose do we submit?", and is scored by that pose's true ligand
RMSD to the crystal pose of the benchmark. MBR (``pose_selection``) picks the pose with the highest mean LDDT-PLI against
the other poses of the pool; it has no fitted parameter, so "leave one ligand out" holds by construction (a ligand's pick
uses its own pool only).

Pre-registered gates (fixed before the full benchmark was scored):
  mbr_selection    MBR over the Boltz samples vs Boltz's first pick: mean RMSD lower, the improvement survives dropping any one
                   ligand (every drop-one mean improvement > 0), and a one-sided paired Wilcoxon p < 0.10 (about 10 ligands:
                   p < 0.05 would need nearly every ligand to improve, so 0.10 is stated openly);
  engine_consensus a second engine is worth adding only if its errors DIFFER from Boltz's (Spearman rho of the per-ligand
                   first-pick RMSDs < 0.7) AND MBR over both engines has a lower mean RMSD than MBR over Boltz alone with
                   the same drop-one robustness.

Usage: python -m cypmode.validation.selection_validation [--table CSV] [--out CSV]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TABLES = (
    "data/structure_track/outputs/cyp3a4/benchmark_engine_pose_validation.csv",
    "data/structure_track/outputs/cyp3a4/benchmark_engine_pose_validation_unguided9.csv",
)
P_VALUE = 0.10


def paired_gate(baseline: pd.Series, rule: pd.Series) -> dict:
    """Improvement = baseline RMSD - rule RMSD per ligand (positive = the rule is better)."""
    from scipy.stats import wilcoxon

    both = pd.concat([baseline, rule], axis=1, keys=["b", "r"]).dropna()
    improvement = (both["b"] - both["r"]).to_numpy()
    if len(improvement) < 3:
        return {"n": len(improvement), "mean_improvement": float(improvement.mean()) if len(improvement) else float("nan"), "wins": int((improvement > 0).sum()), "p": 1.0, "min_drop_one": float("nan"), "met": False}
    p = float(wilcoxon(improvement, alternative="greater", zero_method="wilcox").pvalue) if np.any(improvement != 0) else 1.0
    drop_one = [float(np.delete(improvement, i).mean()) for i in range(len(improvement))]
    mean = float(improvement.mean())
    return {"n": len(improvement), "mean_improvement": mean, "wins": int((improvement > 0).sum()), "p": p, "min_drop_one": float(min(drop_one)), "met": bool(mean > 0 and min(drop_one) > 0 and p < P_VALUE)}


def evaluate_rules(table: pd.DataFrame, matrices: dict[str, tuple[list[tuple[str, int]], np.ndarray]]) -> pd.DataFrame:
    """One row per ligand and rule. ``matrices[compound] = (poses [(engine, sample)], LDDT-PLI matrix over those poses)``."""
    from cypmode.validation.pose_pools import mbr_pick

    rows = []
    truth = table.set_index(["compound", "engine", "sample"])["true_rmsd"]
    for compound, (poses, matrix) in matrices.items():
        rmsd = np.array([truth.get((compound, e, s), np.nan) for e, s in poses])
        boltz = [i for i, (e, _) in enumerate(poses) if e == "boltz" and not np.isnan(rmsd[i])]
        both = [i for i, _ in enumerate(poses) if not np.isnan(rmsd[i])]
        if len(boltz) < 2:
            continue
        row = {"compound": compound, "first_boltz": rmsd[min(boltz, key=lambda i: poses[i][1])], "mbr_boltz": rmsd[mbr_pick(matrix, boltz)],
               "mean_pool_boltz": float(np.mean(rmsd[boltz])), "oracle_boltz": float(np.min(rmsd[boltz]))}
        engines = {poses[i][0] for i in both}
        if {"boltz", "openfold3"} <= engines:
            of3 = [i for i in both if poses[i][0] == "openfold3"]
            row.update(first_openfold3=rmsd[min(of3, key=lambda i: poses[i][1])], mbr_both=rmsd[mbr_pick(matrix, both)], oracle_both=float(np.min(rmsd[both])))
        rows.append(row)
    return pd.DataFrame(rows)


def gates(outcomes: pd.DataFrame) -> dict[str, dict]:
    from scipy.stats import spearmanr

    out = {"mbr_selection": paired_gate(outcomes["first_boltz"], outcomes["mbr_boltz"])}
    if "mbr_both" in outcomes and outcomes["mbr_both"].notna().sum() >= 3:
        rho = float(spearmanr(outcomes["first_boltz"], outcomes["first_openfold3"]).statistic)
        engines = paired_gate(outcomes["mbr_boltz"], outcomes["mbr_both"])
        engines.update(errors_correlation=rho, met=bool(engines["met"] and rho < 0.7))
        out["engine_consensus"] = engines
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=None)
    parser.add_argument("--out", default="data/structure_track/outputs/cyp3a4/selection_validation_outcomes.csv")
    args = parser.parse_args()
    from cypmode.validation.pose_pools import pairwise_lddt, pool_paths

    path = args.table or next((t for t in DEFAULT_TABLES if Path(t).exists()), None)
    if path is None:
        raise SystemExit("no pose table found (run job 160 first)")
    table = pd.read_csv(path)
    benchmark = pd.read_csv("data/structure_track/inputs/structure_benchmark.csv")
    smiles = dict(zip(benchmark.Molecule_Name, benchmark.SMILES, strict=True))
    matrices = {}
    for compound in sorted(table["compound"].unique()):
        pools = [(e, s, p) for e, s, p in pool_paths(compound, ("boltz", "openfold3")) if p.exists()]
        if len(pools) >= 2:
            matrices[compound] = ([(e, s) for e, s, _ in pools], pairwise_lddt([p for _, _, p in pools], smiles=smiles.get(compound)))
    outcomes = evaluate_rules(table, matrices)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(args.out, index=False)
    print(f"pose table {path}: {len(outcomes)} ligands with a pool")
    print(outcomes.drop(columns="compound").agg(["mean", "median"]).round(3).to_string())
    for name, g in gates(outcomes).items():
        extra = f", errors correlation rho {g['errors_correlation']:.2f}" if "errors_correlation" in g else ""
        print(f"  {name}: mean improvement {g['mean_improvement']:+.3f} A over {g['n']} ligands, {g['wins']} wins, Wilcoxon p {g['p']:.3f}, worst drop-one mean {g['min_drop_one']:+.3f}{extra}")
        print(f"STRUCTURE GATE {name} (paired, drop-one robust, p < {P_VALUE}): {'MET' if g['met'] else 'NOT MET'}")


if __name__ == "__main__":
    main()
