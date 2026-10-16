"""SAR-aware pose selection: choose the pose that agrees with the poses of the ligand's analogs (job 169).

For two analogs A and B with a common core, the core RMSD ``C[i, j]`` between pose i of A and pose j of B (complexes superposed on pocket
C-alpha atoms, ``series.core_rmsd``) says how consistent two poses are with the series binding in one way. A pose of A that lies close to
many poses of B is supported by the series. The series score of pose i is minus the mean of ``C[i, :]``; the rule combines it with the
minimum-Bayes-risk consensus inside the pool of A (mean rank of the two, no fitted weight).

Benchmark: three crystal-anchored analog pairs (``series.select_series_benchmark``; unguided Boltz-2, 5 samples per ligand, true RMSD to
the crystal pose). Gate (fixed in advance): over the six ligands, the pick of the combined rule has a lower mean RMSD than the MBR pick,
the improvement survives dropping any one ligand and the one-sided paired Wilcoxon p is below 0.10 (``selection_validation.paired_gate``).

Official ligands: the pairs of the 20 test ligands with Tanimoto >= 0.5 are listed, so the rule can be applied to them in the final recipe.

Usage: python -m cypmode.validation.series_consensus
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

SERIES_CSV = Path("data/structure_track/inputs/structure_benchmark_series.csv")
POSE_TABLE = Path("data/structure_track/outputs/cyp3a4/series_pose_validation.csv")


def combined_pick(mbr: np.ndarray, consensus_rmsd: np.ndarray) -> int:
    """Index of the pose with the best mean rank of its MBR score (higher is better) and its mean core RMSD to the analog's poses (lower is better)."""
    from scipy.stats import rankdata

    valid = np.isfinite(consensus_rmsd)
    if not valid.any():
        return int(np.argmax(mbr))
    score = rankdata(mbr) + np.where(valid, rankdata(-np.nan_to_num(consensus_rmsd, nan=np.inf)), rankdata(mbr))
    return int(np.argmax(score))


def official_pairs(path: str = "data/structure_track/inputs/structure_track_test_blinded.csv", threshold: float = 0.5) -> list[tuple[str, str, float]]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    t = pd.read_csv(path)
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in t.SMILES]
    out = []
    for i, j in itertools.combinations(range(len(t)), 2):
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        if sim >= threshold:
            out.append((t.Molecule_Name[i], t.Molecule_Name[j], float(sim)))
    return out


def main() -> None:
    from cypmode.validation.pose_pools import pairwise_lddt, pool_paths
    from cypmode.validation.selection_validation import P_VALUE, paired_gate
    from cypmode.validation.series import common_core, core_rmsd
    from cypmode.validation.structures import load_structure

    print("official ligand pairs with Tanimoto >= 0.5:", [(a, b, round(s, 2)) for a, b, s in official_pairs()])
    if not SERIES_CSV.exists() or not POSE_TABLE.exists():
        raise SystemExit("series benchmark tables missing (jobs 133 and 142)")
    series, truth = pd.read_csv(SERIES_CSV), pd.read_csv(POSE_TABLE)
    truth = truth[truth.engine == "boltz"].set_index(["compound", "sample"])["true_rmsd"]
    smiles = dict(zip(series.Molecule_Name, series.SMILES, strict=True))
    rows = []
    for _, pair in series.groupby("series"):
        a, b = list(pair.Molecule_Name)
        pools = {n: [(s, p) for _, s, p in pool_paths(n, ("boltz",)) if p.exists() and not np.isnan(truth.get((n, s), np.nan))] for n in (a, b)}
        if min(len(v) for v in pools.values()) < 3:
            continue
        core = common_core(smiles[a], smiles[b])
        structures = {n: [load_structure(p) for _, p in pools[n]] for n in (a, b)}
        c = np.array([[core_rmsd(sa, smiles[a], sb, smiles[b], core) for sb in structures[b]] for sa in structures[a]])
        for n, matrix_c in ((a, c), (b, c.T)):
            matrix = pairwise_lddt([p for _, p in pools[n]])
            mbr = np.array([np.mean([matrix[i, j] for j in range(len(matrix)) if j != i]) for i in range(len(matrix))])
            rmsd = np.array([truth[(n, s)] for s, _ in pools[n]])
            consensus = np.nanmean(matrix_c, axis=1) if np.isfinite(matrix_c).any() else np.full(len(rmsd), np.nan)
            rows.append({"compound": n, "partner": b if n == a else a, "core_atoms": len(core), "poses": len(rmsd), "mean_pool": float(rmsd.mean()),
                         "mbr_pick": float(rmsd[int(np.argmax(mbr))]), "series_pick": float(rmsd[int(np.nanargmin(consensus))]) if np.isfinite(consensus).any() else np.nan,
                         "combined_pick": float(rmsd[combined_pick(mbr, consensus)]), "oracle": float(rmsd.min())})
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))
    g = paired_gate(table["mbr_pick"], table["combined_pick"])
    print(f"mean RMSD: random pose {table.mean_pool.mean():.3f}, MBR {table.mbr_pick.mean():.3f}, series consensus alone {table.series_pick.mean():.3f}, MBR + series {table.combined_pick.mean():.3f}, oracle {table.oracle.mean():.3f}")
    print(f"MBR + series against MBR: mean improvement {g['mean_improvement']:+.3f} A over {g['n']} ligands, {g['wins']} wins, Wilcoxon p {g['p']:.3f}, worst drop-one mean {g['min_drop_one']:+.3f}")
    print(f"STRUCTURE GATE series_consensus (paired, drop-one robust, p < {P_VALUE}): {'MET' if g['met'] else 'NOT MET'}")


if __name__ == "__main__":
    main()
