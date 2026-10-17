"""Confirm the iron-nitrogen bond restraint on crystal ligands that were not used to decide it (the plan is ``restraint_expansion_plan.json``, registered before any run).

``prepare`` lists every eligible crystal ligand by the plan's rule and writes the batch inputs of the two conditions; ``score`` scores the runs and prints the verdict lines.

Usage: python -m cypmode.validation.restraint_expansion prepare
       python -m cypmode.validation.restraint_expansion score
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

INPUTS = Path("data/structure_track/inputs")
OUT = Path("data/structure_track/outputs/cyp3a4")
PLAN_PATH = INPUTS / "restraint_expansion_plan.json"
RESTRAINED_SUFFIX = "-febond"
COORDINATION_DISTANCE_A = 2.6
TRAINING_CUTOFF = "2023-06-01"
ORGANIC = frozenset(["C", "N", "O", "F", "S", "Cl", "Br", "I", "P"])


def load_plan(path: Path = PLAN_PATH) -> dict:
    return json.loads(Path(path).read_text())


def deposition_date(pdb_id: str) -> str | None:
    from cypmode.validation.reference_ligands import CACHE_DIR

    path = Path(CACHE_DIR) / f"{pdb_id}.cif"
    match = re.search(r"_pdbx_database_status\.recvd_initial_deposition_date\s+(\S+)", path.read_text()) if path.exists() else None
    return match.group(1) if match else None


def eligible(references: list[dict], benchmark_smiles: list[str], plan: dict | None = None) -> pd.DataFrame:
    """Every crystal ligand the plan allows, one row per distinct ligand (the entry with the best resolution), ordered by component id, with its stratum."""
    from rdkit import Chem, RDLogger

    from cypmode.validation.build_input import heme_donor_atom
    from cypmode.validation.pose_rescoring import connectivity_key

    RDLogger.DisableLog("rdApp.*")
    rules = (plan or load_plan())["eligibility"]
    excluded = {connectivity_key(s) for s in benchmark_smiles}
    best: dict[str, dict] = {}
    for ref in references:
        smiles, key = ref.get("smiles"), connectivity_key(ref.get("smiles") or "")
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None or not key or key in excluded or len(Chem.GetMolFrags(mol)) > 1 or {a.GetSymbol() for a in mol.GetAtoms()} - ORGANIC:
            continue
        if (ref["resolution"] or 99) > rules["max_resolution_angstrom"] or heme_donor_atom(smiles) is None:
            continue
        stratum = next((name for name, s in rules["strata"].items() if s["min_heavy_atoms"] <= mol.GetNumHeavyAtoms() <= s["max_heavy_atoms"]), None)
        if stratum is None:
            continue
        if key not in best or ref["resolution"] < best[key]["resolution"]:
            best[key] = {"comp_id": ref["comp_id"], "name": ref["name"], "smiles": smiles, "n_heavy": mol.GetNumHeavyAtoms(), "resolution": ref["resolution"],
                         "pdb_id": ref["pdb_id"], "min_fe_distance": ref["min_fe_distance"], "stratum": stratum}
    table = pd.DataFrame(best.values()).sort_values("comp_id").reset_index(drop=True) if best else pd.DataFrame(columns=["comp_id", "stratum"])
    if len(table):
        table["deposited"] = [deposition_date(p) for p in table["pdb_id"]]
    return table


def write_inputs(table: pd.DataFrame) -> list[Path]:
    """The batch CSVs: one pair (unguided, restrained) per stratum, in the order the queue runs them, and the full ligand table."""
    written = []
    for stratum in ("primary", "secondary"):
        part = table[table["stratum"] == stratum]
        names = [f"exp-{c}" for c in part["comp_id"]]
        for suffix, label in (("", "unguided"), (RESTRAINED_SUFFIX, "febond")):
            path = INPUTS / f"restraint_expansion_{stratum}_{label}.csv"
            pd.DataFrame({"Molecule_Name": [n + suffix for n in names], "SMILES": part["smiles"].to_numpy()}).to_csv(path, index=False)
            written.append(path)
    ligands = INPUTS / "restraint_expansion_ligands.csv"
    table.to_csv(ligands, index=False)
    return [*written, ligands]


def paired_statistics(difference: np.ndarray, alpha: float) -> dict:
    """The plan's test on the per-ligand differences (positive = the restraint is better)."""
    from scipy.stats import wilcoxon

    d = np.asarray(difference, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 3 or not np.any(d != 0):
        return {"n": len(d), "mean": float(d.mean()) if len(d) else float("nan"), "wins": int((d > 0).sum()), "p_better": 1.0, "p_worse": 1.0, "min_drop_one": float("nan"), "confirmed": False, "harm": False}
    drop_one = [float(np.delete(d, i).mean()) for i in range(len(d))]
    p_better = float(wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue)
    p_worse = float(wilcoxon(d, alternative="less", zero_method="wilcox").pvalue)
    mean = float(d.mean())
    return {"n": len(d), "mean": mean, "wins": int((d > 0).sum()), "p_better": p_better, "p_worse": p_worse, "min_drop_one": min(drop_one),
            "confirmed": bool(mean > 0 and min(drop_one) > 0 and p_better < alpha), "harm": bool(mean < 0 and p_worse < alpha)}


def per_ligand(poses: pd.DataFrame, column: str, samples: int) -> pd.DataFrame:
    """One row per ligand: the score of each condition (mean over the first ``samples`` samples that exist) and their difference (restrained minus unguided)."""
    boltz = poses[(poses["engine"] == "boltz") & (poses["sample"] < samples)].dropna(subset=[column])
    boltz = boltz.assign(ligand=boltz["compound"].str.removesuffix(RESTRAINED_SUFFIX), condition=np.where(boltz["compound"].str.endswith(RESTRAINED_SUFFIX), "restrained", "unguided"))
    wide = boltz.groupby(["ligand", "condition"])[column].mean().unstack("condition").reindex(columns=["unguided", "restrained"]).dropna()
    wide["difference"] = wide["restrained"] - wide["unguided"]
    return wide


def series_of(ligands: pd.DataFrame, threshold: float) -> dict[str, int]:
    """Series id of each ligand (``exp-<component id>``): the connected components of the graph that links ligands with Tanimoto similarity >= threshold."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in ligands["smiles"]]
    parent = list(range(len(fps)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(fps)):
        for j, similarity in enumerate(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])):
            if similarity >= threshold:
                parent[find(i)] = find(j)
    return {f"exp-{comp}": find(i) for i, comp in enumerate(ligands["comp_id"])}


def verdict_line(label: str, stats: dict) -> str:
    return (f"  {label}: mean change {stats['mean']:+.3f} over {stats['n']} ligands, {stats['wins']} improved, Wilcoxon p {stats['p_better']:.3f} (better) / {stats['p_worse']:.3f} (worse), "
            f"worst drop-one mean {stats['min_drop_one']:+.3f}")


def subgroup_lines(lddt: pd.DataFrame, ligands: pd.DataFrame) -> list[str]:
    """The primary difference by whether the crystal nitrogen coordinates the iron and by whether the entry predates the end of Boltz-2's structure training data."""
    info = ligands.assign(ligand=[f"exp-{c}" for c in ligands["comp_id"]]).set_index("ligand")
    joined = lddt.join(info[["min_fe_distance", "deposited"]], how="left")
    groups = {
        f"the crystal nitrogen coordinates the iron (closest atom <= {COORDINATION_DISTANCE_A} A)": joined["min_fe_distance"] <= COORDINATION_DISTANCE_A,
        "the crystal ligand does not coordinate the iron": joined["min_fe_distance"] > COORDINATION_DISTANCE_A,
        f"deposited before {TRAINING_CUTOFF}": joined["deposited"] < TRAINING_CUTOFF,
        f"deposited on or after {TRAINING_CUTOFF}": joined["deposited"] >= TRAINING_CUTOFF,
    }
    return [f"  {label}: {int(mask.sum())} ligands, mean LDDT-PLI change {joined.loc[mask, 'difference'].mean():+.3f}" for label, mask in groups.items() if mask.any()]


def validity_line(ligands: pd.DataFrame, smiles_of: dict[str, str]) -> str:
    """The share of first samples that pass PoseBusters, per condition, over the ligands whose runs exist."""
    from cypmode.validation.pose_consensus import boltz_cif_paths
    from cypmode.validation.pose_validity import check_pose_validity
    from cypmode.validation.structures import load_structure

    def passes(name: str) -> bool | None:
        paths = boltz_cif_paths(name)
        if not paths:
            return None
        try:
            return bool(check_pose_validity(load_structure(paths[0]), smiles_of[name])["all_passed"])
        except Exception:
            return False

    names = [f"exp-{c}" for c in ligands["comp_id"]]
    pairs = [(passes(n), passes(n + RESTRAINED_SUFFIX)) for n in names]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not pairs:
        return "  PoseBusters: no ligand has both runs"
    return f"  PoseBusters, first samples over {len(pairs)} ligands: {np.mean([a for a, _ in pairs]):.0%} valid without the restraint, {np.mean([b for _, b in pairs]):.0%} with it"


def prepare() -> int:
    from cypmode.validation.reference_ligands import load_reference_set

    benchmark = pd.read_csv(INPUTS / "structure_benchmark.csv")
    table = eligible(load_reference_set(), list(benchmark["SMILES"]))
    paths = write_inputs(table)
    print(f"plan registered {load_plan()['registered']}: {len(table)} eligible crystal ligands ({(table['stratum'] == 'primary').sum()} primary, {(table['stratum'] == 'secondary').sum()} secondary)")
    print(table[["comp_id", "name", "n_heavy", "resolution", "pdb_id", "deposited", "min_fe_distance", "stratum"]].to_string(index=False))
    print("wrote " + ", ".join(p.name for p in paths))
    return 0


def score() -> int:
    from cypmode.validation.mbr_lddt_reevaluation import pose_table
    from cypmode.validation.pose_pools import pool_paths

    plan = load_plan()
    alpha = plan["endpoints"]["alpha"]
    samples = plan["design"]["samples_per_condition"]
    table = pd.read_csv(INPUTS / "restraint_expansion_ligands.csv")
    names = {f"exp-{c}": s for c, s in zip(table["comp_id"], table["smiles"], strict=True)}
    smiles_of = {**names, **{n + RESTRAINED_SUFFIX: s for n, s in names.items()}}
    finished = [n for n in smiles_of if any(p.exists() for _, _, p in pool_paths(n, ("boltz",)))]
    if not finished:
        print("no run of the expansion exists yet")
        return 75
    poses = pose_table(finished, smiles_of)
    poses.to_csv(OUT / "restraint_expansion_poses.csv", index=False)
    rmsd_tables = [OUT / f"restraint_expansion_{stratum}_{label}_poses.csv" for stratum in ("primary", "secondary") for label in ("unguided", "febond")]
    rmsd = pd.concat([pd.read_csv(p) for p in rmsd_tables if p.exists()]) if any(p.exists() for p in rmsd_tables) else pd.DataFrame(columns=["compound", "engine", "sample", "true_rmsd"])
    poses = poses.merge(rmsd[["compound", "engine", "sample", "true_rmsd"]], on=["compound", "engine", "sample"], how="left")
    by_stratum = dict(zip(table["comp_id"].map(lambda c: f"exp-{c}"), table["stratum"], strict=True))
    for stratum in ("primary", "secondary"):
        members = {n for n, s in by_stratum.items() if s == stratum}
        part = poses[poses["compound"].str.removesuffix(RESTRAINED_SUFFIX).isin(members)]
        lddt = per_ligand(part, "lddt_pli", samples)
        lddt.to_csv(OUT / f"restraint_expansion_{stratum}_lddt.csv")
        stats = paired_statistics(lddt["difference"].to_numpy(), alpha)
        print(f"{stratum} stratum: {len(members)} ligands planned, {len(lddt)} with both conditions scored")
        print(verdict_line("LDDT-PLI, restrained minus unguided", stats))
        if len(part["true_rmsd"].dropna()):
            rm = per_ligand(part, "true_rmsd", samples)
            print(verdict_line("RMSD to the crystal (A), unguided minus restrained", paired_statistics(-rm["difference"].to_numpy(), alpha)))
        print("\n".join(subgroup_lines(lddt, table[table["stratum"] == stratum])))
        series = series_of(table, plan["endpoints"]["series_similarity_threshold"])
        by_series = lddt["difference"].groupby(lddt.index.map(series)).mean()
        print(verdict_line(f"LDDT-PLI, one unit per chemical series ({len(by_series)} series)", paired_statistics(by_series.to_numpy(), alpha)))
        print(validity_line(table[table["stratum"] == stratum], smiles_of))
        if stratum == "primary":
            print(f"STRUCTURE GATE restraint_bond_expansion (confirmatory, registered {plan['registered']}, one-sided Wilcoxon p < {alpha}, robust to dropping a ligand): {'MET' if stats['confirmed'] else 'NOT MET'}")
            print(f"STRUCTURE GATE restraint_bond_harm (registered {plan['registered']}, one-sided Wilcoxon that the restraint is worse p < {alpha}): {'MET' if stats['harm'] else 'NOT MET'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "score"))
    return {"prepare": prepare, "score": score}[parser.parse_args().command]()


if __name__ == "__main__":
    sys.exit(main())
