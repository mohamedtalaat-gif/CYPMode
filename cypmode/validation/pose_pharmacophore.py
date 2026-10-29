# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Pharmacophore model of a predicted pose (job 272_pose_pharmacophore_gate, declared on 2026-10-23 before any code ran).

The structure track has had pharmacophore terms for choosing a pose (``pose_awareness``: two distances written by hand) and a shape prior from the crystal ligands (``pose_rescoring``), and no pharmacophore
*model* of a pose. The fields and the sites that ``cypmode.challenge.pocket_pharmacophore`` builds from the public crystal ligands of CYP3A4 are one: a pose is put in the frame of the heme of its own
predicted complex (the iron at the origin, z the porphyrin normal on the side of the ligand, x toward the nitrogen NA) and is scored for where its features and its atoms lie.

Per pose
    field         the mean occupancy at the heavy atoms plus the mean, over the families the ligand has, of the mean family field at its features (the score of 9.13);
    recall        the weight of the sites that a feature of the ligand matches over the weight of all sites (the model of 9.14); ``matched`` their number, ``precision`` the share of the
                  typed features of the ligand that lie in a sphere of a site of their own family;
    outside       the share of the heavy atoms where the occupancy field is below ``SUPPORT_MIN``: the pose leaves the region where crystal ligands have been seen.
Fields and sites are rebuilt without the references of the scored ligand: always the references that are the same compound (connectivity key of the InChIKey), and in the regime of the gate (``novel``,
a novel scaffold with no analogue in the PDB) also every reference with Morgan Tanimoto similarity of at least ``NOVEL_SIMILARITY`` to it, the rule of ``pose_rescoring_validation``. The regime
``compound`` (only the compound itself left out) is printed beside it and is information.

Scores of a pool are mean within-pool ranks (no fitted weight): ``pharmacophore_model`` = ranks of field and recall; ``pharmacophore_model_combined`` = that rank and the rank of the MBR consensus.
Gates (the bar of ``pose_awareness``): over the pooled benchmark poses of ``pose_awareness`` the Spearman correlation between a score and minus the true RMSD, averaged over ligands, exceeds 0.3 and is positive for
at least 60% of the ligands. When a gate is met, a second, paired test decides whether the recipe may use the score (the mean RMSD of its pick below that of MBR, every drop-one difference above
zero, one-sided paired Wilcoxon p < 0.10); it is evaluated only then. The crystal pose of each benchmark ligand is scored the same way and printed beside the predicted poses as information.

Usage: python -m cypmode.validation.pose_pharmacophore --gate      (job 272)
"""

import argparse
import functools
from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.challenge import pocket_pharmacophore as pp
from cypmode.validation.pose_awareness import MIN_FRACTION_POSITIVE, MIN_MEAN_RHO

NOVEL_SIMILARITY = 0.5  # the value of pose_rescoring_validation.NOVEL_SIMILARITY, repeated so that this module does not import that one
SUPPORT_MIN = 0.05
REGIMES = ("novel", "compound")
GATED_REGIME = "novel"
SCORES = ("pharmacophore_model", "pharmacophore_model_combined")
WILCOXON_P, DROP_ONE_MIN = 0.10, 0.0
OUT = Path("data/structure_track/outputs/cyp3a4/pose_pharmacophore_poses.csv")


# -- references and the fields rebuilt without some of them ---------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def references() -> tuple[dict, ...]:
    import json

    return tuple(json.loads(pp.POCKETS["cyp3a4"].read_text())["ligands"])


def kept_indices(smiles: str, regime: str, refs: tuple[dict, ...] | None = None) -> tuple[int, ...]:
    """Indices of the references that remain for a ligand: the references of the compound itself are always out, and in the regime ``novel`` those at least ``NOVEL_SIMILARITY`` similar are out too."""
    from rdkit import DataStructs

    from cypmode.validation.pose_rescoring import connectivity_key, morgan_fingerprint

    refs = references() if refs is None else refs
    own = connectivity_key(smiles)
    query = morgan_fingerprint(smiles)
    kept = []
    for i, ref in enumerate(refs):
        if own is not None and connectivity_key(ref["smiles"]) == own:
            continue
        if regime == "novel" and query is not None and ref.get("smiles"):
            other = morgan_fingerprint(ref["smiles"])
            if other is not None and DataStructs.TanimotoSimilarity(query, other) >= NOVEL_SIMILARITY:
                continue
        kept.append(i)
    return tuple(kept)


@functools.lru_cache(maxsize=64)
def fields_and_sites(kept: tuple[int, ...]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The fields and the sites built from the references with these indices (cached: ligands that leave out the same references share them)."""
    refs = references()
    fields = pp.build_fields([refs[i] for i in kept])
    return fields, pp.sites_of(fields)


# -- a pose in the frame of its heme ------------------------------------------------------------------------------------------------------------------------

def pose_in_frame(structure, smiles: str) -> dict | None:
    """The heavy-atom coordinates, the typed features and the iron-donor and oxidation geometry of the ligand of a predicted complex (a Bio.PDB structure), in the frame of its heme; None when the
    ligand's bond orders cannot be assigned (a distorted pose) or the heme lacks a pyrrole nitrogen."""
    from rdkit import Chem

    from cypmode.validation.structures import _ligand_mol_3d, find_heme_iron

    try:
        mol, _ = _ligand_mol_3d(structure, smiles)
    except ValueError:
        return None
    mol = Chem.RemoveHs(mol)
    iron_atom = find_heme_iron(structure)
    iron = np.asarray(iron_atom.coord, dtype=float)
    nitrogens = {a.get_id(): np.asarray(a.coord, dtype=float) for a in iron_atom.get_parent() if a.get_id() in ("NA", "NB", "NC", "ND") and np.linalg.norm(np.asarray(a.coord) - iron) < 3.0}
    heavy = mol.GetConformer().GetPositions()
    frame = pp.frame_from_nitrogens(nitrogens, iron, heavy)
    if frame is None:
        return None
    features = pp.features_of(mol, pp.feature_factory())
    return {"heavy": pp.to_frame(heavy, iron, frame), "features": {family: pp.to_frame(xyz, iron, frame) if len(xyz) else xyz for family, xyz in features.items()}, "features_raw": features, "mol": mol, "frame": frame, "iron": iron}


def score_pose(heavy: np.ndarray, features: dict[str, np.ndarray], fields: dict[str, np.ndarray], sites: dict[str, np.ndarray]) -> dict[str, float]:
    """field, occupancy, recall, matched, precision and outside of one pose given in the frame of the heme."""
    occupancy_at_atoms = pp.lookup(fields["occupancy"], heavy)
    present = [k for k, family in enumerate(pp.FAMILIES) if len(features[family])]
    parts = [float(pp.lookup(fields[pp.FAMILIES[k]], features[pp.FAMILIES[k]]).mean()) for k in present]
    if present:
        stacked = np.concatenate([features[pp.FAMILIES[k]] for k in present])
        family_of = np.concatenate([np.full(len(features[pp.FAMILIES[k]]), k) for k in present])
        recall, matched, precision = pp.site_match(stacked[None], family_of, present, sites, 1)
    else:
        recall, matched, precision = pp.site_match(None, None, [], sites, 1)
    return {"field": float(occupancy_at_atoms.mean()) + (float(np.mean(parts)) if parts else 0.0), "occupancy": float(occupancy_at_atoms.mean()), "recall": float(recall[0]), "matched": float(matched[0]),
            "precision": float(precision[0]), "outside": float((occupancy_at_atoms < SUPPORT_MIN).mean())}


# -- the pools of the gate -----------------------------------------------------------------------------------------------------------------------------------

def within_pool_scores(pool: pd.DataFrame) -> pd.DataFrame:
    """Add ``pharmacophore_model`` and ``pharmacophore_model_combined`` (mean within-pool ranks; higher is better) to a per-pose table with field, recall and mbr."""
    out = pool.copy()
    rank = lambda column: pool[column].rank(pct=True).to_numpy()
    out["pharmacophore_model"] = np.nanmean(np.column_stack([rank("field"), rank("recall")]), axis=1)
    out["pharmacophore_model_combined"] = np.nanmean(np.column_stack([pd.Series(out["pharmacophore_model"]).rank(pct=True).to_numpy(), rank("mbr")]), axis=1)
    return out


def evaluate(poses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-pose scores and, per ligand, the Spearman rho of each score with minus the true RMSD and the RMSD of the pose that each score picks; ``poses`` has compound, true_rmsd, mbr, field, recall."""
    from scipy.stats import spearmanr

    scored, rows = [], []
    for compound, pool in poses.groupby("compound", sort=True):
        s = within_pool_scores(pool.reset_index(drop=True))
        scored.append(s)
        row = {"compound": compound, "poses": len(s), "mean_pool": float(s.true_rmsd.mean()), "mbr_pick": float(s.loc[s.mbr.idxmax(), "true_rmsd"])}
        for name in (*SCORES, "field", "recall", "precision", "outside"):
            sign = -1.0 if name == "outside" else 1.0
            values = s[name]
            ok = values.notna()
            row[f"rho_{name}"] = float(spearmanr(values[ok] * sign, -s.loc[ok, "true_rmsd"]).statistic) if ok.sum() >= 3 and values[ok].nunique() > 1 else np.nan
            if name in SCORES:
                row[f"{name}_pick"] = float(s.loc[(values * sign).idxmax(), "true_rmsd"])
        rows.append(row)
    return pd.concat(scored, ignore_index=True), pd.DataFrame(rows)


def gate(per_ligand: pd.DataFrame, column: str) -> dict:
    rho = per_ligand[column].dropna()
    mean, fraction = (float(rho.mean()), float((rho > 0).mean())) if len(rho) else (float("nan"), float("nan"))
    return {"mean_rho": mean, "fraction_positive": fraction, "n": len(rho), "met": bool(len(rho) >= 3 and mean > MIN_MEAN_RHO and fraction >= MIN_FRACTION_POSITIVE)}


def use_test(per_ligand: pd.DataFrame, score: str) -> dict:
    """The paired test of the pick by a score against the pick by MBR: the mean RMSD difference, the wins, the worst drop-one mean and the one-sided paired Wilcoxon p."""
    from scipy.stats import wilcoxon

    gain = (per_ligand["mbr_pick"] - per_ligand[f"{score}_pick"]).to_numpy(dtype=float)
    drop_one = [float(np.delete(gain, i).mean()) for i in range(len(gain))]
    p = float(wilcoxon(gain, alternative="greater").pvalue) if len(gain) >= 3 and np.any(gain != 0) else 1.0
    return {"mean_gain": float(gain.mean()), "wins": int((gain > 0).sum()), "n": len(gain), "worst_drop_one": min(drop_one), "p": p, "met": bool(gain.mean() > 0 and min(drop_one) > DROP_ONE_MIN and p < WILCOXON_P)}


def pooled_poses(directory: Path) -> list[tuple[str, str, int, float, Path]]:
    """[(ligand, run, sample, true RMSD, CIF)] of the pooled Boltz-2 poses of ``pose_awareness`` for the benchmark ligands whose structure files exist."""
    from cypmode.validation.pose_awareness import pooled_pose_list
    from cypmode.validation.pose_pools import pool_paths

    smiles = benchmark_smiles()
    out = []
    for ligand, run, sample, rmsd in pooled_pose_list(directory):
        paths = {s: p for _, s, p in pool_paths(run, ("boltz",))}
        if ligand in smiles and sample in paths and paths[sample].exists():
            out.append((ligand, run, sample, rmsd, paths[sample]))
    return out


def benchmark_pools(directory: Path) -> dict[str, list[tuple[str, int, float, Path, float]]]:
    """{ligand: [(run, sample, true RMSD, CIF, MBR consensus)]} for the pooled Boltz-2 poses of ``pose_awareness``, ligands with at least 3 poses."""
    from cypmode.validation.pose_pools import pairwise_lddt

    by_ligand: dict[str, list[tuple[str, int, float, Path]]] = {}
    for ligand, run, sample, rmsd, cif in pooled_poses(directory):
        by_ligand.setdefault(ligand, []).append((run, sample, rmsd, cif))
    out = {}
    for ligand, poses in sorted(by_ligand.items()):
        if len(poses) < 3:
            continue
        matrix = pairwise_lddt([p for *_, p in poses])
        mbr = [float(np.mean([matrix[i, j] for j in range(len(poses)) if j != i])) for i in range(len(poses))]
        out[ligand] = [(*pose, mbr[k]) for k, pose in enumerate(poses)]
    return out


def benchmark_smiles() -> dict[str, str]:
    table = pd.read_csv("data/structure_track/inputs/structure_benchmark.csv")
    return dict(zip(table["Molecule_Name"], table["SMILES"], strict=True))


def score_pools(pools: dict[str, list], regime: str) -> pd.DataFrame:
    from cypmode.validation.structures import load_structure

    smiles = benchmark_smiles()
    rows = []
    for ligand, poses in pools.items():
        fields, sites = fields_and_sites(kept_indices(smiles[ligand], regime))
        for run, sample, rmsd, cif, mbr in poses:
            in_frame = pose_in_frame(load_structure(cif), smiles[ligand])
            if in_frame is not None:
                rows.append({"compound": ligand, "run": run, "sample": sample, "true_rmsd": rmsd, "mbr": mbr, "regime": regime, **score_pose(in_frame["heavy"], in_frame["features"], fields, sites)})
    return pd.DataFrame(rows)


def crystal_scores(regime: str) -> pd.DataFrame:
    """Information, not a gate: the same scores for the crystal pose of each benchmark ligand (the copies of its entries, averaged), with the fields built without the ligand's own references. A model that
    cannot score the bound pose above the predicted ones cannot rank poses either."""
    import gemmi

    from cypmode.validation.pose_rescoring import connectivity_key

    refs = references()
    factory = pp.feature_factory()
    rows = []
    for ligand, smiles in benchmark_smiles().items():
        own = [r for r in refs if connectivity_key(r["smiles"]) == connectivity_key(smiles)]
        fields, sites = fields_and_sites(kept_indices(smiles, regime))
        scores = []
        for record in own:
            mol = pp.crystal_pose(record)
            path = pp.PDB_CACHE / f"{record['pdb_id']}.cif"
            if mol is None or not path.exists():
                continue
            iron, xyz = np.array(record["fe"], dtype=float), np.array(record["coords"], dtype=float)
            frame = pp.heme_frame(gemmi.read_structure(str(path)), iron, xyz)
            if frame is None:
                continue
            found = pp.features_of(mol, factory)
            scores.append(score_pose(pp.to_frame(xyz, iron, frame), {k: pp.to_frame(v, iron, frame) if len(v) else v for k, v in found.items()}, fields, sites))
        if scores:
            rows.append({"compound": ligand, "regime": regime, "copies": len(scores), **{f"crystal_{k}": v for k, v in pd.DataFrame(scores).mean().items()}})
    return pd.DataFrame(rows)


def crystal_report(crystal: pd.DataFrame, scored: pd.DataFrame, regime: str) -> list[str]:
    """One line per score: the mean over the ligands of the crystal pose, of the predicted poses of the pool and of the predicted pose closest to the crystal pose."""
    joined = []
    for row in crystal.itertuples():
        pool = scored[scored.compound == row.compound]
        if len(pool):
            best = pool.sort_values("true_rmsd").iloc[0]
            joined.append({name: (getattr(row, f"crystal_{name}"), pool[name].mean(), best[name]) for name in ("field", "recall", "precision", "outside")})
    lines = [f"information (not a gate), regime {regime}: the crystal pose against the predicted poses, {len(joined)} ligands (mean of the crystal pose / of the pool / of the pose closest to the crystal)"]
    for name in ("field", "recall", "precision", "outside"):
        values = np.array([j[name] for j in joined], dtype=float)
        lines.append(f"  {name}: {np.nanmean(values[:, 0]):.3f} / {np.nanmean(values[:, 1]):.3f} / {np.nanmean(values[:, 2]):.3f}; the crystal pose scores above the mean of its pool for {int((values[:, 0] > values[:, 1]).sum())} of {len(joined)} ligands")
    return lines


def report(per_ligand: pd.DataFrame, regime: str, gated: bool) -> list[str]:
    """The lines of the log for one regime; the lines of the gates only for the gated regime."""
    lines = [f"regime {regime}{' (the gate)' if gated else ' (information)'}: {len(per_ligand)} ligands"]
    for name in (*SCORES, "field", "recall", "precision", "outside"):
        g = gate(per_ligand, f"rho_{name}")
        picks = per_ligand[f"{name}_pick"].mean() if f"{name}_pick" in per_ligand else float("nan")
        lines.append(f"{name}: mean Spearman rho {g['mean_rho']:+.3f}, positive for {g['fraction_positive']:.0%} of {g['n']} ligands" + (f", mean RMSD of the top-scored pose {picks:.3f} (MBR {per_ligand.mbr_pick.mean():.3f}, random pose {per_ligand.mean_pool.mean():.3f})" if name in SCORES else ""))
        if gated and name in SCORES:
            lines.append(f"STRUCTURE GATE pose_{name} (mean rho > {MIN_MEAN_RHO} and positive for >= {MIN_FRACTION_POSITIVE:.0%} of ligands): {'MET' if g['met'] else 'NOT MET'}")
            if g["met"]:
                u = use_test(per_ligand, name)
                lines.append(f"  use of {name}: mean improvement {u['mean_gain']:+.3f} A over {u['n']} ligands, {u['wins']} wins, Wilcoxon p {u['p']:.3f}, worst drop-one mean {u['worst_drop_one']:+.3f}")
                lines.append(f"STRUCTURE GATE pose_{name}_use (paired, drop-one robust, p < {WILCOXON_P}): {'MET' if u['met'] else 'NOT MET'}")
            else:
                lines.append(f"STRUCTURE GATE pose_{name}_use: not evaluated, the gate of the score is not met")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--directory", default="data/structure_track/outputs/cyp3a4")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    if not args.gate:
        raise SystemExit("give --gate")
    pools = benchmark_pools(Path(args.directory))
    tables, lines, crystals = [], [], []
    for regime in REGIMES:
        scored, per_ligand = evaluate(score_pools(pools, regime))
        tables.append(scored)
        print(per_ligand.round(3).to_string(index=False), flush=True)
        lines += report(per_ligand, regime, regime == GATED_REGIME)
        crystal = crystal_scores(regime)
        crystals.append(crystal)
        lines += crystal_report(crystal, scored, regime)
    pd.concat(crystals, ignore_index=True).to_csv(Path(args.out).with_name("pose_pharmacophore_crystal.csv"), index=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(args.out, index=False)
    print(f"{sum(len(v) for v in pools.values())} poses of {len(pools)} ligands")
    print("\n".join(lines))


if __name__ == "__main__":
    from cypmode import provenance

    provenance.announce()
    main()
