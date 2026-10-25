# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Does cypmode.validation.pose_rescoring actually discriminate a
near-native pose from a wrong one? Measured, not assumed -- using the
experimental structures themselves as ground truth.

For every distinct crystal ligand in the reference set, in its own
crystal frame:

- the TRUE pose (RMSD 0) is scored against rigidly displaced DECOYS of
  itself (random rotation about the centroid up to 180 degrees plus a
  random translation up to MAX_SHIFT_A), with RMSDs to the crystal pose
  computed under the identity atom mapping;
- every reference that is the same compound is excluded (leave-compound-
  out), so the score cannot just find the answer in the reference set;
- optionally every reference at least NOVEL_SIMILARITY chemically similar
  is excluded too, mimicking a novel scaffold with no crystallized
  analogs -- the realistic situation for the OCNT-* compounds, and
  the one where the template score has little to say.

Per ligand, per score: Spearman correlation between score and -RMSD over
the pose set, the AUROC for near-native (RMSD < NEAR_NATIVE_A) vs wrong
(RMSD > WRONG_A) poses, and whether the true pose ranks first. This
tests the scoring function's sensitivity to displacement and orientation;
it does NOT test protein-ligand physics (decoys may clash with the
protein, which PoseBusters checks separately), so a good result here is
necessary evidence for using the score as a tie-breaker, not proof.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from cypmode.validation.pose_rescoring import align_references, score_against_aligned
from cypmode.validation.reference_ligands import load_reference_set

NEAR_NATIVE_A = 2.0
WRONG_A = 4.0
RMSD_BINS_A = ((0.5, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, 20.0))
MAX_SHIFT_A = 6.0
NOVEL_SIMILARITY = 0.5
OUT_CSV = Path("data/structure_track/outputs/cyp3a4/pose_rescoring_validation.csv")


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def make_decoys(crystal: np.ndarray, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """n rigid-body perturbations of a ligand pose: random rotation axis,
    angle uniform in [0, 180] degrees, about the centroid, plus a random
    translation of uniform length in [0, MAX_SHIFT_A].
    """
    centroid = crystal.mean(axis=0)
    decoys = []
    for _ in range(n):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rotation = Rotation.from_rotvec(axis * rng.uniform(0, np.pi))
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        decoys.append(rotation.apply(crystal - centroid) + centroid + direction * rng.uniform(0, MAX_SHIFT_A))
    return decoys


def one_per_compound(references: list[dict], min_heavy: int = 15) -> list[dict]:
    """The best-resolution copy of each distinct ligand with at least
    min_heavy heavy atoms (a handful of atoms cannot define an orientation).
    """
    best: dict[str, dict] = {}
    for ref in references:
        if ref["n_heavy"] < min_heavy or not ref.get("smiles"):
            continue
        current = best.get(ref["comp_id"])
        if current is None or (ref["resolution"] or 99) < (current["resolution"] or 99):
            best[ref["comp_id"]] = ref
    return list(best.values())


def score_decoy_set(reference: dict, references: list[dict], n_decoys: int, rng: np.random.Generator, similarity_cut: float | None) -> dict | None:
    """Metrics for one crystal ligand, or None if the leave-out leaves too
    little to score (no references, or no usable template similarity).
    """
    crystal = np.asarray(reference["coords"], dtype=float)
    fe = np.asarray(reference["fe"], dtype=float)
    pocket_ca = {int(k): np.asarray(v, dtype=float) for k, v in reference["pocket_ca"].items()}

    aligned = align_references(reference["smiles"], fe, pocket_ca, references, exclude_compound=reference["smiles"], exclude_similarity_at_least=similarity_cut)
    if len(aligned) < 20:
        return None
    support = max((a[0] for a in aligned if a[0] is not None), default=None)

    poses = [crystal, *make_decoys(crystal, n_decoys, rng)]
    rmsds = np.array([rmsd(p, crystal) for p in poses])
    scores = [score_against_aligned(p, aligned) for p in poses]

    row = {
        "comp_id": reference["comp_id"],
        "pdb_id": reference["pdb_id"],
        "n_heavy": reference["n_heavy"],
        "similarity_cut": similarity_cut,
        "n_refs_used": len(aligned),
        "best_available_similarity": support,
    }
    positive, negative = rmsds < NEAR_NATIVE_A, rmsds > WRONG_A
    for name in ("template_score", "occupancy_score"):
        values = np.array([np.nan if s[name] is None else s[name] for s in scores], dtype=float)
        if np.isnan(values).any():
            row.update({f"{name}_spearman": np.nan, f"{name}_auc": np.nan, f"{name}_true_pose_rank": np.nan})
            continue
        row[f"{name}_spearman"] = float(spearmanr(values, -rmsds).statistic)
        row[f"{name}_auc"] = float(roc_auc_score(np.r_[np.ones(positive.sum()), np.zeros(negative.sum())], np.r_[values[positive], values[negative]]))
        row[f"{name}_true_pose_rank"] = int((values > values[0]).sum()) + 1  # 1 = the crystal pose outscored every decoy
        for lo, hi in RMSD_BINS_A:
            in_bin = (rmsds >= lo) & (rmsds < hi)
            in_bin[0] = False
            row[f"{name}_win_{lo:g}_{hi:g}"] = float((values[0] > values[in_bin]).mean()) if in_bin.any() else np.nan
    return row


def validate(references: list[dict] | None = None, n_decoys: int = 60, seed: int = 0, out_csv: Path | None = OUT_CSV) -> pd.DataFrame:
    references = references or load_reference_set()
    rng = np.random.default_rng(seed)
    rows = []
    for reference in one_per_compound(references):
        for cut in (None, NOVEL_SIMILARITY):
            row = score_decoy_set(reference, references, n_decoys, rng, cut)
            if row is not None:
                rows.append(row)
    table = pd.DataFrame(rows)
    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_csv, index=False)
    return table


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Mean Spearman, mean AUROC and the fraction of ligands whose true
    pose ranked first, per score and per leave-out regime.
    """
    out = []
    for cut, group in table.groupby(table["similarity_cut"].fillna(-1)):
        for name in ("template_score", "occupancy_score"):
            out.append(
                {
                    "regime": "own compound only excluded" if cut == -1 else f"also excl. similarity >= {cut:g}",
                    "score": name,
                    "n_ligands": int(group[f"{name}_auc"].notna().sum()),
                    "mean_spearman": group[f"{name}_spearman"].mean(),
                    "mean_auc": group[f"{name}_auc"].mean(),
                    "true_pose_ranked_first": float((group[f"{name}_true_pose_rank"] == 1).mean()),
                }
            )
    return pd.DataFrame(out)


def win_rates(table: pd.DataFrame) -> pd.DataFrame:
    """How often the TRUE pose outscores a decoy at each displacement --
    the resolution of the score: near-native-vs-wrong at 2-3 A is the
    regime of a real disagreement between two engines, 6+ A is easy.
    Mean over ligands, per score and regime; 0.5 = no better than a coin flip.
    """
    rows = []
    for cut, group in table.groupby(table["similarity_cut"].fillna(-1)):
        for name in ("template_score", "occupancy_score"):
            rows.append(
                {"regime": "own excl." if cut == -1 else f"sim<{cut:g}", "score": name.replace("_score", "")}
                | {f"{lo:g}-{hi:g} A": group[f"{name}_win_{lo:g}_{hi:g}"].mean() for lo, hi in RMSD_BINS_A}
            )
    return pd.DataFrame(rows)


# --- real engine poses vs the crystal structure ---------------------------------------------------


def crystal_mol(smiles: str, elements: list[str], coords: np.ndarray):
    """RDKit mol for a crystal ligand: bonds and aromaticity from the
    compound's SMILES template, 3D coordinates from the crystal (the same
    proximity-bonding + AssignBondOrdersFromTemplate recipe as
    cypmode.validation.structures._ligand_mol_3d, but from plain arrays).
    None if the crystal ligand does not match the template (missing or
    modified atoms) -- such a copy is skipped, never forced.
    """
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem

    lines = [
        f"HETATM{i:>5} {el + str(i):<4} LIG A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2}"
        for i, (el, (x, y, z)) in enumerate(zip(elements, coords, strict=True), start=1)
    ]
    with rdBase.BlockLogs():
        raw = Chem.MolFromPDBBlock("\n".join([*lines, "END"]), sanitize=False, proximityBonding=True)
        template = Chem.MolFromSmiles(smiles)
        if raw is None or template is None:
            return None
        try:
            return AllChem.AssignBondOrdersFromTemplate(template, raw)
        except Exception:
            return None


def true_ligand_rmsd(pose_structure, smiles: str, crystal_records: list[dict]) -> float | None:
    """Pocket-aligned, symmetry-aware heavy-atom RMSD (A) between a predicted
    pose and the crystal pose -- the smallest over the crystal copies
    (different entries/chains of the same ligand sample slightly different
    poses, so "the" crystal pose is the nearest experimental one). None if
    no copy could be compared.
    """
    from rdkit import rdBase
    from rdkit.Chem import rdMolAlign

    from cypmode.validation.pose_rescoring import structure_arrays, superpose_reference
    from cypmode.validation.structures import _ligand_mol_3d

    _, fe, pocket_ca = structure_arrays(pose_structure)
    try:
        pose_mol, _ = _ligand_mol_3d(pose_structure, smiles)
    except ValueError:  # bond orders cannot be assigned to a distorted pose: no RMSD for it, the pose stays in the pool
        return None
    best = None
    for record in crystal_records:
        superposed = superpose_reference(record, fe, pocket_ca)
        if superposed is None:
            continue
        mol = crystal_mol(smiles, record["elements"], superposed[0])
        if mol is None:
            continue
        try:
            with rdBase.BlockLogs():
                value = float(rdMolAlign.CalcRMS(pose_mol, mol))
        except Exception:
            continue
        best = value if best is None else min(best, value)
    return best


def collect_poses(names: list[str], target: str = "cyp3a4", engines: tuple[str, ...] = ("boltz", "openfold3")) -> dict[str, list[tuple[str, int, Path]]]:
    """{compound: [(engine, sample index, CIF path), ...]} for every sampled structure of the named engines (Boltz-2 and OpenFold3 by default)."""
    from cypmode.validation.pose_pools import pool_paths

    return {name: pool_paths(name, engines, target) for name in names}


def validate_engine_poses(
    pose_sources: dict[str, list[tuple[str, int, Path]]],
    references: list[dict] | None = None,
    target: str = "cyp3a4",
    out_csv: Path | None = None,
) -> pd.DataFrame:
    """One row per predicted pose of a compound that has crystal copies in
    the reference set: its TRUE ligand RMSD to the crystal pose, and the
    rescoring scores computed with that compound's own references
    excluded (so the score cannot read the answer from the crystal
    entries), both with only the compound excluded and with every
    reference >= NOVEL_SIMILARITY similar excluded too.
    """
    from cypmode.validation.openfold3 import read_boltz_inputs
    from cypmode.validation.pose_rescoring import (
        align_references,
        connectivity_key,
        score_against_aligned,
        structure_arrays,
    )
    from cypmode.validation.structures import load_structure

    references = references or load_reference_set()
    rows = []
    for name, poses in pose_sources.items():
        smiles = read_boltz_inputs(name, target)["ligand_smiles"]
        key = connectivity_key(smiles)
        crystal_records = [r for r in references if key and connectivity_key(r["smiles"]) == key]
        if not crystal_records:
            continue
        for engine, sample, path in poses:
            structure = load_structure(path)
            ligand, fe, pocket_ca = structure_arrays(structure)
            row = {
                "compound": name,
                "engine": engine,
                "sample": sample,
                "n_crystal_copies": len(crystal_records),
                "true_rmsd": true_ligand_rmsd(structure, smiles, crystal_records),
            }
            for label, cut in (("", None), ("novel_", NOVEL_SIMILARITY)):
                aligned = align_references(smiles, fe, pocket_ca, references, exclude_compound=smiles, exclude_similarity_at_least=cut)
                scores = score_against_aligned(ligand, aligned)
                row[f"{label}template_score"] = scores["template_score"]
                row[f"{label}template_support"] = scores["template_support"]
                row[f"{label}occupancy_score"] = scores["occupancy_score"]
            rows.append(row)
    table = pd.DataFrame(rows)
    if out_csv is not None and len(table):
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_csv, index=False)
    return table


ENGINE_POSE_CSV = Path("data/structure_track/outputs/cyp3a4/engine_pose_validation.csv")
# the reference-panel compounds that also have a CYP3A4 co-crystal structure in the reference set
PANEL_WITH_CRYSTALS = ("azamulin", "clotrimazole", "ketoconazole", "ritonavir", "vardenafil")


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate the pose rescoring scores against crystal structures.")
    parser.add_argument("--engine-poses", action="store_true", help="score real Boltz-2 / OpenFold3 poses against their crystal structures instead of running the decoy test")
    parser.add_argument("--names", nargs="+", default=list(PANEL_WITH_CRYSTALS), help="compounds for --engine-poses")
    parser.add_argument("--engines", nargs="+", default=["boltz", "openfold3"], choices=("boltz", "openfold3", "bioemu2"), help="engines whose poses --engine-poses scores")
    parser.add_argument("--out", default=str(ENGINE_POSE_CSV), help="where --engine-poses writes its table")
    args = parser.parse_args()

    if args.engine_poses:
        table = validate_engine_poses(collect_poses(args.names, engines=tuple(args.engines)), out_csv=Path(args.out))
        print(table.round(3).to_string(index=False))
        print(f"wrote {args.out}")
        return

    table = validate()
    print(summarize(table).round(3).to_string(index=False))
    print("\nP(true pose outscores a decoy) by decoy displacement from the crystal pose:")
    print(win_rates(table).round(3).to_string(index=False))
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    _main()
