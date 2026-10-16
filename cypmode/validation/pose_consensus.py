"""Cross-engine pose consensus: Boltz-2 (the incumbent) vs OpenFold3 (the
added second engine, cypmode.validation.openfold3) for the Structure Track.

The Structure Track's accuracy metrics (LDDT-PLI, BiSyRMSD, LDDT-LP) are
scored against sequestered crystal structures, so no pose can be scored
locally. What CAN be measured per pose is (a) whether it is physically
valid (PoseBusters -- a failing pose scores zero on ALL THREE metrics, see
cypmode.challenge.structure_submission), (b) whether it has the heme
geometry a Type II motif calls for, and (c) whether two independent
engines agree on it. Confidence scores are deliberately NOT used: both
engines' confidence is an unreliable pose selector on P450s (the same
external finding cypmode.challenge.structure_submission cites).

Selection rule (choose_pose) -- conservative on purpose, because the
standing project rule is to ADD methods, never switch: Boltz-2 stays the
default and a pose only changes engine on concrete evidence.

1. Exactly one engine's pose passes PoseBusters -> use that one.
2. Neither passes -> the one with fewer failed checks (tie: Boltz-2).
3. Both pass and agree (pocket-aligned, symmetry-aware ligand RMSD <=
   AGREEMENT_RMSD_A) -> Boltz-2. Two independent engines landing on the
   same pose is the strongest robustness signal available without ground
   truth, and either is then within error of the other.
4. Both pass but disagree -> if the compound carries a Type II motif and
   exactly one engine coordinates the heme iron (closest motif N within
   COORDINATION_DISTANCE_CUTOFF_A), use that one: Type II azole/pyridine
   inhibitors coordinate the iron (ketoconazole: 2.05 A in Boltz-2, 2.14 A
   in OpenFold3).
5. Still unresolved -> the experimental-prior rescoring tie-breaker
   (cypmode.validation.pose_rescoring), which asks how well each pose
   agrees with where crystallized CYP3A4 ligands sit: the template score
   when BOTH poses have a close crystallized analog (similarity >=
   TEMPLATE_MIN_SUPPORT) and differ by >= TEMPLATE_MARGIN, else the
   occupancy score when it differs by >= OCCUPANCY_MARGIN. The margins
   are set from the score gaps measured between a crystal pose and
   rigidly displaced decoys (cypmode.validation.pose_rescoring_validation:
   the median occupancy gap at 2-3 A displacement is 0.093, so 0.08 asks
   for roughly one real 2-3 A displacement's worth of evidence).
   Measured caveat: against the experimental structures of five
   compounds, Boltz-2's own poses are 4-7.5 A off for four of them and
   the occupancy score does not clearly track that error -- real engine
   mistakes are plausible poses inside the occupied pocket, which decoys
   are not. So the margins are deliberately conservative.
Otherwise keep Boltz-2.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.validation.constants import COORDINATION_DISTANCE_CUTOFF_A
from cypmode.validation.openfold3 import model_cif_paths, read_boltz_inputs
from cypmode.validation.pose_rescoring import rescore_pose
from cypmode.validation.pose_validity import check_pose_validity
from cypmode.validation.reference_ligands import load_reference_set
from cypmode.validation.structures import (
    _STANDARD_RESIDUES,
    _ligand_mol_3d,
    find_heme_iron,
    load_structure,
    motif_nitrogen_distances,
)

logger = logging.getLogger(__name__)

AGREEMENT_RMSD_A = 2.0
POCKET_RADIUS_A = 12.0
OCCUPANCY_MARGIN = 0.08
TEMPLATE_MARGIN = 0.10
TEMPLATE_MIN_SUPPORT = 0.5
BOLTZ_OUT_ROOT = Path("data/structure_track/outputs/out")


def boltz_cif_paths(name: str, target: str = "cyp3a4", out_root: Path = BOLTZ_OUT_ROOT) -> list[Path]:
    """Boltz-2's sampled structures for one compound, in rank order
    (model_0 = the model's own top pick, model_1, ...).
    """
    stem = f"{target}_{name}"
    directory = out_root / target / f"boltz_results_{stem}" / "predictions" / stem
    return sorted(directory.glob(f"{stem}_model_*.cif"), key=lambda p: int(p.stem.rsplit("_model_", 1)[1]))


def _pocket_ca_pairs(reference, probe, radius: float) -> tuple[list, list]:
    """Matched CA atoms (by chain + residue number) of every protein
    residue whose CA lies within `radius` of the reference's heme iron --
    a local frame around the active site, where the ligand actually sits,
    rather than the whole fold (whose flexible loops would add noise to
    a ligand RMSD).
    """
    fe = np.asarray(find_heme_iron(reference).coord)
    probe_cas = {
        (res.get_parent().id, res.id[1]): res["CA"]
        for res in probe.get_residues()
        if res.resname in _STANDARD_RESIDUES and "CA" in res
    }
    ref_atoms, probe_atoms = [], []
    for res in reference.get_residues():
        if res.resname not in _STANDARD_RESIDUES or "CA" not in res:
            continue
        key = (res.get_parent().id, res.id[1])
        if key in probe_cas and np.linalg.norm(np.asarray(res["CA"].coord) - fe) <= radius:
            ref_atoms.append(res["CA"])
            probe_atoms.append(probe_cas[key])
    if len(ref_atoms) < 10:
        raise ValueError(f"only {len(ref_atoms)} matched pocket CA atoms within {radius} A of the heme iron -- structures don't correspond")
    return ref_atoms, probe_atoms


def aligned_ligand_rmsd(reference, probe, ligand_smiles: str, radius: float = POCKET_RADIUS_A) -> dict:
    """Pocket-aligned, symmetry-aware heavy-atom RMSD (A) between the
    inhibitor poses in two predicted complexes of the same system: the
    probe is superposed onto the reference on the CA atoms around the
    heme, then RDKit's CalcRMS compares the two ligands in place while
    minimizing over symmetry-equivalent atom mappings (so a flipped
    phenyl ring isn't counted as a pose difference). Both ligand mols are
    rebuilt from the same SMILES template, so their topologies match.
    Returns {"ligand_rmsd": float, "pocket_ca_rmsd": float}.
    """
    from Bio.PDB import Superimposer
    from rdkit.Chem import rdMolAlign

    ref_atoms, probe_atoms = _pocket_ca_pairs(reference, probe, radius)
    superimposer = Superimposer()
    superimposer.set_atoms(ref_atoms, probe_atoms)
    rotation, translation = superimposer.rotran  # moves probe onto reference: x @ rotation + translation

    ref_mol, _ = _ligand_mol_3d(reference, ligand_smiles)
    probe_mol, _ = _ligand_mol_3d(probe, ligand_smiles)
    conformer = probe_mol.GetConformer()
    for i in range(probe_mol.GetNumAtoms()):
        position = np.array(conformer.GetAtomPosition(i)) @ rotation + translation
        conformer.SetAtomPosition(i, position.tolist())

    return {
        "ligand_rmsd": float(rdMolAlign.CalcRMS(probe_mol, ref_mol)),
        "pocket_ca_rmsd": float(superimposer.rms),
    }


def pose_metrics(structure, ligand_smiles: str, references: list[dict] | None = None) -> dict:
    """What can be measured locally about one predicted pose: PoseBusters
    pass/fail (and which checks failed), the Type II heme geometry, and --
    when the crystallized-ligand `references` are given -- the
    experimental-prior rescoring (template_score, template_support,
    occupancy_score; see cypmode.validation.pose_rescoring).
    """
    validity = check_pose_validity(structure, ligand_smiles)
    motifs = motif_nitrogen_distances(structure, ligand_smiles, find_heme_iron(structure))
    closest = motifs[0]["distance_angstrom"] if motifs else None
    metrics = {
        "all_passed": validity["all_passed"],
        "failed_checks": [k for k, v in validity["checks"].items() if not v],
        "has_motif": bool(motifs),
        "closest_motif_n_fe": closest,
        "coordinated": closest is not None and closest <= COORDINATION_DISTANCE_CUTOFF_A,
    }
    if references:
        scores = rescore_pose(structure, ligand_smiles, references)
        metrics["rescore"] = {k: scores[k] for k in ("template_score", "template_support", "occupancy_score")}
    return metrics


def _decision(engine: str, reason: str) -> dict:
    return {"engine": engine, "reason": reason}


def choose_pose(boltz: dict | None, openfold3: dict | None, ligand_rmsd: float | None, agreement_rmsd: float = AGREEMENT_RMSD_A) -> dict:
    """The selection rule from the module docstring, as a pure function of
    the two poses' pose_metrics dicts (None = that engine has no
    prediction for this compound) and their ligand RMSD (None = couldn't
    be computed). Returns {"engine": "boltz" | "openfold3", "reason": str}.
    """
    if openfold3 is None:
        return _decision("boltz", "no_openfold3_prediction")
    if boltz is None:
        return _decision("openfold3", "no_boltz_prediction")

    boltz_ok, of3_ok = boltz["all_passed"], openfold3["all_passed"]
    if boltz_ok != of3_ok:
        engine = "boltz" if boltz_ok else "openfold3"
        return _decision(engine, f"only_{engine}_passes_posebusters")
    if not boltz_ok:
        engine = "openfold3" if len(openfold3["failed_checks"]) < len(boltz["failed_checks"]) else "boltz"
        return _decision(engine, "neither_passes_posebusters_fewest_failed_checks")

    if ligand_rmsd is not None and ligand_rmsd <= agreement_rmsd:
        return _decision("boltz", "engines_agree")
    if boltz["coordinated"] != openfold3["coordinated"]:
        engine = "boltz" if boltz["coordinated"] else "openfold3"
        return _decision(engine, "engines_disagree_heme_coordination_prior")

    rescored = _rescoring_choice(boltz.get("rescore"), openfold3.get("rescore"))
    if rescored is not None:
        return rescored
    return _decision("boltz", "engines_disagree_unresolved_keep_boltz")


def _rescoring_choice(boltz: dict | None, openfold3: dict | None) -> dict | None:
    """The tie-breaker step of choose_pose: template score if both poses
    have a close crystallized analog, else occupancy score, each only past
    its margin. None when there is no usable difference (or no rescoring).
    """
    if not boltz or not openfold3:
        return None
    for key, margin, support_needed in (("template_score", TEMPLATE_MARGIN, True), ("occupancy_score", OCCUPANCY_MARGIN, False)):
        b, o = boltz.get(key), openfold3.get(key)
        if b is None or o is None:
            continue
        if support_needed and min(boltz.get("template_support") or 0.0, openfold3.get("template_support") or 0.0) < TEMPLATE_MIN_SUPPORT:
            continue
        if abs(o - b) >= margin:
            engine = "openfold3" if o > b else "boltz"
            return _decision(engine, f"engines_disagree_{key.split('_')[0]}_rescoring")
    return None


def _best_candidate(cif_paths: list[Path], ligand_smiles: str, references: list[dict] | None = None) -> tuple[Path, object, dict] | None:
    """The first sampled structure (in rank order) that passes PoseBusters,
    or the first one if none pass -- the same "first passing sample"
    convention as cypmode.validation.pose_validity.select_valid_model.
    Returns (cif_path, structure, pose_metrics) or None if there are no
    structures.
    """
    first = None
    for cif in cif_paths:
        structure = load_structure(cif)
        try:
            metrics = pose_metrics(structure, ligand_smiles, references)
        except Exception as exc:
            metrics = {"all_passed": False, "failed_checks": [f"error: {exc}"], "has_motif": False, "closest_motif_n_fe": None, "coordinated": False}
        candidate = (cif, structure, metrics)
        if metrics["all_passed"]:
            return candidate
        first = first or candidate
    return first


def build_consensus_table(names: list[str], target: str = "cyp3a4", out_csv: Path | None = None, references: list[dict] | None = None) -> pd.DataFrame:
    """One row per compound: which engine's pose to use, why, and the
    measurements behind it. A compound with no OpenFold3 prediction yet
    keeps its Boltz-2 pose (reason "no_openfold3_prediction"), so the
    table is valid at any point of a partially finished OpenFold3 batch.
    `references` (cypmode.validation.reference_ligands.load_reference_set)
    switches on the rescoring tie-breaker; without it the rule stops at the
    heme-coordination prior.
    """
    rows = []
    for name in names:
        smiles = read_boltz_inputs(name, target)["ligand_smiles"]
        boltz = _best_candidate(boltz_cif_paths(name, target), smiles, references)
        of3 = _best_candidate(model_cif_paths(name, target), smiles, references)

        ligand_rmsd = pocket_ca_rmsd = rmsd_error = None
        if boltz and of3:
            try:
                alignment = aligned_ligand_rmsd(boltz[1], of3[1], smiles)
                ligand_rmsd, pocket_ca_rmsd = alignment["ligand_rmsd"], alignment["pocket_ca_rmsd"]
            except Exception as exc:
                # An unmeasurable RMSD must not drop the compound: choose_pose treats None as "unknown" (disagreement).
                rmsd_error = str(exc)
                logger.warning("ligand RMSD between engines failed for %s: %s", name, exc)

        decision = choose_pose(boltz[2] if boltz else None, of3[2] if of3 else None, ligand_rmsd)
        chosen = of3 if decision["engine"] == "openfold3" else boltz
        row = {
            "compound": name,
            "engine": decision["engine"],
            "reason": decision["reason"],
            "chosen_cif": str(chosen[0]) if chosen else None,
            "ligand_rmsd": ligand_rmsd,
            "pocket_ca_rmsd": pocket_ca_rmsd,
            "rmsd_error": rmsd_error,
        }
        for label, candidate in (("boltz", boltz), ("of3", of3)):
            m = candidate[2] if candidate else {}
            row[f"{label}_posebusters_ok"] = m.get("all_passed")
            row[f"{label}_failed_checks"] = ";".join(m.get("failed_checks", []))
            row[f"{label}_motif_n_fe"] = m.get("closest_motif_n_fe")
            row[f"{label}_coordinated"] = m.get("coordinated")
            rescore = m.get("rescore") or {}
            row[f"{label}_template_score"] = rescore.get("template_score")
            row[f"{label}_template_support"] = rescore.get("template_support")
            row[f"{label}_occupancy_score"] = rescore.get("occupancy_score")
        rows.append(row)

    table = pd.DataFrame(rows)
    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_csv, index=False)
    return table


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default="cyp3a4")
    parser.add_argument("--test-set", default="data/structure_track/inputs/structure_track_test_blinded.csv")
    parser.add_argument("--out", default=None, help="CSV path (default: data/structure_track/outputs/{target}/engine_consensus.csv)")
    parser.add_argument("--no-rescoring", action="store_true", help="skip the crystal-ligand rescoring tie-breaker (stop at the heme-coordination prior)")
    args = parser.parse_args()

    names = pd.read_csv(args.test_set)["Molecule_Name"].tolist()
    out_csv = Path(args.out or f"data/structure_track/outputs/{args.target}/engine_consensus.csv")
    references = None if args.no_rescoring else load_reference_set()
    table = build_consensus_table(names, args.target, out_csv, references)
    print(table.groupby(["engine", "reason"]).size().to_string())
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    _main()
