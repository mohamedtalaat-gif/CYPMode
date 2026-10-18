"""Gate-driven final structure recipe: decide from the verdicts, run on the 20 official ligands, select, validate, zip.

Nothing is chosen by hand. The gates of the structure track (jobs 110-175) write ``STRUCTURE GATE <name>: MET|NOT MET``
lines to their logs; this module reads them and applies one fixed mapping from verdicts to the recipe:

  restraint       the donor restraint (bond or contact, the one with the lower mean first-pick RMSD if both are MET) is
                  switched on for the whole batch (ligands without a Type II donor run unrestrained by themselves); the bond restraint is
                  withdrawn when restraint_bond_harm is MET (the registered confirmation on new ligands, ``restraint_expansion``);
  pool            base = 5 unguided Boltz samples per ligand (existing samples count), plus 3 runs of every lever whose
                  pilot gate is MET (seeds, crystal conformer templates, step scale 0.8, cryo-EM templates), plus 3 restrained samples when the
                  restraint gate is MET, plus an extra budget of ``--extra`` samples per ligand on average;
  extra budget    allocated by pool disagreement (``ei_sampling``, in rounds) when the ei_sampling gate is MET, else uniform;
  engines         OpenFold3 (3 samples per ligand) only when engine_consensus is MET;
  selection       among PoseBusters-valid poses: MBR when mbr_selection is MET (MBR + heme prior when that gate is MET too),
                  else the first valid pose, taken from the restrained run for a ligand with a Type II donor when the restraint gate is MET
                  (the pick that gate measured) and from the base run otherwise (the Sep 27 behaviour); the physics, pharmacophore and series terms of
                  ``pose_awareness`` / ``series_consensus`` join the mean rank (with MBR) for every gate that is MET.

``--decide-only`` writes the decision and its reasons and stops. The run resumes from the batch checkpoints, uses one heavy
process at a time, never uploads anything, and writes the candidate zip under ``data/structure_track/outputs/cyp3a4/final``
(the earlier Boltz-only zip in ``submissions/structure_track`` stays untouched as the fallback).

Usage: python -m cypmode.validation.final_recipe [--decide-only] [--extra 2]
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.validation.official_posebusters import MARGIN_FLOOR

LOGS = Path("data/structure_track/outputs/cyp3a4/logs")
OUT = Path("data/structure_track/outputs/cyp3a4")
FINAL = OUT / "final"
INPUTS = Path("data/structure_track/inputs")
OFFICIAL = INPUTS / "structure_track_test_blinded.csv"
MSA = INPUTS / "msa" / "cyp3a4_P08684.csv"
GATE_LOGS = ("selection_validation.log", "heme_frame_prior.log", "ei_sampling.log", "structure_gates.log", "pose_awareness.log", "series_consensus.log", "restraint_expansion.log")
GATE_LINE = re.compile(r"^STRUCTURE GATE (\w+)[^:\n]*: (MET|NOT MET)", re.MULTILINE)
BASE_SAMPLES, LEVER_RUNS, RESTRAINT_SAMPLES, OPENFOLD3_SAMPLES = 5, 3, 3, 3
LEVER_SEED_OFFSET = 1000
HOURS_PER_BOLTZ_SAMPLE, HOURS_PER_OPENFOLD3_SAMPLE = 0.12, 0.27  # measured on this Mac: about 7 min and 16 min


@dataclass
class Recipe:
    restraint: str | None = None
    levers: list[str] = field(default_factory=list)
    openfold3: bool = False
    selection: str = "first_valid"
    terms: list[str] = field(default_factory=list)  # extra rank terms combined with MBR when selection is "rank_terms"
    ei_allocation: bool = False
    extra_per_ligand: int = 0
    reasons: list[str] = field(default_factory=list)


def read_gates(logs: Path = LOGS) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for name in GATE_LOGS:
        path = logs / name
        if path.exists():
            out.update({gate: verdict == "MET" for gate, verdict in GATE_LINE.findall(path.read_text())})
    return out


def decide(gates: dict[str, bool], restraint_means: dict[str, float] | None = None, extra_per_ligand: int = 2) -> Recipe:
    r = Recipe(extra_per_ligand=extra_per_ligand)
    met = [g for g in ("restraint_bond", "restraint_contact") if gates.get(g)]
    if gates.get("restraint_bond_harm"):  # the registered check on ligands that were not used to decide the restraint found it worse (restraint_expansion_plan.json)
        met = [g for g in met if g != "restraint_bond"]
        r.reasons.append("restraint 'bond' withdrawn: restraint_bond_harm MET on the confirmation ligands")
    if met:
        means = restraint_means or {}
        r.restraint = min(met, key=lambda g: means.get(g, 0.0)).removeprefix("restraint_")
        r.reasons.append(f"restraint '{r.restraint}': gate MET")
    else:
        r.reasons.append("no restraint: no restraint gate MET" + ("" if any(g.startswith("restraint") for g in gates) else " (no verdict available yet)"))
    r.levers = [g.removeprefix("pilot_") for g in ("pilot_seeds", "pilot_conformers", "pilot_stepscale", "pilot_cryoem") if gates.get(g)]
    r.reasons.append(f"diversity levers: {r.levers or 'none (no pilot gate MET)'}")
    r.openfold3 = bool(gates.get("engine_consensus"))
    r.reasons.append("OpenFold3 added: engine_consensus MET" if r.openfold3 else "Boltz only: engine_consensus not MET")
    if gates.get("mbr_selection"):
        r.selection = "mbr_plus_prior" if gates.get("heme_frame_prior") else "mbr"
        r.reasons.append(f"selection '{r.selection}': mbr_selection MET" + (", heme_frame_prior MET" if gates.get("heme_frame_prior") else ""))
    else:
        r.reasons.append("selection 'first_valid': mbr_selection not MET, so the model's own ranking among valid poses is kept"
                         + ("; for a ligand with a Type II donor the restrained run is read first (the pick the restraint gate measured)" if r.restraint else ""))
    extra = ["physics", "pharmacophore"] if gates.get("pose_combined") else [t for t, g in (("physics", "pose_physics"), ("pharmacophore", "pose_pharmacophore")) if gates.get(g)]
    if gates.get("series_consensus"):
        extra.append("series")
    if extra:
        r.terms, r.selection = extra, "rank_terms"
        r.reasons.append(f"selection 'rank_terms': MBR rank plus {extra}" + (" plus the heme-frame prior" if gates.get("heme_frame_prior") else ""))
    r.ei_allocation = bool(gates.get("ei_sampling"))
    r.reasons.append("extra samples allocated by pool disagreement: ei_sampling MET" if r.ei_allocation else "extra samples allocated uniformly: ei_sampling not MET")
    r.reasons.append(f"margin: among valid poses only those with a relative distance to the receptor of at least {MARGIN_FLOOR} under the scorer's own PoseBusters call are eligible (else the largest margin)")
    return r


def estimate_hours(recipe: Recipe, n_ligands: int, existing: dict[str, int]) -> float:
    boltz = sum(max(0, BASE_SAMPLES - c) for c in existing.values())
    boltz += n_ligands * (LEVER_RUNS * len(recipe.levers) + (RESTRAINT_SAMPLES if recipe.restraint else 0) + recipe.extra_per_ligand)
    return boltz * HOURS_PER_BOLTZ_SAMPLE + (n_ligands * OPENFOLD3_SAMPLES * HOURS_PER_OPENFOLD3_SAMPLE if recipe.openfold3 else 0.0)


def existing_counts(names: list[str]) -> dict[str, int]:
    from cypmode.validation.pose_consensus import boltz_cif_paths

    return {n: len(boltz_cif_paths(n)) for n in names}


def run_groups(recipe: Recipe, official: pd.DataFrame, existing: dict[str, int], references: list[dict]) -> list[dict]:
    """Batch jobs the recipe needs: [{label, rows (DataFrame), flags}], each resumable through its checkpoint JSON."""
    from cypmode.validation.cryoem_templates import build_cryoem_table
    from cypmode.validation.cryoem_templates import load_records as load_cryoem_records
    from cypmode.validation.diversity_pilot import (
        build_pilot_table,
        lever_of,
    )

    groups = []
    need = official[official.Molecule_Name.map(lambda n: existing[n] < BASE_SAMPLES)]
    if len(need):
        rows = pd.DataFrame({"Molecule_Name": need.Molecule_Name + "-u", "SMILES": need.SMILES, "Diffusion_Samples": [BASE_SAMPLES - existing[n] for n in need.Molecule_Name], "Seed": 7})
        groups.append({"label": "base_unguided", "rows": rows, "flags": []})
    if recipe.restraint:
        rows = pd.DataFrame({"Molecule_Name": official.Molecule_Name + "-rs", "SMILES": official.SMILES, "Diffusion_Samples": RESTRAINT_SAMPLES, "Seed": 11})
        groups.append({"label": f"restraint_{recipe.restraint}", "rows": rows, "flags": ["--donor-restraint", recipe.restraint]})
    if recipe.levers:
        table = build_pilot_table(official, references, LEVER_RUNS)
        if "cryoem" in recipe.levers:
            table = pd.concat([table, build_cryoem_table(official, load_cryoem_records(), LEVER_RUNS)], ignore_index=True)
        keep = table[table.Molecule_Name.map(lambda n: lever_of(n) in recipe.levers)]
        groups.append({"label": "diversity_levers", "rows": keep.reset_index(drop=True), "flags": []})
    return groups


def execute(group: dict, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path, results = run_dir / f"{group['label']}.csv", run_dir / f"{group['label']}_results.json"
    group["rows"].to_csv(csv_path, index=False)
    command = [sys.executable, "-m", "cypmode.validation.batch", str(csv_path), str(results), "--no-affinity", "--max-parallel-samples", "1", "--protein-msa", str(MSA), *group["flags"]]
    for attempt in (1, 2):
        subprocess.run([sys.executable, "scripts/clear_batch_errors.py", str(results)], check=False)
        code = subprocess.run(command, check=False).returncode
        failed = [k for k, v in json.loads(results.read_text()).items() if "error" in v] if results.exists() else ["no results file"]
        if code == 0 and not failed:
            return
        print(f"{group['label']}: attempt {attempt} left {len(failed)} failed entries")
    raise RuntimeError(f"{group['label']} did not complete")


def run_order(ligand: str, runs: list[str], donor: bool, restraint: str | None) -> list[str]:
    """The runs of a ligand in the order ``first_valid`` reads them. The restraint gate compared the first pick of the restrained run with the first
    pick of the unguided run on ligands that have a Type II donor, so when that gate is MET the restrained run comes first for such a ligand
    (the base run stays behind it as the fallback when no restrained pose is valid)."""
    restrained = f"{ligand}-rs"
    if restraint and donor and restrained in runs:
        return [restrained] + [r for r in runs if r != restrained]
    return list(runs)


def pool_for(ligand: str, names_by_ligand: dict[str, list[str]], engines: tuple[str, ...]) -> list[tuple[str, str, Path]]:
    """[(run name, engine, CIF path)] for every sample of a ligand across all of its runs."""
    from cypmode.validation.pose_pools import pool_paths

    out = []
    for run in names_by_ligand[ligand]:
        out += [(run, e, p) for e, _, p in pool_paths(run, engines) if p.exists()]
    return out


def margin_candidates(candidates: list[int], ok: np.ndarray, margins: np.ndarray, floor: float = MARGIN_FLOOR) -> list[int]:
    """The poses that may be submitted once the margin to the scorer's PoseBusters thresholds is taken into account.

    Among the valid poses those whose smallest relative distance to the receptor is at least ``floor`` are kept; when none clears it, the valid pose
    with the largest margin is kept alone (a pose that passes by a hair is the last resort, never the first choice); when no pose is valid the
    candidates are unchanged. Nothing here looks at how accurate a pose is: the choice among the survivors stays with the selection rule.
    """
    valid = [i for i in candidates if ok[i] and np.isfinite(margins[i])]
    if not valid:
        return candidates
    safe = [i for i in valid if margins[i] >= floor]
    return safe or [max(valid, key=lambda i: margins[i])]


def select_pose(pool: list[tuple[str, str, Path]], selection: str, valid, prior=None, extras: dict | None = None, smiles: str | None = None, margin=None) -> tuple[int, bool, np.ndarray]:
    """Index of the pose to submit, whether it passed PoseBusters, and the pool's MBR scores.

    ``valid(path) -> bool`` is the PoseBusters verdict; ``prior(path) -> float`` the heme-frame log density; ``extras`` maps a term
    name to one score per pool pose (higher is better) that joins the mean rank when ``selection`` is ``"rank_terms"``;
    ``margin(path) -> float`` the smallest relative distance to the receptor (see ``margin_candidates``).
    """
    from scipy.stats import rankdata

    from cypmode.validation.pose_pools import pairwise_lddt

    ok = np.array([valid(p) for _, _, p in pool])
    candidates = [i for i in range(len(pool)) if ok[i]] or list(range(len(pool)))
    if margin is not None:
        candidates = margin_candidates(candidates, ok, np.array([margin(p) for _, _, p in pool], dtype=float))
    if selection == "first_valid" or len(pool) < 2:
        return candidates[0], bool(ok[candidates[0]]), np.zeros(len(pool))
    matrix = pairwise_lddt([p for _, _, p in pool], smiles=smiles)  # the SMILES lets a pool that mixes engines be compared atom by atom
    mbr = np.array([np.mean([matrix[i, j] for j in candidates if j != i]) if i in candidates else -1.0 for i in range(len(pool))])
    score = rankdata(mbr[candidates])
    if selection in ("mbr_plus_prior", "rank_terms") and prior is not None:
        score = score + rankdata([prior(pool[i][2]) for i in candidates])
    if selection == "rank_terms":
        for values in (extras or {}).values():
            values = np.asarray(values, dtype=float)
            if np.isfinite(values[candidates]).any():
                score = score + rankdata(np.nan_to_num(values[candidates], nan=np.nanmin(values[candidates])))
    best = candidates[int(np.argmax(score))]
    return best, bool(ok[best]), mbr


def pose_extras(ligand: str, smiles: str, pool: list, terms: list[str], partner: dict) -> dict:
    """Per-pose scores (higher is better) for the extra rank terms; poses whose ligand terms cannot be computed get NaN."""
    from cypmode.validation.pose_awareness import (
        ligand_context,
        pose_terms,
        within_pool_scores,
    )

    out: dict[str, np.ndarray] = {}
    if {"physics", "pharmacophore"} & set(terms):
        context = ligand_context(smiles)
        rows = [pose_terms(path, smiles, context) or {} for _, _, path in pool]
        table = pd.DataFrame(rows).reindex(columns=["conf_rmsd", "conf_energy", "burial", "contacts", "clashes", "donor_error", "site_error"])
        table["mbr"] = 0.0
        scored = within_pool_scores(table)
        for name in ("physics", "pharmacophore"):
            if name in terms:
                out[name] = scored[name].to_numpy(dtype=float)
    if "series" in terms and partner:
        from cypmode.validation.series import common_core, core_rmsd
        from cypmode.validation.structures import load_structure

        mates = [b if a == ligand else a for a, b, _ in partner["pairs"] if ligand in (a, b)]
        if mates:
            mate = mates[0]
            mate_pool = pool_for(mate, partner["names_by_ligand"], partner["engines"])
            core = common_core(smiles, partner["smiles"][mate])
            mate_structures = [load_structure(p) for _, _, p in mate_pool]
            own = [load_structure(p) for _, _, p in pool]
            matrix = np.array([[core_rmsd(sa, smiles, sb, partner["smiles"][mate], core) for sb in mate_structures] for sa in own])
            out["series"] = -np.nanmean(matrix, axis=1)
    return out


def ligand_table(official: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, pairs: list[tuple[str, str, float]]) -> pd.DataFrame:
    """One row per official ligand: Type II donor, the closest analog among the official ligands, and the CYP3A4 measurements of the
    same compound in the TDI training set (NaN when it is not there). Informational: no decision rule is attached to it."""
    from rdkit import Chem

    from cypmode.validation.build_input import heme_donor_atom

    canonical = lambda smi: Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    known = {canonical(smi): row for smi, row in zip(train.SMILES, train.itertuples(index=False), strict=True)}
    in_test = {canonical(smi) for smi in test.SMILES}
    mate = {}
    for a, b, similarity in pairs:
        mate[a], mate[b] = (b, similarity), (a, similarity)
    rows = []
    for name, smi in zip(official.Molecule_Name, official.SMILES, strict=True):
        key = canonical(smi)
        record = known.get(key)
        rows.append({
            "ligand": name, "type_ii_donor": heme_donor_atom(smi) is not None, "analog": mate.get(name, (None, None))[0], "analog_tanimoto": mate.get(name, (None, None))[1],
            "in_tdi_train": record is not None, "in_tdi_test": key in in_test,
            "cyp3a4_direct_pic50": getattr(record, "CYP3A4_pIC50_direct_inhibition", float("nan")) if record is not None else float("nan"),
            "cyp3a4_tdi_pic50": getattr(record, "CYP3A4_pIC50_TDI_condition", float("nan")) if record is not None else float("nan"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decide-only", action="store_true")
    parser.add_argument("--extra", type=int, default=2, help="extra Boltz samples per ligand on average")
    args = parser.parse_args()
    from cypmode.challenge.submission import portal_validate_structure
    from cypmode.validation.reference_ligands import load_reference_set

    official = pd.read_csv(OFFICIAL)
    names = official.Molecule_Name.tolist()
    gates = read_gates()
    recipe = decide(gates, extra_per_ligand=args.extra)
    existing = existing_counts(names)
    FINAL.mkdir(parents=True, exist_ok=True)
    estimate = estimate_hours(recipe, len(names), existing)
    from cypmode.validation.series_consensus import official_pairs

    context = ligand_table(official, pd.read_csv("data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"), pd.read_csv("data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"), official_pairs())
    decision = {"gates": gates, "recipe": asdict(recipe), "existing_samples": existing, "estimated_hours": round(estimate, 1), "ligands": json.loads(context.to_json(orient="records"))}
    (FINAL / "recipe.json").write_text(json.dumps(decision, indent=2))
    print("verdicts read:", {k: v for k, v in sorted(gates.items())})
    for reason in recipe.reasons:
        print("  -", reason)
    print(f"estimated compute: {estimate:.1f} h (existing samples counted)")
    print(f"ligands: {int(context.type_ii_donor.sum())} with a Type II donor, {int(context.in_tdi_train.sum())} also in the TDI training set, {int(context.analog.notna().sum())} with an analog among the 20")
    if args.decide_only:
        return

    references = load_reference_set()
    for group in run_groups(recipe, official, existing, references):
        print(f"== run group {group['label']}: {len(group['rows'])} rows")
        execute(group, FINAL / "runs")
    # extra budget: rounds of single samples, to the most disagreeing pools (ei) or to every ligand (uniform)
    from cypmode.validation.pose_pools import pairwise_lddt, pool_disagreement

    names_by_ligand = {n: [n, f"{n}-u", f"{n}-rs", *(f"{n}-seed{k}" for k in range(1, LEVER_RUNS + 1)), *(f"{n}-cf{k}" for k in range(1, LEVER_RUNS + 1)), f"{n}-ss"] for n in names}
    names_by_ligand = {n: [r for r in runs if _has_poses(r)] for n, runs in names_by_ligand.items()}
    donor_of = dict(zip(context.ligand, context.type_ii_donor, strict=True))
    names_by_ligand = {n: run_order(n, runs, bool(donor_of.get(n)), recipe.restraint) for n, runs in names_by_ligand.items()}
    total_extra = recipe.extra_per_ligand * len(names)
    if total_extra:
        allocation = {n: 0 for n in names}
        rounds = max(1, total_extra // 4)
        for round_index in range(rounds):
            share = total_extra // rounds + (1 if round_index < total_extra % rounds else 0)
            if recipe.ei_allocation:
                scores = {n: pool_disagreement(pairwise_lddt([p for _, _, p in pool_for(n, names_by_ligand, ("boltz",))]), list(range(len(pool_for(n, names_by_ligand, ("boltz",)))))) for n in names}
                chosen = sorted(names, key=lambda n: -scores[n])[:share]
            else:
                chosen = names[round_index * share:(round_index + 1) * share] or names[:share]
            rows = pd.DataFrame({"Molecule_Name": [f"{n}-ex{allocation[n] + 1}" for n in chosen], "SMILES": [official.set_index("Molecule_Name").loc[n, "SMILES"] for n in chosen],
                                 "Diffusion_Samples": 1, "Seed": [LEVER_SEED_OFFSET + 31 * round_index + i for i in range(len(chosen))]})
            execute({"label": f"extra_round{round_index}", "rows": rows, "flags": []}, FINAL / "runs")
            for n in chosen:
                allocation[n] += 1
                names_by_ligand[n].append(f"{n}-ex{allocation[n]}")
    engines = ("boltz", "openfold3") if recipe.openfold3 else ("boltz",)
    if recipe.openfold3:
        subprocess.run([sys.executable, "-m", "cypmode.validation.openfold3", "--names", *names, "--samples", str(OPENFOLD3_SAMPLES), "--strict"], check=True)

    from cypmode.validation.heme_frame_prior import (
        HemePrior,
        crystal_descriptors,
        structure_descriptor,
    )
    from cypmode.validation.pose_margins import audit
    from cypmode.validation.structures import load_structure

    smiles = dict(zip(official.Molecule_Name, official.SMILES, strict=True))
    prior_model = None
    partner_pools: dict = {}
    if recipe.selection == "rank_terms" and "series" in recipe.terms:
        from cypmode.validation.series_consensus import official_pairs

        partner_pools = {"pairs": official_pairs(), "names_by_ligand": names_by_ligand, "engines": engines, "smiles": smiles}
    if recipe.selection in ("mbr_plus_prior", "rank_terms") and gates.get("heme_frame_prior"):
        crystal, _, _ = crystal_descriptors(references)
        prior_model = HemePrior(crystal)

    # the scorer's own PoseBusters call on every pose of every pool: the verdict and the margin to its thresholds
    pools = audit(smiles, engines).set_index("cif")
    verdict = {Path(c): (bool(r.passed), float(r.distance_margin)) for c, r in pools.iterrows()}

    rows, details = [], []
    for ligand in names:
        pool = pool_for(ligand, names_by_ligand, engines)

        def valid(path: Path) -> bool:
            return verdict.get(path, (False, float("nan")))[0]

        def margin(path: Path) -> float:
            return verdict.get(path, (False, float("nan")))[1]

        prior = (lambda p: float(prior_model.log_density(structure_descriptor(load_structure(p)))[0])) if prior_model else None
        extras = pose_extras(ligand, smiles[ligand], pool, recipe.terms, partner_pools) if recipe.selection == "rank_terms" else None
        best, ok, _mbr = select_pose(pool, recipe.selection, valid, prior, extras, smiles[ligand], margin)
        run, engine, cif = pool[best]
        rows.append({"compound": ligand, "engine": engine, "chosen_cif": str(cif), "boltz_posebusters_ok": ok if engine == "boltz" else False, "of3_posebusters_ok": ok if engine == "openfold3" else False})
        details.append({"compound": ligand, "pool": len(pool), "valid_in_pool": sum(valid(p) for _, _, p in pool), "run": run, "engine": engine, "chosen_cif": str(cif), "valid": ok, "distance_margin": round(margin(cif), 4)})
    consensus = FINAL / "consensus.csv"
    pd.DataFrame(rows).to_csv(consensus, index=False)
    from cypmode.challenge.structure_submission import build_structure_submission

    summary = build_structure_submission(test_set_path=OFFICIAL, out_dir=FINAL, zip_name="structures_final.zip", consensus_csv=consensus)
    ok, message = portal_validate_structure(summary["zip_path"])
    assert ok, f"final zip rejected by the portal's own check: {message}"
    manifest = {"decision": decision, "zip": str(summary["zip_path"]), "sha256": hashlib.sha256(Path(summary["zip_path"]).read_bytes()).hexdigest(),
                "posebusters_failures": summary["posebusters_failures"], "ligands": details}
    (FINAL / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"wrote {summary['zip_path']} (sha256 {manifest['sha256'][:16]}...); PoseBusters failures: {summary['n_posebusters_failures']}/20")


def _has_poses(run: str) -> bool:
    from cypmode.validation.pose_consensus import boltz_cif_paths

    return bool(boltz_cif_paths(run))


if __name__ == "__main__":
    main()
