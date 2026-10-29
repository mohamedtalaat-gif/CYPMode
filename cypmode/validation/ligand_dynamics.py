# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Ligand dynamics in the frozen pocket (the rule is in ``docs/structure_ligand_dynamics.md``, committed before any number of the experiment).

The question is whether a pharmacophore read over a short trajectory of the ligand in its pocket, or the stability of a pose under dynamics, ranks the predicted poses of a ligand by their
distance to the crystal pose, which the static pharmacophore model of ``pose_pharmacophore`` did not. This is not a dynophore of the literature: the protein and the heme are frozen at their
predicted positions, there is no water (a constant dielectric of 4 stands for it) and the ligand moves for 20 ps.

Three stages, in two environments:

    python -m cypmode.validation.ligand_dynamics --prepare                                  (main environment: the pose files and the manifest)
    ~/miniforge3/envs/openmm/bin/python -m cypmode.validation.ligand_dynamics --simulate     (conda environment ``openmm``; ``--shard i/n`` splits the manifest over processes)
    python -m cypmode.validation.ligand_dynamics --gate                                      (main environment: the scores, the gate and the figures' table)

The simulation is the system of ``relax.py`` (ligand with OpenFF Sage and NAGL charges, Amber ff14SB protein and CHARMM36 heme frozen, an interaction group without cutoff) with the tether of the
heavy atoms switched off and the restraint of a coordinating nitrogen to the iron kept. Langevin at 300 K, friction 1 per ps, 2 fs, seed 42, at most 500 iterations of minimisation, 10,000 steps, a
frame every 1 ps, the first 2 ps not scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TEMPERATURE_K = 300.0
FRICTION_PER_PS = 1.0
TIMESTEP_PS = 0.002
SEED = 42
MINIMISE_ITERATIONS = 500
STEPS = 10_000
FRAME_EVERY = 500  # steps: one frame every 1 ps
EQUILIBRATION_FRAMES = 2
SCORES = ("dynamic_pharmacophore", "stability", "dynamic_combined")
WORK = Path("data/structure_track/outputs/cyp3a4/ligand_dynamics")
MANIFEST = WORK / "manifest.json"
POSES = Path("data/structure_track/outputs/cyp3a4/ligand_dynamics_poses.csv")
LOG_GATE = "STRUCTURE GATE"


# -- the simulation (conda environment openmm) -------------------------------------------------------------------------------------------------------------------

def simulate_pose(pdb: Path, smiles: str, out: Path, threads: int = 2) -> dict:
    """Run the dynamics of the ligand of ``pdb`` in its frozen pocket and write the frames to ``out`` (an npz file); returns the summary that is also stored next to the frames."""
    import openmm as mm
    from openmm import app, unit

    from cypmode.validation import relax

    structure = app.PDBFile(str(pdb))
    topology, positions = structure.topology, structure.positions
    hemes = [r for r in topology.residues() if r.name == "HEM"]
    if len(hemes) != 1:
        raise ValueError(f"{len(hemes)} heme residues in {pdb}; exactly one is expected")
    heme_atoms = list(hemes[0].atoms())
    heme_xyz = np.array([positions[a.index].value_in_unit(unit.nanometer) for a in heme_atoms])
    parameters = relax.heme_parameters()
    iron = next(i for i, a in enumerate(heme_atoms) if a.name == "FE")

    heavy, full = relax.ligand_molecule(pdb, smiles)
    lig_xyz = np.array(full.GetConformer().GetPositions()) * 0.1
    n_heavy = heavy.GetNumAtoms()
    system = relax.ligand_system(full)
    n_lig = system.getNumParticles()
    q_prot, sigma_prot, eps_prot, xyz_prot = relax.protein_environment(pdb, heme_xyz[iron])
    q_env = np.r_[q_prot, [parameters[a.name][0] for a in heme_atoms]]
    sigma_env = np.r_[sigma_prot, [parameters[a.name][1] for a in heme_atoms]]
    eps_env = np.r_[eps_prot, [parameters[a.name][2] for a in heme_atoms]]
    xyz_env = np.vstack([xyz_prot, heme_xyz])
    n_env = len(q_env)
    fe_particle = n_lig + len(q_prot) + iron

    nonbonded = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    ligand_values = [nonbonded.getParticleParameters(i) for i in range(n_lig)]
    for _ in range(n_env):
        system.addParticle(0.0)
        nonbonded.addParticle(0.0, 1.0, 0.0)
    pocket = mm.CustomNonbondedForce(f"4*epsilon*((sigma/r)^12-(sigma/r)^6)+{relax.COULOMB}*q1*q2/({relax.DIELECTRIC}*r); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)")
    for name in ("q", "sigma", "epsilon"):
        pocket.addPerParticleParameter(name)
    for q, sigma, epsilon in ligand_values:
        pocket.addParticle([q.value_in_unit(unit.elementary_charge), sigma.value_in_unit(unit.nanometer), epsilon.value_in_unit(unit.kilojoule_per_mole)])
    for q, sigma, epsilon in zip(q_env, sigma_env, eps_env, strict=True):
        pocket.addParticle([float(q), float(sigma), float(epsilon)])
    for k in range(nonbonded.getNumExceptions()):
        i, j = nonbonded.getExceptionParameters(k)[:2]
        pocket.addExclusion(i, j)
    pocket.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
    pocket.addInteractionGroup(set(range(n_lig)), set(range(n_lig, n_lig + n_env)))
    pocket.setForceGroup(1)
    system.addForce(pocket)

    nitrogens = [a.GetIdx() for a in heavy.GetAtoms() if a.GetSymbol() == "N"]
    distances = {i: float(np.linalg.norm(lig_xyz[i] - heme_xyz[iron])) * 10.0 for i in nitrogens}
    coordinating = min(distances, key=distances.get) if distances and min(distances.values()) < relax.FE_N_MAX_A else None
    if coordinating is not None:  # the force field gives the iron no Lennard-Jones term: the nitrogen is held at its predicted distance, as in the relaxation
        bond = mm.CustomBondForce("0.5*k*(r-r0)^2")
        bond.addPerBondParameter("k")
        bond.addPerBondParameter("r0")
        bond.addBond(coordinating, fe_particle, [relax.FE_N_K, distances[coordinating] * 0.1])
        bond.setForceGroup(3)
        system.addForce(bond)

    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE_K * unit.kelvin, FRICTION_PER_PS / unit.picosecond, TIMESTEP_PS * unit.picoseconds)
    integrator.setRandomNumberSeed(SEED)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CPU"), {"Threads": str(threads)})
    context.setPositions(np.vstack([lig_xyz, xyz_env]))
    mm.LocalEnergyMinimizer.minimize(context, relax.TOLERANCE, MINIMISE_ITERATIONS)
    context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin, SEED)
    frames, interaction, fe_n = [], [], []
    for _ in range(STEPS // FRAME_EVERY):
        integrator.step(FRAME_EVERY)
        moved = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)[:n_lig]
        frames.append(moved[:n_heavy] * 10.0)
        interaction.append(context.getState(getEnergy=True, groups=1 << 1).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        fe_n.append(float(np.linalg.norm(moved[coordinating] - heme_xyz[iron])) * 10.0 if coordinating is not None else float("nan"))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, frames=np.array(frames), start=lig_xyz[:n_heavy] * 10.0, interaction=np.array(interaction), fe_n=np.array(fe_n), coordinating=coordinating is not None)
    return {"pose": pdb.name, "heavy_atoms": n_heavy, "frames": len(frames), "coordinating": coordinating is not None}


def simulate_all(manifest: Path, shard: tuple[int, int], threads: int) -> None:
    jobs = json.loads(manifest.read_text())
    index, count = shard
    for number, job in enumerate(jobs):
        if number % count != index:
            continue
        out = Path(job["out"])
        if out.exists():
            continue
        summary = simulate_pose(Path(job["pdb"]), job["smiles"], out, threads)
        print(json.dumps(summary), flush=True)


# -- the scores (main environment) --------------------------------------------------------------------------------------------------------------------------

def drift(frames: np.ndarray, start: np.ndarray) -> float:
    """Mean over the frames of the heavy-atom RMSD to the start pose, in angstrom."""
    return float(np.sqrt(((frames - start[None]) ** 2).sum(axis=-1).mean(axis=-1)).mean())


def frame_scores(in_frame: dict, frames: np.ndarray, fields: dict, sites: dict) -> pd.DataFrame:
    """The pose-in-frame scores of every frame of a trajectory (the heme is frozen, so the frame of the heme is the one of the start pose)."""
    from cypmode.challenge import pocket_pharmacophore as pp
    from cypmode.validation.pose_pharmacophore import score_pose

    mol, frame, iron = in_frame["mol"], in_frame["frame"], in_frame["iron"]
    factory = pp.feature_factory()
    rows = []
    for xyz in frames:
        conformer = mol.GetConformer()
        for atom, position in enumerate(xyz):
            conformer.SetAtomPosition(atom, [float(v) for v in position])
        features = pp.features_of(mol, factory)
        in_heme_frame = {family: pp.to_frame(positions, iron, frame) if len(positions) else positions for family, positions in features.items()}
        rows.append(score_pose(pp.to_frame(mol.GetConformer().GetPositions(), iron, frame), in_heme_frame, fields, sites))
    return pd.DataFrame(rows)


def within_pool_dynamics(pool: pd.DataFrame) -> pd.DataFrame:
    """Add the three scores (mean within-pool ranks, higher is better) to a per-pose table with the frame means ``field`` and ``recall`` and the ``drift``."""
    out = pool.copy()
    rank = lambda values: pd.Series(values).rank(pct=True).to_numpy()
    out["dynamic_pharmacophore"] = np.nanmean(np.column_stack([rank(pool["field"]), rank(pool["recall"])]), axis=1)
    out["stability"] = rank(-pool["drift"])
    out["dynamic_combined"] = np.nanmean(np.column_stack([rank(out["dynamic_pharmacophore"]), rank(out["stability"])]), axis=1)
    return out


def evaluate(poses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-pose scores and, per ligand, the Spearman rho of each score with minus the true RMSD and the RMSD of the pose each score picks; ``poses`` has compound, true_rmsd, mbr, field, recall, drift and the
    static ``static_field`` and ``static_recall`` of the start pose."""
    from scipy.stats import spearmanr

    scored, rows = [], []
    for compound, pool in poses.groupby("compound", sort=True):
        s = within_pool_dynamics(pool.reset_index(drop=True))
        s["static_pharmacophore_model"] = np.nanmean(np.column_stack([s["static_field"].rank(pct=True).to_numpy(), s["static_recall"].rank(pct=True).to_numpy()]), axis=1)
        scored.append(s)
        row = {"compound": compound, "poses": len(s), "mean_pool": float(s.true_rmsd.mean()), "mbr_pick": float(s.loc[s.mbr.idxmax(), "true_rmsd"])}
        for name in (*SCORES, "static_pharmacophore_model"):
            values = s[name]
            ok = values.notna()
            row[f"rho_{name}"] = float(spearmanr(values[ok], -s.loc[ok, "true_rmsd"]).statistic) if ok.sum() >= 3 and values[ok].nunique() > 1 else np.nan
            if name in SCORES:
                row[f"{name}_pick"] = float(s.loc[values.idxmax(), "true_rmsd"])
        rows.append(row)
    return pd.concat(scored, ignore_index=True), pd.DataFrame(rows)


def report(per_ligand: pd.DataFrame, regime: str, gated: bool) -> list[str]:
    """The lines of the log for one regime; the lines of the gates only for the gated regime."""
    from cypmode.validation.pose_awareness import MIN_FRACTION_POSITIVE, MIN_MEAN_RHO
    from cypmode.validation.pose_pharmacophore import WILCOXON_P, gate, use_test

    lines = [f"regime {regime}{' (the gate)' if gated else ' (information)'}: {len(per_ligand)} ligands"]
    for name in (*SCORES, "static_pharmacophore_model"):
        g = gate(per_ligand, f"rho_{name}")
        lines.append(f"  {name}: mean rho {g['mean_rho']:+.3f}, positive for {g['fraction_positive']:.0%} of {g['n']} ligands")
        if gated and name in SCORES:
            lines.append(f"{LOG_GATE} dynamics_{name} (mean rho > {MIN_MEAN_RHO} and positive for >= {MIN_FRACTION_POSITIVE:.0%} of ligands): {'MET' if g['met'] else 'NOT MET'}")
            if g["met"]:
                u = use_test(per_ligand, name)
                lines.append(f"  paired test against the pick of MBR: mean gain {u['mean_gain']:+.3f} A, wins {u['wins']}/{u['n']}, worst drop-one {u['worst_drop_one']:+.3f}, one-sided Wilcoxon p = {u['p']:.3f}")
                lines.append(f"{LOG_GATE} dynamics_{name}_use (paired, drop-one robust, p < {WILCOXON_P}): {'MET' if u['met'] else 'NOT MET'}")
            else:
                lines.append(f"{LOG_GATE} dynamics_{name}_use: not evaluated, the gate of the score is not met")
    return lines


# -- the stages ----------------------------------------------------------------------------------------------------------------------------------------------

def prepare(directory: Path) -> list[dict]:
    """Write the poses of the pools of the static gate in the submission format and the manifest of the simulation."""
    from cypmode.challenge.structure_submission import convert_cif_to_pdb
    from cypmode.validation.pose_pharmacophore import benchmark_pools, benchmark_smiles

    smiles = benchmark_smiles()
    jobs = []
    for ligand, poses in benchmark_pools(directory).items():
        for run, sample, _, cif, _ in poses:
            pdb = WORK / "poses" / f"{ligand}_{run}_{sample}.pdb"
            if not pdb.exists():
                pdb.parent.mkdir(parents=True, exist_ok=True)
                convert_cif_to_pdb(Path(cif), pdb)
            jobs.append({"ligand": ligand, "run": run, "sample": sample, "pdb": str(pdb), "smiles": smiles[ligand], "out": str(WORK / "trajectories" / f"{ligand}_{run}_{sample}.npz")})
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(jobs, indent=1))
    return jobs


def score_pools(directory: Path, regime: str) -> pd.DataFrame:
    from cypmode.validation.pose_pharmacophore import (
        benchmark_pools,
        benchmark_smiles,
        fields_and_sites,
        kept_indices,
        pose_in_frame,
        score_pose,
    )
    from cypmode.validation.structures import load_structure

    smiles = benchmark_smiles()
    rows = []
    for ligand, poses in benchmark_pools(directory).items():
        fields, sites = fields_and_sites(kept_indices(smiles[ligand], regime))
        for run, sample, rmsd, cif, mbr in poses:
            trajectory = WORK / "trajectories" / f"{ligand}_{run}_{sample}.npz"
            if not trajectory.exists():
                continue
            in_frame = pose_in_frame(load_structure(cif), smiles[ligand])
            data = np.load(trajectory)
            if in_frame is None or in_frame["mol"].GetNumAtoms() != data["frames"].shape[1]:
                continue
            frames = data["frames"][EQUILIBRATION_FRAMES:]
            per_frame = frame_scores(in_frame, frames, fields, sites)
            static = score_pose(in_frame["heavy"], in_frame["features"], fields, sites)
            rows.append({"compound": ligand, "run": run, "sample": sample, "true_rmsd": rmsd, "mbr": mbr, "regime": regime, "field": float(per_frame["field"].mean()), "recall": float(per_frame["recall"].mean()),
                         "drift": drift(frames, data["start"]), "interaction": float(data["interaction"][EQUILIBRATION_FRAMES:].mean()), "static_field": static["field"], "static_recall": static["recall"]})
    return pd.DataFrame(rows)


def gate_stage(directory: Path) -> None:
    tables, lines = [], []
    for regime in ("novel", "compound"):
        scored, per_ligand = evaluate(score_pools(directory, regime))
        tables.append(scored)
        print(per_ligand.round(3).to_string(index=False), flush=True)
        lines += report(per_ligand, regime, regime == "novel")
    POSES.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(POSES, index=False)
    print(f"{len(tables[0])} poses of {tables[0].compound.nunique()} ligands")
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--shard", default="0/1", help="i/n: this process runs the poses whose number is i modulo n")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--directory", default="data/structure_track/outputs/cyp3a4")
    args = parser.parse_args()
    if args.prepare:
        jobs = prepare(Path(args.directory))
        print(f"{len(jobs)} poses of {len({j['ligand'] for j in jobs})} ligands written to {MANIFEST}")
    elif args.simulate:
        index, count = (int(v) for v in args.shard.split("/"))
        simulate_all(MANIFEST, (index, count), args.threads)
    elif args.gate:
        gate_stage(Path(args.directory))
    else:
        parser.error("give --prepare, --simulate or --gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
