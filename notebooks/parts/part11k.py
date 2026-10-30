# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part XI (eleventh part): ligand dynamics in the frozen pocket, the dynophore that the project can afford, and the rule that judged it (11.13)."""

from notebooks.notebook_builder import code, figure, md

DYNAMICS_FACTS = r'''
# The numbers of 11.13: the gate lines of the ligand dynamics (job 286), read from the log that the job left, and the sizes of the system, read from the module.
import pandas as pd

from cypmode.validation import ligand_dynamics as ld

dyn_facts = ld.read_log((ROOT / "data/structure_track/outputs/cyp3a4/logs/ligand_dynamics.log").read_text())
dyn_poses, dyn_ligands = dyn_facts["poses"], dyn_facts["ligands"]
dyn_names = {"static_pharmacophore_model": "the static model of 11.11", "dynamic_pharmacophore": "the dynamic pharmacophore", "stability": "the stability", "dynamic_combined": "the two combined"}
dyn_scores = dyn_facts["scores"]
dyn_met = [dyn_names[n] for n in ld.SCORES if dyn_scores[n]["gate"] == "MET"]
dyn_used = [dyn_names[n] for n in ld.SCORES if dyn_scores[n]["use"] == "MET"]
dyn_table = pd.DataFrame([{"score": dyn_names[n], "mean rho with minus the true RMSD": dyn_scores[n]["rho"], "ligands with a positive rho": f"{dyn_scores[n]['positive']}%", "gate": dyn_scores[n]["gate"] or "information only",
                           "paired test against MBR": dyn_scores[n]["use"] or "not applicable"} for n in dyn_names])
dyn_verdict_sentence = ("None of the three dynamic scores passes the gate: " + ", ".join(f"{dyn_names[n]} {dyn_scores[n]['rho']:+.3f}" for n in ld.SCORES) + f", against {dyn_scores['static_pharmacophore_model']['rho']:+.3f} for the static model, "
                        f"with the bar at {MIN_MEAN_RHO:+.2f}." if not dyn_met else "The gate is met by " + " and ".join(dyn_met) + f"; the static model of the same poses scores {dyn_scores['static_pharmacophore_model']['rho']:+.3f}.")
dyn_decision_sentence = ("No score passes the paired test against MBR, so no candidate D was built and S1 stays the structure entry, as the rule says." if not dyn_used else
                         "The paired test is met by " + " and ".join(dyn_used) + ", so the rule allows candidate D; its construction and its checks are in the paragraph that follows.")
dyn_frame = pd.read_csv(ROOT / ld.POSES, float_precision="round_trip")
dyn_frame = dyn_frame[dyn_frame["regime"] == "novel"]
dyn_rho = pd.DataFrame({n: [ld.pool_rho(pool[n], pool["true_rmsd"]) for _, pool in dyn_frame.groupby("compound", sort=True)] for n in dyn_names})
dyn_between = dyn_rho["static_pharmacophore_model"].corr(dyn_rho["dynamic_pharmacophore"], method="spearman")
dyn_centred = lambda column: dyn_frame[column] - dyn_frame.groupby("compound")[column].transform("mean")
dyn_drift_rho = dyn_centred("drift").corr(dyn_centred("true_rmsd"), method="spearman")
show_table(dyn_table, floatfmt=".2f")
'''

DYNAMICS_CONSTANTS = r'''
# The settings of the experiment, read from the module and from the gate that it reuses, so that no number of the text is typed.
from cypmode.validation import ligand_dynamics as ld
from cypmode.validation.pose_awareness import MIN_FRACTION_POSITIVE, MIN_MEAN_RHO
from cypmode.validation.pose_pharmacophore import NOVEL_SIMILARITY, WILCOXON_P

dyn_step_fs = ld.TIMESTEP_PS * 1000
dyn_time_ps = ld.STEPS * ld.TIMESTEP_PS
dyn_frame_ps = ld.FRAME_EVERY * ld.TIMESTEP_PS
dyn_equilibration_ps = ld.EQUILIBRATION_FRAMES * dyn_frame_ps
dyn_frames_scored = ld.STEPS // ld.FRAME_EVERY - ld.EQUILIBRATION_FRAMES
print(f"{ld.STEPS} steps of {dyn_step_fs:.0f} fs = {dyn_time_ps:.0f} ps at {ld.TEMPERATURE_K:.0f} K, a frame every {dyn_frame_ps:.0f} ps, {dyn_frames_scored} frames scored")
'''

LANGEVIN_TOY = '''
# Langevin dynamics on the simplest system that has an exact answer: one particle of carbon mass in a harmonic well, V = k x^2 / 2 in each of three directions, at the settings of the experiment (300 K, friction 1 per ps,
# time step 2 fs). Statistical mechanics says that at equilibrium every direction has the position variance lv_kT / k and every velocity component the variance lv_kT / m (so a mean kinetic energy of lv_kT / 2 per direction).
# The integrator is written out ("middle" scheme: half kick, half drift, friction and noise, half drift, half kick, as OpenMM's LangevinMiddleIntegrator does) and the same system is run with OpenMM itself.
import json
import subprocess
import textwrap
from pathlib import Path

from cypmode.validation import ligand_dynamics as ld

LV_KB = 0.0019872041  # kcal/mol/K
spring_k, atom_mass, n_particles = 2.0, 12.011, 20000  # kcal/mol/A^2 (the tether of the relaxation, 11.10), amu, independent particles
lv_kT = LV_KB * ld.TEMPERATURE_K
# units: kcal/mol, angstrom, amu; the time unit that makes the equations plain is sqrt(amu A^2 / (kcal/mol)) = 0.04888 ps (the AKMA unit of time, 48.888 fs)
to_ps = 0.04888821
lv_dt, lv_gamma = ld.TIMESTEP_PS / to_ps, ld.FRICTION_PER_PS * to_ps

def langevin_middle(n_steps, lv_dt, lv_gamma, seed=7):
    rng = np.random.default_rng(seed)
    x, v = np.zeros((n_particles, 3)), np.zeros((n_particles, 3))  # a cold start: at rest in the middle of the well
    c1 = np.exp(-lv_gamma * lv_dt)
    c2 = np.sqrt((1 - c1**2) * lv_kT / atom_mass)
    force = -spring_k * x
    for _ in range(n_steps):
        v += 0.5 * lv_dt * force / atom_mass
        x += 0.5 * lv_dt * v
        v = c1 * v + c2 * rng.standard_normal(v.shape)
        x += 0.5 * lv_dt * v
        force = -spring_k * x
        v += 0.5 * lv_dt * force / atom_mass
    return x, v

n_steps_run = 10000  # 20 ps, the length of the experiment
lv_x, lv_v = langevin_middle(n_steps_run, lv_dt, lv_gamma)
var_x, var_v = float(lv_x.var()), float(lv_v.var())
n_samples = lv_x.size
rel_error = float(np.sqrt(2 / n_samples))  # the relative standard error of a variance estimated from n_samples normal values
expected_x, expected_v = lv_kT / spring_k, lv_kT / atom_mass
mean_kinetic = float(0.5 * atom_mass * (lv_v**2).sum(axis=1).mean())

openmm_source = textwrap.dedent(f"""
    import json
    import numpy as np
    import openmm as mm
    from openmm import unit
    n, k, T, lv_gamma, lv_dt, steps = {n_particles}, {spring_k}, {ld.TEMPERATURE_K}, {ld.FRICTION_PER_PS}, {ld.TIMESTEP_PS}, {n_steps_run}
    system = mm.System()
    force = mm.CustomExternalForce("0.5 * k * (x^2 + y^2 + z^2)")
    force.addGlobalParameter("k", k * 4.184 * 100)  # kcal/mol/A^2 in kJ/mol/nm^2
    for i in range(n):
        system.addParticle({atom_mass})
        force.addParticle(i, [])
    system.addForce(force)
    integrator = mm.LangevinMiddleIntegrator(T * unit.kelvin, lv_gamma / unit.picosecond, lv_dt * unit.picoseconds)
    integrator.setRandomNumberSeed({ld.SEED})
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CPU"))
    context.setPositions(np.zeros((n, 3)))
    context.setVelocities(np.zeros((n, 3)))
    integrator.step(steps)
    state = context.getState(getPositions=True, getVelocities=True)
    x = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    v = state.getVelocities(asNumpy=True).value_in_unit(unit.angstrom / unit.picosecond)
    print(json.dumps({{"var_x": float(x.var()), "var_v": float(v.var())}}))
""")
openmm_python = Path.home() / "miniforge3/envs/openmm/bin/python"
lv_done = subprocess.run([str(openmm_python), "-c", openmm_source], capture_output=True, text=True, timeout=600)
assert lv_done.returncode == 0, lv_done.stderr[-400:]
openmm_result = json.loads(lv_done.stdout.strip().splitlines()[-1])
# OpenMM reports velocities in A/ps; the toy works in A per internal time unit
var_v_openmm = openmm_result["var_v"] * to_ps**2
lv_rows = [{"quantity": "position variance per direction (A^2)", "lv_kT / k": expected_x, "this integrator": var_x, "OpenMM": openmm_result["var_x"]},
        {"quantity": "velocity variance per direction (A^2 per unit time^2)", "lv_kT / m": expected_v, "this integrator": var_v, "OpenMM": var_v_openmm}]
dynamics_table = pd.DataFrame(lv_rows)
show_table(dynamics_table, floatfmt=".5f")
lv_checks = {"toy position": var_x / expected_x - 1, "toy velocity": var_v / expected_v - 1, "OpenMM position": openmm_result["var_x"] / expected_x - 1, "OpenMM velocity": var_v_openmm / expected_v - 1}
assert all(abs(e) < 4 * rel_error for e in lv_checks.values()), lv_checks  # all four agree with the exact value within four standard errors
'''

CELLS = [
    md("""
## 11.13 Ligand dynamics in the frozen pocket: a dynophore that the project can afford

> **Field:** Computational chemistry · Statistics

*Where this section sits.* Section 11.11 asked whether a pharmacophore model that is read at one frame of a predicted pose ranks the poses of a ligand by their distance to the crystal pose, and the answer was no. A single frame is a snapshot of a molecule that moves. This section asks whether reading the same model over a short trajectory of the ligand in its pocket changes the answer, and it also tests a quantity that no earlier rule used: how stable a pose stays when it is allowed to move. It teaches the three ideas it needs: molecular dynamics, the dynophore, and what a small test can and cannot show.
"""),
    code(DYNAMICS_CONSTANTS),
    md("""
**Molecular dynamics.** The atoms of a molecule are points with masses, and a **force field** (the same kind of energy function as in 11.10) gives the force on each atom for any arrangement. **Molecular dynamics** moves the atoms by Newton's second law in very small steps: at each step the forces are computed and the positions and velocities are updated. Here a step is {{=dyn_step_fs:.0f}} femtoseconds (a femtosecond is a millionth of a billionth of a second); {{=ld.STEPS:,}} steps make {{=dyn_time_ps:.0f}} picoseconds (ps). To keep the molecule at a temperature, a **Langevin thermostat** adds a friction and random kicks to the atoms, which together make the average kinetic energy that of {{=ld.TEMPERATURE_K:.0f}} K. The result is a **trajectory**, a list of arrangements (frames), here one every {{=dyn_frame_ps:.0f}} ps. The next cell runs the thermostat on a system whose exact answer is known.
"""),
    code(LANGEVIN_TOY),
    md("""
**The thermostat, on a system with an exact answer.** The cell moves particles of carbon mass in a harmonic well, $V = k x^2/2$ in each of three directions, with the spring constant $k$ of the tether of the relaxation (11.10), at the settings of the experiment: {{=ld.TEMPERATURE_K:.0f}} K, a friction of {{=ld.FRICTION_PER_PS:.0f}} per ps and a step of {{=dyn_step_fs:.0f}} fs. Statistical mechanics gives the exact answer. At equilibrium the position along each direction has the variance $k_B T/k$, and each component of the velocity the variance $k_B T/m$, where $k_B$ is the Boltzmann constant, $T$ the temperature and $m$ the mass; that is the statement that every direction holds a kinetic energy of $k_B T/2$ on average. The integrator is written out in the cell: half a kick from the force, half a drift of the position, the friction and the random kick, half a drift, half a kick. That is the "middle" scheme of OpenMM's `LangevinMiddleIntegrator`, which the experiment uses. {{=n_particles:,}} independent particles start at rest in the middle of the well; after {{=n_steps_run:,}} steps (20 ps, the length of the experiment) the position variance is {{=var_x:.4f}} Å² against the exact {{=expected_x:.4f}}, a difference of {{=lv_checks['toy position']:+.1%}}, and the velocity variance differs from the exact value by {{=lv_checks['toy velocity']:+.1%}}. OpenMM's own integrator on the same system differs by {{=lv_checks['OpenMM position']:+.1%}} and {{=lv_checks['OpenMM velocity']:+.1%}}. The statistical error of a variance estimated from {{=n_samples:,}} values is {{=rel_error:.1%}}, so all four numbers agree with the exact value (the cell requires it, within four standard errors). The mean kinetic energy is {{=mean_kinetic:.3f}} kcal/mol, against {{=1.5 * lv_kT:.3f}} for three directions. This is what "the thermostat keeps the temperature at {{=ld.TEMPERATURE_K:.0f}} K" means, and it can be checked because the correct distribution is known.

**The dynophore.** A **pharmacophore** (9.14, 11.11) is the set of chemical features of a ligand that matter for binding, with their positions. A **dynophore** is a pharmacophore that is followed over a molecular dynamics simulation, so that each feature carries the share of the time in which it is present and where it sits. A published account describes the approach as one that "supplements the static structure-based-pharmacophore information with a dynamic component" (Janežič and colleagues, *International Journal of Molecular Sciences* 2021, doi 10.3390/ijms222413474, Appendix C). In that work the simulation is of the whole complex of protein and ligand in water.

**What this project does instead, and why it is not a dynophore.** A simulation of the whole enzyme in water for nanoseconds is out of reach on this machine for the valid poses of the 20 official ligands (the note of the rule gives the measured speed of the system below, for the ligand alone). The project therefore runs *ligand dynamics in a frozen pocket*: the system of the force-field relaxation of 11.10, in which the protein and the heme stay at their predicted positions and only the ligand moves, with two changes written down before the first number (`docs/structure_track/structure_ligand_dynamics.md`): the tether of the ligand to its predicted positions is switched off, because it would give every pose the same stability, and the restraint of a coordinating nitrogen to the iron is kept at its predicted distance, because the force field gives the iron no repulsive term. The thermostat is at {{=ld.TEMPERATURE_K:.0f}} K, the first {{=dyn_equilibration_ps:.0f}} ps are not scored, and the poses are those of the gate of 11.11. The name in this notebook is always *ligand dynamics in a frozen pocket*, for the reasons that follow.

**What it cannot show.** (1) The protein is the predicted one and does not respond, so a pose that needs the pocket to open or a loop to move cannot be recognised. (2) There is no water: a water molecule that bridges the ligand and the protein, and the cost of removing water from the ligand, are absent, and a constant dielectric of 4 is a crude stand-in. (3) {{=dyn_time_ps:.0f}} ps is the time of vibrations and of small rotations; the slow motions of a ligand, a ring that flips or a chain that swings, are not sampled. (4) The force field is the same as in the relaxation, so the stability of a pose measures its agreement with that force field, which the physics terms of 11.8 already did without ranking the poses. (5) The frame scores reuse the fields and the sites of 11.11, so any error of that model is carried into the average. (6) The nitrogen of a Type II ligand is held at the iron by construction, so the anchor of such a pose is stable whether or not it is right.

**The rule, fixed before any number.** The ten ligands are the crystal-anchored benchmark of 11.5, picked for their chemical diversity from the CYPMode PDB entries and not for any property of the dynamics; the table of 11.5 lists them with their PDB entries. The pools are those of the gate of 11.11, the regime in which the references at least {{=NOVEL_SIMILARITY}} similar to the ligand are left out of the pocket fields. Three scores, all within-pool ranks without a fitted weight: the **dynamic pharmacophore** (the ranks of the field score and of the site recall, each averaged over the {{=dyn_frames_scored}} scored frames), the **stability** (minus the mean heavy-atom distance of the frames to the start pose) and the two combined. Scores that are equal in exact arithmetic are ties: the mean of two ranks is often the same number for different poses, so the scores are rounded to {{=ld.TIE_DECIMALS}} decimals before they are ranked, and the rounding noise of the last binary digit cannot order them. The gate is the bar of 11.8 and 11.11: the correlation of a score with minus the true RMSD, averaged over the ligands, exceeds {{=MIN_MEAN_RHO}} and is positive for at least {{=MIN_FRACTION_POSITIVE:.0%}} of them; only then a paired test against the pose that MBR picks decides whether the score may be used (the mean RMSD of its pick is below that of MBR, every drop-one difference is above zero, one-sided paired Wilcoxon p below {{=WILCOXON_P}}). Three scores are judged without a correction among them. The structure entry stays S1 unless a paired test is met, and a candidate that is built is never uploaded to the live board to decide anything.
"""),
    code(DYNAMICS_FACTS),
    md("""
**The result.** The pool is the one of 11.11: {{=dyn_poses}} poses of {{=dyn_ligands}} benchmark ligands, {{=dyn_time_ps:.0f}} ps for each pose. {{=dyn_verdict_sentence}} {{=dyn_decision_sentence}}

*What the number of ligands allows.* {{=dyn_ligands}} ligands cannot detect a modest effect: with the bar at a correlation of {{=MIN_MEAN_RHO}} and a share of {{=MIN_FRACTION_POSITIVE:.0%}}, a score that helps a little will often fail and a score that does nothing will sometimes pass. A pass would therefore be weak evidence and a failure says nothing about an effect that is smaller than the gate can see. The benchmark ligands are also chemically far from the 20 official ligands, so even a clear result on them would not transfer with certainty.
"""),
    figure("struct_dyn_gate"),
    md("""
The figure shows the gate ligand by ligand. A dot to the right of zero means that within the pool of that ligand the score puts the poses that are closer to the crystal pose first; the dashed line is the bar of the gate. The dynamic pharmacophore and the static model of the same poses follow each other from ligand to ligand (Spearman correlation {{=dyn_between:+.2f}} over the {{=dyn_ligands}} ligands, an estimate that so few ligands cannot make precise): where the field ranks the poses of a ligand wrongly at one frame it mostly does so over the whole trajectory. That fits a score that reads the same field at more frames, and it is why the dynamic pharmacophore adds little to the static model.
"""),
    figure("struct_dyn_drift"),
    md("""
The second figure asks the question of the stability directly. After the drift and the error of each pose are centred on the mean of its ligand, their Spearman correlation is {{=dyn_drift_rho:+.2f}} over the {{=dyn_poses}} poses, so in these data a pose that moves little is not a pose that is closer to the crystal pose. One reading is that the force field has a minimum near every pose that the prediction produced, so that the drift measures how steep that minimum is and not where the crystal pose lies; the experiment does not test that reading.
"""),
    figure("struct_dyn_trace"),
    md("""
The third figure shows what the dynamic pharmacophore averages. For the ligand whose poses differ most in their distance to the crystal pose, the left panel is the distance of the ligand to its start pose over the 20 ps and the right panel is the field score of the pose, frame by frame. The grey band is the equilibration that is not scored. The best and the worst pose against the crystal are drawn thick.

*On what actually ran.* `cypmode/validation/ligand_dynamics.py` (`simulate_pose`, `frame_scores`, `within_pool_dynamics`, `evaluate`, `report`) is the experiment; it reuses the force field of `relax.py`, the pools and the gate functions of `pose_pharmacophore.py` and the bar of `pose_awareness.py`. The queue job `286_ligand_dynamics` prepares the pose files, runs the simulation in four processes of two threads in the conda environment `openmm`, and writes the log `ligand_dynamics.log` that this section reads; the figures are drawn by `docs/figures/gen/structure_dynamics.py` and the tests are in `tests/test_ligand_dynamics.py`. The rule is in `docs/structure_track/structure_ligand_dynamics.md`, the method report states the verdicts in section 11, and the limitation L16 says what the experiment cannot show.
"""),
]
