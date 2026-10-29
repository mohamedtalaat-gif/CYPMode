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

CELLS = [
    md("""
## 11.13 Ligand dynamics in the frozen pocket: a dynophore that the project can afford

> **Field:** Computational chemistry · Statistics

*Where this section sits.* Section 11.11 asked whether a pharmacophore model that is read at one frame of a predicted pose ranks the poses of a ligand by their distance to the crystal pose, and the answer was no. A single frame is a snapshot of a molecule that moves. This section asks whether reading the same model over a short trajectory of the ligand in its pocket changes the answer, and it also tests a quantity that no earlier rule used: how stable a pose stays when it is allowed to move. It teaches the three ideas it needs: molecular dynamics, the dynophore, and what a small test can and cannot show.
"""),
    code(DYNAMICS_CONSTANTS),
    md("""
**Molecular dynamics.** The atoms of a molecule are points with masses, and a **force field** (the same kind of energy function as in 11.10) gives the force on each atom for any arrangement. **Molecular dynamics** moves the atoms by Newton's second law in very small steps: at each step the forces are computed and the positions and velocities are updated. Here a step is {{=dyn_step_fs:.0f}} femtoseconds (a femtosecond is a millionth of a billionth of a second); {{=ld.STEPS:,}} steps make {{=dyn_time_ps:.0f}} picoseconds (ps). To keep the molecule at a temperature, a **Langevin thermostat** adds a friction and random kicks to the atoms, which together make the average kinetic energy that of {{=ld.TEMPERATURE_K:.0f}} K. The result is a **trajectory**, a list of arrangements (frames), here one every {{=dyn_frame_ps:.0f}} ps.

**The dynophore.** A **pharmacophore** (9.14, 11.11) is the set of chemical features of a ligand that matter for binding, with their positions. A **dynophore** is a pharmacophore that is followed over a molecular dynamics simulation, so that each feature carries the share of the time in which it is present and where it sits. A published account describes the approach as one that "supplements the static structure-based-pharmacophore information with a dynamic component" (Janežič and colleagues, *International Journal of Molecular Sciences* 2021, doi 10.3390/ijms222413474, Appendix C). In that work the simulation is of the whole complex of protein and ligand in water.

**What this project does instead, and why it is not a dynophore.** A simulation of the whole enzyme in water for nanoseconds is out of reach on this machine for the valid poses of the 20 official ligands (the note of the rule gives the measured speed of the system below, for the ligand alone). The project therefore runs *ligand dynamics in a frozen pocket*: the system of the force-field relaxation of 11.10, in which the protein and the heme stay at their predicted positions and only the ligand moves, with two changes written down before the first number (`docs/structure_track/structure_ligand_dynamics.md`): the tether of the ligand to its predicted positions is switched off, because it would give every pose the same stability, and the restraint of a coordinating nitrogen to the iron is kept at its predicted distance, because the force field gives the iron no repulsive term. The thermostat is at {{=ld.TEMPERATURE_K:.0f}} K, the first {{=dyn_equilibration_ps:.0f}} ps are not scored, and the poses are those of the gate of 11.11. The name in this notebook is always *ligand dynamics in a frozen pocket*, for the reasons that follow.

**What it cannot show.** (1) The protein is the predicted one and does not respond, so a pose that needs the pocket to open or a loop to move cannot be recognised. (2) There is no water: a water molecule that bridges the ligand and the protein, and the cost of removing water from the ligand, are absent, and a constant dielectric of 4 is a crude stand-in. (3) {{=dyn_time_ps:.0f}} ps is the time of vibrations and of small rotations; the slow motions of a ligand, a ring that flips or a chain that swings, are not sampled. (4) The force field is the same as in the relaxation, so the stability of a pose measures its agreement with that force field, which the physics terms of 11.8 already did without ranking the poses. (5) The frame scores reuse the fields and the sites of 11.11, so any error of that model is carried into the average. (6) The nitrogen of a Type II ligand is held at the iron by construction, so the anchor of such a pose is stable whether or not it is right.

**The rule, fixed before any number.** The pools are those of the gate of 11.11, the regime in which the references at least {{=NOVEL_SIMILARITY}} similar to the ligand are left out of the pocket fields. Three scores, all within-pool ranks without a fitted weight: the **dynamic pharmacophore** (the ranks of the field score and of the site recall, each averaged over the {{=dyn_frames_scored}} scored frames), the **stability** (minus the mean heavy-atom distance of the frames to the start pose) and the two combined. Scores that are equal in exact arithmetic are ties: the mean of two ranks is often the same number for different poses, so the scores are rounded to {{=ld.TIE_DECIMALS}} decimals before they are ranked, and the rounding noise of the last binary digit cannot order them. The gate is the bar of 11.8 and 11.11: the correlation of a score with minus the true RMSD, averaged over the ligands, exceeds {{=MIN_MEAN_RHO}} and is positive for at least {{=MIN_FRACTION_POSITIVE:.0%}} of them; only then a paired test against the pose that MBR picks decides whether the score may be used (the mean RMSD of its pick is below that of MBR, every drop-one difference is above zero, one-sided paired Wilcoxon p below {{=WILCOXON_P}}). Three scores are judged without a correction among them. The structure entry stays S1 unless a paired test is met, and a candidate that is built is never uploaded to the live board to decide anything.
"""),
    code(DYNAMICS_FACTS),
    md("""
**The result.** The pool is the one of 11.11: {{=dyn_poses}} poses of {{=dyn_ligands}} benchmark ligands, {{=dyn_time_ps:.0f}} ps for each pose. {{=dyn_verdict_sentence}} {{=dyn_decision_sentence}}

*What the number of ligands allows.* {{=dyn_ligands}} ligands cannot detect a modest effect: with the bar at a correlation of {{=MIN_MEAN_RHO}} and a share of {{=MIN_FRACTION_POSITIVE:.0%}}, a score that helps a little will often fail and a score that does nothing will sometimes pass. A pass would therefore be weak evidence and a failure says nothing about an effect that is smaller than the gate can see. The benchmark ligands are also chemically far from the 20 official ligands, so even a clear result on them would not transfer with certainty.
"""),
    figure("struct_dyn_gate"),
    md("""
The figure shows the gate ligand by ligand. A dot to the right of zero means that within the pool of that ligand the score puts the poses that are closer to the crystal pose first; the dashed line is the bar of the gate. The dynamic scores and the static model do not move together from ligand to ligand, which is what a gate on so few ligands has to live with.
"""),
    figure("struct_dyn_drift"),
    md("""
The second figure asks the question of the stability directly. A pose that stays where it was put is not a pose that is close to the crystal: the force field has a minimum near every pose that the prediction produced, and the drift measures how steep that minimum is, not where the crystal pose is.
"""),
    figure("struct_dyn_trace"),
    md("""
The third figure shows what the dynamic pharmacophore averages. For the ligand whose poses differ most in their distance to the crystal pose, the left panel is the distance of the ligand to its start pose over the 20 ps and the right panel is the field score of the pose, frame by frame. The grey band is the equilibration that is not scored. The best and the worst pose against the crystal are drawn thick.

*On what actually ran.* `cypmode/validation/ligand_dynamics.py` (`simulate_pose`, `frame_scores`, `within_pool_dynamics`, `evaluate`, `report`) is the experiment; it reuses the force field of `relax.py`, the pools and the gate functions of `pose_pharmacophore.py` and the bar of `pose_awareness.py`. The queue job `286_ligand_dynamics` prepares the pose files, runs the simulation in four processes of two threads in the conda environment `openmm`, and writes the log `ligand_dynamics.log` that this section reads; the figures are drawn by `docs/figures/gen/structure_dynamics.py` and the tests are in `tests/test_ligand_dynamics.py`. The rule is in `docs/structure_track/structure_ligand_dynamics.md`, the method report states the verdicts in section 11, and the limitation L16 says what the experiment cannot show.
"""),
]
