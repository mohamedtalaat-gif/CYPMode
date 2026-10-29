# Structure track: ligand dynamics in the frozen pocket, and the rule that is fixed before any number

This note is dated by the commit that adds it, which precedes every number of the experiment (2026-10-29).

**The question.** The static pharmacophore model of a pose (`cypmode/validation/pose_pharmacophore.py`) did not rank the predicted poses of the benchmark ligands (`pose_pharmacophore_model`: NOT MET). A pharmacophore that is read over a short trajectory of the ligand in its pocket, instead of at one frame, could carry a signal that the single frame lacks, and the stability of a pose under dynamics is a second quantity that no earlier rule used. The experiment asks whether either of them ranks the predicted poses of a ligand by their distance to the crystal pose.

**What it is not.** A dynophore of the literature is built from a molecular dynamics simulation of the whole complex in water, with a flexible protein, over nanoseconds. Nothing of that is done here. The protein and the heme are frozen at their predicted positions, there is no water (a constant dielectric of 4 stands for it), the ligand moves for 20 ps, and the force field is the one of the relaxation. The name used everywhere is therefore *ligand dynamics in a frozen pocket*. It cannot show that a pose is unstable because the pocket would open, or that a water molecule would bridge it.

**The system.** The one of `cypmode/validation/relax.py` (OpenMM on the CPU, ligand with OpenFF Sage and NAGL charges, protein with Amber ff14SB, heme with the CHARMM36 template, an interaction group of the ligand and the frozen environment without cutoff). Two changes: the tether of the heavy atoms to their predicted positions is switched off, because it would give every pose the same stability, and the harmonic restraint of a coordinating nitrogen to the iron is kept at its predicted distance, because the force field gives the iron no Lennard-Jones term. Langevin integrator at 300 K, friction 1 per ps, time step 2 fs, seed 42, a minimisation of at most 500 iterations first, 10,000 steps (20 ps), one frame every 1 ps; the first 2 ps are equilibration and are not scored. Measured speed on this machine: 101 steps per second with 8 threads (about 17 ns per day), so one pose takes about 100 s.

**The poses.** The pools of the gate of `pose_pharmacophore`: the pooled Boltz-2 poses of the benchmark ligands that have at least 3 poses and a crystal reference (104 poses of 10 ligands at the time of the static gate). The regime `novel` (the references that are at least 0.5 similar to the ligand are left out of the pocket fields) is the gated regime and the regime `compound` is information, as in the static gate.

**The scores** (mean within-pool ranks, no fitted weight, higher is better):

1. `dynamic_pharmacophore`: the ranks of the mean field score and of the mean site recall of the frames, the scores of `pose_pharmacophore.score_pose` averaged over the 18 frames; it is the dynamic counterpart of `pharmacophore_model`.
2. `stability`: the rank of minus the mean heavy-atom RMSD of the frames to the start pose.
3. `dynamic_combined`: the mean of the two ranks.

The score of the static model of the same poses is printed beside them as information, so that the effect of the dynamics can be read.

**The gate.** The bar of the static gate and of `pose_awareness`, unchanged: over the pooled poses of a ligand the Spearman correlation of a score with minus the true RMSD, averaged over the ligands, exceeds 0.3 and is positive for at least 60% of the ligands. Only when it is met the paired test decides whether the score may be used: the mean RMSD of the pose that the score picks is below that of the pose that MBR picks, every drop-one difference is above zero and the one-sided paired Wilcoxon p is below 0.10. Three scores are judged and no correction is applied among them; this is stated here so that a pass of one of them is read with it.

**Power.** Ten ligands cannot detect a modest effect. A pass would be weak evidence, and a failure would say nothing about an effect smaller than the gate can see. The benchmark ligands are also chemically far from the 20 official ligands (the best similarity of an official ligand to the reference set is 0.12 to 0.34).

**The decision.** The structure entry stays S1 (`docs/structure_track/structure_candidate_c.md`) unless the paired test of a score is met in the gated regime. If it is, candidate D is built: for every ligand the eligible pose (the 22 checks of the scorer's PoseBusters call, the stereochemistry of the official SMILES, the margin floor 0.8) with the highest registered score, a ligand with fewer than 3 eligible poses keeps the pose of S1, followed by the registered relaxation. D replaces S1 only if all 20 poses pass the checks. D is never uploaded to the live board to decide anything.

**The time limit.** The benchmark result is read by 2026-10-31 12:00 UTC. The official pools are simulated only if the gate of a score is met. What is not finished by 2026-11-01 00:00 UTC is dropped and S1 stays.

**What is written whatever the result.** The log, a section of the notebook, a paragraph of the method report, a limitation, the figures of the experiment and the entry of the innovation summary.
