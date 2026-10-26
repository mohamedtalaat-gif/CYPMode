# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part XI (ninth part): the pharmacophore model of a predicted pose, the check of the conformational bias, the check of the binding-site representation and the expected pharmacophore of the 20 official ligands (11.11)."""

from notebooks.notebook_builder import code, figure, md

S8_FACTS = r'''
# The numbers of 11.11: the gate lines of the pharmacophore model of the poses (job 272), the coverage of the bound conformation by the ensembles (job 273) and the checks of the pocket and of the support (job 274),
# every one read from the logs that those jobs left.
import re

import pandas as pd

s8_logs = ROOT / "data/structure_track/outputs/cyp3a4/logs"
gate_text = (s8_logs / "pose_pharmacophore_model.log").read_text()
novel_text = gate_text.split("regime compound")[0]
score_line = re.compile(r"^(pharmacophore_model_combined|pharmacophore_model|field|recall|precision|outside): mean Spearman rho ([+-][\d.]+), positive for (\d+)% of (\d+) ligands(?:, mean RMSD of the top-scored pose ([\d.]+) \(MBR ([\d.]+), random pose ([\d.]+)\))?", re.MULTILINE)
s8_scores = {name: {"rho": float(rho), "positive": int(positive), "n": int(n), "pick": float(pick) if pick else None, "mbr": float(mbr) if mbr else None, "random": float(rand) if rand else None} for name, rho, positive, n, pick, mbr, rand in score_line.findall(novel_text)}
s8_gates = dict(re.findall(r"STRUCTURE GATE pose_(pharmacophore_model(?:_combined)?) \(.*?\): (MET|NOT MET)", novel_text))
s8_crystal = re.search(r"information \(not a gate\), regime novel: .*?\n  field: (-?[\d.]+) / (-?[\d.]+) / (-?[\d.]+); the crystal pose scores above the mean of its pool for (\d+) of (\d+) ligands\n  recall: (-?[\d.]+) / (-?[\d.]+) / (-?[\d.]+); the crystal pose scores above the mean of its pool for (\d+) of (\d+) ligands", gate_text)
crystal_field, pool_field, best_field, crystal_field_above, crystal_n, crystal_recall, pool_recall, best_recall, crystal_recall_above = (float(s8_crystal.group(i)) for i in (1, 2, 3, 4, 5, 6, 7, 8, 9))
s8_pose_count = re.search(r"(\d+) poses of (\d+) ligands", gate_text)
n_gate_poses, n_gate_ligands = int(s8_pose_count.group(1)), int(s8_pose_count.group(2))
gate_met = [name for name, verdict in s8_gates.items() if verdict == "MET"]

conformer_text = (s8_logs / "conformer_bias.log").read_text()
c_evaluated = re.search(r"crystal ligand copies evaluated: (\d+) of (\d+) \((\d+) distinct ligands\)", conformer_text)
c_cover = re.search(r"a conformer within ([\d.]+) A of the crystal pose for (\d+)% of the copies; median nearest RMSD ([\d.]+) A", conformer_text)
c_windows = {int(w): (int(k), int(n), int(p)) for w, k, n, p in re.findall(r"within (\d+) kcal/mol of the lowest conformer for (\d+) of (\d+) covered copies \((\d+)%\)", conformer_text)}
c_block = re.search(r"a member within ([\d.]+) A of the crystal pose for (\d+)% of the (\d+) copies; median nearest RMSD ([\d.]+) A", conformer_text)
c_needed = re.search(r"median energy that the bound state needs ([\d.]+) kcal/mol, 90th percentile ([\d.]+) kcal/mol", conformer_text)
c_verdict = re.search(r"CONFORMER CHECK block_ensemble_covers_the_bound_state .*: (YES|NO)", conformer_text).group(1)
c_poses = re.search(r"farther than 1.0 A for (\d+)%, farther than 2.0 A for (\d+)%", conformer_text)
n_copies, n_copies_all, n_ligands_cb = (int(v) for v in c_evaluated.groups())
bias_table = pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/conformer_bias.csv")
bias_done = bias_table[bias_table.nearest_rmsd.notna()]
size_covered, size_missed = float(bias_done[bias_done.nearest_rmsd <= 1.0].n_heavy.median()), float(bias_done[bias_done.nearest_rmsd > 1.0].n_heavy.median())
share_300, median_rmsd_300 = int(c_cover.group(2)), float(c_cover.group(3))
share_block, median_rmsd_block = int(c_block.group(2)), float(c_block.group(4))
needed_median, needed_p90 = float(c_needed.group(1)), float(c_needed.group(2))
poses_beyond_1, poses_beyond_2 = (int(c_poses.group(1)), int(c_poses.group(2))) if c_poses else (None, None)

site_text = (s8_logs / "site_representation.log").read_text()
v_site = re.search(r"(\d+) crystal copies, (\d+) core residues .*? median ([\d.]+) A, 90th percentile ([\d.]+) A, 95th percentile ([\d.]+) A", site_text)
v_pred = re.search(r"predicted pockets: (\d+) of (\d+) poses compared; .*? median ([\d.]+) A, (\d+)% beyond the flag; Spearman rho with the true RMSD of the pose ([+-][\d.]+) over all poses", site_text)
v_support = re.search(r"crystal copies \(fields built without their compound\) median ([\d.]+), 95th percentile ([\d.]+)", site_text)
v_beyond = re.search(r"predicted poses beyond the cut: (\d+)% of (\d+); Spearman rho of the outside share with the true RMSD ([+-][\d.]+) over all poses", site_text)
n_core = int(v_site.group(2))
nn_median, nn_p90, pocket_flag = (float(v_site.group(i)) for i in (3, 4, 5))
pred_median, pred_beyond_share, pred_rho = float(v_pred.group(3)), int(v_pred.group(4)), float(v_pred.group(5))
support_median, support_cut = float(v_support.group(1)), float(v_support.group(2))
support_beyond_share, support_rho = int(v_beyond.group(1)), float(v_beyond.group(3))

s8_table = pd.DataFrame([{"score": name, "mean rho with minus the true RMSD": values["rho"], "ligands with a positive rho": f"{values['positive']}%", "RMSD of the top-scored pose": values["pick"] if values["pick"] is not None else float("nan"),
                          "gate": s8_gates.get(name, "information")} for name, values in s8_scores.items()])
mbr_pick = next(v["mbr"] for v in s8_scores.values() if v["mbr"] is not None)
random_pick = next(v["random"] for v in s8_scores.values() if v["random"] is not None)
s8_table.round(3)
'''

OFFICIAL_FACTS = r'''
# The expected pharmacophore of the 20 official ligands (job 925; HELD BACK: it describes predicted poses of blinded ligands), read from the folder that the job wrote.
import pandas as pd

official_dir = ROOT / "data/structure_track/outputs/cyp3a4/pharmacophore_official20"
official = pd.read_csv(official_dir / "summary.csv")
common_model = pd.read_csv(official_dir / "common_sites.csv")
receptor = pd.read_csv(official_dir / "receptor_side.csv")
n_official = len(official)
n_common = len(common_model)
n_flag_pocket, n_flag_support = int(official.beyond_pocket_flag.sum()), int(official.beyond_support_cut.sum())
n_outside_ensemble, n_strained = int(official.outside_ensemble.sum()), int(official.strained.sum())
n_unsatisfied = int((receptor.unsatisfied.astype(bool)).sum())
donor_ligands = official[official.donor_error.notna()]
median_donor_error = float(donor_ligands.donor_error.median()) if len(donor_ligands) else float("nan")
median_recall, median_pool_spread = float(official.recall.median()), float(official.pool_spread.median())
show_table(official[["ligand", "features", "donor_error", "site_error", "recall", "matched", "precision", "pool_recall_mean", "pool_spread", "conf_rmsd", "pocket_rmsd", "unsatisfied_polar"]].round(2), floatfmt=".2f")
'''

CELLS = [
    md("""
## 11.11 Pharmacophore models of poses, and two biases that such models carry

> **Field:** Computational chemistry · Structural biology · Cheminformatics · Statistics

*Where this section sits:* 9.13 and 9.14 built, for the time-dependent inhibition track, fields and sites that say where the crystal ligands of CYP3A4 place their donors, acceptors, rings and hydrophobic groups, in the frame of the heme. Section 11.8 asked how to choose among the predicted poses of a ligand and tried a pharmacophore term of two distances written by hand. This section asks the same question with a pharmacophore *model*: a pose is put in the frame of its own heme and judged by where its features and atoms lie against what the crystals show. It also reports two checks of what such a model cannot see, the two biases named at the start of the work on it: that an ensemble of low-energy conformers over-represents low-energy states and can miss the bound one, and that a model read from complexes represents a holo pocket and says nothing of an apo or a cryptic one.

**The pose in the frame of its heme.** Every predicted complex has its own heme, and the frame of 9.13 (the iron at the origin, $z$ along the normal of the porphyrin plane on the side of the ligand, $x$ toward the pyrrole nitrogen NA) is found in it from the coordinates of the four nitrogens. The typed features of the ligand are taken at the coordinates of the pose, not of a conformer, and its heavy atoms are scored with the same field and the same sites as in 9.13 and 9.14: the **field score** (the mean occupancy at the heavy atoms plus the mean, over the families the ligand has, of the mean family field at its features), the **recall** (the weight of the sites that a feature of the ligand matches over the weight of all sites), the number of matched sites and the **precision** (the share of the typed features that lie in a sphere of a site of their own family). The fields and the sites are rebuilt for each scored ligand *without its own references*: the crystal ligands that are the same compound and, in the regime that stands for a novel scaffold and carries the gate, also every crystal ligand whose Morgan Tanimoto similarity to it is at least 0.5, the rule that was used for the shape prior of the same benchmark. Without that step a pose of ketoconazole would be scored against the crystal ligand ketoconazole, and the gate would read the answer from the key.

The score of a pose is a rank inside the pool of the poses of its ligand, with no weight fitted: the mean of the ranks of its field score and its recall (`pharmacophore_model`), and the mean of that rank and the rank of the minimum-Bayes-risk consensus of 11.8 (`pharmacophore_model_combined`). The gate is the bar of 11.8: over the pooled benchmark poses the **Spearman correlation** (the correlation of the ranks) between a score and minus the true RMSD, averaged over the ligands, exceeds 0.3 and is positive for at least 60% of the ligands. If a gate is met, whether the recipe may use the score is a second, paired test against the minimum-Bayes-risk pick, and only then.
"""),
    code(S8_FACTS),
    md("""
The pool is the one of 11.8: {{=n_gate_poses}} poses of {{=n_gate_ligands}} benchmark ligands, with the true RMSD of each. The table gives, for each score, the mean correlation and the share of ligands in which it is positive; the pick of `pharmacophore_model` has a mean RMSD of {{=s8_scores['pharmacophore_model']['pick']:.2f}} Å and that of `pharmacophore_model_combined` of {{=s8_scores['pharmacophore_model_combined']['pick']:.2f}} Å, against {{=mbr_pick:.2f}} Å for the minimum-Bayes-risk pick and {{=random_pick:.2f}} Å for a random pose. The gate is met for **{{=', '.join(gate_met) if gate_met else 'neither score'}}**. {{=('A score that fails its gate is written down as a negative result and is not tuned: the score was not changed after its first run.' if not gate_met else 'A met gate is only the first of two conditions: the use of the score in the recipe has to pass the paired test that was declared in advance, and that verdict is read from the log by the final recipe, not from this sentence.')}}

*A model that cannot tell the bound pose from a predicted one cannot rank poses.* For the benchmark ligands whose crystal entries are in the reference set, the crystal pose was scored in the same way, with the fields built without its own compound. The field score of the crystal pose is {{=crystal_field:.2f}} on average, against {{=pool_field:.2f}} for the poses of the pool and {{=best_field:.2f}} for the predicted pose closest to the crystal; its recall is {{=crystal_recall:.2f}} against {{=pool_recall:.2f}} and {{=best_recall:.2f}}. The crystal pose scores above the mean of its pool for {{=int(crystal_field_above)}} of {{=int(crystal_n)}} ligands by field score and for {{=int(crystal_recall_above)}} of them by recall. A pose that lies in the right region (every predicted pose is close to the heme, as every crystal pose is) fills the same fields as the true pose does; the fields are broad, they were built to say where ligands sit and not how, and at that resolution a pool of poses that differ by a few ångströms cannot be told apart.
"""),
    md("""
**First bias: the conformers that an ensemble holds.** A **conformer** is one three-dimensional shape of a molecule, and a flexible molecule has very many. An ensemble generator (ETKDG here, which draws shapes from the distances and torsion angles seen in crystals, then relaxes each with the force field MMFF) keeps those of low energy, since a molecule in solution is mostly in them. The ensembles of 9.13 and 9.14 hold up to twelve conformers within 5 kcal/mol of the lowest. But a ligand in a pocket is held by the protein, and the shape that fits can cost energy: the bound conformer is often not the lowest. A model that places and describes only low-energy shapes then over-represents them and can miss the one that matters. The check measures it directly on the crystal ligands, whose bound shapes are known. For each of {{=n_copies}} crystal ligand copies ({{=n_ligands_cb}} distinct ligands; {{=n_copies_all - n_copies}} could not be treated) 300 conformers were generated, the heavy-atom RMSD (symmetry aware) of the nearest to the crystal pose was found, and its energy above the lowest. A conformer within 1.0 Å of the crystal pose exists among the 300 for {{=share_300}}% of the copies (the median nearest RMSD is {{=median_rmsd_300:.2f}} Å); among those, the bound state lies within 5 kcal/mol of the lowest conformer for {{=c_windows[5][2]}}%, within 10 for {{=c_windows[10][2]}}% and within 20 for {{=c_windows[20][2]}}%, and the median energy that it needs is {{=needed_median:.1f}} kcal/mol (90th percentile {{=needed_p90:.1f}}). The ensemble of the block itself, twelve conformers within 5 kcal/mol, contains a member within 1.0 Å of the crystal pose for {{=share_block}}% of the copies (median nearest RMSD {{=median_rmsd_block:.2f}} Å). The registered question is whether at least 80% are covered, and the answer is **{{=c_verdict}}**{{=('. The verdicts of the pocket block and of the site model, in 9.13 and 9.14, were therefore reached with an ensemble that misses the bound conformation of ' + str(100 - share_block) + '% of the crystal ligands, and they are reported with that sentence.' if c_verdict == 'NO' else '.')}} The copies that are not covered by the 300 conformers have a median of {{=size_missed:.0f}} heavy atoms against {{=size_covered:.0f}} for those that are: a whole-molecule RMSD of 1 Å between a vacuum conformer and a bound pose is a demanding test for a large flexible molecule, and the check measures how often the bound state is reached, not whether the ensemble is good for the small ones. For the {{=n_gate_poses}} predicted poses of the gate, the nearest conformer of the ensemble of 11.8 (within 8 kcal/mol) is {{=poses_beyond_1}}% farther than 1.0 Å and {{=poses_beyond_2}}% farther than 2.0 Å: a predicted pose often resembles no conformer of the low-energy ensemble, which is what a bound shape that costs energy looks like from the side of the ensemble. For that reason the pharmacophore of a predicted pose is reported for the pool of its poses and not only for the top one.
"""),
    figure("struct_conformer_window"),
    md("""
**Second bias: a pocket that the references do not show.** The fields and the sites are **holo**: they are read from ligands in complexes, and a holo model says where ligands sit, not what the pocket looks like before a ligand arrives (the **apo** state) or in a state that no crystal shows (a **cryptic pocket**, which opens only on binding, or for ligands of a kind not yet crystallised). CYP3A4 is a flexible enzyme, so the question is real. Three measurements, none a gate, give the size of the problem as far as the data allow. The variability of the site: the pocket C-alpha atoms of the {{=n_core}} residues that at least 90% of the crystal copies share were superposed by the Kabsch method (the rigid motion that minimises the RMSD) and each copy compared with the nearest copy of another PDB entry and another ligand; the median is {{=nn_median:.2f}} Å, the 90th percentile {{=nn_p90:.2f}} Å and the 95th percentile {{=pocket_flag:.2f}} Å, which is the flag of an unseen pocket conformation. The pockets of the {{=n_gate_poses}} predicted complexes lie {{=pred_median:.2f}} Å from the nearest crystal pocket at the median (their own compound left out), {{=pred_beyond_share}}% of them beyond the flag, and their distance correlates with the true RMSD of the pose with a Spearman correlation of {{=pred_rho:+.2f}}. The support of the fields: for the crystal poses the share of heavy atoms where the occupancy field built without their compound is below 0.05 is {{=support_median:.3f}} at the median and {{=support_cut:.3f}} at the 95th percentile, the cut at which a pose leaves the region where ligands have been seen; {{=support_beyond_share}}% of the predicted poses are beyond it, and the share of atoms outside the support correlates with the true RMSD with {{=support_rho:+.2f}}.

**What these measurements do and do not say.** {{=('They say that, by their backbone, the predicted pockets are inside the range that the crystals span: the median distance of a predicted pocket to the nearest crystal pocket (' + format(pred_median, '.2f') + ' Å) is of the size of the spread among the crystals themselves (' + format(nn_median, '.2f') + ' Å).' if pred_beyond_share == 0 else 'They say that, by their backbone, ' + str(pred_beyond_share) + '% of the predicted pockets lie beyond the range that the crystals span, which marks those complexes for care.')}} They cannot say more. The C-alpha atoms of the core residues are a coarse description of a pocket, since what opens a cryptic pocket is a side chain turning or a loop moving, and the residues that move are exactly those that fewer than 90% of the entries share and that are left out of the core. A pocket that Boltz-2 shapes around a ligand can be plausible or spurious, and the predicted complex alone cannot say which. A cryptic pocket that no entry shows is, by construction, outside what fields read from 113 entries of one enzyme can represent. Both flags are therefore carried by every official pose as a mark to read it with care, and nothing is corrected by them.
"""),
    figure("struct_site_check"),
    md("""
**The expected pharmacophore of the 20 official ligands.** For each official ligand the same machinery gives what its final submitted pose offers to the pocket, and what its pool of valid poses says about how certain that is: the typed features with their coordinates in the frame of the heme; the distance of the Type II donor nitrogen to the iron or of the most easily oxidised carbon; the site model of the pocket against the pose (field score, recall, matched sites, precision), with the references of the ligand and of its analogues left out; for each site matched by the final pose the share of the pool's poses that match it, and the spread of the feature positions across the pool; the receptor side of the final complex (for each donor and acceptor the nearest polar protein atom within 3.5 Å or the iron within 2.6 Å, for each ring and hydrophobic group the residues within 4.5 Å, and a flag for a buried polar feature without a partner); a common-feature model of the 20 poses, the sites at which at least 14 of them place a feature of the same family within 1.5 Å; and the two flags of the second bias. These are predictions for the blinded ligands, and they are held back until the challenge has closed: the folder `pharmacophore_official20` and the figure that shows it are not published before then, and the cells of this section that read them are marked so that the public edition of the notebook shows only a note in their place.
""", held_back=True),
    code(OFFICIAL_FACTS, held_back=True),
    md("""
For the {{=n_official}} official poses the median recall of the sites is {{=median_recall:.2f}}; the pharmacophore moves by {{=median_pool_spread:.2f}} Å at the median across the valid poses of the pool; the common-feature model has {{=n_common}} sites; the Type II donors lie within {{=median_donor_error:.2f}} Å of the ideal distance to the iron at the median. Of the 20 poses {{=n_flag_pocket}} lie beyond the pocket flag, {{=n_flag_support}} beyond the support cut, {{=n_outside_ensemble}} are farther than 1.0 Å from every conformer of the low-energy ensemble and {{=n_strained}} are strained by the registered criterion (an energy of the nearest conformer more than 10 kcal/mol above the lowest, which the ensemble of 8 kcal/mol cannot show; the distance to the ensemble is the usable measure); the receptor side lists {{=n_unsatisfied}} buried polar features without a partner. The model of 11.11 was not used to choose any of these poses (its gate decides that, above); this is a description of the chosen poses, with the uncertainty that the pool and the flags give, and not a measure of their accuracy, which only the organisers can make.
""", held_back=True),
    figure("struct_pharm_official20", held_back=True),
    md("""
*On what actually ran.* `cypmode/validation/pose_pharmacophore.py` (`pose_in_frame`, `score_pose`, `kept_indices`, `fields_and_sites`, `evaluate`, `gate`, `use_test`, `crystal_scores`) is the model and its gate, run by the job `272_pose_pharmacophore_gate`; `cypmode/validation/conformer_bias.py` is the first check, run by `273_conformer_bias`; `cypmode/validation/site_representation.py` is the second, run by `274_site_representation`, and it writes the two flags; `cypmode/validation/official_pharmacophore.py` describes the 20 official poses in the final chain, in the job `925_official_pharmacophore`. The frame of a heme from its four nitrogens is `frame_from_nitrogens` of `cypmode/challenge/pocket_pharmacophore.py`. The tests are `tests/test_pose_pharmacophore.py` (the gate and its rule on poses that track the true RMSD and on noise, the leave-out of the scored ligand and its analogues, the crystal pose), `tests/test_conformer_bias.py` (a crystal pose that is the lowest or a strained conformer, the consequence rule), `tests/test_site_representation.py` (the Kabsch RMSD, the core residues, the nearest other entry) and `tests/test_official_pharmacophore.py` (the common sites, the similarity, the receptor side), and the figures are tested on synthetic files in `tests/test_structure_pharmacophore_figures.py`. The conformer ensembles of this part and of 9.13 and 9.14 are drawn with the seed 42: in the installed version of RDKit the seed 0 returns the same conformer for every draw, which turned the first run of the pocket block into a run with one conformer per compound, was found while this check was being prepared, and is why that first run is kept in the logs as a record and not as a verdict. The limits are written next to each result; the largest is that the model describes where a pose lies against ligands that were seen, so that it can reward a pose for being like the references and cannot reward it for being right where they are silent.
"""),
]
