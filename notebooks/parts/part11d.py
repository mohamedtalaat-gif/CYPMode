# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part XI (fourth part): the iron-nitrogen restraint and its confirmation (11.6)."""

from notebooks.notebook_builder import code, md

RESTRAINT_BENCHMARK = '''
# The restraint on the benchmark. The six benchmark ligands that have a Type II donor were run three ways with Boltz-2 (three samples each): without a restraint, with the donor nitrogen bonded to the iron, and with a contact
# between them (11.3). The decision rule was written beforehand (cypmode/validation/structure_gates.py): the restraint is kept if the mean RMSD of the first sample to the crystal pose is lower than without it AND the share of first samples
# that pass PoseBusters is not lower. The first-sample RMSD of each ligand is read from the pose tables of the three conditions, and the validity shares from the log of the gate (job 165_structure_gate_verdicts).
import re

from cypmode.validation.restraint_expansion import paired_statistics
from cypmode.validation.structure_benchmark_report import variant_comparison

restraint_tables = {name: pd.read_csv(ROOT / f"data/structure_track/outputs/cyp3a4/benchmark_engine_pose_validation_{stem}.csv") for name, stem in (("unguided", "unguided_boltz"), ("febond", "febond"), ("fecontact", "fecontact"))}
comparison = variant_comparison({"unguided": restraint_tables["unguided"], "febond": restraint_tables["febond"], "fecontact": restraint_tables["fecontact"]}, samples=3)
first_pick = comparison.pivot(index="compound", columns="variant", values="first_pick")[["unguided", "febond", "fecontact"]].dropna()
first_pick.columns = ["without a restraint (A)", "bond to the iron (A)", "contact with the iron (A)"]
show_table(first_pick.reset_index().round(2))
gate_log = (ROOT / "data/structure_track/outputs/cyp3a4/logs/structure_gates.log").read_text()
validity_shares = {name: tuple(float(x) for x in re.search(rf"restraint_{name}: .*'valid_unguided': ([\\d.]+), 'valid_variant': ([\\d.]+)", gate_log).groups()) for name in ("bond", "contact")}
n_restraint_ligands = len(first_pick)
mean_without, mean_bond, mean_contact = (float(first_pick.iloc[:, i].mean()) for i in range(3))
bond_gain = paired_statistics((first_pick.iloc[:, 0] - first_pick.iloc[:, 1]).to_numpy(), 0.05)
contact_gain = paired_statistics((first_pick.iloc[:, 0] - first_pick.iloc[:, 2]).to_numpy(), 0.05)
n_under_2 = {name: int((first_pick[col] < 2).sum()) for name, col in (("without", first_pick.columns[0]), ("bond", first_pick.columns[1]), ("contact", first_pick.columns[2]))}
gate_met = {name: bool(re.search(rf"STRUCTURE GATE restraint_{name}: MET", gate_log)) for name in ("bond", "contact")}
'''

PLAN_FACTS = '''
# The registered plan of the confirmation (data/structure_track/inputs/restraint_expansion_plan.json), read before anything else: its date, its alpha, and the number of ligands that met its eligibility rule.
from cypmode.validation import restraint_expansion as expansion

plan = expansion.load_plan()
alpha_plan = plan["endpoints"]["alpha"]
registered_date = plan["registered"]
lddt_primary = pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/restraint_expansion_primary_lddt.csv", index_col=0)
lddt_secondary = pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/restraint_expansion_secondary_lddt.csv", index_col=0)
n_primary, n_secondary = len(lddt_primary), len(lddt_secondary)
'''

RESTRAINT_CONFIRMATION = '''
# The confirmation of the restraint on ligands that were not used to decide it. The plan was registered on 2026-10-12, before any run (data/structure_track/inputs/restraint_expansion_plan.json): every crystal ligand with a Type II donor,
# organic, at most 30 heavy atoms (the primary stratum; the official ligands have 18 to 27) and a resolution of at most 3.5 angstrom, and not in the benchmark, is run, none picked; two Boltz-2 samples per ligand in each of two conditions;
# the score of a ligand is the mean LDDT-PLI to the crystal; the test is the one-sided Wilcoxon signed-rank test that the restraint raises it, with the mean difference staying positive when any one ligand is dropped; alpha 0.05. A second
# stratum of slightly larger ligands (31 to 34 heavy atoms) is reported and never used for the decision. The cell reads the plan and the per-ligand tables, repeats the statistics with the plan's own function, and then repeats them with
# the scores of OpenStructure (11.4), which is what the registered consequence of the disagreement required.
from cypmode.validation.ost_regate import regate

primary_stats = expansion.paired_statistics(lddt_primary["difference"].to_numpy(), alpha_plan)
secondary_stats = expansion.paired_statistics(lddt_secondary["difference"].to_numpy(), alpha_plan)
expansion_ligands = pd.read_csv(ROOT / "data/structure_track/inputs/restraint_expansion_ligands.csv").assign(comp_id=lambda t: t["comp_id"].astype(str))
expansion_poses = pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/restraint_expansion_poses.csv")
ost_scores = pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/ost_vs_ours.csv")
ost_stats = regate(expansion_poses, ost_scores, expansion_ligands, plan["design"]["samples_per_condition"], alpha_plan)
confirmation_table = pd.DataFrame([
    {"measure": "primary stratum, our LDDT-PLI", **{k: primary_stats[k] for k in ("n", "mean", "wins", "p_better", "p_worse", "min_drop_one")}},
    {"measure": "primary stratum, OpenStructure's LDDT-PLI", **{k: ost_stats["openstructure"][k] for k in ("n", "mean", "wins", "p_better", "p_worse", "min_drop_one")}},
    {"measure": "secondary stratum, our LDDT-PLI", **{k: secondary_stats[k] for k in ("n", "mean", "wins", "p_better", "p_worse", "min_drop_one")}}])
show_table(confirmation_table, floatfmt=".3f")
joined = lddt_primary.join(expansion_ligands.assign(ligand=lambda t: "exp-" + t["comp_id"]).set_index("ligand")[["min_fe_distance", "deposited", "n_heavy"]])
coordinating = joined[joined["min_fe_distance"] <= expansion.COORDINATION_DISTANCE_A]
not_coordinating = joined[joined["min_fe_distance"] > expansion.COORDINATION_DISTANCE_A]
before_cutoff, after_cutoff = joined[joined["deposited"] < expansion.TRAINING_CUTOFF], joined[joined["deposited"] >= expansion.TRAINING_CUTOFF]
subgroups = pd.DataFrame([
    {"subgroup": "the crystal nitrogen coordinates the iron", "ligands": len(coordinating), "mean change of LDDT-PLI": float(coordinating["difference"].mean())},
    {"subgroup": "the crystal ligand does not coordinate the iron", "ligands": len(not_coordinating), "mean change of LDDT-PLI": float(not_coordinating["difference"].mean())},
    {"subgroup": f"deposited before {expansion.TRAINING_CUTOFF}", "ligands": len(before_cutoff), "mean change of LDDT-PLI": float(before_cutoff["difference"].mean())},
    {"subgroup": f"deposited on or after {expansion.TRAINING_CUTOFF}", "ligands": len(after_cutoff), "mean change of LDDT-PLI": float(after_cutoff["difference"].mean())}])
show_table(subgroups, floatfmt=".3f")
from scipy import stats as scipy_stats

difference_sd = float(lddt_primary["difference"].std(ddof=1))
detectable_change = float((scipy_stats.t.ppf(0.95, n_degrees := len(lddt_primary) - 1) + scipy_stats.t.ppf(0.80, n_degrees)) * difference_sd / np.sqrt(len(lddt_primary)))
best_gain_ligand = lddt_primary["difference"].idxmax()
worst_gain_ligand = lddt_primary["difference"].idxmin()
biggest_gain, biggest_loss = float(lddt_primary["difference"].max()), float(lddt_primary["difference"].min())
'''

CELLS = [
    md("""
## 11.6 The iron-nitrogen restraint: what it adds to the input and what the evidence shows

> **Field:** Structural biology · Statistics

*Where this section sits:* 11.5 showed that the unguided model places the iron-binding nitrogen on the iron for most of the ligands that have one and still misplaces the rest of the molecule. This section tests the one lever that acts on the nitrogen directly. It is the only lever of the part that passed its first gate, and the only one that was then put to a second test on new ligands. The account is given in full because the second test did not confirm the first.

**The idea and where it comes from.** The nitrogen of a Type II ligand (1.9) binds the iron of the heme, and in the crystal structures it does so in the large majority of cases (11.5: {{=share_coordinating:.0%}} of the distinct ligands, {{=coordination_table.iloc[1, 2]:.0%}} of those that have a donor). A co-folding model that does not place it there has missed the most informative contact of the pose. The idea of steering the co-folding model with a forced contact between the iron and the donor atom of the ligand was stated as a plan in the public repository `openadmet-cyp-structure` (github.com/xX-its-amit-Xx/openadmet-cyp-structure), whose text reports that 83 of the 101 deposited CYP3A4 ligands it counts coordinate the iron directly, almost always through a nitrogen; the repository presents the steering as a plan and not as a measured gain. The project's count, made with its own criterion (11.5), is {{=int(round(share_coordinating * n_distinct))}} of {{=n_distinct}}. The project built its own version, tested it on the benchmark of 11.5 and then on ligands it had not used, and the numbers below are its own.

**The two forms.** Both use the constraints of 11.3 and the donor atom that `heme_donor_atom` picks. A **bond restraint** declares a covalent bond between the donor nitrogen and the iron: the model is told that the two atoms are bonded, and the nitrogen ends up about 2 Å from the iron (1.95 Å in the file of 11.1). A **contact restraint** states only that the two atoms lie within 4 Å of each other, without enforcement. A ligand with no donor of that kind receives neither and is run as the base run; 12 of the 20 official ligands have a donor.

**The first gate.** Six benchmark ligands have a donor. The rule, written before the runs: keep the restraint if the mean RMSD of the first sample falls and the share of first samples that pass PoseBusters does not. The cell reads the three conditions.
"""),
    code(RESTRAINT_BENCHMARK),
    md("""
The mean first-sample RMSD over the {{=n_restraint_ligands}} ligands is {{=mean_without:.2f}} Å without a restraint, {{=mean_bond:.2f}} Å with the bond and {{=mean_contact:.2f}} Å with the contact; {{=n_under_2['bond']}} ligand{{='s' if n_under_2['bond'] != 1 else ''}} are within 2 Å with the bond against {{=n_under_2['without']}} without. The share of first samples that pass PoseBusters rises from {{=validity_shares['bond'][0]:.2f}} to {{=validity_shares['bond'][1]:.2f}} (for both restraints). Both gates are {{='MET' if gate_met['bond'] else 'NOT MET'}} (bond) and {{='MET' if gate_met['contact'] else 'NOT MET'}} (contact) by the rule, and the recipe takes the one with the lower mean RMSD, the bond restraint. The gain is carried by two ligands: `bench-A1A4S` falls from {{=first_pick.loc['bench-A1A4S'].iloc[0]:.1f}} to {{=first_pick.loc['bench-A1A4S'].iloc[1]:.1f}} Å and `bench-PK9` from {{=first_pick.loc['bench-PK9'].iloc[0]:.1f}} to {{=first_pick.loc['bench-PK9'].iloc[1]:.1f}}; the other four change by a few tenths of an ångström. The paired statistics, given for information because six ligands cannot carry a significance test, are a mean improvement of {{=bond_gain['mean']:+.2f}} Å, {{=bond_gain['wins']}} wins in {{=bond_gain['n']}} and a Wilcoxon p-value of {{=bond_gain['p_better']:.3f}} for the bond, and {{=contact_gain['mean']:+.2f}} Å with {{=contact_gain['wins']}} wins for the contact.

The **Wilcoxon signed-rank test** is the test used throughout the part. For paired measurements (the same ligand under two conditions) it ranks the absolute values of the differences, adds up the ranks of the positive ones, and asks how likely a sum that large would be if the differences were as often positive as negative. It does not assume that the differences follow a normal distribution, which suits a handful of ligands with a few wild values, but with $n$ ligands the smallest p-value it can give is $1/2^n$ for the one-sided test: with six it cannot go below 0.016.

"""),
    code(PLAN_FACTS),
    md("""
**The second test.** The six ligands had decided the restraint, and two of them carried the gain. The rule of the project for such a case is to test the decision on ligands that played no part in it, with a plan that fixes everything before any result exists. The plan was registered on {{=registered_date}} (`data/structure_track/inputs/restraint_expansion_plan.json`): every crystal ligand of the reference set that has a donor, is organic and has a resolution of at most 3.5 Å, with at most 30 heavy atoms (the official ligands have 18 to 27), and is not in the benchmark, is run, none picked; two Boltz-2 samples in each of the two conditions; the score of a ligand is its mean LDDT-PLI to the crystal; the test is the one-sided Wilcoxon test that the restraint raises it at $\\alpha = {{=alpha_plan}}$, with the mean difference staying positive when any one ligand is dropped. If confirmed, the restraint is confirmed; if the one-sided test that it *lowers* the score is significant, the restraint is withdrawn from the recipe; otherwise it stays as decided by its earlier gate and is reported as not confirmed. Only {{=n_primary}} reference ligands met the rule, so all {{=n_primary}} were run; six slightly larger ones (31 to 34 heavy atoms) form a secondary stratum that is reported and never used for the decision. The run took about 12 hours for the primary stratum (jobs `176_restraint_expansion_prepare`, `177_boltz_restraint_expansion_primary`, `178_score_restraint_expansion`, `186_boltz_restraint_expansion_secondary`, `187_score_restraint_expansion_secondary`).
"""),
    code(RESTRAINT_CONFIRMATION),
    md("""
The restraint changes the mean LDDT-PLI of the {{=n_primary}} new ligands by {{=primary_stats['mean']:+.3f}}, in {{=primary_stats['wins']}} of them for the better, with a one-sided p-value of {{=primary_stats['p_better']:.3f}}; and when the ligand with the largest gain is dropped the mean falls to {{=primary_stats['min_drop_one']:+.3f}}, so that it does not stay positive. The restraint is therefore **not confirmed**. Neither is it shown to harm: the one-sided p-value that it lowers the score is {{=primary_stats['p_worse']:.3f}}. With OpenStructure's LDDT-PLI the picture is the same in kind ({{=ost_stats['openstructure']['mean']:+.3f}}, {{=ost_stats['openstructure']['wins']}} of {{=ost_stats['openstructure']['n']}} better, p = {{=ost_stats['openstructure']['p_better']:.3f}}), so that both gates, the confirmation and the harm check, are NOT MET on either yardstick, and the recipe stays as it was. The subgroups show no pattern that would explain the gap: the effect is {{=subgroups.iloc[0, 2]:+.3f}} where the crystal nitrogen coordinates the iron ({{=subgroups.iloc[0, 1]}} ligands) and {{=subgroups.iloc[1, 2]:+.3f}} where it does not ({{=subgroups.iloc[1, 1]}}), and {{=subgroups.iloc[2, 2]:+.3f}} for entries deposited before the end of the model's structure training data against {{=subgroups.iloc[3, 2]:+.3f}} after it. The changes of single ligands run from {{=biggest_loss:+.2f}} to {{=biggest_gain:+.2f}}, with a standard deviation of {{=difference_sd:.2f}}; a test on {{=n_primary}} ligands with a spread that size recognises a true mean change of about {{=detectable_change:.2f}} in 8 of 10 repetitions, which is more than the {{=primary_stats['mean']:.3f}} seen. The reading is that the restraint helps by something between nothing and a small amount in LDDT-PLI, that the earlier benchmark, in which two ligands carried the gain, is consistent with having overstated it, and that this design cannot say more. The project keeps the restraint in the recipe because its first gate was met and the registered rule for exactly this outcome keeps it, and says so; the cost of keeping it is small: on the {{=n_primary}} new ligands it did not lower the score on average ({{=primary_stats['wins']}} ligands better, {{=primary_stats['n'] - primary_stats['wins']}} worse).

*On what actually ran.* `cypmode/validation/build_input.py` (`heme_donor_atom`, `build_boltz_yaml` with `donor_restraint`), `cypmode/validation/structure_gates.py` (the first gate, queue job `165_structure_gate_verdicts`, with the runs of `110_boltz_restraint_bond`, `120_boltz_restraint_contact` and `140_score_restraint_and_pilot`), `cypmode/validation/restraint_expansion.py` (`eligible`, `paired_statistics`, `per_ligand`; the plan file), `cypmode/validation/mbr_lddt_reevaluation.py` (the exploratory re-scoring of the first benchmark with LDDT-PLI, queue job `173_mbr_lddt_reevaluation`: +0.133, 4 of 6 better, p = 0.109) and `cypmode/validation/ost_regate.py` (job `241_ost_regate`). The logs are `structure_gates.log`, `restraint_expansion.log` and `ost_regate.log`.
"""),
]
