"""Part XI (sixth part): choosing among poses (11.8)."""

from notebooks.notebook_builder import code, md

MBR_TOY = '''
# Minimum Bayes risk selection, on the toy complex of 11.4. A pool of poses is given; each pose in turn is treated as the truth, and every other pose is scored against it with LDDT-PLI; the expected score of a pose is the mean over
# the others, and the pose with the highest expected score is chosen (cypmode/validation/pose_selection.py, pose_pools.mbr_pick). It needs no confidence score and no fitted parameter. Two pools of six poses: in the first, four poses
# lie close to the true pose and two are far from it and from each other; in the second, the four that agree with each other are all wrong (the same displacement of 4 angstrom) and the two poses near the truth are the minority.
from cypmode.validation.pose_pools import mbr_pick
from cypmode.validation.pose_selection import expected_lddt

rng_pool = np.random.default_rng(8)
def jitter(scale):
    return ligand + scale * rng_pool.standard_normal(ligand.shape)
def unit(vector):
    return vector / np.linalg.norm(vector)
away = unit(np.array([1.0, 0.7, -0.4]))
pools_toy = {
    "most poses near the truth": [jitter(0.2), jitter(0.25), jitter(0.3), jitter(0.2), ligand + 5.0 * unit(rng_pool.standard_normal(3)), ligand - 5.0 * unit(rng_pool.standard_normal(3))],
    "most poses agree and are wrong": [ligand + 4.0 * away + 0.2 * rng_pool.standard_normal(ligand.shape) for _ in range(4)] + [jitter(0.2), jitter(0.3)],
}
toy_rows = []
for name, poses in pools_toy.items():
    expected = expected_lddt([complex_dicts(p, pocket) for p in poses])
    chosen = int(expected.argmax())
    errors = [rmsd(p, ligand) for p in poses]
    toy_rows.append({"pool": name, "chosen pose": chosen, "RMSD of the chosen pose (A)": errors[chosen], "RMSD of the best pose in the pool (A)": min(errors), "mean RMSD of the pool (A)": float(np.mean(errors))})
mbr_toy_table = pd.DataFrame(toy_rows)
show_table(mbr_toy_table, floatfmt=".2f")
'''

SELECTION_RESULTS = '''
# The rules of selection on the benchmark of 11.5. Every rule answers one question for each ligand, "which single pose do we submit?", and is scored by that pose's RMSD to the crystal pose. The table of each ligand's outcomes is the one
# written by cypmode/validation/selection_validation.py (job 170_mbr_selection_validation); the gates are recomputed with its own function. "Oracle" is the best pose that exists in the pool: the ceiling that a rule could reach and
# no rule can apply. The second table adds the experimental-prior rescoring of 11.5 (pose_rescoring.py), which scores a pose by how well it agrees with where deposited ligands sit in the pocket, on the same ligands.
from cypmode.validation.selection_validation import gates as selection_gates

selection_outcomes = pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/selection_validation_outcomes.csv")
selection_g = selection_gates(selection_outcomes)
summary_means = selection_outcomes.drop(columns="compound").mean()
rule_table = pd.DataFrame({"mean RMSD (A)": summary_means}).rename(index={"first_boltz": "Boltz-2, first sample", "mbr_boltz": "Boltz-2, minimum Bayes risk", "mean_pool_boltz": "Boltz-2, a random sample", "oracle_boltz": "Boltz-2, best sample (oracle)",
                                                                          "first_openfold3": "OpenFold3, first sample", "mbr_both": "both engines, minimum Bayes risk", "oracle_both": "both engines, best sample (oracle)"})
show_table(rule_table.reset_index().rename(columns={"index": "rule"}), floatfmt=".2f")
rescoring = pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/benchmark_selection_outcomes.csv")
rescoring_table = rescoring.groupby("rule")["rmsd"].agg(["mean", lambda x: (x < 2).mean()]).rename(columns={"mean": "mean RMSD (A)", "<lambda_0>": "share within 2 A"})
show_table(rescoring_table.reset_index().rename(columns={"rule": "rule"}), floatfmt=".2f")
mbr_gate, engine_gate = selection_g["mbr_selection"], selection_g["engine_consensus"]
lddt_outcomes = pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/selection_lddt_outcomes.csv")
lddt_gate = selection_gates(lddt_outcomes.assign(first_boltz=-lddt_outcomes["first_boltz"], mbr_boltz=-lddt_outcomes["mbr_boltz"]).drop(columns=["mbr_both", "first_openfold3"]))["mbr_selection"]
lddt_means = lddt_outcomes[["first_boltz", "mbr_boltz", "oracle_boltz"]].mean()
first_pick_mean, mbr_mean, oracle_mean = float(summary_means["first_boltz"]), float(summary_means["mbr_boltz"]), float(summary_means["oracle_boltz"])
share_of_ceiling = (first_pick_mean - mbr_mean) / (first_pick_mean - oracle_mean)
reeval_text = (ROOT / "data/structure_track/outputs/cyp3a4/logs/mbr_lddt_reevaluation.log").read_text()
restrained_found = re.search(r"mbr_on_restrained_lddt_exploratory: mean LDDT-PLI gain ([+-][\\d.]+) over (\\d+) ligands, (\\d+) wins", reeval_text)
restrained_gain, restrained_n, restrained_wins = float(restrained_found.group(1)), int(restrained_found.group(2)), int(restrained_found.group(3))
from scipy import stats as scipy_stats

improvement_sd = float((selection_outcomes['first_boltz'] - selection_outcomes['mbr_boltz']).std(ddof=1))
n_selection = len(selection_outcomes)
detectable_improvement = float((scipy_stats.t.ppf(0.90, n_selection - 1) + scipy_stats.t.ppf(0.80, n_selection - 1)) * improvement_sd / np.sqrt(n_selection))
template_rule = rescoring_table.loc["template score over Boltz samples"]
first_rule = rescoring_table.loc["boltz first pick"]
openfold_first = float(summary_means["first_openfold3"])
'''

POSE_TERMS = '''
# The other things that were tried to choose a pose, each with a rule written before its result (the logs of the gates of jobs 168 to 175): the heme-frame prior (every pose described in the heme's own frame, scored by how
# typical it is of the 148 crystal ligands); three terms that describe a pose with knowledge the co-folding model does not use (agreement with the low-energy conformers of the ligand and its strain, from GFN2-xTB; packing and
# clashes; the distance of the donor nitrogen to the iron, or of the best oxidation site to it); the agreement of a pose with the poses of the ligand's analogues; and expected-improvement sampling (give the next sample to the
# ligand whose pool still disagrees most, at equal total compute). For the first four the statistic is the mean over ligands of the Spearman correlation between the term and minus the RMSD to the crystal (a term that ranks the
# poses must have it above 0.3, and positive for at least 60 percent of the ligands). The values are read from the logs of the gates.
log_dir = ROOT / "data/structure_track/outputs/cyp3a4/logs"
heme_text, ei_text = (log_dir / "heme_frame_prior.log").read_text(), (log_dir / "ei_sampling.log").read_text()
aware_text, series_text = (log_dir / "pose_awareness.log").read_text(), (log_dir / "series_consensus.log").read_text()
def number(pattern, text):
    found = re.search(pattern, text, re.M)
    return float(found.group(1)) if found else np.nan
def verdict_of(name, text):
    found = re.search(rf"STRUCTURE GATE {name} .*: (MET|NOT MET)$", text, re.M)
    return found.group(1) if found else "no verdict"
random_pose_rmsd = number(r"random pose ([\\d.]+)", aware_text)
terms_rows = [{"term": "heme-frame prior", "mean rho": number(r"mean Spearman rho\\(prior, -RMSD\\) (-?[\\d.]+)", heme_text), "ligands with rho > 0 (%)": number(r"positive for (\\d+)% of 10", heme_text),
               "RMSD of the picked pose": number(r"prior alone ([\\d.]+)", heme_text), "RMSD with minimum Bayes risk": number(r"MBR ([\\d.]+),", heme_text), "gate": verdict_of("heme_frame_prior", heme_text)}]
for label, key, name in (("physics and molecule term", "physics", "pose_physics"), ("pharmacophore term", "pharmacophore", "pose_pharmacophore"), ("the three terms combined", "combined", "pose_combined")):
    terms_rows.append({"term": label, "mean rho": number(rf"{key}: mean Spearman rho \\+?(-?[\\d.]+)", aware_text), "ligands with rho > 0 (%)": number(rf"{key}: .*positive for (\\d+)%", aware_text),
                       "RMSD of the picked pose": number(rf"{key}: .*top-scored pose ([\\d.]+)", aware_text), "RMSD with minimum Bayes risk": number(rf"{key}: .*\\(MBR ([\\d.]+)", aware_text), "gate": verdict_of(name, aware_text)})
terms_rows.append({"term": "agreement with the analogues (6 ligands)", "mean rho": np.nan, "ligands with rho > 0 (%)": np.nan, "RMSD of the picked pose": number(r"MBR \\+ series ([\\d.]+)", series_text), "RMSD with minimum Bayes risk": number(r"MBR ([\\d.]+), series", series_text),
                   "gate": verdict_of("series_consensus", series_text)})
pose_terms_table = pd.DataFrame(terms_rows)
show_table(pose_terms_table, floatfmt=".2f")
ei_one = number(r"ei vs uniform: mean improvement (\\+?-?[\\d.]+)", ei_text)
ei_two = number(r"extra budget 2 samples.*\\n  ei vs uniform: mean improvement (\\+?-?[\\d.]+)", ei_text)
ei_p = number(r"ei vs uniform: mean improvement .*Wilcoxon p ([\\d.]+)", ei_text)
ei_verdict = verdict_of("ei_sampling", ei_text)
n_terms_not_met = int((pose_terms_table["gate"] == "NOT MET").sum()) + int(ei_verdict == "NOT MET")
ost_summary = re.findall(r"^  (\\w+): (MET|NOT MET) -> (MET|NOT MET)$", (log_dir / "ost_pool_gates.log").read_text(), re.M)
n_ost_repeated, n_ost_changed = len(ost_summary), sum(a != b for _, a, b in ost_summary)
n_ost_pose_rows = len(pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/ost_pools.csv"))
plumbing = re.search(r"CHECK of the plumbing: (\\d+) gate verdicts from unchanged copies of the tables, (\\d+) equal", (log_dir / "ost_pool_gates.log").read_text())
plumbing_total, plumbing_equal = int(plumbing.group(1)), int(plumbing.group(2))
'''

CELLS = [
    md("""
## 11.8 Choosing among poses: minimum Bayes risk, a second engine, priors and the physics

> **Field:** Structural biology · Statistics

*Where this section sits:* 11.7 tried to make the pool of poses differ. This section asks the question that follows, which is whether any rule picks the best pose from a pool better than the model's own first sample does. The model's confidence cannot do it (11.5). The rules below use only what is available for the 20 official ligands, which have no answer key.

**Minimum Bayes risk.** The score is LDDT-PLI against the unknown true complex (11.4). If the pool is a sample of what the model thinks the truth might be, the pose with the best expected score is the one that agrees best with the others: treat each pose in turn as the truth, score all the others against it, and take the pose with the highest mean score. This is **minimum Bayes risk** (MBR) selection (`cypmode/validation/pose_selection.py`, `pose_pools.py`). It has no parameter to fit, so that a leave-one-ligand-out test is automatic, and it needs no confidence. It is the consensus of the pool. Whether the consensus is right is another matter, which the first cell shows with two pools of six poses of the toy complex of 11.4.
"""),
    code(MBR_TOY),
    md("""
In the first pool, four of the six poses lie near the truth, and the choice is one of them ({{=mbr_toy_table.iloc[0, 2]:.2f}} Å from the truth, against a pool mean of {{=mbr_toy_table.iloc[0, 4]:.2f}}). In the second the majority is wrong in the same way (four poses 4 Å away in one direction), and MBR chooses a member of it ({{=mbr_toy_table.iloc[1, 2]:.2f}} Å) although the pool holds two poses within {{=mbr_toy_table.iloc[1, 3]:.1f}} Å of the truth. Consensus is a good guide when the errors of the model are random and a bad one when they are systematic, and the observation of 11.5 that the five samples of a prediction agree with each other and are wrong together is the second case.

**MBR on the benchmark.** The cell takes the ten ligands of the benchmark with their pools (five Boltz-2 samples each) and compares the rules, scored by the RMSD of the chosen pose to the crystal pose, with the gate that was registered before the full benchmark was scored (`cypmode/validation/selection_validation.py`, queue job `170_mbr_selection_validation`): MBR over the Boltz samples against the first sample, with a lower mean RMSD, an improvement that survives dropping any one ligand, and a one-sided Wilcoxon p-value below 0.10 (stated openly: with about ten ligands, 0.05 would need almost every ligand to improve).
"""),
    code(SELECTION_RESULTS),
    md("""
The first sample of Boltz-2 is {{=first_pick_mean:.2f}} Å from the crystal pose on average. The best pose that exists in the five-sample pools (the oracle, which no rule can apply) is {{=oracle_mean:.2f}} Å: choosing within a Boltz pool can therefore gain at most {{=first_pick_mean - oracle_mean:.2f}} Å. MBR reaches {{=mbr_mean:.2f}} Å, a gain of {{=mbr_gate['mean_improvement']:.2f}} Å, which is {{=share_of_ceiling:.0%}} of the ceiling, with {{=mbr_gate['wins']}} of {{=mbr_gate['n']}} ligands better and a Wilcoxon p-value of {{=mbr_gate['p']:.3f}}. The mean stays positive when any one ligand is dropped (the worst drop-one mean is {{=mbr_gate['min_drop_one']:+.3f}}) but the p-value is above 0.10: the gate `mbr_selection` is NOT MET. Scored with the challenge's own metric, which is the one that MBR optimises by construction, the picture is the same (LDDT-PLI {{=lddt_means['first_boltz']:.3f}} for the first sample, {{=lddt_means['mbr_boltz']:.3f}} for MBR and {{=lddt_means['oracle_boltz']:.3f}} for the oracle; a gain of {{=lddt_gate['mean_improvement']:+.3f}}, {{=lddt_gate['wins']}} wins in {{=lddt_gate['n']}}, p = {{=lddt_gate['p']:.3f}}, exploratory because the registered gate used RMSD, queue job `173_mbr_lddt_reevaluation`). Inside the pools of the bond-restrained runs, MBR gains {{=restrained_gain:+.3f}} in LDDT-PLI over their first samples ({{=restrained_wins}} win{{='' if restrained_wins == 1 else 's'}} in {{=restrained_n}}).

**The experimental prior.** A pose can also be scored by how well it agrees with where deposited ligands sit in the pocket (`cypmode/validation/pose_rescoring.py`: each reference ligand is superposed into the frame of the pose by the pocket's C-alpha atoms, and two scores are computed from the coordinates alone). The second table shows the rule beside the first sample: the mean RMSD of the pose it picks from the Boltz samples is {{=template_rule['mean RMSD (A)']:.2f}} Å against {{=first_rule['mean RMSD (A)']:.2f}}, and {{=template_rule['share within 2 A']:.0%}} of its picks are within 2 Å against {{=first_rule['share within 2 A']:.0%}}. For the 20 official ligands the template score is switched off by design, because their best similarity to a reference ligand is below the support floor of 0.5 (11.5): the prior can only help with a ligand that resembles a deposited one, and none does.

**A second engine.** The gate `engine_consensus` asks whether a second engine adds information: its errors must differ from Boltz-2's (rank correlation of the per-ligand RMSDs below 0.7) and MBR over both engines must beat MBR over Boltz-2 alone. OpenFold3 (11.9) is the second engine. On the benchmark its first sample is {{=openfold_first:.2f}} Å from the crystal pose, worse than Boltz-2's {{=first_pick_mean:.2f}}. The errors do differ (rank correlation {{=engine_gate['errors_correlation']:.2f}}), which is the first condition, but MBR over both engines is {{=engine_gate['mean_improvement']:+.2f}} Å against MBR over Boltz-2 alone, with {{=engine_gate['wins']}} wins in {{=engine_gate['n']}} (p = {{=engine_gate['p']:.3f}}): NOT MET. A second opinion that is on average worse than the first does not improve a vote.

**Terms that describe a pose with knowledge the model does not use.** Four more rules were tried, each scoring the poses of a pool and each judged by the Spearman correlation between its score and minus the RMSD to the crystal (a term that ranks the poses has it above 0.3 and positive for at least 60 percent of the ligands). The **heme-frame prior** describes each pose in the heme's own frame (the iron as origin, the porphyrin normal as axis: the height of the nearest ligand atom above the plane, its radial distance, the tilt of the ligand's long axis) by seven rotation-invariant numbers, and asks how typical they are of the 148 crystal ligands (`cypmode/validation/heme_frame_prior.py`). The **physics and molecule term** (`cypmode/validation/pose_awareness.py`) compares the pose with the low-energy conformers of the ligand, ranked by GFN2-xTB in implicit water (10.4), and measures packing and clashes; the **pharmacophore term** measures, for a ligand with a donor, how far its iron-nitrogen distance is from the 2.2 Å of a dative bond and, for the others, whether the most easily oxidised carbon (the lowest hydrogen-abstraction energy of 10.5) lies in the 3.5 to 6.0 Å window from the iron; the **series term** (`cypmode/validation/series.py` and `series_consensus.py`) takes a pose of a ligand to be supported by the poses of its analogues, by the agreement of their common core after the two complexes are superposed on the pocket (three crystal-anchored pairs of analogues, six ligands). **Expected-improvement sampling** (`cypmode/validation/ei_sampling.py`) is a different kind of rule: it does not pick a pose but spends the extra samples where the pool disagrees most, and is compared with uniform allocation at equal compute.
"""),
    code(POSE_TERMS),
    md("""
None passes. The heme-frame prior has a mean correlation of {{=pose_terms_table.iloc[0, 1]:+.3f}} and is positive for {{=pose_terms_table.iloc[0, 2]:.0f}} percent of the ligands, which is no better than a coin; the three pose terms reach {{=pose_terms_table.iloc[1, 1]:+.3f}}, {{=pose_terms_table.iloc[2, 1]:+.3f}} and {{=pose_terms_table.iloc[3, 1]:+.3f}}, none above 0.3, and the pose each picks is {{=pose_terms_table.iloc[1, 3]:.2f}}, {{=pose_terms_table.iloc[2, 3]:.2f}} and {{=pose_terms_table.iloc[3, 3]:.2f}} Å from the crystal pose against a random pose's {{=random_pose_rmsd:.2f}}; the series term over six ligands picks poses {{=pose_terms_table.iloc[4, 3]:.2f}} Å away against MBR's {{=pose_terms_table.iloc[4, 4]:.2f}}. Expected-improvement sampling gains {{=ei_one:+.3f}} Å over uniform sampling with one extra sample per ligand and {{=ei_two:+.3f}} Å with two (p = {{=ei_p:.3f}}). In all, {{=n_terms_not_met}} of the rules of this subsection are NOT MET. The pharmacophore term is the nearest to anything: its picks average {{=pose_terms_table.iloc[2, 3]:.2f}} Å, better than the random pose's {{=random_pose_rmsd:.2f}} by {{=random_pose_rmsd - pose_terms_table.iloc[2, 3]:.2f}} Å, on a correlation of {{=pose_terms_table.iloc[2, 1]:+.3f}}; the rule requires 0.3 and it was not relaxed.

**What limits these tests.** Ten ligands, about four ångströms of typical error, and a ceiling of {{=first_pick_mean - oracle_mean:.2f}} Å between the first sample and the best sample that exists. The improvement of a rule over the first sample varies from ligand to ligand with a standard deviation of {{=improvement_sd:.2f}} Å for MBR; a test on {{=n_selection}} ligands recognises a true mean improvement of about {{=detectable_improvement:.2f}} Å in 8 of 10 repetitions at the one-sided level 0.10, more than the {{=mbr_gate['mean_improvement']:.2f}} Å that MBR gains. A verdict of NOT MET therefore says that the benchmark does not show an effect, not that the rule has none; the same qualification as for the physics features of 10.6. The registered consequence is that no rule is used: the recipe takes the model's own order, with the restraint of 11.6 where it applies, and chooses the first sample that passes the validity checks with a safe margin (11.10).

**The same gates with the organisers' yardstick.** The disagreement of 11.4 between the project's LDDT-PLI and OpenStructure's required the gates that rest on the project's measure to be repeated with the organisers'. {{=n_ost_repeated}} gates of this part were repeated on the {{=n_ost_pose_rows}} poses of their tables (job `244_ost_pools_score` scored the poses, about 4 seconds each, in the conda environment `ost`; job `245_ost_pool_gates` applied the same code and the same rules to copies of the tables in which the RMSD is OpenStructure's BiSyRMSD): the verdicts changed for {{=n_ost_changed}} of them. Before the repeat, the code was run on unchanged copies of the tables and stopped if any verdict differed from the original log; {{=plumbing_equal}} of {{=plumbing_total}} were equal. The copies are written to `ost_yardstick/`, and the verdict lines carry the suffix `_ost`.

*On what actually ran.* `cypmode/validation/selection_validation.py` (`paired_gate`, `gates`, `evaluate_rules`), `pose_selection.py` (`lddt_pli_arrays`, `expected_lddt`, `select_pose`), `pose_pools.py` (`pairwise_lddt`, `mbr_pick`, `pool_disagreement`, `symmetric_lddt_pli`), `pose_rescoring.py` and `pose_rescoring_validation.py` (the prior and its test against decoys; `reference_ligands.py` supplies the references), `heme_frame_prior.py`, `pose_awareness.py`, `series.py`, `series_consensus.py`, `ei_sampling.py`, `mbr_lddt_reevaluation.py`, `lddt_to_crystal.py`, `ost_pools.py`, `ost_pool_gates.py` and `structure_benchmark_report.py` (the comparison of all rules); queue jobs `160_score_all_and_report`, `165_structure_gate_verdicts`, `168_pose_awareness`, `169_series_consensus`, `170_mbr_selection_validation`, `172_ei_sampling`, `173_mbr_lddt_reevaluation`, `175_heme_frame_prior`, `133_series_benchmark_runs`, `142_series_scoring`, `244_ost_pools_score` and `245_ost_pool_gates`. The logs are `selection_validation.log`, `mbr_lddt_reevaluation.log`, `heme_frame_prior.log`, `pose_awareness.log`, `series_consensus.log`, `ei_sampling.log` and `ost_pool_gates.log`.
"""),
]
