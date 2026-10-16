"""Part VI (third part): why the project stopped using Tukey's HSD (6.6), the pre-declared gates and what a design could detect (6.7), and the checks that the split and the pipeline do not leak (6.8)."""

from notebooks.notebook_builder import code, figure, md

SCORES_COUNT = '''
# What the saved tables hold: for each experiment, the MCC of every variant on every (repeat, fold) of a repeated cross-validation, for each enzyme. The first of them (the ChemBERTa experiment) is counted here.
COMPARISONS = ROOT / "data/tdi_track/outputs/comparisons"
chemberta_raw = pd.read_csv(COMPARISONS / "chemberta_significance_raw.csv")
scores_per_variant, n_variants = int(chemberta_raw.groupby(["isoform", "variant"]).size().iloc[0]), int(chemberta_raw["variant"].nunique())
n_repeats_saved, n_folds_saved = int(chemberta_raw["repeat"].nunique()), int(chemberta_raw["fold"].nunique())
print(f"{chemberta_raw['isoform'].nunique()} enzymes x {n_variants} variants x {n_repeats_saved} repeats x {n_folds_saved} folds = {len(chemberta_raw)} saved scores; {scores_per_variant} per variant and enzyme")
'''

TUKEY_VS_PAIRED = '''
# The two analyses on the same saved scores. Five earlier experiments saved the per-fold MCC of every variant (three to four variants each, repeated cross-validation). They were first analysed with Tukey's HSD, which
# treats the fold scores as independent observations of each variant. Here both analyses are recomputed from the saved scores, Tukey's HSD with the library and the corrected paired t with Holm's adjustment with the
# project's function (the one of 6.5), and set against the result files the project wrote.
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from cypmode.challenge.stacking_significance import paired_comparison_summary

comparison_names = ["stacking", "chemberta", "pharmacophore_fingerprint", "qm_reactivity", "quantum_pharmacophore"]
summary_rows, flipped_frames = [], []
for name in comparison_names:
    raw = pd.read_csv(COMPARISONS / f"{name}_significance_raw.csv")
    tukey_frames, paired_frames = [], []
    for isoform, part in raw.groupby("isoform"):
        result = pairwise_tukeyhsd(part["mcc"].to_numpy(), part["variant"].to_numpy(), alpha=0.05)
        tukey_frames.append(pd.DataFrame(result._results_table.data[1:], columns=result._results_table.data[0]).assign(isoform=isoform))
        paired_frames.append(paired_comparison_summary(part, "mcc").assign(isoform=isoform))
    tukey_all, paired_all = pd.concat(tukey_frames), pd.concat(paired_frames)
    both = tukey_all.merge(paired_all, on=["isoform", "group1", "group2"], suffixes=(" Tukey", " paired"))
    stored_tukey = pd.read_csv(COMPARISONS / f"{name}_significance_tukey.csv")
    stored_paired = pd.read_csv(COMPARISONS / f"{name}_significance_paired.csv")
    summary_rows.append({"experiment": name, "pairs compared": len(both), "Tukey says different": int(both["reject Tukey"].sum()), "corrected paired t says different": int(both["reject paired"].sum()),
                         "files written earlier agree": bool(stored_tukey["reject"].sum() == both["reject Tukey"].sum() and stored_paired["reject"].sum() == both["reject paired"].sum())})
    flipped_frames.append(both[both["reject Tukey"] & ~both["reject paired"]].assign(experiment=name))
summary_table = pd.DataFrame(summary_rows)
show_table(summary_table)
flipped = pd.concat(flipped_frames)
show_table(flipped.assign(pair=flipped["group1"] + " vs " + flipped["group2"])[["experiment", "isoform", "pair", "meandiff Tukey", "p-adj Tukey", "p-adj paired"]].rename(columns={"meandiff Tukey": "difference of the mean MCC", "p-adj Tukey": "Tukey p", "p-adj paired": "corrected paired t, Holm p"}), floatfmt=".4f")
n_pairs_all, n_tukey_all, n_paired_all = int(summary_table["pairs compared"].sum()), int(summary_table["Tukey says different"].sum()), int(summary_table["corrected paired t says different"].sum())
n_flipped = len(flipped)
files_agree = bool(summary_table["files written earlier agree"].all())
flip_experiments = sorted(flipped["experiment"].unique())
example = flipped[(flipped["isoform"] == "CYP3A4") & (flipped["group1"] == "M3") & (flipped["group2"] == "M3+ChemBERTa")].iloc[0]
example_diff, example_tukey_p, example_paired_p = float(example["meandiff Tukey"]), float(example["p-adj Tukey"]), float(example["p-adj paired"])
'''

FROZEN_BOOTSTRAP = '''
# The other shortcut, on two real members of the stack: resampling the compounds while the predictions stay frozen. The difference of the AUC of two members (the deep LightGBM on feature set A plus fingerprints, and its
# shallow twin) is given an interval in two ways: by a bootstrap over the compounds of the frozen out-of-fold predictions of the first repeat, and by the corrected t over the 15 outer folds of the plan.
from scipy import stats

from cypmode.challenge.model_comparison import corrected_resampled_t, plan_sizes

member_a, member_b = "lgbm_A+fps_deep", "lgbm_A+fps_shallow"
fold_differences, frozen = [], {}
for repeat, outer in plan.outer_folds():
    refit = plan.task(repeat, outer, OUTER_REFIT)
    y_part, s_a, s_b = plan.labels(refit.evaluate), cache.get(member_a, refit), cache.get(member_b, refit)
    fold_differences.append(roc_auc_score(y_part, s_a) - roc_auc_score(y_part, s_b))
    if repeat == 0:
        frozen.update({int(row): (a_, b_, label) for row, a_, b_, label in zip(refit.evaluate, s_a, s_b, y_part)})
fold_differences = np.array(fold_differences)
corrected = corrected_resampled_t(fold_differences, *plan_sizes(plan))
half_width = float(stats.t.ppf(0.975, len(fold_differences) - 1) * corrected.standard_error)
a_frozen, b_frozen, y_frozen = (np.array([frozen[r][i] for r in sorted(frozen)]) for i in range(3))
rng_boot = np.random.default_rng(0)
boot_diffs = []
for _ in range(2000):
    pick = rng_boot.integers(0, len(y_frozen), len(y_frozen))
    boot_diffs.append(roc_auc_score(y_frozen[pick], a_frozen[pick]) - roc_auc_score(y_frozen[pick], b_frozen[pick]))
boot_low, boot_high = np.percentile(boot_diffs, [2.5, 97.5])
frozen_rows = [{"how the interval is made": "bootstrap over compounds, predictions frozen (repeat 0)", "difference": float(roc_auc_score(y_frozen, a_frozen) - roc_auc_score(y_frozen, b_frozen)), "95 % interval": f"{boot_low:+.4f} to {boot_high:+.4f}", "width": float(boot_high - boot_low)},
               {"how the interval is made": "corrected resampled t over the 15 outer folds", "difference": float(fold_differences.mean()), "95 % interval": f"{fold_differences.mean() - half_width:+.4f} to {fold_differences.mean() + half_width:+.4f}", "width": 2 * half_width}]
show_table(pd.DataFrame(frozen_rows), floatfmt=".4f")
width_ratio = (2 * half_width) / (boot_high - boot_low)
bootstrap_excludes_zero, corrected_excludes_zero = bool(boot_low > 0 or boot_high < 0), bool(fold_differences.mean() - half_width > 0 or fold_differences.mean() + half_width < 0)
corrected_p = corrected.p_value
print(f"the corrected interval is {width_ratio:.1f} times as wide as the bootstrap one; corrected p = {corrected_p:.3f}")
'''

GATE_REPRODUCTION = '''
# The gate of the project, reproduced from the cache for four additions to the stack (the multitask network; the same with the screen and Emax as auxiliary heads; TabICL; plain Chemprop). For each, the stack of the 23
# default members plus the addition is compared with the 23 alone, on the nested plan, with the weighted yardstick of 6.4 at both cut-offs, by the corrected t of 6.5. The p-values are then set against the ones the
# project's own experiment logged when it ran. The standard deviation of the fold differences, recovered from the standard error, feeds the next cell.
import re

from cypmode.challenge.cached_candidates import GROUPS
from cypmode.challenge.label_confidence import confident_weights

n_train_plan, n_test_plan = plan_sizes(plan)
ratio_5 = n_test_plan / n_train_plan
J_5 = len(plan.outer_folds())
scores_of = {}
reproduced_rows, fold_sd_5 = [], []
for group in ("mlp", "mlp_aux", "tabicl", "chemprop"):
    added = [c.name for c in GROUPS[group]()]
    log_text = (ROOT / f"data/structure_track/outputs/cyp3a4/logs/aux_{group}_CYP3A4.log").read_text(errors="ignore")
    logged = {z0: float(p) for z0, p in re.findall(r"z0=(1\\.0|1\\.96)\\s+stacked base\\+added vs base: AUC [+-]\\d\\.\\d+ \\(p=([\\d.]+)", log_text)}
    row = {"addition": group}
    for z0 in (1.0, 1.96):
        weight_fn = confident_weights(matcher, z_all, z0)
        if z0 not in scores_of:
            scores_of[z0] = nested_evaluate(cache, default_names, [StackedLogit()], weight_fn, target_prevalence, "auc").set_index(["repeat", "outer"])["outer_score"]
        with_addition = nested_evaluate(cache, default_names + added, [StackedLogit()], weight_fn, target_prevalence, "auc").set_index(["repeat", "outer"])["outer_score"]
        result = corrected_resampled_t((with_addition - scores_of[z0]).to_numpy(), n_train_plan, n_test_plan)
        row[f"gain z0 {z0}"], row[f"p z0 {z0}"], row[f"logged p z0 {z0}"] = result.mean_difference, result.p_value, logged[str(z0)]
        fold_sd_5.append(result.standard_error / np.sqrt(1 / J_5 + ratio_5))
    reproduced_rows.append(row)
reproduced = pd.DataFrame(reproduced_rows)
show_table(reproduced, floatfmt=".3f")
agrees_with_logs = bool(all(abs(r[f"p z0 {z0}"] - r[f"logged p z0 {z0}"]) < 0.0015 for _, r in reproduced.iterrows() for z0 in (1.0, 1.96)))
n_gate_met = int(sum(all(r[f"p z0 {z0}"] < 0.05 and r[f"gain z0 {z0}"] > 0 for z0 in (1.0, 1.96)) for _, r in reproduced.iterrows()))
sd_5 = float(np.median(fold_sd_5))
print(f"median standard deviation of the 15 fold differences over the eight comparisons: {sd_5:.4f}; reproduced p-values equal the logged ones: {agrees_with_logs}; additions meeting the gate: {n_gate_met} of 4")
'''

DESIGN_FLOOR = '''
# The smallest true AUC difference that a design could recognise. The test rejects when |mean| > t_crit * SE with SE = s_d * sqrt(1/J + n_test/n_train), s_d the standard deviation of the fold differences. If the true
# difference is delta, the mean is about delta with standard error SE, so the test rejects 8 times in 10 when delta exceeds the critical value by the 0.8 quantile of t times SE:
# MDE = (t_{0.975, J-1} + t_{0.8, J-1}) * SE (the central-t approximation, close for these degrees of freedom). s_d is measured, not assumed: for the 5-fold design from the eight comparisons just reproduced,
# and for the 10-fold replicate from its own logs (the stacked comparisons of the physics blocks and of the neural pair), recovering SE from each logged (difference, p).
def minimum_detectable_effect(j, ratio, sd):
    return float((stats.t.ppf(0.975, j - 1) + stats.t.ppf(0.8, j - 1)) * sd * np.sqrt(1 / j + ratio))

STACKED_LINE = re.compile(r"z0=(\\d\\.\\d+)\\s+stacked base\\+added vs base: AUC ([+-]\\d\\.\\d+) \\(p=(\\d\\.\\d+)")
J_10, ratio_10 = 30, 1 / 9
sd_10_values = []
for stem in ("physics_experiment_f10", "physics_compact_f10", "physics_tierb_f10", "aux_neural_f10"):
    path = ROOT / f"data/structure_track/outputs/cyp3a4/logs/{stem}_CYP3A4.log"
    for z0, gain, p in STACKED_LINE.findall(path.read_text(errors="ignore") if path.exists() else ""):
        gain, p = float(gain), float(p)
        if 0.01 < p < 0.99 and gain != 0:
            standard_error = abs(gain) / stats.t.isf(p / 2, J_10 - 1)
            sd_10_values.append(standard_error / np.sqrt(1 / J_10 + ratio_10))
sd_10 = float(np.median(sd_10_values))
floor_rows = [{"design": "5 outer folds x 3 repeats (the main design), corrected t", "folds J": 15, "variance factor": 1 / 15 + 1 / 4, "fold sd measured": sd_5, "smallest difference found 8 times in 10": minimum_detectable_effect(15, 1 / 4, sd_5)},
              {"design": "10 outer folds x 3 repeats (the replicate), corrected t", "folds J": 30, "variance factor": 1 / 30 + 1 / 9, "fold sd measured": sd_10, "smallest difference found 8 times in 10": minimum_detectable_effect(30, 1 / 9, sd_10)},
              {"design": "main design if the folds really were independent (what the naive test assumes)", "folds J": 15, "variance factor": 1 / 15, "fold sd measured": sd_5, "smallest difference found 8 times in 10": minimum_detectable_effect(15, 0.0, sd_5)}]
show_table(pd.DataFrame(floor_rows), floatfmt=".4f")
floor_main, floor_replicate, floor_naive = (r["smallest difference found 8 times in 10"] for r in floor_rows)
n_sd_10 = len(sd_10_values)
def power_at(delta, j, ratio, sd):
    standard_error = sd * np.sqrt(1 / j + ratio)
    t_crit, ncp = stats.t.ppf(0.975, j - 1), delta / standard_error
    return float(stats.nct.sf(t_crit, j - 1, ncp) + stats.nct.cdf(-t_crit, j - 1, ncp))
power_main, power_replicate = power_at(0.004, 15, 1 / 4, sd_5), power_at(0.004, 30, 1 / 9, sd_10)
gains_logged = family["gain at z0 1.0"].to_numpy()                         # the stacked gains of every logged CYP3A4 addition (the family of 6.5)
n_above_main, n_above_replicate, n_logged = int((gains_logged > floor_main).sum()), int((gains_logged > floor_replicate).sum()), len(gains_logged)
median_gain = float(np.median(gains_logged))
'''

SPLIT_BLOCK = '''
# Is a split informative? Two stored results. The first is the similarity profile of the project's own cross-validation folds and of the blinded test compounds, each against the training set it is judged from
# (cypmode/challenge/split_validity.py). The second is the test-like hold-out (cypmode/challenge/block_holdout.py): the analogue-expansion round of the training file held out whole, the candidates fitted on the rest and
# scored on the block, next to their cross-validation AUC. The log of that run is read back and the agreement of the two yardsticks is recomputed from its table.
from io import StringIO

from scipy.stats import spearmanr

split_table = pd.read_csv(COMPARISONS / "split_validity.csv")
show_table(split_table.rename(columns={"median_similarity": "median similarity", "frac_similarity_ge_0.5": "share at 0.5 or more", "frac_similarity_ge_0.7": "share at 0.7 or more"})[["comparison", "n", "median similarity", "share at 0.5 or more", "share at 0.7 or more"]], floatfmt=".3f")
log_lines = (ROOT / "data/structure_track/outputs/cyp3a4/logs/block_holdout.log").read_text().splitlines()
start = next(i for i, line in enumerate(log_lines) if line.strip().startswith("candidate") and "block_auc" in line)
end = next(i for i in range(start + 1, len(log_lines)) if not log_lines[i].strip())
block_table = pd.read_csv(StringIO("\\n".join(log_lines[start:end])), sep=r"\\s+")
rho_block = float(spearmanr(block_table["cv_auc_unweighted"], block_table["block_auc"]).statistic)
rho_stored = float(re.search(r"block AUC = ([\\d.]+)", "\\n".join(log_lines)).group(1))
best_by_cv = block_table.sort_values("cv_auc_unweighted", ascending=False).iloc[0]
size_line = next(line for line in log_lines if "held-out block rows" in line)
committee_line = next(line for line in log_lines if line.startswith("equal-weight rank committee"))
show_table(block_table.head(6)[["candidate", "cv_auc_unweighted", "block_auc", "block_auc_lo", "block_auc_hi"]], floatfmt=".4f")
print(f"{len(block_table)} candidates; Spearman rho between the cross-validation AUC and the block AUC, recomputed from the table printed to four decimals: {rho_block:.3f}; the run itself, from unrounded values: {rho_stored:.3f}")
print(size_line.split(' ', 2)[2]); print(committee_line)
block_rank_of_cv_best = int((block_table["block_auc"] > best_by_cv["block_auc"]).sum() + 1)
mean_offset = float((block_table["block_auc"] - block_table["cv_auc_unweighted"]).mean())
block_rate = float(re.search(r"positive rate ([\\d.]+)", size_line).group(1))
n_block, n_train_block = (int(x) for x in re.search(r"train rows (\\d+), held-out block rows (\\d+)", size_line).groups()[::-1])
n_block_positives = round(n_block * block_rate)
own_split, test_split = split_table.iloc[0], split_table.iloc[1]
'''

LEAK_TOY = '''
# Why a label-permutation control can catch a leak, shown on data where there is nothing to find. 200 compounds with 2,000 pure-noise features and random labels: any honest score must be 0.5. Two pipelines, both
# "select the 20 features most correlated with the label, fit a logistic regression, score by 5-fold cross-validation": the leaky one selects the features once on all compounds, before the folds are made; the
# clean one selects them again inside each training part. Each is run on the true (random) labels and on 20 random permutations of them.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

def top_features(X, y, k=20):
    centred = X - X.mean(axis=0)
    score = np.abs(centred.T @ (y - y.mean()))
    return np.argsort(score)[-k:]

def cross_validated_auc(X, y, leaky, seed=0):
    if leaky:
        X = X[:, top_features(X, y)]
    out = np.zeros(len(y))
    for fit_rows, held in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        columns = slice(None) if leaky else top_features(X[fit_rows], y[fit_rows])
        out[held] = LogisticRegression(max_iter=300).fit(X[fit_rows][:, columns], y[fit_rows]).decision_function(X[held][:, columns])
    return roc_auc_score(y, out)

rng_leak = np.random.default_rng(5)
X_noise = rng_leak.normal(size=(200, 2000))
y_noise = rng_leak.integers(0, 2, 200)
leak_rows = []
for label, leaky in (("selects the features before the folds (leaky)", True), ("selects the features inside each training part (clean)", False)):
    permuted = [cross_validated_auc(X_noise, rng_leak.permutation(y_noise), leaky) for _ in range(20)]
    leak_rows.append({"pipeline": label, "AUC on the true labels": cross_validated_auc(X_noise, y_noise, leaky), "mean AUC on 20 permuted label sets": float(np.mean(permuted)), "range over the permutations": f"{min(permuted):.3f} to {max(permuted):.3f}"})
show_table(pd.DataFrame(leak_rows), floatfmt=".3f")
leaky_auc, clean_auc = leak_rows[0]["AUC on the true labels"], leak_rows[1]["AUC on the true labels"]
leaky_permuted_mean, clean_permuted_mean = leak_rows[0]["mean AUC on 20 permuted label sets"], leak_rows[1]["mean AUC on 20 permuted label sets"]
'''

LEAKAGE = '''
# The leakage controls of the project, as checks that can fail. (1) The negative control: the whole nested pipeline run with the labels of the pool permuted at random must give chance; its stored result is read
# back and the verdict is recomputed from the numbers with the gate's own tolerances. (2) The feature blocks take structures and nothing else: every block function of the feature store is listed with the names of its
# parameters, and a block that took a label, a fold or a table of results would show here. (3) The guard on the laboratory's releases: rows of a blinded test compound are removed, shown on a made-up frame.
import inspect

from cypmode.challenge.feature_store import BLOCKS as FEATURE_BLOCKS
from cypmode.challenge.official_extra import exclude_test_compounds, test_compounds

control_text = (ROOT / "data/structure_track/outputs/cyp3a4/logs/permutation_control_CYP3A4.log").read_text()
control = {z0: dict(zip(("real_auc", "real_mcc", "perm_auc", "perm_mcc"), map(float, vals))) for z0, *vals in re.findall(r"z0=(1\\.0|1\\.96)\\s+real labels: AUC ([\\d.]+), MCC ([+-][\\d.]+) \\| permuted: AUC ([\\d.]+) \\(range [\\d.]+ to [\\d.]+\\), MCC ([+-][\\d.]+)", control_text)}
control_rows = [{"cut-off": f"z0 = {z0}", "real labels: AUC": c["real_auc"], "permuted: AUC": c["perm_auc"], "real labels: MCC": c["real_mcc"], "permuted: MCC": c["perm_mcc"]} for z0, c in control.items()]
show_table(pd.DataFrame(control_rows), floatfmt=".3f")
control_met = bool(all(abs(c["perm_auc"] - 0.5) <= 0.04 and c["real_auc"] - c["perm_auc"] >= 0.1 and abs(c["perm_mcc"]) <= 0.08 for c in control.values()))
stored_control_met = "GATE (negative control, permuted labels give chance): MET" in control_text
parameters = pd.DataFrame([{"feature block": name, "parameters of its function": ", ".join(inspect.signature(function).parameters)} for name, function in sorted(FEATURE_BLOCKS.items())])
only_structures = bool((parameters["parameters of its function"] == "mols").all())
n_blocks_checked = len(parameters)
n_permutations = int(re.search(r"(\\d+) permutations", control_text).group(1))
ids = sorted(test_compounds())
made_up = pd.DataFrame({"ocnt_batch": [f"{ids[0]}-AA-001", "OCNT-0000001-AA-001", f"{ids[1]}-AB-002", "OCNT-0000002-AA-001"], "value": [1.0, 2.0, 3.0, 4.0]})
kept, dropped = exclude_test_compounds(made_up, "ocnt_batch")
guard_works = bool(len(kept) == 2 and len(dropped) == 2 and set(kept["value"]) == {2.0, 4.0})
print(f"negative control recomputed: {'MET' if control_met else 'NOT MET'} (stored verdict: {'MET' if stored_control_met else 'NOT MET'}); {n_blocks_checked} feature blocks, every one takes only the structures: {only_structures}; the guard keeps {len(kept)} of 4 made-up rows and drops {len(dropped)} blinded compounds: {guard_works}")
real_auc_control, permuted_auc_control = control["1.0"]["real_auc"], control["1.0"]["perm_auc"]
'''

FORKING = '''
# What fixing the rule in advance protects against. For an addition that truly adds nothing, the signed z-value of its gain at the two cut-offs is a pair of correlated standard normal numbers. Their correlation is
# measured on the stacked comparisons of the project's logs (every addition, both cut-offs: the family of 6.5); pairs are then simulated under "no true gain" and three rules are applied to them: one cut-off fixed
# beforehand, the gate of the project (a positive gain with p < 0.05 at both cut-offs), and "report whichever cut-off looks better" (p < 0.05 at either).
from scipy.stats import norm

signed_z = []
for path in sorted(LOG_DIR.glob("*CYP3A4.log")):
    found = {z0: (float(gain), float(p)) for z0, gain, p in stacked_line.findall(path.read_text(errors="ignore"))}
    if "1.0" in found and "1.96" in found:
        signed_z.append([np.sign(found[z][0]) * norm.isf(max(found[z][1], 1e-12) / 2) for z in ("1.0", "1.96")])
signed_z = np.array(signed_z)
rho_cuts = float(np.corrcoef(signed_z.T)[0, 1])
null_z = np.random.default_rng(3).multivariate_normal([0.0, 0.0], [[1.0, rho_cuts], [rho_cuts, 1.0]], size=400_000)
limit = norm.isf(0.025)                                                  # a positive gain whose two-sided p-value is below 0.05
pass_rates = {"one cut-off, fixed beforehand": float((null_z[:, 0] > limit).mean()), "both cut-offs (the gate)": float((null_z > limit).all(axis=1).mean()), "either cut-off, chosen after looking": float((null_z > limit).any(axis=1).mean())}
fork_table = pd.DataFrame({"rule": list(pass_rates), "share of additions with no true gain that pass": list(pass_rates.values())})
fork_table["real additions of the logs that pass"] = [int((signed_z[:, 0] > limit).sum()), int((signed_z > limit).all(axis=1).sum()), int((signed_z > limit).any(axis=1).sum())]
show_table(fork_table, floatfmt=".4f")
n_family_pairs = len(signed_z)
print(f"correlation of the signed z-values at the two cut-offs over {n_family_pairs} logged comparisons: {rho_cuts:.2f}")
'''

GATE_LEDGER = '''
# The gates of the project, counted from its logs. An experiment prints a line "GATE ...: MET" or "... NOT MET" when it ends. All such lines of the project's log directory are collected and sorted into three
# groups: the negative controls (for which MET is the good outcome), the structure track, and the additions to the TDI stack (alone or stacked).
gate_line = re.compile(r"GATE\\b.*:\\s*(NOT MET|MET)\\s*$")
ledger_rows = []
for path in sorted(LOG_DIR.glob("*.log")):
    for line in path.read_text(errors="ignore").splitlines():
        found = gate_line.search(line)
        if found:
            group = "negative controls" if "negative control" in line else ("structure track" if "STRUCTURE GATE" in line else "TDI stack additions")
            ledger_rows.append({"log": path.name, "line": line.strip(), "group": group, "verdict": found.group(1)})
ledger = pd.DataFrame(ledger_rows)
by_group = ledger.groupby(["group", "verdict"]).size().unstack(fill_value=0).reindex(columns=["MET", "NOT MET"], fill_value=0).reset_index()
show_table(by_group)
met_lines = ledger[(ledger["verdict"] == "MET") & (ledger["group"] != "negative controls")]
show_table(met_lines[["log", "line"]].assign(line=met_lines["line"].str.replace(r"\\s+", " ", regex=True)))
n_ledger_lines, n_ledger_logs = len(ledger), ledger["log"].nunique()
n_improve = int((ledger["group"] != "negative controls").sum())
n_improve_met = int(((ledger["group"] != "negative controls") & (ledger["verdict"] == "MET")).sum())
n_control_met = int(((ledger["group"] == "negative controls") & (ledger["verdict"] == "MET")).sum())
n_control = int((ledger["group"] == "negative controls").sum())
'''

REPLICATE = '''
# The 10-fold replicate of the project, read from the logs of the experiments through the module that decides it (cypmode/challenge/replicate_verdict.py): for each addition put through it, the stacked comparison
# in the main design (5 outer folds x 3 repeats) and in the replicate (10 outer folds x 3 repeats), at both cut-offs, and the verdict: an addition is replicated only if the replicate gives p < 0.05 and a positive
# gain at both cut-offs AND the main design has a positive gain at both cut-offs.
from cypmode.challenge.replicate_verdict import GROUPS as REPLICATE_GROUPS
from cypmode.challenge.replicate_verdict import LOGS as REPLICATE_LOGS
from cypmode.challenge.replicate_verdict import read_stacked, verdict

replicate_rows, replicate_values = [], {}
for group, (five_name, ten_name) in REPLICATE_GROUPS.items():
    five, ten = read_stacked(REPLICATE_LOGS / f"{five_name}_CYP3A4.log"), read_stacked(REPLICATE_LOGS / f"{ten_name}_CYP3A4.log")
    replicate_values[group] = (five, ten)
    row = {"addition": group}
    for z0 in (1.0, 1.96):
        row[f"5 folds, z0 {z0}"] = f"{five[z0][0]:+.4f} (p={five[z0][1]:.3f})" if z0 in five else "no result"
        row[f"10 folds, z0 {z0}"] = f"{ten[z0][0]:+.4f} (p={ten[z0][1]:.3f})" if z0 in ten else "no result"
    row["replicated"] = ("yes" if verdict(five, ten) else "no") if len(ten) == 2 else "not run yet"
    replicate_rows.append(row)
replicate_table = pd.DataFrame(replicate_rows)
show_table(replicate_table)
n_replicate_groups = len(replicate_table)
n_replicated = int((replicate_table["replicated"] == "yes").sum())
n_replicate_pending = int((replicate_table["replicated"] == "not run yet").sum())
n_not_replicated = int((replicate_table["replicated"] == "no").sum())
neural_five, neural_ten = replicate_values["neural"]
neural_status = (f"the 10-fold design gives {neural_ten[1.0][0]:+.4f} (p = {neural_ten[1.0][1]:.3f}) at z0 = 1.0 and {neural_ten[1.96][0]:+.4f} (p = {neural_ten[1.96][1]:.3f}) at z0 = 1.96"
                 if len(neural_ten) == 2 else "its 10-fold run had not finished when this page was built")
compact_five, compact_ten = replicate_values["physics_compact"]
'''


CELLS = [
    md("""
## 6.6 Why Tukey's HSD was dropped, and how the older results were recomputed

> **Field:** Statistics

*Where this section sits:* 6.5 derived the test the project uses now. It was not the project's first choice. The earliest significance experiments (Part VII reports what each of them tested) compared their variants with **Tukey's HSD**, the standard method for comparing several groups at once, and the method answers a question that is subtly different from the one being asked. This section explains what the method assumes, recomputes the old analyses from their saved scores with the corrected test of 6.5 to show where the two disagree, and then examines one more shortcut of the same family: resampling the compounds while the predictions stay fixed. Nothing here is a new model. Every number comes from scores that were saved when the experiments ran.

"""),
    code(SCORES_COUNT),
    md("""
**What the method does.** An experiment of this kind has a few variants of a model (the base model alone, the base model with one added block of features, and so on) and, for each variant and each enzyme, the MCC (section 4.10) on every validation fold of a repeated cross-validation: here {{=scores_per_variant}} scores, from {{=n_repeats_saved}} repeats of {{=n_folds_saved}} folds. Tukey's HSD (the "honestly significant difference" of Tukey, 1949) asks, for every pair of variants, whether the difference between their mean scores is larger than chance explains, and it guards against the many-comparisons problem of 6.5 by widening its yardstick when more variants are compared. A pair is called different when

$$|\\bar x_a - \\bar x_b| \\;>\\; q_{\\alpha,\\,k,\\,N-k}\\,\\sqrt{\\frac{\\mathrm{MS}_{\\text{within}}}{n}},$$

where $\\bar x_a$ and $\\bar x_b$ are the mean scores of the two variants, $k$ is the number of variants, $n$ the number of scores per variant, $N = kn$ the number of scores in all, $\\mathrm{MS}_{\\text{within}}$ the average over the variants of the variance of their scores around their own mean, $\\alpha$ the accepted false-alarm probability (0.05), and $q$ a critical value of the *studentized range distribution*, the distribution of the largest difference among $k$ means in units of their standard error. The right-hand side is a standard error of a mean, $\\sqrt{\\mathrm{MS}_{\\text{within}}/n}$, made wider by $q$ for the number of pairs. That standard error treats the $n$ scores of a variant as $n$ independent observations: the mean of {{=scores_per_variant}} of them is taken to be known {{=scores_per_variant ** 0.5:.0f}} times more precisely than one of them.

They are not independent, for the reason given in 6.5: the folds of one repeat share most of their training compounds, and the five repeats are five arrangements of the same compounds. The method also does not use the pairing of the scores: the fold in which one variant is scored is the very fold in which another is scored, so a hard fold lowers every variant together, and an unpaired method counts that shared variation as noise in each of them. The first fault makes the method too ready to find differences, the second makes it too slow to find them, and which dominates depends on the data. The cell recomputes both analyses on the saved scores: Tukey's HSD with the library, and the corrected paired t with Holm's adjustment with the function of 6.5, for each pair of variants and each enzyme.
"""),
    code(TUKEY_VS_PAIRED),
    md("""
Over the {{=len(comparison_names)}} experiments there are {{=n_variants}} variants, hence {{=n_variants * (n_variants - 1) // 2}} pairs per enzyme and {{=n_pairs_all}} pairs in all. Tukey's HSD calls {{=n_tukey_all}} of them different; the corrected paired test calls {{=n_paired_all}}. The {{=n_flipped}} pairs that Tukey calls different and the corrected test does not all belong to {{='one experiment' if len(flip_experiments) == 1 else str(len(flip_experiments)) + ' experiments'}} ({{=', '.join(flip_experiments)}}), and the second table lists them. The one that matters most is the CYP3A4 row that compares M3 with M3+ChemBERTa: adding ChemBERTa embeddings (section 7.9) to the base features changes the mean MCC by {{=example_diff:+.3f}}, which Tukey reports at p = {{=example_tukey_p:.4f}} and the corrected paired test at p = {{=example_paired_p:.2f}}. The direction of the finding is unchanged, since ChemBERTa did not help and the project did not use it. What changes is the sentence that the data allow. "Significantly worse" was overstated; what is supported is "no sign of a gain, and a loss of about {{=abs(example_diff):.2f}} MCC that this design cannot tell from chance". The files written when the experiments ran agree with the recomputation: {{=files_agree}}.

**How the older results were recomputed.** The experiments saved the MCC of every variant on every (repeat, fold) in a `*_significance_raw.csv` file, and that choice is what made a correction possible without refitting a single model. The module `cypmode/challenge/recompute_paired_summaries.py` reads each raw table, groups it by enzyme, runs `paired_comparison_summary` and writes `*_significance_paired.csv` next to it; the Tukey files are kept, so that the two analyses can be set side by side as above. The lesson for any experiment of this kind is to save the per-fold scores and not only a summary of them: a statistic that later proves wrong can then be replaced.

**Resampling the compounds of fixed predictions.** A second shortcut is tempting once the folds have been run and their predictions saved. The out-of-fold predictions of a model are a table with one score per compound; resampling the compounds of that table with replacement (a bootstrap, section 6.5) and recomputing the AUC difference of two models on each resample gives a distribution, and the 2.5th and 97.5th percentiles of it give an interval. The interval is real, but it answers a narrower question. It holds the fitted models fixed and asks how much the difference would move if the *validation compounds* were different. The corrected test asks how much it would move if the *whole procedure* were run again, with different training compounds as well, and training compounds change the model. The cell makes the comparison for two members of the stack, the deep and the shallow LightGBM on feature set A with fingerprints (section 8.1): a bootstrap of 2,000 resamples on the frozen predictions of the first repeat, and the corrected interval from the 15 outer folds.
"""),
    code(FROZEN_BOOTSTRAP),
    md("""
The two members differ in AUC by {{=fold_differences.mean():+.4f}} on average over the 15 folds. The bootstrap interval is {{=boot_low:+.4f}} to {{=boot_high:+.4f}}, of width {{=boot_high - boot_low:.4f}}; the corrected interval is {{=fold_differences.mean() - half_width:+.4f}} to {{=fold_differences.mean() + half_width:+.4f}}, of width {{=2 * half_width:.4f}}, {{=width_ratio:.1f}} times wider, and its p-value is {{=corrected_p:.2f}}. Neither interval excludes zero, so the decision would be the same here, and one pair of models is an example and not a proof. The direction of the difference in width is what the argument predicts: the narrower interval leaves out the variation that comes from refitting on other training compounds. It is why the project does not use a bootstrap of saved predictions as the *test* between two models. The bootstrap has a different, legitimate use in the project, to put an interval around a *single* number on one fixed set of predictions (the AUC of a final model on a hold-out block in 6.8, the smoothing of a threshold in section 8.4), where the question is exactly how much that number depends on which compounds were scored.

*On what actually ran.* `cypmode/challenge/stacking_significance.py` holds `repeated_cv_stacking_comparison`, which produced the raw tables (five repeats of five folds, with the nearest-neighbour label features recomputed inside every fold), and `paired_comparison_summary`, the corrected paired t with Holm's adjustment, its `meandiff` being the second variant minus the first. `cypmode/challenge/recompute_paired_summaries.py` applies it to the raw tables of the older experiments. The Tukey tables are the ones the experiments wrote, and the cell above recomputes them with `statsmodels` as a check. The notebook recomputes everything in this section from the saved scores and the cache of the nested plan; it fits nothing.
"""),
    md("""
## 6.7 Pre-declared gates: the rule, the floor of the design, and the replicate

> **Field:** Statistics · Data science

*Where this section sits:* 6.4 gave a yardstick and 6.5 a test. What is still missing is a rule for what to do with the outcome: which comparisons count, at what strength of evidence, and who decides before the outcome is known. The rule the project uses for every proposed addition to its model is a gate, the pass/fail rule fixed before a result exists, and this section states it exactly, shows why it is written in advance, reproduces the verdict of four additions from the cache, counts how the gates of the whole project came out, asks how small a gain the design could ever have recognised, and describes the second design, with ten outer folds, that is used to confirm a result when the first design's floor is too high.

**The rule.** A new block of features or a new family of models is *an addition* to the stack of 23 default members (section 8.1). It is judged in two ways. In the *stacked* comparison, the stack of the 23 default members plus the addition is compared with the stack of the 23 alone: both go through the same nested plan (6.3), with the same selector (the logistic stack of section 8.2) and the weighted yardstick of 6.4, and the 15 differences of the outer scores are tested with the corrected t of 6.5. In the *alone* comparison, a single model with the addition is compared with the same model without it. An addition **passes the gate** when its mean difference is positive and the corrected p-value is below 0.05 at *both* confidence cut-offs, $z_0 = 1.0$ and $z_0 = 1.96$ (section 6.4: the cut-off decides which compounds count as scored). Both the level and the cut-offs are constants of the code (`GATE_ALPHA` and `Z_CUTS` in `cypmode/challenge/addition_experiment.py`), set before any addition was run, and the last line an experiment prints is its verdict, MET or NOT MET.

**Why the rule is fixed before the results.** The reason is statistical and it can be measured. Consider an addition that truly adds nothing. Its gain at the two cut-offs is then two noisy numbers that are almost the same noise, since the two yardsticks share most of their compounds. If the analyst may choose which cut-off to report after seeing both, the chance of a false alarm is larger than the 5 percent the test promises; if both must pass, it is smaller. The cell measures the correlation between the two cut-offs on the project's own logged comparisons, simulates additions with no true gain that have that correlation, and counts how many pass under three rules.
"""),
    code(FORKING),
    md("""
The signed z-values of the logged comparisons are correlated at {{=rho_cuts:.2f}} between the two cut-offs. For an addition with no true gain, a single cut-off fixed beforehand lets {{=pass_rates['one cut-off, fixed beforehand']:.1%}} through, the 2.5 percent that a positive difference with a two-sided p-value below 0.05 gives by chance; the gate, which asks for both, lets {{=pass_rates['both cut-offs (the gate)']:.1%}} through; and choosing the better cut-off after looking lets {{=pass_rates['either cut-off, chosen after looking']:.1%}} through, {{=pass_rates['either cut-off, chosen after looking'] / pass_rates['one cut-off, fixed beforehand']:.1f}} times as many as the fixed one. The two cut-offs are only one of several ways in which a result can be looked at. The metric (AUC or average precision), the design (five or ten outer folds) and the set of compounds are others, and each extra way multiplies the chances in the same fashion. A rule written down before the first run removes the freedom, and with it the need to estimate how much of it was used. Of the {{=n_family_pairs}} logged comparisons, the number that pass the gate is {{=int(fork_table['real additions of the logs that pass'].iloc[1])}}, against {{=int(fork_table['real additions of the logs that pass'].iloc[2])}} that would have passed under the choose-afterwards rule.

**The gate on four real additions.** The next cell reproduces the verdict of four additions from the cache of the nested plan and sets it against what the project's own experiment logged when it ran: the multitask network (`mlp`), the same network with the organisers' extra readings as auxiliary targets (defined in 9.10; `mlp_aux`), the pretrained tabular foundation model TabICL, and plain Chemprop. For each, the 23 default members plus the addition are compared with the 23 alone, at both cut-offs, with the weighted yardstick. The standard error of each comparison is converted back into the standard deviation of the 15 fold differences, which the design-floor calculation below needs.
"""),
    code(GATE_REPRODUCTION),
    md("""
Every p-value recomputed here equals the one in the log of its experiment: {{=agrees_with_logs}}. {{=n_gate_met}} of the four additions meet the gate. The best of them, `mlp`, reaches p = {{=reproduced.loc[reproduced['addition'] == 'mlp', 'p z0 1.0'].iloc[0]:.3f}} at $z_0 = 1.0$ and misses at $z_0 = 1.96$ with p = {{=reproduced.loc[reproduced['addition'] == 'mlp', 'p z0 1.96'].iloc[0]:.3f}}; with a gain of {{=reproduced.loc[reproduced['addition'] == 'mlp', 'gain z0 1.0'].iloc[0]:+.3f}} AUC it is also the kind of result that would have been reported as an improvement under the choose-afterwards rule. The median standard deviation of the 15 fold differences over the eight comparisons is {{=sd_5:.4f}} AUC.

**The gates of the whole project.** Every experiment ends with such a line, and the lines can be counted. The cell collects all of them from the project's log directory and sorts them into **negative controls** (runs on data for which the right answer is known to be nothing, section 6.8; for them MET is the good outcome), the structure track (Part XI), and additions to the TDI stack.
"""),
    code(GATE_LEDGER),
    md("""
The logs hold {{=n_ledger_lines}} gate lines from {{=n_ledger_logs}} logs. Of the {{=n_improve}} that judge a proposed improvement (additions to the stack and levers of the structure track), {{=n_improve_met}} are MET, the ones listed in the second table, and of the {{=n_control}} negative-control lines {{=n_control_met}} are MET, as they should be. Many of the lines are repeats of one experiment (an addition has an alone and a stacked line for each enzyme, and the ten-fold replicate repeats the stacked ones), so the count says how often the gate was asked and not how many independent ideas were tried. It is still the right picture of what a strict pre-declared rule does to a research programme with many ideas: most of them do not clear it, and the few that do are the ones the project builds on. A claim that something helps, in this project, is the exception that survived a test it could have failed.

**What the design could ever have recognised.** A test that fails to reject does not show that nothing is there. It shows that the design could not see it, and whether it could have is a calculation, the power of the test (6.5): the probability that it rejects when the effect is real. The **minimum detectable effect** is the smallest true difference that the test rejects with a stated probability, here 0.8, the conventional "8 times in 10". The test of 6.5 rejects when $|\\bar d| > t_{0.975,\\,J-1} \\cdot \\mathrm{SE}$ with $\\mathrm{SE} = s_d\\sqrt{1/J + n_{\\text{test}}/n_{\\text{train}}}$, where $s_d$ is the standard deviation of the $J$ fold differences. If the true difference is $\\delta$, the mean difference is about $\\delta$ with standard error SE, so it exceeds the critical value in 80 percent of experiments when $\\delta$ is larger than that value by the 0.8 quantile of the t distribution times SE:

$$\\mathrm{MDE} \\;\\approx\\; \\left(t_{0.975,\\,J-1} + t_{0.8,\\,J-1}\\right)\\, s_d \\sqrt{\\frac{1}{J} + \\frac{n_{\\text{test}}}{n_{\\text{train}}}}.$$

(The central t distribution is used for both quantiles; the exact calculation uses a non-central one and differs little at these degrees of freedom.) The standard deviation is measured and not assumed: for the main design it is the median over the eight comparisons just reproduced, and for the replicate described below it is recovered from the logs of the ten-fold experiments, each logged (gain, p-value) pair being turned back into a standard error. The third row of the table is a comparison: the floor that the main design would have if its 15 folds were independent, the assumption of the naive test of 6.5.
"""),
    code(DESIGN_FLOOR),
    md("""
The main design can recognise a true gain of {{=floor_main:.4f}} AUC or more, 8 times in 10, and nothing smaller. The replicate lowers the floor to {{=floor_replicate:.4f}}, and the number in the third row, {{=floor_naive:.4f}}, is the price of the correlation between folds: the honest test needs a true gain about {{=floor_main / floor_naive:.1f}} times larger than the naive test would claim to need. Set against this floor, the gains the project logged for its additions are small: the median of the {{=n_logged}} logged stacked gains at $z_0 = 1.0$ is {{=median_gain:+.4f}}, and the number of them above the floor of the main design is {{=n_above_main}}, above that of the replicate {{=n_above_replicate}}. This is the quantitative form of the statement made at the end of 6.5. An addition that really improves the AUC by 0.004 would be found {{=power_main:.0%}} of the time by the main design and {{=power_replicate:.0%}} of the time by the replicate. A failure to clear the gate is therefore a statement about the size of the effect: a gain much larger than about {{=floor_main:.2f}} AUC is unlikely, and a smaller one is neither shown nor excluded. It also fixes what the project can claim in the other direction. What clears the gate is, besides, more likely an overestimate than an underestimate of the true gain, since a result is selected because it came out large (the winner's curse); the replicate exists partly for that reason.

**The replicate.** When the gains in question are of the size of the main design's floor or below it, a second design is available, and it is declared in advance in the same way. The factor in the standard error is $1/J + n_{\\text{test}}/n_{\\text{train}}$. With 10 outer folds repeated three times, $J = 30$ and the ratio falls from $1/4$ to $1/9$, so the factor falls from 0.317 to 0.144: the standard error of the same difference would be about a third smaller if the folds were as noisy, and they are not, since each outer fold is a tenth of the pool and not a fifth, so that its score is noisier (the measured standard deviation of the fold differences is {{=sd_10:.4f}} against {{=sd_5:.4f}}). The replicate counts an addition as **replicated** only when two things hold: the ten-fold comparison gives p < 0.05 and a positive gain at both cut-offs, and the five-fold comparison gives a positive gain at both cut-offs, so that the replicate cannot rescue a result whose sign the main design contradicts. The rule for the physics blocks of Part X was written down before any ten-fold result existed, and the rule for the neural pair (section 9.9) was added on 2026-10-16, after its fifteen-fold result and before any ten-fold one; the module's docstring records both. The cell reads the logs through the module that decides.
"""),
    code(REPLICATE),
    md("""
The replicate covers {{=n_replicate_groups}} additions: {{=n_replicated}} replicated, {{=n_not_replicated}} did not{{=', and ' + str(n_replicate_pending) + ' had not been run when this page was built' if n_replicate_pending else ''}}. The cases show what the rule does. The block of compact physics features (`physics_compact`) comes close: at $z_0 = 1.96$ its ten-fold comparison is {{=compact_ten[1.96][0]:+.4f}} with p = {{=compact_ten[1.96][1]:.3f}}, below 0.05, but at $z_0 = 1.0$ it is p = {{=compact_ten[1.0][1]:.3f}}, and the rule asks for both, so it is not replicated. The rule can be read as too strict for such a case, and it was fixed before the case existed. The neural pair is the other kind of case: it cleared the main gate, so the replicate is the test that decides whether it is kept, and {{=neural_status}}. Sections 10.6 and 9.9 give the full account of the two.

*On what actually ran.* `cypmode/challenge/addition_experiment.py` runs one addition through the gate (it fills the cache for the candidates if asked, builds the stacked and the alone comparisons, prints one line per cut-off, and ends with the verdict line that the ledger above counts); `cypmode/challenge/cached_candidates.py` defines the groups of candidates an experiment adds (`GROUPS`); `cypmode/challenge/model_comparison.py` supplies `compare_candidate_sets`, which this notebook replaces with explicit `nested_evaluate` calls so that the fold differences can be seen; `cypmode/challenge/replicate_verdict.py` is the module whose `verdict` function decided the replicate above. The designs are the plans of 6.3, built with `--outer 5` for the main design and `--outer 10` and the root `nested_f10` for the replicate; the queue jobs that build and fill them are described in section 12.4.
"""),
    md("""
## 6.8 Checking that a split is informative, the test-like hold-out, and the leakage controls

> **Field:** Statistics · Cheminformatics

*Where this section sits:* every number of Part VI is a number on validation compounds, and it is worth something only if two things are true: the validation compounds resemble the ones the project is scored on, and they were never seen by the procedure that is being scored. 6.4 acted on the first by weighting. This section examines both from outside: it measures how far the project's folds are from the training set compared with the blinded test, scores the candidates on a block of the training file that was built like the test and held out whole, and then tests for the second condition, the absence of **data leakage**: information from the validation rows reaching the fit.

**Is the split informative?** A cross-validation score is a forecast of the score on new compounds only if the folds are a fair stand-in for them. A fold made of near-copies of training compounds is too easy and over-predicts; a fold of compounds unlike anything in training is too hard and under-predicts. The first stored result is the profile of the nearest-training similarity (Tanimoto, section 3.2) for the project's own cross-validation folds against their own training parts, and for the 750 blinded test compounds against the whole training file (`cypmode/challenge/split_validity.py`). The second stored result is a harder test of the same idea. The training file is not homogeneous: its compound numbers fall into blocks that were screened in different rounds (`cypmode/challenge/library_blocks.py`), and the highest block is an *analogue-expansion round*, built like the blinded test set, with a positive rate of about 0.30 for CYP3A4 like the test set's. Holding that whole block out and fitting everything on the rest gives a score in a regime that resembles the test far more than any random or similarity-weighted fold does, and the same candidates' cross-validation AUC can be set next to it (`cypmode/challenge/block_holdout.py`). The candidate set and the "best" candidate are chosen from the cross-validation only, never from the block, so the block stays a test. The cell reads both results and recomputes the agreement of the two yardsticks from the logged table.
"""),
    code(SPLIT_BLOCK),
    md("""
The median nearest-training similarity is {{=own_split['median_similarity']:.3f}} for the cross-validation folds and {{=test_split['median_similarity']:.3f}} for the blinded test; {{=own_split['frac_similarity_ge_0.5']:.0%}} of the fold compounds have a training neighbour at 0.5 or more, against {{=test_split['frac_similarity_ge_0.5']:.0%}} of the test compounds, but at 0.7 or more the order reverses ({{=own_split['frac_similarity_ge_0.7']:.0%}} against {{=test_split['frac_similarity_ge_0.7']:.0%}}). The test compounds are therefore typically analogues of training compounds at moderate similarity and seldom near-copies, and the folds are on this one count the harder exam. That is a statement about a single property, the distance to the nearest neighbour, and a necessary condition for a fair stand-in and not a sufficient one; the weights of 6.4 correct for it where they can.

On the held-out block of {{=n_block}} compounds, trained on the other {{=n_train_block}}, the candidate that cross-validation ranked first scores {{=best_by_cv['block_auc']:.3f}} AUC (interval {{=best_by_cv['block_auc_lo']:.3f}} to {{=best_by_cv['block_auc_hi']:.3f}}, a bootstrap over the block's compounds, the legitimate use named in 6.6) against {{=best_by_cv['cv_auc_unweighted']:.3f}} in cross-validation, and has rank {{=block_rank_of_cv_best}} of the {{=len(block_table)}} candidates on the block. The two yardsticks agree on the order of the candidates, with a Spearman correlation of {{=rho_block:.3f}} recomputed from the printed table (the run itself, from the unrounded values, printed {{=rho_stored:.3f}}). On the *level* the block is {{='kinder' if mean_offset > 0 else 'harder'}}: its AUC exceeds the cross-validation AUC by {{=mean_offset:+.3f}} on average over the candidates. That fits a block whose compounds have closer training neighbours than a random fold has, but this notebook has not measured that, and the block differs from a random fold in other ways too (it is one round of one chemistry), so the offset is reported and not explained. The equal-weight committee of all candidates scores about the same as the single best, and the two intervals overlap almost completely, so the block cannot tell a good selection from a committee, a limit of {{=n_block}} compounds with about {{=n_block_positives}} positives. What the block does show is that the cross-validation was a faithful guide to the *order* of the candidates in a regime like the test, which is the use the project makes of it.

**Leakage.** A pipeline leaks when the label of a validation compound, or something computed from it, has influenced the model that scores it. The result is a score that is high for a reason that will not repeat on new compounds. A well-known case is selecting features before cross-validation (Ambroise and McLachlan, 2002, showed how large its effect is on gene-expression data): if the 20 features most correlated with the label are chosen on all the compounds and only the model is cross-validated, the chosen features have already seen every validation label. The cell shows it on data in which there is nothing to find: 200 compounds, 2,000 features of pure noise, random labels. Two pipelines are identical except for the place of the selection, and each is run on the true (random) labels and on 20 permutations of them. A label permutation destroys any real relationship while keeping everything else, so a procedure without leakage must score 0.5 on permuted labels, and a procedure that leaks does not.
"""),
    code(LEAK_TOY),
    md("""
The leaky pipeline scores {{=leaky_auc:.2f}} AUC on labels that are random, and {{=leaky_permuted_mean:.2f}} on average on permuted ones: it "finds" a signal in noise that is not there, and it would find it on any labels. The clean pipeline, with the selection inside each training part, scores {{=clean_auc:.2f}} on the same labels and {{=clean_permuted_mean:.2f}} on average on the permuted ones, which is chance. The permuted-label score is the diagnostic: it needs no knowledge of where the leak is. This is the **negative control** of the project, a run on data for which the right answer is known to be "nothing" (Leach, *Molecular Modelling*, 2nd ed., section 12.13.1, describes it for regression models as a randomisation test), and the project applies it to the *whole* nested pipeline: inner folds, selector, outer folds, and the yardstick of 6.4.

The next cell reads the stored result of the control, and checks it against the tolerances written into the experiment. The gate is that the permuted-label AUC is within 0.04 of 0.5 and at least 0.1 below the real-label AUC, and the permuted MCC within 0.08 of 0; the tolerances are about two sampling standard errors of one run, not tuned values. It also makes two structural checks. The first is that every feature block of the feature store is a function of the structures alone: the names of the parameters of all of them are listed, and a block that took a label, a fold or a table of results would show. The second is the guard on releases from the laboratory that measured the challenge: some of its files contain readings of compounds that are in the blinded test set, and the function `exclude_test_compounds` removes their rows before anything is computed, so that a model never reads a measurement of a test compound. The guard is shown on a made-up frame with two rows of blinded compounds and two of others, and the identifiers are not printed.
"""),
    code(LEAKAGE),
    md("""
On the real pipeline, the control gives {{=real_auc_control:.3f}} AUC for the real labels at $z_0 = 1.0$ and {{=permuted_auc_control:.3f}} for permuted ones, and the recomputed verdict is {{='MET' if control_met else 'NOT MET'}}, as the stored one is. All {{=n_blocks_checked}} feature blocks take only the structures: {{=only_structures}}. The guard keeps the rows of the other compounds and drops those of the blinded ones: {{=guard_works}}. Three limits belong with this. The parameter check is a necessary condition, since a function that takes only structures cannot read a label through its arguments, though it could read one from a global, and it is the permutation control that tests the whole pipeline, not the signature. The control was run for a few permutations ({{=n_permutations}} in the stored run), so its range is wide; the conclusion rests on the mean and on the tolerance, not on any one permutation. And the control cannot find leakage between the training file and the blinded test, a test compound that is also in the training file for example, because it never touches the test set; that is checked separately, by comparing the identifiers and the structures of the training file with those of the blinded test (section 8.6).

Features that use labels are computed inside the folds. The nearest-neighbour label features of 7.3 are recomputed for every fold from the training part only (`add_knn_label_features_for_fold`), and in the nested plan every fit of every candidate happens inside a task (6.3), which sees only its own training rows. With the controls of this section, the weights of 6.4, the test of 6.5 and the gate of 6.7, the measurement chain of Part VI is complete: Part VII now applies it, from the first model.

*On what actually ran.* `cypmode/challenge/split_validity.py` computes the similarity profile (it uses `tanimoto_ood_distance` of `cypmode/challenge/reliability.py`); `cypmode/challenge/block_holdout.py` fits the candidates on the training part, scores them on the block, and prints the table and the agreement that this notebook reads back; `cypmode/challenge/permutation_control.py` runs the negative control on six candidates, one repeat of five outer folds and three permutations, with its caches in a temporary directory that is discarded; `cypmode/challenge/official_extra.py` holds `exclude_test_compounds`, the guard, and `cypmode/challenge/feature_store.py` the feature blocks. The toy leak above is computed in the notebook.
"""),
]
