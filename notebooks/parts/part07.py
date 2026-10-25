# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part VII (first part): the challenge's own baseline reproduced (7.1), the staircase of feature blocks M0 to M3 (7.2) and nearest-neighbour read-across features (7.3)."""

from notebooks.notebook_builder import code, figure, md

BASELINE_DATA = '''
# What the challenge tutorial's baseline is given. It reads the training file, keeps every compound that has a label for at least one of the two enzymes (load_tdi_training_data), and trains one LightGBM
# classifier per enzyme on the RDKit descriptors of the compound. The table counts what that gives per enzyme and how many of the labelled rows were measured in both assay arms (section 2.3); the others have
# no direct-inhibition reading, a distinction that matters from 7.4 on.
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from cypmode.challenge.tdi_baseline import TDI_ISOFORMS, compute_descriptors, cross_validate_baseline, cross_validate_fixed_seed, load_tdi_training_data
from cypmode.challenge.tdi_targets import BOTH_ARMS, NO_DIRECT, training_view

COMPARISONS = ROOT / "data/tdi_track/outputs/comparisons"
train_file = pd.read_csv(ROOT / "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv")
tutorial_frame = load_tdi_training_data()
data_rows = []
for iso in TDI_ISOFORMS:
    view = training_view(train_file, iso)
    labelled = view[view["label"].notna()]
    data_rows.append({"enzyme": iso, "labelled compounds": len(labelled), "time-dependent inhibitors": int(labelled["label"].sum()), "share": float(labelled["label"].mean()),
                      "measured in both arms": int((labelled["population"] == BOTH_ARMS).sum()), "no direct reading": int((labelled["population"] == NO_DIRECT).sum())})
data_table = pd.DataFrame(data_rows).set_index("enzyme")
show_table(data_table.reset_index(), floatfmt=".3f")
n_tutorial = len(tutorial_frame)
'''

BASELINE_RUNS = '''
# The baseline run twice. First as the tutorial runs it: ten random 75/25 splits per enzyme, a fresh LightGBM on each, the 0.5 threshold. The splits are unseeded in the tutorial, so the seed of numpy is fixed here for the
# run to repeat. Then with the project's fixed-seed five-fold protocol (cross_validate_fixed_seed), which is the same on every run and is what later comparisons use. The stored tables of both are read next to the new runs.
described = compute_descriptors(tutorial_frame)                 # the descriptors of useful_rdkit_utils, as in the tutorial; about 40 s for the labelled compounds
np.random.seed(0)
tutorial_runs = cross_validate_baseline(described, n_iterations=10)
fixed_runs = cross_validate_fixed_seed(described)
stored_tutorial, stored_fixed = pd.read_csv(COMPARISONS / "baseline_reproduction.csv"), pd.read_csv(COMPARISONS / "baseline_fixed_seed.csv")
n_descriptors = len(described["descriptors"].iloc[0])
baseline_rows = []
for name, frame in (("tutorial protocol, run here", tutorial_runs), ("tutorial protocol, stored", stored_tutorial), ("fixed-seed five-fold, run here", fixed_runs), ("fixed-seed five-fold, stored", stored_fixed)):
    for iso in TDI_ISOFORMS:
        part = frame[frame["isoform"] == iso]
        baseline_rows.append({"protocol": name, "enzyme": iso, "splits": len(part), "MCC mean": part["mcc"].mean(), "MCC sd": part["mcc"].std(), "AUC mean": part["roc_auc"].mean(), "accuracy mean": part["accuracy"].mean()})
baseline_table = pd.DataFrame(baseline_rows)
show_table(baseline_table, floatfmt=".3f")
fixed_matches_stored = bool(np.allclose(fixed_runs.sort_values(["isoform", "fold"])["mcc"].to_numpy(), stored_fixed.sort_values(["isoform", "fold"])["mcc"].to_numpy()))
def baseline_value(protocol, iso, column):
    return float(baseline_table[(baseline_table["protocol"] == protocol) & (baseline_table["enzyme"] == iso)][column].iloc[0])
'''

THRESHOLD_VIEW = '''
# Why an AUC of 0.80 and an MCC of 0.29 belong together. The out-of-fold probabilities of the fixed-seed five-fold run, pooled, for each enzyme, and what the threshold of 0.5 does with them: how many compounds it calls
# positive against how many are positive, the accuracy against the accuracy of calling everything negative, and the MCC at every threshold from 0.02 to 0.9.
oof_probabilities, oof_labels = {}, {}
for iso in TDI_ISOFORMS:
    column = f"{iso}_is_TDI"
    sub = described.dropna(subset=[column]).reset_index(drop=True)
    y = sub[column].astype(int).to_numpy()
    X = np.stack(sub["descriptors"])
    p = np.zeros(len(y))
    for fit_rows, held in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        p[held] = LGBMClassifier(verbose=-1, random_state=42).fit(X[fit_rows], y[fit_rows]).predict_proba(X[held])[:, 1]
    oof_probabilities[iso], oof_labels[iso] = p, y
thresholds = np.linspace(0.02, 0.9, 89)
call_rows, mcc_curves = [], {}
for iso in TDI_ISOFORMS:
    p, y = oof_probabilities[iso], oof_labels[iso]
    curve = np.array([matthews_corrcoef(y, (p >= t).astype(int)) for t in thresholds])
    mcc_curves[iso] = curve
    called = p >= 0.5
    call_rows.append({"enzyme": iso, "positive share (truth)": y.mean(), "called positive at 0.5": called.mean(), "accuracy at 0.5": float((called == y).mean()), "accuracy of calling everything negative": 1 - y.mean(),
                      "AUC": roc_auc_score(y, p), "MCC at 0.5": matthews_corrcoef(y, called.astype(int)), "best MCC over the thresholds": float(curve.max()), "threshold of the best": float(thresholds[curve.argmax()])})
call_table = pd.DataFrame(call_rows).set_index("enzyme")
show_table(call_table.reset_index(), floatfmt=".3f")
fig, axis = plt.subplots(figsize=(6.6, 3.6))
for iso, colour in zip(TDI_ISOFORMS, (theme["blue"], theme["orange"])):
    axis.plot(thresholds, mcc_curves[iso], color=colour, lw=2.4, label=iso)
    axis.plot(thresholds[mcc_curves[iso].argmax()], mcc_curves[iso].max(), "o", color=colour, ms=7)
axis.axvline(0.5, color=theme["muted"], ls=":", lw=1.2)
axis.set(xlabel="threshold on the predicted probability", ylabel="MCC of the pooled out-of-fold calls", title="the baseline's MCC as the threshold moves")
axis.legend(frameon=False)
tidy(axis, theme)
fig.tight_layout()
'''

BOARD_OFFSET = '''
# The same baseline on the challenge's own board. The interim leaderboards (the full test set, frozen at the 2026-09-24 deadline; section 2.10) list the organisers' reference entry, "Official LightGBM baseline", which is
# this model trained on the whole training file and scored on the 750 blinded compounds. Its board MCC is set against the cross-validation estimates of the same model on the training file.
board_rows = []
for iso in TDI_ISOFORMS:
    board = pd.read_csv(ROOT / f"data/tdi_track/inputs/leaderboard_snapshots/interim_{iso}_2026-10-04.csv")
    reference = board[board["Model Report Link"] == "Official LightGBM baseline"].iloc[0]
    board_rows.append({"enzyme": iso, "rank on the interim board": int(reference["Rank"]), "entries": len(board), "board MCC": float(reference["MCC"]), "board MCC sd (bootstrap)": float(reference["MCC std"]),
                       "fixed-seed CV MCC": baseline_value("fixed-seed five-fold, run here", iso, "MCC mean"), "tutorial-protocol MCC": baseline_value("tutorial protocol, run here", iso, "MCC mean")})
board_table = pd.DataFrame(board_rows).set_index("enzyme")
board_table["board minus fixed-seed CV"] = board_table["board MCC"] - board_table["fixed-seed CV MCC"]
show_table(board_table.reset_index(), floatfmt=".3f")
'''

ABLATION_RERUN = '''
# The M0 to M3 staircase (cypmode/challenge/tdi_ablation.py). Four models, the same five folds and the same LightGBM; each adds one block of columns to the one before: M0 the RDKit descriptors, M1 adds the motif flag
# (any of the six heme-binding nitrogen motifs of section 5.1, one column), M2 adds the trialkylamine flag of section 5.2 (one column), M3 adds the Morgan fingerprint of section 3.1 (2,048 bits). The stored table is the
# run of 2026-09-26. The trialkylamine pattern has changed since (section 5.2: three more exclusions), so the staircase is run twice here: once with the pattern as it was on the day of the stored run, which must reproduce
# the stored table, and once with the pattern of today.
from unittest import mock

import cypmode.metrics.amines as amine_module
from rdkit import Chem

from cypmode.challenge.tdi_ablation import MODELS, add_ablation_features, run_ablation

PATTERN_OF_THE_STORED_RUN = {
    "tertiary_trialkylamine": Chem.MolFromSmarts("[NX3;H0;!$(NC=[O,S]);!$(N=*);!a;!$(N-a)]"),
    "secondary_trialkylamine": Chem.MolFromSmarts("[NX3;H1;!$(NC=[O,S]);!$(N=*);!a;!$(N-a)]"),
}
stored_ablation = pd.read_csv(COMPARISONS / "ablation_m0_m3.csv")
with mock.patch.object(amine_module, "_TRIALKYLAMINE_PATTERNS", PATTERN_OF_THE_STORED_RUN):
    staircase_then = add_ablation_features(described)
    ablation_then = run_ablation(staircase_then)
staircase_now = add_ablation_features(described)
ablation_now = run_ablation(staircase_now)
key = ["model", "isoform", "fold"]
then_matches_stored = float(np.abs(ablation_then.sort_values(key)["mcc"].to_numpy() - stored_ablation.sort_values(key)["mcc"].to_numpy()).max())
widths = {model: len(staircase_now[f"{model.lower()}_features"].iloc[0]) for model in MODELS}
motif_share = float(staircase_now["motif_flag"].map(lambda a: a[0]).mean())
amine_share_now, amine_share_then = (float(frame["amine_flag"].map(lambda a: a[0]).mean()) for frame in (staircase_now, staircase_then))
rows = []
for model in MODELS:
    for iso in TDI_ISOFORMS:
        pick = lambda frame: frame[(frame["model"] == model) & (frame["isoform"] == iso)]["mcc"]
        rows.append({"model": model, "columns": widths[model], "enzyme": iso, "stored (pattern of that day)": pick(stored_ablation).mean(), "re-run, pattern of that day": pick(ablation_then).mean(), "re-run, pattern of today": pick(ablation_now).mean(), "sd over folds (today)": pick(ablation_now).std()})
ablation_table = pd.DataFrame(rows)
show_table(ablation_table, floatfmt=".3f")
print(f"largest difference between the re-run with the old pattern and the stored table: {then_matches_stored:.1e}; amine flag set for {amine_share_then:.1%} of the compounds with the old pattern, {amine_share_now:.1%} with today's; motif flag for {motif_share:.1%}")
'''

ABLATION_TESTS = '''
# Which of the steps could the five folds recognise? The paired differences between consecutive models, fold by fold, with the corrected resampled t of 6.5 (J = 5 folds, n_test / n_train = 1/4), for the stored table and for
# the re-run with today's pattern, and the smallest true difference this design could have found (section 6.7: 8 times in 10).
from scipy import stats

from cypmode.challenge.model_comparison import corrected_resampled_t

def step_tests(frame, label):
    out = []
    for iso in TDI_ISOFORMS:
        for before, after in (("M0", "M1"), ("M1", "M2"), ("M2", "M3")):
            a = frame[(frame["model"] == after) & (frame["isoform"] == iso)].sort_values("fold")["mcc"].to_numpy()
            b = frame[(frame["model"] == before) & (frame["isoform"] == iso)].sort_values("fold")["mcc"].to_numpy()
            result = corrected_resampled_t(a - b, 4, 1)
            out.append({"table": label, "enzyme": iso, "step": f"{before} to {after}", "mean MCC change": result.mean_difference, "p (corrected t)": result.p_value, "folds in which it rose": f"{result.n_better} of {result.n_folds}", "sd of the fold differences": float(np.std(a - b, ddof=1))})
    return out
step_table = pd.DataFrame(step_tests(stored_ablation, "stored") + step_tests(ablation_now, "today"))
show_table(step_table, floatfmt=".3f")
median_sd_folds = float(step_table["sd of the fold differences"].median())
mde_five_folds = float((stats.t.ppf(0.975, 4) + stats.t.ppf(0.8, 4)) * median_sd_folds * np.sqrt(1 / 5 + 1 / 4))
stored_amine_step = step_table[(step_table["table"] == "stored") & (step_table["enzyme"] == "CYP2D6") & (step_table["step"] == "M1 to M2")].iloc[0]
today_amine_step = step_table[(step_table["table"] == "today") & (step_table["enzyme"] == "CYP2D6") & (step_table["step"] == "M1 to M2")].iloc[0]
print(f"smallest true MCC change that five folds find 8 times in 10, with the median fold sd {median_sd_folds:.3f}: {mde_five_folds:.3f}")
'''

PARTITION_NOISE = '''
# One partition of the compounds into five folds is one draw. The same comparison of the amine step (M1 against M2, today's pattern) on five different partitions, five folds each (the 5 x 5 design of the later
# experiments, section 6.6): the mean change of MCC within each partition, and the corrected resampled t over all 25 folds.
rng_partitions = [100 + k for k in range(5)]
partition_rows = []
for iso in TDI_ISOFORMS:
    column = f"{iso}_is_TDI"
    sub = staircase_now.dropna(subset=[column]).reset_index(drop=True)
    y = sub[column].astype(int).to_numpy()
    features = {m: np.stack(sub[f"{m.lower()}_features"]) for m in ("M1", "M2")}
    for repeat, seed in enumerate(rng_partitions):
        for fold, (fit_rows, held) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(sub, y)):
            for m, X in features.items():
                p = LGBMClassifier(verbose=-1, random_state=0).fit(X[fit_rows], y[fit_rows]).predict_proba(X[held])[:, 1]
                partition_rows.append({"enzyme": iso, "repeat": repeat, "fold": fold, "model": m, "mcc": matthews_corrcoef(y[held], (p >= 0.5).astype(int))})
partition_scores = pd.DataFrame(partition_rows)
partition_summary = []
for iso in TDI_ISOFORMS:
    part = partition_scores[partition_scores["enzyme"] == iso]
    m1, m2 = (part[part["model"] == m].sort_values(["repeat", "fold"])["mcc"].to_numpy() for m in ("M1", "M2"))
    per_repeat = (part[part["model"] == "M2"].groupby("repeat")["mcc"].mean() - part[part["model"] == "M1"].groupby("repeat")["mcc"].mean()).to_numpy()
    result = corrected_resampled_t(m2 - m1, 4, 1)
    partition_summary.append({"enzyme": iso, "change of MCC in each of the 5 partitions": ", ".join(f"{v:+.3f}" for v in per_repeat), "lowest": float(per_repeat.min()), "highest": float(per_repeat.max()),
                              "mean over the 25 folds": result.mean_difference, "p (corrected t, J = 25)": result.p_value})
partition_summary = pd.DataFrame(partition_summary)
show_table(partition_summary, floatfmt=".3f")
sd_25 = float(np.std(m2 - m1, ddof=1))
mde_25_folds = float((stats.t.ppf(0.975, 24) + stats.t.ppf(0.8, 24)) * sd_25 * np.sqrt(1 / 25 + 1 / 4))
cyp2d6_partitions = partition_summary[partition_summary["enzyme"] == "CYP2D6"].iloc[0]
'''

CELLS = [
    md("""
# Part VII. The first TDI models and what they taught

Part VI built the measurement. This part uses it, in the order in which the work happened, from the challenge's own baseline to the first assembled models, the work of the first weeks of the project. It has eleven sections. Section 7.1 reproduces the baseline; 7.2 adds the project's two nitrogen screens to it one block at a time; 7.3 gives each compound the labels of its nearest neighbours; 7.4 describes a feature that looked like the largest gain of the project and was retracted; 7.5 and 7.6 are about the yes/no threshold and about uncertainty; 7.7 to 7.9 bring in external data, quantum-chemical and pharmacophore features and learned embeddings; 7.10 asks whether predicted structures help a classifier; 7.11 assembles the first complete models and records the decision to build the nested evaluation of 6.3.

Two things are worth knowing before reading it. The first is that the experiments of this part were run *before* the nested evaluation existed. They used five-fold cross-validation (later experiments repeat it five times), a threshold of 0.5 or one tuned inside the folds, and the MCC of section 4.10 as the score. The corrected resampled t of 6.5 was applied to them afterwards wherever the per-fold scores had been saved (6.6). Each section says which of its numbers are stored tables of the day and which are recomputed for this notebook, and when a recomputation disagrees with the stored table the section says why.
The second is the population. The early CYP3A4 numbers were computed on the *mixed* population of section 2.4: a third of the labelled CYP3A4 compounds have no direct-inhibition reading and are all labelled negative, and training and scoring on them rewards a shortcut that does not exist in the blinded test set. Section 7.4 shows the shortcut at work. The models of Part VIII use the corrected population, and the counterpart of each experiment of this part in the nested design is named in its section.

What this part hands on is mostly a set of lessons, and they are the reason the later parts are built as they are. A score without its spread cannot be compared with another (7.1, 7.2). A feature that is mechanistically sensible usually does not help once it is tested properly (7.2, 7.7, 7.8, 7.9, 7.10). A result that looks too good is more often a shortcut than a discovery (7.4). A decision taken on a point estimate of five folds is not a decision (7.2). The rule that no idea enters the model without passing a gate written in advance (6.7) is the answer to all four.
"""),
    md("""
## 7.1 Reproducing the challenge's baseline from the raw file

> **Field:** Machine learning · Data science

*Where this section sits:* Part VI said how to measure. The first thing to measure is the reference. A project that wants to improve on a baseline must know what the baseline scores *in its own measurement*, because a number recalled from elsewhere comes with a protocol that the reader does not control. This section reproduces the organisers' tutorial baseline from the raw training file, shows why its MCC is low even though its AUC is not, and sets it beside the score the organisers' reference entry earns on the challenge's own board.

**The baseline.** The tutorial of the challenge (the repository `OpenADMET/CYP-Challenge-Tutorial`) trains one LightGBM classifier (section 4.3) for each of the two enzymes, with the library's default settings, on the RDKit descriptors of the compounds (section 3.3). It splits the labelled compounds at random into 75 percent for fitting and 25 percent for scoring, keeping the share of positives the same in both parts (a *stratified* split, section 4.1), and reports accuracy, F1, MCC and AUC at the threshold of 0.5, for ten such random splits. The reference row of the board is the organisers' version of this model. The project's version of it is `cypmode/challenge/tdi_baseline.py`: `load_tdi_training_data` keeps every compound that has a label for at least one enzyme, `compute_descriptors` uses the same descriptor library as the tutorial, so that the numbers are comparable, and `cross_validate_baseline` is the tutorial's protocol. The first cell counts what the baseline is given.
"""),
    code(BASELINE_DATA),
    md("""
The tutorial's frame holds {{=n_tutorial:,}} compounds with at least one label. For CYP3A4 {{=data_table.loc['CYP3A4', 'labelled compounds']:,}} of them are labelled, {{=data_table.loc['CYP3A4', 'share']:.1%}} as time-dependent inhibitors, and {{=data_table.loc['CYP3A4', 'no direct reading']:,}} of the labelled compounds have no direct-inhibition reading: they are the third of the CYP3A4 labels that follow the convention of section 2.4, and the baseline, like every early experiment, trains and is scored on them together with the {{=data_table.loc['CYP3A4', 'measured in both arms']:,}} compounds measured in both arms. For CYP2D6 {{=data_table.loc['CYP2D6', 'labelled compounds']:,}} are labelled, {{=data_table.loc['CYP2D6', 'share']:.1%}} positive, and only {{=data_table.loc['CYP2D6', 'no direct reading']}} lack a direct reading. In both enzymes about one compound in five is positive, so a model that says "negative" for everything is right four times in five. That fact, which section 4.10 called the accuracy paradox, shapes everything below.

The next cell runs the baseline two ways. The first is the tutorial's protocol, ten random splits per enzyme. The tutorial does not seed its random splits, so two runs of it give different numbers, which is itself a finding; the seed of numpy is fixed here so that this notebook's run repeats. The second is the project's fixed-seed five-fold protocol (`cross_validate_fixed_seed`): the same five folds every time, so that a later model scored on them is compared with the baseline on the same compounds. The tables stored when the project first ran them are read alongside.
"""),
    code(BASELINE_RUNS),
    md("""
The descriptor set has {{=n_descriptors}} columns. On the tutorial's protocol the baseline scores an MCC of {{=baseline_value('tutorial protocol, run here', 'CYP3A4', 'MCC mean'):.3f}} for CYP3A4 and {{=baseline_value('tutorial protocol, run here', 'CYP2D6', 'MCC mean'):.3f}} for CYP2D6 in this run, and {{=baseline_value('tutorial protocol, stored', 'CYP3A4', 'MCC mean'):.3f}} and {{=baseline_value('tutorial protocol, stored', 'CYP2D6', 'MCC mean'):.3f}} in the run stored when the project began. The two runs use the same protocol on the same data, and for CYP2D6 they differ by {{=abs(baseline_value('tutorial protocol, run here', 'CYP2D6', 'MCC mean') - baseline_value('tutorial protocol, stored', 'CYP2D6', 'MCC mean')):.3f}}, about as much as the standard deviation of one split ({{=baseline_value('tutorial protocol, run here', 'CYP2D6', 'MCC sd'):.3f}}). That is the reason for the second protocol: with five fixed folds the baseline scores {{=baseline_value('fixed-seed five-fold, run here', 'CYP3A4', 'MCC mean'):.3f}} and {{=baseline_value('fixed-seed five-fold, run here', 'CYP2D6', 'MCC mean'):.3f}}, the re-run equals the stored table to the last digit ({{=fixed_matches_stored}}), and every comparison of the next sections uses these folds. The standard deviation over the five folds is {{=baseline_value('fixed-seed five-fold, run here', 'CYP3A4', 'MCC sd'):.3f}} for CYP3A4 and {{=baseline_value('fixed-seed five-fold, run here', 'CYP2D6', 'MCC sd'):.3f}} for CYP2D6. Keep these two numbers in mind: they are as large as nearly every improvement that the next sections will claim, and they are why 6.5 was needed.

**Why an AUC of 0.80 and an MCC of 0.29 belong together.** The AUC (section 4.10) asks only whether the model ranks positives above negatives. The MCC asks about the *calls* made at a threshold, and the baseline makes them at the probability 0.5. A gradient-boosted model fitted to data in which one compound in five is positive produces probabilities that stay near that rate, and few compounds reach 0.5. The next cell shows what happens: the pooled out-of-fold probabilities of the fixed-seed run, the share of compounds called positive at 0.5 against the share that are positive, the accuracy against that of calling everything negative, and the MCC as the threshold moves.
"""),
    code(THRESHOLD_VIEW),
    md("""
For CYP3A4 the threshold of 0.5 calls {{=call_table.loc['CYP3A4', 'called positive at 0.5']:.1%}} of the compounds positive, while {{=call_table.loc['CYP3A4', 'positive share (truth)']:.1%}} are; the accuracy is {{=call_table.loc['CYP3A4', 'accuracy at 0.5']:.3f}}, barely above the {{=call_table.loc['CYP3A4', 'accuracy of calling everything negative']:.3f}} of a model that never says positive. The MCC is {{=call_table.loc['CYP3A4', 'MCC at 0.5']:.3f}}, but the same probabilities reach {{=call_table.loc['CYP3A4', 'best MCC over the thresholds']:.3f}} at a threshold of {{=call_table.loc['CYP3A4', 'threshold of the best']:.2f}}. The ranking was good; the decision was bad. The best-threshold number is chosen on the very predictions it is computed on, so it is optimistic, and section 7.5 and section 8.4 show how a threshold is set inside the folds and transferred as a *rate* of calls. For CYP2D6 the model is a different case. Its AUC is {{=call_table.loc['CYP2D6', 'AUC']:.3f}}, only a little better than the 0.5 of guessing, it calls {{=call_table.loc['CYP2D6', 'called positive at 0.5']:.1%}} of the compounds positive, and no threshold gives it much: the best MCC over the thresholds is {{=call_table.loc['CYP2D6', 'best MCC over the thresholds']:.3f}}. CYP2D6 is the hard enzyme of the challenge from this first number, and the figure shows it: the CYP2D6 curve stays close to zero everywhere.

**The same model on the board.** The interim leaderboards (section 2.10) list the organisers' reference row, the entry whose report link reads "Official LightGBM baseline": the tutorial's model, scored by the organisers on the 750 blinded compounds. Its board MCC is set against the cross-validation estimates of the same model on the training file.
"""),
    code(BOARD_OFFSET),
    md("""
The reference row scores {{=board_table.loc['CYP3A4', 'board MCC']:.3f}} on CYP3A4 (rank {{=board_table.loc['CYP3A4', 'rank on the interim board']}} of {{=board_table.loc['CYP3A4', 'entries']}}) and {{=board_table.loc['CYP2D6', 'board MCC']:.3f}} on CYP2D6 (rank {{=board_table.loc['CYP2D6', 'rank on the interim board']}}). The board's own bootstrap standard deviations are {{=board_table.loc['CYP3A4', 'board MCC sd (bootstrap)']:.3f}} and {{=board_table.loc['CYP2D6', 'board MCC sd (bootstrap)']:.3f}}, and the board scores are higher than the cross-validation estimates by {{=board_table.loc['CYP3A4', 'board minus fixed-seed CV']:.3f}} and {{=board_table.loc['CYP2D6', 'board minus fixed-seed CV']:.3f}}. The cross-validation of the training file is therefore *not* a forecast of the board. Part VI named the reasons: the blinded compounds are analogues of top hits and sit closer to the training file than a cross-validation fold does (6.8), their positive rates are not the training file's (2.10), and the score of the board is computed on a different set of compounds with its own sampling noise. The project uses cross-validation to put candidates in the right *order*, never to predict the board number, and 6.4 built the yardstick that makes the order meaningful for this test set.

*On what actually ran.* `cypmode/challenge/tdi_baseline.py` holds `load_tdi_training_data`, `compute_descriptors`, `cross_validate_baseline` (the tutorial's protocol, with the unseeded splits) and `cross_validate_fixed_seed` (the project's protocol, five stratified folds with `random_state=42`, a LightGBM with the same seed, the threshold 0.5); the stored tables are `baseline_reproduction.csv` and `baseline_fixed_seed.csv` in `data/tdi_track/outputs/comparisons/`. The numbers of the board are read from the snapshot files of `data/tdi_track/inputs/leaderboard_snapshots/`, which hold no test label. Everything in this section is on the mixed population of section 2.4.
"""),
    md("""
## 7.2 The staircase M0 to M3: adding one block of features at a time

> **Field:** Machine learning · Cheminformatics

*Where this section sits:* 7.1 fixed the reference. The first question the project asked of its own chemistry was whether the nitrogen screens of Part V help a classifier that already has the descriptors. The method is an **ablation** in its additive form, a staircase: the same compounds, the same folds and the same model, one block of columns added at each step, so that the change of the score from one step to the next can be attributed to the block. This section runs the staircase, finds that the table the project stored is reproducible to the last digit, finds also that a later change in one of the blocks changes the table, and uses that to measure how much the staircase could ever have shown.

**The four steps.** M0 is the baseline of 7.1, the RDKit descriptors alone. M1 adds the *motif flag* of section 5.1: one column, 1 when any of the six heme-binding nitrogen patterns matches. M2 adds the *trialkylamine flag* of section 5.2, one more column. M3 adds the 2,048-bit Morgan fingerprint of section 3.1. The module is `cypmode/challenge/tdi_ablation.py`; its docstring states the rule under which it was written: if M2 does not beat M1, the amine flag is dropped, whatever the project's attachment to it.

A detail makes the staircase a good case study. The trialkylamine pattern of section 5.2 was extended after the stored run: three more exclusions (the sulfonamide, the enamine, the amidine and guanidine nitrogens) were added to the pattern, which sets the flag for fewer compounds. A stored result is the result of the code *as it was*, and the way to find out whether it is still the result of the code as it is, is to run it again. The cell therefore runs the staircase twice on the same five folds, once with the pattern of the stored run, written out below in full, and once with the pattern of today.
"""),
    code(ABLATION_RERUN),
    md("""
The four models have {{=widths['M0']}}, {{=widths['M1']}}, {{=widths['M2']}} and {{=widths['M3']:,}} columns, and the motif flag is set for {{=motif_share:.1%}} of the compounds. The re-run with the pattern of the stored run equals the stored table to {{=then_matches_stored:.0e}}: the staircase is exactly reproducible, which is worth stating because it means every difference below is a difference of the code and not of the machine. The old pattern sets the amine flag for {{=amine_share_then:.1%}} of the compounds and today's for {{=amine_share_now:.1%}}. For CYP3A4 the change in the pattern makes no visible difference. For CYP2D6 it makes a large one: the stored table has the amine step lowering the MCC from {{=ablation_table[(ablation_table['model'] == 'M1') & (ablation_table['enzyme'] == 'CYP2D6')]['stored (pattern of that day)'].iloc[0]:.3f}} to {{=ablation_table[(ablation_table['model'] == 'M2') & (ablation_table['enzyme'] == 'CYP2D6')]['stored (pattern of that day)'].iloc[0]:.3f}}, which is why the project of the day dropped the flag, and the table with today's pattern has it raising the MCC to {{=ablation_table[(ablation_table['model'] == 'M2') & (ablation_table['enzyme'] == 'CYP2D6')]['re-run, pattern of today'].iloc[0]:.3f}}. The same step has the opposite sign.

Neither sign is a finding, and the next two cells show why. The first applies the corrected resampled t of 6.5 to each step of each table, with the five folds as the five paired observations ($J = 5$, $n_{\\text{test}}/n_{\\text{train}} = 1/4$), and computes with the formula of 6.7 the smallest true change in MCC that a design of five folds recognises 8 times in 10, from the spread of the fold differences.
"""),
    code(ABLATION_TESTS),
    md("""
No step of either table reaches p = 0.05; the smallest p-value is {{=step_table['p (corrected t)'].min():.3f}}, for the motif flag on CYP3A4, a gain of {{=step_table[(step_table['table'] == 'stored') & (step_table['enzyme'] == 'CYP3A4') & (step_table['step'] == 'M0 to M1')]['mean MCC change'].iloc[0]:+.3f}} MCC, positive in 4 of the 5 folds and too small for five folds to separate from zero. The amine step on CYP2D6 has p = {{=stored_amine_step['p (corrected t)']:.2f}} in the stored table and p = {{=today_amine_step['p (corrected t)']:.2f}} in today's; the standard deviation of its fold differences is {{=stored_amine_step['sd of the fold differences']:.3f}} and {{=today_amine_step['sd of the fold differences']:.3f}}. With the median spread of the fold differences of this table, {{=median_sd_folds:.3f}}, five folds recognise a true change of {{=mde_five_folds:.2f}} MCC or more and nothing smaller. Every change in the table is smaller than that, so that for a typical step the answer of the staircase was decided by chance. The staircase of five folds could not have answered the question it was built for.

The second check removes the dependence on one partition. The compounds are partitioned into five folds in one way only in the stored run; a different random partition would have given different folds. The cell repeats the amine step (M1 against M2, today's pattern) on five different partitions, 25 folds in all, which is the design that the later experiments used (section 6.6).
"""),
    code(PARTITION_NOISE),
    md("""
For CYP2D6 the change of MCC from M1 to M2 is {{=cyp2d6_partitions['change of MCC in each of the 5 partitions']}} in the five partitions: it runs from {{=cyp2d6_partitions['lowest']:+.3f}} to {{=cyp2d6_partitions['highest']:+.3f}}, and the partition decides the sign. Over the 25 folds the mean change is {{=cyp2d6_partitions['mean over the 25 folds']:+.3f}} and the corrected p-value is {{=cyp2d6_partitions['p (corrected t, J = 25)']:.2f}}; even the 25-fold design, with the spread of this comparison, recognises only a true change of {{=mde_25_folds:.2f}} MCC or more. The amine flag neither helps nor hurts as far as this data can say, and the opposite signs of the stored and the re-run tables are two draws from the same noise.

What the staircase established is therefore not what the project of the day recorded. The decision "the amine flag is dropped" was taken on the point estimate of one partition of five folds, and the evidence for it does not survive a look: it was neither established nor refuted. Whether the decision did harm cannot be told, since the effect is within the noise, and it matters little, for a reason that is instructive: the final models are not built on this staircase. Their base feature set, the one called A (section 8.1), contains the Morgan counts and the RDKit descriptors and neither flag; the motif patterns reached the models as a block of seven columns inside the larger set called A+fps; the amine flag is a block of the feature store (section 3.7) that no member of the final stack uses. The lesson that the project drew, and that built the rest of this part, is a method and not a verdict: report the spread beside the mean; test a step with a design that can see it (repeated partitions, the correction of 6.5); and write the rule down before the result (6.7).

*On what actually ran.* `cypmode/challenge/tdi_ablation.py` builds the four feature columns (`add_ablation_features`; it also builds `m3_no_amine_features`, the set that the downstream modules were later switched to, and the module's comments record that for a time they silently kept the amine flag after the decision to drop it) and runs the staircase (`run_ablation`, which calls `cross_validate_fixed_seed` for each model). `cypmode/metrics/motifs.py` and `cypmode/metrics/amines.py` hold the two flags. The stored table is `ablation_m0_m3.csv`. The re-run with the pattern of the stored run replaces the pattern in memory for the length of one cell and restores it; no file of the repository is changed. This section too is on the mixed population of section 2.4.
"""),
    figure("ml_m0_m3_ablation"),
]
