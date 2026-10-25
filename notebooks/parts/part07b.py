# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part VII (second part): nearest-neighbour read-across features (7.3), the direct-inhibition bridge and its retraction (7.4)."""

from notebooks.notebook_builder import code, md

KNN_DEMO = '''
# The four read-across features for one compound. A labelled CYP3A4 compound that is a time-dependent inhibitor and whose ten nearest neighbours (Tanimoto similarity on the Morgan bits, section 3.2) include three
# inhibitors. The neighbours are searched among all the other labelled compounds, the compound itself left out; the features are then computed by hand and by the project's function.
from rdkit import DataStructs

from cypmode.challenge.knn_features import KNN_FEATURE_NAMES, _morgan_fp, add_knn_label_features_for_fold, knn_label_features
from cypmode.challenge.tdi_ablation import add_ablation_features

staircase = add_ablation_features(described)                                  # the columns of 7.2: descriptors, motif flag, amine flag and Morgan bits, one row per labelled compound

cyp3a4_frame = described.dropna(subset=["CYP3A4_is_TDI"]).reset_index(drop=True)
y_3a4 = cyp3a4_frame["CYP3A4_is_TDI"].astype(int).to_numpy()
fps_3a4 = [_morgan_fp(s) for s in cyp3a4_frame["SMILES"]]
example_index = None
for i in range(len(cyp3a4_frame)):
    similarities = np.array(DataStructs.BulkTanimotoSimilarity(fps_3a4[i], fps_3a4))
    similarities[i] = -1.0                                                   # the compound is not its own neighbour
    nearest = np.argsort(-similarities)[:10]
    if y_3a4[i] == 1 and y_3a4[nearest].sum() == 3 and 0.45 < similarities[nearest].mean() < 0.6:
        example_index = i
        break
neighbour_table = pd.DataFrame({"rank": np.arange(1, 11), "Tanimoto similarity": similarities[nearest], "neighbour is an inhibitor": y_3a4[nearest]})
show_table(neighbour_table, floatfmt=".2f")
by_hand = [y_3a4[nearest].mean(), float((similarities[nearest] * y_3a4[nearest]).sum() / similarities[nearest].sum()), similarities[nearest].mean(), 10.0]
others = np.arange(len(y_3a4)) != example_index
by_function = knn_label_features([fps_3a4[example_index]], [fps_3a4[j] for j in np.where(others)[0]], y_3a4[others])[0]
by_hand_equals_function = bool(np.allclose(by_hand, by_function))
show_table(pd.DataFrame({"feature": KNN_FEATURE_NAMES, "by hand": by_hand, "by the project's function": by_function}), floatfmt=".3f")
example_name = cyp3a4_frame["Molecule_Name"].iloc[example_index]
'''

KNN_LEAKS = '''
# Where the label of a compound may enter its neighbours' features. Five versions of the same experiment on the same five folds (fixed seed), scored by AUC and by MCC at 0.5 on the validation folds, per enzyme: the
# model without the read-across features (M3 without the amine flag: descriptors, motif flag, Morgan bits); the project's construction, in which the features are built inside each fold from the training compounds only
# (the compound itself left out for the training rows); features built once on all labelled compounds, each compound's own label left out (global leave-one-out); the training rows' features built with the compound
# itself in its own neighbour list; and the validation rows' features built with the compound itself in the list. Only the first two and the last are about the validation compounds' labels, and they are the ones to compare.
leak_rows = []
for iso in TDI_ISOFORMS:
    column = f"{iso}_is_TDI"
    sub = staircase.dropna(subset=[column]).reset_index(drop=True)
    y = sub[column].astype(int).to_numpy()
    X = np.stack(sub["m3_no_amine_features"])
    fps = [_morgan_fp(s) for s in sub["SMILES"]]
    global_loo = np.zeros((len(y), 4))
    for i in range(len(y)):
        keep = np.where(np.arange(len(y)) != i)[0]
        global_loo[i] = knn_label_features([fps[i]], [fps[j] for j in keep], y[keep])[0]
    for fold, (fit_rows, held) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
        inside = add_knn_label_features_for_fold(fps, y, fit_rows, held)
        train_self = np.zeros((len(y), 4))
        train_fps = [fps[i] for i in fit_rows]
        train_self[fit_rows] = knn_label_features(train_fps, train_fps, y[fit_rows])        # every training compound is in its own list
        train_self[held] = inside[held]
        valid_self = inside.copy()
        valid_self[held] = knn_label_features([fps[i] for i in held], [fps[i] for i in range(len(y))], y)     # every validation compound is in its own list
        versions = {"no read-across features": X, "built inside the fold": np.hstack([X, inside]), "global leave-one-out": np.hstack([X, global_loo]),
                    "training compound in its own list": np.hstack([X, train_self]), "validation compound in its own list": np.hstack([X, valid_self])}
        for name, features in versions.items():
            p = LGBMClassifier(verbose=-1, random_state=42).fit(features[fit_rows], y[fit_rows]).predict_proba(features[held])[:, 1]
            leak_rows.append({"enzyme": iso, "version": name, "fold": fold, "MCC": matthews_corrcoef(y[held], (p >= 0.5).astype(int)), "AUC": roc_auc_score(y[held], p)})
leak_scores = pd.DataFrame(leak_rows)
leak_table = leak_scores.groupby(["enzyme", "version"], sort=False)[["MCC", "AUC"]].mean().reset_index()
show_table(leak_table, floatfmt=".3f")
def leak_value(iso, version, column):
    return float(leak_table[(leak_table["enzyme"] == iso) & (leak_table["version"] == version)][column].iloc[0])
'''

KNN_STORED = '''
# The stored result and its repeats. The ablation of the day (knn_features_ablation.csv: M3 against M3 plus the read-across features, five folds) and the five repeated-partition experiments of 6.6, every one of which
# contains the pair M3 and M3+KNN: the mean MCC change from adding the features, and the corrected paired p-value, for each enzyme.
from cypmode.challenge.model_comparison import corrected_resampled_t

stored_knn = pd.read_csv(COMPARISONS / "knn_features_ablation.csv")
stored_knn_means = stored_knn.groupby("isoform")[["mcc_m3", "mcc_m3_plus_knn"]].mean()
knn_rows = [{"experiment": "ablation of the day (5 folds)", "enzyme": iso, "mean MCC change": float((part["mcc_m3_plus_knn"] - part["mcc_m3"]).mean()), "p (corrected t)": corrected_resampled_t((part["mcc_m3_plus_knn"] - part["mcc_m3"]).to_numpy(), 4, 1).p_value}
            for iso, part in stored_knn.groupby("isoform")]
for name in ("stacking", "chemberta", "pharmacophore_fingerprint", "qm_reactivity", "quantum_pharmacophore"):
    raw = pd.read_csv(COMPARISONS / f"{name}_significance_raw.csv")
    for iso, part in raw.groupby("isoform"):
        pair = part.pivot_table(index=["repeat", "fold"], columns="variant", values="mcc")
        difference = (pair["M3+KNN"] - pair["M3"]).to_numpy()
        knn_rows.append({"experiment": f"{name} (5 x 5)", "enzyme": iso, "mean MCC change": float(difference.mean()), "p (corrected t)": corrected_resampled_t(difference, 4, 1).p_value})
knn_table = pd.DataFrame(knn_rows)
show_table(knn_table, floatfmt=".3f")
knn_5x5 = knn_table[knn_table["experiment"].str.contains("5 x 5")]
knn_range = {iso: (float(knn_5x5[knn_5x5["enzyme"] == iso]["mean MCC change"].min()), float(knn_5x5[knn_5x5["enzyme"] == iso]["mean MCC change"].max())) for iso in TDI_ISOFORMS}
knn_smallest_p = float(knn_table["p (corrected t)"].min())
'''


BRIDGE_FACTS = '''
# What the bridge is and what it was given. The label rule of section 2.3 makes a compound's direct-inhibition pIC50 d part of its TDI label (positive when the TDI-arm pIC50 t exceeds max(d, 4) by more than log10 2), and the
# blinded test file shows no such value. The bridge (cypmode/challenge/oof_bridge.py) is a regressor of d from the structure, with the inverse-variance weights of section 2.5, whose out-of-fold predictions were to feed the
# classifier. The stored predictions are read, merged on to the labelled CYP3A4 compounds, and the facts that matter are counted.
from sklearn.metrics import roc_auc_score

stored_bridge = pd.read_csv(COMPARISONS / "oof_direct_inhibition.csv")
test_file = pd.read_csv(ROOT / "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv")
bridge_frame = staircase.dropna(subset=["CYP3A4_is_TDI"]).reset_index(drop=True).merge(
    stored_bridge[["Molecule_Name", "CYP3A4_pIC50_direct_inhibition", "CYP3A4_pIC50_direct_inhibition_std", "CYP3A4_pIC50_oof", "CYP3A4_pIC50_oof_abs_error"]], on="Molecule_Name", how="left")
y_bridge = bridge_frame["CYP3A4_is_TDI"].astype(int).to_numpy()
unmeasured = bridge_frame["CYP3A4_pIC50_direct_inhibition"].isna().to_numpy()
measured_pair = bridge_frame.dropna(subset=["CYP3A4_pIC50_direct_inhibition"])
facts_rows = [
    {"quantity": "labelled CYP3A4 compounds", "value": len(bridge_frame)},
    {"quantity": "of which with no direct-inhibition reading", "value": int(unmeasured.sum())},
    {"quantity": "positives among those", "value": int(y_bridge[unmeasured].sum())},
    {"quantity": "AUC of the single fact 'a direct reading exists' as a score for the label", "value": float(roc_auc_score(y_bridge, (~unmeasured).astype(float)))},
    {"quantity": "mean absolute error of the bridge's out-of-fold prediction (pIC50 units)", "value": float(measured_pair["CYP3A4_pIC50_oof_abs_error"].mean())},
    {"quantity": "correlation of the prediction with the reading", "value": float(np.corrcoef(measured_pair["CYP3A4_pIC50_direct_inhibition"], measured_pair["CYP3A4_pIC50_oof"])[0, 1])},
    {"quantity": "columns of the blinded test file", "value": ", ".join(test_file.columns)},
]
show_table(pd.DataFrame(facts_rows), floatfmt=".3f")
bridge_facts = {row["quantity"]: row["value"] for row in facts_rows}
n_unmeasured, n_bridge_rows = int(unmeasured.sum()), len(bridge_frame)
flag_auc = bridge_facts["AUC of the single fact 'a direct reading exists' as a score for the label"]
bridge_mae = bridge_facts["mean absolute error of the bridge's out-of-fold prediction (pIC50 units)"]
bridge_corr = bridge_facts["correlation of the prediction with the reading"]
'''

BRIDGE_CV = '''
# The feature, tried three ways on the same five folds. The bridge's prediction is added to the classifier's columns. For a compound with no direct reading the stored prediction is empty (LightGBM treats an empty value as a
# value of its own: it learns on which side of each split to send it). (1) The naive evaluation: the validation compounds keep their empty values, as in the experiment of the day. (2) A deployment-like evaluation: the
# validation compounds are given what the test compounds will be given, a prediction for every compound, made by a regressor fitted on the compounds that have a reading (cypmode.challenge.oof_bridge.fit_final_model); the
# classifier is the same one, trained on the same empty values. (3) The classifier without the feature. MCC is taken by calling the same share of compounds positive as the training set's positive rate (section 6.4),
# which needs no threshold, next to the AUC.
from cypmode.challenge.oof_bridge import fit_final_model

bridge_oof = bridge_frame["CYP3A4_pIC50_oof"].to_numpy()                         # empty where there is no reading
regressor = fit_final_model(bridge_frame, "CYP3A4")
final_prediction = regressor.predict(np.stack(bridge_frame["descriptors"]))
bridge_deployment = np.where(unmeasured, final_prediction, bridge_oof)
X_bridge = np.stack(bridge_frame["m3_no_amine_features"])
def call_top_share(score, share):
    k = int(round(share * len(score)))
    return (score >= np.sort(score)[::-1][k - 1]).astype(int)
cv_rows = []
for fold, (fit_rows, held) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X_bridge, y_bridge)):
    with_feature = LGBMClassifier(verbose=-1, random_state=42).fit(np.column_stack([X_bridge, bridge_oof])[fit_rows], y_bridge[fit_rows])
    without = LGBMClassifier(verbose=-1, random_state=42).fit(X_bridge[fit_rows], y_bridge[fit_rows])
    for name, model, held_features in (("without the feature", without, X_bridge[held]),
                                       ("with the feature, validation compounds as in the experiment of the day", with_feature, np.column_stack([X_bridge, bridge_oof])[held]),
                                       ("with the feature, validation compounds as the test compounds will be", with_feature, np.column_stack([X_bridge, bridge_deployment])[held])):
        p = model.predict_proba(held_features)[:, 1]
        cv_rows.append({"version": name, "fold": fold, "AUC": roc_auc_score(y_bridge[held], p), "MCC (top share called)": matthews_corrcoef(y_bridge[held], call_top_share(p, y_bridge.mean()))})
# (4) the corrected population of section 2.4: only the compounds with a reading, so that nothing is empty and the feature has to earn its place as a prediction from the structure.
both_arms = ~unmeasured
X_both, y_both, bridge_both = X_bridge[both_arms], y_bridge[both_arms], bridge_oof[both_arms]
for fold, (fit_rows, held) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X_both, y_both)):
    for name, features in (("corrected population, without the feature", X_both), ("corrected population, with the feature", np.column_stack([X_both, bridge_both]))):
        p = LGBMClassifier(verbose=-1, random_state=42).fit(features[fit_rows], y_both[fit_rows]).predict_proba(features[held])[:, 1]
        cv_rows.append({"version": name, "fold": fold, "AUC": roc_auc_score(y_both[held], p), "MCC (top share called)": matthews_corrcoef(y_both[held], call_top_share(p, y_both.mean()))})
cv_frame = pd.DataFrame(cv_rows)
bridge_cv = cv_frame.groupby("version", sort=False)[["AUC", "MCC (top share called)"]].mean().reset_index()
show_table(bridge_cv, floatfmt=".3f")
bridge_values = bridge_cv.set_index("version")
v_without, v_naive, v_deploy, v_both_without, v_both_with = (bridge_values.index[i] for i in range(5))
fold_gain = cv_frame[cv_frame["version"] == v_both_with].sort_values("fold")["AUC"].to_numpy() - cv_frame[cv_frame["version"] == v_both_without].sort_values("fold")["AUC"].to_numpy()
corrected_population_test = corrected_resampled_t(fold_gain, 4, 1)
day_with = pd.read_csv(ROOT / "data/tdi_track/outputs/cyp3a4/oof_bridge_feature_tuned_cyp3a4.csv")
day_without = pd.read_csv(ROOT / "data/tdi_track/outputs/cyp3a4/threshold_tuning_cyp3a4_no_amine.csv")
day_mcc_with, day_mcc_without, day_auc_with = float(day_with["mcc_tuned"].mean()), float(day_without["mcc_tuned"].mean()), float(day_with["roc_auc"].mean())
fig, axis = plt.subplots(figsize=(8.4, 3.6))
bars = [bridge_values.loc[v_without, "AUC"], bridge_values.loc[v_naive, "AUC"], bridge_values.loc[v_deploy, "AUC"], flag_auc, bridge_values.loc[v_both_without, "AUC"], bridge_values.loc[v_both_with, "AUC"]]
labels = ["without the\\nfeature", "naive\\nevaluation", "deployment-like\\nevaluation", "the 'reading exists'\\nfact alone", "corrected population\\nwithout", "corrected population\\nwith the feature"]
axis.bar(labels, bars, color=[theme["muted"], theme["orange"], theme["red"], theme["blue"], theme["muted"], theme["green"]])
axis.axhline(0.5, color=theme["muted"], ls=":", lw=1.2)
for position, value in enumerate(bars):
    axis.text(position, value + 0.01, f"{value:.3f}", ha="center", color=theme["text"], fontsize=10)
axis.set(ylabel="AUC on the validation folds", ylim=(0.4, 0.95), title="the bridge feature, judged four ways")
axis.tick_params(axis="x", labelsize=8)
tidy(axis, theme)
fig.tight_layout()
'''

CELLS = [
    md("""
## 7.3 Reading across: the labels of a compound's nearest neighbours

> **Field:** Cheminformatics · Machine learning

*Where this section sits:* 7.2 added chemistry that the project had written. This section adds what a medicinal chemist does first with a compound that has to be judged: look at the known compounds that look most like it. The idea is old and has a name, the *similar-property principle* (the volume edited by Johnson and Maggiora, 1990): structurally similar compounds tend to have similar properties. Section 4.5 turned it into a model, the nearest-neighbour classifier. Here it becomes something else, four columns that are handed to the gradient-boosted model of 7.1 together with the descriptors, so that the global model can use the local evidence when it helps and ignore it when it does not. The section also holds the project's first leakage lesson, and it is worth reading even by someone who does not care about this feature.

**The four columns.** For a compound $i$ with the Morgan bit fingerprint of section 3.1, let $N_k(i)$ be the set of the $k = 10$ other labelled compounds with the highest Tanimoto similarity $s_{ij}$ to it (section 3.2), and let $y_j \\in \\{0, 1\\}$ be the label of compound $j$, 1 for a time-dependent inhibitor. Call the four columns $f_1$ to $f_4$. They are

$$f_1 = \\frac{1}{k} \\sum_{j \\in N_k(i)} y_j, \\qquad f_2 = \\frac{\\sum_{j \\in N_k(i)} s_{ij}\\, y_j}{\\sum_{j \\in N_k(i)} s_{ij}}, \\qquad f_3 = \\frac{1}{k} \\sum_{j \\in N_k(i)} s_{ij}, \\qquad f_4 = k.$$

The first is the share of inhibitors among the neighbours, the second the same share with the closer neighbours counted more, the third the mean similarity of the neighbours, which tells the model how far the neighbourhood can be trusted (ten neighbours at 0.9 are evidence, ten at 0.3 are not), and the fourth the number of neighbours found, which is $k$ whenever the pool holds at least $k$ compounds and so is constant here. The next cell computes the four columns for one compound by hand, listing its neighbours, and by the project's function.
"""),
    code(KNN_DEMO),
    md("""
The compound {{=example_name}} is itself an inhibitor; its ten nearest neighbours, at similarities from {{=neighbour_table['Tanimoto similarity'].max():.2f}} down to {{=neighbour_table['Tanimoto similarity'].min():.2f}}, include {{=int(neighbour_table['neighbour is an inhibitor'].sum())}} inhibitors, so that $f_1 = 0.3$, and the closer neighbours are mostly negative, so that $f_2$ is slightly lower. The by-hand values equal the function's: {{=by_hand_equals_function}}. The example is a small lesson of its own: the neighbours of an inhibitor are mostly not inhibitors, and a feature that is only moderately informative is what read-across usually provides.

**Where a compound's label may not enter its own features.** The columns use labels, and that is where the danger is. A compound is its own nearest neighbour, at similarity 1, so that if it is left in its own neighbour list, its label is in its own feature, and the model needs no chemistry to read the answer. The project's construction (`add_knn_label_features_for_fold`) follows two rules. A training compound's list is searched among the other training compounds of the fold, itself left out; a validation compound's list is searched among the training compounds of the fold only. The cell compares the construction with its four ways of going wrong, on the same five folds, for both enzymes: no read-across columns at all; the columns built inside the fold as just described; columns built once on all the labelled compounds with each compound's own label left out ("global leave-one-out"); the training compounds' columns built with the compound itself in its list; and the validation compounds' columns built the same way. The model is the M3 set without the amine flag (descriptors, motif flag, Morgan bits).
"""),
    code(KNN_LEAKS),
    md("""
The two failures that put a compound in its own list have opposite effects. When the *validation* compound is in its own list, the score is inflated: CYP3A4 reaches an AUC of {{=leak_value('CYP3A4', 'validation compound in its own list', 'AUC'):.3f}}, against {{=leak_value('CYP3A4', 'built inside the fold', 'AUC'):.3f}} for the correct construction and {{=leak_value('CYP3A4', 'no read-across features', 'AUC'):.3f}} for no read-across columns at all. The inflation, {{=leak_value('CYP3A4', 'validation compound in its own list', 'AUC') - leak_value('CYP3A4', 'built inside the fold', 'AUC'):+.3f}} AUC, is several times the gain that the corrected feature gives ({{=leak_value('CYP3A4', 'built inside the fold', 'AUC') - leak_value('CYP3A4', 'no read-across features', 'AUC'):+.3f}}), which is how a leak looks in practice: not an absurd score, a plausible improvement. When the *training* compounds are in their own lists the effect is the reverse: the model sees a feature that is almost perfect in training and learns to rely on it, and on the validation compounds, whose corrected columns are far weaker, it fails, with an AUC of {{=leak_value('CYP3A4', 'training compound in its own list', 'AUC'):.3f}} and an MCC of {{=leak_value('CYP3A4', 'training compound in its own list', 'MCC'):.3f}}. The same mistake is invisible in one direction and ruinous in the other, and only the evaluation of the right side shows which.

The third wrong version, the global leave-one-out, is the one the module's docstring warns about: a training compound's column then contains labels of compounds that are in the validation fold. Here it does nothing measurable: the AUC is {{=leak_value('CYP3A4', 'global leave-one-out', 'AUC'):.3f}} against {{=leak_value('CYP3A4', 'built inside the fold', 'AUC'):.3f}}. The measured result is therefore that the second-order leak is within the noise at this size, and the rule is kept all the same, for a reason that is not statistical: the construction inside the fold is cheap, and it cannot be wrong by omission, while a global one has to be argued for each time it is used. For CYP2D6 no version of the feature changes the score beyond the spread of the folds, which is the second recurring theme of this part.

**What the feature did.** The stored ablation of the day compared the same five folds with and without the four columns, and the project's record of the day called its outcome a positive result on both enzymes. The cell sets that table beside the five repeated-partition experiments of 6.6, each of which contains the same pair, M3 and M3 plus the read-across columns.
"""),
    code(KNN_STORED),
    md("""
The stored ablation had the MCC of CYP3A4 rising from {{=stored_knn_means.loc['CYP3A4', 'mcc_m3']:.3f}} to {{=stored_knn_means.loc['CYP3A4', 'mcc_m3_plus_knn']:.3f}} and that of CYP2D6 from {{=stored_knn_means.loc['CYP2D6', 'mcc_m3']:.3f}} to {{=stored_knn_means.loc['CYP2D6', 'mcc_m3_plus_knn']:.3f}}, with corrected p-values of {{=knn_table.iloc[1]['p (corrected t)']:.2f}} and {{=knn_table.iloc[0]['p (corrected t)']:.2f}}. In the five repeated experiments the CYP3A4 change is between {{=knn_range['CYP3A4'][0]:+.3f}} and {{=knn_range['CYP3A4'][1]:+.3f}} MCC in the five repeated experiments and the smallest corrected p-value of the whole table is {{=knn_smallest_p:.2f}}. For CYP2D6 the change runs from {{=knn_range['CYP2D6'][0]:+.3f}} to {{=knn_range['CYP2D6'][1]:+.3f}}: its sign is the sign of the run, and the experiments that differ by it differ only in the few compounds each added block could not be computed for, which were left out of that run and so changed the partitions. So the early reading, a gain on both enzymes and the largest for CYP2D6, was not established: it was one draw. The feature itself is sound and it stayed. In the nested design of Part VIII two members of the stack are built on it (`lgbm_A_deep_knnfeat` and `lgbm_A+fps_deep_knnfeat`, with $k = 10$), and the gate of 6.7 judges what they add inside the stack. The lesson of this section for everything that follows is the one about the evaluation: a feature that reads labels must be built, for every compound that is scored, from labels that compound's score could not have seen.

*On what actually ran.* `cypmode/challenge/knn_features.py` holds `knn_label_features` (the four columns for any query against any reference pool, with no notion of identity), `add_knn_label_features_for_fold` (the fold-internal construction, with the leave-one-out for the training rows) and `cross_validate_with_knn_features` (the ablation of the day). The module's tests include an adversarial case with two identical structures and different labels, which confirms that the self-exclusion works on the compound and not on the structure. The stored table is `knn_features_ablation.csv`; the five repeated-partition tables are those of 6.6. These experiments are on the mixed population of section 2.4.
"""),
    md("""
## 7.4 The direct-inhibition bridge: a gain that was a shortcut

> **Field:** Data science · Machine learning

*Where this section sits:* 7.3 used the labels of other compounds. This section uses a quantity that is part of the *definition* of the label, and the way it went wrong is the most useful single lesson of Part VII. It is told in full because the same mistake can be made, and was made once, with a result that looked like the best of the project at that point.

**The idea.** The label of a compound depends on two measurements (section 2.3): the potency in the TDI arm, $t$, and the potency of direct inhibition, $d$, and it is positive when $t - \\max(d, 4) > \\log_{10} 2$. The blinded test file shows neither: it has two columns, the name and the structure. A model that wants potency as an input must therefore predict it. The *bridge* (`cypmode/challenge/oof_bridge.py`) is a regression of $d$ on the RDKit descriptors, LightGBM with each compound weighted by the inverse of its measured noise variance (section 2.5), whose prediction becomes a column of the classifier. It is built *out of fold*: the prediction for each compound comes from a regressor that never saw that compound, so that the column has in training the distribution it will have on the test compounds. A first plan was a censored regression (the Tobit model) for the readings below the assay floor; the data were checked first and showed no pile-up at the floor but a noise that grows five- to tenfold below it, which the inverse-variance weights describe and the censored model does not.

The cell reads the stored out-of-fold predictions, merges them on to the labelled CYP3A4 compounds and counts the facts that matter.
"""),
    code(BRIDGE_FACTS),
    md("""
Of the {{=n_bridge_rows:,}} labelled CYP3A4 compounds, {{=n_unmeasured:,}} have no direct reading, and of those {{=bridge_facts['positives among those']}} are positives: the convention of section 2.4 labels them all negative. The bridge predicts the reading of the others with a mean absolute error of {{=bridge_mae:.2f}} pIC50 units and a correlation of {{=bridge_corr:.2f}} with the measured value, a fair regression. And the single fact "a direct reading exists", used as a score for the label with no chemistry at all, has an AUC of {{=flag_auc:.3f}}. That is the shortcut. The stored predictions are empty for exactly the compounds that are certain to be negative, and LightGBM treats an empty value as a value of its own, sending it down a learned side of each split, so that a classifier given the bridge's column learns "empty means negative" in its first split. The next cell measures what that does, four ways. The first three are on the mixed population: without the feature; with it, the validation compounds keeping their empty values, which is how the experiment of the day was evaluated; and with it, the validation compounds given what the test compounds will be given, which is a prediction for every compound from a regressor that did not see it, with no empty value. The fourth is on the corrected population of section 2.4, with only the compounds that have a reading, where nothing is empty.
"""),
    code(BRIDGE_CV),
    md("""
The experiment of the day is reproduced. Without the feature the tuned MCC of the stored table is {{=day_mcc_without:.3f}}; with it, {{=day_mcc_with:.3f}} and an AUC of {{=day_auc_with:.3f}}, and the cell's naive evaluation reproduces that AUC ({{=bridge_values.loc[v_naive, 'AUC']:.3f}}; without the feature {{=bridge_values.loc[v_without, 'AUC']:.3f}}). It was the largest gain the project had found, +{{=day_mcc_with - day_mcc_without:.2f}} MCC for a feature with a plausible mechanism, and it was not believed for long. Three facts removed it. The test file exposes no direct reading, so the column the model had learned to read could not exist at deployment. The model, shown the deployment-like input, scores an AUC of {{=bridge_values.loc[v_deploy, 'AUC']:.3f}} and an MCC of {{=bridge_values.loc[v_deploy, 'MCC (top share called)']:.3f}}, far below the {{=bridge_values.loc[v_without, 'AUC']:.3f}} and {{=bridge_values.loc[v_without, 'MCC (top share called)']:.3f}} of the model without the feature: it is worse than having nothing, because it has learned to trust a signal that is absent. And on the corrected population, where the shortcut is gone, the feature moves the AUC from {{=bridge_values.loc[v_both_without, 'AUC']:.3f}} to {{=bridge_values.loc[v_both_with, 'AUC']:.3f}}, a change of {{=corrected_population_test.mean_difference:+.4f}} with a corrected p-value of {{=corrected_population_test.p_value:.2f}}: the prediction of $d$ from the descriptors carries no information about the label that the descriptors did not already carry.

The record of the day re-checked the claim with an evaluation in which the validation compounds were scored with no awareness of their own missingness, and found the tuned MCC at 0.378, below the 0.387 of the model without the feature, and retracted the result. The cell above shows a harsher version of the same check, in which the classifier is also trained with the empty values, and the corrected-population row is the version the project's design adopted in the end: the nested design of Part VIII trains and scores on the compounds with both arms measured, and a model of the direct-inhibition arm, where it appears, is one candidate among the others, a regression of the two arms (sections 4.6 and 9.6), and not a feature handed to the classifier.

Four rules come out of this section, and the project has kept them. The first is that every feature must be computable from the structure alone for the blinded compounds, and the cell above checks the file for it: two columns. The second is that the evaluation of a model must give the validation compounds exactly the information that the test compounds will have, no more and no less. The third is that a gain which is large, arrives at once and has a mechanism that is easy to tell is the kind to attack hardest, not the kind to report first. The fourth is that a gain of a feature must be compared with the gain of the single fact that could explain it: here "a reading exists" alone has an AUC of {{=flag_auc:.2f}}, and no model needed to be better than that to show the shortcut.

*On what actually ran.* `cypmode/challenge/oof_bridge.py` holds `compute_oof_predictions` (a regressor per fold, inverse-variance weights from the measured noise), `fit_final_model` (the regressor fitted on every compound with a reading, which is what would have run on the test compounds) and `predict_test`. The stored table is `oof_direct_inhibition.csv`; the numbers of the day are in `oof_bridge_feature_tuned_cyp3a4.csv` and `threshold_tuning_cyp3a4_no_amine.csv`. The test that the module's out-of-fold construction does not leak is a weak one (it asks only for a correlation above 0.3), as the audit of 2026-09-26 noted; the construction itself was checked directly and is correct, and the retraction concerned the *use* of the feature and not its construction.
"""),
]
