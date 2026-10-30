# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part VII (fourth part): external data as an auxiliary signal (7.7)."""

from notebooks.notebook_builder import code, md

EXT_SOURCES = '''
# The external sources of the first TDI models, counted from the files the project cached. Each is a set of compounds with a reading of how strongly, or whether, they inhibit a cytochrome enzyme, from an assay
# other than the challenge's. None has a time-dependent-inhibition label. For each source the table gives what is measured, how many compounds are left after the loaders of the modules collapse repeated readings
# (7.7 explains why that matters), and the share of actives where the source has a binary outcome.
from cypmode.challenge.bindingdb_auxiliary import load_bindingdb_isoform
from cypmode.challenge.chembl_auxiliary import load_chembl_isoform
from cypmode.challenge.pubchem_auxiliary import AID1851_RAW_PATH, AID1851_ISOFORMS, SINGLE_ISOFORM_AID_PATHS, load_aid1851, load_single_isoform_aid

INPUTS = ROOT / "data/tdi_track/inputs"
aid1851 = load_aid1851(ROOT / AID1851_RAW_PATH)
source_rows = []
for iso in ("CYP3A4", "CYP2D6"):
    active = aid1851[f"{iso}_active"]
    source_rows.append({"source": "PubChem AID 1851, the five-enzyme panel (NCATS qHTS)", "enzyme": iso, "measures": "inhibition in a luminescence assay, active / inactive / inconclusive", "compounds": int(active.notna().sum()), "share active": float(active.mean())})
for iso in ("CYP3A4", "CYP2D6"):
    single = load_single_isoform_aid(ROOT / SINGLE_ISOFORM_AID_PATHS[iso], iso)
    source_rows.append({"source": f"PubChem AID {884 if iso == 'CYP3A4' else 891}, the single-enzyme screen", "enzyme": iso, "measures": "the same assay, one enzyme", "compounds": int(single[f"{iso}_active"].notna().sum()), "share active": float(single[f"{iso}_active"].mean())})
for iso, uniprot, chembl in (("CYP3A4", "P08684", "CHEMBL340"), ("CYP2D6", "P10635", "CHEMBL289")):
    bindingdb = load_bindingdb_isoform(INPUTS / f"bindingdb_cache/{iso}_{uniprot}.json")
    chembl_table = load_chembl_isoform(INPUTS / f"chembl_cache/{iso}_{chembl}.json")
    source_rows.append({"source": "BindingDB", "enzyme": iso, "measures": "Ki, IC50 or EC50 from many papers, combined by the median", "compounds": len(bindingdb), "share active": np.nan})
    source_rows.append({"source": "ChEMBL", "enzyme": iso, "measures": "pChEMBL value from many papers, combined by the median", "compounds": len(chembl_table), "share active": np.nan})
for iso in ("cyp3a4", "cyp2d6"):
    veith = pd.read_csv(INPUTS / f"tdc_cache/{iso}_veith.tab", sep="\\t")
    source_rows.append({"source": "TDC / Veith panel (the same NCATS screen, curated)", "enzyme": iso.upper(), "measures": "inhibitor or not", "compounds": len(veith), "share active": float(veith["Y"].mean())})
source_table = pd.DataFrame(source_rows)
show_table(source_table, floatfmt=".3f")
n_aid_unique = len(aid1851)
'''

EXT_BUG = '''
# A data-handling bug that the project met, reproduced. The raw file of AID 1851 has one row per compound and panel (the five enzymes), but some pairs have several rows, replicate measurements of the same sample. The loader
# builds one row per compound by merging the five panels one after another on the compound identifier. Done without first keeping one row per compound and panel, each merge multiplies the rows of the compounds that
# have replicates, and the five merges multiply them one on top of the other. The cell counts the rows both ways.
raw_aid = pd.read_csv(ROOT / AID1851_RAW_PATH, skiprows=[1, 2, 3, 4], low_memory=False)
raw_aid = raw_aid.dropna(subset=["PUBCHEM_EXT_DATASOURCE_SMILES"]).rename(columns={"PUBCHEM_EXT_DATASOURCE_SMILES": "SMILES"})
per_pair = raw_aid.groupby(["PUBCHEM_CID", "Panel Name"]).size()
naive = raw_aid[["PUBCHEM_CID", "SMILES"]].drop_duplicates(subset=["PUBCHEM_CID"]).reset_index(drop=True)
for iso, panel_name in AID1851_ISOFORMS.items():
    panel = raw_aid[raw_aid["Panel Name"] == panel_name][["PUBCHEM_CID", "PUBCHEM_ACTIVITY_OUTCOME"]].rename(columns={"PUBCHEM_ACTIVITY_OUTCOME": f"{iso}_outcome"})
    naive = naive.merge(panel, on="PUBCHEM_CID", how="left")
bug_rows = [{"quantity": "rows of the raw file", "value": len(raw_aid)}, {"quantity": "distinct compounds", "value": raw_aid["PUBCHEM_CID"].nunique()},
            {"quantity": "(compound, panel) pairs with more than one row", "value": int((per_pair > 1).sum())}, {"quantity": "most rows for one pair", "value": int(per_pair.max())},
            {"quantity": "rows after the five merges without collapsing", "value": len(naive)}, {"quantity": "rows after collapsing first (the loader)", "value": len(aid1851)}]
show_table(pd.DataFrame(bug_rows))
bug_values = {row["quantity"]: row["value"] for row in bug_rows}
inflation = bug_values["rows after the five merges without collapsing"] / bug_values["rows after collapsing first (the loader)"]
rows_per_compound = naive.groupby("PUBCHEM_CID").size()
most_copies = int(rows_per_compound.max())
'''

EXT_OVERLAP = '''
# Matching a compound across datasets. Plain canonical SMILES write a salt and its free base as two different strings, so an overlap check built on them misses a compound that is in both files. The standardised parent structure
# (cypmode/challenge/structure_standardization.py, the structure pipeline of ChEMBL: salts stripped, charges neutralised) puts them together. The cell shows the two spellings of one drug, then counts the compounds of AID 1851
# that are also in the challenge's training or test file by each way of matching.
from rdkit import Chem

from cypmode.challenge.structure_standardization import standardized_parent_smiles

ketoconazole_free = "CC(=O)N1CCN(CC1)c1ccc(OC[C@H]2CO[C@](Cn3ccnc3)(O2)c2ccc(Cl)cc2Cl)cc1"
ketoconazole_salt = ketoconazole_free + ".Cl"
def plain_canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None
spelling_rows = [{"spelling": "plain canonical SMILES", "free base": plain_canonical(ketoconazole_free), "hydrochloride": plain_canonical(ketoconazole_salt), "same string": plain_canonical(ketoconazole_free) == plain_canonical(ketoconazole_salt)},
                 {"spelling": "standardised parent", "free base": standardized_parent_smiles(ketoconazole_free), "hydrochloride": standardized_parent_smiles(ketoconazole_salt), "same string": standardized_parent_smiles(ketoconazole_free) == standardized_parent_smiles(ketoconazole_salt)}]
show_table(pd.DataFrame(spelling_rows))
test_frame = pd.read_csv(ROOT / "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv")
challenge_smiles = pd.concat([train_file["SMILES"], test_frame["SMILES"]])
plain_overlap = int(aid1851["SMILES"].map(plain_canonical).isin(set(challenge_smiles.map(plain_canonical).dropna())).sum())
standardised_overlap = int(aid1851["SMILES"].map(standardized_parent_smiles).isin(set(challenge_smiles.map(standardized_parent_smiles).dropna())).sum())
print(f"compounds of AID 1851 that are also in the challenge files: {plain_overlap} by plain canonical SMILES, {standardised_overlap} by standardised parent, of {len(aid1851):,}")
salt_spelling_differs = not spelling_rows[0]["same string"]
'''

EXT_RESULTS = '''
# What the external signals did. Each source is turned into one column or a few (a LightGBM model fitted on the external compounds only, applied to every challenge compound; no challenge label is involved and the external
# compounds that are also challenge compounds are removed first), and added to M3. Two sets of results: the ablations of the day, five folds each, and the repeated-partition tests (5 x 5) of BindingDB (stacking_significance)
# and of AID 1851 and ChEMBL (external_auxiliary_significance), re-analysed with the paired corrected t of 6.5 and 6.6.
from cypmode.challenge.model_comparison import corrected_resampled_t
from cypmode.challenge.stacking_significance import paired_comparison_summary

single_split_rows = []
for source, file_name, aux_column in (("AID 1851", "aid1851_auxiliary_ablation", "mcc_m3_plus_aux"), ("AID 884 / 891", "aid891_884_auxiliary_ablation", "mcc_m3_plus_aux"), ("BindingDB", "bindingdb_auxiliary_ablation", "mcc_m3_plus_bindingdb"),
                                      ("ChEMBL", "chembl_auxiliary_ablation", "mcc_m3_plus_aux"), ("AID 1851 + BindingDB + KNN", "combined_features_ablation", "mcc_combined")):
    table = pd.read_csv(COMPARISONS / f"{file_name}.csv")
    for iso, part in table.groupby("isoform"):
        difference = (part[aux_column] - part["mcc_m3"]).to_numpy()
        single_split_rows.append({"source": source, "enzyme": iso, "MCC of M3": float(part["mcc_m3"].mean()), "MCC with the source": float(part[aux_column].mean()), "change": float(difference.mean()), "p (corrected t, 5 folds)": corrected_resampled_t(difference, 4, 1).p_value})
single_split = pd.DataFrame(single_split_rows)
show_table(single_split, floatfmt=".3f")
repeated_rows = []
for iso, directory in (("CYP3A4", "cyp3a4"), ("CYP2D6", "cyp2d6")):
    raw = pd.read_csv(ROOT / f"data/tdi_track/outputs/{directory}/external_auxiliary_significance_raw.csv")
    summary = paired_comparison_summary(raw)
    for _, row in summary[summary["group1"] == "M3"].iterrows():
        repeated_rows.append({"experiment": "AID 1851 and ChEMBL (5 x 5)", "enzyme": iso, "comparison": f"{row['group1']} to {row['group2']}", "mean change": row["meandiff"], "p (corrected t, Holm)": row["p-adj"]})
stacking_raw = pd.read_csv(COMPARISONS / "stacking_significance_raw.csv")
for iso, part in stacking_raw.groupby("isoform"):
    summary = paired_comparison_summary(part)
    for _, row in summary[(summary["group1"] == "M3") & (summary["group2"].str.contains("BindingDB"))].iterrows():
        repeated_rows.append({"experiment": "BindingDB and KNN (5 x 5)", "enzyme": iso, "comparison": f"{row['group1']} to {row['group2']}", "mean change": row["meandiff"], "p (corrected t, Holm)": row["p-adj"]})
repeated = pd.DataFrame(repeated_rows)
show_table(repeated, floatfmt=".3f")
largest_single = single_split.loc[single_split["change"].idxmax()]
repeated_2d6 = repeated[(repeated["enzyme"] == "CYP2D6") & (repeated["comparison"].str.contains("AID1851|ChEMBL")) & ~repeated["comparison"].str.contains("KNN")]
smallest_repeated_p = float(repeated["p (corrected t, Holm)"].min())
n_single_positive = int((single_split["change"] > 0).sum())
'''


QM_UNIVARIATE = '''
# The two quantum-chemical families, as they were cached for the labelled compounds, and what each number says alone. For each of the 16 columns (the 8 whole-molecule GFN2-xTB numbers of cypmode/challenge/quantum_pharmacophore.py
# and the 8 reactivity numbers of cypmode/challenge/qm_reactivity_features.py) the AUC of the column as a score for the label, in the better of its two directions, with the standard error that an AUC has when the column
# carries no information, sqrt((n + 1) / (12 n+ n-)) for n+ positives and n- negatives (Hanley and McNeil, 1982). Two ordinary descriptors are added for scale. Sixteen columns for each of two enzymes make 32 tests, and
# a column is counted as signal only if its AUC exceeds 0.5 by more than the Bonferroni limit, the 0.05/32 two-sided normal quantile times the standard error.
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors
from scipy import stats

quantum_cache = pd.read_csv(ROOT / "data/tdi_track/outputs/caches/quantum_pharmacophore_cache.csv")
reactivity_cache = pd.read_csv(ROOT / "data/tdi_track/outputs/caches/qm_reactivity_cache.csv")
labelled = tutorial_frame.merge(quantum_cache, on="SMILES", how="left").merge(reactivity_cache, on="SMILES", how="left")
mols_labelled = [Chem.MolFromSmiles(s) for s in labelled["SMILES"]]
labelled["molecular weight"] = [Descriptors.MolWt(m) for m in mols_labelled]
labelled["Crippen logP"] = [Crippen.MolLogP(m) for m in mols_labelled]
quantum_columns, reactivity_columns = [c for c in quantum_cache.columns if c != "SMILES"], [c for c in reactivity_cache.columns if c != "SMILES"]
coverage = {"whole-molecule GFN2-xTB numbers": float(labelled[quantum_columns].notna().all(axis=1).mean()), "reactivity numbers (any)": float(labelled[reactivity_columns].notna().any(axis=1).mean()),
            "a basic nitrogen found (reactivity)": float((labelled["qm_has_basicN"] == 1).mean())}
bonferroni = stats.norm.isf(0.05 / 64)
auc_rows = []
for iso in TDI_ISOFORMS:
    column = f"{iso}_is_TDI"
    for name in [*quantum_columns, *reactivity_columns, "molecular weight", "Crippen logP"]:
        known = labelled[[column, name]].dropna()
        y = known[column].astype(int).to_numpy()
        auc = roc_auc_score(y, known[name].to_numpy())
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        standard_error = float(np.sqrt((n_pos + n_neg + 1) / (12 * n_pos * n_neg)))
        auc_rows.append({"enzyme": iso, "column": name, "n": len(known), "AUC (better direction)": max(auc, 1 - auc), "standard error under no information": standard_error, "signal after Bonferroni": bool(abs(auc - 0.5) > bonferroni * standard_error)})
auc_table = pd.DataFrame(auc_rows)
show_table(auc_table.sort_values(["enzyme", "AUC (better direction)"], ascending=[True, False]).groupby("enzyme").head(6), floatfmt=".3f")
n_signal_quantum = int(auc_table[auc_table["column"].isin([*quantum_columns, *reactivity_columns])]["signal after Bonferroni"].sum())
best_quantum = auc_table[auc_table["column"].isin([*quantum_columns, *reactivity_columns])].sort_values("AUC (better direction)", ascending=False).iloc[0]
descriptor_aucs = auc_table[auc_table["column"].isin(["molecular weight", "Crippen logP"])].set_index(["enzyme", "column"])["AUC (better direction)"]
'''

QM_NULLS = '''
# The three feature families against the plain model, in the repeated-partition design (5 x 5; section 6.6), re-analysed with the paired corrected t: the change of the MCC from adding the family to M3, with the Holm-adjusted
# p-value over the six pairs of its experiment.
from cypmode.challenge.stacking_significance import paired_comparison_summary

null_rows = []
for family, file_stem, label in (("GFN2-xTB whole-molecule numbers", "quantum_pharmacophore", "Quantum"), ("qmdesc reactivity numbers", "qm_reactivity", "QM-Reactivity"), ("three-point pharmacophore fingerprint", "pharmacophore_fingerprint", "Pharmacophore")):
    raw = pd.read_csv(COMPARISONS / f"{file_stem}_significance_raw.csv")
    for iso, part in raw.groupby("isoform"):
        summary = paired_comparison_summary(part)
        row = summary[(summary["group1"] == "M3") & (summary["group2"] == f"M3+{label}")].iloc[0]
        pair = part.pivot_table(index=["repeat", "fold"], columns="variant", values="mcc")
        sd = float((pair[f"M3+{label}"] - pair["M3"]).std(ddof=1))
        floor = float((stats.t.ppf(0.975, 24) + stats.t.ppf(0.8, 24)) * sd * np.sqrt(1 / 25 + 1 / 4))
        null_rows.append({"family": family, "enzyme": iso, "mean MCC change": row["meandiff"], "95 % interval": f"{row['lower']:+.3f} to {row['upper']:+.3f}", "p (corrected t, Holm)": row["p-adj"], "smallest change found 8 times in 10": floor})
null_table = pd.DataFrame(null_rows)
show_table(null_table, floatfmt=".3f")
smallest_null_p = float(null_table["p (corrected t, Holm)"].min())
mde_range = (float(null_table["smallest change found 8 times in 10"].min()), float(null_table["smallest change found 8 times in 10"].max()))
quantum_single = pd.read_csv(COMPARISONS / "quantum_pharmacophore_ablation.csv").groupby("isoform")[["mcc_m3", "mcc_m3_plus_quantum"]].mean()
change_range = (float(null_table["mean MCC change"].min()), float(null_table["mean MCC change"].max()))
'''


CELLS = [
    md("""
## 7.7 Outside data as an auxiliary signal: PubChem, BindingDB, ChEMBL and the cytochrome panel

> **Field:** Cheminformatics · Data science

*Where this section sits:* the training file holds a few thousand compounds with a time-dependent-inhibition label. The literature holds tens of thousands of compounds with some reading of how strongly they inhibit a cytochrome enzyme, none of them with a TDI label. The obvious idea is to use that knowledge. This section explains how an outside source becomes a feature without leaking, shows two mistakes in handling such data that the project made and caught (one of them changed the sign of a result), and reports what the sources gave. It continues the sources of section 2.8.

**How an outside source becomes a feature.** The sources measure direct inhibition or binding, not the shift between the two assay arms. What they offer is therefore a *correlate* of the label, since a potent inhibitor is more often also a time-dependent one, and not a label. The recipe has three steps. A model is fitted on the source, structure in and the source's outcome out (here a LightGBM classifier on the Morgan bits, section 3.1, for an active-or-not outcome, a regressor for a potency). The model is then applied to every challenge compound, training and blinded alike, and its prediction becomes a column of the classifier. Before the fitting, every source compound that is also a challenge compound is removed. That removal is what makes the column safe to compute once: the prediction for a challenge compound comes from a model that has never seen it or its reading, as a prediction for a new compound would, so the column is out of sample by construction, no challenge label is involved, and no nesting is needed. This is the difference from the bridge of 7.4, where the model that produced the column was trained on the very compounds the classifier was trained on, so that the construction had to be made out of fold.

The first cell lists the sources and counts the compounds in each, after the loaders of the project have collapsed repeated readings of the same compound, which is the first of the two mistakes below.
"""),
    code(EXT_SOURCES),
    md("""
The five-enzyme panel of PubChem AID 1851 (an AID is the number that PubChem gives to an assay; this one is an NCATS screen read by a luminescence assay) gives {{=n_aid_unique:,}} compounds, with an activity outcome for {{=int(source_table.iloc[0]['compounds']):,}} of them on CYP3A4 and {{=int(source_table.iloc[1]['compounds']):,}} on CYP2D6; the single-enzyme screens AID 884 and 891 are the individual assays from which part of that panel was drawn; BindingDB and ChEMBL are literature-curated and hold affinities from many papers, so that a compound can have several readings. The TDC version of the Veith panel (the same NCATS screen, curated into a benchmark) is the source of the block used in the final model (below). The positive rates are far from the 21 percent of the challenge: {{=source_table.iloc[0]['share active']:.0%}} of the AID 1851 compounds are active on CYP3A4, a library of drug-like and bioactive compounds and not a set of compounds promoted to dose-response.

**The first mistake: counting replicate rows as compounds.** The raw file of AID 1851 has one row for each compound and panel, but some pairs have several, replicate readings of the same sample. The loader makes one row per compound by merging the five panels one after another on the compound identifier. If the panels are not first reduced to one row per compound, every merge multiplies the rows of the compounds that have replicates, and the five merges multiply them on top of one another. The cell reproduces the count.
"""),
    code(EXT_BUG),
    md("""
The file has {{=bug_values['rows of the raw file']:,}} rows for {{=bug_values['distinct compounds']:,}} compounds; {{=bug_values['(compound, panel) pairs with more than one row']:,}} (compound, panel) pairs have more than one row, up to {{=bug_values['most rows for one pair']}}. Merged without collapsing, the five panels give {{=bug_values['rows after the five merges without collapsing']:,}} rows, {{=inflation:.1f}} times the {{=bug_values['rows after collapsing first (the loader)']:,}} compounds, and the most replicated compound appears in {{=most_copies:,}} rows: seven replicates in each of the five panels multiply to {{=7 ** 5:,}}, a single compound that makes up about a fifth of the table. A model fitted on those rows counts that compound {{=most_copies:,}} times, and counts every compound with replicates more than the others. The record of the day shows what this did: the first version of the AID 1851 feature *lowered* the CYP2D6 MCC, to 0.020, and the corrected one raised it. The error was found because the surprising sign was investigated before it was believed, the discipline that the project had applied to leakage and now applied to a data pipeline: an unexpected result is first a question about the pipeline. The same lesson, in the other direction, belongs to 7.4.

**The second mistake: a salt is not a different compound.** The overlap between a source and the challenge files must be removed, and the overlap is found by comparing structures. Plain canonical SMILES write a compound and its salt as different strings. The standardised parent structure, produced by the structure pipeline that ChEMBL uses to curate its own entries (salts stripped, charges neutralised; `cypmode/challenge/structure_standardization.py`), writes them alike. The cell shows one drug in both forms, and counts how many compounds of AID 1851 are also in the challenge's training or test file by each way of matching.
"""),
    code(EXT_OVERLAP),
    md("""
Ketoconazole and its hydrochloride are two strings under plain canonical SMILES ({{=salt_spelling_differs}}) and one under the standardised parent. Of the {{=n_aid_unique:,}} compounds of AID 1851, {{=plain_overlap}} are found in the challenge files by plain matching and {{=standardised_overlap}} by standardised matching: {{=standardised_overlap - plain_overlap}} compounds that a plain overlap check would have left in the external training set, each of them a challenge compound whose reading the auxiliary model would have seen. The difference is small and the principle is not: an overlap check is a leakage control, and a leakage control that misses a spelling of the same molecule does not control the leak. `exclude_exact_overlap` in the loaders of PubChem, BindingDB and ChEMBL, and the overlap check of `cypmode/challenge/split_validity.py`, match on the standardised parent.

**What the sources gave.** The last cell sets the results of the day, five folds each, beside the repeated-partition tests. Every number is the change of the MCC when the source's column is added to M3, and the p-value is the corrected paired t of 6.5, for five folds in the first table and with the Holm adjustment over the pairs in the second.
"""),
    code(EXT_RESULTS),
    md("""
On the single partition of five folds, {{=n_single_positive}} of the {{=len(single_split)}} source and enzyme pairs have a positive change. The largest is {{=largest_single['change']:+.3f}} MCC ({{=largest_single['source']}}, {{=largest_single['enzyme']}}), with a corrected p-value of {{=largest_single['p (corrected t, 5 folds)']:.2f}}, and the two sources of the 'consistent' CYP2D6 bump, AID 1851 and ChEMBL, are at {{=single_split[(single_split['source'] == 'AID 1851') & (single_split['enzyme'] == 'CYP2D6')]['change'].iloc[0]:+.3f}} and {{=single_split[(single_split['source'] == 'ChEMBL') & (single_split['enzyme'] == 'CYP2D6')]['change'].iloc[0]:+.3f}}. The project of the day took the agreement of two independent sources on one effect size to be a stronger signal than anything it had seen for CYP2D6, and, on the rule that a surprising result is tested before it is believed, repeated both over 25 folds. In the repeated design the same two comparisons are {{=repeated_2d6.iloc[0]['mean change']:+.3f}} (AID 1851) and {{=repeated_2d6.iloc[2]['mean change']:+.3f}} (ChEMBL), and the smallest corrected p-value of the second table is {{=smallest_repeated_p:.2f}}. The two sources were not independent in the way that mattered: both were scored on the same five folds, a partition that happened to favour them together, and two results from one partition are one result. The same holds for BindingDB, whose single-partition CYP2D6 change of {{=single_split[(single_split['source'] == 'BindingDB') & (single_split['enzyme'] == 'CYP2D6')]['change'].iloc[0]:+.3f}} is as far below zero as the others are above it.

An external signal still entered the final model, for a different reason from the one it was first tried for. The TDC version of the Veith panel feeds the block `ext_activity` of the feature store (section 3.7): five probabilities of inhibition, for CYP1A2, 2C9, 2C19, 2D6 and 3A4, from models trained on the panels only, with every challenge compound removed from the external training set (`cypmode/challenge/external_activity.py`). The module's own reasoning is about the population and not about a gain in the folds: the challenge's training pool consists of compounds that were promoted to dose-response, so that it knows little of what is *inactive* (for CYP2D6 only about 9 percent of its compounds fall below the assay floor), while the blinded set is full of such compounds; an external panel that knows the inactive space is a prior for exactly the region where the training file is thin. The block is part of three of the 23 default members of the stack (`lgbm_A+ext_deep`, `lgbm_A+fps+ext_deep` and `lgbm_A+fps+ext+alerts_deep`, section 8.1), and what it adds to the stack is judged by the gates of Part IX.

*On what actually ran.* `cypmode/challenge/pubchem_auxiliary.py` (AID 1851, 884 and 891; `load_aid1851` collapses the replicates), `bindingdb_auxiliary.py` and `chembl_auxiliary.py` (targeted pulls for one enzyme, cached as JSON in `data/tdi_track/inputs/`, with `fetch_*` functions that talk to the services and `load_*` functions that read the cache), `external_activity.py` (the Veith panels and the block), `structure_standardization.py`, `combined_features.py` (the first stacking of the signals) and `external_auxiliary_significance.py` and `stacking_significance.py` (the 5 x 5 tests). The auxiliary models are fitted once, on the external compounds, and applied alike to every challenge compound. The stored tables are the ablations of the day (`aid1851_auxiliary_ablation.csv`, `aid891_884_auxiliary_ablation.csv`, `bindingdb_auxiliary_ablation.csv`, `chembl_auxiliary_ablation.csv`, `combined_features_ablation.csv`) and the raw fold scores of the repeated tests. The ablations are on the mixed population of section 2.4.
"""),
    md("""
## 7.8 Electrons and pharmacophores: the quantum-chemical features and the three-point fingerprint

> **Field:** Quantum chemistry · Cheminformatics

*Where this section sits:* the descriptors and fingerprints of Part III describe how a molecule is *connected*. A cytochrome oxidises a molecule by taking electrons from it, and how readily a molecule gives up electrons is a property of its electrons and not of its connections. This section tries three ways of giving the classifier that kind of information: numbers from an approximate quantum calculation of the whole molecule, numbers from a neural network that imitates a better quantum calculation atom by atom, and the three-point pharmacophore fingerprint of section 3.5. All three were mechanistically sensible, all three were tested properly, and all three were null. The theory of the calculations is left to Part X, which returns to the same machinery with more of it and a different design; this section says only what the numbers are, and what they did.

**What the numbers are.** The electrons of a molecule occupy **orbitals**: regions of space, each with an energy, which the electrons fill from the lowest energy upwards, two to an orbital. The highest filled orbital is the **HOMO** (highest occupied molecular orbital), and its energy says how loosely the most loosely held electrons are bound: a high HOMO energy means a molecule that gives up an electron easily, which is the first step of many oxidations, including the ones the iron-oxygen species of the cycle of section 1.3 carries out. The lowest empty orbital is the **LUMO**, and the gap between the two measures how easily the molecule is excited or reacts. Energies are given in **hartree**, the atomic unit of energy (one hartree is 27.2 electronvolts). The **dipole moment** measures how unevenly the charge is spread, a molecule with a positive end and a negative end having a large one. A **partial charge** is the share of electronic charge that an atom holds compared with the neutral atom; the charges of a molecule add up to its net charge. A **Fukui index** is the change of the electron density at an atom when an electron is added or removed, and so marks the atoms that give up or accept electrons first: the project's network returns one index for an atom acting as a nucleophile, a donor of a lone pair (`fukui_neu`), and one for an atom acting as an electrophile (`fukui_elec`).

Two ways of obtaining the numbers were used. A **semi-empirical method** is an approximate quantum calculation whose parameters are fitted to reference results so that it can be run about a thousand times faster than an exact one; the method used, GFN2-xTB (Bannwarth, Ehlert and Grimme, 2019; xTB stands for extended tight binding, and GFN for the geometries, frequencies and noncovalent interactions that its parameters were fitted to), takes about one second per molecule on one processor core. For each compound the project makes a three-dimensional shape (the embedding of RDKit followed by a force-field optimisation), runs the calculation once and keeps eight numbers: the HOMO and LUMO energies, their gap, the dipole moment, the total energy, and the smallest, the largest and the spread of the atomic partial charges. A **surrogate model** is a neural network trained to reproduce a calculation, here a graph network (the `qmdesc` network of Guan and co-workers) trained on results of density-functional calculations; it gives per-atom partial charges and Fukui indices from the structure alone in milliseconds. The project reads it at the nitrogen atoms that the amine screen of section 5.2 selects, and over the whole molecule, and keeps eight numbers: three at the most reactive basic nitrogen (and a flag that says whether there is one), and the mean and the largest of the two Fukui indices over all atoms. Where the compound has no such nitrogen the three values are left empty and the flag is 0, so that an empty value never claims a measurement that was not made.

The numbers were computed once for every compound and cached, and the notebook reads the caches, for a reason that is a lesson of its own. The calculation library (`tblite`) and the gradient-boosting library (LightGBM) each ship their own parallel-computing runtime, and loaded into one process they deadlock: the program stops at 0 percent processor use without an error. The same kind of conflict exists between the neural-network library and LightGBM, where it ends in a crash. The project's answer is a rule that no module imports both: each extraction runs in a process of its own and writes a cache, and the evaluation runs in another process and only reads it. The audit of September (finding 3) found one module that broke the rule and had not yet failed only because its cache happened to be complete. This notebook, which loads LightGBM, therefore cannot recompute the quantum numbers; it reads what the extraction processes wrote.

The cell asks the simplest question of each cached column: how well does it separate the labels on its own?
"""),
    code(QM_UNIVARIATE),
    md("""
The caches cover {{=coverage['whole-molecule GFN2-xTB numbers']:.1%}} of the labelled compounds with all eight whole-molecule numbers and {{=coverage['reactivity numbers (any)']:.1%}} with the reactivity numbers; {{=coverage['a basic nitrogen found (reactivity)']:.1%}} have a basic nitrogen in the sense of the amine pattern as it was when the cache was made (the old pattern of 7.2). Taken one at a time the numbers are weak. The best of the 16 quantum columns, {{=best_quantum['column']}} on {{=best_quantum['enzyme']}}, has an AUC of {{=best_quantum['AUC (better direction)']:.3f}}, and {{=n_signal_quantum}} of the 32 column and enzyme pairs exceed 0.5 by more than the Bonferroni limit. For scale, the molecular weight alone has an AUC of {{=descriptor_aucs[('CYP3A4', 'molecular weight')]:.3f}} on CYP3A4 and {{=descriptor_aucs[('CYP2D6', 'molecular weight')]:.3f}} on CYP2D6, and the Crippen logP {{=descriptor_aucs[('CYP3A4', 'Crippen logP')]:.3f}} and {{=descriptor_aucs[('CYP2D6', 'Crippen logP')]:.3f}}. A column that is weak alone can still help in a model, because it can carry what the other columns do not; the repeated-partition tests of the three families, M3 against M3 plus the family, answer that.
"""),
    code(QM_NULLS),
    md("""
No family changes the MCC by more than {{=max(abs(change_range[0]), abs(change_range[1])):.3f}} in either direction, every interval contains zero, and the smallest adjusted p-value of the six comparisons is {{=smallest_null_p:.2f}} (the adjustment multiplies a p-value by the number of comparisons of its experiment, and caps it at 1). The single-partition result of the day had pointed the same way for the whole-molecule numbers (CYP3A4 from {{=quantum_single.loc['CYP3A4', 'mcc_m3']:.3f}} to {{=quantum_single.loc['CYP3A4', 'mcc_m3_plus_quantum']:.3f}}, CYP2D6 from {{=quantum_single.loc['CYP2D6', 'mcc_m3']:.3f}} to {{=quantum_single.loc['CYP2D6', 'mcc_m3_plus_quantum']:.3f}} on the five folds of the stored table `quantum_pharmacophore_ablation.csv`), and the project's rule, applied after the lessons of 7.2 and 7.7, was to take such a result to 25 folds before believing it in either direction. The reading is the one the table supports and no more: not harmful, not helpful, and indistinguishable from nothing at the size of this data. Two readings of the null are possible. The numbers might carry no information about time-dependent inhibition that the 217 descriptors and the fingerprint do not already carry; or the information is there and the design cannot see an effect smaller than its floor, which for the 25 folds of these tests, with the spread of their own fold differences, is between {{=mde_range[0]:.2f}} and {{=mde_range[1]:.2f}} MCC (the formula of section 6.7). The data do not separate the two, and the question was reopened in Part X with a much larger set of atom-level and site-level descriptors, a design that could see smaller effects (the nested one of Part VI), and a pre-declared gate; the answer there was the same.

The three-point pharmacophore fingerprint of section 3.5 is the third null of the table, and the one that came with the cleanest story. It encodes the arrangement in space of hydrogen-bond donors and acceptors, aromatic rings, hydrophobic regions and ionisable groups, which neither the connectivity fingerprint nor the scalar quantum numbers capture; it costs about 0.35 second per compound (about 36 minutes for the training file on one core); and it leaned slightly negative (the last two rows of the table, with a mean change of {{=null_table[null_table['family'].str.contains('pharmacophore')]['mean MCC change'].mean():+.3f}} MCC), as the others did. The audit found that two compounds of the file had an all-zero fingerprint, the sign of a failed embedding and not the claim that there is no pharmacophoric feature, and that the repeated test had scored them as if the zero were a measurement; the test now leaves such rows out. The effect was small here, and it is exactly the kind of error that a larger share of failures, such as unusual chemotypes in the blinded set, would make large.

These three nulls, with the five of 7.7 and the ChemBERTa result of 7.9, make the pattern that the project recorded as its main early finding: none of the hand-built or externally derived features moved the baseline by an amount that survives a proper test. The record offers an explanation that the next section examines and that Part X revisits, that the ceiling is the combination of the classical descriptors with the gradient-boosted tree rather than a missing feature.

*On what actually ran.* `cypmode/challenge/quantum_pharmacophore.py` (the 3D embedding, the GFN2-xTB calculation through the `tblite` library, the eight numbers, a cache keyed by SMILES that a failed compound gets NaN in), `quantum_pharmacophore_significance.py` (the evaluation, in its own process: the audit's finding was that the extraction and the evaluation had been in one module), `qm_reactivity_features.py` and `qm_reactivity_significance.py` (the surrogate network `qmdesc`, patched at installation for two incompatibilities with newer libraries: it imported `pkg_resources`, which modern setuptools removed, and its checkpoint stores a pickled argument namespace that newer PyTorch refuses to load by default; the second is fixed by allowing that one class and not by disabling the safety check), `pharmacophore_fingerprint.py` and the caches in `data/tdi_track/outputs/caches/`. The stored 5 x 5 tables are the `*_significance_raw.csv` files of `comparisons/`. These experiments are on the mixed population of section 2.4.
"""),
]
