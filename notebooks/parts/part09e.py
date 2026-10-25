# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part IX (fifth part): the pharmacophore read on molecules, in 2D and in 3D, with the functional groups and the structure-activity relationships, for both tracks (9.15)."""

from notebooks.notebook_builder import code, figure, md

ATLAS_FACTS = r'''
# The numbers of 9.15: the share of time-dependent inhibitors among the compounds that carry each functional group, and the matched pairs that gain one group, for both enzymes. The groups are the fixed table of
# cypmode/challenge/functional_groups.py; the pairs are the ones that the matched-pair features of 9.2 used (the cache of cypmode/challenge/mmp_transfer.py).
import numpy as np
import pandas as pd

from cypmode.challenge import functional_groups as groups_module
from cypmode.challenge.mmp_transfer import get_pairs
from cypmode.challenge.tdi_targets import BOTH_ARMS, training_view

train_file = pd.read_csv(ROOT / "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv")
test_file = pd.read_csv(ROOT / "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv")
all_smiles = list(train_file["SMILES"]) + list(test_file["SMILES"])
pairs_cache = get_pairs(all_smiles)
group_tables, pair_tables, base_rates, n_pool = {}, {}, {}, {}
for enzyme in ("CYP3A4", "CYP2D6"):
    view = training_view(train_file, enzyme)
    keep = (view["population"] == BOTH_ARMS).to_numpy()
    labels = view["label"][keep].astype(int).to_numpy()
    base_rates[enzyme], n_pool[enzyme] = float(labels.mean()), int(len(labels))
    group_tables[enzyme] = groups_module.group_rates(groups_module.group_matrix(train_file["SMILES"][keep].tolist()), labels).set_index("group")
    pair_tables[enzyme], pair_detail = groups_module.group_gain_pairs(all_smiles, pairs_cache, np.flatnonzero(keep), labels, 5)
    pair_tables[enzyme] = pair_tables[enzyme].set_index("group")
    n_single_gain = {enzyme: int((pair_detail["gained"] >= 0).sum())} if enzyme == "CYP3A4" else {**n_single_gain, enzyme: int((pair_detail["gained"] >= 0).sum())}


def group_rate(group, enzyme):
    """(rate, number of compounds) of a group for an enzyme, (nan, 0) when fewer than 10 compounds carry it."""
    table = group_tables[enzyme]
    return (float(table.loc[group, "rate"]), int(table.loc[group, "n"])) if group in table.index else (float("nan"), 0)


def group_gain(group, enzyme):
    """(mean change of the label, standard error, number of pairs) of the pairs that gain a group, (nan, nan, 0) when fewer than 5 pairs do."""
    table = pair_tables[enzyme]
    return (float(table.loc[group, "mean"]), float(table.loc[group, "se"]), int(table.loc[group, "pairs"])) if group in table.index else (float("nan"), float("nan"), 0)


assert group_tables["CYP3A4"].index[0] == "methylenedioxyphenyl" and group_tables["CYP2D6"].index[0] == "methylenedioxyphenyl", "the text of 9.15 says that this group has the highest rate on both enzymes"
mdp_3a4, mdp_3a4_n = group_rate("methylenedioxyphenyl", "CYP3A4")
mdp_2d6, mdp_2d6_n = group_rate("methylenedioxyphenyl", "CYP2D6")
acid_3a4, acid_3a4_n = group_rate("carboxylic acid", "CYP3A4")
azole_3a4, azole_3a4_n = group_rate("azole or pyridine (heme binder)", "CYP3A4")
azole_2d6, azole_2d6_n = group_rate("azole or pyridine (heme binder)", "CYP2D6")
thiophene_3a4, thiophene_3a4_n = group_rate("thiophene", "CYP3A4")
gain_tbutyl, gain_tbutyl_se, gain_tbutyl_n = group_gain("tert-butyl", "CYP3A4")
gain_acid, gain_acid_se, gain_acid_n = group_gain("carboxylic acid", "CYP3A4")
assert thiophene_3a4 > base_rates["CYP3A4"] and group_rate("furan", "CYP3A4")[0] > base_rates["CYP3A4"], "the text says that thiophenes and furans are above the base rate on CYP3A4"
assert acid_3a4 < 0.1 and azole_3a4 < base_rates["CYP3A4"] + 0.03 and azole_2d6 < base_rates["CYP2D6"] + 0.03, "the text says that acids are rare among inhibitors and that heme binders are not enriched"
assert gain_tbutyl > 0 > gain_acid, "the text reads the direction of these two changes"
n_groups_3a4, n_groups_2d6 = len(pair_tables["CYP3A4"]), len(pair_tables["CYP2D6"])
gain_summary = pair_tables["CYP3A4"].reset_index()[["group", "pairs", "mean", "se"]].head(8).round(2)
'''

ATLAS_TABLE = r'''
gain_summary
'''

OFFICIAL_ATLAS_FACTS = r'''
# The 20 official ligands in 2D and in 3D (job 925; HELD BACK: it describes the final poses of blinded ligands), read from the folder that the job wrote.
import pandas as pd

official_dir = ROOT / "data/structure_track/outputs/cyp3a4/pharmacophore_official20"
atlas_summary = pd.read_csv(official_dir / "summary.csv")
atlas_donors = atlas_summary[atlas_summary.donor_error.notna()]
atlas_non_donors = atlas_summary[atlas_summary.donor_error.isna()]
n_atlas_donors, n_atlas_non_donors = len(atlas_donors), len(atlas_non_donors)
n_donor_close = int((atlas_donors.donor_error <= 0.2).sum())
n_site_outside = int((atlas_non_donors.site_error > 0).sum())
recall_donors, recall_non_donors = float(atlas_donors.recall.median()), float(atlas_non_donors.recall.median())
'''

CELLS = [
    code(ATLAS_FACTS, hide_input=True),
    md("""
## 9.15 Reading the pharmacophore on molecules: functional groups, structure-activity relationships and the pocket, in 2D and in 3D

> **Field:** Chemistry · Cheminformatics · Structural biology · Pharmacology

*Where this section sits:* 9.13 and 9.14 describe the pharmacophore of a compound with fields, sites and hypotheses, and 11.11 does the same for a predicted pose. Those descriptions are numbers. This section draws them: it puts each representation and each model on real molecules, flat on the page and in space, and connects them to the two things that a chemist reads first in a structure, its **functional groups** and the **structure-activity relationship** (SAR) that they carry. It covers both tracks, so that the reader can follow one idea, a feature of a molecule, from the formula of a compound to its place in a pocket.

**Three words.** A **functional group** is a small, named arrangement of atoms that behaves as a unit in a molecule: the carboxylic acid $-\\mathrm{COOH}$, the amide $-\\mathrm{CONH}-$, a chlorine on an aromatic ring. A **pharmacophore feature** is what a group (or an atom, or a ring) offers to a binding site: a hydrogen-bond donor, an acceptor, an aromatic ring, a hydrophobic patch, a group that is positively or negatively charged at the pH of the body. One functional group usually offers several features (an amide is both a donor and an acceptor), and the table of `cypmode/challenge/functional_groups.py` says which. A **structure-activity relationship** is the way a change in the structure of a molecule changes its activity. It is stated as a rule of thumb ("adding a tert-butyl group makes a compound a better time-dependent inhibitor") and tested on **matched molecular pairs**, two compounds that differ by one small fragment at the same place, so that nothing else changed.

**The functional groups of the table.** The table was written before any rate was looked at and is not changed: eighteen groups, each a SMARTS pattern (the pattern language of 2.1) with the families of features that it carries and, where the literature names one, why it matters for a cytochrome P450. The first figure takes each group in turn and asks how often the compounds that carry it are time-dependent inhibitors. For CYP3A4, {{=n_pool['CYP3A4']}} labelled compounds, of which {{=base_rates['CYP3A4']:.0%}} are inhibitors, and for CYP2D6, {{=n_pool['CYP2D6']}} compounds with {{=base_rates['CYP2D6']:.0%}}; the interval drawn on each rate is the **Wilson interval**, the range of values of a proportion that the counts do not exclude, which unlike the interval of a mean plus or minus two standard errors cannot leave the range from 0 to 1 when a group is rare.
"""),
    code(ATLAS_TABLE),
    figure("pharm_functional_groups"),
    md("""
**What the figure says.** The group with the highest rate on CYP3A4 is the methylenedioxyphenyl ring, with {{=mdp_3a4:.0%}} of {{=mdp_3a4_n}} compounds inhibiting (the base rate is {{=base_rates['CYP3A4']:.0%}}), and on CYP2D6 it is again the group with the highest rate, {{=mdp_2d6:.0%}} of {{=mdp_2d6_n}}. This is the textbook case of a **mechanism-based inactivator**: the enzyme oxidises the ring, the product is a carbene that binds the heme iron, and the enzyme is lost for as long as the complex lives. Thiophenes ({{=thiophene_3a4:.0%}} of {{=thiophene_3a4_n}} on CYP3A4) and furans are other groups that P450s turn into reactive intermediates, and the figure shows both above the base rate (the furans only just); terminal alkynes, a third such group, are too rare in the data to appear. At the other end, carboxylic acids are almost never inhibitors of CYP3A4 ({{=acid_3a4:.0%}} of {{=acid_3a4_n}}): CYP3A4 prefers neutral and basic compounds and an acid is held outside the pocket. The groups that bind the heme, the azoles and the pyridines, are *not* enriched ({{=azole_3a4:.0%}} of {{=azole_3a4_n}} on CYP3A4, {{=azole_2d6:.0%}} of {{=azole_2d6_n}} on CYP2D6): they inhibit by binding the iron, and a compound that does so is a **reversible** inhibitor whose inhibition is already counted in the direct arm that the label subtracts (1.8). The label of this challenge asks for the loss of activity that grows with time, which needs a chemistry that the heme binder alone does not supply. This is the reason why the heme-binding pharmacophore of 9.13 and 9.14 is a weak predictor of the label, and why a pharmacophore model for this track has to be read as a model of *who is inhibited*, not of *who is inactivated*.
"""),
    md("""
**The structure-activity relationship as a change.** The rates above compare groups of different compounds, so a group can look good only because the compounds that carry it have something else in common. A matched pair removes that confound. `cypmode/challenge/mmp_transfer.py` (the pair analysis of 9.2) finds, by single-bond fragmentation, the pairs of training compounds that differ by one fragment of at most ten heavy atoms. The second figure counts a pair for a functional group when the second compound has the group, the first does not and no other group of the table changes, and gives for each group the mean change of the label over those pairs (0 for a non-inhibitor, 1 for an inhibitor). The pairs are few at the level of the exact exchange of fragments (no exchange has as many as five labelled pairs), which is why they are grouped by the group they gain. CYP3A4 has {{=n_single_gain['CYP3A4']}} labelled pairs that gain exactly one group, {{=n_groups_3a4}} groups with at least five of them, and CYP2D6 has {{=n_single_gain['CYP2D6']}} pairs and {{=n_groups_2d6}} groups.
"""),
    figure("pharm_mmp_sar"),
    md("""
**Reading the pairs.** Gaining a tert-butyl group moves the label by {{=gain_tbutyl:+.2f}} (standard error {{=gain_tbutyl_se:.2f}}, {{=gain_tbutyl_n}} pairs) on CYP3A4, and gaining a carboxylic acid by {{=gain_acid:+.2f}} ({{=gain_acid_se:.2f}}, {{=gain_acid_n}} pairs): for the acid the pairs agree with the low rate of the first figure. For the tert-butyl group they say something that the rates do not: the compounds that carry it are inhibitors as often as the base rate says, yet the pairs that gain it become inhibitors, so that the group co-occurs with other features that hide its effect in the comparison of groups and only the pair, which holds everything else fixed, shows it. A change in a group that moves the label by less than its standard error is not an SAR and is not read as one; the table gives the first eight groups. Each real pair under the bars is the one whose change is the most typical of its group, not the most extreme, and the line under it gives what the pair did to the pharmacophore features: a transformation that changes the label and the features together is a candidate for the pharmacophore explanation, one that changes the label and no feature is not.
"""),
    md("""
**The same compounds as features.** The surest compounds of the training set, drawn twice: the atoms of their functional groups, and the atoms of their pharmacophore features by the RDKit definitions. The two drawings answer different questions. The groups say what a chemist calls the parts of the molecule; the features say what a binding site sees. A ring nitrogen that is part of an azole is, at once, an acceptor, a member of an aromatic ring and the heme-binding atom, and the figure shows that an atom belongs to several features. The 3D blocks of 9.13 and 9.14 place exactly these features in a pocket.
"""),
    figure("pharm_features_examples"),
    md("""
**What the ligand-based model looks for.** The hypothesis model of 9.14 was described there by a list of feature types and distances. Here each of its strongest hypotheses is put back on a molecule that realises it, so that the three or two features are seen as atoms of a structure and as spheres on a 3D conformer joined by their distances, the triangle that the program calls a three-point pharmacophore. A hypothesis that favours inhibitors and one that excludes them are shown for each enzyme. The conformers are low-energy shapes drawn with the seed 42, not bound poses (11.11 measures how often such a shape contains the bound one), and the model is fitted on all training rows here, so the figure describes it and does not measure how well it predicts.
"""),
    figure("pharm_hypotheses_2d3d"),
    md("""
**One compound, every representation.** The atlas takes ketoconazole, the reference ligand of the CYP3A4 complex 2V0M, and draws it in each of the representations of the project: its functional groups, its feature families, a three-point pharmacophore on a 3D conformer, the pharmacophore level of the physics block (the carbon atoms by the energy needed to abstract a hydrogen, and the heme-binding nitrogen), the electronic level (the partial charge of each atom, with the scalars of the quantum block), and the placement of the pocket block with the sites of 9.14 in the real crystal structure. It is the figure to read first if the vocabulary of Part IX is new.
"""),
    figure("pharm_atlas"),
    md("""
**The pocket model on the protein it came from.** The sites of 9.14 were found in the frame of the heme from the ligands of many crystal complexes. The next figure puts them back into one real complex of each enzyme, as translucent spheres of their tolerance radius, so that the reader sees the model in the cavity that produced it: a crystal structure of CYP3A4 with ketoconazole and one of CYP2D6 with the ligand whose atoms come closest to the iron. The model is holo, which is to say that it was read from complexes; what an apo or a cryptic pocket looks like is not in it, and 11.11 says how far that limit can be measured.
"""),
    figure("pharm_pocket_3d"),
    md("""
**The structure track.** Section 11.11 asked whether the pharmacophore model of a pose can choose the better pose from a pool, and the answer there is the verdict of its gate. The next figure shows why the question is hard, on real poses. For three benchmark ligands, chosen by a rule fixed in advance (the smallest, the median and the largest first-sample RMSD of the unguided runs), it draws the best and the worst pose of the pool in the pocket of the predicted complex, with the crystal pose of the same ligand in white, the sites of the model as spheres (firmer where a feature of the pose matches, nearly transparent where none does) and the features of the pose as small solid spheres. A pose a few ångströms from the crystal one can match as many sites as the true one: the sites are broad and every pose is in the same region of the same pocket.
"""),
    figure("struct_pharm_pose_3d"),
    md("""
**Structure-activity relationship in the structure track.** The series benchmark of 11.5 has three pairs of analogues that differ by one change of linker (a thioether where a pair has an amine, a sulfonamide where another has an amide). Each member is drawn in 2D with the atoms that the pair changes highlighted (the atoms outside the maximum common substructure of the pair, and those that change element), next to its first predicted pose in 3D with the crystal pose and the sites of the model rebuilt without the references of either member. The pyridine nitrogen of every member lies at the distance of a Type II bond from the iron, so the series shares its heme-binding pharmacophore; what the pair changes is what the linker puts on the way out of the pocket. This is the structure track's form of an SAR: a change of structure read as a change of pose and of feature.
"""),
    figure("struct_pharm_series"),
    md("""
**The 20 official ligands.** The last two figures draw the ligands that are submitted: their structures in 2D, grouped into chemotypes (ligands whose Morgan similarity to a neighbour is at least 0.45, with the maximum common substructure of each group), with their functional groups and the pharmacophore numbers of their final poses; and their final poses in 3D in the pocket, with the sites of the model and the features of the pose. They are predictions for blinded ligands and are held back until the challenge has closed; the cells that read them are marked so that the public edition of the notebook shows only a note in their place.
""", held_back=True),
    code(OFFICIAL_ATLAS_FACTS, held_back=True),
    md("""
Of the 20 final poses {{=n_atlas_donors}} are of ligands with a Type II donor, and {{=n_donor_close}} of those lie within 0.2 Å of the ideal distance between the donor nitrogen and the iron; the other {{=n_atlas_non_donors}} have no donor, and for {{=n_site_outside}} of them the carbon that is easiest to oxidise lies outside the window of 3.5 to 6.0 Å from the iron. The median share of the weight of the crystal sites that a final pose matches is {{=recall_donors:.2f}} for the ligands with a donor and {{=recall_non_donors:.2f}} for the others: the pocket model describes the heme-binding mode and says little of the poses that do not bind the iron, which is also where the structure of the pocket has the fewest crystal ligands to say what is expected.
""", held_back=True),
    figure("struct_pharm_official20_2d", held_back=True),
    figure("struct_pharm_official20_3d", held_back=True),
    md("""
*On what actually ran.* The functional groups are defined once in `cypmode/challenge/functional_groups.py` (the table `GROUPS`, `groups_of`, `group_matrix`, `group_rates`, `wilson`, and `group_gain_pairs` for the pairs), with its tests in `tests/test_functional_groups.py`. The figures are made by `docs/figures/gen/pharmacophore_atlas.py` (the TDI track: the groups, the pairs, the features, the hypotheses, the atlas, the pockets) and `docs/figures/gen/structure_pharmacophore_atlas.py` (the structure track: the poses, the series, the 20 ligands), through the figures job 940; the 2D drawings are made by RDKit from the SMILES, the 3D pictures of crystal and predicted complexes by PyMOL from their coordinates, and the 3D conformers of the hypotheses are drawn with the seed 42 and said to be so. The tests are `tests/test_pharmacophore_atlas.py` (the drawings, the fragment of a pair, the typical pair, the chemotypes, the changed atoms of a pair and the spheres of a pose) and the figures of the three first groups are drawn in it from the real data. The hypotheses of 9.14 carry pmapper's letters, in which the lower-case a is the aromatic ring and the upper-case A the acceptor; an earlier description of those letters had them the other way round, which changed only the words of the descriptions and not a signature or a column. Limits: a functional group is a pattern, so a compound with the same chemistry written in another way is missed; the matched pairs are few for CYP2D6 and for any single exchange; the SAR of this section is that of the label, which is a time-dependent inhibition, not of the potency; and every 3D conformer is a low-energy shape, not a bound one.
"""),
]
