"""Part III (second half): the three-point pharmacophore fingerprint, chemical series, and the feature store."""

from notebooks.notebook_builder import code, figure, md

PHARMACOPHORE = '''
# The example compound as the project's pharmacophore fingerprint sees it: one 3D conformer (ETKDG distance geometry, then a MMFF force-field minimisation), the pharmacophoric features that pmapper
# finds in it, and the hashed fingerprint of all triples of features with their binned distances.
from pmapper.pharmacophore import Pharmacophore

from cypmode.challenge.pharmacophore_fingerprint import PMAPPER_NBITS, compute_pharmacophore_fingerprint, _embed_3d_conformer

conformer_molecule = _embed_3d_conformer(example)
pharmacophore = Pharmacophore()
pharmacophore.load_from_mol(conformer_molecule)
feature_names = {"A": "hydrogen-bond acceptor", "D": "hydrogen-bond donor", "H": "hydrophobic region", "a": "aromatic ring", "P": "positive ionisable", "N": "negative ionisable"}
feature_counts = dict(pharmacophore.get_features_count())
feature_table = pd.DataFrame([{"feature": f"{feature_names.get(key, key)} ({key})", "found in the example": value} for key, value in sorted(feature_counts.items())])
show_table(feature_table)
fingerprint = compute_pharmacophore_fingerprint(example)
assert fingerprint.shape == (PMAPPER_NBITS,)
n_phar_bits = int(fingerprint.sum())
print(f"{sum(feature_counts.values())} features give {n_phar_bits} bits switched on among {PMAPPER_NBITS} (all triples and pairs of features, with the distances between them in bins)")

# One molecule, many shapes: the distance between two atoms depends on the conformer. Ten ETKDG conformers of the example; the distance between the pyridine nitrogen and the sulfonamide sulfur.
with_h = Chem.AddHs(mol)
conformer_ids = AllChem.EmbedMultipleConfs(with_h, numConfs=10, randomSeed=7)
AllChem.MMFFOptimizeMoleculeConfs(with_h)
nitrogen = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.GetIsAromatic()][0]
sulfur = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "S"][0]
heavy = [a.GetIdx() for a in with_h.GetAtoms() if a.GetAtomicNum() > 1]
coordinates = np.array([[list(with_h.GetConformer(c).GetAtomPosition(i)) for i in heavy] for c in conformer_ids])            # (conformers, heavy atoms, 3)
pair_distance = np.linalg.norm(coordinates[:, :, None, :] - coordinates[:, None, :, :], axis=-1)                                # (conformers, heavy atoms, heavy atoms)
spread = pair_distance.max(axis=0) - pair_distance.min(axis=0)
i_widest, j_widest = np.unravel_index(spread.argmax(), spread.shape)
fixed = pair_distance[:, heavy.index(nitrogen), heavy.index(sulfur)]
widest_atoms = f"{with_h.GetAtomWithIdx(heavy[i_widest]).GetSymbol()}{heavy[i_widest]} and {with_h.GetAtomWithIdx(heavy[j_widest]).GetSymbol()}{heavy[j_widest]}"
distance_low, distance_high, n_conformers = float(fixed.min()), float(fixed.max()), len(conformer_ids)
widest_low, widest_high = float(pair_distance[:, i_widest, j_widest].min()), float(pair_distance[:, i_widest, j_widest].max())
print(f"over {n_conformers} conformers the distance between the pyridine nitrogen and the sulfur runs from {distance_low:.1f} to {distance_high:.1f} angstrom, "
      f"while the distance between {widest_atoms} runs from {widest_low:.1f} to {widest_high:.1f}")
'''

PHARMACOPHORE_RESULT = '''
# What the project measured with this block, and which compounds it exists for: the cache of the fingerprints, the paired comparison of the base model of the M0 to M3 ablation (section 7.2) with and
# without the block, per isoform, and what is missing.
cache = pd.read_csv(ROOT / "data/tdi_track/outputs/caches/pharmacophore_fingerprint_cache.csv", usecols=["SMILES"])
cached = set(cache["SMILES"])
coverage_train, coverage_test = int(train["SMILES"].isin(cached).sum()), int(test["SMILES"].isin(cached).sum())
paired = pd.read_csv(ROOT / "data/tdi_track/outputs/comparisons/pharmacophore_fingerprint_significance_paired.csv")
result = paired[(paired["group1"] == "M3") & (paired["group2"] == "M3+Pharmacophore")][["isoform", "meandiff", "p-adj", "lower", "upper"]].rename(
    columns={"meandiff": "mean difference of the outer AUC", "p-adj": "adjusted p", "lower": "interval, lower", "upper": "interval, upper"})
print(f"the cache holds {len(cached)} compounds: {coverage_train} of the {len(train)} training compounds and {coverage_test} of the {len(test)} test compounds")
show_table(result, floatfmt=".4f")
phar_result = result.set_index("isoform")
'''

SERIES = '''
# Chemical series. Scaffolds (Bemis and Murcko) and Taylor-Butina clusters of the CYP3A4 compounds with both arms, then what keeping a series together does to a validation split: for the compounds held out
# in 5-fold cross-validation, the highest Tanimoto similarity to the compounds the model is trained on, once for a random split and once for a split that keeps every cluster on one side.
from sklearn.model_selection import GroupKFold, KFold

from cypmode.challenge.grouping import butina_groups, morgan_bit_matrix, scaffold_groups

both = training_view(train, "CYP3A4").query("population == 'both_arms'")
subset = train.loc[both.index, "SMILES"].tolist()
clusters = butina_groups(morgan_bit_matrix(subset), 0.6)
scaffolds = scaffold_groups(subset)
cluster_sizes = np.bincount(clusters)
generator_series = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps_series = [generator_series.GetFingerprint(Chem.MolFromSmiles(s)) for s in subset]

def held_out_neighbour(splits):
    out = []
    for fit_part, held_out in splits:
        reference = [fps_series[j] for j in fit_part]
        out += [max(DataStructs.BulkTanimotoSimilarity(fps_series[j], reference)) for j in held_out]
    return np.array(out)

random_nn = held_out_neighbour(KFold(5, shuffle=True, random_state=0).split(fps_series))
group_nn = held_out_neighbour(GroupKFold(5).split(fps_series, groups=clusters))
series_rows = [{"quantity": "compounds", "value": len(subset)}, {"quantity": "Bemis-Murcko generic scaffolds", "value": len(set(scaffolds))},
               {"quantity": "Butina clusters at similarity 0.6", "value": len(set(clusters))}, {"quantity": "clusters with one compound", "value": int((cluster_sizes == 1).sum())},
               {"quantity": "largest cluster", "value": int(cluster_sizes.max())}]
show_table(pd.DataFrame(series_rows))
split_rows = [{"split": "random 5-fold", "median similarity of a held-out compound to the training part": float(np.median(random_nn)), "share at 0.5 or more": float((random_nn >= 0.5).mean()), "share at 0.6 or more": float((random_nn >= 0.6).mean())},
              {"split": "5-fold keeping every cluster whole", "median similarity of a held-out compound to the training part": float(np.median(group_nn)), "share at 0.5 or more": float((group_nn >= 0.5).mean()), "share at 0.6 or more": float((group_nn >= 0.6).mean())}]
show_table(pd.DataFrame(split_rows), floatfmt=".3f")
n_clusters, n_singletons, n_scaffolds, largest_cluster = len(set(clusters)), int((cluster_sizes == 1).sum()), len(set(scaffolds)), int(cluster_sizes.max())
share_random_06, share_group_06 = float((random_nn >= 0.6).mean()), float((group_nn >= 0.6).mean())
'''

STORE_CHECK = '''
# The feature store's two promises, checked on the real functions: a block is a function of the structure only (computing it twice, or in a different order, gives the same numbers), and a table has
# one row per compound in the order the compounds were given. 25 compounds of the training file, default blocks.
from cypmode.challenge.feature_store import FeatureStore

sample = train["SMILES"].iloc[:25].tolist()
first = FeatureStore.build(sample)
second = FeatureStore.build(sample[::-1])
identical = {name: bool(np.array_equal(first.blocks[name], second.blocks[name][::-1])) for name in first.blocks}
print(f"{len(first.blocks)} default blocks computed on 25 compounds, and again with the compounds in the reverse order: identical for {sum(identical.values())} of {len(identical)} blocks")
matrix = first.matrix(["morgan_c2", "maccs"], rows=np.array([3, 0]))
print(f"matrix(['morgan_c2', 'maccs'], rows=[3, 0]) has shape {matrix.shape}: {first.blocks['morgan_c2'].shape[1]} + {first.blocks['maccs'].shape[1]} columns, rows in the order requested")
stored_nan = {name: int(np.isnan(store[name]).sum()) for name in store.files if name != "smiles"}
print("not-a-number entries in the stored blocks:", {k: v for k, v in stored_nan.items() if v} or "none")
n_identical, n_default_blocks = sum(identical.values()), len(identical)
nan_blocks_text = ", ".join(f"{name} ({count} entries)" for name, count in stored_nan.items() if count) or "none"
'''

CELLS = [
    md("""
## 3.5 Three-point pharmacophore fingerprints

> **Field:** Cheminformatics · Structural biology

A **pharmacophore** is the arrangement in space of the features of a molecule that let it interact with a given protein: a hydrogen-bond donor here, an acceptor there, an aromatic ring, a greasy region, a charged group, at distances from each other that fit the pocket. Two molecules with different skeletons can present the
same arrangement and bind the same site, which is why the concept is used in virtual screening. The fingerprints of the previous sections are blind to it: a Morgan fingerprint says that a piece is present and not where it lies in three dimensions. A **three-point pharmacophore fingerprint** records the arrangement. It finds the features in a 3D shape of the molecule, takes
every set of two or three of them, and writes down the types of the features together with the distances between them, divided into bins so that small differences do not matter; each combination is hashed to a position of a fixed row, exactly as in section 3.1. The project uses the open-source program pmapper (Polishchuk and co-workers) through
`cypmode/challenge/pharmacophore_fingerprint.py`, with a width of 2048 positions.

A 3D shape is needed, and a molecule has no single one: it can bend about its single bonds, and each shape, a **conformer**, gives different distances. The module generates one conformer with the ETKDG method (Riniker and Landrum, 2015), which builds a structure from the pairwise distances allowed by the bonds and corrects the torsion angles
with preferences that were learned from crystal structures, and relaxes it with the MMFF94 force field (Halgren, 1996), a set of simple formulas for the energy of bond stretching, bending, twisting and non-bonded contacts that is minimised until nothing moves. A molecule whose 3D structure cannot be built gets a fingerprint of zeros, which is an honest "no bits", and not an error that would stop a batch of thousands. The cell shows the features found in the example, its fingerprint, and
the effect of the conformer on a distance.
"""),
    code(PHARMACOPHORE),
    md("""
The example compound presents {{=sum(feature_counts.values())}} features, which combine into {{=n_phar_bits}} bits. Over {{=n_conformers}} conformers of the same molecule the distance between the pyridine nitrogen and the sulfur runs only from {{=distance_low:.1f}} to {{=distance_high:.1f}} angstrom (the two atoms are held apart by a ring and one chain), but the distance between {{=widest_atoms}} runs from {{=widest_low:.1f}} to {{=widest_high:.1f}} angstrom: how much a distance depends on the conformer depends on the pair of atoms, and for flexible pairs a single conformer gives a fingerprint that is only one of several possible ones, which is the reason the bins are coarse.
The fingerprint is slow compared with the others (about a third of a second per compound for the embedding and the minimisation), so it was computed once and cached. **On what actually ran.** The cache holds a fingerprint for the compounds that were in the training file when the block was tested, not for the whole of today's file, and it holds none for the test compounds; the block was
therefore never a column of the feature store, and the result below is a comparison made at that earlier stage.
"""),
    code(PHARMACOPHORE_RESULT),
    md("""
The cache holds {{=coverage_train}} of the {{=len(train)}} training compounds and {{=coverage_test}} test compounds. The paired comparison of the base model of the M0 to M3 ablation (section 7.2) with and without the block (the corrected resampled test that Part VI builds, so the adjusted $p$ is the one after the correction for several comparisons) shows a difference of the outer AUC of {{=phar_result.loc['CYP3A4', 'mean difference of the outer AUC']:.4f}} on CYP3A4 and {{=phar_result.loc['CYP2D6', 'mean difference of the outer AUC']:.4f}} on CYP2D6, with an adjusted $p$ of
{{=phar_result.loc['CYP3A4', 'adjusted p']:.2f}} and {{=phar_result.loc['CYP2D6', 'adjusted p']:.2f}}: the block adds nothing that can be told from chance, and it was not carried forward. It is kept in the code and in this record because the idea, that the arrangement of features in space tells what a compound can do, comes back with the physics of Part X, where the distances are computed from a quantum-chemical description and not from a force field.

## 3.6 Chemical series, scaffolds and clusters

> **Field:** Cheminformatics · Statistics (clustering)

Compounds in a medicinal-chemistry programme are made in **series**: a chemist takes a compound that works, changes one part of it, and makes the next. A series is a group of near relatives, and in the data of this challenge the series are many and tight (section 2.7: the median test compound has a relative at a similarity above 0.5). Why this matters for validation is that a random split of such data
puts members of the same series on both sides, so the model is asked about a compound whose near relative it has just been shown. The question of this section is how to find the series, and how much it matters. Two tools are used (`cypmode/challenge/grouping.py`).

**Bemis-Murcko scaffolds** (Bemis and Murcko, 1996). Remove from a molecule all side chains and keep the rings and the chains that link them; what is left is the scaffold, the framework of the molecule. The *generic* scaffold goes one step further, turning every atom into carbon and every bond into a single bond, so that molecules with the same shape but different atoms share one. Two compounds with the same scaffold belong
together, but two members of a series with different scaffolds (a ring was changed) do not, which is why scaffolds are too strict.

**Taylor-Butina clustering** (Butina, 1999). Choose a similarity cut-off (here a Tanimoto similarity of 0.6, a distance of $1 - 0.6 = 0.4$). Count, for every compound, the others within the cut-off. The compound with the most neighbours becomes a cluster *centre*, and it and all of its neighbours form a cluster, which is removed; the next centre is picked among the compounds that are left, and the process repeats until
none is left. A compound with no neighbour in range forms a cluster of one. Every member is within the cut-off of the centre of its cluster, but two members can be farther apart than that, and the clusters are not nested. The cell clusters the CYP3A4 compounds that have both arms and then compares two ways of splitting them for five-fold cross-validation: at random, and with every cluster kept whole (scikit-learn's `GroupKFold`, which assigns the groups to folds so that no group is divided).
"""),
    code(SERIES),
    md("""
The {{=len(subset)}} compounds fall into {{=n_clusters}} clusters at the similarity cut-off of 0.6, of which {{=n_singletons}} have a single compound; the largest holds {{=largest_cluster}}, and there are {{=n_scaffolds}} generic scaffolds. The training set is therefore mostly made of compounds that are not close relatives of one another at that level, with a minority in tight series. That minority has a
large effect on a validation: with a random split, {{=share_random_06:.1%}} of the held-out compounds have a training compound at a similarity of 0.6 or more; keeping each cluster together reduces it to {{=share_group_06:.1%}}, as it should, since the clusters were defined by exactly that cut-off. The project uses both kinds of split (Part VI): the random one measures what a model does on compounds
like those it has seen, which is what the test set resembles (section 2.7), and the clustered one measures what it does on a new series. A number is only meaningful together with the kind of split that produced it.

## 3.7 The feature store: all the blocks, computed once

> **Field:** Software engineering · Cheminformatics

The blocks of sections 3.1 to 3.5 are needed by hundreds of model fits (Part VI). Computing a fingerprint takes milliseconds, a descriptor table seconds, the physics block hours, so recomputing per fit would be wasteful and, worse, could produce small differences between fits. The module `cypmode/challenge/feature_store.py`
therefore keeps one table: a dictionary of *blocks*, each a function of a list of molecules that returns a table with one row per molecule, evaluated once for all {{=len(train) + len(test)}} compounds of both files and written to a single compressed file (`data/tdi_track/outputs/caches/feature_store.npz`, with the list of strings that it was computed for as its key: a cache built for a different list is discarded and rebuilt). A model chooses blocks by name and the store returns the columns side by side, for
the rows it asks for (`FeatureStore.matrix`).

Two rules make this safe. **A block reads no label.** Every block is a function of the structure and nothing else, so that it cannot carry an answer from a test compound's label (which no one has) or from a validation compound into a fit. The consequence is that a block can be computed once for everything, including the test structures, which is allowed because a structure is not a label. Features that do depend on labels, such as the labels of
a compound's neighbours (section 7.3), are *not* blocks: they depend on which compounds are in the training part of each fold, so the runner builds them separately inside every fold (Part VI). **Opt-in blocks are read, never computed**: the physics, pretrained-embedding and external-activity blocks come from caches written by processes of their own (some need a library that conflicts with the one used for model fitting
when both run in one process), and asking for one that has not been built is an error and not a silent recomputation. The cell checks the first promise on real compounds and counts the missing values in the stored table.
"""),
    code(STORE_CHECK),
    md("""
The {{=n_default_blocks}} default blocks of the store give the same numbers when the compounds are computed in the reverse order ({{=n_identical}} of {{=n_default_blocks}} identical), so that a row depends on its compound and on nothing else. Blocks of the stored table with undefined (not-a-number) entries: {{=nan_blocks_text}}. The figure shows the widths of the blocks.
"""),
    figure("mol_feature_blocks"),
]
