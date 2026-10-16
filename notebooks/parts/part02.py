"""Part II (first half): the data. A molecule as text and as a graph, the two challenge files, the label rule, the contaminated rows, measurement noise."""

from notebooks.notebook_builder import code, figure, md

MOLECULE_AS_GRAPH = '''
# The example compound of the challenge (OCNT-2395436) read as a graph: RDKit turns the SMILES text into atoms (the vertices) and bonds (the edges). Hydrogens are not stored as atoms;
# they are counted from the valence, which is how SMILES leaves them out.
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

example_name = "OCNT-2395436"
example = train.set_index("Molecule_Name").loc[example_name, "SMILES"]
mol = Chem.MolFromSmiles(example)
atom_rows = []
for atom in mol.GetAtoms():
    bond_orders = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())          # aromatic bonds count 1.5
    atom_rows.append({"element": atom.GetSymbol(), "bonded heavy atoms": atom.GetDegree(), "hydrogens": atom.GetTotalNumHs(), "valence": bond_orders + atom.GetTotalNumHs()})
by_element = pd.DataFrame(atom_rows).groupby("element").agg(atoms=("valence", "size"), valence_found=("valence", lambda v: ", ".join(f"{x:g}" for x in sorted(set(v))))).reset_index()
usual = {"C": "4", "N": "3", "O": "2", "S": "2, 4 or 6", "Br": "1"}
by_element["usual valence"] = by_element["element"].map(usual)
by_element = by_element.rename(columns={"valence_found": "valence found in the molecule"})

adjacency = Chem.GetAdjacencyMatrix(mol)          # A[i, j] = 1 when atoms i and j share a bond
print(f"SMILES: {example}")
print(f"formula {rdMolDescriptors.CalcMolFormula(mol)}, molecular weight {Descriptors.MolWt(mol):.2f}; {mol.GetNumAtoms()} heavy atoms, {mol.GetNumBonds()} bonds")
print(f"adjacency matrix {adjacency.shape[0]} x {adjacency.shape[1]}: symmetric {bool((adjacency == adjacency.T).all())}, row sums (the number of bonded heavy atoms) {sorted({int(x) for x in adjacency.sum(axis=1)})}, half the sum of all entries = {adjacency.sum() // 2} bonds")
print(f"rings: {[len(r) for r in mol.GetRingInfo().AtomRings()]} atoms; aromatic atoms: {sum(a.GetIsAromatic() for a in mol.GetAtoms())}")
show_table(by_element)
n_atoms, n_bonds, n_aromatic_atoms = mol.GetNumAtoms(), mol.GetNumBonds(), sum(a.GetIsAromatic() for a in mol.GetAtoms())
'''

HUCKEL = '''
# Aromaticity by counting electrons. In one Kekule drawing of a ring, every double bond inside the ring carries two pi electrons; a ring atom with a lone pair that is not in a double bond (the N-H of
# pyrrole, the O of furan) lends two more. Hueckel's rule: a flat, fully conjugated ring with 4n + 2 pi electrons is aromatic.
def pi_electrons(smiles):
    ring_mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(ring_mol, clearAromaticFlags=True)
    ring = ring_mol.GetRingInfo().AtomRings()[0]
    double_in_ring = sum(1 for b in ring_mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE and b.GetBeginAtomIdx() in ring and b.GetEndAtomIdx() in ring)
    lone_pair_donors = sum(1 for i in ring if ring_mol.GetAtomWithIdx(i).GetSymbol() in ("N", "O", "S") and not any(
        b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtomIdx(i) in ring for b in ring_mol.GetAtomWithIdx(i).GetBonds()))
    return 2 * double_in_ring + 2 * lone_pair_donors, len(ring)

reference_rings = {"benzene": "c1ccccc1", "pyridine": "c1ccncc1", "pyrrole": "c1cc[nH]c1", "furan": "c1ccoc1", "imidazole": "c1c[nH]cn1", "cyclohexane": "C1CCCCC1", "cyclooctatetraene": "C1=CC=CC=CC=C1"}
ring_rows = []
for name, smiles in reference_rings.items():
    electrons, size = pi_electrons(smiles)
    huckel = electrons % 4 == 2
    flat_conjugated = all(a.GetIsAromatic() for a in Chem.MolFromSmiles(smiles).GetAtoms())
    ring_rows.append({"ring": name, "atoms in the ring": size, "pi electrons": electrons, "4n + 2 ?": "yes" if huckel else "no", "RDKit calls it aromatic": "yes" if flat_conjugated else "no"})
show_table(pd.DataFrame(ring_rows))
ring_in_example = [r for r in mol.GetRingInfo().AtomRings() if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)][0]
ring_atoms = sorted(mol.GetAtomWithIdx(i).GetSymbol() for i in ring_in_example)
print(f"the aromatic ring of the example compound has the atoms {', '.join(ring_atoms)}: a pyridine ring, {pi_electrons('c1ccncc1')[0]} pi electrons, with the nitrogen lone pair in the plane of the ring")
'''

CANONICAL = '''
# One molecule, many strings. A SMILES string is a walk over the graph; the walk can start at any atom and take any branch first. 2000 random walks over the example compound give many different
# strings; every one of them parses back to the same graph, and the canonical form (a fixed rule for the start and the order of the branches) turns them all into one string.
random_strings = {Chem.MolToSmiles(mol, doRandom=True, canonical=False) for _ in range(2000)}
canonical = Chem.MolToSmiles(mol)
same_again = all(Chem.MolToSmiles(Chem.MolFromSmiles(s)) == canonical for s in random_strings)
print(f"canonical form: {canonical}")
print(f"2000 random walks gave {len(random_strings)} different strings, for instance:")
for s in sorted(random_strings, key=len)[:2]:
    print("   ", s)
print(f"every one of them has the canonical form above: {same_again}")

# The strings of the training file are written in Kekule form (alternating double bonds in the rings, no lower-case aromatic atoms); after reading them and writing them back the aromatic atoms are lower case.
import re

aromatic_letters = re.compile(r"(?<![A-Z])[cnos]")                        # a lower-case letter that does not follow a capital (that would be Cl, Br, ...)
kekule_style = np.mean([not aromatic_letters.search(s) for s in train["SMILES"]])
as_given = np.mean([Chem.MolToSmiles(Chem.MolFromSmiles(s)) == s for s in train["SMILES"]])
print(f"share of the {len(train)} training strings written in Kekule style: {kekule_style:.1%}; share already in RDKit's canonical form: {as_given:.1%}; the example, as given: {example}")
'''

STEREO = '''
# Stereochemistry. A carbon with four different neighbours exists in two mirror-image forms that the flat graph cannot tell apart. SMILES marks them with @ and @@ (the order of the neighbours
# seen from the first neighbour: anticlockwise or clockwise). The example below is a training compound with one such centre; flipping the mark gives its mirror image.
from rdkit.Chem import AllChem

chiral_name = "OCNT-0493927"
chiral_smiles = train.set_index("Molecule_Name").loc[chiral_name, "SMILES"]
chiral = Chem.MolFromSmiles(chiral_smiles)
mirror = Chem.MolFromSmiles(chiral_smiles.replace("@@", "!").replace("@", "@@").replace("!", "@"))
centres = Chem.FindMolChiralCenters(chiral, useLegacyImplementation=False)
mirror_centres = Chem.FindMolChiralCenters(mirror, useLegacyImplementation=False)
print(f"{chiral_name}: {chiral_smiles}\\n  stereocentre (atom index, R or S): {centres}")
print(f"  mirror image: {Chem.MolToSmiles(mirror)}\\n  stereocentre: {mirror_centres}")
print(f"  same graph, different molecule: the two canonical strings are different ({Chem.MolToSmiles(chiral) != Chem.MolToSmiles(mirror)}), the graphs without the marks are equal "
      f"({Chem.MolToSmiles(chiral, isomericSmiles=False) == Chem.MolToSmiles(mirror, isomericSmiles=False)})")

# How many compounds of the training file carry such centres, and for how many of them is the configuration given?
molecules_train = [Chem.MolFromSmiles(s) for s in train["SMILES"]]
n_possible = np.array([len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)) for m in molecules_train])
n_given = np.array([len(Chem.FindMolChiralCenters(m, includeUnassigned=False, useLegacyImplementation=False)) for m in molecules_train])
stereo_table = pd.DataFrame([
    {"compounds": "with no stereocentre", "number": int((n_possible == 0).sum())},
    {"compounds": "with a stereocentre, configuration not given", "number": int(((n_possible > 0) & (n_given == 0)).sum())},
    {"compounds": "with a stereocentre, configuration given (at least one)", "number": int((n_given > 0).sum())}])
stereo_table["share"] = stereo_table["number"] / len(train)
show_table(stereo_table, floatfmt=".3f")
n_unspecified = int(((n_possible > 0) & (n_given == 0)).sum())

# Compounds that differ only in configuration share one flat graph. Are there such pairs in the file, and do their labels disagree?
flat = pd.Series([Chem.MolToSmiles(m, isomericSmiles=False) for m in molecules_train], index=train.index)
group_sizes = flat.value_counts()
n_shared_graphs = int((group_sizes > 1).sum())
n_conflicts = 0
for enzyme in ("CYP3A4", "CYP2D6"):
    labelled_rows = train[f"{enzyme}_is_TDI"].notna()
    n_conflicts += int((train.loc[labelled_rows, f"{enzyme}_is_TDI"].astype(bool).groupby(flat[labelled_rows]).nunique() > 1).sum())
print(f"flat graphs shared by more than one training compound: {n_shared_graphs}; of these, labelled differently on CYP3A4 or CYP2D6: {n_conflicts}")
'''

FILE_STATS = '''
# Per enzyme: how many training compounds have each of the two arms, both of them, and a label. (Counts of non-empty cells; section 2.3 uses the compounds with both arms.)
rows = []
for enzyme in enzymes:
    t_arm, d_arm = train[f"{enzyme}_pIC50_TDI_condition"].notna(), train[f"{enzyme}_pIC50_direct_inhibition"].notna()
    labelled = train[f"{enzyme}_is_TDI"].notna() if f"{enzyme}_is_TDI" in train.columns else pd.Series(False, index=train.index)
    rows.append({"enzyme": enzyme, "TDI arm measured": int(t_arm.sum()), "direct arm measured": int(d_arm.sum()), "both arms": int((t_arm & d_arm).sum()),
                 "TDI arm only": int((t_arm & ~d_arm).sum()), "direct arm only": int((~t_arm & d_arm).sum()), "label given": int(labelled.sum()) if labelled.any() else "none",
                 "positive labels": int(train.loc[labelled, f"{enzyme}_is_TDI"].astype(bool).sum()) if labelled.any() else "none"})
coverage_table = pd.DataFrame(rows)
show_table(coverage_table)
measured_any = train[[f"{e}_pIC50_TDI_condition" for e in enzymes]].notna().any(axis=1)
measured_all = train[[f"{e}_pIC50_TDI_condition" for e in enzymes]].notna().all(axis=1)
print(f"compounds with the TDI arm measured on at least one enzyme: {int(measured_any.sum())} of {len(train)}; on all four: {int(measured_all.sum())}")
numbers = lambda names: names.str.extract(r"(\\d+)$")[0].astype(int)
print(f"compound numbers: training {numbers(train['Molecule_Name']).min()} to {numbers(train['Molecule_Name']).max()}, blinded test {numbers(test['Molecule_Name']).min()} to {numbers(test['Molecule_Name']).max()}")
only_label_missing = train.loc[train["CYP3A4_is_TDI"].notna() & train["CYP3A4_pIC50_TDI_condition"].isna(), ["Molecule_Name", "CYP3A4_is_TDI", "CYP3A4_pIC50_direct_inhibition"]]
print("labelled on CYP3A4 although the TDI arm is empty:", only_label_missing.to_dict("records"))
tdi_only = coverage_table.set_index("enzyme")["TDI arm only"]
direct_only_total = int(coverage_table["direct arm only"].sum())
'''

LABEL_RULE = '''
# The challenge's rule as it is written, and the single threshold that replaces it. The literal rule has two branches; the single threshold s > log10(2) with s = t - max(d, 4) must give the
# same answer for every possible pair (d, t), which is checked here on 200000 random pairs (and a grid that includes the boundaries) and on the real compounds.
from cypmode.challenge.tdi_targets import ASSAY_FLOOR, SHIFT_THRESHOLD, label_from_arms, label_from_shift, training_view

rng = np.random.default_rng(0)
d_random, t_random = rng.uniform(1, 8, 200_000), rng.uniform(1, 8, 200_000)
literal = np.array([label_from_arms(d, t) for d, t in zip(d_random, t_random)])
single = label_from_shift(t_random - np.maximum(d_random, ASSAY_FLOOR)) == 1
grid = np.round(np.arange(1.0, 8.0001, 0.05), 2)
grid_disagree = sum(label_from_arms(d, t) != (label_from_shift(t - max(d, ASSAY_FLOOR)) == 1) for d in grid for t in grid if d != ASSAY_FLOOR)
print(f"floor {ASSAY_FLOOR}, threshold log10(2) = {SHIFT_THRESHOLD:.5f}; disagreements between the two forms: {int((literal != single).sum())} of 200000 random pairs, {grid_disagree} of {grid.size ** 2 - grid.size} grid pairs")

rule_rows, intervals = [], {}
for enzyme in ("CYP3A4", "CYP2D6"):
    view = training_view(train, enzyme)
    both = view[view["population"] == "both_arms"]
    given = both["label"].astype(bool)
    reproduced = label_from_shift(both["shift"]) == 1
    # the thresholds consistent with every label: above the largest shift of a negative and at or below the smallest shift of a positive
    lower, upper = both.loc[~given, "shift"].max(), both.loc[given, "shift"].min()
    intervals[enzyme] = (lower, upper)
    assert (reproduced == given).all(), "the rule does not reproduce every label"
    rule_rows.append({"enzyme": enzyme, "compounds with both arms": len(both), "labels reproduced by the rule": int((reproduced == given).sum()),
                      "positives with d >= 4": int((given & (both["direct"] >= 4)).sum()), "positives with d < 4 (inferred)": int((given & (both["direct"] < 4)).sum()),
                      "highest shift among negatives": lower, "lowest shift among positives": upper})
show_table(pd.DataFrame(rule_rows), floatfmt=".5f")
n_exact_four = int(sum((training_view(train, e)["direct"] == ASSAY_FLOOR).sum() for e in ("CYP3A4", "CYP2D6")))
both_3a4 = training_view(train, "CYP3A4").query("population == 'both_arms'")
near_threshold = int(((both_3a4["shift"] - SHIFT_THRESHOLD).abs() < 0.1).sum())
print(f"compounds with d exactly 4: {n_exact_four}; CYP3A4 compounds with a shift within 0.1 of the threshold: {near_threshold} of {len(both_3a4)}")
print(f"log10(2) = {SHIFT_THRESHOLD:.5f} lies inside both intervals; 0.3 does not: the largest negative shift of CYP3A4 is {intervals['CYP3A4'][0]:.5f} > 0.3")
'''

GRID_SEARCH = '''
# Recovering the rule instead of assuming it: for every pair (floor, threshold) of a grid count the labels of the compounds with both arms that the rule t - max(d, floor) > threshold gets wrong.
# Only one cell of the grid has no error, for each enzyme.
grids = {}
for enzyme in ("CYP3A4", "CYP2D6"):
    both = training_view(train, enzyme).query("population == 'both_arms'")
    floors, thresholds = [3.5, 3.9, 4.0, 4.1, 4.5], [0.2, 0.25, 0.3, SHIFT_THRESHOLD, 0.35, 0.4]
    errors = pd.DataFrame([[int(((both["tdi"] - np.maximum(both["direct"], f) > th) != both["label"].astype(bool)).sum()) for th in thresholds] for f in floors],
                          index=[f"floor {f}" for f in floors], columns=["0.2", "0.25", "0.3", "log10 2 = 0.301", "0.35", "0.4"])
    grids[enzyme] = errors
    show_table(errors.reset_index().rename(columns={"index": "floor"}), caption=f"{enzyme}: number of labels among the {len(both)} compounds with both arms that the rule with this floor and threshold gets wrong")
zero_cells = {enzyme: int((g == 0).to_numpy().sum()) for enzyme, g in grids.items()}
smallest_error = {enzyme: int(g.to_numpy()[g.to_numpy() > 0].min()) for enzyme, g in grids.items()}
print("grid cells without any error:", zero_cells, "; smallest number of errors in any other cell:", smallest_error)
'''

POPULATIONS = '''
# The three kinds of CYP3A4 rows: both arms measured; labelled although the direct arm is empty; no label at all. (training_view tags them.)
view3 = training_view(train, "CYP3A4")
pop_rows = []
for name, label in (("both_arms", "both arms measured"), ("no_direct_label_by_convention", "label given, direct arm empty"), ("unlabeled", "no usable label (the TDI arm is empty)")):
    part = view3[view3["population"] == name]
    given = part["label"].dropna().astype(bool)
    pop_rows.append({"CYP3A4 rows": label, "compounds": len(part), "positive": int(given.sum()), "median TDI-arm pIC50": part["tdi"].median(),
                     "share with t above 4.301": float((part["tdi"] > ASSAY_FLOOR + SHIFT_THRESHOLD).mean()) if part["tdi"].notna().any() else float("nan")})
population_table = pd.DataFrame(pop_rows)
show_table(population_table.fillna("-"), floatfmt=".3f")
no_direct = view3[view3["population"] == "no_direct_label_by_convention"]
n_no_direct, share_would_be_positive, median_t_no_direct = len(no_direct), float((no_direct["tdi"] > ASSAY_FLOOR + SHIFT_THRESHOLD).mean()), no_direct["tdi"].median()
both_rows = view3[view3["population"] == "both_arms"]
print(f"if the direct pIC50 of those {n_no_direct} compounds were below the floor, the written rule would call every compound with t > 4.301 positive: {share_would_be_positive:.0%} of them; the file calls {int(no_direct['label'].astype(bool).sum())} of them positive")
print(f"positive share among the compounds with both arms: {both_rows['label'].astype(bool).mean():.3f}")
'''

CONTAMINATION_EFFECT = '''
# What the 1249 rows do to a model. The same LightGBM model on the same features (Morgan counts, radius 2, 1024 positions) is trained twice in 5-fold cross-validation, once on the
# compounds with both arms only and once with the 1249 labelled-negative rows added to every training part. Both are scored on the same held-out compounds with both arms, because
# those are the compounds that resemble the scored test set.
from lightgbm import LGBMClassifier
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
smiles_all = train.loc[view3.index[view3["population"] != "unlabeled"], "SMILES"]
features = np.array([generator.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles_all], dtype=np.float32)
status = view3.loc[smiles_all.index, "population"].to_numpy()
y_all = view3.loc[smiles_all.index, "label"].astype(int).to_numpy()
is_both = status == "both_arms"

def cross_validated_auc(include_no_direct: bool, seed: int = 0) -> float:
    scores, truths = [], []
    folds = StratifiedKFold(5, shuffle=True, random_state=seed).split(features[is_both], y_all[is_both])
    both_index = np.flatnonzero(is_both)
    extra = np.flatnonzero(~is_both)
    for train_part, test_part in folds:
        fit_index = np.concatenate([both_index[train_part], extra]) if include_no_direct else both_index[train_part]
        model = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, random_state=seed, n_jobs=2, verbose=-1)
        model.fit(features[fit_index], y_all[fit_index])
        scores.append(model.predict_proba(features[both_index[test_part]])[:, 1])
        truths.append(y_all[both_index[test_part]])
    return roc_auc_score(np.concatenate(truths), np.concatenate(scores))

auc_clean, auc_mixed = cross_validated_auc(False), cross_validated_auc(True)
assert auc_clean > auc_mixed, "the text below states that the extra rows lower the AUC"
print(f"AUC on held-out compounds with both arms: trained on both-arms rows only {auc_clean:.3f}; with the {int((~is_both).sum())} labelled-negative rows added {auc_mixed:.3f}; difference {auc_clean - auc_mixed:+.3f}")
'''

NOISE = '''
# What the confidence limits and the standard deviation of a pIC50 are. For a normally distributed estimate the interval mean +- 1.96 sd holds 95 % of the probability; the file's limits
# should therefore be 1.96 standard deviations from the value on each side.
from scipy.stats import norm

print(f"norm.ppf(0.975) = {norm.ppf(0.975):.5f}")
noise_rows = []
for enzyme in ("CYP3A4", "CYP2D6"):
    for arm, arm_name in (("TDI_condition", "TDI arm"), ("direct_inhibition", "direct arm")):
        column = f"{enzyme}_pIC50_{arm}"
        part = train[[column, column + "_conf_low", column + "_conf_high", column + "_std"]].dropna()
        half_width = (part[column + "_conf_high"] - part[column + "_conf_low"]) / 2 / part[column + "_std"]
        centred = (part[column] - part[column + "_conf_low"]) / (part[column + "_conf_high"] - part[column])
        noise_rows.append({"enzyme": enzyme, "arm": arm_name, "measurements": len(part), "median standard deviation": part[column + "_std"].median(),
                           "half-width / sd, median": half_width.median(), "half-width / sd, 10th to 90th percentile": f"{half_width.quantile(0.1):.2f} to {half_width.quantile(0.9):.2f}",
                           "value is the middle of the interval": f"{centred.median():.2f}"})
show_table(pd.DataFrame(noise_rows), floatfmt=".3f")
sd_values = {(r["enzyme"], r["arm"]): r["median standard deviation"] for r in noise_rows}
'''

PROPAGATION = '''
# Error propagation for the shift, checked by simulation. A compound with t = 5.4 +- 0.06 and d = 4.6 +- 0.09 (the median standard deviations of CYP3A4): draw both measurements 200000 times
# from their normal distributions, form s = t - max(d, 4) each time, and compare the spread of s with sqrt(sd_t^2 + sd_d^2). Then the same for a compound whose direct pIC50 is below the floor.
rng = np.random.default_rng(1)
sd_t, sd_d = sd_values[("CYP3A4", "TDI arm")], sd_values[("CYP3A4", "direct arm")]
prop_rows = []
for d_mean, t_mean in ((4.6, 5.4), (3.2, 5.4)):
    d_draw, t_draw = rng.normal(d_mean, sd_d, 200_000), rng.normal(t_mean, sd_t, 200_000)
    s_draw = t_draw - np.maximum(d_draw, ASSAY_FLOOR)
    formula = np.sqrt(sd_t**2 + (sd_d**2 if d_mean >= ASSAY_FLOOR else 0.0))
    prop_rows.append({"d": d_mean, "t": t_mean, "sd(s) by simulation": s_draw.std(), "sd(s) by formula": formula, "mean of s": s_draw.mean()})
show_table(pd.DataFrame(prop_rows), floatfmt=".4f")
'''

CONFIDENCE = '''
# The confidence z = |s - log10 2| / sd(s) of every labelled compound and the probability that its label is wrong, Phi(-z), where Phi is the standard normal distribution function.
from scipy.special import erf

from cypmode.challenge.label_confidence import confidence_z

def Phi(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))

conf_rows, z_by_enzyme = [], {}
for enzyme in ("CYP3A4", "CYP2D6"):
    view = training_view(train, enzyme)
    both = view["population"] == "both_arms"
    z = confidence_z(train, enzyme)[both.to_numpy()]
    z_by_enzyme[enzyme] = z
    row = {"enzyme": enzyme, "compounds": int(both.sum()), "median z": float(np.median(z)), "expected wrong labels": float(Phi(-z).sum())}
    for z0 in (1.0, 1.96, 2.5):
        row[f"|z| >= {z0}"] = int((z >= z0).sum())
    conf_rows.append(row)
show_table(pd.DataFrame(conf_rows), floatfmt=".1f")
print("probability that a label is wrong, Phi(-z):", {z0: round(float(Phi(-z0)), 4) for z0 in (0.5, 1.0, 1.96, 2.5, 3.0)})
quantiles = np.percentile(z_by_enzyme["CYP3A4"], [5, 25, 50, 75, 95])
print("CYP3A4 z at the 5th, 25th, 50th, 75th, 95th percentile:", np.round(quantiles, 2))
n_both = {r["enzyme"]: r["compounds"] for r in conf_rows}
expected_wrong = {r["enzyme"]: r["expected wrong labels"] for r in conf_rows}
n_confident = {r["enzyme"]: r["|z| >= 1.96"] for r in conf_rows}
share_below_2 = float((z_by_enzyme["CYP3A4"] < 2).mean())
'''

CELLS = [
    md("""
# Part II. The data

A model learns from numbers, and the first task of any project is to know exactly what the numbers are: where they come from, what they measure, what is missing, and how far the data given for learning
resemble the data on which the model will be judged. This part describes the molecules and the two files of the challenge (sections 2.1 and 2.2), the rule that turns two measurements into a label and the problem that
was found in some of the labels (2.3 to 2.5), the structure of the data (2.6 and 2.7), the other sources the project looked at (2.8 and 2.9), and the public leaderboard as a source of information (2.10).

## 2.1 A molecule as text and as a graph

> **Field:** Cheminformatics · Chemistry

Every compound of the challenge reaches the computer as a line of text, so the first question is how a line of text can describe a molecule. A **molecule** is a set of **atoms** held together by **bonds**. An atom is
named by its element (C for carbon, N for nitrogen, O for oxygen, S for sulfur, Br for bromine). A bond is a pair of electrons shared by two atoms, and it comes as a **single bond**, a **double bond** (two shared pairs) or a **triple bond**
(three). Each element forms a characteristic number of bonds, its **valence**: carbon 4, nitrogen 3, oxygen 2, hydrogen and the halogens 1. The numbers follow from the way electrons fill shells. An atom is stable
when its outer shell holds eight electrons (two, for hydrogen), carbon has four outer electrons and so needs four more, which it gets by sharing four electron pairs; nitrogen has five and shares three, oxygen has six and
shares two. Sulfur, which sits below oxygen, can go beyond eight and has the valences 2, 4 and 6; the sulfonamide group of the example below is the case of 6.

A molecule is therefore a **molecular graph**: a mathematical object made of vertices (the atoms) and edges (the bonds), with a label on each vertex (the element) and each edge (single, double, triple, aromatic). The graph is
stored as an **adjacency matrix** $A$, a square table with one row and one column per atom in which $A_{ij} = 1$ if atoms $i$ and $j$ share a bond and $0$ otherwise. Since a bond joins two atoms in both
directions, $A$ is symmetric, the sum of row $i$ is the number of neighbours of atom $i$, and the sum of all entries is twice the number of bonds. The hydrogens attached to carbon are almost never written down: given the valence of
an atom and its bonds to other heavy atoms, the number of hydrogens is what is missing, $n_{\\mathrm{H}} = \\text{valence} - \\sum \\text{bond orders}$. The cell below reads the example compound of the challenge and
checks every one of these statements on it.
"""),
    code(MOLECULE_AS_GRAPH),
    md("""
The compound has {{=n_atoms}} heavy atoms and {{=n_bonds}} bonds; the valence of every atom is the usual one, which is why the hydrogens can be left out of the graph. The number of bonds (the half-sum of $A$) is larger than the number of atoms
minus one, which is what a tree would have, because the molecule has rings: a cyclobutane ring and a pyridine ring. A ring of six atoms in which the bonds are neither single nor double but something in
between is called **aromatic**, and it deserves a few lines because the heme-binding nitrogen that this project looks for (section 1.9) sits in such rings.

**Aromaticity.** Benzene, $\\mathrm{C_6H_6}$, was drawn by Kekule in 1865 as a ring with alternating single and double bonds. Measurements show that all six bonds have the same length, between the single-bond and the double-bond
length, and that the molecule is much more stable than three isolated double bonds would be. Each of the six carbons uses three of its four outer electrons in single bonds (two to carbon neighbours, one to hydrogen) and
keeps one in a $p$ orbital that points out of the plane of the ring; the six $p$ electrons are shared round the whole ring, a cloud above and below it. The quantum-mechanical treatment of Hueckel (1931) shows that such a
closed loop of $\\pi$ electrons is specially stable when it holds $4n + 2$ of them ($n = 0, 1, 2, \\dots$: 2, 6, 10, ...) and unstable when it holds $4n$. An **aromatic** ring is therefore flat, fully conjugated (every ring atom has
a $p$ orbital) and holds $4n + 2$ $\\pi$ electrons. Pyridine replaces one CH of benzene by a nitrogen. That nitrogen takes part in the $\\pi$ system with one electron, like the carbon it replaced, and keeps its
**lone pair** in the plane of the ring, pointing away from it: this is the electron pair that can bind the heme iron. In pyrrole and furan the heteroatom has no double bond in the ring and lends two electrons from its lone pair to the $\\pi$
system, which also gives six. The cell counts $\\pi$ electrons in a Kekule drawing (two for each double bond inside the ring, two for each ring heteroatom without one) and applies Hueckel's rule to seven rings, two of them
deliberately not aromatic.
"""),
    code(HUCKEL),
    md("""
The rule agrees with RDKit's decision for all seven rings: cyclohexane has no $\\pi$ system, and cyclooctatetraene has $8 = 4 \\cdot 2$ electrons, a ring that is not flat and not aromatic. (RDKit's own model of aromaticity is
a more elaborate version of the same idea, so that it also handles fused rings and rings with exocyclic double bonds.)

**SMILES.** The *simplified molecular-input line-entry system* (Weininger, 1988) writes the graph as one line of text by walking over it. The grammar has only a few rules.

- An atom is written by its element symbol; the hydrogens are implied by the valence. Atoms outside the organic subset (B, C, N, O, P, S, F, Cl, Br, I) or with a charge go in square brackets, as in `[NH4+]`.
- Bonds are `-` (single, usually omitted), `=` (double), `#` (triple); atoms next to each other in the text are bonded.
- A **branch** is written in parentheses and the walk returns to the atom before it, so `CC(C)C` is isobutane; `CC(C)(C)N` is a nitrogen on a carbon that carries three methyl groups.
- A **ring** is written by giving the same digit to the two atoms where the ring is cut open: `C1CCCCC1` is cyclohexane, and the digits are reused once closed.
- Aromatic atoms are written in lower case (`c1ccncc1` is pyridine); the Kekule drawing with alternating `=` and `-` is equally valid and describes the same molecule.
- Mirror-image forms are marked by `@` and `@@` after an atom (below), and charged atoms by `+` and `-` inside brackets.

The example compound is `{{=example}}`: a tert-butyl group `CC(C)(C)` on the nitrogen of a sulfonamide `NS(=O)(=O)`, whose sulfur is attached to a cyclobutane `C1(...)CCC1`, which carries a
`CNC2=C(Br)C=NC=C2C#N` chain: a bromo-cyano-pyridine joined through an amine. A walk can start at any atom and choose any branch first, so one molecule has many valid strings. The **canonical SMILES** (Weininger and
co-workers, 1989) removes that freedom: the atoms are first ranked by an invariant of their environment (the same idea as the Morgan algorithm of section 3.1) and ties are broken so that the ranking is unique, then the walk starts at the
atom of lowest rank and takes the branches in order of rank. Two strings that describe the same molecule therefore have the same canonical form, and a molecule can be looked up by its canonical string. The next cell takes 2000 random walks.
"""),
    code(CANONICAL),
    md("""
The strings of the file are not in the canonical form of the software used in this project (only {{=as_given:.1%}} of them are), because they are written in the Kekule style; reading them with RDKit and writing them again gives the
canonical aromatic string. This is the reason all comparisons between compounds of this project, here and in the other data sources (section 2.9), are made on strings that were first read into a molecule and written back, and never on the text as given.

**Stereochemistry.** A carbon with four different neighbours exists in two forms that are mirror images of each other and cannot be turned into one another without breaking bonds: **enantiomers**. The pair of hands is the
usual picture. The two forms have the same graph and often very different effects in the body, since an enzyme is itself made of one hand of building blocks. SMILES records the configuration with `@` and `@@`
(the neighbours of the marked atom, looked at from the first one, run anticlockwise or clockwise), and the software derives the labels R and S of the Cahn-Ingold-Prelog convention from it.
"""),
    code(STEREO),
    md("""
Most compounds that could carry a stereocentre do not say which configuration they have: {{=n_unspecified}} of the {{=len(train)}} training compounds have a possible centre without a configuration, and for them the file does not distinguish a molecule from its
mirror image (the sample was either racemic, a 1:1 mixture of the two forms, or not specified). A model that reads this file can only learn from the flat graph for these compounds, and for the rest the
marks add one more way in which two otherwise identical strings differ. The file has {{=n_shared_graphs}} flat graphs that are shared by two compounds, that is {{=n_shared_graphs}} pairs of stereoisomers, and the pairs are labelled differently {{=n_conflicts}} times; the consequences of stereochemistry for the features are in section 3.1.
""" ),
    md("""
## 2.2 The training and blinded test files, column by column

> **Field:** Data science

The challenge supplies two files (section 0.3). The **training file** has {{=len(train)}} rows, one per compound, and {{=train.shape[1]}} columns: the compound's name and structure (`Molecule_Name`, `SMILES`), and for each of four enzymes
the following set of columns, whose names follow one pattern, `<enzyme>_<quantity>`:

- `pIC50_direct_inhibition`: the potency in the **direct arm**, $d$ (section 1.8): enzyme, compound and substrate mixed at once;
- `pIC50_TDI_condition`: the potency in the **TDI arm**, $t$: the enzyme and the compound are left together before the substrate is added;
- `_conf_low` and `_conf_high`, appended to either name: the lower and upper limit of a 95 % **confidence interval** of the pIC50, a range that holds the true value with 95 % probability (section 2.5);
- `_std`, appended to either name: its standard deviation;
- `_is_TDI`, only for CYP3A4 and CYP2D6: the challenge's yes/no label, the thing to be predicted.

The **blinded test file** has {{=len(test)}} rows and two columns, the name and the structure. It holds no measurement and no label: the organisers keep them and compute the score from the predictions that are uploaded (section 0.3).
The cell counts what the training file holds.
"""),
    code(FILE_STATS),
    md("""
Three kinds of absence can be read from the table. **Never measured**: the quantity is simply empty, because the compound was not tested on that enzyme. **Measured only in one arm**: the TDI arm exists and the direct arm does not
({{=tdi_only['CYP3A4']}} compounds of CYP3A4 and {{=tdi_only['CYP2D6']}} of CYP2D6, section 2.4); the opposite case, a direct arm without a TDI arm, occurs {{=direct_only_total}} time(s) in all. And only two enzymes have a label. The file does not state why a compound was tested on one enzyme and not on another; what the data
reveal about this is the subject of section 2.6. The compound numbers of the training and test files overlap in range: the two files are drawn from the same numbering, and the test compounds are interleaved with the
training ones and not separated from them by date or by number.

One more feature of the table is a single compound with a CYP3A4 label and a direct-arm value, but with no TDI-arm value, labelled negative; it is one row of {{=len(train)}} and is not used when the label is
studied against the two measurements. The figure shows the same table as a picture.
"""),
    figure("data_coverage"),
    md("""
## 2.3 The label rule: how $s = t - \\max(d, 4) > \\log_{10} 2$ was recovered and checked

> **Field:** Data science · Pharmacology

The challenge's FAQ gives the label as a four-way rule on the two potencies. Let $d$ be the direct pIC50 and $t$ the TDI-arm pIC50. The compound is a time-dependent inhibitor when

- $d > 4$ and $t - d > \\log_{10} 2$ (it became more than twice as potent after the head start), or
- $d < 4$ and $t > 4 + \\log_{10} 2$ (the **inferred positive**: the direct arm showed no measurable effect, below the floor of section 1.7, and the TDI arm did),

and it is not one in all other cases. The two branches are one rule. Write $s = t - \\max(d, 4)$, the shift measured against the direct value or the floor, whichever is larger. If $d > 4$ the maximum is $d$ and $s = t - d$; if $d < 4$
the maximum is 4 and $s = t - 4$, so $s > \\log_{10} 2 \\iff t > 4 + \\log_{10} 2$. In both cases the label is positive exactly when

$$s = t - \\max(d, 4) \\;>\\; \\log_{10} 2 \\approx 0.30103 .$$

(A compound with $d$ exactly 4 is covered by the second form and by the first, which agree there, so the choice does not matter.) The point of writing it as one inequality is practical: $s$ is a **continuous** number carrying the whole information of the pair, while the label is only its
sign relative to a threshold. A model can learn the continuous quantity and be turned into a classifier at the end, which the project does (Part VII). The cell checks the equivalence of the two forms on 200,000 random pairs and on a regular grid,
and then applies the rule to the real compounds of the training file.
"""),
    code(LABEL_RULE),
    md("""
No compound of the file has $d$ exactly 4 (the cell counted {{=n_exact_four}}), so the two forms never meet at their boundary. The rule reproduces **every** label of the compounds with both arms ({{=rule_rows[0]['labels reproduced by the rule']}} of {{=rule_rows[0]['compounds with both arms']}} for CYP3A4 and {{=rule_rows[1]['labels reproduced by the rule']}} of {{=rule_rows[1]['compounds with both arms']}} for CYP2D6). This was not
read off the FAQ and trusted: the numbers 4 and $\\log_{10} 2$ can also be *recovered from the data*, and the table of intervals shows how tightly. The thresholds that give no error at all must lie above the highest shift of any negative and
at or below the lowest shift of any positive: for CYP3A4 that is the interval from {{=intervals['CYP3A4'][0]:.5f}} to {{=intervals['CYP3A4'][1]:.5f}}, an interval of width {{=intervals['CYP3A4'][1] - intervals['CYP3A4'][0]:.5f}}. The value 0.3, which one might round to,
is excluded, and $\\log_{10} 2$ is inside; the data therefore say that the threshold is a factor of two in potency and not one tenth of a pIC50 unit rounded. The floor can be tested the same way, by trying other floors and thresholds on a grid:
"""),
    code(GRID_SEARCH),
    md("""
Each enzyme has exactly {{=zero_cells['CYP3A4']}} grid cell with no wrong label, the one at floor 4 and threshold $\\log_{10} 2$; every other cell makes at least {{=smallest_error['CYP3A4']}} error(s) for CYP3A4 and {{=smallest_error['CYP2D6']}} for CYP2D6 (the smallest is the neighbour with threshold 0.3, which the largest negative shift just rules out), and the cells away from the optimum make dozens to hundreds. The label of this challenge is
a deterministic function of two measured numbers, with a floor at the weakest concentration the assay can distinguish (pIC50 4, an IC50 of 100 micromolar) and a threshold at a two-fold gain in potency. The files do not say *why* these values were chosen; they
only show that these are the values used. The figure draws the rule on the real compounds, and the second shows where the shifts of the compounds lie relative to the threshold.
"""),
    figure("data_label_rule"),
    figure("data_shift_distribution"),
    md("""
The picture holds a warning that the next sections follow up. A large number of compounds lie close to the threshold: {{=near_threshold}} of the
{{=len(both_3a4)}} CYP3A4 compounds have a shift within 0.1 pIC50 units of the threshold. A measurement of a pIC50 is not exact (section 1.7), so for such a compound a repeated experiment could well fall on the other side, and
its label would be different. That is the subject of section 2.5.

## 2.4 The contamination found in the labels

> **Field:** Data science

The cell of section 2.2 showed that {{=tdi_only['CYP3A4']:,}} CYP3A4 compounds have a label although their direct arm is empty. They are the single most important irregularity of the training file. The cell below
tabulates the three kinds of CYP3A4 rows.
"""),
    code(POPULATIONS),
    md("""
**What is wrong with these rows.** For a compound with a measured direct arm, the label is computed from $s$. A compound without one has no baseline from which to compute a shift. The {{=n_no_direct}} rows nevertheless carry a label, and
it is *negative for every one of them*. Their TDI-arm potency is not low, however: the median is {{=median_t_no_direct:.2f}}, higher than the median of the compounds with both arms, and {{=share_would_be_positive:.0%}} of them have $t > 4.301$. If the missing direct pIC50 had been below the
floor, the written rule would have called most of them positive. So these rows follow another convention than the rule: *no direct measurement, therefore negative*. The blinded test set is, according to the organisers, assayed
in full dose-response mode in both arms, so it contains no such compound. The training file thus mixes two populations with different labelling conventions, and only one of them resembles what will be scored.

**How it came to light.** The project's first CYP3A4 results used every row that had a label. Recovering the label rule (section 2.3) showed that it can be applied only to the compounds with both arms, and counting the
rest gave the table above. The early CYP3A4 numbers had therefore been computed on the mixed population, and the early ablations were redone on the corrected one (Part VII says, experiment by experiment, which numbers are of which kind). CYP2D6 is hardly affected: only {{=tdi_only['CYP2D6']}} of its labelled compounds lack the direct arm.

**The fix.** In the training of every model and in every validation set the project uses the compounds with both arms only (`training_view` tags the three populations; `population == both_arms`). The labelled-negative rows with an empty direct arm are not discarded from
knowledge altogether: their TDI-arm potency is real data, and a regression model can use it through `shift_imputed`, which fills the missing baseline with the floor, $s = t - 4$, the value they would have if the direct arm had been below
the floor. This is *not* a measurement and is used with that reading only. The cell measures what the contamination does to a model, in the cleanest way available: the same **gradient-boosted trees** (a model made of many small **decision trees**, flowcharts of yes/no questions about the features, fitted one after another, each correcting the mistakes of those before it; LightGBM is the software used, section 4.3) trained twice, and always scored on held-out compounds that have both arms.
"""),
    code(CONTAMINATION_EFFECT),
    md("""
The model trained with the extra rows ranks the compounds with both arms **worse** ({{=auc_mixed:.3f}} against {{=auc_clean:.3f}} in terms of the **AUC**, here the area under the ROC curve of section 4.10 and not the area under the concentration curve of section 1.1: the probability that a randomly chosen positive compound is scored above a randomly chosen negative one, 0.5 for chance).
The harm is the one expected: the rows teach the model that a potent TDI arm without a direct arm means "negative", which is false for the compounds that will be scored. A model evaluated on the mixed population would also have reported a number that
describes the mixture and not the problem. This is one of the findings of the project that is not about a model at all, and it is recorded in the checklist and in the method report as the contaminated no-direct rows.

## 2.5 Measurement noise: the standard deviation, the confidence limits and the confidence $z$

> **Field:** Statistics

Every pIC50 of the file is an estimate from a fitted curve (section 1.7), and the file gives its uncertainty twice, as a **standard deviation** and as the limits of a 95 % **confidence interval**. The standard deviation of an estimate is the
typical size of its error: if the experiment were repeated many times, the estimates would scatter around the true value with this spread. For a normally distributed error, 95 % of the repetitions fall within 1.96 standard deviations of the true value, because the standard
normal distribution function $\\Phi(x) = \\tfrac12[1 + \\mathrm{erf}(x/\\sqrt 2)]$ equals 0.975 at $x = 1.96$; a 95 % interval is therefore the value plus and minus 1.96 times the standard deviation. The two columns should then be the same
information, and the cell checks it.
"""),
    code(NOISE),
    md("""
The interval is centred on the value and its half-width is close to 1.96 standard deviations in every row of the table (the ratio varies a little between compounds because the limits come from the fitting procedure and not from the formula). The standard
deviation is therefore the one number needed, and the typical pIC50 error is {{=sd_values[('CYP3A4', 'TDI arm')]:.2f}} (TDI arm) and {{=sd_values[('CYP3A4', 'direct arm')]:.2f}} (direct arm) for CYP3A4.

**Error propagation.** The label depends on the *difference* $s = t - \\max(d, 4)$ of two noisy numbers, so the noise of $s$ has to be derived. For two independent random quantities $X$ and $Y$ and constants $a$ and $b$, the variance of the combination
$aX + bY$ is $a^2\\sigma_X^2 + b^2\\sigma_Y^2$; the term $2ab\\,\\mathrm{Cov}(X, Y)$ that would otherwise appear vanishes when the quantities are independent, as the two assay arms are taken to be, since they are separate experiments. With $a = 1$ and $b = -1$ the
variance of the difference is the sum of the variances. When the direct pIC50 is above the floor, $s = t - d$ and

$$\\mathrm{sd}(s)^2 = \\mathrm{sd}(t)^2 + \\mathrm{sd}(d)^2 ,$$

where $\\mathrm{sd}(\\cdot)$ is the standard deviation of the column of the same name. When $d$ is below the floor the second term is the constant 4, with no noise, and $\\mathrm{sd}(s)^2 = \\mathrm{sd}(t)^2$. The formula is the one in the module
`cypmode/challenge/label_confidence.py` (the term $[d \\ge 4]\\,\\mathrm{sd}(d)^2$, where $[\\cdot]$ is 1 if the statement in the brackets is true and 0 if not). A simulation can check it, by drawing both measurements from normal distributions and forming $s$.
"""),
    code(PROPAGATION),
    md("""
The spread of the simulated shifts agrees with the formula in both cases. In the second row, where $d$ is far below the floor, the noise of $d$ makes no difference, as the formula says.

**The confidence $z$.** How sure can one be of a label? The label is "the shift is above the threshold". The measured shift $s$ has the uncertainty $\\mathrm{sd}(s)$; treating the true shift as normally distributed around the measurement (the file's
standard deviations are posterior standard deviations, so this is the natural reading), the probability that the true shift lies on the other side of the threshold from the measured one is the area of the normal curve beyond the distance of the threshold, measured in standard deviations ($P$ stands for probability):

$$z = \\frac{|s - \\log_{10} 2|}{\\mathrm{sd}(s)}, \\qquad P(\\text{label wrong}) = \\Phi(-z).$$

$z = 1$ is a one-in-six chance of the opposite label, $z = 1.96$ is 2.5 %, $z = 2.5$ is 0.6 %. The module `label_confidence` computes $z$ for every row (`confidence_z`) and offers weights that use it (`confident_weights`, `training_weights`).
The challenge's FAQ says that predictions are requested for every test compound, but that only compounds whose label can be assigned with confidence contribute to the score. The exact rule is not published, so every analysis of the project is repeated across several cut-offs $z_0$, and a compound counts as
*confident* if $z \\ge z_0$. The cell computes $z$ for the training compounds.
"""),
    code(CONFIDENCE),
    md("""
Of the {{=n_both['CYP3A4']}} CYP3A4 compounds with both arms, about {{=expected_wrong['CYP3A4']:.0f}} of the labels are expected to be wrong (the sum of $\\Phi(-z)$ over the compounds) and {{=n_confident['CYP3A4']}} reach $z \\ge 1.96$; the corresponding numbers for CYP2D6 are
{{=expected_wrong['CYP2D6']:.0f}} wrong in {{=n_both['CYP2D6']}} and {{=n_confident['CYP2D6']}} confident. The median CYP3A4 compound is {{=quantiles[2]:.1f}} standard deviations from the threshold and the lower quartile only {{=quantiles[1]:.1f}}; {{=share_below_2:.0%}} of the compounds are less than two standard deviations from it.
Two consequences are drawn in later parts. A validation score computed on *all* compounds measures a partly random problem, whereas the official score is computed on the confident subset; section 6.4 builds a yardstick that restricts to confident compounds and shows that
the official baseline's own cross-validated score rises with the cut-off in the direction of its leaderboard score. And a model can be trained with the labels of uncertain compounds down-weighted, which is one of the experiments of Part IX.
"""),
]
