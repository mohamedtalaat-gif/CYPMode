"""Part III (first half): molecules as numbers. Morgan fingerprints, Tanimoto similarity, descriptors, the other fingerprint blocks."""

from notebooks.notebook_builder import code, figure, md

MORGAN = '''
# The Morgan algorithm written out (a toy that uses nothing but the graph of the example compound) and compared with RDKit's own. Every atom starts with an identifier made from five properties
# (atomic number, number of bonded heavy atoms, number of hydrogens, charge, ring or not). In each round an atom's new identifier is a hash of its old identifier and of the sorted list of
# (bond order, identifier) of its neighbours, so after r rounds it stands for everything within r bonds of the atom.
from rdkit.Chem import rdFingerprintGenerator

def toy_identifiers(molecule, radius):
    ids = [hash((a.GetAtomicNum(), a.GetDegree(), a.GetTotalNumHs(), a.GetFormalCharge(), int(a.IsInRing()))) for a in molecule.GetAtoms()]
    rounds = [ids[:]]
    for _ in range(radius):
        updated = []
        for atom in molecule.GetAtoms():
            neighbours = sorted((molecule.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble(), ids[n.GetIdx()]) for n in atom.GetNeighbors())
            updated.append(hash((ids[atom.GetIdx()], tuple(neighbours))))
        ids = updated
        rounds.append(ids[:])
    return rounds

def classes(identifiers):
    groups = {}
    for atom, identifier in enumerate(identifiers):
        groups.setdefault(identifier, []).append(atom)
    return sorted(map(tuple, groups.values()))

rounds = toy_identifiers(mol, 2)
rdkit_rows, agree = [], []
for redundant in (True, False):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, includeRedundantEnvironments=redundant)
    extra = rdFingerprintGenerator.AdditionalOutput()
    extra.AllocateBitInfoMap()
    generator.GetSparseCountFingerprint(mol, additionalOutput=extra)
    environment = {(atom, r): identifier for identifier, entries in extra.GetBitInfoMap().items() for atom, r in entries}      # (atom, radius) -> unfolded identifier
    for r in range(3):
        theirs = {a: identifier for (a, rr), identifier in environment.items() if rr == r}
        same = classes([theirs[a] for a in sorted(theirs)]) == classes([rounds[r][a] for a in sorted(theirs)]) if len(theirs) == mol.GetNumAtoms() else None
        if redundant:
            agree.append(same)
    rdkit_rows.append((redundant, {r: len({i for (a, rr), i in environment.items() if rr == r}) for r in range(3)}))
by_radius = pd.DataFrame([{"radius": r, "atoms in the round": mol.GetNumAtoms(), "distinct identifiers in the toy": len(set(rounds[r])),
                           "distinct identifiers in RDKit, all environments": rdkit_rows[0][1][r], "distinct identifiers in RDKit, redundant ones dropped": rdkit_rows[1][1][r],
                           "same grouping of the atoms as RDKit's": "yes" if agree[r] else "no"} for r in range(3)])
show_table(by_radius)

# Folding: an identifier is a 32-bit number; the fingerprint position is the identifier modulo the width (here 2048). Different environments can land on one position.
unfolded = rdFingerprintGenerator.GetMorganGenerator(radius=2).GetSparseCountFingerprint(mol).GetNonzeroElements()
folded = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)
k, m_bits = len(unfolded), 2048
expected_on = m_bits * (1 - (1 - 1 / m_bits) ** k)
print(f"{k} distinct environments are written to {folded.GetNumOnBits()} of the {m_bits} positions; if the positions were chosen at random one would expect {expected_on:.1f} occupied positions")
print(f"{sum(unfolded.values())} environments counted with repeats; {sum(1 for c in unfolded.values() if c > 1)} of the {k} occur more than once")
n_environments, n_on_bits, n_expected_on = k, folded.GetNumOnBits(), expected_on
'''

MORGAN_STORE = '''
# What the Morgan blocks of the stored feature table look like over all 6895 compounds (the file is built in section 3.7): how many positions are on in a molecule, and how often each position is on.
store = np.load(ROOT / "data/tdi_track/outputs/caches/feature_store.npz", allow_pickle=True)
store_smiles = list(store["smiles"])
assert store_smiles == list(train["SMILES"]) + list(test["SMILES"])             # rows 0 .. 6144 are the training compounds, 6145 .. 6894 the test compounds
bits = store["morgan_bits"]
on_per_molecule = bits.sum(axis=1)
frequency = bits.mean(axis=0)
morgan_rows = [{"quantity": "positions switched on per molecule (median, 5th to 95th percentile)", "value": f"{np.median(on_per_molecule):.0f} ({np.percentile(on_per_molecule, 5):.0f} to {np.percentile(on_per_molecule, 95):.0f})"},
               {"quantity": "positions that are on in no compound", "value": f"{int((frequency == 0).sum())} of 2048"},
               {"quantity": "positions that are on in more than half of the compounds", "value": f"{int((frequency > 0.5).sum())}"},
               {"quantity": "positions that are on in fewer than 1 % of the compounds", "value": f"{int((frequency < 0.01).sum())}"},
               {"quantity": "share of all entries of the table that are 1", "value": f"{bits.mean():.1%}"}]
show_table(pd.DataFrame(morgan_rows))
sparsity = float(1 - bits.mean())
radius3 = store["morgan_c3"]
counts2 = store["morgan_c2"]
print(f"the count block with radius 2 has a largest count of {int(counts2.max())}; the radius-3 block sums to {radius3.sum(axis=1).mean():.1f} environments per molecule against {counts2.sum(axis=1).mean():.1f} for radius 2")
'''

TANIMOTO = '''
# The Tanimoto similarity by hand and by RDKit, and why the obvious alternative, the fraction of positions on which two fingerprints agree, does not work for fingerprints that are mostly zeros.
generator_2048 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
example_fp = generator_2048.GetFingerprint(mol)
all_train_fp = [generator_2048.GetFingerprint(Chem.MolFromSmiles(s)) for s in train["SMILES"]]
similarities = np.array(DataStructs.BulkTanimotoSimilarity(example_fp, all_train_fp))
similarities[train["Molecule_Name"].to_numpy() == example_name] = -1
nearest = int(similarities.argmax())
other_fp = all_train_fp[nearest]
on_a, on_b = set(example_fp.GetOnBits()), set(other_fp.GetOnBits())
a, b, c = len(on_a), len(on_b), len(on_a & on_b)
by_hand = c / (a + b - c)
print(f"the example has a = {a} positions on, its nearest training compound ({train['Molecule_Name'][nearest]}) has b = {b}, and c = {c} are on in both")
print(f"Tanimoto by hand c / (a + b - c) = {by_hand:.4f}; RDKit gives {DataStructs.TanimotoSimilarity(example_fp, other_fp):.4f}")

# 4000 random pairs of training compounds: Tanimoto against the agreement of all 2048 positions (1 minus the Hamming distance divided by the width).
rng = np.random.default_rng(0)
pairs = [(int(i), int(j)) for i, j in rng.integers(0, 1000, size=(4000, 2)) if i != j]
tanimoto_values = np.array([DataStructs.TanimotoSimilarity(all_train_fp[i], all_train_fp[j]) for i, j in pairs])
agreement_values = np.array([1 - (all_train_fp[i] ^ all_train_fp[j]).GetNumOnBits() / 2048 for i, j in pairs])
compare_rows = [{"measure": "Tanimoto similarity", "smallest": tanimoto_values.min(), "median": np.median(tanimoto_values), "largest": tanimoto_values.max()},
                {"measure": "fraction of the 2048 positions that agree", "smallest": agreement_values.min(), "median": np.median(agreement_values), "largest": agreement_values.max()}]
show_table(pd.DataFrame(compare_rows))

# For counts instead of bits the same idea is the min / max form: sum of the smaller counts over sum of the larger counts.
count_a = generator_2048.GetCountFingerprintAsNumPy(mol).astype(float)
count_b = generator_2048.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(train["SMILES"][nearest])).astype(float)
minmax = np.minimum(count_a, count_b).sum() / np.maximum(count_a, count_b).sum()
print(f"the min / max form on the counts of the same two molecules: {minmax:.4f}")
nearest_compound, tanimoto_median, agreement_median = train["Molecule_Name"][nearest], float(np.median(tanimoto_values)), float(np.median(agreement_values))
'''

DESCRIPTORS = '''
# Some of the 217 RDKit descriptors, computed for the example compound, and two of them rebuilt from their parts: the Wildman-Crippen logP and the topological polar surface area are
# sums over atoms, so the contributions of the single atoms must add up to the value of the descriptor.
from rdkit.Chem import Crippen

with_h = Chem.AddHs(mol)
crippen = rdMolDescriptors._CalcCrippenContribs(with_h)                    # (logP contribution, molar refractivity contribution) of every atom, hydrogens included
tpsa_parts = rdMolDescriptors._CalcTPSAContribs(mol)                       # contribution of every heavy atom in square angstrom
log_p_sum, tpsa_sum = sum(x[0] for x in crippen), sum(tpsa_parts)
assert abs(log_p_sum - Crippen.MolLogP(mol)) < 1e-9 and abs(tpsa_sum - rdMolDescriptors.CalcTPSA(mol)) < 1e-9
polar_atoms = [(mol.GetAtomWithIdx(i).GetSymbol(), round(v, 2)) for i, v in enumerate(tpsa_parts) if v > 0]
by_atom = pd.DataFrame({"element": [with_h.GetAtomWithIdx(i).GetSymbol() for i in range(with_h.GetNumAtoms())], "logP contribution": [x[0] for x in crippen]})
by_group = by_atom.groupby("element").agg(atoms=("logP contribution", "size"), total=("logP contribution", "sum")).sort_values("total").reset_index()
assert {symbol for symbol, _ in [(mol.GetAtomWithIdx(i).GetSymbol(), v) for i, v in enumerate(tpsa_parts) if v > 0]} <= {"N", "O"}
show_table(by_group, floatfmt=".3f")
descriptor_rows = [{"descriptor": "molecular weight (g/mol)", "value": Descriptors.MolWt(mol)}, {"descriptor": "Wildman-Crippen logP", "value": Crippen.MolLogP(mol)},
                   {"descriptor": "topological polar surface area (A^2)", "value": rdMolDescriptors.CalcTPSA(mol)}, {"descriptor": "hydrogen-bond donors", "value": rdMolDescriptors.CalcNumHBD(mol)},
                   {"descriptor": "hydrogen-bond acceptors", "value": rdMolDescriptors.CalcNumHBA(mol)}, {"descriptor": "rotatable bonds", "value": rdMolDescriptors.CalcNumRotatableBonds(mol)},
                   {"descriptor": "rings", "value": rdMolDescriptors.CalcNumRings(mol)}, {"descriptor": "fraction of sp3 carbons", "value": rdMolDescriptors.CalcFractionCSP3(mol)}]
show_table(pd.DataFrame(descriptor_rows), floatfmt=".2f")
print(f"logP: the sum of the {len(crippen)} atom contributions is {log_p_sum:.4f} and the descriptor is {Crippen.MolLogP(mol):.4f}; TPSA: the polar atoms {polar_atoms} add up to {tpsa_sum:.2f}")
n_descriptors = len(Descriptors.descList)
'''

DESCRIPTOR_SHIFT = '''
# The descriptors of the training compounds against those of the test compounds (the cached block, columns named by the descriptor list).
names = [name for name, _ in Descriptors.descList]
table = store["rdkit_desc"]
assert table.shape[1] == len(names)
columns = {"MolWt": "molecular weight", "MolLogP": "logP", "TPSA": "polar surface area", "NumHDonors": "H-bond donors", "NumHAcceptors": "H-bond acceptors", "NumRotatableBonds": "rotatable bonds", "FractionCSP3": "fraction sp3"}
shift_rows = []
for key, title in columns.items():
    column = table[:, names.index(key)]
    shift_rows.append({"descriptor": title, "training median": np.median(column[: len(train)]), "test median": np.median(column[len(train):]), "training 5th to 95th": f"{np.percentile(column[: len(train)], 5):.1f} to {np.percentile(column[: len(train)], 95):.1f}",
                       "test 5th to 95th": f"{np.percentile(column[len(train):], 5):.1f} to {np.percentile(column[len(train):], 95):.1f}"})
show_table(pd.DataFrame(shift_rows), floatfmt=".2f")
mw_train, mw_test = shift_rows[0]["training median"], shift_rows[0]["test median"]
'''

BLOCKS = '''
# Every block of the feature store, with its real width in the stored table, its kind, and what the example compound looks like in it. The default blocks are computed here from the compound with
# the same functions the store uses (BLOCKS in feature_store.py); the opt-in blocks come from caches built by their own processes and are only read.
from cypmode.challenge.feature_store import BLOCKS as BLOCK_FUNCTIONS, OPT_IN_BLOCKS

meaning = {
    "morgan_bits": ("bits", "Morgan environments up to radius 2 present or absent (section 3.1)"),
    "morgan_c2": ("counts", "the same, with the number of occurrences"),
    "morgan_c3": ("counts", "Morgan environments up to radius 3, counted"),
    "fmorgan_c2": ("counts", "Morgan with atoms described by their pharmacophoric kind, counted"),
    "atompair_c": ("counts", "pairs of atom types with the number of bonds between them, counted"),
    "torsion_c": ("counts", "paths of four bonded atoms, counted"),
    "maccs": ("bits", "166 predefined structural questions, plus an unused first position"),
    "avalon": ("bits", "hashed path and feature fragments (Avalon)"),
    "erg": ("real", "extended reduced graph: pairs of pharmacophore-like nodes at binned distances"),
    "rdkit_desc": ("real", "all 217 descriptors of RDKit (section 3.3)"),
    "motif": ("bits", "six heme-binding nitrogen motifs and whether any is present (Part V)"),
    "tdi_alerts": ("counts", "30 structural alerts of section 1.10, a ring-size flag and the largest ring"),
    "tdi_alert_context": ("real", "the alerts weighted by lipophilicity, size, fluorine nearby and labile groups"),
    "amine": ("bits", "a trialkylamine that is not a sulfonamide, enamine or guanidine (section 5.2)"),
    "ext_activity": ("real", "five probabilities from models trained on the TDC panel only (section 2.8)"),
    "chemeleon": ("real", "64 principal components of a frozen pretrained embedding (section 9.4)"),
    "physics": ("real", "GFN2-xTB electronic and site descriptors, 133 columns (Part X)"),
    "physics_b": ("real", "Boltzmann-weighted and relaxed physics, 22 columns (Part X)"),
    "basic_centre": ("real", "protonation states at pH 6.4 to 8.4 and the distances between the charged nitrogen and the aromatic rings (Part IX)"),
    "monroe": ("real", "64 principal components of a frozen pretrained graph-transformer embedding (Part IX)"),
    "ext_cont": ("real", "predicted response and potency of the external cytochrome panel, a pair for each of five enzymes (Part IX)"),
    "ext_tox": ("real", "predicted response and potency of the Tox21 cytochrome screen, a pair for each of five enzymes (Part IX)"),
    "ext_adme": ("real", "two predicted properties, logD and mouse brain protein binding, from surrogate networks (Part IX)"),
    "atomjepa": ("real", "64 principal components of the frozen embeddings of a pretrained equivariant network (Part IX)"),
}
example_blocks = {name: np.asarray(BLOCK_FUNCTIONS[name]([mol])).ravel() for name in BLOCK_FUNCTIONS if name not in OPT_IN_BLOCKS}
block_rows = []
for name in store.files:
    if name == "smiles":
        continue
    width = store[name].shape[1]
    row = {"block": name, "columns": width, "kind": meaning[name][0], "what it holds": meaning[name][1], "opt-in": "yes" if name in OPT_IN_BLOCKS else "no"}
    if name in example_blocks:
        assert example_blocks[name].shape[0] == width
        row["non-zero for the example"] = int((example_blocks[name] != 0).sum())
    else:
        row["non-zero for the example"] = "-"
    block_rows.append(row)
block_table = pd.DataFrame(block_rows)
show_table(block_table)
n_blocks, n_columns_total = len(block_table), int(block_table["columns"].sum())
n_opt_in = int((block_table["opt-in"] == "yes").sum())
'''

CELLS = [
    md("""
# Part III. Molecules as numbers

A learning algorithm takes a table of numbers: one row per compound, one column per measured or computed quantity, all rows of the same length. A molecule is a graph of atoms (section 2.1), of any size and shape, so the first
job is to turn it into a row of a fixed length that keeps what matters for the question. There is no single right way, and different ways keep different things, so the project uses many of them side by side and lets the models decide. This part builds them
one by one: the fingerprints, which record which small pieces of structure are present (3.1 to 3.4), the three-dimensional pharmacophore fingerprint (3.5), the grouping of compounds into chemical series, which is needed for honest validation
(3.6), and the feature store in which all the blocks are kept (3.7). Two rules hold for everything in this part: a feature is computed from the structure alone, never from a label, so that it cannot leak an answer into a model, and a feature that the project defined
and measured as useless is still described here, because what it was meant to do is part of the record.

## 3.1 Morgan (circular) fingerprints

> **Field:** Cheminformatics · Mathematics (hashing)

**The need.** A molecule of 23 atoms and a molecule of 60 atoms must both become a row of the same length, and two molecules that share a chemically meaningful piece, for example the same pyridine ring with the same substituents, must share something in their rows. A table with one column
per atom would have the wrong length and no meaning for a column. The idea of the **fingerprint** (which section 0.2 introduced as a list of yes/no answers) is to ask a long, fixed list of questions of the form "does the molecule contain *this* small piece of structure?" and to record the answers.
The questions need not be written by hand. The *Morgan algorithm* (Morgan, 1965, for identifying molecules in a database; used for fingerprints by Rogers and Hahn, 2010, under the name extended-connectivity fingerprints, ECFP) generates them from the molecule itself.

**The algorithm.** Describe each atom by its **environment**, everything within a given number of bonds, the **radius**. At radius 0 the environment is the atom alone; at radius 1 the atom with its bonded neighbours; at radius 2 also the neighbours of the neighbours. The algorithm builds the description
of these environments by repeated rounds. (1) Every atom gets a starting integer identifier computed from simple properties of the atom: its atomic number, the number of heavy atoms bonded to it, its number of hydrogens, its charge and whether it is in a ring. (2) In each round every atom collects the identifiers of its neighbours together with the order of the bond to each (single, double, ...), sorts that list so that it
does not depend on the numbering of the atoms, and combines its own identifier with the list into a new identifier by a **hash function**. A hash function is a fixed recipe that turns any input into a number so that equal inputs always give equal numbers and different inputs almost always give different ones. After round $r$ the identifier of an atom stands for its whole
environment of radius $r$. (3) All identifiers of all rounds up to the chosen radius are collected; the set (or, for a count fingerprint, the multiset) is the molecule's fingerprint. The fingerprint of radius 2 is also called ECFP4, because the largest environment is 4 bonds across. (4) To get a fixed length, each identifier is **folded**: the position in the row is the identifier modulo the width $m$ (the remainder after division by $m$), here $m = 2048$.

Folding has a price: two different environments can fall on the same position, a **collision**. If $k$ distinct environments were thrown at random into $m$ positions, a given position stays empty with probability $(1 - 1/m)^k$, so the expected number of occupied positions is $m\\,[1 - (1 - 1/m)^k]$, slightly fewer than $k$. A **bit** fingerprint records only whether a position is occupied; a **count** fingerprint records how many environments fell on it, which also tells a molecule with three methyl groups from a molecule with one.
The cell implements the rounds in a few lines, compares them with RDKit's implementation on the example compound, and checks the arithmetic of folding.
"""),
    code(MORGAN),
    md("""
The toy and RDKit group the atoms in the same way at every radius: the algorithm is exactly the one described. RDKit by default drops an environment when it covers exactly the same bonds as an earlier one (a terminal atom's environment at radius 2 may be a copy of what its neighbour already had), which
is why its default count at radius 2 is lower than the toy's. The example compound has {{=n_environments}} distinct environments, folded onto {{=n_on_bits}} positions of the 2048, where {{=n_expected_on:.1f}} occupied positions are expected if the
identifiers behaved like random numbers: the hashing spreads the environments about as evenly as chance would, and few collide. The figure shows the algorithm on a real training compound, radius by radius.
"""),
    figure("mol_morgan_steps"),
    md("""
The cell below reads the stored table of the project (built in section 3.7) and shows how the Morgan block looks over all {{=len(train) + len(test)}} compounds.
"""),
    code(MORGAN_STORE),
    md("""
A fingerprint is **sparse**: only {{=1 - sparsity:.1%}} of the entries of the bit table are 1, which is why the bits are described as questions most molecules answer "no" to, and why the right measure of similarity is not the usual
one (section 3.2). The radius-3 count block records more environments per molecule than radius 2, as it must, since it contains everything of radius 2 and the environments of one more round. Which of these representations a model uses is decided by the validation of Part VI and not by taste.

## 3.2 Tanimoto similarity and why it suits fingerprints

> **Field:** Cheminformatics · Mathematics (set similarity)

To compare two molecules one needs a number that is large when they share many structural pieces and small when they share few. Two fingerprints can be seen as two sets, $A$ and $B$, the sets of positions that are on, with $a = |A|$ and $b = |B|$ elements, of which $c = |A \\cap B|$ are common. The **Tanimoto similarity**
(the index that Jaccard proposed in 1901 for comparing plant communities, introduced for chemistry by Tanimoto in 1958) is the share of the positions on in *either* fingerprint that are on in *both*:

$$T(A, B) \\;=\\; \\frac{|A \\cap B|}{|A \\cup B|} \\;=\\; \\frac{c}{a + b - c}, \\qquad 0 \\le T \\le 1,$$

where the second form follows because the union counts the common positions once, $|A \\cup B| = a + b - c$. It is 1 when the two sets are the same and 0 when they have nothing in common. For count fingerprints with entries $x_i$ and $y_i$ the same idea gives the min/max form
$T = \\sum_i \\min(x_i, y_i) \\,/\\, \\sum_i \\max(x_i, y_i)$, which reduces to the first form when every entry is 0 or 1. The distance $1 - T$ obeys the triangle inequality (Lipkus, 1999), so it is a true distance, which is what clustering (section 3.6) and nearest-neighbour methods (section 4.5) need.

**Why not simply count the positions where the fingerprints agree?** Because almost all positions are 0 in both. Take two molecules with 46 positions on each and none in common: they disagree on 92 of the 2048 positions and agree on the other 1956, an agreement of 95 %, although the molecules share nothing. The absence of a piece is almost never informative: no one would call two
molecules similar because both lack a piece out of two thousand. The Tanimoto similarity leaves the shared zeros out of both numerator and denominator, which is the whole point. The cell computes it by hand for the example and its closest training relative, and compares it with the agreement of all positions on 4000 random pairs.
"""),
    code(TANIMOTO),
    md("""
The Tanimoto similarity of random pairs of compounds covers almost the whole range, with a median of {{=tanimoto_median:.2f}}, while the fraction of agreeing positions is at least {{=compare_rows[1]['smallest']:.2f}} for every pair: it cannot tell any pair of molecules from any other. The example compound's closest relative in the training set is {{=nearest_compound}}, at a similarity of {{=by_hand:.2f}}.
The meaning of a value is empirical and not a law: a Tanimoto similarity of 0.5 on Morgan fingerprints of radius 2 usually marks a close relative of the same series (the figure shows the example with its closest relative among the compounds that have both arms and with a compound of median similarity), and the project uses 0.5 and 0.6 as practical cut-offs (sections 2.7 and 3.6). Similarity of structure is not similarity of behaviour: two close relatives can differ in the potency by orders of magnitude, a limit of every method that
works with neighbours (section 7.3).
"""),
    figure("mol_tanimoto"),
    md("""
## 3.3 Descriptors: a molecule as a few dozen physical quantities

> **Field:** Cheminformatics · Chemistry

A fingerprint says which pieces are there; a **descriptor** says how big, how greasy or how polar the whole molecule is. The two kinds are complementary: a fingerprint cannot add up the contributions of the many pieces, and a descriptor cannot say where they are. The project uses all {{=len(Descriptors.descList)}} descriptors of RDKit as one block (`rdkit_desc`), with non-finite values replaced by 0 and all values clipped to $\\pm 10^6$. The best known are:

- **Molecular weight**, the sum of the atomic masses in g/mol: the size of the molecule.
- **logP**, the logarithm of the **partition coefficient** between oil and water. If a compound is shaken with equal volumes of 1-octanol (a stand-in for the oily interior of a membrane or of a protein pocket) and water until it has distributed itself, $P = [X]_{\\text{octanol}} / [X]_{\\text{water}}$ is the ratio of its concentrations in the two layers, and $\\log P = \\log_{10} P$; a positive logP means that the compound prefers oil. It cannot be measured for a compound that does not exist
  yet, so it is estimated. The Wildman-Crippen method (1999) assigns every atom a type from the atom, its aromaticity and its neighbours (several dozen types, found by matching patterns) and adds up the contribution $a_t$ that was fitted by regression to measured values: $\\log P = \\sum_{i} a_{t(i)}$ over all atoms $i$, including hydrogens.
- **Topological polar surface area** (TPSA, Ertl, Rohde and Selzer, 2000) in square angstrom: the surface that nitrogen and oxygen atoms (with their hydrogens) would present, again a sum of fragment contributions, $\\mathrm{TPSA} = \\sum_{f} s_f$, and again read from the graph and not from a 3D structure. It is a measure of how polar a molecule is, and polar molecules cross membranes less readily.
- **Hydrogen-bond donors and acceptors** (the counts of N-H and O-H groups, and of N and O atoms, that can give or accept a hydrogen bond, the definitions of Lipinski), **rotatable bonds** (single bonds between non-terminal heavy atoms that are not in a ring: the flexibility), the number of rings and the **fraction of carbons that are sp3** (carbons with four single bonds, the "non-flat" ones).

Lipinski and co-workers (1997) noted that most orally active drugs have a molecular weight of at most 500, a logP of at most 5, at most 5 donors and at most 10 acceptors; such numbers describe the space that drug-like molecules occupy. The cell computes these for the example compound and rebuilds two of them from their atoms to show that the descriptors are sums: the atom contributions to logP and TPSA must add up to the descriptors.
"""),
    code(DESCRIPTORS),
    md("""
The contributions of the atoms add up to the descriptors exactly (the cell asserts it): the table groups the contributions to logP by element, and the polar surface of {{=tpsa_sum:.1f}} square angstrom comes from the nitrogens and oxygens alone, the only atoms that the version of the method used here counts. The same table for the whole data: do the test compounds differ from the training compounds in these simple quantities?
"""),
    code(DESCRIPTOR_SHIFT),
    md("""
The median test compound has a molecular weight of {{=mw_test:.0f}} against {{=mw_train:.0f}} for the training compounds, and the other rows give the full comparison. A shift in such plain quantities is another face of the covariate shift of section 2.7; whether it matters for a model depends on whether the model leans on these columns, which is what the validation of Part VI measures.

## 3.4 The other fingerprint and descriptor blocks

> **Field:** Cheminformatics

The Morgan fingerprint is one of many ways to ask structural questions. The feature store holds several more, because each makes different questions possible and no one of them is known in advance to be the best for this problem.

- **Atom pairs** (Carhart, Smith and Venkataraghavan, 1985): for every pair of atoms, the types of the two atoms and the number of bonds on the shortest path between them, hashed and counted (2048 positions). It encodes where things are relative to each other without any 3D geometry.
- **Topological torsions** (Nilakantan, Bauman, Dixon and Venkataraghavan, 1987): every path of four bonded atoms, with the atom types and the numbers of attached heavy atoms, hashed and counted (1024 positions).
- **Feature-Morgan** (FCFP, Rogers and Hahn, 2010): the Morgan algorithm, but the starting identifier of an atom is not its element but its pharmacophoric kind (hydrogen-bond donor, acceptor, positive or negative ionisable, aromatic, halogen), so that two molecules with different atoms in the same roles look alike (1024 counts).
- **MACCS keys** (Durant, Leland, Henry and Nourse, 2002): 166 hand-written structural questions, such as "is there an N-O bond?" or "are there at least two rings?", defined by MDL for database searching. The table has 167 columns because the first one is unused.
- **Avalon fingerprint** (Gedeck, Rohde and Bartels, 2006): hashed fragments of paths and atom-centred features enumerated by a different procedure (1024 bits).
- **Extended reduced graphs, ErG** (Stiefl, Watson, Baumann and Zaliani, 2006): the molecule is first collapsed into a graph of pharmacophore-like nodes (a ring becomes one node of its kind, a functional group another), and the fingerprint is the number of node pairs of each type at each topological distance (315 real values). It is deliberately coarse, so that molecules built on different skeletons but with the same pattern of roles look alike.
- **The project's own blocks**: the six heme-binding nitrogen motifs of Part V, 30 structural alerts and two ring features (section 1.10), 15 *alert-context* columns that weight an alert by the lipophilicity and size of the molecule, the presence of fluorine near the alert and the number of labile groups (medicinal chemists slow bioactivation with just such changes, Leach and co-workers, 2014), and one column for a trialkylamine (section 5.2).
- **Blocks that are read from a cache** instead of being computed on demand: five probabilities from models of external data (section 2.8), 64 principal components of a pretrained embedding (section 9.4) and the two physics blocks (Part X). Their computation takes minutes to hours and runs in its own processes, so the store only reads them.

The cell lists every block with its real width in the stored table and shows how many columns are non-zero for the example compound.
"""),
    code(BLOCKS),
    md("""
The store holds {{=n_blocks}} blocks and {{=n_columns_total:,}} columns in all; {{=n_opt_in}} of the blocks are opt-in, which means that they are read from caches and are not part of a default build. The blocks overlap in what they say: Morgan bits and Morgan counts hold the same environments, and the radius-3 block contains the radius-2 one.
A model that uses many of them is not given independent facts but several descriptions of the same compound, and a tree-based model (Part IV) copes with that by simply preferring whichever columns split the data best. The figure of section 3.7 shows the sizes of the blocks side by side.
"""),
]
