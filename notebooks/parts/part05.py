"""Part V (first half): the Type II motif screen. The idea and the patterns that encode it (5.1), the trialkylamine alert (5.2), the panel data (5.3), the test against potency (5.4)."""

from notebooks.notebook_builder import code, md

PATTERNS = '''
# The six patterns of the motif screen, read back from the project's own table (cypmode/metrics/motifs.py): what each pattern is made of, and on how many compounds of the challenge it matches.
import pandas as pd
from rdkit import Chem

from cypmode.metrics.motifs import TYPE_II_MOTIFS, detect_motifs

train_file = pd.read_csv(ROOT / "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv")
test_file = pd.read_csv(ROOT / "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv")
challenge = pd.concat([train_file[["Molecule_Name", "SMILES"]], test_file], ignore_index=True)
import time

started = time.perf_counter()
challenge["motifs"] = [detect_motifs(s) for s in challenge["SMILES"]]
seconds_for_screen = time.perf_counter() - started
pattern_rows = []
for name, smarts in TYPE_II_MOTIFS.items():
    pattern = Chem.MolFromSmarts(smarts)
    atoms = " ".join(a.GetSmarts().replace("[n&H1]", "[nH]") for a in pattern.GetAtoms())
    pattern_rows.append({"pattern": name, "SMARTS": smarts, "atoms of the pattern": atoms, "atoms": pattern.GetNumAtoms(), "bonds": pattern.GetNumBonds(),
                         "compounds that match": int(sum(name in m for m in challenge["motifs"]))})
show_table(pd.DataFrame(pattern_rows))
n_challenge, n_with_motif = len(challenge), int(challenge["motifs"].apply(bool).sum())
n_train_file, n_test_file = len(train_file), len(test_file)
share_with_motif = n_with_motif / n_challenge
'''

PROBES = '''
# Probing the patterns with small molecules whose chemistry is known. For every probe the screen's call, and the nitrogen atoms that the matched patterns contain, split by what kind of nitrogen they are:
# a "pyridine-type" aromatic nitrogen has two neighbours and no hydrogen, so its lone pair lies in the plane of the ring and is free; a "pyrrole-type" one has a third neighbour (a hydrogen or a substituent)
# and its lone pair is part of the ring's shared electrons.
probes = {"imidazole": "c1cnc[nH]1", "1-methylimidazole": "Cn1ccnc1", "1,2,4-triazole": "c1ncn[nH]1", "pyridine": "c1ccncc1", "thiazole": "c1cscn1", "2-pyridone": "O=c1cccc[nH]1",
          "pyrrole": "c1cc[nH]c1", "benzene": "c1ccccc1", "aniline": "Nc1ccccc1"}

def matched_nitrogens(smiles):
    """Indices of the nitrogen atoms that belong to at least one match of the six patterns."""
    mol = Chem.MolFromSmiles(smiles)
    found = set()
    for smarts in TYPE_II_MOTIFS.values():
        for match in mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)):
            found |= {i for i in match if mol.GetAtomWithIdx(i).GetSymbol() == "N"}
    return mol, sorted(found)

def free_lone_pair(atom):
    """An aromatic nitrogen with two heavy neighbours and no hydrogen: the lone pair is in the plane of the ring and available."""
    return atom.GetIsAromatic() and atom.GetDegree() == 2 and atom.GetTotalNumHs() == 0

probe_rows = []
for label, smiles in probes.items():
    mol, nitrogens = matched_nitrogens(smiles)
    free = sum(free_lone_pair(mol.GetAtomWithIdx(i)) for i in nitrogens)
    probe_rows.append({"molecule": label, "SMILES": smiles, "patterns that match": ", ".join(detect_motifs(smiles)) or "none", "matched nitrogens": len(nitrogens),
                       "of them with a free lone pair": free})
show_table(pd.DataFrame(probe_rows))
pyridone_calls = next(r for r in probe_rows if r["molecule"] == "2-pyridone")
pyrrole_calls = next(r for r in probe_rows if r["molecule"] == "pyrrole")
'''

POPULATION = '''
# The same question over the whole challenge: of the compounds that the screen flags, how many are flagged only through nitrogens that have no free lone pair?
rows_flagged = challenge[challenge["motifs"].apply(bool)]
free_counts = []
for smiles in rows_flagged["SMILES"]:
    mol, nitrogens = matched_nitrogens(smiles)
    free_counts.append(sum(free_lone_pair(mol.GetAtomWithIdx(i)) for i in nitrogens))
free_counts = np.array(free_counts)
n_no_free = int((free_counts == 0).sum())
population_rows = [{"compounds flagged by the screen": len(rows_flagged), "with at least one matched nitrogen that has a free lone pair": int((free_counts > 0).sum()),
                    "flagged only through nitrogens without one": n_no_free, "share": n_no_free / len(rows_flagged)}]
show_table(pd.DataFrame(population_rows))
share_no_free = n_no_free / len(rows_flagged)
'''

STAGES = r'''
# The trialkylamine pattern of the project (cypmode/metrics/amines.py) built up one exclusion at a time. Stage 0 is "a nitrogen with three connections and no hydrogen" (tertiary; for the secondary
# amines the same with one hydrogen); each later stage adds one exclusion, in the order they stand in the final pattern. A probe is a small molecule of one chemical type.
import numpy as np
from cypmode.challenge.tdi_targets import training_view
from cypmode.metrics.amines import has_trialkylamine
from cypmode.validation.tdi_reference import REFERENCE_SMILES

labelled = training_view(train_file, "CYP3A4").query("population == 'both_arms'")        # the compounds with both arms measured and a label that can be trusted (section 2.3)
n_labelled = len(labelled)

EXCLUSIONS = [("any nitrogen with three connections and no hydrogen (tertiary)", ""), ("not an amide or thioamide", "!$(NC=[O,S])"), ("no double bond on the nitrogen", "!$(N=*)"),
              ("not aromatic and no aromatic neighbour", "!a;!$(N-a)"), ("not a sulfonamide or phosphoramide", "!$(N-[S,P]=[O,S,N])"), ("not an enamine", "!$(N-C=C)"), ("not an amidine or guanidine", "!$(N-C=N)")]

def stage_patterns(stage):
    pieces = "".join(";" + e for _, e in EXCLUSIONS[1:stage + 1] if e)
    return [Chem.MolFromSmarts(f"[NX3;H{h}{pieces}]") for h in (0, 1)]

def flagged_at(mol, stage):
    return any(mol.HasSubstructMatch(p) for p in stage_patterns(stage))

amine_probes = {"triethylamine": "CCN(CC)CC", "N-methylpiperidine": "CN1CCCCC1", "diethylamine (secondary)": "CCNCC", "N,N-dimethylaniline": "CN(C)c1ccccc1", "N,N-dimethylacetamide": "CC(=O)N(C)C",
                "N,N-dimethylmethanesulfonamide": "CS(=O)(=O)N(C)C", "N,N-dimethylvinylamine (an enamine)": "C=CN(C)C", "N,N-dimethylacetamidine": "CC(=N)N(C)C", "tetramethylguanidine": "CN=C(N(C)C)N(C)C",
                "1-acetyl-4-phenylpiperazine": "CC(=O)N1CCN(c2ccccc2)CC1", "1-methyl-4-phenylpiperazine": "CN1CCN(c2ccccc2)CC1"}
probe_rows = []
for label, smiles in amine_probes.items():
    mol = Chem.MolFromSmiles(smiles)
    probe_rows.append({"probe": label, **{f"stage {k}": ("flagged" if flagged_at(mol, k) else "-") for k in range(len(EXCLUSIONS))}, "final pattern of the project": "flagged" if has_trialkylamine(smiles) else "-"})
show_table(pd.DataFrame(probe_rows))
final_agrees = all((r["stage 6"] == "flagged") == (r["final pattern of the project"] == "flagged") for r in probe_rows)
'''

STAGE_COUNTS = r'''
# The same stages on the real collection: how many of the compounds each stage flags, and how often the flagged compounds that were labelled turned out to be time-dependent inhibitors of CYP3A4
# (the compounds with both arms measured, section 2.3). The lift of a stage is that share divided by the share among all labelled compounds.
y_label = labelled["label"].astype(int)
base_share = float(y_label.mean())
challenge_mols = [Chem.MolFromSmiles(s) for s in challenge["SMILES"]]
stage_rows = []
for stage, (description, _) in enumerate(EXCLUSIONS):
    flags = np.array([flagged_at(m, stage) for m in challenge_mols])
    labelled_flags = flags[: len(train_file)][labelled.index]
    share = float(y_label[labelled_flags].mean())
    stage_rows.append({"stage": stage, "rule added": description, "compounds flagged": int(flags.sum()), "share of the collection": float(flags.mean()), "labelled and flagged": int(labelled_flags.sum()),
                       "share that are TDI": share, "standard error of the share": float(np.sqrt(share * (1 - share) / labelled_flags.sum())), "lift": share / base_share})
show_table(pd.DataFrame(stage_rows))
stage_table = pd.DataFrame(stage_rows)
first_stage, last_stage = stage_table.iloc[0], stage_table.iloc[-1]
'''

REFERENCE_DRUGS = r'''
# The six drugs of the reference panel (section 5.7) at the first and at the final stage: ketoconazole, a reversible inhibitor with no time dependence, is flagged by the first pattern only because of the nitrogen
# that carries a phenyl ring.
reference_rows = []
for name, smiles in REFERENCE_SMILES.items():
    mol = Chem.MolFromSmiles(smiles)
    reference_rows.append({"drug": name, "stage 0": "flagged" if flagged_at(mol, 0) else "-", "stage 3 (aromatic nitrogens excluded)": "flagged" if flagged_at(mol, 3) else "-",
                           "final pattern": "flagged" if has_trialkylamine(smiles) else "-"})
show_table(pd.DataFrame(reference_rows))
ketoconazole_row = next(r for r in reference_rows if r["drug"] == "ketoconazole")
flagged_final = [r["drug"] for r in reference_rows if r["final pattern"] == "flagged"]
'''

CELLS = [
    md("""
# Part V. The Type II motif screen

Parts I to IV gave the chemistry, the data, the numbers that describe a molecule and the machine learning that turns such numbers into a prediction. Before the project measured anything with the careful procedures of Part VI, it began with something much smaller
and older than a learned model: a rule written by a chemist, applied by a program. This part tells that first piece of work in full, for three reasons. It is the origin of two of the feature blocks that every later model receives (the motif block and the amine alert, section 3.4). It
is the only part of the project that needs no training at all, so every claim about it can be tested completely and the logic of the test can be seen without the machinery of cross-validation. And it introduces, on a small scale, the habits of the whole project: a claim is tested on
data that did not shape it, the size of the effect is reported next to its probability, and the text says what the test cannot show.

The part answers one question, from five directions. *Does a molecule that carries a nitrogen able to bind the heme iron behave differently from one that does not?* Section 5.1 turns the chemistry of section 1.9 into a pattern that a program can search for. Section 5.2 builds the second rule, for
the amines that inactivate the enzyme by a different route (section 1.10). Section 5.3 asks how many of the known inhibitors of five enzymes carry the pattern. Section 5.4 asks whether the pattern goes with potency, on a screen of more than a thousand compounds that the rule never saw. Section 5.5 asks
whether a structure predicted by Boltz-2 puts the nitrogen where the chemistry says it should be, and section 5.6 adds two measurements on the same structures, how deeply the molecule is buried and how large its pocket is. Section 5.7 compares the rule with six drugs whose time-dependent inhibition
was measured, and section 5.8 describes the small web application in which the original screen can be used.

What this part hands on: the patterns of 5.1 and 5.2 reappear as feature blocks in Part VII (the early ablation of the motif and amine features, 7.2); the distance from the iron to the nitrogen of 5.5 is the quantity that Part XI measures for every pose; the two pocket measurements of 5.6 are the structural
evidence of section 7.10. A reader who wants only the final models can read 5.1 and 5.2, which say what the two blocks mean, and return to the rest when a later part refers to it.
"""),
    md("""
## 5.1 The idea: a nitrogen that can bind the iron, written as a pattern

> **Field:** Chemistry · Cheminformatics

*Where this section sits:* section 1.9 explained, in words, why an aromatic nitrogen can take the place of the water on the heme iron. This section turns that sentence into something a program can apply to a million molecules in seconds. The next section adds a second rule, and section 3.4
shows how both became columns of the feature store.

**A screen** is a fast rule that looks at every molecule of a collection and flags those worth a closer look; it is cheap because it asks only a yes or no question of the structure of the molecule, and it does not try to predict how strongly anything binds. The question of this screen is: does the molecule contain one
of six small groups of atoms, each of which has a nitrogen of the kind that binds the heme iron? The groups are those that textbooks of drug metabolism name as the classical Type II binders (de Groot, 2006; Correia and Ortiz de Montellano, 2005): the **imidazole** (a flat ring of five atoms, three of them
carbon and two nitrogen, the two nitrogens separated by one carbon), the **1,2,4-triazole** and the **1,2,3-triazole** (rings of five atoms with three nitrogens, arranged differently), the **pyridine** (a flat ring of six atoms with one nitrogen) and the **thiazole** (five atoms with one sulfur and one nitrogen).
Ketoconazole carries an imidazole, fluconazole and itraconazole a triazole, and ritonavir a thiazole; these antifungal and antiviral drugs are classical Type II inhibitors of CYP3A4.

**How a group of atoms is written.** The notation of section 2.1 for a molecule, SMILES, has a sister notation for a *pattern*, **SMARTS** (section 0.2). Both describe a graph, but a SMILES string describes the one graph of one molecule, and a SMARTS string describes a small graph whose atoms and bonds may be described loosely,
so that the pattern fits many molecules. The letters are those of SMILES: a lower-case `c` is an aromatic carbon and `n` an aromatic nitrogen; an upper-case `N` is a nitrogen that is not aromatic; a square bracket gives the atom a fuller description, `[nH]` being an aromatic nitrogen that carries one hydrogen. A digit marks a
ring: the atom that carries it and the later atom that carries the same digit are joined by a bond, which closes the ring. So `c1cnc[nH]1` reads: an aromatic carbon (which opens ring 1), another carbon, a nitrogen, a carbon, and a nitrogen carrying a hydrogen that closes ring 1; five atoms in a ring, two of them nitrogen with one
carbon between them, which is the imidazole. The pattern `c1cncn1` is the same ring without the demand that either nitrogen carry a hydrogen: its `n` matches an aromatic nitrogen whatever else is attached, so it fits the imidazoles whose nitrogen carries a substituent as well as those that carry a hydrogen. The first cell prints
all six patterns, built into small graphs by the toolkit, and counts how many compounds of the two challenge files (the training file and the blinded test file, section 2.2) each one matches.
"""),
    code(PATTERNS),
    md("""
**What "matches" means.** Let $x$ be a molecule, seen as a graph whose nodes are its atoms and whose edges are its bonds (section 2.1), and $p$ a pattern, a smaller graph of the same kind. The pattern $p$ **matches** $x$ when its atoms can be laid on a choice of atoms of $x$, no two on the
same atom, so that every atom of $p$ lies on an atom of the kind it describes and every bond of $p$ lies on a bond of $x$ of the kind it describes. (The name in graph theory is *subgraph isomorphism*; the toolkit searches for such a laying-on by trying atom after atom and abandoning a try as soon as a bond fails, which for patterns of five or six atoms is fast: reading the {{=n_challenge:,}} structures and applying the six patterns to each took {{=seconds_for_screen:.1f}} seconds in the cell above.)
The screen's call for a molecule $x$ is a single number $m(x)$, with $p_1, \\dots, p_6$ the six patterns and the index $j$ running over them:

$$m(x) \\;=\\; \\max_{j=1,\\dots,6}\\; \\mathbb{1}\\bigl[\\, p_j \\text{ matches } x \\,\\bigr],$$

where the symbol $\\mathbb{1}[\\cdot]$, the **indicator**, is 1 when the statement inside the brackets is true and 0 when it is false. In words: $m(x)=1$ when at least one of the six patterns matches. Of the {{=n_challenge:,}} compounds of the challenge, {{=n_with_motif:,}}
({{=share_with_motif:.0%}}) are flagged. That is a large share, and it is the first thing to understand about this screen: the library of the challenge is full of nitrogen-containing aromatic rings, because such rings are common in drugs and in the molecules built for drug discovery, so a flag by itself does not
distinguish most compounds from the rest. The flag is useful only to the extent that it goes with something that matters, and the following sections measure whether it does.

**What the patterns cannot see.** A pattern describes which atoms are connected to which; it knows nothing about the electrons. A nitrogen can bind the iron only if its lone pair is free. In an aromatic ring of five atoms such as the imidazole, one nitrogen puts its lone pair into the electrons shared round the ring (the pyrrole-type nitrogen, which carries a hydrogen or a substituent as its third neighbour) and so cannot use
it; the other nitrogen (the pyridine-type nitrogen, with two neighbours and no hydrogen) keeps its lone pair in the plane of the ring, free to bind, and that is the one that goes to the iron. The pattern `c1cncn1` matches both nitrogens without distinction. The next cell makes the point with small molecules whose chemistry is known, and counts, for each probe,
how many of the matched nitrogens have a free lone pair (the definition is two neighbours and no hydrogen, which is chemically a simplification, but a useful one).
"""),
    code(PROBES),
    md("""
The imidazole and 1-methylimidazole each have a nitrogen with a free lone pair and one without, as the chemistry says, and the pattern matches both. Pyridine, thiazole and the triazole each match with at least one free nitrogen. Two probes show the weakness. The pyridine pattern matches 2-pyridone ({{=pyridone_calls['matched nitrogens']}} matched nitrogen,
{{=pyridone_calls['of them with a free lone pair']}} of them with a free lone pair) because the toolkit, like the chemist, treats the ring as aromatic, but the nitrogen of a pyridone carries a hydrogen and uses its lone pair in the ring; and pyrrole, which has one nitrogen of the same kind, is not matched at all only because every pattern for a five-atom ring demands a second heteroatom (a second nitrogen or a sulfur). So a match is a clue
and not a proof. The cell below asks how often it matters on the real collection: among the compounds that the screen flags, how many are flagged only through nitrogens that have no free lone pair.
"""),
    code(POPULATION),
    md("""
Of the {{=population_rows[0]['compounds flagged by the screen']:,}} flagged compounds, {{=n_no_free}} ({{=share_no_free:.1%}}) are flagged only through nitrogens without a free lone pair. These are the compounds for which the flag is most likely to be wrong chemically; the screen as it was written and used does not remove them, and the
section on the structures (5.5) is where such cases show themselves, because a predicted structure places the nitrogen at a measured distance from the iron and a nitrogen that cannot bind does not go there.

*On what actually ran.* The screen of the project (`cypmode/metrics/motifs.py`) is the six patterns and the function that reports which of them match: `detect_motifs` returns the names, and `classify_mechanism` turns a non-empty list into the sentence "Type II (heme-coordinating motif present)" and an empty list into "No coordinating motif detected (Type I or non-classical)".
It does not test the lone pair, as the probes above show; it does not know where in the pocket the molecule will sit; and the sentence it prints is a statement about a *capability*. The same six patterns are used unchanged by the original panel analysis (5.3), by the potency check (5.4), by the structural check (5.5) and by the early feature blocks of Part VII.
"""),
    md("""
## 5.2 The trialkylamine alert, and the false positives it had to exclude

> **Field:** Chemistry · Cheminformatics

*Where this section sits:* the screen of 5.1 looks for molecules that bind the iron themselves. Section 1.10 described a second way in which a molecule can switch an enzyme off, with no group that binds the iron to begin with: the enzyme itself turns an amine into something that does. This section
builds the pattern for that route. It was written second, as a separate screen (`cypmode/metrics/amines.py`), and it is a good case study, because the first version of the pattern flagged almost every compound of the collection and the work consisted in finding out, with chemistry and with data, what it had to leave out.

**The chemistry.** A nitrogen with three single bonds and a lone pair is an **amine**: the three bonds point to three corners of a pyramid and the lone pair to the fourth. Call a carbon **alkyl** when its bonds are all single and it is not part of an aromatic ring. A **trialkylamine**, in the sense used here, is an amine whose nitrogen is joined to alkyl
carbons and, at most, to one hydrogen; the name is a little loose, since it covers the secondary amines (one hydrogen) and the tertiary ones (none). Such a nitrogen is a classical target of the enzyme (section 1.10). The oxidising iron-oxygen species of the catalytic cycle removes an electron from the lone pair, or a hydrogen from a carbon next to the
nitrogen. The product splits into a smaller amine and an aldehyde (**N-dealkylation**), or the nitrogen itself is oxidised, in steps, to a nitroso compound whose nitrogen binds the iron of the heme and stays there, a metabolic-intermediate complex. Whether the enzyme gets that far depends on one thing at the start: how freely available the nitrogen's lone pair is.

The lone pair is less available the more it is shared with a neighbouring group. Five neighbours do it. A nitrogen bonded to an aromatic ring (an *aniline*, written $N$–aryl) shares the pair with the ring's electrons. In an amide (the nitrogen next to a carbon double-bonded to oxygen) the pair is spread over nitrogen, carbon and oxygen, which is why amides are planar and almost
unreactive. A sulfonamide (nitrogen next to a sulfur double-bonded to oxygen) does the same through the sulfur. In an enamine (nitrogen next to a carbon-carbon double bond) the pair runs into the double bond, and in an amidine or a guanidine (nitrogen next to a carbon double-bonded to another nitrogen) it runs towards that nitrogen. The measure of the sharing that every chemistry
textbook gives is basicity: the acid of triethylamine has a pKa of about 10.7, that of aniline about 4.6 and that of acetamide about -0.5 (Clayden and co-workers, 2012); the less basic the nitrogen, the less its pair is available, for protonation or for oxidation. A pattern for the amine route must therefore keep the nitrogens of the first kind and leave out those of the five others.

**Writing the rule.** The pattern language of 5.1 needs a few more symbols. `N` is an aliphatic (not aromatic) nitrogen, `X3` says that the atom has three connections in all (hydrogens counted), `H0` and `H1` say that it carries no hydrogen or one; `;` means *and*, `!` means *not*, and `,` inside a bracket means *or*. A **recursive** description `$(...)` stands for "an atom at which the pattern inside the
parentheses matches": `!$(NC=[O,S])` says that the nitrogen is not the nitrogen of a group in which it is bonded to a carbon that is double-bonded to oxygen or sulfur, which is the amide and the thioamide. The final pattern of the project is a nitrogen with three connections and no hydrogen (or one) with all of these exclusions at once. The cell builds it in stages, each stage adding one exclusion, and applies
every stage to eleven small molecules, each of one chemical type, so that the reason for every exclusion can be read off the table.
"""),
    code(STAGES),
    md("""
Stage 0 flags every probe that has a nitrogen with three connections, the amide, the aniline and the guanidine as readily as the amine. Each later stage removes a type: the amide at stage 1, the aniline and the nitrogens of aromatic rings at stage 3, the sulfonamide at stage 4, the enamine at stage 5 and the amidine and guanidine at stage 6; and the two piperazines show the
logic on a molecule with two nitrogens. The piperazine of ketoconazole has an acetyl nitrogen and a phenyl nitrogen, neither of which is an amine in the sense of the screen (1-acetyl-4-phenylpiperazine in the table); with a methyl on the second nitrogen, as in the last probe, the molecule keeps a true trialkylamine and stays flagged to the end. The last column
is the function of the project itself (`has_trialkylamine`); it agrees with stage 6 on all eleven probes: {{=final_agrees}}.

**What the exclusions do on the real collection.** The probes are chemistry; the collection is data. The next cell applies every stage to the {{=n_challenge:,}} compounds of the challenge and asks, for the compounds that were labelled (those with both arms measured, {{=n_labelled:,}} of them, section 2.3), how often a flagged compound turned out to be a time-dependent inhibitor of CYP3A4.
Two numbers describe the usefulness of an alert. Let $a$ stand for the statement "the compound is flagged" and $T$ for "the compound is a time-dependent inhibitor". The probability $P(T)$ is the share of time-dependent inhibitors among all the labelled compounds, and $P(T \\mid a)$, read "the probability of $T$ given $a$", is the share among the flagged ones: it is
the **precision** of the alert in the sense of section 4.10. Their ratio is the **lift**, $L$,

$$L \\;=\\; \\frac{P(T \\mid a)}{P(T)},$$

which is 1 for an alert that tells nothing (the flagged compounds are no more often inhibitors than the rest) and larger the more the alert concentrates the inhibitors.
"""),
    code(STAGE_COUNTS),
    md("""
Without any exclusion, stage 0 flags {{=first_stage['compounds flagged']:,}} compounds, {{=first_stage['share of the collection']:.0%}} of the collection, and the lift is {{=first_stage['lift']:.2f}}: the alert says nothing, as it must when it is applied to nearly everything. With all the exclusions the alert flags
{{=last_stage['compounds flagged']:,}} compounds ({{=last_stage['share of the collection']:.0%}}), the share of time-dependent inhibitors among the {{=last_stage['labelled and flagged']}} labelled ones that are flagged is {{=last_stage['share that are TDI']:.3f}} against {{=base_share:.3f}} for all labelled compounds, and the lift is {{=last_stage['lift']:.2f}}. Overall, then, the
exclusions remove compounds whose nitrogens cannot be oxidised in the way that matters, and what is left is a little richer in inhibitors. A lift of {{=last_stage['lift']:.2f}} means that a flagged compound is a time-dependent inhibitor about {{=(last_stage['lift'] - 1) * 100:.0f}} percent more often than an average one; it does not make the alert a classifier.

The rise is not steady, and the way to read the dip is a lesson of its own. At stage 3, where the nitrogens that carry or touch an aromatic ring are excluded, the share falls from {{=stage_table.loc[2, 'share that are TDI']:.3f}} to {{=stage_table.loc[3, 'share that are TDI']:.3f}}. A share estimated from $n$ compounds, of which a fraction $p$ are inhibitors, has a **standard error**
(the spread of such an estimate if the same experiment were repeated on other compounds of the same kind) of

$$\\mathrm{SE} \\;=\\; \\sqrt{\\frac{p\\,(1-p)}{n}},$$

because one compound is a yes-or-no draw with variance $p(1-p)$ and the average of $n$ independent draws has $n$ times less variance. For the {{=stage_table.loc[3, 'labelled and flagged']}} labelled compounds of stage 3 it is {{=stage_table.loc[3, 'standard error of the share']:.3f}}, which is as large as the dip: the data neither confirm nor contradict this exclusion. It stands on the chemistry, since the nitrogen of an aniline shares its pair with the ring, and on the ketoconazole case below, and the
table should not be read as proof of it. The same caution applies to the whole column: whether the difference between {{=last_stage['share that are TDI']:.3f}} and the {{=base_share:.3f}} of all labelled compounds is distinguishable from chance needs the statistics that the next two sections build from the beginning (sections 5.3 and 5.4); here the point is the direction and the way the numbers move.

The project checked the pattern, before using it, on a smaller and sharper test: six named drugs whose time-dependent inhibition of CYP3A4 had been measured (section 5.7). The cell lists them at the first and at the final stage.
"""),
    code(REFERENCE_DRUGS),
    md("""
Ketoconazole, a reversible Type II inhibitor that shows no time-dependent shift (5.7), is flagged by the first patterns through the nitrogen that carries a phenyl ring and not by the final one ({{=ketoconazole_row['final pattern']}}). The drugs that the final pattern does flag are {{=', '.join(flagged_final)}}, the three of the panel that carry a
true trialkylamine and that are time-dependent inhibitors; the other two, azamulin and clotrimazole, have no amine, and each is a case of a different mechanism, which section 5.7 takes up. Six drugs are far too few to measure how good a pattern is, and the project does not use them that way: they are a sanity check that the refined pattern does what the chemistry says on molecules whose behaviour is known, and the collection of
the preceding cell is the quantitative evidence.

*On what actually ran.* `has_trialkylamine` in `cypmode/metrics/amines.py` applies the final pattern (the two patterns for the secondary and the tertiary amine) and raises an error for a string that is empty or cannot be read; the module's docstring records the history that the cell above reproduces. The amine flag was one of the two flags that Part VII (section 7.2) put into the
first models of the challenge; that section reports what became of it. A reader should not take the alert as a prediction that a compound inactivates the enzyme: it records that the compound carries a nitrogen of the kind that can be oxidised to a species that binds the iron, no more.
"""),
]
