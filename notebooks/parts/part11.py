# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part XI (first part): the structure track, what a structure is (11.1), how a co-folding model predicts one (11.2) and how its input is built (11.3)."""

from notebooks.notebook_builder import code, md

STRUCTURE_FILE = '''
# What a predicted complex file holds. The file is one of the 20 structures of the final submission (the ligand OCNT-2312554 in CYP3A4), in the layout the challenge asks for: the protein as chain A, the ligand as the residue
# LIG on chain B, the heme as the residue HEM on chain C. The first four lines are shown as they are in the file, then the file is read with Biopython and counted. The last number in each line of coordinates before the element is the
# "B-factor" column, which in a crystal file records how much an atom is smeared; the co-folding model writes its own per-atom confidence into it (a number from 0 to 100).
from Bio.PDB import PDBParser

pose_path = ROOT / "data/structure_track/outputs/cyp3a4/final/_pdb_files/OCNT-2312554.pdb"
raw_lines = pose_path.read_text().splitlines()
print("\\n".join(raw_lines[:4]))
hetero_start = next(i for i, line in enumerate(raw_lines) if line.startswith("HETATM"))
print("...")
print("\\n".join(raw_lines[hetero_start:hetero_start + 2]))
pose = PDBParser(QUIET=True).get_structure("pose", str(pose_path))[0]
chain_rows = []
for chain in pose:
    residues = list(chain)
    atoms = [a for r in residues for a in r]
    chain_rows.append({"chain": chain.id, "residues": len(residues), "first residue": residues[0].get_resname(), "atoms": len(atoms), "mean confidence (B-factor column)": float(np.mean([a.get_bfactor() for a in atoms]))})
chain_table = pd.DataFrame(chain_rows)
show_table(chain_table, floatfmt=".1f")
iron = [a for r in pose["C"] for a in r if a.get_name() == "FE"][0]
cysteine = [r for r in pose["A"] if r.id[1] == 442][0]
sulfur = cysteine["SG"]
ligand_nitrogens = [a for r in pose["B"] for a in r if a.element == "N"]
iron_to_sulfur = float(np.linalg.norm(iron.coord - sulfur.coord))
nearest_nitrogen = min(ligand_nitrogens, key=lambda a: np.linalg.norm(iron.coord - a.coord))
iron_to_nitrogen = float(np.linalg.norm(iron.coord - nearest_nitrogen.coord))
n_protein_atoms, n_ligand_atoms, n_heme_atoms = (int(chain_table.loc[chain_table["chain"] == c, "atoms"].iloc[0]) for c in ("A", "B", "C"))
n_residues_protein = int(chain_table.loc[chain_table["chain"] == "A", "residues"].iloc[0])
cysteine_name = cysteine.get_resname()
'''

SEQUENCE_FACTS = '''
# The protein of the structure track: CYP3A4 (UniProt P08684), 503 residues, as it stands in the Boltz input of every ligand. The project finds the cysteine that binds the heme iron from the conserved signature
# F-x-x-G-x-x-x-C-x-G of all P450 enzymes (cypmode/validation/build_input.py, find_coordinating_cysteine): the cell checks that it finds one match and that the residue is the one the structure file names.
import re
import yaml

from cypmode.validation.build_input import CYS_MOTIF_PATTERN, find_coordinating_cysteine

input_yaml = yaml.safe_load((ROOT / "data/structure_track/inputs/cyp3a4_OCNT-2312554.yaml").read_text())
cyp3a4_sequence = input_yaml["sequences"][0]["protein"]["sequence"]
cysteine_position = find_coordinating_cysteine(cyp3a4_sequence)
signature = re.search(CYS_MOTIF_PATTERN, cyp3a4_sequence)
composition = pd.Series(list(cyp3a4_sequence)).value_counts()
sequence_length = len(cyp3a4_sequence)
signature_text = signature.group(0)
most_common = ", ".join(f"{a} {composition[a]}" for a in composition.index[:3])
'''

COEVOLUTION_TOY = '''
# What an alignment of related sequences can tell about the shape. A toy "protein" of 12 positions with 4 kinds of residue. 400 related sequences are drawn at random, except that position 9 copies position 2 in 85 percent of them
# (positions 3 and 10 counted from 1, 2 and 9 counted from 0; two residues that touch each other in the folded protein and must change together to keep fitting). The mutual information of two columns, I(i, j) = sum over pairs of residues f_ij log( f_ij / (f_i f_j) ), where f_i is the
# share of each residue at position i and f_ij the share of pairs at i and j, is zero when the two positions vary independently and large when one tells the other.
rng_msa = np.random.default_rng(3)
n_sequences, n_positions, n_letters = 400, 12, 4
alignment = rng_msa.integers(0, n_letters, size=(n_sequences, n_positions))
copied = rng_msa.random(n_sequences) < 0.85
alignment[:, 9] = np.where(copied, alignment[:, 2], alignment[:, 9])

def mutual_information_matrix(msa, letters):
    n, length = msa.shape
    single = np.stack([np.bincount(msa[:, i], minlength=letters) / n for i in range(length)])
    information = np.zeros((length, length))
    for i in range(length):
        for j in range(i + 1, length):
            joint = np.zeros((letters, letters))
            for a, b in zip(msa[:, i], msa[:, j]):
                joint[a, b] += 1
            joint /= n
            seen = joint > 0
            information[i, j] = information[j, i] = (joint[seen] * np.log(joint[seen] / np.outer(single[i], single[j])[seen])).sum()
    return information

information = mutual_information_matrix(alignment, n_letters)
pair_index = np.triu_indices(n_positions, 1)
ranked = np.argsort(-information[pair_index])[:4]
coevolution_table = pd.DataFrame({"position i": pair_index[0][ranked] + 1, "position j": pair_index[1][ranked] + 1, "mutual information (nats)": information[pair_index][ranked]})
show_table(coevolution_table, floatfmt=".3f")
median_information = float(np.median(information[pair_index]))
top_pair = (int(coevolution_table.iloc[0, 0]), int(coevolution_table.iloc[0, 1]))
'''

TRIANGLE_TOY = '''
# One triangle update of the pair representation, with random weights, to see what it computes. z[i, j] is a vector (a "channel" of 8 numbers) for the pair of positions i and j. The outgoing-edge update of AlphaFold 2
# (Jumper et al., 2021, algorithm 11) is  z_ij <- z_ij + g_ij * W_o LN( sum_k a_ik * b_jk ),  with a = gate * (W_a LN(z)) and b = gate * (W_b LN(z)), g a gate, LN a normalisation over the channels: the pair (i, j)
# collects, for every third position k, the two edges (i, k) and (j, k). The test: relabel the positions by a random permutation and compute again; an update that treats positions by what they are connected to, not by
# their number, gives the same result relabelled.
rng_tri = np.random.default_rng(3)
def layer_norm(z):
    return (z - z.mean(-1, keepdims=True)) / np.sqrt(z.var(-1, keepdims=True) + 1e-5)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def triangle_update_outgoing(z, weights):
    zn = layer_norm(z)
    a = (zn @ weights["a"]) * sigmoid(zn @ weights["a_gate"])
    b = (zn @ weights["b"]) * sigmoid(zn @ weights["b_gate"])
    aggregated = np.einsum("ikc,jkc->ijc", a, b)
    return z + sigmoid(zn @ weights["gate"]) * (layer_norm(aggregated) @ weights["out"])

n_nodes, n_channels = 7, 8
triangle_weights = {name: rng_tri.standard_normal((n_channels, n_channels)) * 0.3 for name in ("a", "b", "a_gate", "b_gate", "gate", "out")}
pair_state = rng_tri.standard_normal((n_nodes, n_nodes, n_channels))
relabel = rng_tri.permutation(n_nodes)
updated = triangle_update_outgoing(pair_state, triangle_weights)
updated_of_relabelled = triangle_update_outgoing(pair_state[np.ix_(relabel, relabel)], triangle_weights)
equivariance_error = float(np.abs(updated[np.ix_(relabel, relabel)] - updated_of_relabelled).max())
change_size = float(np.abs(updated - pair_state).mean() / np.abs(pair_state).mean())
print(f"largest difference after relabelling the positions: {equivariance_error:.1e}; mean size of the update relative to the state: {change_size:.2f}")
'''

DIFFUSION_TOY = '''
# A diffusion sampler in two dimensions where the ideal denoiser is known exactly, so that the sampler itself can be examined. The "data" are two clusters of points, standing for two different poses of a ligand, at (-3, 0)
# and (3, 0), each a Gaussian of standard deviation 0.3 in each direction. For data that are a mixture of Gaussians the best denoiser D(x, sigma) = E[x_0 | x_sigma], the expected clean point given a point with noise of size sigma,
# has a closed form (the clusters weighted by how well they explain x, each pulling x toward its centre by the share s^2 / (s^2 + sigma^2)). The sampler is that of the text: start from pure noise of size 20, go down 200 noise levels,
# at each level add some noise back (the "churn") and take a step toward the denoised point whose length is scaled by the step scale eta.
cluster_centres = np.array([[-3.0, 0.0], [3.0, 0.0]])
cluster_std = 0.3

def ideal_denoiser(x, sigma):
    variance = cluster_std**2 + sigma**2
    log_weight = np.stack([-((x - centre) ** 2).sum(-1) / (2 * variance) for centre in cluster_centres], -1)
    weight = np.exp(log_weight - log_weight.max(-1, keepdims=True))
    weight /= weight.sum(-1, keepdims=True)
    pulled = np.stack([centre + (cluster_std**2 / variance) * (x - centre) for centre in cluster_centres], 1)
    return (weight[..., None] * pulled).sum(1)

def sample_points(n, step_scale, steps=200, sigma_max=20.0, sigma_min=0.002, rho=7, churn=0.8, noise_scale=1.003, seed=0):
    generator = np.random.default_rng(seed)
    levels = (sigma_max ** (1 / rho) + np.arange(steps + 1) / steps * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    x = levels[0] * generator.standard_normal((n, 2))
    for i in range(steps):
        gamma = churn if levels[i + 1] > 1.0 else 0.0
        raised = levels[i] * (1 + gamma)
        noisy = x + noise_scale * np.sqrt(max(raised**2 - levels[i] ** 2, 0)) * generator.standard_normal(x.shape)
        direction = (noisy - ideal_denoiser(noisy, raised)) / raised
        x = noisy + step_scale * (levels[i + 1] - raised) * direction
    return x

rows = []
for step_scale in (0.5, 0.8, 1.0, 1.5, 2.0):
    points = sample_points(4000, step_scale)
    left = points[:, 0] < 0
    nearest = np.minimum(np.linalg.norm(points - cluster_centres[0], axis=1), np.linalg.norm(points - cluster_centres[1], axis=1))
    rows.append({"step scale": step_scale, "share in the left cluster": float(left.mean()), "root-mean-square distance to the cluster centre": float(np.sqrt((nearest**2).mean()))})
diffusion_table = pd.DataFrame(rows)
show_table(diffusion_table, floatfmt=".3f")
true_spread = float(np.sqrt(2) * cluster_std)
spread_at = {row["step scale"]: row["root-mean-square distance to the cluster centre"] for row in rows}
'''

BUILD_INPUT_CELL = '''
# The input of a prediction, built by the project's own function (cypmode/validation/build_input.py) for the ligand of the previous cell: the protein, the ligand, the heme as the chemical component HEM, and the covalent bond between
# the sulfur of the cysteine and the iron of the heme. The three versions differ in what is said about the ligand: nothing (the base run), a bond from its donor nitrogen to the iron, or a contact between them. The long protein
# sequence is shortened for display.
from cypmode.validation.build_input import boltz_atom_name, build_boltz_yaml, heme_donor_atom

ligand_smiles = input_yaml["sequences"][1]["ligand"]["smiles"]
donor = heme_donor_atom(ligand_smiles)
yaml_versions = {mode: build_boltz_yaml(cyp3a4_sequence, ligand_smiles, cysteine_position, predict_affinity=False, donor_restraint=mode) for mode in (None, "bond", "contact")}
def shortened(text):
    return re.sub(r"(sequence: \\w{20})\\w+", r"\\1...", text)
print(shortened(yaml_versions["bond"]))
base_lines = yaml_versions[None].count("\\n")
bond_extra = [line for line in yaml_versions["bond"].splitlines() if line not in yaml_versions[None].splitlines()]
contact_extra = [line for line in yaml_versions["contact"].splitlines() if line not in yaml_versions[None].splitlines()]
donor_index, donor_name = donor
'''

CELLS = [
    md("""
# Part XI. The structure track

The challenge has a second track besides the classification of time-dependent inhibition. For 20 compounds the organisers ask for the three-dimensional pose of the compound in CYP3A4: a file of coordinates in which the protein, the heme and the ligand sit where they would in a crystal. The files are scored against experimental structures that are not public. The primary score is **LDDT-PLI**, a measure of how well the contacts between the ligand and the protein are kept (11.4); a second one is **BiSyRMSD**, the distance of the ligand atoms from their true positions after the pocket is superposed; and a pose that fails the organisers' physical-validity checks (the **PoseBusters** checks, 11.4) scores zero on all of them.

The project's position is that of a track with no answer key. The 20 compounds have no analogue in any deposited P450 structure (the best similarity to a deposited ligand is 0.35), so that nothing can be learned from a neighbour, and the models that predict poses are known to be confident when they are wrong. What the project has is the Protein Data Bank, which holds 129 CYP3A4 entries with a ligand in the active site, 148 ligand positions in all (11.5). They can serve as a **crystal-anchored benchmark**: ligands whose pose is known, on which every lever can be tried and scored before it is used on the 20. Most of this part is the account of levers that were tried this way, each with a rule written before its result, and most of them did not pass. The part is therefore also an account of how little the available evidence allows one to claim, and of the one lever, putting the ligand's iron-binding nitrogen on the heme iron, that passed its first gate and then did not pass its confirmation.

The order is the order of the pipeline. Sections 11.1 to 11.3 say what a structure file is, how a co-folding model predicts one and how its input is built. Section 11.4 says how a pose is scored. Sections 11.5 to 11.9 are the benchmarks and the levers: the restraint, the diversity levers, the choice among poses, and the second engine. Section 11.10 is the final recipe and the checks on the 20 files that are submitted. Section 11.11 is the pharmacophore model of a pose, with the checks of what such a model cannot see.
"""),
    code(SEQUENCE_FACTS),
    md("""
## 11.1 Proteins, folding and the structure of CYP3A4; what a complex file contains

> **Field:** Structural biology · Biochemistry

*Where this section sits:* Part I explained the heme and the catalytic cycle of a P450 enzyme, and 5.5 and 5.6 used a predicted structure to check the motif screen. This section starts the structure track from the beginning: what is being predicted, and what the file that holds the answer contains.

**Amino acids, the chain and the fold.** A **protein** is a chain of **amino acids**, twenty kinds, joined end to end by peptide bonds. Each unit of the chain, a **residue**, contributes the same three-atom stretch to the **backbone** (a nitrogen, a central carbon called C-alpha, and a carbonyl carbon) and a **side chain** attached to the C-alpha that differs from kind to kind: small and neutral, large and oily, charged, or able to bind a metal. The **sequence** is the string of residue letters. The **fold** is the three-dimensional shape that the chain takes up, which the sequence determines through many weak interactions: hydrogen bonds between backbone atoms that make helices and sheets, oily side chains that hide from water in the interior, opposite charges that pair. CYP3A4 (UniProt entry P08684) is a chain of {{=sequence_length}} residues. The commonest residues are {{=most_common}}, a composition typical of an enzyme that binds oily molecules. Its first stretch anchors it in the membrane of the cell's endoplasmic reticulum; the rest folds into a mostly helical catalytic domain around a heme.

**The heme and the cysteine.** The heme is a flat ring (a porphyrin) holding an iron atom (1.3). In every P450 enzyme the iron is held from one side by the sulfur of a cysteine, and the surrounding stretch of the sequence is so well conserved that a pattern finds it: phenylalanine, two residues, glycine, three residues, the cysteine, one residue, glycine. In CYP3A4 the pattern matches once, at {{=signature_text}}, and the cysteine is residue {{=cysteine_position}}. The other face of the iron looks into the pocket where substrates and inhibitors bind: it holds a water molecule at rest, binds the oxygen that the enzyme activates, and can be bound by a nitrogen of an inhibitor (1.9). The structure track is the question of how a ligand sits in that pocket relative to that iron.

**Where structures come from.** Experimental structures come from **X-ray crystallography** (the protein is crystallised, the diffraction of X-rays gives an electron density map into which atoms are fitted; the **resolution** in ångströms says how fine the map is: below 2 Å individual atoms are clear, at 3 Å only the shape of side chains) and, increasingly, **cryo-electron microscopy** (images of frozen molecules combined into a map; the five CYP3A4 entries released on 2026-09-30 are at 3.2 to 3.7 Å and are trimers). They are deposited in the **Protein Data Bank**, a public archive; each entry has a four-character identifier. The structure track's reference set is made from the CYP3A4 entries of the archive (11.5). Predicted structures come from models (11.2) and are written in the same formats.

**What the file holds.** The simplest format is that of the Protein Data Bank itself, a text file with one line for each atom. The cell prints the first lines of one of the 20 submitted files and reads it with a library for structures. Each line gives the kind of record (`ATOM` for the residues of the protein, `HETATM` for everything else, here the ligand and the heme), a serial number, the atom name, the residue name, the chain, the residue number, the three coordinates in ångströms, an occupancy, the **B-factor** column and the element. In a crystal file the B-factor records how much an atom is smeared; the co-folding model writes into it its own confidence in the atom's position, from 0 to 100. The format stores no bonds for the ligand, and no bond orders at all: a program that needs them, as the scorer does, takes them from the SMILES string of the compound. The challenge asks for the protein as chain A, the ligand as the residue `LIG` and the heme as `HEM`.
"""),
    code(STRUCTURE_FILE),
    md("""
The file holds {{=n_residues_protein}} residues of protein in {{=n_protein_atoms:,}} atoms, a ligand of {{=n_ligand_atoms}} atoms and the heme of {{=n_heme_atoms}}. The model's confidence is high for the protein ({{=chain_table.iloc[0, 4]:.0f}} on average) and the heme ({{=chain_table.iloc[2, 4]:.0f}}) and lower for the ligand ({{=chain_table.iloc[1, 4]:.0f}}). The sulfur of the cysteine {{=cysteine_position}} is {{=iron_to_sulfur:.2f}} Å from the iron, the distance of a coordinate bond, and the nearest nitrogen of this ligand is {{=iron_to_nitrogen:.2f}} Å from it on the other side. That is the geometry of a Type II ligand (1.9): iron bound by the sulfur on one face and by the nitrogen of the inhibitor on the other. How often the models place a ligand so, and whether they place it right, is the content of the rest of the part.

*On what actually ran.* The cells read `final/_pdb_files/OCNT-2312554.pdb` of the final recipe (11.10) and the input `data/structure_track/inputs/cyp3a4_OCNT-2312554.yaml`, with Biopython and `cypmode/validation/build_input.py` (`find_coordinating_cysteine`, `CYS_MOTIF_PATTERN`). The count of the structures of the archive is the project's own reference set, built by `cypmode/validation/reference_ligands.py` (11.5).
"""),
    md("""
## 11.2 Co-folding models: how a structure is predicted from a sequence and a molecule

> **Field:** Machine learning · Structural biology

*Where this section sits:* 11.1 said what a complex file holds. This section explains the family of models that writes it, from the sequence of the protein and the formula of the ligand alone. The project uses two members of the family, Boltz-2 and OpenFold3, and it uses them as instruments, not as objects of study; the section explains enough of their working for the reader to understand what the levers of 11.6 to 11.9 act on.

**The evolutionary record.** A single sequence says little about the fold. A set of sequences of the same protein from many species, aligned position by position (a **multiple sequence alignment**, MSA), says a great deal, because positions that touch each other in the folded protein must stay compatible: if one changes in the course of evolution, the other tends to change with it. The cell is a toy of this. It draws 400 related sequences of 12 positions at random, makes position 10 follow position 3 in 85 percent of them (positions counted from 1), and measures for every pair of positions the **mutual information** $I(i,j) = \\sum f_{ij}\\log\\big(f_{ij}/(f_i f_j)\\big)$, where $f_i$ is the share of each residue at position $i$ and $f_{ij}$ the share of each pair at positions $i$ and $j$: it is zero when the two positions vary independently and large when one tells the other.
"""),
    code(COEVOLUTION_TOY),
    md("""
The pair ({{=top_pair[0]}}, {{=top_pair[1]}}) stands out with a mutual information of {{=coevolution_table.iloc[0, 2]:.2f}} nats against a median of {{=median_information:.3f}} for all pairs, and the next pair has {{=coevolution_table.iloc[1, 2]:.3f}}. Real alignments hold thousands of sequences and the signal is weaker and spread over many pairs, and the models of this section learn to read it rather than to compute one number; but it is the same evidence. The alignments are made by searching a database of sequences; the model used here can ask a public server for them (the **ColabFold** service, Mirdita and co-workers, 2022), and the project keeps one alignment for CYP3A4 and reuses it for every ligand, which both saves the search and removes a dependence that failed once (11.3).

**The architecture.** The first model of this kind to be accurate on the folding problem was AlphaFold 2 (Jumper and co-workers, 2021), a network that turned an alignment into coordinates. AlphaFold 3 (Abramson and co-workers, 2024) extended it from proteins to complexes of proteins with other molecules, ligands included, and the open models used here follow its design: Boltz-1 (Wohlwend and co-workers, 2024) and its successor Boltz-2 (Passaro, Saro and co-workers, 2025), which also predicts a binding affinity, and OpenFold3 (the OpenFold consortium), a reimplementation of AlphaFold 3. All have two parts. The **trunk** reads the sequences, the alignment and the ligands and builds two representations: one vector for every **token** (a residue of the protein or an atom of a ligand) and one vector for every *pair* of tokens, the **pair representation** $z_{ij}$, which holds what the model has worked out about how $i$ and $j$ relate in space. The **diffusion module** then turns these representations into coordinates for every heavy atom. The trunk is run once for each prediction; the diffusion module is run once for each requested **sample**, from the same trunk output. The samples of one prediction are therefore not independent opinions: they share the trunk's reading of the problem and differ only in what the sampling adds.

**The pair representation and the triangle.** Distances in space obey the triangle inequality: $d_{ij} \\le d_{ik} + d_{kj}$ for any third point $k$. If two positions are both close to a third, they cannot be far from each other. The trunk is built to use this. In a **triangle update** the vector of the pair $(i,j)$ is refreshed with information from the pairs $(i,k)$ and $(j,k)$ for every third token $k$:

$$z_{ij} \\;\\leftarrow\\; z_{ij} + g_{ij}\\odot W_o\\,\\mathrm{LN}\\Big(\\sum_k a_{ik}\\odot b_{jk}\\Big),$$

where $a_{ik}$ and $b_{jk}$ are learned linear transformations of $z_{ik}$ and $z_{jk}$ (each multiplied by a learned gate), $\\mathrm{LN}$ normalises over the channels, $W_o$ is a learned matrix, $g_{ij}$ a learned gate between 0 and 1, and $\\odot$ is multiplication element by element. A second operation, **triangle attention**, lets the pair $(i,j)$ attend over the pairs $(i,k)$ with a weight that depends on $z_{jk}$. The cell runs one update with random weights. It cannot show what a trained model has learned, but it shows what the operation is: it treats positions by what they are connected to and not by their number, so that relabelling the positions and computing gives the relabelled result.
"""),
    code(TRIANGLE_TOY),
    md("""
Relabelling the seven positions changes the result by {{=equivariance_error:.0e}}, which is rounding, and the update is {{=change_size:.0%}} of the size of the state. Triangle attention needs memory proportional to the cube of the number of tokens, which is what failed on this machine (11.9).

**The diffusion module.** A **diffusion model** generates by reversing a corruption. Take the coordinates $x_0$ of a structure and add noise of size $\\sigma$: $x_\\sigma = x_0 + \\sigma\\varepsilon$ with $\\varepsilon$ drawn from a normal distribution. A network $D_\\theta(x_\\sigma, \\sigma)$, the **denoiser**, is trained to return the expected clean structure given the noisy one, $D \\approx E[x_0 \\mid x_\\sigma]$, by minimising the squared distance to the true $x_0$ over many structures and noise sizes; here it is also given the trunk's representations. To generate, start from pure noise of a large size $\\sigma_{\\max}$ and lower the noise in steps $\\sigma_{\\max} = t_0 > t_1 > \\dots > t_n \\approx 0$. In the form used by AlphaFold 3 and Boltz (after Karras and co-workers, 2022) a step is: raise the noise level a little, $\\hat t = t_i(1+\\gamma)$, by adding fresh noise to match; then move toward the denoised estimate,

$$x_{i+1} \\;=\\; \\hat x + \\eta\\,(t_{i+1} - \\hat t)\\,\\frac{\\hat x - D_\\theta(\\hat x, \\hat t)}{\\hat t},$$

with $\\hat x$ the noisy structure, $\\gamma$ the size of the added noise (the "churn") and $\\eta$ the **step scale**. With $\\eta = 1$ and no added noise this is a plain numerical solution of the equation that carries the noisy distribution back to the data. The step scale of the models is a setting of the sampler (`--step_scale`, 1.5 by default for Boltz-2, and the documentation says that lower values give more diversity among the samples). The cell builds a sampler of this form in two dimensions where the ideal denoiser can be written down: the "data" are two clusters of points at $(\\pm 3, 0)$, standing for two different poses of a ligand, each a Gaussian of width 0.3.
"""),
    code(DIFFUSION_TOY),
    md("""
Two things can be read. The share of points in the left cluster stays at one half for every step scale ({{=diffusion_table['share in the left cluster'].min():.2f}} to {{=diffusion_table['share in the left cluster'].max():.2f}}): the step scale does not change *which* pose is chosen, it changes how far a sample strays from the pose it chose. And that stray distance falls steeply as the step scale rises: the true spread of a cluster is {{=true_spread:.2f}} (the root-mean-square distance of a point to the centre of a Gaussian of width 0.3 in two dimensions), the sampler reproduces it at a step scale of 1 ({{=spread_at[1.0]:.2f}}), and at the default of 1.5 for the real models it gives samples that are {{=spread_at[1.5] / true_spread:.0%}} as wide as the distribution they sample from. A step scale above 1 makes the samples hug the mode: they are narrower than the distribution they come from. At 0.5 the samples are {{=spread_at[0.5] / true_spread:.1f}} times too wide. That is the lever tried in 11.7, to get samples that differ.

**The confidence.** The models also output confidence scores for each atom and each pair (the per-atom confidence in the B-factor column of 11.1 is one) and for the complex as a whole, trained to predict how close the model's own structure is to the truth. They are useful as warnings of a disordered region. They are not a reliable ranking of poses of one ligand: on the crystal benchmark of 11.5, Boltz-2's confidence is 0.91 to 0.96 even for poses 5 to 6 Å from the crystal pose, which is why the project does not select by it (11.8).

*On what actually ran.* The cells above are toys written for the notebook; none runs a real co-folding model, which takes tens of minutes for one prediction. The real models run through `cypmode/validation/batch.py` (Boltz-2, 11.3) and `cypmode/validation/openfold3.py` (OpenFold3, 11.9). Boltz-2 and Boltz-1 are released under the MIT licence; the documentation of the program is the source of the step-scale statement and its default.
"""),
    md("""
## 11.3 Building the input of a prediction

> **Field:** Software engineering · Structural biology

*Where this section sits:* 11.2 described the model. This section describes what it is given, and how the project builds that input for any ligand with its own code (`cypmode/validation/build_input.py`), so that every choice that matters, such as the covalent bond between the cysteine and the heme or a restraint on the ligand, is stated once and can be tested.

**The input file.** Boltz-2 reads a YAML file with a list of molecules (the **sequences** section: a protein chain with its sequence, a ligand given by a SMILES string or by the code of a chemical component, here the heme with the code `HEM`), a list of **constraints**, optional **templates** and optional **properties** (the affinity head, which runs its own diffusion five times and costs from 55 minutes to over ten hours on this machine, so that it is switched off for the structure track). A **constraint** can be a covalent **bond** between two named atoms, a **contact** (two atoms or residues within a maximum distance, which may be enforced by a potential that pushes the structure toward the condition, or not) or a **pocket** (a set of residues that the ligand should touch). A **template** is the structure of the protein, or part of it, from an experiment, given as a soft guide for the backbone. The program also takes settings of the sampler: the number of sampling steps (200 by default), the number of recycling steps of the trunk (3), the number of samples from one trunk pass, and the step scale of 11.2.

**What the project always says.** The heme is not part of the protein and the model does not know that it is bound to the cysteine, so the input says so with a bond constraint between the sulfur of the cysteine and the iron of the heme, with the cysteine's position found by the signature of 11.1. Everything else about the ligand's position is left to the model. The cell builds the input for the ligand of 11.1 in its three forms, with the project's function `build_boltz_yaml`. The first is the base run. The second adds a bond from the ligand's donor nitrogen to the iron, and the third a contact between them at the tightest distance the program accepts, 4 Å, without enforcement. The donor is chosen by `heme_donor_atom`: among the unsubstituted aromatic nitrogens that belong to a Type II motif (imidazole, triazole, pyridine, thiazole, 5.1) the one with the fewest heavy atoms within two bonds, so that a nitrogen flanked by substituents loses to a bare one. The program names the atoms of a ligand by its own ranking, and a constraint has to quote that name, so `boltz_atom_name` reproduces the naming.
"""),
    code(BUILD_INPUT_CELL),
    md("""
The input differs between the base run and the bond-restrained run by {{=len(bond_extra)}} lines, which are the bond from the ligand's nitrogen {{=donor_name}} (atom {{=donor_index}} of the SMILES) to the iron, and between the base run and the contact-restrained run by {{=len(contact_extra)}} lines. A ligand without a donor of that kind gets no extra lines and runs as the base run. Three details of this code carry the history of the project. The alignment of the protein is given as a file (`protein_msa`) and not searched for each ligand: an outage of the public search server failed four runs in a row during a long batch, and a stored alignment removes the dependence. The restraints need the affinity head off, because with it on the program renames the atoms of the ligand after standardising its SMILES and the names in the constraint no longer match. And the templates are protein files given with the identifier of the chain to use, so that a trimer from cryo-electron microscopy can supply one chain (11.7).

**Running a batch.** One prediction takes about 40 minutes for five samples on this machine, one sample at a time, and a run of the 20 ligands takes a day. `cypmode/validation/batch.py` runs a list of ligands with the properties that a long run needs: each ligand's result is written as it finishes, a failure of one ligand is recorded and does not stop the others, and a second call skips the ligands that succeeded. Its options carry the levers of the following sections: `--donor-restraint bond|contact`, `--max-parallel-samples` (the number of samples drawn at once; 1 after a swap-exhaustion crash of the machine), `--step-scale`, and per-row columns for a template, a seed and a number of samples. `cypmode/validation/pipeline.py` wraps one compound from SMILES to coordination result, and `cypmode/validation/build_summary.py` collects the summary.

*On what actually ran.* `cypmode/validation/build_input.py` (the functions named above), executed by the cell; `batch.py`, `pipeline.py` and `build_summary.py` as described. No prediction is run by the notebook. The settings quoted are those of the Boltz documentation (version in `environment/`).
"""),
]
