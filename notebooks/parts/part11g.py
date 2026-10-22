# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part XI (seventh part): OpenFold3, the second engine, the memory failure and its fix (11.9)."""

from notebooks.notebook_builder import code, md

OF3_RUN = '''
# OpenFold3 as it was run for the benchmark of 11.5: the query of one ligand, the package and the weights, the time of every run (the log of job 150_openfold3_benchmark) and the size of each problem in tokens. Everything is
# read from the files that the runs left, none is typed in.
import json

import gemmi
from rdkit import Chem

import cypmode.validation.openfold3 as of3_module

of3_root = ROOT / "data/structure_track/outputs/openfold3/cyp3a4"
of3_logs = ROOT / "data/structure_track/outputs/cyp3a4/logs"
of3_chains = json.loads((of3_root / "bench-YNV/query.json").read_text())["queries"]["cyp3a4_bench-YNV"]["chains"]
of3_weights = json.loads((of3_root / "bench-YNV/experiment_config.json").read_text())["inference_ckpt_name"]
of3_version = re.search(r"^Version: (\\S+)", next((ROOT / ".venv-openfold3/lib/python3.13/site-packages").glob("openfold3-*.dist-info/METADATA")).read_text(), re.M).group(1)
of3_notes = re.sub(r"\\s+", " ", of3_module.__doc__)
of3_n_fe, of3_boltz_n_fe = (float(v) for v in re.search(r"imidazole nitrogen (\\d\\.\\d+) A from the iron with no constraints at all \\(Boltz-2: (\\d\\.\\d+) A\\)", of3_notes).groups())
of3_s_fe = float(re.search(r"Cys442 SG to Fe (\\d\\.\\d+) A", of3_notes).group(1))
of3_plddt_without = int(re.search(r"useless \\(pLDDT (\\d+)\\)", of3_notes).group(1))

def first_sample(name, suffix):
    return sorted((of3_root / name).glob(f"cyp3a4_{name}/seed_*/*sample_1_{suffix}"))[0]

structure = gemmi.read_structure(str(first_sample("bench-YNV", "model.cif")))
chain_sizes = {chain.name: sum(len(residue) for residue in chain) for chain in structure[0]}
chain_rows = []
for chain in of3_chains:
    letter = chain["chain_ids"][0]
    if chain["molecule_type"] == "protein":
        chain_rows.append({"chain": letter, "molecule": "CYP3A4", "given as": "amino-acid sequence", "tokens": len(chain["sequence"])})
    elif "smiles" in chain:
        chain_rows.append({"chain": letter, "molecule": "the inhibitor", "given as": "SMILES", "tokens": Chem.MolFromSmiles(chain["smiles"]).GetNumHeavyAtoms()})
    else:
        chain_rows.append({"chain": letter, "molecule": "heme", "given as": "component code " + chain["ccd_codes"][0], "tokens": chain_sizes[letter]})
chain_table = pd.DataFrame(chain_rows)
show_table(chain_table, floatfmt=".0f")

benchmark_smiles = pd.read_csv(ROOT / "data/structure_track/inputs/structure_benchmark.csv").set_index("Molecule_Name")["SMILES"]
run_lines = re.finditer(r"^\\s+(bench-\\S+): (\\d+) sample\\(s\\), ([\\d.]+) s", (of3_logs / "bench_of3.log").read_text(), re.M)
of3_runs = pd.DataFrame([(m.group(1), int(m.group(2)), float(m.group(3))) for m in run_lines], columns=["ligand", "samples", "seconds"])
of3_runs["heavy atoms"] = [Chem.MolFromSmiles(benchmark_smiles[n]).GetNumHeavyAtoms() for n in of3_runs["ligand"]]
of3_runs["tokens"] = [len(json.loads(first_sample(n, "confidences.json").read_text())["pae"]) for n in of3_runs["ligand"]]
of3_runs["minutes"] = of3_runs["seconds"] / 60
show_table(of3_runs[["ligand", "samples", "heavy atoms", "tokens", "minutes"]], floatfmt=".1f")

residues = chain_table.loc[chain_table["molecule"] == "CYP3A4", "tokens"].iloc[0]
heme_atoms = chain_table.loc[chain_table["molecule"] == "heme", "tokens"].iloc[0]
token_count_checks = bool((of3_runs["tokens"] == residues + of3_runs["heavy atoms"] + heme_atoms).all())
of3_seeds = sorted({p.name for n in of3_runs["ligand"] for p in (of3_root / n).glob("cyp3a4_*/seed_*")})
ligand_atoms = int(chain_table.loc[chain_table["molecule"] == "the inhibitor", "tokens"].iloc[0])
of3_seed_number = of3_seeds[0].replace("seed_", "")
of3_hours, of3_min, of3_max, of3_mean = float(of3_runs["seconds"].sum() / 3600), float(of3_runs["minutes"].min()), float(of3_runs["minutes"].max()), float(of3_runs["minutes"].mean())
assert token_count_checks and len(of3_seeds) == 1  # a failing cell fails the build
print(f"tokens = residues + heavy atoms of the ligand + atoms of the heme for all {len(of3_runs)} ligands; one seed folder: {of3_seeds[0]}")
'''

OOM_ARITH = '''
# The failure, from the two messages that were logged and the arithmetic that explains them. The messages: the probe of job 149 (the runner file alone, three samples of the largest ligand, bench-YNV) and the log that an earlier
# attempt on bench-TCI left. The sizes are those the run itself wrote: the number of tokens is the side of the matrix of expected errors that every run saves, and the number of heads, of blocks and of samples are in the configuration
# that the run saved.
probe_text = (of3_logs / "openfold3_probe_config.log").read_text()
logged_probe = float(re.search(r"Invalid buffer size: ([\\d.]+) GiB", probe_text).group(1))
error_text = (of3_root / "bench-TCI/logs/predict_err_rank0.log").read_text()
allocated_tci, max_allowed, logged_tci = (float(v) for v in re.search(r"MPS allocated: ([\\d.]+) GiB.*?max allowed: ([\\d.]+) GiB\\)\\. Tried to allocate ([\\d.]+) GiB", error_text).groups())
frames = re.findall(r'File "[^"]*/(openfold3/[^"]+)", line (\\d+), in (\\w+)', error_text)
for path, line, name in frames:
    if path.endswith(("heads/prediction_heads.py", "utils/chunk_utils.py")):
        print(f"{path}, line {line}, in {name}")

model_config = json.loads((of3_root / "bench-YNV/model_config.json").read_text())["architecture"]
head_stack = model_config["heads"]["pairformer_embedding"]["pairformer"]
n_heads, n_blocks_head, n_blocks_trunk = head_stack["no_heads_pair"], head_stack["no_blocks"], model_config["pairformer"]["no_blocks"]
n_samples = model_config["shared"]["diffusion"]["no_full_rollout_samples"]
bytes_per_number = 4
of3_runs["predicted GiB"] = bytes_per_number * n_samples * n_heads * of3_runs["tokens"] ** 3 / 2 ** 30
logged_gib = {"bench-YNV": logged_probe, "bench-TCI": logged_tci}
of3_runs["logged GiB"] = [logged_gib.get(n, float("nan")) for n in of3_runs["ligand"]]
show_table(of3_runs[["ligand", "tokens", "predicted GiB", "logged GiB"]], floatfmt=".2f")
relative_errors = {n: float(abs(of3_runs.set_index("ligand").loc[n, "predicted GiB"] / g - 1)) for n, g in logged_gib.items()}
largest_error = max(relative_errors.values())
n_ynv = int(of3_runs.set_index("ligand").loc["bench-YNV", "tokens"])
predicted_ynv = float(of3_runs.set_index("ligand").loc["bench-YNV", "predicted GiB"])
bias_mib = bytes_per_number * n_heads * n_ynv ** 2 / 2 ** 20
copies = n_samples * n_ynv
'''

COPY_TOY = '''
# Broadcasting and flattening, on small arrays. The bias of triangle attention is one array of shape (H, N, N) per sample, shared by all N rows of the pair representation. The layer sees it as an array of shape (S, N, H, N, N),
# S samples times N rows, without storing the repeats, and the chunking routine then flattens the first two axes into one. numpy follows the same rules as the library of the program (checked for PyTorch, in its own environment,
# by a test of the repository).
rng_of3 = np.random.default_rng(3)
n_toy, heads_toy = 48, 4
copy_rows = []
for samples_toy in (1, 2, 3):
    bias_toy = rng_of3.standard_normal((samples_toy, 1, heads_toy, n_toy, n_toy)).astype(np.float32)
    seen_toy = np.broadcast_to(bias_toy, (samples_toy, n_toy, heads_toy, n_toy, n_toy))
    flat_toy = seen_toy.reshape(-1, heads_toy, n_toy, n_toy)
    shares_toy = bool(np.shares_memory(flat_toy, bias_toy))
    copy_rows.append({"samples": samples_toy, "stored bias (KiB)": bias_toy.nbytes / 1024, "flattened array shares that memory": shares_toy, "memory held after flattening (KiB)": (bias_toy.nbytes if shares_toy else flat_toy.nbytes) / 1024})
copy_table = pd.DataFrame(copy_rows)
copy_table["growth"] = copy_table["memory held after flattening (KiB)"] / copy_table["stored bias (KiB)"]
show_table(copy_table, floatfmt=".1f")
'''

CHUNK_TOY = '''
# The chunked layer, three ways, on the toy of 11.2 made larger: starting-node triangle attention on S = 3 pair representations of N = 48 tokens with H = 4 heads. "Reference" computes every row at once. "Flatten first" is the program's
# routine as it came: it broadcasts the bias to every row, flattens (a copy for S > 1) and then cuts chunks of rows. "Views" is the low-memory path: it keeps the broadcast view and cuts every chunk out of it, so that only the chunk is
# ever materialised. The peak memory is measured with tracemalloc, which follows the arrays of numpy; the difference of the results is the largest absolute difference from the reference.
import tracemalloc

S_toy, N_c, C_c, H_c, D_c, rows_c = 3, 48, 16, 4, 8, 16
rng_c = np.random.default_rng(11)
x_c = rng_c.standard_normal((S_toy, N_c, N_c, C_c)).astype(np.float32)
w_q, w_k, w_v = (rng_c.standard_normal((C_c, H_c * D_c)).astype(np.float32) / np.sqrt(C_c) for _ in range(3))
tri_c = rng_c.standard_normal((S_toy, 1, H_c, N_c, N_c)).astype(np.float32)  # one bias per sample, shared by all rows
mask_c = (-1e9 * (rng_c.random((S_toy, N_c, 1, 1, N_c)) < 0.1)).astype(np.float32)

def softmax_last(a):
    a = a - a.max(axis=-1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=-1, keepdims=True)

def projections(x):
    split = lambda w: (x @ w).reshape(*x.shape[:-1], H_c, D_c)
    return split(w_q), split(w_k), split(w_v)

def reference():
    q, k, v = projections(x_c)
    logits = np.einsum("sijhd,sikhd->sihjk", q, k) / np.sqrt(D_c) + tri_c + mask_c
    return np.einsum("sihjk,sikhd->sijhd", softmax_last(logits), v).reshape(S_toy, N_c, N_c, H_c * D_c)

def chunked(views):
    q, k, v = (a.reshape(S_toy * N_c, N_c, H_c, D_c) for a in projections(x_c))
    seen = np.broadcast_to(tri_c, (S_toy, N_c, H_c, N_c, N_c))
    if not views:
        flat_bias = seen.reshape(-1, H_c, N_c, N_c)  # the copy
        flat_mask = mask_c.reshape(S_toy * N_c, 1, 1, N_c)
    out = np.empty((S_toy * N_c, N_c, H_c, D_c), dtype=np.float32)
    for lo in range(0, S_toy * N_c, rows_c):
        hi = min(lo + rows_c, S_toy * N_c)
        if views:
            s_index, i_index = np.divmod(np.arange(lo, hi), N_c)
            bias_c, mask_part = seen[s_index, i_index], mask_c[s_index, i_index]
        else:
            bias_c, mask_part = flat_bias[lo:hi], flat_mask[lo:hi]
        logits = np.einsum("rjhd,rkhd->rhjk", q[lo:hi], k[lo:hi]) / np.sqrt(D_c) + bias_c + mask_part
        out[lo:hi] = np.einsum("rhjk,rkhd->rjhd", softmax_last(logits), v[lo:hi])
    return out.reshape(S_toy, N_c, N_c, H_c * D_c)

def measured(function, *args):
    tracemalloc.start()
    result = function(*args)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return result, peak / 2 ** 20

expected_c, peak_reference = measured(reference)
flat_c, peak_flat = measured(chunked, False)
views_c, peak_views = measured(chunked, True)
chunk_table = pd.DataFrame({"way": ["reference (all rows at once)", "flatten first (as it came)", "views (low memory)"], "peak memory (MiB)": [peak_reference, peak_flat, peak_views],
                            "largest difference from the reference": [0.0, float(np.abs(flat_c - expected_c).max()), float(np.abs(views_c - expected_c).max())]})
show_table(chunk_table, floatfmt=".1e")
bias_copy_mib = tri_c.nbytes * N_c / 2 ** 20
chunk_mib_real = rows_c * bytes_per_number * n_heads * n_ynv ** 2 / 2 ** 20
'''

FIX_FACTS = '''
# What the fix is checked by: the self-check of the wrapper (run in the OpenFold3 environment by job 149 and by a test of the repository), the probe of the same job with the wrapper in force, and the mode that the probe chose.
selfcheck_line = [line for line in (of3_logs / "openfold3_lowmem_selfcheck.log").read_text().splitlines() if line.startswith("self-check")][-1]
probe_ok = re.search(r"(bench-\\S+): (\\d+) sample\\(s\\), ([\\d.]+) s", (of3_logs / "openfold3_probe_lowmem.log").read_text())
probe_ligand, probe_samples, probe_seconds = probe_ok.group(1), int(probe_ok.group(2)), float(probe_ok.group(3))
memory_mode = (ROOT / "data/structure_track/inputs/openfold3_memory_mode.txt").read_text().strip()
wrapper_source = (ROOT / "scripts/run_openfold_lowmem.py").read_text()
tolerance_unchunked, tolerance_original = re.findall(r"atol=([\\d.e-]+)", wrapper_source)
print(selfcheck_line)
print(f"probe with the wrapper: {probe_ligand}, {probe_samples} samples, {probe_seconds:.1f} s; memory mode written by the probe: {memory_mode}")
'''

BENCH = '''
# The benchmark of the second engine: the ten ligands of 11.5 with five Boltz-2 samples and three OpenFold3 samples each. The pose table holds the RMSD of every sample to the crystal pose; the outcomes table holds, for every ligand, the RMSD of
# the first sample of each engine and of the pose chosen by minimum Bayes risk (11.8) from the pool of Boltz-2 alone and from the pool of both engines. The gate is recomputed with the function that registered it, and the repetition with
# OpenStructure's yardstick is read from the log of job 245_ost_pool_gates.
import cypmode.validation.final_recipe as final_recipe_module
from cypmode.validation.selection_validation import gates as pose_gates

of3_outcomes = pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/selection_validation_outcomes.csv")
of3_gate = pose_gates(of3_outcomes)["engine_consensus"]
sample_table = pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/benchmark_engine_pose_validation.csv")
by_engine = sample_table.groupby("engine")["true_rmsd"].agg(samples="count", mean="mean", median="median", best="min")
by_engine.index = by_engine.index.map({"boltz": "Boltz-2", "openfold3": "OpenFold3"})
show_table(by_engine.reset_index().rename(columns={"engine": "engine", "mean": "mean RMSD (A)", "median": "median RMSD (A)", "best": "best sample (A)"}), floatfmt=".2f")
first_closer = int((of3_outcomes["first_openfold3"] < of3_outcomes["first_boltz"]).sum())
first_gap = float((of3_outcomes["first_openfold3"] - of3_outcomes["first_boltz"]).mean())
change = of3_outcomes["mbr_both"] - of3_outcomes["mbr_boltz"]
n_changed, n_worse, n_better = int((change.abs() > 1e-9).sum()), int((change > 1e-9).sum()), int((change < -1e-9).sum())
ligand_table = of3_outcomes[["compound", "first_boltz", "first_openfold3", "mbr_boltz", "mbr_both"]].rename(columns={"compound": "ligand", "first_boltz": "Boltz-2 first sample", "first_openfold3": "OpenFold3 first sample",
                                                                                                                  "mbr_boltz": "MBR, Boltz-2 pool", "mbr_both": "MBR, both pools"})
show_table(ligand_table, floatfmt=".2f")
oracle_boltz, oracle_both = float(of3_outcomes["oracle_boltz"].mean()), float(of3_outcomes["oracle_both"].mean())
# A look after the verdict, not a test: the ligand whose pick got worst when the second pool was added, and the same comparison without it.
worst_ligand = of3_outcomes.loc[change.idxmax()]
rest = of3_outcomes.drop(change.idxmax())
rest_gate = pose_gates(rest)["engine_consensus"]
rest_gain = -float((rest["mbr_both"] - rest["mbr_boltz"]).mean())
rest_verdict = "would meet" if rest_gate["met"] else "would not meet"
mbr_boltz_mean, mbr_both_mean = float(of3_outcomes["mbr_boltz"].mean()), float(of3_outcomes["mbr_both"].mean())
ost_found = re.search(r"engine_consensus: mean improvement ([+-][\\d.]+) A over (\\d+) ligands, (\\d+) wins, Wilcoxon p ([\\d.]+), worst drop-one mean ([+-][\\d.]+), errors correlation rho ([\\d.]+)", (of3_logs / "ost_pool_gates.log").read_text())
ost_improvement, ost_wins, ost_p, ost_rho = float(ost_found.group(1)), int(ost_found.group(3)), float(ost_found.group(4)), float(ost_found.group(6))
n_test_ligands = len(pd.read_csv(ROOT / "data/structure_track/inputs/structure_track_test_blinded.csv"))
of3_extra_hours = n_test_ligands * final_recipe_module.OPENFOLD3_SAMPLES * final_recipe_module.HOURS_PER_OPENFOLD3_SAMPLE
final_recipe_openfold = bool(json.loads((ROOT / "data/structure_track/outputs/cyp3a4/final/recipe.json").read_text())["recipe"]["openfold3"])
'''

BIOEMU = '''
# BioEmu2, examined as a third structure engine: the record of the first attempts and, once job 259 has run, the timings of the probe and the registered feasibility rule applied to them. Every number is read from the files that the attempts left.
import json
import re

import yaml

import cypmode.validation.bioemu2 as bioemu_module

bioemu_logs = ROOT / "data/structure_track/outputs/cyp3a4/logs"
bioemu_first = (bioemu_logs / "exploratory_bioemu2_first_attempts.log").read_text()
bioemu_cpu_s, bioemu_gpu_s = (int(v) for v in re.findall(r"in (\\d+) s", bioemu_first))
bioemu_peptide_atoms = int(re.search(r"(\\d+) atoms", bioemu_first).group(1))
bioemu_stops = [(int(a), int(b) / 1000) for a, b in re.findall(r"WATCHDOG: free (\\d+)% swap (\\d+)M", bioemu_first)]
bioemu_system = yaml.safe_load((ROOT / "data/structure_track/inputs/cyp3a4_bench-MWS.yaml").read_text())
bioemu_residues = len(next(e["protein"]["sequence"] for e in bioemu_system["sequences"] if "protein" in e))
bioemu_probe_log = (bioemu_logs / "exploratory_bioemu2_probe.log").read_text().strip().splitlines()
bioemu_rule_line = bioemu_probe_log[-1]
bioemu_limit_h = bioemu_module.MAX_EXCLUSIVE_HOURS
bioemu_benchmark, bioemu_official, bioemu_samples = bioemu_module.BENCHMARK_LIGANDS, bioemu_module.OFFICIAL_LIGANDS, bioemu_module.SAMPLES_PER_LIGAND
bioemu_timings_file = ROOT / "data/structure_track/outputs/bioemu2/probe/cyp3a4" / bioemu_module.PROBE_NAME / "timings.json"
if bioemu_timings_file.exists():
    bioemu_t = json.loads(bioemu_timings_file.read_text())
    bioemu_sentence = (f"The probe completed: the trunk took {bioemu_t['trunk_s']:.0f} s, the two samples {bioemu_t['sample_s'][0]:.0f} s and {bioemu_t['sample_s'][1]:.0f} s, "
                       f"and the process held at most {bioemu_t['peak_rss_mb'] / 1000:.1f} GB of memory.")
else:
    bioemu_sentence = "The probe did not complete; its log records how far it came."
print(bioemu_rule_line)
'''

CELLS = [
    md("""
## 11.9 OpenFold3: the second engine, the memory failure and what the second engine bought

> **Field:** Software engineering · Structural biology

*Where this section sits:* 11.2 described the family of co-folding models and 11.3 the batch that runs Boltz-2 on every ligand. 11.8 asked whether any rule picks a better pose from a pool than the model's own first sample, and one of its questions was whether a second engine adds information. This section is that second engine in full: what was run and how long it took, the failure that stopped its first runs on this machine and its cause, the change that made them fit and the check that the change leaves the answers alone, the benchmark and the verdict. The failure is an example of a kind of problem that the other sections do not show, a program that is correct on paper and does not fit in the memory of the computer, and of what it takes to show that a change which makes it fit does not change what it computes.
"""),
    code(OF3_RUN),
    md("""
**What was run.** OpenFold3 is the reimplementation of AlphaFold 3 named in 11.2. The package (version {{=of3_version}}, Apache licence 2.0) lives in an isolated environment of its own, the pattern used for Boltz-2, and the weights are the checkpoint `{{=of3_weights}}`. For each ligand the runner writes one query: the protein as a sequence (chain A), the inhibitor as a SMILES string (chain B) and the heme as a code of the chemical-component dictionary, HEM (chain C). These are the three chains and the same letters as in the Boltz-2 input, and the runner reads the sequence and the SMILES from that very input file, so that the two engines are given exactly the same system. The first table is the query of one ligand with the size of each chain in tokens (11.2: a residue of the protein, an atom of a ligand): {{=int(chain_table['tokens'].sum())}} tokens for the ligand with {{=ligand_atoms}} heavy atoms, which is the side of the matrix of predicted aligned errors (PAE, the model's estimate of the error in the position of one token when the structure is aligned on another) that every run saves; the cell checks that the same sum, protein residues plus heavy atoms of the ligand plus the {{=heme_atoms}} atoms of the heme, gives the side of that matrix for all ten ligands of the second table.

Three choices separate these runs from the Boltz-2 ones. *The alignments*: each run asks the ColabFold server for the alignment of the protein, and keeps it in its own folder, because without alignments the model is of no use: the first trial without them reached a pLDDT of {{=of3_plddt_without}}. *No restraint*: the Boltz-2 batch can be told that the nitrogen of the ligand binds the iron (11.6). The OpenFold3 query carries no such restraint, since the program names the atoms of a bond by integer positions, which are easy to get wrong, and in the first trial, with ketoconazole, a crystallised inhibitor of CYP3A4, the model placed the nitrogen {{=of3_n_fe}} Å from the iron and the sulfur of cysteine 442 {{=of3_s_fe}} Å from it without any constraint (Boltz-2: {{=of3_boltz_n_fe}} Å). *One run at a time*: the runner refuses to start a prediction while another OpenFold3 process is alive, and none is run beside Boltz-2, because a single prediction already takes most of the memory of the machine.

A run of three samples takes {{=of3_min:.0f}} to {{=of3_max:.0f}} minutes depending on the size of the ligand ({{=of3_mean:.0f}} on average), so the ten ligands of the benchmark took {{=of3_hours:.1f}} hours of machine time. The run folders carry the same seed number, {{=of3_seed_number}}, for all ten ligands (the query file asks for seed 42; the folder name is the record of the seed used).

The machine has 16 gibibytes of **unified memory**: memory that the processor and the graphics processor share, so that what a model holds on the graphics processor is taken from the same 16 gibibytes as everything else. A **gibibyte** (GiB) is $2^{30} = 1{,}073{,}741{,}824$ bytes, the unit that the messages below are written in; a gigabyte counts $10^9$ and is 7 percent smaller. The graphics backend of PyTorch on this machine (MPS, Apple's Metal Performance Shaders) is allowed to hold only a part of that at once (the logs below state the limit) and refuses a single block that is too large for it.
"""),
    code(OOM_ARITH),
    md("""
**The failure.** The probe of job 149 asked for three samples of the largest ligand of the benchmark, bench-YNV, with the runner file alone. It failed with `Invalid buffer size: {{=logged_probe}} GiB`: the program asked the backend for one block of that size. An earlier attempt on another ligand, bench-TCI, stopped with `Tried to allocate {{=logged_tci}} GiB` while {{=allocated_tci}} GiB were already in use and {{=max_allowed}} GiB were allowed. The traceback that the second attempt kept says where, in the lines printed above the table: in the *confidence head*, a second pair stack of {{=n_blocks_head}} blocks that runs after sampling to compute the confidence scores (the trunk's own pair stack has {{=n_blocks_trunk}} blocks and runs once), inside the routine `chunk_layer`, which does the **chunking**: it cuts the work of a layer into pieces of a few rows so that only one piece is in memory at a time. The line that failed flattens the batch dimensions of the inputs of triangle attention before the first piece is cut.

**The arithmetic.** A number in single precision takes 4 bytes, so an array of $E$ numbers takes $4E$ bytes. Let $S$ be the number of samples that the confidence head works on at once ({{=n_samples}} in the benchmark, the number requested), $N$ the number of tokens and $H$ the number of heads of attention in its pair stack ({{=n_heads}}). The bias of triangle attention has shape $(N, H, N, N)$ once it has been repeated for each of the $N$ rows, and the head works on the $S$ samples together, so the flattened array has $S\\,N\\,H\\,N\\,N$ numbers:

$$\\text{bytes} \\;=\\; 4\\,S\\,N\\,H\\,N\\,N \\;=\\; 4\\,S\\,H\\,N^{3}.$$

For the ligand of the probe, $N = {{=n_ynv}}$, which gives {{=predicted_ynv:.2f}} GiB. The third column of the table is this formula for all ten ligands and the fourth is what the logs say for the two that were logged: they agree to within {{=largest_error:.1%}}. The cube is the point. The bias as stored is small, $4HN^2$ bytes, {{=bias_mib:.1f}} MiB; its flattened copy is $S N = {{=copies}}$ times larger, and the cube grows fast with the ligand: the tokens of the ten ligands run from {{=int(of3_runs['tokens'].min())}} to {{=int(of3_runs['tokens'].max())}}, and the allocations from {{=of3_runs['predicted GiB'].min():.1f}} to {{=of3_runs['predicted GiB'].max():.1f}} GiB. Even the smallest of them, {{=of3_runs['predicted GiB'].min():.1f}} GiB, is of the size of the request that failed in bench-TCI when {{=allocated_tci}} GiB were already in use.

**Why a copy.** In triangle attention (11.2) each row of the pair representation attends along the row, and the weights are shifted by a bias that depends on the pair of positions and not on the row: the same array serves all $N$ rows. The program treats the stored array as if it were repeated $N$ times without storing the repeats; this is **broadcasting**, and the repeated array is a *view*, a different description of the same memory. The chunking routine begins by giving every input the same batch shape and then flattens the batch dimensions into one, so that it can count off rows. Flattening a view costs nothing when the result can still be described as one regular block of memory, with the same step from each row to the next, and costs a copy when it cannot. With one sample the batch dimensions are $(1, N)$ and it can. With two or more they are $(S, N)$: the $N$ rows of the first sample repeat the first sample's array and the next $N$ rows repeat another, and no single step describes that. The cell shows it on small arrays.
"""),
    code(COPY_TOY),
    md("""
With one sample the flattened array shares the stored memory and holds nothing beyond it; with two or three samples it holds {{=copy_table.loc[2, 'growth']:.0f}} times what is stored, which is $N$ = {{=n_toy}}, the number of rows over which each bias is repeated (and $S\\,N$ times the bias of one sample). The same holds for the library the program uses, PyTorch (a test of the repository runs the same shapes in its environment). It agrees with the first trial: the single-sample prediction of ketoconazole ran in the memory of the machine, because with one sample the flattening is a view; the benchmark asked for three.

**Why the chunk size did not help.** The chunking routine exists to bound memory: it evaluates the layer on a few rows at a time. But the flattening happens before the first chunk is cut, so the size of the chunk does not enter. The runner file of the project lowers the upper bound of the chunk size to 128 (the preset sets 1024 and lets a tuner raise it to what fits the first block), which caps the buffers that depend on the chunk size. The probe of job 149 shows that this was not enough: the run with the runner file alone is the mode `config`, and it failed at the confidence head.

**The change.** The chunking routine already has the option that the failing path lacks, `low_mem`, which keeps the broadcast view and cuts every chunk out of it, so that only the chunk is ever materialised; no call site of the program sets it. The script `scripts/run_openfold_lowmem.py` makes it the default at every call site and then hands over to the program's own command line; it changes no file of the package. What it must not change is the answer. Chunking evaluates the same attention row by row, so the argument is that the arithmetic is the same, and the cell checks it on the toy: the three ways of computing the layer, all rows at once, flatten first as it came, and the low-memory path, with the peak memory of each.
"""),
    code(CHUNK_TOY),
    md("""
The three give the same layer to {{=max(chunk_table['largest difference from the reference']):.0e}} (rounding of single-precision sums in a different order). The memory differs: flattening first holds a copy of the bias, {{=bias_copy_mib:.1f}} MiB here, on top of what the chunks need, and the low-memory path holds only the chunks, so its peak is {{=peak_views / peak_flat:.0%}} of the other's. At the real sizes the copy is {{=predicted_ynv:.1f}} GiB, while a chunk of $r$ rows needs $4\\,r\\,H\\,N^{2}$ bytes: for $r$ = {{=rows_c}} and $N$ = {{=n_ynv}} that is {{=chunk_mib_real:.0f}} MiB (an example of the order of size, not the chunk that the program used).
"""),
    code(FIX_FACTS),
    md("""
**What checks the change.** The self-check of the script builds a real triangle-attention layer of OpenFold3, at 24 tokens with random weights and in both directions (starting and ending node), computes it three ways, unchunked, chunked as it came and chunked by the script, and requires the script's result to equal the unchunked one to {{=tolerance_unchunked}} and the original chunking to {{=tolerance_original}}; it also requires that the call site of the layer has been replaced. Its last line in the log of job 149 reads: *{{=selfcheck_line}}*. A test of the repository (`tests/test_openfold3_lowmem.py`) runs the same check in the OpenFold3 environment. With the script in force the probe completed {{=probe_samples}} samples of {{=probe_ligand}} in {{=probe_seconds:.0f}} seconds, and the memory mode `{{=memory_mode}}` was written for every later run. What is not shown, and cannot be: that whole predictions are identical, because the program as it came cannot produce one for these inputs. The evidence is the equality on the layer where the memory was exhausted, which is the same function evaluated row by row; the other call sites of `chunk_layer` that the script changes are the same routine on other layers, and only the triangle attention was checked.

**The benchmark.** With the change in force, the ten ligands of the benchmark of 11.5 were predicted with three OpenFold3 samples each, and compared with the five Boltz-2 samples that each already had. The question of the pre-registered gate `engine_consensus` (11.8) has two parts: do the engines make different errors (the rank correlation of the per-ligand errors of their first samples must be below 0.7, so that a second opinion is a different opinion), and does minimum Bayes risk over both pools reach a lower mean RMSD than over the Boltz-2 pool alone, with the same safeguards as the other selection gates (the gain must survive dropping any one ligand, and a one-sided paired Wilcoxon p must be below 0.10)?
"""),
    code(BENCH),
    md("""
The engines do make different errors: the rank correlation of the errors of their first samples is {{=of3_gate['errors_correlation']:.2f}}, below 0.7, and OpenFold3's first sample is the closer of the two for {{=first_closer}} of the {{=len(of3_outcomes)}} ligands. But it is the worse engine on average, by {{=first_gap:.2f}} Å at the first sample, and over all samples {{=by_engine.loc['OpenFold3', 'mean']:.2f}} Å against {{=by_engine.loc['Boltz-2', 'mean']:.2f}} Å. The second part fails: minimum Bayes risk over both pools gives a mean RMSD of {{=mbr_both_mean:.3f}} Å and over the Boltz-2 pool alone {{=mbr_boltz_mean:.3f}} Å, so adding the engine makes the chosen pose worse by {{=-of3_gate['mean_improvement']:.3f}} Å on average, although in sign it helps more often than it harms: the choice changes for {{=n_changed}} of the {{=len(of3_outcomes)}} ligands, for the better in {{=n_better}} and for the worse in {{=n_worse}}. The Wilcoxon p is {{=of3_gate['p']:.3f}} and the worst drop-one mean {{=of3_gate['min_drop_one']:+.3f}} Å. The gate is NOT MET. With OpenStructure's yardstick (11.4) the figures are {{=ost_improvement:+.3f}} Å, {{=ost_wins}} wins, p {{=ost_p:.3f}} and a correlation of {{=ost_rho:.2f}}, and the verdict is the same.

Two things are visible in the tables. Minimum Bayes risk chooses the pose that agrees best with the others of the pool, so that every added pose is a vote, and the votes of the second engine are for poses that are mostly farther from the crystal pose: its first sample is the closer to it for only {{=first_closer}} of the ten ligands. And one ligand carries the verdict. For {{=worst_ligand['compound']}} the Boltz-2 pool alone chose a pose {{=worst_ligand['mbr_boltz']:.2f}} Å from the crystal pose (the first sample of that engine is {{=worst_ligand['first_boltz']:.2f}} Å away) and the pool of both engines one {{=worst_ligand['mbr_both']:.2f}} Å away, a loss of {{=worst_ligand['mbr_both'] - worst_ligand['mbr_boltz']:.2f}} Å on its own. A look made after the verdict, and so not a test: without that ligand the pool of both engines is the better by {{=rest_gain:.2f}} Å on average over the other {{=len(rest)}}, with p {{=rest_gate['p']:.2f}} and a worst drop-one mean of {{=rest_gate['min_drop_one']:+.2f}} Å, which {{=rest_verdict}} the registered rule. The rule was applied to the ten ligands as they are and its verdict stands, because choosing which ligand to leave out after seeing the result is the move that a gate written beforehand exists to prevent. What the look does show is how little the verdict rests on: one ligand turns it either way, so ten ligands cannot say that the second engine is harmful, only that it was not shown to help under the rule that was written before the result. The ceiling does rise when the pool grows (the best pose that exists in the pool, which no rule can apply, has a mean RMSD of {{=oracle_both:.2f}} Å with both engines against {{=oracle_boltz:.2f}} Å with Boltz-2 alone), so the second engine does produce good poses that Boltz-2 does not; what is missing is a way to tell which of them they are. The confidence of the engines cannot tell them apart: Boltz-2's is high even for poses several ångströms from the crystal pose (11.2), and the module that would have combined the engines records, as an external finding, that the confidence of both engines is an unreliable selector of poses for these enzymes.

**What follows.** The final recipe of the Structure Track therefore runs Boltz-2 alone (the recipe file records `openfold3 = {{=final_recipe_openfold}}` and the reason, "engine_consensus not MET"). Had the gate been met, the recipe would have added three OpenFold3 samples for each of the {{=n_test_ligands}} ligands at about 16 minutes each, {{=of3_extra_hours:.1f}} hours of machine time. The code stays in the repository, tested, as the record of what was run and as an engine that a later benchmark could re-test; it is not part of any submitted file.

**What limits the verdict.** Ten ligands, so a gate that can certify only a large difference (11.8 gives its size); a Wilcoxon threshold of 0.10 that was stated openly because 0.05 would have needed nearly every ligand to improve; one seed, the weights as released, no templates and no restraint, and alignments requested from a public server; and a yardstick, the RMSD to the crystal pose, that is not the organisers' score (the repetition with OpenStructure's scorer gave the same verdict). A different set of ligands, a restraint expressed in OpenFold3's own terms, or a selection rule that used something other than agreement could change the picture; none was tried.

*On what actually ran.* `cypmode/validation/openfold3.py` (`build_query`, `read_boltz_inputs`, `run_openfold3`, `run_openfold3_batch`, `default_memory_mode`, `failure_reason`, and the guard `_another_prediction_running`), `scripts/run_openfold_lowmem.py` (`patch_chunk_layer`, `self_check`), the runner file `data/structure_track/inputs/openfold3_runner.yml`, `cypmode/validation/selection_validation.py` (`gates`, `paired_gate`), `cypmode/validation/pose_consensus.py` (the rule that would have combined the engines), `cypmode/validation/final_recipe.py` (the constants of cost and the decision), the tests `tests/test_openfold3.py` and `tests/test_openfold3_lowmem.py`, and the queue jobs `149_openfold3_memory_probe`, `150_openfold3_benchmark`, `170_mbr_selection_validation`, `180_structure_final_recipe_run` and `245_ost_pool_gates`. The cells of this section read the files those runs left; the arrays of the two toys are written for the notebook. OpenFold3 is distributed under the Apache licence 2.0, and its weights under the terms stated with the checkpoint; the repository contains neither the package nor the weights.
"""),
    code(BIOEMU),
    md("""
**A third engine that was examined: BioEmu2.** After the benchmark of the second engine a third model was looked at, BioEmu2: a diffusion model of the kind described in 11.2 (from Microsoft Research, under the MIT licence, with about five hundred million parameters according to its model card) that builds an all-atom structure of a protein with ligands from the sequence, an alignment of related sequences and optional contacts. It is meant for Linux and an NVIDIA graphics card, and the configuration of its public checkpoint names one layer routine that exists only for those cards. Its code has a plain PyTorch version of the same layer, so a local copy of the configuration that names it, with the layers cut into chunks that lower the peak memory without changing the arithmetic, lets the model run on this machine; the weights are untouched (`cypmode/validation/bioemu2.py`, tested by `tests/test_bioemu2.py`). The inputs are those of Boltz-2: the sequence, the SMILES of the ligand, the heme and the bond of Cys442 to the iron, all read from the same input file and converted by the program's own converter, and the first rows of the alignment that Boltz-2 used, so that the engines differ in the model and not in what they are given. Contacts to the atoms of a ligand that is given as a SMILES string are not supported by the program, so none are passed.

**What the machine allows.** On a peptide of twenty residues ({{=bioemu_peptide_atoms}} atoms) two samples took {{=bioemu_cpu_s}} s on the processor and {{=bioemu_gpu_s}} s on the graphics chip. A benchmark system is much larger: the protein alone has {{=bioemu_residues}} residues, and with the ligand and the heme the model works on several hundred tokens. Two timing probes of that system were stopped by a memory watchdog while a graphics job of the queue was running beside them, at {{=bioemu_stops[0][0]}}% and {{=bioemu_stops[1][0]}}% free memory with {{=bioemu_stops[0][1]:.1f}} GB and {{=bioemu_stops[1][1]:.1f}} GB of swap in use (swap is the disk space that the operating system falls back on when memory is full, and it is slow); the second probe used the chunked layers and stopped all the same. Beside the queue the system does not fit, so the probe became a queue job, `259_bioemu2_probe`, that waits until the long graphics jobs have settled and then runs alone. {{=bioemu_sentence}}

**The rule that was written before the probe was read.** The benchmark of the engine is run only if its projected cost is within a limit: one trunk per ligand and the samples at the median speed of the probe, for the {{=bioemu_benchmark}} benchmark ligands and the {{=bioemu_official}} official ligands with {{=bioemu_samples}} samples each, at most {{=bioemu_limit_h:.0f}} hours of exclusive machine time. The log of the probe ends with the rule applied: {{=bioemu_rule_line}}. If it says NOT ALLOWED, the engine is recorded as not run, with the measurements above, and nothing else changes. If it says ALLOWED, the benchmark is declared in the checklist before any ligand is predicted, and its gate is the engine-consensus gate of this section with BioEmu2 as the added engine; the second engine did not pass that gate, so the prior for a third is not high.

*On what actually ran.* `cypmode/validation/bioemu2.py` (`prepare_checkpoint`, `pure_torch_config`, `msa_a3m`, `run`, `projected_hours`, `feasibility_line`) ran for the attempts and for the probe, and `259_bioemu2_probe` is the job that applies the rule. The file `exploratory_bioemu2_first_attempts.log` was assembled by hand from the logs of the five first attempts and its lines are the programs' own output; `exploratory_bioemu2_probe.log` is written by the job.
"""),
]
