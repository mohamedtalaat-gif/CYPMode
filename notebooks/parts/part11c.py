# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part XI (third part): the reference ligands and the crystal-anchored benchmark (11.5)."""

from notebooks.notebook_builder import code, md

REFERENCE_SET = '''
# The reference set (cypmode/validation/reference_ligands.py): every CYP3A4 entry of the Protein Data Bank (UniProt P08684) that holds a drug-like ligand in the active site, one record for each placement of a ligand, with the
# coordinates of the ligand, of the heme iron and of the C-alpha atoms of the pocket. The cell counts it, and asks three questions of it. (1) How do the ligands sit relative to the iron: the nearest ligand atom to the iron and its
# element. (2) Does the presence of a Type II motif (5.1) predict that the nitrogen binds the iron? (3) How far are the 20 ligands of the challenge from this set, and from the rest of the challenge data (the training and the blinded files)?
import json
from collections import defaultdict

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from cypmode.validation.build_input import heme_donor_atom

RDLogger.DisableLog("rdApp.*")
reference = json.loads((ROOT / "data/structure_track/inputs/cyp3a4_reference_ligands.json").read_text())
placements = reference["ligands"]
nearest = []
for record in placements:
    coords, fe = np.asarray(record["coords"], dtype=float), np.asarray(record["fe"], dtype=float).reshape(-1, 3)[0]
    distances = np.linalg.norm(coords - fe, axis=1)
    nearest.append({"comp_id": record["comp_id"], "smiles": record["smiles"], "pdb_id": record["pdb_id"], "distance": float(distances.min()), "element": record["elements"][int(distances.argmin())]})
nearest = pd.DataFrame(nearest)
distinct = nearest.sort_values("distance").groupby("comp_id", as_index=False).first()
distinct["coordinates through nitrogen"] = (distinct["distance"] <= 2.6) & (distinct["element"] == "N")
distinct["has a Type II donor"] = distinct["smiles"].map(lambda s: heme_donor_atom(s) is not None)
n_entries, n_placements, n_distinct = reference["n_entries"], len(placements), len(distinct)
share_coordinating = float(distinct["coordinates through nitrogen"].mean())
with_donor, without_donor = distinct[distinct["has a Type II donor"]], distinct[~distinct["has a Type II donor"]]
coordination_table = pd.DataFrame([
    {"ligands": "all distinct ligands", "number": n_distinct, "coordinate the iron through a nitrogen": float(distinct["coordinates through nitrogen"].mean())},
    {"ligands": "with a Type II donor in the structure", "number": len(with_donor), "coordinate the iron through a nitrogen": float(with_donor["coordinates through nitrogen"].mean())},
    {"ligands": "without one", "number": len(without_donor), "coordinate the iron through a nitrogen": float(without_donor["coordinates through nitrogen"].mean())}])
show_table(coordination_table, floatfmt=".2f")
median_resolution = float(np.median([r["resolution"] for r in placements if r.get("resolution")]))

generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fingerprint = lambda smiles: generator.GetFingerprint(Chem.MolFromSmiles(smiles))
final_recipe = json.loads((ROOT / "data/structure_track/outputs/cyp3a4/final/recipe.json").read_text())
official = [row["ligand"] for row in final_recipe["ligands"]]
official_smiles = {name: yaml.safe_load((ROOT / f"data/structure_track/inputs/cyp3a4_{name}.yaml").read_text())["sequences"][1]["ligand"]["smiles"] for name in official}
reference_fingerprints = [fingerprint(r["smiles"]) for r in placements]
best_to_reference = np.array([max(DataStructs.BulkTanimotoSimilarity(fingerprint(s), reference_fingerprints)) for s in official_smiles.values()])
challenge_smiles = list(pd.read_csv(ROOT / "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv")["SMILES"]) + list(pd.read_csv(ROOT / "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv")["SMILES"])
challenge_fingerprints = [fingerprint(s) for s in challenge_smiles]
best_in_challenge = []
for smiles in official_smiles.values():
    similarity = np.sort(np.array(DataStructs.BulkTanimotoSimilarity(fingerprint(smiles), challenge_fingerprints)))[::-1]
    best_in_challenge.append(similarity[1] if similarity[0] >= 0.999 else similarity[0])   # the compound itself is in the challenge files
best_in_challenge = np.array(best_in_challenge)
n_with_analog = int((best_in_challenge >= 0.5).sum())
n_official_donor = int(sum(row["type_ii_donor"] for row in final_recipe["ligands"]))
print(f"{n_entries} entries, {n_placements} placements, {n_distinct} distinct ligands; best similarity of an official ligand to the reference set: {best_to_reference.min():.2f} to {best_to_reference.max():.2f}; official ligands with an analog (>= 0.5) in the challenge data: {n_with_analog} of {len(official)}")
'''

BENCHMARK_UNGUIDED = '''
# The crystal-anchored benchmark and the first measurement on it. Ten ligands of the reference set were picked by a fixed rule (cypmode/validation/structure_benchmark.py): 20 to 60 heavy atoms, resolution at most 3.0 angstrom,
# organic, and the most mutually dissimilar (maximin on Morgan fingerprints, starting from the compounds already predicted, so that each new pick is far from the earlier ones). Unguided Boltz-2 (the base input of 11.3) drew five
# samples for each, one at a time. The table gives for each ligand the RMSD of the first sample to the crystal pose (pocket-superposed, symmetry-corrected, the best over the crystal copies), the best of the five, how far apart the
# five are, the model's own confidence in the first sample, and whether the first sample puts a nitrogen of a Type II motif on the iron.
from scipy import stats

benchmark_poses = pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/benchmark_engine_pose_validation_unguided9.csv")
benchmark_results = pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/structure_benchmark_results.csv").set_index("compound")
per_ligand = benchmark_poses.groupby("compound")["true_rmsd"].agg(first="first", best="min", spread="std", mean="mean")
per_ligand["confidence"] = benchmark_results.loc[per_ligand.index, "confidence_score"]
per_ligand["nitrogen on the iron"] = benchmark_results.loc[per_ligand.index, "coordinated"]
per_ligand["has a donor"] = benchmark_results.loc[per_ligand.index, "smiles"].map(lambda s: heme_donor_atom(s) is not None)
bench_table = per_ligand.reset_index().rename(columns={"compound": "ligand", "first": "first sample (A)", "best": "best of five (A)", "spread": "spread of the five (A, sd)", "mean": "mean of the five (A)", "confidence": "model confidence"})
show_table(bench_table.round(2))
n_bench = len(per_ligand)
first_mean, best_mean = float(per_ligand["first"].mean()), float(per_ligand["best"].mean())
n_first_under_2, n_best_under_2 = int((per_ligand["first"] < 2).sum()), int((per_ligand["best"] < 2).sum())
confidence_range = (float(per_ligand["confidence"].min()), float(per_ligand["confidence"].max()))
failures = per_ligand[per_ligand["first"] > 4]
confidence_on_failures = (float(failures["confidence"].min()), float(failures["confidence"].max()), len(failures))
confidence_rank = float(stats.spearmanr(per_ligand["confidence"], per_ligand["first"])[0])
donor_ligands = per_ligand[per_ligand["has a donor"]]
n_donor, n_donor_coordinated = len(donor_ligands), int(donor_ligands["nitrogen on the iron"].sum())
donor_coordinated_wrong = int(((donor_ligands["nitrogen on the iron"]) & (donor_ligands["first"] > 2)).sum())
'''

CELLS = [
    md("""
## 11.5 The reference ligands and the crystal-anchored benchmark

> **Field:** Structural biology · Statistics

*Where this section sits:* 11.4 defined how a pose is scored against a crystal pose; this section says where the crystal poses come from, how far the 20 official ligands are from them, and what the unguided model does on a benchmark made from them. Every verdict of 11.6 to 11.9 is a comparison on this benchmark, so what it can and cannot show is stated here.

**The reference set.** The Protein Data Bank holds structures of CYP3A4 with ligands in the pocket. `cypmode/validation/reference_ligands.py` collects them from the archive (UniProt P08684) and keeps a ligand only if it is an inhibitor-sized molecule in the active site, that is, if its nearest heavy atom lies within a set distance of a heme iron; the peripheral site where CYP3A4 also binds some molecules is left out, as are buffers, ions and detergents. For each ligand it stores the coordinates, the iron and the C-alpha atoms of the pocket, so that the ligand can be placed in the frame of any predicted complex. The cell reads the set and asks how the ligands sit relative to the iron.
"""),
    code(REFERENCE_SET),
    md("""
The set holds {{=n_placements}} placements of {{=n_distinct}} distinct ligands in {{=n_entries}} entries, at a median resolution of {{=median_resolution:.2f}} Å. {{=share_coordinating:.0%}} of the distinct ligands bind the iron through a nitrogen (the nearest ligand atom is a nitrogen within 2.6 Å of the iron). The presence of a Type II donor in the formula predicts it: {{=coordination_table.iloc[1, 2]:.0%}} of the {{=coordination_table.iloc[1, 1]}} ligands that have a donor coordinate, and none of the {{=coordination_table.iloc[2, 1]}} that have not. This is the empirical basis of the restraint of 11.6: a nitrogen that can bind the iron usually does.

**How far the 20 are.** The 20 official ligands are not in the set and are not near it: the best Tanimoto similarity (Morgan fingerprints of radius 2, 2,048 bits, 3.2) of an official ligand to any ligand of the set lies between {{=best_to_reference.min():.2f}} and {{=best_to_reference.max():.2f}}, so that no deposited structure is a good guide to any of them. They are not alone in the world, though: {{=n_with_analog}} of the 20 have an analogue (similarity of at least 0.5) among the compounds of the challenge's own files, which have measurements and no structures. {{=n_official_donor}} of the 20 have a Type II donor. For half of the official ligands, then, the iron-binding nitrogen is predictable from the formula, and for the other half it is not.

**The benchmark.** Since the 20 have no answer key, a benchmark is built from the set: ten ligands chosen by a rule fixed in advance (`cypmode/validation/structure_benchmark.py`): 20 to 60 heavy atoms, resolution of at most 3.0 Å, organic, one fragment, and the most mutually dissimilar by a maximin pick on the fingerprints, starting from the five compounds already predicted so that each pick is far from them and from each other. The rule is deterministic and the list is not hand-written. One caveat is kept visible: the models may have seen these entries in training, so that the absolute accuracy on the benchmark is optimistic. The benchmark is for *comparing* settings and selection rules, not for forecasting a score. Unguided Boltz-2 drew five samples for each ligand, one at a time (about 40 minutes per ligand); nine of the ten ligands were scored. The table gives the result.
"""),
    code(BENCHMARK_UNGUIDED),
    md("""
**What the model does without help.** The first sample is {{=first_mean:.2f}} Å from the crystal pose on average, and only {{=n_first_under_2}} of the {{=n_bench}} are within 2 Å. The best of the five is {{=best_mean:.2f}} Å away and {{=n_best_under_2}} of {{=n_bench}} are within 2 Å: drawing five samples buys almost nothing, because the five are close to each other (the spread of a ligand's five RMSD values is a few tenths of an ångström for most). The model's own confidence is no guide: it lies between {{=confidence_range[0]:.2f}} and {{=confidence_range[1]:.2f}} for every ligand, and between {{=confidence_on_failures[0]:.2f}} and {{=confidence_on_failures[1]:.2f}} for the {{=confidence_on_failures[2]}} ligands whose first sample is more than 4 Å from the crystal; its rank correlation with the error is {{=confidence_rank:.2f}} over the {{=n_bench}} ligands. And the iron anchor is not the bottleneck: of the {{=n_donor}} benchmark ligands with a donor, {{=n_donor_coordinated}} already have the donor nitrogen on the iron in the first sample, and {{=donor_coordinated_wrong}} of those {{=n_donor_coordinated}} are still more than 2 Å from the crystal pose, because the rest of the molecule is wrongly placed. These four observations frame the sections that follow. A lever that forces the nitrogen onto the iron (11.6) can help only the ligands where the model does not already do it; a lever that makes the samples differ (11.7) has to widen a pool that is narrow; and a rule for choosing among poses (11.8) can only choose among poses that differ, and cannot use the confidence.

*On what actually ran.* `cypmode/validation/reference_ligands.py` (`build_reference_set`, `load_reference_set`; the raw mmCIF files are cached in `data/structure_track/inputs/pdb_cache`, regenerable, and the compact `cyp3a4_reference_ligands.json` is what the repository tracks), `cypmode/validation/structure_benchmark.py` (`candidates`, `pick_diverse`), `cypmode/validation/structure_benchmark_report.py` (the report that compares engines and selection rules on the benchmark), and the Boltz-2 runs of queue job `100_boltz_unguided_resume`. The tables are `benchmark_engine_pose_validation_unguided9.csv` and `structure_benchmark_results.csv`. Scripts: `scripts/run_structure_benchmark.sh` runs the benchmark end to end and `scripts/clear_batch_errors.py` clears the failed rows of a batch so that a resumed run retries them.
"""),
]
