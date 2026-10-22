# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part XI (fifth part): the diversity levers that did not pass (11.7)."""

from notebooks.notebook_builder import code, md

POCKET_RIGIDITY = '''
# How different are the pockets of the crystal structures? If the C-alpha atoms of the pocket sit nearly in the same place in every structure, a different crystal template cannot change the prediction much. For each pair of the
# entries of the reference set the C-alpha atoms of the pocket residues that both entries have (69 residues in the first) are superposed with the Kabsch algorithm of 11.4 and the RMSD is taken.
pockets = {}
for record in placements:
    pockets.setdefault(record["pdb_id"], record["pocket_ca"])
entry_ids = list(pockets)
pair_rmsd = []
for i in range(len(entry_ids)):
    for j in range(i + 1, len(entry_ids)):
        a, b = pockets[entry_ids[i]], pockets[entry_ids[j]]
        common = sorted(set(a) & set(b), key=int)
        if len(common) < 20:
            continue
        x, y = np.array([a[c] for c in common]), np.array([b[c] for c in common])
        pair_rmsd.append(rmsd(kabsch(x, y)(x), y))
pair_rmsd = np.array(pair_rmsd)
pocket_median, pocket_p95, pocket_max = float(np.median(pair_rmsd)), float(np.percentile(pair_rmsd, 95)), float(pair_rmsd.max())
n_pockets, n_pocket_residues, n_pocket_pairs = len(entry_ids), len(next(iter(pockets.values()))), len(pair_rmsd)
print(f"{n_pockets} entries, {n_pocket_pairs} pairs: median pocket C-alpha RMSD {pocket_median:.2f} A, 95th percentile {pocket_p95:.2f}, largest {pocket_max:.2f}")
'''

PILOT_TABLE = '''
# The pilot of the diversity levers (cypmode/validation/diversity_pilot.py). Three benchmark ligands whose first sample is wrong (bench-A1CIW 6.3 A, bench-PK9 4.8, bench-TPF 3.6) were run with four levers: three different random
# seeds (a new pass of the trunk for each), three different crystal conformations of the pocket as the protein template, three cryo-electron-microscopy structures as the template, and a step scale of 0.8 (three samples).
# A lever "creates diversity" for a ligand when the spread of its poses (the mean pairwise ligand RMSD in the pool) is at least 1.5 times that of the unguided pool, and "reaches the right pose" when the best pose in its pool is at
# least 0.75 angstrom closer to the crystal than the best of the unguided pool; the lever passes if both hold for at least half of the ligands. Both thresholds were fixed before the runs. The pools are read from the log of the gate.
from cypmode.validation.diversity_pilot import DIVERSITY_FACTOR, GAIN_A, verdicts

import re

gate_text = (ROOT / "data/structure_track/outputs/cyp3a4/logs/structure_gates.log").read_text()
pool_rows = re.findall(r"^\\s*(bench-\\S+)\\s+(seeds|conformers|cryoem)\\s+(\\d+)\\s+([\\d.]+)\\s+([\\d.]+)\\s+([\\d.]+)\\s+([\\d.]+)\\s*$", gate_text, re.M)
pools = pd.DataFrame(pool_rows, columns=["ligand", "lever", "poses", "spread", "unguided_spread", "best_rmsd", "unguided_best"]).astype({"poses": int, "spread": float, "unguided_spread": float, "best_rmsd": float, "unguided_best": float})
pools["wider than 1.5 x"] = pools["spread"] >= DIVERSITY_FACTOR * pools["unguided_spread"]
pools["best pose better by 0.75 A"] = pools["unguided_best"] - pools["best_rmsd"] >= GAIN_A
show_table(pools.round(2))
pilot_verdicts = verdicts(pools).set_index("lever")
step_scale_poses = pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/diversity_pilot_poses.csv")
step_scale_poses = step_scale_poses[step_scale_poses["compound"].str.endswith("-ss")]
n_step_scale, n_step_scale_unusable = len(step_scale_poses), int(step_scale_poses["true_rmsd"].isna().sum())
pilot_gate_text = {lever: ("MET" if re.search(rf"STRUCTURE GATE pilot_{lever}: MET", gate_text) else "NOT MET") for lever in ("seeds", "conformers", "cryoem")}
'''

CELLS = [
    md("""
## 11.7 Levers that did not pass: seeds, crystal templates, cryo-EM templates and the step scale

> **Field:** Structural biology · Statistics

*Where this section sits:* 11.5 found that the five samples of one prediction agree with each other, so that choosing among them cannot rescue a wrong pose; if a selection rule (11.8) is to have anything to choose from, the pool has to differ. This section tests four ways to make it differ, on a pilot of three ligands and a rule fixed before the runs. None passed.

**The four levers.** (1) **Seeds.** The same input with different random seeds. A new seed gives a new pass of the trunk, as in AlphaFold 3, so the diversity should be more than that of five samples from one pass. (2) **Crystal templates.** A template (11.3) is the protein structure of an experiment, given to the model as a soft guide; CYP3A4 can be given a different crystal conformation of its pocket, chosen from the reference set by the spread of the pocket's C-alpha atoms (`cypmode/validation/conformers.py`), and never from an entry whose ligand is the compound or a close analogue of it (Tanimoto similarity of at least 0.5): a template must not carry the answer. (3) **Cryo-EM templates.** The five cryo-EM structures of human CYP3A4 released on 2026-09-30 (the entries 38FU, 38FV, 38FW, 38FX and 38FY: apo, ritonavir, ketoconazole, azamulin, vardenafil; 3.2 to 3.7 Å; trimers, of which chain A serves as a one-chain template) add pocket shapes that the X-ray set may lack (`cypmode/validation/cryoem_templates.py`). (4) **The step scale.** A value of 0.8 instead of 1.5 (11.2), where the toy suggested that the samples would spread.

The reason to doubt the second and third lever is a number that the reference set can give. The cell superposes the pocket C-alpha atoms of every pair of the crystal entries and measures how much the pocket moves.
"""),
    code(POCKET_RIGIDITY),
    md("""
Across {{=n_pockets}} entries ({{=n_pocket_pairs:,}} pairs, {{=n_pocket_residues}} pocket residues) the pocket's C-alpha atoms are {{=pocket_median:.2f}} Å apart at the median, {{=pocket_p95:.2f}} Å at the 95th percentile and {{=pocket_max:.2f}} Å at most. The backbone of the pocket is almost the same in every structure, whatever ligand sits in it; a template can change the side chains and some loops but not the shape of the cavity by much. That does not prove that templates are useless, since small changes can move a ligand, but it sets the expectation.

**The pilot.** The rule, fixed in `cypmode/validation/diversity_pilot.py` before any run: a lever *creates diversity* for a ligand if the spread of its poses (the mean pairwise ligand RMSD within the pool, after superposing the pockets) is at least 1.5 times the spread of the unguided pool, and *reaches the right pose* if the best pose in its pool is at least 0.75 Å closer to the crystal pose than the best of the unguided pool; a lever passes if both hold for at least half of the ligands. Three ligands whose unguided first sample is wrong were used, `bench-A1CIW` (6.3 Å), `bench-PK9` (4.8) and `bench-TPF` (3.6), with three runs of each lever per ligand.
"""),
    code(PILOT_TABLE),
    md("""
The verdicts are **NOT MET** for all three levers that could be evaluated: seeds (the pool is wider for {{=pilot_verdicts.loc['seeds', 'ligands_with_wider_pool']:.0%}} of the ligands and the best pose better for {{=pilot_verdicts.loc['seeds', 'ligands_with_better_best_pose']:.0%}}), crystal templates ({{=pilot_verdicts.loc['conformers', 'ligands_with_wider_pool']:.0%}} and {{=pilot_verdicts.loc['conformers', 'ligands_with_better_best_pose']:.0%}}) and cryo-EM templates ({{=pilot_verdicts.loc['cryoem', 'ligands_with_wider_pool']:.0%}} and {{=pilot_verdicts.loc['cryoem', 'ligands_with_better_best_pose']:.0%}}); a lever needed two of the three ligands. The table shows what happens. For `bench-A1CIW` and `bench-PK9` the lever pools are *not* wider than the unguided pool, which was already broad for them (spreads of 4.2 and 3.4 Å): the unguided samples there were different but wrong, and the levers gave different but wrong poses too. For `bench-TPF` the unguided pool was unusually tight (0.9 Å) and every lever widens it (3.3 to 3.6 Å), and one of them, the crystal templates, finds a pose closer to the crystal (2.3 Å against 3.6) by drawing from a wider pool. That is one ligand of three, and a pool wide enough to include a better pose will usually include one if it is wide enough; whether the better pose can be *chosen* is the question of 11.8.

The step-scale lever could not be evaluated at all: all {{=n_step_scale_unusable}} of the {{=n_step_scale}} poses made at a step scale of 0.8 were unusable, because RDKit finds no match between the predicted ligand and its SMILES, so that neither the RMSD to the crystal nor the PoseBusters checks can be computed. A ligand whose geometry cannot be matched to its own formula is not a pose that anyone should submit. This is consistent with what the sampler of 11.2 does at a step scale below 1, which leaves residual noise in the coordinates (in the toy, 0.8 gives samples {{=spread_at[0.8] / true_spread:.1f}} times wider than the true spread); it was not tested further. The recipe does not use it.

The consequence for the recipe is that no diversity lever is used: the pools of 11.10 are made of repeated unguided samples, restrained ones and, where the budget allows, extra samples of the same input.

*On what actually ran.* `cypmode/validation/diversity_pilot.py` (`build_pilot_table`, `pool_spread`, `verdicts`; the thresholds), `cypmode/validation/conformers.py` (`entry_pockets`, `excluded_entries`, `pick_conformers`), `cypmode/validation/conformer_benchmark.py` (the batch input of the template experiment on the whole benchmark), `cypmode/validation/cryoem_templates.py` (`fetch`, the three most different entries per ligand), `cypmode/validation/structure_gates.py` (the verdict lines) and `scripts/run_diversity_pilot.sh` (which waits for the OpenFold3 stage, pauses the main queue, runs the pilot and restarts it). The queue jobs are `130_boltz_diversity_pilot`, `135_boltz_cryoem_pilot`, `140_score_restraint_and_pilot`, `144_score_cryoem_pilot` and `165_structure_gate_verdicts`.
"""),
]
