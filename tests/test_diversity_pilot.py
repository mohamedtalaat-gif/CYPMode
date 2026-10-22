# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pandas as pd

from cypmode.validation.diversity_pilot import (
    GAIN_A,
    SAMPLES,
    STEP_SCALE,
    build_pilot_table,
    lever_of,
    ligand_of,
    verdicts,
)

LIGANDS = pd.DataFrame({"Molecule_Name": ["bench-AAA"], "SMILES": ["c1ccncc1C"]})


def _refs(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    base = {str(r): rng.normal(scale=8.0, size=3).tolist() for r in range(100, 140)}
    refs = []
    for i, pdb in enumerate(["P1", "P2", "P3", "P4"]):
        refs.append({"pdb_id": pdb, "resolution": 2.0, "smiles": "CCCCCCCCCCCC(=O)OCC", "pocket_ca": {k: [x + i * 0.3 * (int(k) % 2), y, z] for k, (x, y, z) in base.items()}})
        (tmp_path / f"{pdb}.cif").write_text("data_x\n")
    monkeypatch.setattr("cypmode.validation.conformers.CACHE_DIR", tmp_path)
    monkeypatch.setattr("cypmode.validation.conformers.pick_conformers.__defaults__", (frozenset(), tmp_path))
    return refs


def test_pilot_table_has_one_row_per_run_with_the_options_each_lever_needs(tmp_path, monkeypatch):
    table = build_pilot_table(LIGANDS, _refs(tmp_path, monkeypatch), n_runs=3)
    seeds = table[table.Molecule_Name.str.contains("-seed")]
    assert len(seeds) == 3 and seeds.Seed.nunique() == 3 and (seeds.Diffusion_Samples == 1).all()
    conformers = table[table.Molecule_Name.str.contains("-cf")]
    assert len(conformers) == 3 and conformers.Template_PDB.nunique() == 3
    step = table[table.Molecule_Name.str.endswith("-ss")].iloc[0]
    assert step.Step_Scale == STEP_SCALE and step.Diffusion_Samples == SAMPLES


def test_names_map_back_to_ligand_and_lever():
    assert [lever_of(n) for n in ("bench-A-seed2", "bench-A-cf1", "bench-A-ss", "bench-A-em3")] == ["seeds", "conformers", "stepscale", "cryoem"]
    assert ligand_of("bench-A1CIW-seed2") == "bench-A1CIW"


def test_verdict_needs_a_wider_pool_and_a_better_best_pose_for_enough_ligands():
    pools = pd.DataFrame(
        [
            # seeds: wider and closer for both ligands
            {"ligand": "a", "lever": "seeds", "spread": 3.0, "unguided_spread": 1.0, "best_rmsd": 2.0, "unguided_best": 4.0},
            {"ligand": "b", "lever": "seeds", "spread": 3.0, "unguided_spread": 1.0, "best_rmsd": 3.0, "unguided_best": 3.0 + GAIN_A},
            # conformers: wider but the best pose is no better
            {"ligand": "a", "lever": "conformers", "spread": 3.0, "unguided_spread": 1.0, "best_rmsd": 4.0, "unguided_best": 4.0},
            {"ligand": "b", "lever": "conformers", "spread": 3.0, "unguided_spread": 1.0, "best_rmsd": 3.0, "unguided_best": 3.0},
        ]
    )
    result = verdicts(pools).set_index("lever")
    assert bool(result.loc["seeds", "creates_diversity"]) and bool(result.loc["seeds", "reaches_right_pose"])
    assert bool(result.loc["conformers", "creates_diversity"]) and not bool(result.loc["conformers", "reaches_right_pose"])
