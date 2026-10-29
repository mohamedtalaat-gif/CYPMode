# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The scores and the gate of the ligand dynamics in the frozen pocket: the drift, the within-pool ranks, the gate and the use test on synthetic pools, the lines of the log as the ledger reads them, and the split of the simulation over processes."""

import json

import numpy as np
import pandas as pd

from cypmode.audit.method_report import GATE
from cypmode.validation import ligand_dynamics as ld


def pools(perfect: bool, n_ligands: int = 6, n_poses: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_ligands):
        error = np.sort(rng.uniform(0.5, 6.0, n_poses))
        quality = -error if perfect else rng.normal(size=n_poses)
        for e, q in zip(error, quality, strict=True):
            rows.append({"compound": f"bench-L{k}", "true_rmsd": e, "mbr": rng.normal(), "field": q, "recall": q, "drift": -q, "static_field": rng.normal(), "static_recall": rng.normal(), "regime": "novel"})
    return pd.DataFrame(rows)


def test_the_drift_of_a_rigid_shift_is_its_length():
    start = np.zeros((4, 3))
    frames = np.stack([start + np.array([1.0, 0, 0]), start + np.array([0, 2.0, 0])])
    assert ld.drift(frames, start) == 1.5


def test_a_higher_field_and_recall_and_a_lower_drift_rank_higher():
    pool = pd.DataFrame({"field": [0.1, 0.5, 0.9], "recall": [0.2, 0.4, 0.8], "drift": [3.0, 2.0, 1.0]})
    out = ld.within_pool_dynamics(pool)
    assert out["dynamic_pharmacophore"].is_monotonic_increasing and out["stability"].is_monotonic_increasing and out["dynamic_combined"].idxmax() == 2


def test_a_score_that_orders_the_poses_by_their_error_passes_the_gate_and_the_use_test():
    scored, per_ligand = ld.evaluate(pools(perfect=True))
    lines = ld.report(per_ligand, "novel", True)
    verdicts = {g[0]: g[3] for g in GATE.findall("\n".join(lines))}
    assert verdicts["dynamics_dynamic_pharmacophore"] == "MET" and verdicts["dynamics_dynamic_pharmacophore_use"] == "MET"
    assert (per_ligand["dynamic_pharmacophore_pick"] <= per_ligand["mbr_pick"]).all() and len(scored) == 36


def test_a_score_without_information_does_not_pass_and_its_use_test_is_not_evaluated():
    _, per_ligand = ld.evaluate(pools(perfect=False, n_ligands=10, seed=3))
    lines = ld.report(per_ligand, "novel", True)
    text = "\n".join(lines)
    assert "dynamics_dynamic_pharmacophore (" in text and "NOT MET" in text
    assert "dynamics_stability_use: not evaluated" in text or "dynamics_stability_use (paired" in text


def test_the_lines_of_an_information_regime_carry_no_gate_line():
    _, per_ligand = ld.evaluate(pools(perfect=True))
    assert not [line for line in ld.report(per_ligand, "compound", False) if line.startswith(ld.LOG_GATE)]


def test_every_pose_is_simulated_by_exactly_one_shard(tmp_path, monkeypatch):
    jobs = [{"pdb": f"p{i}.pdb", "smiles": "C", "out": str(tmp_path / f"{i}.npz")} for i in range(10)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(jobs))
    seen = []
    monkeypatch.setattr(ld, "simulate_pose", lambda pdb, smiles, out, threads: seen.append(pdb.name) or {"pose": pdb.name})
    for shard in range(4):
        ld.simulate_all(manifest, (shard, 4), 1)
    assert sorted(seen) == sorted(j["pdb"] for j in jobs) and len(set(seen)) == 10
