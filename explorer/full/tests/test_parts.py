# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from simulators import (
    additions,
    ddi,
    heme,
    quantum,
    structure,
    validation,
)

PYTHON = HERE.parents[1] / ".venv-marimo/bin/python"


def test_the_ratio_of_exposures_follows_the_static_model_and_cannot_pass_one_over_one_minus_fm():
    assert ddi.aucr(0.8, 9.0) == pytest.approx(1 / (0.8 / 10 + 0.2))
    assert ddi.aucr(0.9, 1e9) == pytest.approx(10.0, rel=1e-6)
    assert ddi.aucr(0.5, 0.0) == pytest.approx(1.0)
    assert [ddi.classify(x) for x in (1.1, 1.3, 2.0, 5.0)] == ["no relevant", "weak", "moderate", "strong"]


def test_an_inactivated_enzyme_falls_to_the_steady_level_of_synthesis_against_loss():
    k_deg, k_obs = 0.02, 0.06
    level = ddi.enzyme_fraction([0.0, 1e6], k_deg, k_obs)
    assert level[0] == pytest.approx(1.0) and level[1] == pytest.approx(k_deg / (k_deg + k_obs))
    assert ddi.inactivation_rate(0.1, 5.0, 5.0) == pytest.approx(0.05)
    assert ddi.aucr(0.8, 0.0, 0.25) > ddi.aucr(0.8, 0.0, 1.0)


def test_the_cycle_has_nine_steps_and_the_difference_spectra_have_the_peak_where_they_say():
    assert len(heme.CYCLE) == 9 and heme.CYCLE[0][1].startswith("Fe(III), low spin") and heme.CYCLE[-1][1].startswith("Fe(III), low spin")
    w = np.arange(360.0, 471.0)
    type_ii, type_i = heme.difference_spectrum("Type II", 1.0, w), heme.difference_spectrum("Type I", 1.0, w)
    assert abs(w[type_ii.argmax()] - 430) <= 2 and abs(w[type_i.argmax()] - 390) <= 2 and w[type_ii.argmin()] < 400 < w[type_i.argmin()]


def test_a_titration_recovers_the_dissociation_constant():
    _points, k_fit, se = heme.titration(5.0, 0.1, 0.002, seed=3)
    assert abs(k_fit - 5.0) < 4 * se and se < 1.0


@pytest.mark.skipif(not PYTHON.exists(), reason="the explorer environment .venv-marimo is not on this machine")
def test_every_registered_simulator_makes_its_page_with_the_default_controls():
    run = subprocess.run([str(PYTHON), str(HERE / "tests/render_parts.py")], capture_output=True, text=True, check=False, cwd=HERE)
    result = json.loads(run.stdout.strip().splitlines()[-1])
    assert result and {k: v for k, v in result.items() if v != "ok"} == {}


def test_the_grid_reproduces_the_known_energies_of_the_box_and_the_oscillator():
    box = quantum.schrodinger_1d("particle in a box of width a", 2.0)
    assert np.allclose(box["energies"], box["exact"], rtol=1e-3)
    oscillator = quantum.schrodinger_1d("harmonic oscillator of frequency w", 2.0)
    assert np.allclose(oscillator["energies"], oscillator["exact"], rtol=2e-3)


def test_the_self_consistent_field_of_the_dimer_has_the_known_energies():
    restricted = quantum.hubbard_dimer_scf(1.0, 1.0, 0.2, 0.5)
    assert restricted["energy"] == pytest.approx(0.5 - 2.0) and restricted["exact"] == pytest.approx(0.5 - np.sqrt(0.25 + 4.0))
    broken = quantum.hubbard_dimer_scf(1.0, 6.0, 0.2, 0.5)
    assert broken["converged"] and broken["energy"] == pytest.approx(-2 * 1.0**2 / 6.0, abs=1e-6) and broken["energy"] > broken["exact"]


def test_the_exchange_of_the_hydrogen_density_and_the_pi_gap_of_benzene_are_the_textbook_values():
    h = quantum.hydrogenic(1.0)
    assert h["electrons"] == pytest.approx(1.0) and h["LDA exchange"] == pytest.approx(-0.2680, abs=5e-4) and h["E exact"] == pytest.approx(-0.5)
    assert quantum.hydrogenic(0.9555)["E LDA"] == pytest.approx(-0.4565, abs=5e-4)
    benzene = quantum.huckel("c1ccccc1")
    assert benzene["gap"] == pytest.approx(2.0) and benzene["electrons"] == 6 and benzene["atoms"] == 6


def test_the_distance_in_the_trimmed_pocket_file_is_the_one_of_the_validation_summary():
    import json

    summary = json.loads((HERE.parents[1] / "data/structure_track/outputs/cyp3a4/validation_summary.json").read_text())
    for name in ("ketoconazole", "ritonavir", "azamulin"):
        contact = structure.iron_nitrogen(structure.parse(structure.pdb_text(f"validation_{name}")))
        assert contact["distance"] == pytest.approx(min(r["distance_angstrom"] for r in summary[name]["all_ligand_nitrogen_distances"]), abs=0.011)


def test_an_rmsd_without_alignment_sees_a_shift_and_the_aligned_one_does_not():
    r = structure.rmsd_demo("CC(C)Cc1ccc(cc1)C(C)C(=O)O", 5.0, 90.0)
    assert r["in place"] > 3.0 and r["after the best rigid alignment"] < 2.0 and r["after alignment with the best match of symmetric atoms"] <= r["after the best rigid alignment"] + 1e-6
    assert structure.pair_tensor_memory(600, 128, 4, 8, 4)["total (GB)"] > structure.pair_tensor_memory(300, 128, 4, 8, 4)["total (GB)"] * 4


def test_the_gate_of_a_selected_population_helps_only_where_non_hits_are_scored_and_tukey_is_the_looser_test():
    none, many = additions.selection_auc(0.0), additions.selection_auc(0.8)
    assert many["multiplied by the gate P(hit)"] - many["fitted on the hits alone"] > none["multiplied by the gate P(hit)"] - none["fitted on the hits alone"]
    r = validation.false_alarm_rates(5, 10, 0.3, 80, 1)
    assert r["Tukey HSD"] > r["corrected t with Holm"]


def test_the_ledger_replay_names_the_families_it_shows(tmp_path):
    from simulators import experiments

    assert experiments.family_of("aux_neural_f10_CYP3A4.log") == "aux_neural" and experiments.family_of("confidence_CYP2D6_r3.log") == "confidence"
    assert set(experiments.verdicts(["aux_neural"])["stem"]) == {"aux_neural"} and len(experiments.comparisons(["aux_neural"], "CYP3A4")) <= len(experiments.comparisons(["aux_neural"]))


@pytest.mark.skipif(not PYTHON.exists(), reason="the explorer environment .venv-marimo is not on this machine")
def test_every_control_of_every_simulator_changes_the_page():
    run = subprocess.run([str(PYTHON), str(HERE / "tests/audit_controls.py")], capture_output=True, text=True, check=False, cwd=HERE)
    result = json.loads(run.stdout.strip().splitlines()[-1])
    bad = {k: v for k, v in result.items() if v not in ("changes the page", "not varied here")}
    assert not bad, bad
