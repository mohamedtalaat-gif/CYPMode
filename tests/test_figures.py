# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import sys
from pathlib import Path

import numpy as np
import pytest

FIGURES = Path("docs/figures")
sys.path[:0] = [str(FIGURES), str(FIGURES.parent.parent)]

from catalog import REGISTRY, Figure, caption_text, figure


def needs(path: str):
    """Skip a test whose input is written by the chain or held back until the challenge closes, so that a clean checkout skips it and the machine of the build runs it."""
    return pytest.mark.skipif(not Path(path).exists(), reason=f"{path} is not in a clean checkout")


FINAL = "data/tdi_track/outputs/final/manifest.json"

pytestmark = pytest.mark.skipif(not Path("data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv").exists(), reason="the training data is not in this checkout")


def test_a_figure_id_can_only_be_registered_once():
    figure("test_only_unique", "test", "t", "c")(lambda theme: None)
    try:
        with pytest.raises(ValueError):
            figure("test_only_unique", "test", "t", "c")(lambda theme: None)
    finally:
        REGISTRY.pop("test_only_unique", None)


def test_a_caption_can_be_computed_from_the_data():
    fig = Figure("x", "g", "t", lambda: "from data", lambda theme: None)
    assert caption_text(fig) == "from data"
    assert caption_text(Figure("y", "g", "t", "plain", lambda theme: None)) == "plain"


def test_files_of_a_figure_are_attributed_to_its_id():
    from build_all import figure_id_of

    assert figure_id_of(Path("kin_x-light.svg")) == "kin_x"
    assert figure_id_of(Path("kin_x-dark.svg")) == "kin_x"
    assert figure_id_of(Path("kin_x-anim-light.gif")) == "kin_x"
    assert figure_id_of(Path("kin_x-anim-dark.gif")) == "kin_x"
    assert figure_id_of(Path("kin_x-light.png")) == "kin_x"
    assert figure_id_of(Path("kin_x-dark.png")) == "kin_x"


@pytest.mark.parametrize("isoform", ["CYP3A4", "CYP2D6"])
def test_the_label_rule_drawn_in_the_figure_reproduces_every_label(isoform):
    from gen.data import LOG2, labelled_arms

    arms = labelled_arms(isoform)
    assert len(arms) > 1000
    assert ((arms["shift"] > LOG2) == arms["label"]).all()


def test_the_integrated_inactivation_curves_follow_the_closed_form_rate():
    from gen.kinetics import K_I, K_INACT, integrate, k_obs_of

    for inhibitor in (5.0, 20.0, 100.0):
        t, activity = integrate(inhibitor)
        window = (t > 2) & (activity > 2e-3)
        slope = -np.polyfit(t[window], np.log(activity[window]), 1)[0]
        assert slope == pytest.approx(k_obs_of(inhibitor), rel=0.03)
    assert k_obs_of(K_I) == pytest.approx(K_INACT / 2)


def test_the_memory_formula_gives_the_size_the_runtime_refused():
    from gen.structure import allocation_gib

    assert allocation_gib(592) == pytest.approx(9.27, abs=0.005)  # bench-YNV: 503 residues + 46 ligand atoms + 43 heme atoms; the log says 9.27 GiB
    assert allocation_gib(592, 128) < allocation_gib(592) / 4


# -- the concept and result figures added on 2026-10-26 (gen/ml_more.py, gen/structure_more.py, gen/data_more.py) -----------------------------------------

def test_the_backward_pass_of_the_network_figure_matches_finite_differences():
    from gen.ml_more import trained_network

    net = trained_network()
    assert net["worst_gradient_error"] < 1e-6
    assert net["losses"][-1] < net["losses"][0]


def test_the_tanimoto_similarity_is_a_positive_semidefinite_kernel_on_real_compounds():
    from gen.ml_more import kernel_study

    study = kernel_study()
    assert study["eigenvalues"][-1] > -1e-8
    assert 0.5 < study["svm_auc"] < 1.0


def test_the_attention_rows_sum_to_one_and_the_scaling_flattens_them():
    from gen.ml_more import attention_weights, tokens

    scaled, unscaled = attention_weights(), attention_weights(scaled=False)
    assert scaled.shape == (len(tokens()), len(tokens()))
    assert np.allclose(scaled.sum(axis=1), 1.0) and np.allclose(unscaled.sum(axis=1), 1.0)
    assert unscaled.max(axis=1).mean() > scaled.max(axis=1).mean()


def test_a_message_reaches_exactly_the_atoms_within_k_bonds():
    from gen.ml_more import drawn_molecule

    mol, _coords, distances = drawn_molecule()
    donor = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.GetIsAromatic() and a.GetDegree() == 2)
    reached = [int((distances[donor] <= k).sum()) for k in range(4)]
    assert reached[0] == 1 and reached == sorted(reached) and reached[3] < mol.GetNumAtoms()


@needs(FINAL)
def test_the_confusion_matrix_of_the_default_stack_has_the_mcc_of_the_final_build_closely():
    from gen.ml_more import confusion_counts, table_mcc

    for isoform in ("CYP3A4", "CYP2D6"):
        c = confusion_counts(isoform)
        assert abs(c["tp"] + c["fp"] + c["fn"] + c["tn"] - 1.0) < 1e-9
        assert abs(c["mcc"] - table_mcc(isoform)) < 0.04  # the build transfers the rate inside each fold; the figure calls the same top share


@needs("data/tdi_track/outputs/caches/feature_store.npz")
def test_the_principal_components_explain_less_variance_one_after_the_other():
    from gen.ml_more import component_study

    explained = component_study()["explained"]
    assert (np.diff(explained) <= 1e-12).all() and explained.sum() < 1.0


def test_the_kabsch_fit_recovers_a_rotation_and_a_shift():
    from gen.structure_more import kabsch

    rng = np.random.default_rng(0)
    fixed = rng.normal(size=(12, 3))
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    moving = (fixed - np.array([1.0, -2.0, 0.5])) @ rotation.T
    r, t = kabsch(moving, fixed)
    assert np.allclose(moving @ r + t, fixed, atol=1e-9)


def test_the_lddt_pairs_of_the_figure_score_what_the_project_scores():
    from gen.structure_more import lddt_pairs

    from cypmode.validation.pose_pools import symmetric_lddt_pli, template_automorphisms

    rng = np.random.default_rng(1)
    reference_ligand = rng.normal(0, 1.5, (6, 3))
    protein = {(i, "CA"): rng.normal(0, 4, 3) for i in range(60)}
    model_ligand = reference_ligand + rng.normal(0, 0.4, reference_ligand.shape)
    model_protein = {k: v + rng.normal(0, 0.3, 3) for k, v in protein.items()}
    permutations = template_automorphisms("c1ccccc1")
    score, reference_d, model_d = lddt_pairs(model_ligand, model_protein, reference_ligand, protein, permutations)
    assert abs(score - symmetric_lddt_pli(model_ligand, model_protein, reference_ligand, protein, permutations)) < 1e-12
    assert len(reference_d) == len(model_d) > 0


def test_the_triangle_update_does_not_depend_on_the_order_of_the_positions():
    from gen.structure_more import triangle_state

    d, _z, _update, equivariance = triangle_state()
    assert equivariance < 1e-9
    assert d[0, 1] <= d[0, 2] + d[2, 1] + 1e-9  # the triangle inequality of the real distances


def test_the_pose_pool_figures_read_the_same_numbers_as_the_selection_validation():
    from gen.structure_more import pool_caption

    assert "10 crystal-anchored benchmark ligands" in pool_caption()


@needs("data/tdi_track/inputs/official_extra/cyp_challenge/cyp-challenge-single-concentration-TRAIN.csv")
def test_the_screen_decides_which_cyp2d6_compounds_reached_the_tdi_assay():
    from gen.data_more import gate_study

    study = gate_study()
    assert study["CYP2D6"]["rho"] < -0.7 and study["CYP2D6"]["max_assayed"] < study["CYP2D6"]["min_not"] + 0.2
    assert abs(study["CYP3A4"]["rho"]) < 0.3


def test_the_compounds_without_a_direct_curve_are_all_negative():
    from gen.data_more import nodirect_study

    study = nodirect_study()
    assert study["CYP3A4"]["none"] > 1000 and study["CYP3A4"]["rate_none"] == 0.0 and study["CYP3A4"]["rate_both"] > 0.2


def test_the_negative_control_and_the_stage_rule_are_read_from_their_logs(tmp_path):
    from gen.data_more import CONTROL, STAGE, control_rows, stage_rows

    line = "  z0=1.0  real labels: AUC 0.8007, MCC +0.4111 | permuted: AUC 0.5096 (range 0.4802 to 0.5351), MCC +0.0020 (range -0.0376 to +0.0287)"
    assert CONTROL.search(line).groups()[:6] == ("1.0", "0.8007", "+0.4111", "0.5096", "0.4802", "0.5351")
    stage = "  z0=1.0  stacked base+added vs base: AUC +0.0035 (p=0.640, better 4/5) AP +0.0006 (p=0.975, better 3/5)"
    assert STAGE.search(stage).groups() == ("+0.0035", "0.640", "4", "5")
    controls = control_rows()
    assert (controls["permuted"] - 0.5).abs().max() < 0.05 and (controls["real"] > 0.7).all()
    assert [bool(r["continue"]) for r in stage_rows()] == [False, True]


def test_the_pretrained_models_that_joined_pass_at_both_cuts_and_the_others_do_not():
    from gen.data_more import pretrained_rows

    table = pretrained_rows()
    both = {(idea, iso) for (idea, iso), part in table.groupby(["idea", "enzyme"]) if ((part["p"] < 0.05) & (part["auc_gain"] > 0)).sum() == 2}
    assert both == {("Chemprop from the two OpenADMET pretraining weights (staged)", "CYP3A4"), ("the neural pair (multitask network and plain Chemprop) in the stack", "CYP3A4")}  # the two groups of the final CYP3A4 stack


# -- the overview figures (gen/overview_more.py) -------------------------------------------------------------------------------------------------------

def test_the_ledger_heat_map_marks_the_two_groups_of_the_final_stack_as_met_on_cyp3a4_only():
    from gen.overview_more import ledger_matrix

    rows, columns, cells, kinds = ledger_matrix()
    assert columns == ["CYP3A4", "CYP2D6", "structure"] and cells.shape == (len(rows), 3) and len(kinds) == len(rows) and set(np.unique(cells)) <= {0, 1, 2}
    neural = rows.index("the neural pair (multitask network and plain Chemprop) in the stack")
    foundation = rows.index("Chemprop from the two OpenADMET pretraining weights (staged)")
    assert cells[neural, 0] == 2 and cells[neural, 1] == 1 and cells[foundation, 0] == 2 and cells[foundation, 1] == 0


@needs(FINAL)
def test_the_final_results_page_reads_the_manifest_of_the_final_build():
    import json

    from gen.overview_more import final_tables

    manifest = json.loads(Path("data/tdi_track/outputs/final/manifest.json").read_text())["isoforms"]
    tables = final_tables()
    for iso, row in manifest.items():
        assert tables[iso]["name"] == row["stack"] and tables[iso]["n"] == row["n_candidates"]
        assert tables[iso]["final"]["auc_z1.0"] >= tables[iso]["default"]["auc_z1.0"] - 1e-9


def test_the_butina_clusters_cover_every_compound_once():
    from gen.overview_more import butina_study

    study = butina_study()
    assert int(study["sizes"].sum()) == study["n"] and study["sizes"].max() > 5


def test_the_probability_of_a_positive_label_falls_with_the_distance_of_the_shift_from_the_threshold():
    from gen.overview_more import label_gaussians

    table = label_gaussians()
    assert abs(table["p_positive"].iloc[0] - 0.5) < 0.1 and table["p_positive"].is_monotonic_decreasing and table["p_positive"].iloc[-1] < 0.01


def test_the_two_arms_of_one_enzyme_correlate_more_than_any_two_enzymes():
    from gen.overview_more import ARM_ISOFORMS, arm_correlations

    _labels, matrix, _counts = arm_correlations()
    n = len(ARM_ISOFORMS)
    assert np.allclose(matrix, matrix.T, equal_nan=True)
    assert min(matrix[i, i + n] for i in range(n)) > 0.85
    assert abs(matrix[ARM_ISOFORMS.index("CYP2D6"), ARM_ISOFORMS.index("CYP3A4")]) < 0.2


def test_the_logs_of_the_calibration_and_of_the_selection_models_are_read():
    from gen.overview_more import (
        CALIBRATION,
        DEPLOY,
        SENSITIVITY,
        calibration_rows,
        selection_tables,
    )

    line = " CYP3A4 isotonic 0.0189 0.1726  0.9252     0.0048     0.3120    0.3200      True"
    assert CALIBRATION.search(line).groups()[:3] == ("CYP3A4", "isotonic", "0.0189")
    sensitivity = "  z0=1.0  non-hit weight 0.25 | S0 AUC 0.606 MCC +0.088 | S1 AUC 0.535 MCC +0.050 | S2 AUC 0.551 MCC +0.041 | S3 AUC 0.565 MCC +0.062"
    assert SENSITIVITY.search(sensitivity).group(3) == "0.606"
    assert DEPLOY.search("    w=0.25 hits_ 0.598|+0.069  selec 0.651|+0.078  blend 0.655|+0.101   MCC").groups()[:2] == ("0.25", "0.598")
    table = calibration_rows()
    assert len(table) == 6 and table[table["isoform"] == "CYP2D6"].set_index("method").loc["isotonic", "met"]
    sensitivity_table, deploy = selection_tables()
    assert len(deploy) == 7 and sensitivity_table["z0"].nunique() == 2


@needs("data/structure_track/outputs/cyp3a4/final/official_posebusters_pools.csv")
def test_the_three_engines_and_the_pool_of_the_recipe_are_all_there():
    from gen.overview_more import engine_table, recipe_pool

    assert set(engine_table()["engine"]) == {"boltz", "openfold3", "bioemu2"}
    pools, chosen = recipe_pool()
    assert len(pools) == 200 and len(chosen) == 20 and set(chosen.values()) <= set(pools["kind"])


def test_the_repository_map_counts_every_file_that_git_tracks():
    import subprocess

    from gen.overview_more import repository_files

    tracked = [n for n in subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.splitlines() if Path(n).is_file()]
    assert len(repository_files()) == len(tracked)
