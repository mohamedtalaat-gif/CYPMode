from pathlib import Path

import pytest

pytest.importorskip("Bio")

from cypmode.validation.structures import summarize_compound

KETOCONAZOLE_SMILES = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"
VARDENAFIL_SMILES = "CCCc1nc(C)c2c(=O)nc(-c3cc(S(=O)(=O)N4CCN(CC)CC4)ccc3OCC)[nH]n12"

RESULT_DIR = Path("data/boltz_test/out/boltz_results_cyp3a4_ketoconazole")
CIF_PATH = RESULT_DIR / "predictions" / "cyp3a4_ketoconazole" / "cyp3a4_ketoconazole_model_0.cif"

pytestmark = pytest.mark.skipif(
    not CIF_PATH.exists(),
    reason="requires a local Boltz-2 run; see README.md's Validation section",
)


def test_summarize_ketoconazole_coordinates():
    # Real, known-correct answer: the imidazole nitrogen (CIF atom N32) sits
    # 2.05 A from the heme iron -- confirmed independently by
    # github.com/PatWalters/cyp-heatmap's ProLIF analysis of the same
    # compound (see README.md's Independent cross-check section).
    result = summarize_compound(RESULT_DIR, KETOCONAZOLE_SMILES)
    assert result["compound"] == "cyp3a4_ketoconazole"
    assert result["coordinated"] is True
    assert result["closest_motif_nitrogen_distance_angstrom"] == pytest.approx(2.05, abs=0.01)
    assert result["motif_nitrogen_distances"][0]["atom"] == "N32"
    assert result["motif_nitrogen_distances"][0]["motif"] == "imidazole_Nsub"
    assert "affinity_pred_value" in result


def test_motif_distance_ignores_closer_non_motif_nitrogen():
    # The regression this module exists to prevent: vardenafil's
    # geometrically closest ligand nitrogen (N50, ~4.84 A) is not part of
    # any matched Type II motif. The motif-restricted distance must not
    # silently fall back to it.
    result_dir = Path("data/boltz_test/out/boltz_results_cyp3a4_vardenafil")
    if not (result_dir / "predictions" / "cyp3a4_vardenafil" / "cyp3a4_vardenafil_model_0.cif").exists():
        pytest.skip("requires a local Boltz-2 run for vardenafil")

    result = summarize_compound(result_dir, VARDENAFIL_SMILES)
    motif_atoms = {d["atom"] for d in result["motif_nitrogen_distances"]}
    assert "N50" not in motif_atoms
    assert result["closest_motif_nitrogen_distance_angstrom"] > 4.84
    assert result["coordinated"] is False

    all_atoms = {d["atom"] for d in result["all_ligand_nitrogen_distances"]}
    assert "N50" in all_atoms  # still visible for manual review, just not driving the call
