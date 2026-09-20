from pathlib import Path

import pytest

pytest.importorskip("Bio")

from cypmode.validation.structures import summarize_compound

RESULT_DIR = Path("data/boltz_test/out/boltz_results_cyp3a4_ketoconazole")
CIF_PATH = RESULT_DIR / "predictions" / "cyp3a4_ketoconazole" / "cyp3a4_ketoconazole_model_0.cif"

pytestmark = pytest.mark.skipif(
    not CIF_PATH.exists(),
    reason="requires a local Boltz-2 run; see README.md's Validation section",
)


def test_summarize_ketoconazole():
    result = summarize_compound(RESULT_DIR)
    assert result["compound"] == "cyp3a4_ketoconazole"
    assert result["fe_n_distance_angstrom"] > 0
    assert isinstance(result["coordinated"], bool)
    assert "affinity_pred_value" in result
