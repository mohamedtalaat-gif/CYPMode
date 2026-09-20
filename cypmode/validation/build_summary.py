"""Regenerates data/boltz_test/validation_summary.json from a local Boltz-2 run.

Run after `boltz predict` has produced all four
data/boltz_test/out/boltz_results_cyp3a4_<compound>/ directories -- see
README.md's Validation section for the exact command. Requires the
`validate` extra (`pip install -e ".[validate]"`) for Biopython.
"""

import json
from pathlib import Path

from cypmode.validation.structures import summarize_compound

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "boltz_test" / "out"
SUMMARY_PATH = ROOT / "data" / "boltz_test" / "validation_summary.json"

COMPOUNDS = ["ketoconazole", "ritonavir", "azamulin", "vardenafil"]


def main() -> None:
    results = {}
    for name in COMPOUNDS:
        result_dir = OUT_DIR / f"boltz_results_cyp3a4_{name}"
        results[name] = summarize_compound(result_dir)

    SUMMARY_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
