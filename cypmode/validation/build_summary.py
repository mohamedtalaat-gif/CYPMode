"""Regenerates data/boltz_test/validation_summary.json from local Boltz-2 runs.

Discovers compounds from data/boltz_test/out/ itself rather than a fixed
list, and reads each one's ligand SMILES from its own input YAML
(data/boltz_test/cyp3a4_<compound>.yaml) rather than a second, separately
maintained mapping -- one source of truth per fact. Run after `boltz
predict` has produced the result directories -- see README.md's Validation
section. Requires the `validate` extra (`pip install -e ".[validate]"`) for
Biopython.
"""

import json
import re
from pathlib import Path

from cypmode.validation.structures import summarize_compound

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "boltz_test" / "out"
INPUT_DIR = ROOT / "data" / "boltz_test"
SUMMARY_PATH = ROOT / "data" / "boltz_test" / "validation_summary.json"

_SMILES_LINE = re.compile(r"smiles:\s*'([^']*)'")


def _read_ligand_smiles(yaml_path: Path) -> str:
    """Extract the ligand SMILES from a Boltz-2 input YAML without a full YAML
    parser -- the input files here are simple enough that a regex over the
    `smiles: '...'` line is a smaller dependency than adding PyYAML just for this.
    """
    text = yaml_path.read_text()
    matches = _SMILES_LINE.findall(text)
    if not matches:
        raise ValueError(f"no smiles: '...' line found in {yaml_path}")
    if len(matches) > 1:
        raise ValueError(f"expected exactly one ligand smiles in {yaml_path}, found {len(matches)}")
    return matches[0]


def discover_compounds() -> list[str]:
    """Compound names with both a finished Boltz-2 run and an input YAML."""
    names = []
    for result_dir in sorted(OUT_DIR.glob("boltz_results_cyp3a4_*")):
        name = result_dir.name.removeprefix("boltz_results_cyp3a4_")
        if (INPUT_DIR / f"cyp3a4_{name}.yaml").exists():
            names.append(name)
    return names


def main() -> None:
    compounds = discover_compounds()
    if not compounds:
        raise SystemExit(f"no finished Boltz-2 runs found under {OUT_DIR}")

    results = {}
    for name in compounds:
        result_dir = OUT_DIR / f"boltz_results_cyp3a4_{name}"
        ligand_smiles = _read_ligand_smiles(INPUT_DIR / f"cyp3a4_{name}.yaml")
        results[name] = summarize_compound(result_dir, ligand_smiles)

    SUMMARY_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {SUMMARY_PATH} ({len(results)} compound(s): {', '.join(compounds)})")


if __name__ == "__main__":
    main()
