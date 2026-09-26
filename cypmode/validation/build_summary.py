"""Regenerates data/structure_track/outputs/validation_summary.json from local Boltz-2 runs.

Discovers compounds from data/structure_track/outputs/out/ itself rather than a fixed
list, and reads each one's ligand SMILES from its own input YAML
(data/structure_track/inputs/cyp3a4_<compound>.yaml) rather than a second, separately
maintained mapping -- one source of truth per fact. Run after `boltz
predict` has produced the result directories -- see README.md's Validation
section. Requires the `validate` extra (`pip install -e ".[validate]"`) for
Biopython.
"""

import json
import re
from pathlib import Path

# Deliberately not imported at module level: summarize_compound needs
# Biopython (the `validate` extra), but discover_compounds() and
# _read_ligand_smiles() below don't, and should stay importable/testable
# without it -- imported lazily inside main() instead.

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "structure_track" / "outputs" / "out"
INPUT_DIR = ROOT / "data" / "structure_track" / "inputs"
SUMMARY_PATH = ROOT / "data" / "structure_track" / "outputs" / "validation_summary.json"

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


def discover_compounds(target: str = "cyp3a4") -> list[str]:
    """Compound names with both a finished Boltz-2 run and an input YAML,
    for the given target protein (used only in file/directory naming --
    was a hardcoded "cyp3a4" literal until now, which would have silently
    found zero CYP2D6 runs rather than a real, distinct set of them).
    """
    names = []
    for result_dir in sorted(OUT_DIR.glob(f"boltz_results_{target}_*")):
        name = result_dir.name.removeprefix(f"boltz_results_{target}_")
        if (INPUT_DIR / f"{target}_{name}.yaml").exists():
            names.append(name)
    return names


def main(target: str = "cyp3a4", summary_path: Path | None = None) -> None:
    from cypmode.validation.structures import summarize_compound

    compounds = discover_compounds(target)
    if not compounds:
        raise SystemExit(f"no finished Boltz-2 runs found under {OUT_DIR} for target={target!r}")

    results = {}
    for name in compounds:
        result_dir = OUT_DIR / f"boltz_results_{target}_{name}"
        ligand_smiles = _read_ligand_smiles(INPUT_DIR / f"{target}_{name}.yaml")
        results[name] = summarize_compound(result_dir, ligand_smiles)

    out_path = summary_path or SUMMARY_PATH
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out_path} ({len(results)} compound(s): {', '.join(compounds)})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default="cyp3a4",
        help="protein name used in file/directory naming (e.g. 'cyp2d6' for a CYP2D6 batch)",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="defaults to data/structure_track/outputs/validation_summary.json -- "
        "pass a different path for a non-cyp3a4 target so its summary doesn't "
        "overwrite the real CYP3A4 one",
    )
    args = parser.parse_args()
    main(target=args.target, summary_path=args.summary_path)
