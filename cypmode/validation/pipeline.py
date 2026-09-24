"""End-to-end structural validation: SMILES in, coordination result out.

Wraps three steps that were, until now, separate manual actions: building
the Boltz-2 input (build_input.py), running Boltz-2 itself, and summarizing
the result (structures.py). Boltz-2 is a slow (tens of minutes), external,
GPU-bound model -- this module shells out to it rather than pretending to
skip or fake that step, and rather than importing it into this process
(Boltz-2 needs its own virtual environment; see README.md's Reproducing
this section).
"""

import subprocess
from pathlib import Path

from cypmode.validation.build_input import build_boltz_yaml, find_coordinating_cysteine
from cypmode.validation.constants import CYP3A4_SEQUENCE
from cypmode.validation.structures import summarize_compound

ROOT = Path(__file__).resolve().parents[2]


def run_structural_validation(
    name: str,
    ligand_smiles: str,
    protein_sequence: str = CYP3A4_SEQUENCE,
    boltz_bin: str | Path = ROOT / ".venv-boltz" / "bin" / "boltz",
    out_dir: str | Path = ROOT / "data" / "boltz_test" / "out",
    cache_dir: str | Path = ROOT / "third_party" / "boltz_cache",
    input_dir: str | Path = ROOT / "data" / "boltz_test",
    accelerator: str = "gpu",
    override: bool = True,
    stream_output: bool = True,
) -> dict:
    """Build the input, run Boltz-2 (blocking -- tens of minutes), and summarize
    the result. Raises RuntimeError with Boltz-2's own output on failure,
    rather than a bare non-zero exit code.

    override defaults to True: confirmed directly (itraconazole, a fresh
    compound with no prior cache at all) that Boltz-2 can exit 0 having
    written the structure but silently skipped the affinity JSON, and that
    --override is what makes it actually run that step -- not just a stale
    old-cache issue reusing a name. --override still reuses the expensive
    MSA/preprocessing cache when the input hasn't changed (confirmed in the
    same run: "All inputs are already processed"), so this costs nothing
    beyond forcing the prediction step to actually execute.

    stream_output prints Boltz-2's own progress live instead of buffering
    it until the process exits -- useful for a run that takes tens of
    minutes, since capture_output=True previously gave no signal at all
    until the very end.
    """
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = input_dir / f"cyp3a4_{name}.yaml"

    coordinating_cys = find_coordinating_cysteine(protein_sequence)
    yaml_path.write_text(build_boltz_yaml(protein_sequence, ligand_smiles, coordinating_cys))

    command = [
        str(boltz_bin),
        "predict",
        str(yaml_path),
        "--out_dir",
        str(out_dir),
        "--cache",
        str(cache_dir),
        "--accelerator",
        accelerator,
        "--use_msa_server",
    ]
    if override:
        command.append("--override")

    if stream_output:
        lines: list[str] = []
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        ) as proc:
            for line in proc.stdout:
                print(line, end="", flush=True)
                lines.append(line)
            returncode = proc.wait()
        if returncode != 0:
            raise RuntimeError(f"boltz predict failed for {name!r} (exit {returncode}):\n" + "".join(lines[-100:]))
    else:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"boltz predict failed for {name!r} (exit {result.returncode}):\n"
                f"--- stdout (tail) ---\n{result.stdout[-2000:]}\n"
                f"--- stderr (tail) ---\n{result.stderr[-2000:]}"
            )

    result_dir = Path(out_dir) / f"boltz_results_cyp3a4_{name}"
    return summarize_compound(result_dir, ligand_smiles)


def _main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="short identifier, e.g. a compound name (used in file/directory names)")
    parser.add_argument("smiles", help="ligand SMILES")
    parser.add_argument("--protein-sequence", default=CYP3A4_SEQUENCE, help="defaults to CYP3A4 (UniProt P08684)")
    parser.add_argument(
        "--no-override",
        dest="override",
        action="store_false",
        help="reuse existing predictions instead of forcing a fresh run (default: force a fresh run -- "
        "Boltz-2 can exit 0 having silently skipped the affinity step otherwise, confirmed directly)",
    )
    parser.add_argument("--no-stream", dest="stream_output", action="store_false", help="buffer output instead of printing it live")
    args = parser.parse_args()

    result = run_structural_validation(
        args.name,
        args.smiles,
        protein_sequence=args.protein_sequence,
        override=args.override,
        stream_output=args.stream_output,
    )
    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
