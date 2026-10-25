# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

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
CPU_WRAPPER = Path(__file__).resolve().with_name("boltz_cpu_fp32.py")


def boltz_command(boltz_bin: str | Path, accelerator: str) -> list[str]:
    """The start of the Boltz command: its executable, or on the CPU the Python of its environment with the float32 wrapper."""
    if accelerator == "cpu":
        return [str(Path(boltz_bin).with_name("python")), "-P", str(CPU_WRAPPER)]
    return [str(boltz_bin)]


def run_structural_validation(
    name: str,
    ligand_smiles: str,
    protein_sequence: str = CYP3A4_SEQUENCE,
    target: str = "cyp3a4",
    boltz_bin: str | Path = ROOT / ".venv-boltz" / "bin" / "boltz",
    out_dir: str | Path = ROOT / "data" / "structure_track" / "outputs" / "out",
    cache_dir: str | Path = ROOT / "third_party" / "boltz_cache",
    input_dir: str | Path = ROOT / "data" / "structure_track" / "inputs",
    accelerator: str = "gpu",
    override: bool = True,
    stream_output: bool = True,
    predict_affinity: bool = True,
    recycling_steps: int | None = None,
    max_msa_seqs: int | None = None,
    num_subsampled_msa: int | None = None,
    diffusion_samples: int | None = None,
    diffusion_samples_affinity: int | None = None,
    max_parallel_samples: int | None = None,
    donor_restraint: str | None = None,
    templates: list | None = None,
    protein_msa: str | Path | None = None,
    step_scale: float | None = None,
    seed: int | None = None,
    write_embeddings: bool = False,
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

    target names the protein this run is against (used in file/directory
    naming, e.g. "cyp3a4_<name>.yaml") -- defaults to "cyp3a4" to match
    every existing file this project has, but was a hardcoded literal
    until now, which would have silently collided CYP2D6 runs' files
    with CYP3A4's own under the same names. Pass target="cyp2d6" (with a
    real CYP2D6 protein_sequence) for a different real isoform, rather
    than reusing this default against a sequence it doesn't actually
    name.

    out_dir is the shared base for every target's results -- this
    function always writes to out_dir/target/, not out_dir directly
    (e.g. the real default resolves to .../outputs/out/cyp3a4/ or
    .../outputs/out/cyp2d6/), so two targets' Boltz-2 results are never
    mixed in one flat directory.
    """
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = input_dir / f"{target}_{name}.yaml"

    coordinating_cys = find_coordinating_cysteine(protein_sequence)
    yaml_path.write_text(
        build_boltz_yaml(
            protein_sequence, ligand_smiles, coordinating_cys, predict_affinity=predict_affinity, donor_restraint=donor_restraint,
            templates=templates,
            protein_msa=protein_msa,
        )
    )

    # Nested by target so CYP3A4 and CYP2D6 (or any future isoform) results
    # never land in the same flat directory -- out_dir is the shared base,
    # this is where boltz is actually told to write for this specific run.
    target_out_dir = Path(out_dir) / target

    command = [
        *boltz_command(boltz_bin, accelerator),
        "predict",
        str(yaml_path),
        "--out_dir",
        str(target_out_dir),
        "--cache",
        str(cache_dir),
        "--accelerator",
        accelerator,
        "--use_msa_server",
    ]
    if override:
        command.append("--override")
    if recycling_steps is not None:
        command += ["--recycling_steps", str(recycling_steps)]
    if max_msa_seqs is not None:
        command += ["--max_msa_seqs", str(max_msa_seqs)]
    if num_subsampled_msa is not None:
        command += ["--num_subsampled_msa", str(num_subsampled_msa)]
    if diffusion_samples is not None:
        command += ["--diffusion_samples", str(diffusion_samples)]
    if diffusion_samples_affinity is not None:
        command += ["--diffusion_samples_affinity", str(diffusion_samples_affinity)]
    if step_scale is not None:
        command += ["--step_scale", str(step_scale)]
    if seed is not None:
        command += ["--seed", str(seed)]
    if max_parallel_samples is not None:
        command += ["--max_parallel_samples", str(max_parallel_samples)]
    if write_embeddings:
        command.append("--write_embeddings")

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
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"boltz predict failed for {name!r} (exit {result.returncode}):\n"
                f"--- stdout (tail) ---\n{result.stdout[-2000:]}\n"
                f"--- stderr (tail) ---\n{result.stderr[-2000:]}"
            )

    result_dir = target_out_dir / f"boltz_results_{target}_{name}"
    return summarize_compound(
        result_dir, ligand_smiles, expect_affinity=predict_affinity, expect_trunk_embedding=write_embeddings
    )


def _main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="short identifier, e.g. a compound name (used in file/directory names)")
    parser.add_argument("smiles", help="ligand SMILES")
    parser.add_argument("--protein-sequence", default=CYP3A4_SEQUENCE, help="defaults to CYP3A4 (UniProt P08684)")
    parser.add_argument(
        "--target",
        default="cyp3a4",
        help="protein name used in file/directory naming only (e.g. 'cyp3a4_<name>.yaml') -- "
        "must be passed explicitly alongside --protein-sequence for a different real isoform "
        "(e.g. --target cyp2d6), or its files would silently collide with cyp3a4's own",
    )
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
        target=args.target,
        override=args.override,
        stream_output=args.stream_output,
    )
    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
