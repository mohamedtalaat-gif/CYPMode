# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Resumable BioEmu2 runner -- a candidate third structure engine next to Boltz-2 and OpenFold3, never a replacement.

BioEmu2 (Microsoft Research, MIT licence, an all-atom diffusion model of about 500 million parameters, public checkpoint
``bioemu2-preview`` in the Hugging Face repository ``microsoft/bioemu2``) predicts the structure of a protein with ligands from the
sequence, an alignment and optional contacts. Its documentation names Linux and an NVIDIA GPU as the supported platform. The
checkpoint's own configuration selects the triangle-multiplication kernel of cuEquivariance, which exists only for NVIDIA GPUs,
so a local copy of the configuration is used here: the pure-PyTorch kernel (``trimul_variant: full``) and chunked transition and
attention-pair-weighting layers, which lower the peak memory without changing the mathematics. The weights are not touched (the
copy links to the downloaded file).

Each compound's protein sequence, ligand SMILES, heme and the covalent bond of Cys442 to the iron come from the SAME Boltz-2
input file that the other engines use (``data/structure_track/inputs/{target}_{name}.yaml``, converted with BioEmu2's own
``convert-boltz``), and the alignment is the first ``--rows`` rows of the alignment that Boltz-2 used, so that the engines differ in
the model and not in the evidence they are given. No contact conditioning is passed: contacts to atoms of a ligand that is given
as a SMILES string are not supported by BioEmu2.

Memory is the binding constraint on this 16 GB machine: the run refuses to start while another prediction process is alive, and
the queue's memory guard (``scripts/guards.sh``) stops it when free memory stays low.

Every compound ends with ``sample_<i>.cif`` files (one per sample, in the chain layout A protein, B ligand, C heme) and a
``timings.json`` (seconds of the trunk and of each sample, the peak resident memory, the MSA depth, the device).

Usage (from the isolated environment .venv-bioemu2):
  python -m cypmode.validation.bioemu2 --probe                       # one benchmark system, two samples, timings only
  python -m cypmode.validation.bioemu2 --names bench-MWS ... --samples 3
"""

import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path

CACHE_DIR = Path(".bioemu2-cache")
CHECKPOINT_NAME = "bioemu2-preview"
LOCAL_DIR = CACHE_DIR / "local" / f"{CHECKPOINT_NAME}-pure-torch"
INPUTS_DIR = Path("data/structure_track/inputs")
MSA_CSV = INPUTS_DIR / "msa" / "cyp3a4_P08684.csv"
OUT_ROOT = Path("data/structure_track/outputs/bioemu2")
PROBE_NAME = "bench-MWS"
DEFAULT_ROWS = 256
CHUNK_SIZE = 64  # transition and pair-weighted-averaging chunk (rows); smaller saves memory and costs time


def pure_torch_config(text: str, chunk_size: int = CHUNK_SIZE) -> str:
    """The checkpoint's configuration with the pure-PyTorch kernel and chunked layers; fails when the file is not the expected one."""
    steps = (
        (r"(?m)^(\s+trimul_variant:) cueq$", r"\1 full", 1),
        (r"(?m)^(\s+chunk_size:) null$", rf"\1 {chunk_size}", 2),
        (r"(?m)^(\s+pwa_chunk_heads:) false$", r"\1 true", 1),
    )
    for pattern, replacement, expected in steps:
        text, count = re.subn(pattern, replacement, text)
        if count != expected:
            raise ValueError(f"expected {expected} match(es) of {pattern!r} in the checkpoint configuration, found {count}")
    return text


def snapshot_dir(cache_dir: Path = CACHE_DIR) -> Path | None:
    found = sorted(cache_dir.glob(f"models--microsoft--bioemu2/snapshots/*/checkpoints/{CHECKPOINT_NAME}"))
    return found[-1] if found else None


def prepare_checkpoint(local_dir: Path = LOCAL_DIR, cache_dir: Path = CACHE_DIR) -> Path:
    """The path of the checkpoint whose configuration is the local copy; downloads the public checkpoint once when it is missing."""
    source = snapshot_dir(cache_dir)
    if source is None:
        from huggingface_hub import snapshot_download

        snapshot_download("microsoft/bioemu2", allow_patterns=[f"checkpoints/{CHECKPOINT_NAME}/*"], cache_dir=cache_dir)
        source = snapshot_dir(cache_dir)
    if source is None:
        raise FileNotFoundError(f"no {CHECKPOINT_NAME} checkpoint under {cache_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    link = local_dir / "checkpoint.ckpt"
    if not link.exists():
        link.symlink_to((source / "checkpoint.ckpt").resolve())
    (local_dir / "config.yaml").write_text(pure_torch_config((source / "config.yaml").read_text()))
    return link


def msa_a3m(rows: int, path: Path = MSA_CSV) -> str:
    """The first ``rows`` rows of the Boltz-2 alignment (the query first) as A3M text."""
    with open(path) as handle:
        sequences = [row["sequence"] for _, row in zip(range(rows), csv.DictReader(handle), strict=False)]
    return "".join(f">{i}\n{sequence}\n" for i, sequence in enumerate(sequences))


def sample_cifs(name: str, target: str = "cyp3a4", out_root: Path = OUT_ROOT) -> list[Path]:
    """Every sample written for this compound, in sample order."""
    found = (out_root / target / name).glob("sample_*.cif")
    return sorted(found, key=lambda p: int(p.stem.split("_")[1]))


def _another_prediction_running() -> bool:
    result = subprocess.run(["pgrep", "-f", "boltz predict|run_openfold|cypmode.validation.bioemu2"], capture_output=True, text=True, check=False)
    return any(pid != str(os.getpid()) for pid in result.stdout.split())


def _peak_rss_mb() -> float:
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)  # bytes on macOS


def run(names: list[str], samples: int, rows: int, device: str, seed: int, target: str = "cyp3a4", out_root: Path = OUT_ROOT) -> dict:
    """Sequential, resumable: a compound that already has ``samples`` structures is skipped; a failure stops the run."""
    import torch
    from bioemu2 import (
        MSA,
        EmuSession,
        convert_boltz,
        load_model,
        load_system_and_contacts,
    )
    from bioemu2.trajectory_io import save_mdtraj_cif_frame

    if _another_prediction_running():
        raise RuntimeError("another prediction process is running -- predictions must not overlap (16 GB RAM)")
    model = load_model(checkpoint_path=prepare_checkpoint(), device=device, precision="32-true", matmul_precision=None)
    alignment = MSA(unpaired_a3m={"A": msa_a3m(rows)})
    summary = {}
    for name in names:
        out_dir = out_root / target / name
        if len(sample_cifs(name, target, out_root)) >= samples:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        system_path = out_dir / "system.yaml"
        convert_boltz(INPUTS_DIR / f"{target}_{name}.yaml", system_path)
        system, contacts = load_system_and_contacts(system_path)
        session = EmuSession(system, model=model, mode="structure", msa=alignment, contacts=contacts)
        started = time.time()
        session._get_embedding()  # the trunk, timed on its own
        trunk_s = time.time() - started
        sample_s, atoms = [], 0
        for i in range(samples):
            started = time.time()
            trajectory = session.sample(num_samples=1, batch_size=1, seed=seed + i, progress=False)
            sample_s.append(round(time.time() - started, 1))
            atoms = trajectory.n_atoms
            save_mdtraj_cif_frame(trajectory[0], out_dir / f"sample_{i}.cif")
        summary[name] = {"trunk_s": round(trunk_s, 1), "sample_s": sample_s, "atoms": atoms, "msa_rows": rows, "device": device,
                         "chunk_size": CHUNK_SIZE, "peak_rss_mb": _peak_rss_mb(), "seed": seed}
        if device == "mps":
            summary[name]["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 1e6, 1)
        (out_dir / "timings.json").write_text(json.dumps(summary[name], indent=2) + "\n")
        session.cache_clear()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--names", nargs="*", help="Molecule_Names to run (the benchmark ligands, or official ligands later)")
    parser.add_argument("--samples", type=int, default=3, help="structures per compound")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="alignment rows taken from the Boltz-2 alignment")
    parser.add_argument("--device", default="mps", choices=("mps", "cpu"))
    parser.add_argument("--seed", type=int, default=0, help="seed of the first sample; sample i uses seed + i")
    parser.add_argument("--probe", action="store_true", help=f"timings on one benchmark system ({PROBE_NAME}), two samples, written under <out>/probe")
    parser.add_argument("--target", default="cyp3a4")
    args = parser.parse_args()
    out_root = OUT_ROOT / "probe" if args.probe else OUT_ROOT
    names = [PROBE_NAME] if args.probe else (args.names or [])
    samples = 2 if args.probe else args.samples
    if not names:
        raise SystemExit("give --names or --probe")
    summary = run(names, samples, args.rows, args.device, args.seed, args.target, out_root)
    for name, entry in summary.items():
        print(f"  {name}: trunk {entry['trunk_s']} s, samples {entry['sample_s']} s, {entry['atoms']} atoms, peak {entry['peak_rss_mb']} MB resident")


if __name__ == "__main__":
    from cypmode import provenance

    provenance.announce()
    main()
