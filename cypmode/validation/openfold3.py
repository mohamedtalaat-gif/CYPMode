# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Resumable OpenFold3 batch runner -- an ADDED second structure engine
next to Boltz-2 (cypmode.validation.pipeline), never a replacement.

Verified working on this Apple M5 (16 GB unified RAM, MPS) with the
default OpenBind-0 checkpoint, run from the isolated .venv-openfold3
(same isolation pattern as .venv-boltz): CYP3A4 + ketoconazole + heme
with ColabFold MSAs took ~16 min for one structure, placed the heme on
Cys442 and the imidazole nitrogen 2.14 A from the iron with no
constraints at all (Boltz-2: 2.05 A). Without MSAs the same model is
useless (pLDDT 33), so MSAs are always on here.

Memory is the binding constraint: free RAM dipped to ~9% with ~7 GB of
swap at the peak of a single prediction. So predictions run strictly one
at a time, and this module refuses to start one while another
run_openfold process is alive -- never alongside Boltz-2 either (see
_another_prediction_running).

Each compound's ligand SMILES, protein sequence and heme come from the
SAME Boltz-2 input YAML the existing pipeline used
(data/structure_track/inputs/{target}_{name}.yaml), so both engines are
guaranteed to predict exactly the same system. No covalent-bond
constraint is passed: OpenFold3 specifies bond atoms by integer index,
which is easy to get wrong, and it reproduces the Cys442-Fe ligation
unaided (Cys442 SG to Fe 2.27 A in the ketoconazole run).
"""

import json
import subprocess
import time
from pathlib import Path

import yaml

OPENFOLD3_BIN = Path(".venv-openfold3/bin/run_openfold")
OPENFOLD3_PYTHON = Path(".venv-openfold3/bin/python3")
LOWMEM_SCRIPT = Path("scripts/run_openfold_lowmem.py")
RUNNER_YAML = Path("data/structure_track/inputs/openfold3_runner.yml")
MEMORY_MODE_FILE = Path("data/structure_track/inputs/openfold3_memory_mode.txt")  # written by the memory probe: the first mode that completes the largest benchmark ligand
MEMORY_MODES = ("config", "lowmem")  # config: runner YAML with a smaller chunk size; lowmem: that plus the low-memory chunk_layer path
INPUTS_DIR = Path("data/structure_track/inputs")
OUT_ROOT = Path("data/structure_track/outputs/openfold3")
DEFAULT_TIMEOUT_S = 3600  # a normal CYP3A4 prediction takes ~16 min; a stalled one should not hold the machine forever


def read_boltz_inputs(name: str, target: str = "cyp3a4", inputs_dir: Path = INPUTS_DIR) -> dict:
    """The protein sequence and ligand SMILES from the compound's own
    Boltz-2 input YAML -- the single source of truth for what both
    engines predict. The heme (CCD HEM) is implied for every CYP, so it
    is not read from the YAML.
    """
    spec = yaml.safe_load((inputs_dir / f"{target}_{name}.yaml").read_text())
    protein = ligand = None
    for entry in spec["sequences"]:
        if "protein" in entry:
            protein = entry["protein"]["sequence"]
        elif "ligand" in entry and "smiles" in entry["ligand"]:
            ligand = entry["ligand"]["smiles"]
    if protein is None or ligand is None:
        raise ValueError(f"{target}_{name}.yaml needs one protein and one SMILES ligand entry")
    return {"protein_sequence": protein, "ligand_smiles": ligand}


def build_query(query_name: str, protein_sequence: str, ligand_smiles: str) -> dict:
    """OpenFold3's InferenceQuerySet JSON for one protein + inhibitor +
    heme complex: chain A protein, chain B inhibitor (SMILES), chain C
    heme (CCD HEM) -- the same A/B/C layout Boltz-2 uses, so downstream
    code can treat both engines' output identically. MSAs on (see module
    docstring for why).
    """
    return {
        "seeds": [42],
        "queries": {
            query_name: {
                "chains": [
                    {"molecule_type": "protein", "chain_ids": ["A"], "sequence": protein_sequence},
                    {"molecule_type": "ligand", "chain_ids": ["B"], "smiles": ligand_smiles},
                    {"molecule_type": "ligand", "chain_ids": ["C"], "ccd_codes": ["HEM"]},
                ],
                "use_msas": True,
            }
        },
    }


def predictions_dir(name: str, target: str = "cyp3a4", out_root: Path = OUT_ROOT) -> Path:
    return out_root / target / name


def model_cif_paths(name: str, target: str = "cyp3a4", out_root: Path = OUT_ROOT) -> list[Path]:
    """Every sampled structure OpenFold3 wrote for this compound, in
    sample order (…_sample_1_model.cif, …_sample_2_model.cif, …).
    """
    query_name = f"{target}_{name}"
    found = (predictions_dir(name, target, out_root) / query_name).glob("seed_*/*_model.cif")
    return sorted(found, key=lambda p: (p.parent.name, int(p.stem.split("_sample_")[1].split("_")[0])))


def ranking_scores(cif_paths: list[Path]) -> list[float | None]:
    """OpenFold3's own per-sample ranking score from the aggregated
    confidences JSON next to each CIF (None if absent). Informational
    only: confidence scores are an unreliable pose selector for P450s
    (see cypmode.challenge.structure_submission's module docstring), so
    cypmode.validation.pose_consensus never selects on it.
    """
    scores = []
    for cif in cif_paths:
        agg = cif.with_name(cif.name.replace("_model.cif", "_confidences_aggregated.json"))
        scores.append(json.loads(agg.read_text()).get("sample_ranking_score") if agg.exists() else None)
    return scores


def _another_prediction_running() -> bool:
    """True if any other OpenFold3 prediction process is alive. Two at
    once would exceed this 16 GB machine's memory (a single prediction
    already pushes free RAM to ~9%).
    """
    result = subprocess.run(["pgrep", "-f", "run_openfold"], capture_output=True, text=True, check=False)
    return bool(result.stdout.strip())


def default_memory_mode(path: Path = MEMORY_MODE_FILE) -> str:
    mode = path.read_text().strip() if path.exists() else "config"
    if mode not in MEMORY_MODES:
        raise ValueError(f"{path} must contain one of {MEMORY_MODES}, found {mode!r}")
    return mode


def failure_reason(out_dir: Path) -> str:
    """Last error line of OpenFold3's own log: it exits 0 when a query fails, so the reason is only in the log."""
    log = Path(out_dir) / "logs" / "predict_err_rank0.log"
    if not log.exists():
        return "no error log"
    lines = [line.strip() for line in log.read_text(errors="ignore").splitlines() if "Error" in line]
    return lines[-1][:240] if lines else "no error line in the log"


def run_openfold3(
    name: str,
    protein_sequence: str,
    ligand_smiles: str,
    target: str = "cyp3a4",
    num_diffusion_samples: int = 1,
    out_root: Path = OUT_ROOT,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    memory: str = "config",
) -> dict:
    """Runs one OpenFold3 prediction (blocking) and returns its summary.
    Raises if another prediction is already running, if the process
    fails/times out, or if it produced no structure -- the batch runner
    records that as an error and moves on.
    """
    if _another_prediction_running():
        raise RuntimeError("another run_openfold process is already running -- predictions must not overlap (16 GB RAM)")

    query_name = f"{target}_{name}"
    out_dir = predictions_dir(name, target, out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    query_path = out_dir / "query.json"
    query_path.write_text(json.dumps(build_query(query_name, protein_sequence, ligand_smiles), indent=2) + "\n")
    for stale in (out_dir / "logs" / "predict_err_rank0.log", out_dir / "summary.txt"):
        stale.unlink(missing_ok=True)  # an earlier attempt's failure must not be reported as this run's

    if memory not in MEMORY_MODES:
        raise ValueError(f"memory must be one of {MEMORY_MODES}")
    launcher = [str(OPENFOLD3_BIN)] if memory == "config" else [str(OPENFOLD3_PYTHON), str(LOWMEM_SCRIPT)]
    command = [
        *launcher,
        "predict",
        f"--runner-yaml={RUNNER_YAML}",
        f"--query-json={query_path}",
        f"--output-dir={out_dir}",
        "--use-msa-server=true",
        "--use-templates=false",
        f"--num-diffusion-samples={num_diffusion_samples}",
        "--num-model-seeds=1",
    ]
    started = time.time()
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s, check=False)
    runtime_s = time.time() - started
    if result.returncode != 0:
        raise RuntimeError(f"run_openfold failed for {query_name} (exit {result.returncode}):\n{result.stderr[-1500:]}")

    cifs = model_cif_paths(name, target, out_root)
    if not cifs:
        raise RuntimeError(f"run_openfold exited 0 but wrote no *_model.cif under {out_dir}: {failure_reason(out_dir)}")
    return {
        "model_cifs": [str(c) for c in cifs],
        "ranking_scores": ranking_scores(cifs),
        "runtime_s": round(runtime_s, 1),
    }


def run_openfold3_batch(
    names: list[str],
    target: str = "cyp3a4",
    num_diffusion_samples: int = 1,
    out_root: Path = OUT_ROOT,
    inputs_dir: Path = INPUTS_DIR,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    memory: str = "config",
) -> dict:
    """Sequential, resumable batch: results are checkpointed to
    {out_root}/{target}/results.json after every compound, a compound
    that already has a successful entry (with its CIFs still on disk) is
    skipped, and a failed one is recorded as {"error": ...} and retried
    on the next call -- same contract as cypmode.validation.batch.run_batch.
    """
    results_path = out_root / target / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for name in names:
        existing = results.get(name)
        if existing and "error" not in existing and all(Path(c).exists() for c in existing["model_cifs"]):
            continue
        try:
            inputs = read_boltz_inputs(name, target, inputs_dir)
            results[name] = run_openfold3(
                name, inputs["protein_sequence"], inputs["ligand_smiles"], target, num_diffusion_samples, out_root, timeout_s, memory
            )
        except Exception as exc:
            results[name] = {"error": str(exc)}
        results_path.write_text(json.dumps(results, indent=2) + "\n")
    return results


def _main() -> None:
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default="cyp3a4")
    parser.add_argument("--names", nargs="*", help="Molecule_Names to run (default: every compound in --test-set)")
    parser.add_argument(
        "--test-set",
        default="data/structure_track/inputs/structure_track_test_blinded.csv",
        help="CSV with a Molecule_Name column; used when --names is not given",
    )
    parser.add_argument("--limit", type=int, default=None, help="only the first N compounds (dry runs)")
    parser.add_argument("--samples", type=int, default=1, help="diffusion samples per compound")
    parser.add_argument("--memory", choices=MEMORY_MODES, default=None, help="config: runner YAML with a smaller chunk size; lowmem: that plus the low-memory chunk_layer path (default: the mode in openfold3_memory_mode.txt, else config)")
    parser.add_argument("--strict", action="store_true", help="exit with status 1 when any requested compound has no structure")
    args = parser.parse_args()

    names = args.names or pd.read_csv(args.test_set)["Molecule_Name"].tolist()
    if args.limit:
        names = names[: args.limit]
    print(f"running OpenFold3 on {len(names)} compound(s), one at a time (~16 min each)")
    memory = args.memory or default_memory_mode()
    print(f"memory mode: {memory}")
    results = run_openfold3_batch(names, args.target, args.samples, memory=memory)
    for name in names:
        r = results[name]
        print(f"  {name}: " + (f"ERROR {r['error'][:200]}" if "error" in r else f"{len(r['model_cifs'])} sample(s), {r['runtime_s']} s"))
    failed = [n for n in names if "error" in results[n]]
    if args.strict and failed:
        print(f"{len(failed)} of {len(names)} compound(s) without a structure: {', '.join(failed)}")
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
