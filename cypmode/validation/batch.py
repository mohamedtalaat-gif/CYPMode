"""Batch orchestration: run structural validation across many compounds, one call.

cypmode.validation.pipeline.run_structural_validation handles one compound
end to end, but a real batch (e.g. every compound in an external blinded
test set) needs that call repeated across many compounds, with two things
a plain loop doesn't give for free: one compound's failure (a bad SMILES,
a Boltz-2 crash) shouldn't lose every other compound's result, and a run
spanning many hours needs its results checkpointed as it goes, not only
written out at the very end.
"""

import json
from pathlib import Path

from cypmode.validation.pipeline import run_structural_validation


def run_batch(compounds: dict[str, str], results_path: str | Path, **pipeline_kwargs) -> dict:
    """Run run_structural_validation for every name->SMILES pair in `compounds`.

    Writes results_path after every compound, so a multi-hour run's
    progress survives an interruption. Resumes automatically: names already
    present in an existing results_path are skipped rather than re-run. A
    per-compound failure is recorded as {"error": ...} rather than raised,
    so one bad compound doesn't stop the rest of the batch.
    """
    results_path = Path(results_path)
    results: dict = json.loads(results_path.read_text()) if results_path.exists() else {}

    for name, smiles in compounds.items():
        if name in results:
            continue
        try:
            results[name] = run_structural_validation(name, smiles, **pipeline_kwargs)
        except Exception as exc:
            results[name] = {"error": str(exc)}
        results_path.write_text(json.dumps(results, indent=2) + "\n")

    return results


def _main() -> None:
    import argparse
    import csv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="CSV with Molecule_Name,SMILES columns")
    parser.add_argument("results_path", help="JSON file to checkpoint results to (resumes if it already exists)")
    parser.add_argument("--no-stream", dest="stream_output", action="store_false")
    args = parser.parse_args()

    with open(args.csv_path, newline="") as f:
        compounds = {row["Molecule_Name"]: row["SMILES"] for row in csv.DictReader(f)}

    results = run_batch(compounds, args.results_path, stream_output=args.stream_output)
    n_errors = sum(1 for r in results.values() if "error" in r)
    print(f"done: {len(results)} compound(s), {n_errors} error(s), written to {args.results_path}")


if __name__ == "__main__":
    _main()
