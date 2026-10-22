# Contributing

The repository holds a heme-coordination motif screen with a structure-based check, and the code for an entry to the OpenADMET CYP
blind challenge (TDI classification and CYP3A4 structure prediction). Issues and pull requests are welcome; scope changes are
easiest to agree on before writing code, so open an issue first for anything beyond a small fix.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,challenge]"
```

Extras (see [README.md](README.md) for what each is for):

- `dev` - the motif screen, data loaders and `app.py`.
- `challenge` - LightGBM, scikit-learn, statsmodels, tblite and the rest of the TDI engine.
- `validate` - Biopython and PoseBusters for the structure tools.
- `multitask` - PyTorch, for the shared-trunk model and ChemBERTa features.
- `tdc` - refreshes the CYP panel cache from source instead of the checked-in `data/tdi_track/inputs/tdc_cache/`.

Boltz-2, OpenFold3 and chemprop each live in their own virtual environment (`.venv-boltz`, `.venv-openfold3`, `.venv-ml`); the package
versions of `.venv` and `.venv-ml` are pinned in `environment/`.

## Running tests

```bash
python -m pytest tests -q
python -m pytest tests/test_multitask.py -q
python -m pytest tests/test_chemberta_features.py -q
python -m pytest tests/test_quantum_pharmacophore.py tests/test_physics_block.py tests/test_physics_tierb.py -q
```

The last three commands each run in their own process. PyTorch and tblite bundle OpenMP runtimes that crash or deadlock when they are
loaded into the same process as LightGBM, so `pyproject.toml` ignores those files in the main run. Tests that need the TDC panel
cache, Boltz-2 outputs or the large caches skip cleanly when those are absent.

## Checks before a pull request

```bash
python -m cypmode.audit.checklist       # every job, module and file named in SUBMISSION_CHECKLIST.md exists
bash scripts/run_audit.sh               # full audit report, generated twice and compared byte for byte
python -m cypmode.audit.method_report --check   # METHOD_REPORT.md and NEGATIVE_RESULTS.md equal a fresh generation
python -m cypmode.audit.public_readiness        # private paths, credential-shaped strings, large files, third-party text
python -m cypmode.audit.checklist --status      # done, running and queued items of the checklist
```

Documents that describe the architecture (the sketches in `README.md` and `SUBMISSION_CHECKLIST.md`, the repository map, this file) are
updated in the same change as the code they describe.

## Experiments

A new model component is a candidate group in `cypmode/challenge/nested_runners.py` with a driver built on
`cypmode/challenge/addition_experiment.py`: it is evaluated alone and inside the assembled stack, with the corrected resampled t at both
confidence cut-offs, and it enters the final stack only through the gate (`cypmode/challenge/final_tdi_build.py`). Long runs are queue
jobs (`queue/jobs/NNN_name.sh`, started with `bash scripts/queue_start.sh`; helpers in `queue/lib.sh`, memory and battery guards in
`scripts/guards.sh`). Physics additions are also judged with 10 outer folds (`python -m cypmode.challenge.replicate_verdict`, candidate files
from `final_tdi_build.py --design 10`); `bash scripts/reproduce_tdi.sh` rebuilds the candidate files from the raw files.

Structure-track additions follow the same pattern: a term or lever gets a gate on the crystal-anchored benchmark (`cypmode/validation/pose_awareness.py`,
`series_consensus.py`, `selection_validation.py`, `structure_gates.py`), prints a `STRUCTURE GATE name: MET|NOT MET` line, and enters the
recipe only through `cypmode/validation/final_recipe.py`.

The notebook is rebuilt with `python -m notebooks.build_notebook` (executes every cell, checks that every defined term is defined where it is first used and that every file of the repository is explained, then exports HTML and PDF). A new module, script, queue job or test file has to be named in the text of the notebook before it builds.

## Conventions

- No premature abstraction: concrete functions rather than shared base classes or generic frameworks.
- Every heuristic or motif choice traces back to a citable source (a paper, a named method) in a comment or docstring, as in
  `cypmode/metrics/motifs.py`.
- Comments explain why, not what, and are written in English.
- Prose (docstrings, comments, commit messages, this file) is plain and direct: the change and its rationale.
- Numbers quoted in documents come from the corrected procedure (nested cross-validation, corrected resampled t, permutation nulls) and are
  produced by code, not typed in.

## Reporting bugs / requesting features

Use the issue templates. For anything touching `cypmode/metrics/`, include a concrete SMILES string and the motif call you got against
what you expected.

## License

The code is licensed under the GNU Affero General Public License, version 3 or later, with the additional terms of `NOTICE`; the documents under CC BY-NC-SA 4.0 and the generated data under CC BY 4.0 (`LICENSES/`, `REUSE.toml`). Unless stated otherwise, a contribution submitted for inclusion is licensed under the licence of the part of the repository that it changes, without additional conditions.
