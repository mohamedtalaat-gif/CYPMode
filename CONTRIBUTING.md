# Contributing

This is a small, early-stage project — a motif screen plus a validation
harness, not a large application. Issues and pull requests are welcome;
scope changes are easiest to agree on before writing code, so open an issue
first for anything beyond a small fix.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The `dev` extra is enough for `cypmode/metrics/`, `cypmode/data/`, and
`app.py`. Two more extras exist for the rest of the codebase — see
[README.md](README.md)'s Install section for what each needs and why they're
kept separate:

- `tdc` — refreshes the CYP panel cache from source instead of the
  checked-in `data/tdc_cache/`.
- `validate` — parses the Boltz-2 structures in
  `cypmode/validation/structures.py`.

## Running tests

```bash
pytest
```

Tests that need the TDC panel cache or Boltz-2 structure outputs skip
cleanly when those aren't present — you don't need either to contribute to
`cypmode/metrics/` or `app.py`.

## Conventions

- No premature abstraction: this codebase intentionally avoids shared base
  classes and generic frameworks where a few concrete functions would do.
- Every heuristic or motif choice should trace back to a citable source (a
  paper, a named method) in a comment or docstring — see
  `cypmode/metrics/motifs.py` for the pattern. "Seemed reasonable" is not
  sufficient justification for a structural alert.
- Comments explain *why*, not *what* — if removing a comment wouldn't
  confuse a future reader, it shouldn't be there.
- Keep prose (docstrings, comments, commit messages, this file) in plain,
  direct English — no meta-narration about how or why a change was made,
  just the change and its rationale.

## Reporting bugs / requesting features

Use the issue templates. For anything touching `cypmode/metrics/`, include a
concrete SMILES string and the motif call you got vs. what you expected —
that's usually enough to reproduce.
