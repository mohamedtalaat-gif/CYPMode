# Structure Prediction track — submission

## Required format (verified directly against the live HF Space's `submission.py`, 2026-09-26 —
quoting the literal help text shown on the submission form itself, not a paraphrase)

> "Submit a `.zip` archive containing exactly **20 `.pdb` files**, one per
> compound, named after the compound identifier (e.g. `x00011-1.pdb`).
> Each file must be a full protein–ligand complex with the ligand residue
> named `LIG`."

Concretely, for our own real 20-compound official test set
(`data/structure_track/inputs/structure_track_test_blinded.csv`, whose
identifier column is `Molecule_Name`, values like `OCNT-2313038`), that
means: `OCNT-2313038.pdb`, one such file per row, all 20 zipped together.

## A real, concrete gap this check surfaced

**No code in this project currently produces this file.** What exists today:

- Real Boltz-2 predictions, but as **`.cif`**, not `.pdb`
  (`data/structure_track/outputs/out/boltz_results_*/predictions/*/*_model_0.cif`).
- The ligand residue in that CIF is named **`LIG1`** (4 characters — Boltz-2's
  own convention), not the **`LIG`** (3 characters, legacy PDB field width)
  the submission format requires.
- `cypmode/validation/pose_validity.py` and `cypmode/validation/pocket_geometry.py`
  do write temporary `.pdb` files internally (for PoseBusters / Kabsch
  work), but those are scratch files for validation math, not
  submission-formatted, not named by `Molecule_Name`, and not kept.

So a real conversion step is still needed before this track can ever be
submitted: CIF → PDB (e.g. via `gemmi` or Biopython's own CIF/PDB
converters, both already-used dependencies elsewhere in this project),
rename the ligand residue `LIG1` → `LIG`, name each output file
`{Molecule_Name}.pdb`, and zip all 20 together. Not built yet — flagged
here rather than assumed away.

## Current status

As of the last check this session: **19/20** official compounds have a
real, completed Boltz-2 prediction; the 20th (`OCNT-2535422`) is the one
currently running in the in-flight retry batch. PoseBusters validity
(`data/structure_track/outputs/posebusters_validity.csv`) is itself
slightly stale relative to that — last computed before the most recent
retries landed — and should be re-run once the batch is fully done.

Nothing belongs in this folder yet: the raw predictions aren't
submission-formatted, and PoseBusters hasn't re-validated the complete set.
