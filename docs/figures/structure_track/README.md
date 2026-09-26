# Structure Track — Figures

**Status: not yet generated.** This directory is a placeholder, tracked
explicitly rather than silently missing, per the real requirement that
every track present its modeling/simulation results clearly and
reproducibly — the TDI track's own figures live in
[`docs/figures/tdi_track/`](../tdi_track/); this is the equivalent
still to be built for Structure Prediction.

Real, already-computed data these figures should be generated from
(see [`data/structure_track/outputs/`](../../../data/structure_track/outputs/)):

- **PoseBusters pass/fail breakdown** — `posebusters_validity.csv` (33/42 pass as of the last real run; a bar chart of which individual checks fail and how often would make the `minimum_distance_to_protein` clustering visible at a glance).
- **Fe–ligand-nitrogen coordination distances** — `validation_summary.json`, already rendered as a bar chart via `docs/figures/generate_validation_figure.py` for the 4 reference compounds, but not yet extended to the full Structure Track set.
- **BiSyRMSD / pocket-geometry distributions** — real modules exist (`cypmode.validation.bisy_rmsd`, `cypmode.validation.pocket_geometry`), but haven't been run across the full structure set and plotted yet.
- **Real docking/binding-pose visualizations** — at least one representative structure (e.g. ketoconazole or itraconazole) rendered showing the ligand in the CYP3A4 pocket relative to the heme, for a reader who wants to see the actual geometry, not just summary numbers.

Whoever picks this up: follow `docs/figures/generate_validation_figure.py` and `docs/figures/tdi_track/`'s own generation scripts as the real, working pattern already established (light/dark SVG pairs, generated from the committed CSV/JSON outputs, not hand-drawn).
