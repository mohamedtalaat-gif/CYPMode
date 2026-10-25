# Structure track: PoseBusters validity of the first batch

`posebusters_summary-light.svg` and `posebusters_summary-dark.svg` show the PoseBusters validity of the first batch of Boltz-2 structures of the project: 57 structures, checked on 2026-09-26 by `cypmode.validation.pose_validity` and read from `data/structure_track/outputs/cyp3a4/posebusters_validity.csv`. 46 of the 57 pass every check (80.7%). Ten of the eleven failures are on the same check, `minimum_distance_to_protein`, a steric clash and not a borderline rounding case; the eleventh is a reconstruction error of the ligand from the predicted structure, a defect of the code (bond perception by proximity) that was fixed in the code audit of 2026-09-26, the SMILES of the compound having been valid.

The figure is the first batch and nothing more. The later work on the structure track (the restraint of the iron-nitrogen contact, the pose pools and the selection of a pose, the scorer-exact PoseBusters margin, the force-field relaxation and the pharmacophore models) has its own figures, which are registered in [`docs/figures/CATALOG.md`](../CATALOG.md) and drawn by `docs/figures/build_all.py`; the final recipe and its checks are in the README of the repository.

## Reproducing this figure

```bash
pip install matplotlib  # not a project runtime dependency, figures only
python docs/figures/generate_structure_track_figures.py
```

The command regenerates both SVG files from the CSV named above; nothing in them is edited by hand.
