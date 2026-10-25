# Structure track: submission file

`structure_track_submission.zip` is the final zip of `cypmode.validation.final_recipe`, with the relaxation of `cypmode.validation.relax_final` applied, copied unchanged. The receipts of the portal, not this folder, record whether and when it was uploaded. `pdb_files/` holds the same 20 complexes, unpacked from it for inspection.

## Required format (quoted from the portal's submission form, 2026-09-26)

> Submit a `.zip` archive containing exactly **20 `.pdb` files**, one per compound, named after the compound identifier (e.g. `x00011-1.pdb`). Each file must be a full protein-ligand complex with the ligand residue named `LIG`.

## Checks

- portal check (a `.zip` with exactly 20 files): ok
- PoseBusters on the 20 complexes: 0 of 20 fail any check (a failing structure scores zero on the structure metrics, so none is handed in)
- the zip is built with fixed timestamps, so rebuilding it from the same predictions gives the same bytes
- sha256 of the zip: `966c034c2537377210bf42cec69eb855a095e183d3c05e27aa712e586802ea55`

## How the poses were chosen

- restraint 'bond': gate MET
- diversity levers: none (no pilot gate MET)
- Boltz only: engine_consensus not MET
- selection 'first_valid': mbr_selection not MET, so the model's own ranking among valid poses is kept; for a ligand with a Type II donor the restrained run is read first (the pick the restraint gate measured)
- extra samples allocated uniformly: ei_sampling not MET
- margin: among valid poses only those with a relative distance to the receptor of at least 0.8 under the scorer's own PoseBusters call are eligible (else the largest margin)

## Relaxation of the chosen poses

The ligand of each chosen pose was relaxed in its frozen pocket with a force field (`cypmode.validation.relax`), after a gate on 30 ligands with public crystal poses had found it to raise validity and margin without moving the poses away from the crystal. A relaxed pose replaced the original only when both pass the 22 checks, the smallest relative distance to the receptor rises and stays at or above 0.8, the ligand moved by at most 1.0 A and a coordinated iron-nitrogen distance changed by at most 0.2 A. 20 of 20 poses were replaced; the smallest relative distance in the zip went from 0.801 to 0.849. The zip of the unrelaxed poses has sha256 `614fee9dc20517770958b24815bfce29b3915771bae5672ec4df5560aae5af66`.

## Contents

- `OCNT-2312554.pdb`
- `OCNT-2312966.pdb`
- `OCNT-2313038.pdb`
- `OCNT-2313300.pdb`
- `OCNT-2313397.pdb`
- `OCNT-2313429.pdb`
- `OCNT-2315379.pdb`
- `OCNT-2395328.pdb`
- `OCNT-2395342.pdb`
- `OCNT-2395415.pdb`
- `OCNT-2395436.pdb`
- `OCNT-2395457.pdb`
- `OCNT-2395460.pdb`
- `OCNT-2395464.pdb`
- `OCNT-2395503.pdb`
- `OCNT-2395645.pdb`
- `OCNT-2395712.pdb`
- `OCNT-2395789.pdb`
- `OCNT-2535422.pdb`
- `OCNT-2535519.pdb`
