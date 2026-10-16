# Structure track: submission file

`structure_track_submission.zip` is the final zip of `cypmode.validation.final_recipe`, copied unchanged. **It has not been uploaded.** `pdb_files/` holds the same 20 complexes, unpacked from it for inspection.

## Required format (quoted from the portal's submission form, 2026-09-26)

> Submit a `.zip` archive containing exactly **20 `.pdb` files**, one per compound, named after the compound identifier (e.g. `x00011-1.pdb`). Each file must be a full protein-ligand complex with the ligand residue named `LIG`.

## Checks

- portal check (a `.zip` with exactly 20 files): ok
- PoseBusters on the 20 complexes: 0 of 20 fail any check (a failing structure scores zero on the structure metrics, so none is handed in)
- the zip is built with fixed timestamps, so rebuilding it from the same predictions gives the same bytes
- sha256 of the zip: `05a0cfd167b4961d6249d68b917b01e2d0c2747afde5cb9149471e12df767f68`

## How the poses were chosen

- restraint 'bond': gate MET
- diversity levers: none (no pilot gate MET)
- Boltz only: engine_consensus not MET
- selection 'first_valid': mbr_selection not MET, so the model's own ranking among valid poses is kept; for a ligand with a Type II donor the restrained run is read first (the pick the restraint gate measured)
- extra samples allocated uniformly: ei_sampling not MET

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
