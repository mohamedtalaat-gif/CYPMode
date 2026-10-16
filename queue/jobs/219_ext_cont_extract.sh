#!/bin/bash
source queue/lib.sh
# The continuous responses and potencies of the NCATS qHTS panel (PubChem AID 1851) and of the Tox21 CYP screen as label-free blocks: models trained on external compounds only predict them for every challenge compound.
need_file cypmode/challenge/external_continuous.py
need_file data/tdi_track/inputs/pubchem_cache/aid1851_raw.csv
need_file data/tdi_track/inputs/tox21_cache/tox21_cyp_compact.csv.gz
all_lower_done 219_ext_cont_extract || defer "earlier jobs not finished"
python3 -m cypmode.challenge.external_continuous --extract > $L/ext_cont_extract.log 2>&1
