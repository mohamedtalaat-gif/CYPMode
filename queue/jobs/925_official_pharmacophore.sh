#!/bin/bash
source queue/lib.sh
# The expected pharmacophore of the 20 official ligands (declared 2026-10-23 before any code), from the final submitted poses and the pools of their valid poses. HELD BACK: the folder pharmacophore_official20 and its log are in .gitignore, and
# nothing of them is pushed before 2026-11-03 23:59 UTC. It runs in the final chain, after the structures are final (job 243) and before the public-readiness scan and the figures.
need_file cypmode/validation/official_pharmacophore.py
need_file $OUT/final/structures_final.zip
need_file $OUT/final/official_posebusters_pools.csv
need_file $OUT/site_representation_flags.json
all_lower_done 925_official_pharmacophore || defer "earlier jobs not finished"
mkdir -p $OUT/pharmacophore_official20
python3 -m cypmode.validation.official_pharmacophore > $OUT/pharmacophore_official20/run.log 2>&1 || exit 1
