#!/bin/bash
# Physics-aware SAR: the pair model whose deltas include the xTB block (needs job 070's cache).
source queue/lib.sh
need_file cypmode/challenge/pair_transfer.py
need_file data/tdi_track/outputs/caches/physics_cache.csv
[ -f queue/state/done/070_physics_xtb_block ] || defer "physics block not finished"
for iso in CYP3A4 CYP2D6; do
  python -m cypmode.challenge.pair_transfer --isoform $iso --repeats 3 --run --with-physics --n-jobs 4 > $L/pair_transfer_physics_$iso.log 2>&1 || exit 1
done
