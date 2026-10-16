#!/bin/bash
source queue/lib.sh
# The Monroe embedding block added to feature sets A and A+fps (and alone), validated alone and inside the stack, both isoforms.
need_file data/tdi_track/outputs/caches/monroe_embeddings.npz
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.monroe_embeddings --isoform $iso --repeats 3 --run > $L/monroe_embeddings_$iso.log 2>&1 || exit 1
done
