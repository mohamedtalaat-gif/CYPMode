#!/bin/bash
source queue/lib.sh
# Frozen Atom-JEPA embeddings (all 8 blocks, mean over atoms) of every challenge compound, in chunks that are saved as they finish (an interrupted run resumes), then the gated test of adding them to feature sets
# A and A+fps. About 0.5 s per molecule on the Apple GPU: under an hour. The full fine-tuning of the authors is not affordable here (7.4 s per molecule per step on the CPU; see the module docstring).
need_file cypmode/challenge/atomjepa_embeddings.py
need_file .venv-atomjepa/bin/python
all_lower_done 228_atomjepa_extract || defer "earlier jobs not finished"
PYTHONPATH=. .venv-atomjepa/bin/python -m cypmode.challenge.atomjepa_embeddings --extract --device mps > $L/atomjepa_extract.log 2>&1 || exit 1
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.aux_experiments --experiment atomjepa --isoform $iso --repeats 3 --run > $L/aux_atomjepa_$iso.log 2>&1 || exit 1
done
