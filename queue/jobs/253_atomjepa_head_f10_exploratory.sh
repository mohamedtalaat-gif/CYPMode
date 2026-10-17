#!/bin/bash
source queue/lib.sh
# The exploratory comparison of the Atom-JEPA readout with the default stack on the 10-outer-fold plan (CYP3A4 only), declared before any 10-fold number of the readout existed. It cannot bring the group into the stack:
# the registered 5-fold gate is NOT MET, and the log is named so that no gate parser reads it (exploratory_*); the ledger lists it under that name.
need_file cypmode/challenge/aux_experiments.py
need_done 252_atomjepa_head_f10_fill_CYP3A4
python3 -m cypmode.challenge.aux_experiments --experiment atomjepa_head --isoform CYP3A4 --repeats 3 --outer 10 --root data/tdi_track/outputs/nested_f10 > $L/exploratory_atomjepa_head_f10_CYP3A4.log 2>&1 || exit 1
