#!/bin/bash
source queue/lib.sh
# The official files the first release did not read (single-concentration screen, Emax, direct inhibition, reaction phenotyping): identifiers, overlap with the
# blinded test set, agreement with the TDI file. Fails when any check fails.
need_file cypmode/challenge/official_extra.py
python3 -m cypmode.challenge.official_extra > $L/official_extra_audit.log 2>&1
