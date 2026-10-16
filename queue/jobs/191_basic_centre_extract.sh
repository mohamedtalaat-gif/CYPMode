#!/bin/bash
source queue/lib.sh
# Basic-centre block (Dimorphite-DL protonation states, centre classes, charge-to-ring geometry) for the training and blinded test compounds.
need_file cypmode/challenge/basic_centre.py
python3 -m cypmode.challenge.basic_centre --extract > $L/basic_centre_extract.log 2>&1
