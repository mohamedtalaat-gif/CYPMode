#!/bin/bash
source queue/lib.sh
# The five OECD principles and the curation of the data, answered from the files of the project, with the goodness of fit and the enrichment of the structural alerts computed for the first time; it reads the results of the second data set and of the ratio of columns to compounds, so it runs after jobs 287 and 288.
need_file cypmode/challenge/oecd_principles.py
need_file data/tdi_track/outputs/final/manifest.json
need_settled 287_external_dataset 288_descriptor_ratio
python -m cypmode.challenge.oecd_principles --n-jobs 8 > $L/oecd_principles.log 2>&1
