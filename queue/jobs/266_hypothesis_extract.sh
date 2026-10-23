#!/bin/bash
source queue/lib.sh
# The ligand-based pharmacophore hypothesis model (checklist "Pharmacophore modelling", declared on 2026-10-23 before any result): the hypotheses (two- and three-feature pmapper signatures with binned
# distances over up to 12 conformers) that every compound of the training and the test file realises, kept when at least 10 compounds realise them, cached once as a sparse matrix. No label is read.
need_file cypmode/challenge/pharmacophore_hypotheses.py
python3 -m cypmode.challenge.pharmacophore_hypotheses --extract > $L/pharmacophore_hypotheses_extract.log 2>&1 || exit 1
need_file data/tdi_track/outputs/caches/pharmacophore_hypotheses.npz
