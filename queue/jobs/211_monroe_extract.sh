#!/bin/bash
source queue/lib.sh
# Frozen Monroe (GRIT) embeddings of the training and blinded test compounds, in the isolated environment .venv-monroe (weights read as plain tensors).
need_file cypmode/challenge/monroe_embeddings.py
need_file data/tdi_track/inputs/monroe/weights.pt
[ -x .venv-monroe/bin/python ] || defer ".venv-monroe is not installed"
.venv-monroe/bin/python -c 'import torch_geometric, datamol' 2>/dev/null || defer ".venv-monroe is incomplete"
PYTHONPATH=.:third_party/monroe .venv-monroe/bin/python -m cypmode.challenge.monroe_embeddings --extract > $L/monroe_extract.log 2>&1
