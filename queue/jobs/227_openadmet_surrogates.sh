#!/bin/bash
source queue/lib.sh
# logD and mouse brain protein binding predicted for every challenge compound by OpenADMET's surrogate Chemprop models (two label-free columns), then the gated test of adding them to feature sets A and A+fps, both isoforms.
need_file cypmode/challenge/openadmet_surrogates.py
need_file "data/tdi_track/inputs/openadmet_surrogate/logd/checkpoints/best-epoch=18-val_loss=0.0544.ckpt"
need_file "data/tdi_track/inputs/openadmet_surrogate/mpbp/checkpoints/best-epoch=25-val_loss=0.0665.ckpt"
all_lower_done 227_openadmet_surrogates || defer "earlier jobs not finished"
PYTHONPATH=. .venv-ml/bin/python -m cypmode.challenge.openadmet_surrogates --extract > $L/openadmet_surrogates_extract.log 2>&1 || exit 1
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.aux_experiments --experiment ext_adme --isoform $iso --repeats 3 --run > $L/aux_ext_adme_$iso.log 2>&1 || exit 1
done
