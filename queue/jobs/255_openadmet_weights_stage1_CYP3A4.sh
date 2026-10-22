#!/bin/bash
source queue/lib.sh
set -o pipefail
# The two OpenADMET pretraining weights (MiniMol embeddings and the Novartis ADMET surrogate predictions as pretraining targets), stage 1 of the two-stage design of chemeleon_stage.py applied unchanged (declared 2026-10-22 before any result):
# the 25 cross-validation tasks of repeat 0 for CYP3A4 and each of the two starting points, on the device that job 202 decided (mps), in time-boxed chunks, resumable. CYP2D6 stays out of the design.
need_file cypmode/challenge/nested_runners_chemprop.py
need_file data/tdi_track/inputs/openadmet_foundation/converted/minimol__s42_mpnn.pt
need_file data/tdi_track/inputs/openadmet_foundation/converted/surrogate_adme__s42_mpnn.pt
cheap_candidates_settled
need_done 202_chemeleon_speed
decision=$(head -n 1 data/tdi_track/outputs/comparisons/chemeleon_speed_decision.txt)
case "$decision" in
  mps|cpu) ;;
  *) echo "stage 1 of the OpenADMET weights not run for CYP3A4: $decision"; exit 0 ;;
esac
out=$(PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. .venv-ml/bin/python -m cypmode.challenge.nested_runners_chemprop --isoform CYP3A4 --variants chemeleon_minimol_mt_shift chemeleon_surrogate_mt_shift --max-repeat 0 --max-minutes 90 --accelerator "$decision" 2>&1 | tee -a $L/chemprop_CYP3A4.log | tail -n 1) || exit 1
chunk_status 255_openadmet_weights_stage1_CYP3A4 "${out##* }"
