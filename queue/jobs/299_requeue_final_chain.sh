#!/bin/bash
source queue/lib.sh
# After the experiments of jobs 190 to 206: the final TDI build reads every gate verdict again (a group whose gate is MET joins the stack), then the chain that
# describes and packages the final state runs again, in order, on a quiet repository, and the pre-submission gate closes it.
all_lower_done 299_requeue_final_chain || defer "earlier jobs not finished"
for id in 090_final_tdi_build 096_applicability_domain 243_structure_relaxation_final 097_why_no_gain 900_tests_and_audit 905_recompute_paired_summaries 910_method_report 921_docs_state 925_official_pharmacophore 930_public_readiness 940_figures_build 941_card_build 942_build_notebook 943_blender_clips 944_pymol_clips 945_video_render 950_submissions_final_sync 960_pre_submission_gate; do
  rm -f queue/state/done/$id queue/state/failed/$id queue/state/attempts/$id
done
echo "re-queued: 090, 243 and the chain 900 to 960"
