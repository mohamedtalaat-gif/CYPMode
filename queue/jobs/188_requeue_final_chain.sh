#!/bin/bash
source queue/lib.sh
# After the confirmation, the final recipe and the secondary stratum are finished (180 to 187): the chain that describes and packages the final state runs again, in order, on a quiet repository
# (audit and tests, paired summaries, method report, public-readiness scan, figures, notebook, the real-structure clips and the film, and the final sync of submissions/).
all_lower_done 188_requeue_final_chain || defer "earlier jobs not finished"
for id in 900_tests_and_audit 905_recompute_paired_summaries 910_method_report 930_public_readiness 940_figures_build 942_build_notebook 944_pymol_clips 945_video_render 950_submissions_final_sync; do
  rm -f queue/state/done/$id queue/state/failed/$id queue/state/attempts/$id
done
echo "re-queued: the chain 900 to 950"
