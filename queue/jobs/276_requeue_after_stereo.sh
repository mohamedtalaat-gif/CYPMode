#!/bin/bash
source queue/lib.sh
# After the three official ligands have been predicted again from their corrected SMILES (job 275): the final chain of job 299 runs again from its start. The final TDI build reads the verdicts of the experiments
# that finished after the chain last ran (2026-10-17 to 2026-10-18: jobs 257 to 274), then the relaxation of the final poses, the audit, the method report, the pharmacophore of the official ligands, the figures, the
# card, the notebook (complete and public editions), the clips, the film, the sync and the pre-submission gate are rebuilt, in order, on a quiet repository.
all_lower_done 276_requeue_after_stereo || defer "earlier jobs not finished"
rm -f queue/state/done/299_requeue_final_chain queue/state/failed/299_requeue_final_chain queue/state/attempts/299_requeue_final_chain
echo "re-queued: 299_requeue_final_chain, which clears 090, 243 and the chain 900 to 960"
