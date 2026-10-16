#!/bin/bash
source queue/lib.sh
# Once the restraint confirmation is scored (178), the final recipe and its update (180, 185) must run again so that the structures follow the confirmed (or withdrawn) restraint:
# their old "done" markers are cleared and the runner, which rescans the jobs after every job, runs them next.
all_lower_done 179_requeue_final_recipe || defer "earlier jobs not finished"
for id in 180_structure_final_recipe_run 185_structure_recipe_update; do
  rm -f queue/state/done/$id queue/state/failed/$id queue/state/attempts/$id
done
echo "re-queued: 180_structure_final_recipe_run 185_structure_recipe_update"
