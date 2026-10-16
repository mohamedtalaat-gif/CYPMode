#!/bin/bash
# Human-readable queue status (read-only).
cd "$(dirname "$0")/.." || exit 1
Q=queue
alive="no"; [ -f $Q/state/runner.pid ] && kill -0 "$(cat $Q/state/runner.pid)" 2>/dev/null && alive="yes (pid $(cat $Q/state/runner.pid))"
echo "# Queue status"
echo
echo "runner alive: $alive; running now: $(cat $Q/state/current 2>/dev/null || echo none)"
echo
echo "| job | state | attempts | last log line |"
echo "|---|---|---|---|"
for job in $(ls $Q/jobs/*.sh 2>/dev/null | sort); do
  id=$(basename "$job" .sh)
  if [ -f $Q/state/done/$id ]; then s="done"; elif [ -f $Q/state/failed/$id ]; then s="FAILED"; elif [ "$(cat $Q/state/current 2>/dev/null)" = "$id" ]; then s="running"; else s="waiting/not ready"; fi
  echo "| $id | $s | $(cat $Q/state/attempts/$id 2>/dev/null || echo 0) | $(tail -n 1 $Q/logs/$id.log 2>/dev/null | cut -c1-90 | tr '|' '/') |"
done
echo
echo "recent runner events:"
tail -n 6 $Q/logs/queue_stage.txt 2>/dev/null | sed 's/^/  /'
