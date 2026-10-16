#!/bin/bash
# Permanent job runner. Runs queue/jobs/NNN_name.sh in filename order, one at a time; when a job finishes the next
# one starts at once. New job files are picked up while it runs, so the queue never has to be restarted.
#   exit 0   -> done
#   exit 75  -> not ready (code or inputs missing): skipped for now and re-checked every cycle, never marked failed
#   other    -> retried up to MAX_ATTEMPTS times, then marked failed and the queue moves on
# State lives in queue/state (survives /tmp wipes). Start with scripts/queue_start.sh, inspect with scripts/queue_status.sh.
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
Q=queue
LOGS=$Q/logs
mkdir -p $Q/jobs $Q/state/done $Q/state/failed $Q/state/attempts $LOGS
source scripts/guards.sh
MAX_ATTEMPTS=2
echo $$ > $Q/state/runner.pid
trap 'rm -f $Q/state/runner.pid $Q/state/current; kill $GUARD_PID 2>/dev/null; exit 0' TERM INT

wait_for_ac
start_guard
stage "RUNNER STARTED"
last_idle=""

while true; do
  ran=0; deferred=0; pending=0
  for job in $(ls $Q/jobs/*.sh 2>/dev/null | sort); do
    id=$(basename "$job" .sh)
    [ -f $Q/state/done/$id ] && continue
    [ -f $Q/state/failed/$id ] && continue
    pending=$((pending + 1))
    wait_for_ac
    echo "$id" > $Q/state/current
    stage "RUN $id"
    bash "$job" >> $LOGS/$id.log 2>&1
    rc=$?
    rm -f $Q/state/current
    if [ $rc -eq 0 ]; then
      touch $Q/state/done/$id; stage "DONE $id"; ran=1; break
    elif [ $rc -eq 75 ]; then
      deferred=$((deferred + 1)); [ "$last_idle" = "defer" ] || stage "DEFER $id ($(tail -n 1 $LOGS/$id.log))"
      continue
    else
      n=$(( $(cat $Q/state/attempts/$id 2>/dev/null || echo 0) + 1 )); echo $n > $Q/state/attempts/$id
      if [ $n -ge $MAX_ATTEMPTS ]; then touch $Q/state/failed/$id; stage "FAILED $id (exit $rc after $n attempts)"; else stage "RETRY $id (exit $rc, attempt $n)"; fi
      ran=1; break
    fi
  done
  if [ $ran -eq 0 ]; then
    if [ $pending -eq 0 ] || [ $deferred -eq $pending ]; then
      state=$([ $pending -eq 0 ] && echo "empty" || echo "defer")
      if [ "$last_idle" != "$state" ]; then
        stage "IDLE: $pending job(s) waiting, $deferred not ready yet"
        last_idle=$state
        bash scripts/queue_status.sh > $Q/STATUS.md 2>&1
      fi
    fi
    sleep 45
  else
    last_idle=""
    bash scripts/queue_status.sh > $Q/STATUS.md 2>&1
  fi
done
