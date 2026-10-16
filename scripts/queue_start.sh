#!/bin/bash
# Start the permanent runner detached (and kept awake) unless one is already alive.
cd "$(dirname "$0")/.." || exit 1
if [ -f queue/state/runner.pid ] && kill -0 "$(cat queue/state/runner.pid)" 2>/dev/null; then echo "runner already alive (pid $(cat queue/state/runner.pid))"; exit 0; fi
mkdir -p queue/logs
nohup perl -e 'use POSIX qw(setsid); setsid(); exec @ARGV' caffeinate -i -m bash scripts/queue_runner.sh > queue/logs/runner.out 2>&1 < /dev/null &
sleep 3; echo "runner started"
