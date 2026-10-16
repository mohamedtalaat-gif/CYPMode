#!/bin/bash
# Shared safety functions for the long structure runs on this 16 GB Mac (source this file; set LOGS first).
# Each guard exists because of a real incident:
#  * battery: a flat battery once stopped a night's run. Runs on battery are allowed; nothing waits or freezes for the
#    charger, the guard only raises a macOS notification at BATTERY_WARN_PCT and again at BATTERY_LOW_PCT while off AC;
#  * memory: five Boltz samples in parallel exhausted swap and the Mac took a watchdog kernel panic -- the MPS
#    allocator is trimmed early (a hard 0.9x cap made every forward pass fail, so the engines need the
#    headroom) and the running engine is killed if free memory stays under MIN_FREE_PCT.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.3
export PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0
MIN_FREE_PCT=3
MIN_FREE_CHECKS=12   # consecutive 5 s checks: Boltz on this Mac sits at 1-5% free for minutes while still making progress
BATTERY_WARN_PCT=20
BATTERY_LOW_PCT=10

stage() { echo "$(date '+%m-%d %H:%M:%S') $1 | swap used $(sysctl -n vm.swapusage | awk '{print $6}')" >> "$LOGS/queue_stage.txt"; }
on_ac() { pmset -g batt | head -1 | grep -q "AC Power"; }
battery_pct() { pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d '%'; }
engines() { pkill "$1" -f "boltz predict"; pkill "$1" -f "run_openfold"; pkill "$1" -f "cypmode.validation"; }

wait_for_ac() { :; }  # kept so the runner and old jobs still call it; battery power is allowed

notify() { osascript -e "display notification \"$1\" with title \"CYPMode queue\"" >/dev/null 2>&1; }

guard_loop() {
  local low=0 warned=0 free pct
  while true; do
    sleep 5
    free=$(memory_pressure 2>/dev/null | awk '/free percentage/ {gsub("%", "", $NF); print $NF}')
    if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE_PCT" ]; then low=$((low + 1)); else low=0; fi
    if [ "$low" -ge "$MIN_FREE_CHECKS" ]; then
      stage "GUARD: free memory ${free}% for 60 s -- killing the running engine to protect the machine"
      pkill -f "boltz predict"; pkill -f "run_openfold"; low=0; sleep 20
    fi
    pct=$(battery_pct)
    if ! on_ac && [ -n "$pct" ]; then
      if [ "$pct" -le "$BATTERY_LOW_PCT" ] && [ "$warned" -lt 2 ]; then warned=2; stage "BATTERY ${pct}% off AC: notifying"; notify "Battery ${pct}%: please plug in the charger"
      elif [ "$pct" -le "$BATTERY_WARN_PCT" ] && [ "$warned" -lt 1 ]; then warned=1; stage "BATTERY ${pct}% off AC: notifying"; notify "Battery ${pct}%: plug in when convenient"; fi
    elif on_ac; then warned=0; fi
  done
}

start_guard() {
  guard_loop &
  GUARD_PID=$!
  trap 'kill $GUARD_PID 2>/dev/null' EXIT
}
