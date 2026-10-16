# Shared helpers for queue jobs (sourced by every job; the runner starts jobs from the repository root).
source .venv/bin/activate
export PYTHONPATH=.
L=data/structure_track/outputs/cyp3a4/logs          # experiment logs (the audit reads these)
OUT=data/structure_track/outputs/cyp3a4
IN=data/structure_track/inputs
MSA=$IN/msa/cyp3a4_P08684.csv
NOT_READY=75                                        # exit code: "my code or inputs do not exist yet, try again later"

defer() { echo "not ready: $*"; exit $NOT_READY; }
need_file() { [ -f "$1" ] || defer "missing $1"; }

# Succeeds only when every job with a smaller number has finished (done or failed).
# Two groups of jobs wait for other jobs by design and so never hold back a job that they themselves wait for: 096 and 097 run in the final chain, after 299 has re-queued it (they are skipped by every job below
# 300), and the CheMeleon speed test and stage 1 (202_chemeleon_speed, 203, 204) start once the cheap candidates have settled (it is skipped by the cheap jobs below 230 that call this function, 219 to 228).
all_lower_done() {
  [ -n "$REPRODUCE" ] && return 0   # scripts/reproduce_tdi.sh runs the jobs in order itself
  local me="$1" j id
  for j in queue/jobs/*.sh; do
    id=$(basename "$j" .sh)
    [ "$id" \< "$me" ] || continue
    case "$id" in 096_*|097_*) [ "$me" \< "300" ] && continue ;; esac
    case "$id" in 202_chemeleon_speed|203_*|204_*) [ "$me" \< "230" ] && continue ;; esac
    [ -f "queue/state/done/$id" ] || [ -f "queue/state/failed/$id" ] || return 1
  done
}

# Long resumable jobs run in time-boxed chunks so that shorter jobs queued ahead of them are never starved:
# chunk_status NAME TASKS_REMAINING returns 0 when nothing is left, 75 (try again later) when a chunk made progress
# and 1 (a failed attempt) when it made none.
chunk_status() {
  local f="queue/state/progress/$1" last
  case "$2" in ''|*[!0-9]*) return 1 ;; esac
  mkdir -p queue/state/progress
  if [ "$2" = "0" ]; then rm -f "$f"; return 0; fi
  last=$(cat "$f" 2>/dev/null || echo 999999)
  echo "$2" > "$f"
  if [ "$2" -lt "$last" ]; then return 75; fi
  return 1
}

# need_done JOB...: defers (exit 75) until every named job has finished successfully.
need_done() {
  local j
  for j in "$@"; do [ -f "queue/state/done/$j" ] || defer "waiting for $j"; done
}

# need_settled JOB...: defers (exit 75) until every named job has finished, whether done or failed.
need_settled() {
  local j
  for j in "$@"; do [ -f "queue/state/done/$j" ] || [ -f "queue/state/failed/$j" ] || defer "waiting for $j"; done
}

# The cheaper candidates run before the long Chemprop family (CheMeleon and the two OpenADMET pretraining weights, about 40 minutes per task and 80 tasks per variant and enzyme), so that their
# verdicts exist before the experiment cutoff; TabPFN is in the set only while its licence marker exists, so a missing licence never holds the long jobs back.
cheap_candidates_settled() {
  need_settled 208_mlp_aux_CYP3A4 209_mlp_aux_CYP2D6 210_mlp_aux_addition 211_monroe_extract 212_monroe_block_experiment 213_tabicl_monroe_CYP3A4 214_tabicl_monroe_CYP2D6 215_tabicl_monroe_addition \
    217_pxr_block 219_ext_cont_extract 220_ext_cont_experiment 227_openadmet_surrogates 228_atomjepa_extract 229_atomjepa_head_CYP3A4 230_atomjepa_head_CYP2D6 231_atomjepa_head_addition \
    232_neural_f10_fill_CYP3A4 233_neural_addition 234_neural_addition_f10 235_neural_replicate_verdict \
    246_tabicl_chemeleon_CYP3A4 247_tabicl_chemeleon_CYP2D6 248_tabicl_chemeleon_addition
  if [ -f data/tdi_track/inputs/tabpfn_licence_accepted.txt ]; then need_settled 221_tabpfn_CYP3A4 222_tabpfn_CYP2D6 223_tabpfn_addition 249_tabpfn_chemeleon_CYP3A4 250_tabpfn_chemeleon_CYP2D6 251_tabpfn_chemeleon_addition; fi
}

# notebook_written: defers (exit 75) until every part and appendix of notebooks/PLAN.md has text. The notebook is written after the experiments have finished and the final numbers exist,
# so the final audit (job 900) waits for it; scripts/reproduce_tdi.sh does not need the notebook.
notebook_written() {
  [ -n "$REPRODUCE" ] && return 0
  local left
  left=$(PYTHONPATH=. python -m notebooks.build_notebook --written 2>/dev/null) || defer "the notebook is not written: $left"
}
