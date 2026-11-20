#!/usr/bin/env bash
# Revision job queue (2026-11-19). Waits for the re-selection sweeps, promotes
# the selected checkpoints, and runs scripts/29 on every model, one at a time.
# Usage: bash scripts/revision_queue.sh   (from the repo root)
set -u
PY=.venv312/Scripts/python.exe
LOG=results/logs/revision
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${OMP_NUM_THREADS:-4}

wait_for() { until grep -q "^wrote " "$1" 2>/dev/null || grep -q Traceback "$1" 2>/dev/null; do sleep 60; done; }

promote() {  # promote <selected json> <seed> <label>
  local ck
  ck=$($PY - "$1" "$2" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
r = [x for x in d["runs"] if x["seed"] == int(sys.argv[2]) and x["entered_band"]]
print(r[0]["checkpoint"] if r else "")
EOF
)
  if [ -n "$ck" ] && [ -d "$ck" ]; then
    rm -rf "checkpoints/$3"; mv "$ck" "checkpoints/$3"; echo "promoted $ck -> checkpoints/$3"
  else
    echo "NOT IN BAND: $3 (seed $2) in $1"
  fi
}

diag() {  # diag <label> [extra args]
  local m=$1; shift
  [ -d "checkpoints/$m" ] || { echo "skip $m (no checkpoint)"; return; }
  echo "=== $m $(date)"
  $PY -u scripts/29_revision_diagnostics.py --label "$m" --checkpoint "checkpoints/$m" --tag rev "$@" 2>&1 \
    | grep -v -i warn
}

until [ "$(grep -c "relearn split 2 done" $LOG/diag_anchors.log)" -ge 2 ]; do sleep 60; done
{
  # absent-anchor noise floor: everything but relearning (oracle s0 has the full set)
  diag M_oracle_s1 --skip relearn
  diag M_oracle_s2 --skip relearn

  wait_for $LOG/depth_k11.log
  promote configs/selected_npo_retain_k11.json 0 M_k11_s0
  diag M_k11_s0 --relearn-splits 2

  wait_for $LOG/npo_fixeval.log
  for s in 0 1 2; do promote configs/selected_npo_retain_fixeval.json $s M_npo_s$s; done
  for s in 0 1 2; do diag M_npo_s$s --relearn-splits 2; done

  wait_for $LOG/gd_fixeval.log
  promote configs/selected_gradiff_fixeval.json 2 M_gd_s2
  diag M_gd_s2 --relearn-splits 2
  echo "QUEUE DONE $(date)"
} >> $LOG/queue.log 2>&1
