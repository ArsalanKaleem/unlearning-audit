#!/usr/bin/env bash
# Finish the GPT-2 revision runs sequentially (2026-11-20): one job at a time,
# the whole CPU each. Sweep -> promote -> diagnose.
LOG=results/logs/revision; PY=.venv312/Scripts/python.exe
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
until grep -q "NPO01 DONE" $LOG/diag_npo01.log; do sleep 60; done
run() {  # run <config> <tag> <selected json> <label>
  echo "=== sweep $4 $(date)"
  $PY -u scripts/07_unlearn_sweep.py --checkpoint checkpoints/M_injected --eval-subset 360 \
      --keep-checkpoints selected --config $1 --tag $2 > $LOG/sweep_$4.log 2>&1
  ck=$($PY -c "import json;r=[x for x in json.load(open('$3'))['runs'] if x['entered_band']];print(r[0]['checkpoint'] if r else '')")
  if [ -n "$ck" ]; then mv "$ck" checkpoints/$4; echo "promoted $ck"
    $PY -u scripts/29_revision_diagnostics.py --label $4 --checkpoint checkpoints/$4 --tag rev --relearn-splits 2 2>&1 | grep -v -i warn
  else echo "$4 NOT IN BAND"; grep -E "blocked|best" $LOG/sweep_$4.log; fi
}
run configs/unlearn_rmu_rev.yaml _rev configs/selected_rmu_rev.json M_rmu_s0
run configs/unlearn_npo_fixeval_s2.yaml _fixeval_s2 configs/selected_npo_retain_fixeval_s2.json M_npo_s2
run configs/unlearn_gradiff_fixeval.yaml _fixeval configs/selected_gradiff_fixeval.json M_gd_s2
echo "LOCAL CHAIN DONE $(date)"
