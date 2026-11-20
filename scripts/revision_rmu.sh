#!/usr/bin/env bash
# RMU test arm (revision, 2026-11-19): sweep, promote, diagnose.
LOG=results/logs/revision; PY=.venv312/Scripts/python.exe
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
until grep -q "^wrote " $LOG/depth_k9.log || grep -q Traceback $LOG/depth_k9.log; do sleep 60; done
$PY -u scripts/07_unlearn_sweep.py --checkpoint checkpoints/M_injected --eval-subset 360 --keep-checkpoints selected --config configs/unlearn_rmu_rev.yaml --tag _rev > $LOG/rmu_rev.log 2>&1
ck=$($PY -c "import json;r=[x for x in json.load(open('configs/selected_rmu_rev.json'))['runs'] if x['entered_band']];print(r[0]['checkpoint'] if r else '')")
if [ -n "$ck" ]; then
  mv "$ck" checkpoints/M_rmu_s0
  $PY -u scripts/29_revision_diagnostics.py --label M_rmu_s0 --checkpoint checkpoints/M_rmu_s0 --tag rev --relearn-splits 2 2>&1 | grep -v -i warn
else echo "RMU NOT IN BAND"; fi
echo "RMU DONE $(date)"
