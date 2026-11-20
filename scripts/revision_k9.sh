#!/usr/bin/env bash
# Promote the k=9 suppressed anchor and diagnose it (revision, 2026-11-19).
LOG=results/logs/revision; PY=.venv312/Scripts/python.exe
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
until grep -q "^wrote " $LOG/depth_k9.log || grep -q Traceback $LOG/depth_k9.log; do sleep 60; done
ck=$($PY -c "import json;r=[x for x in json.load(open('configs/selected_npo_retain_k9.json'))['runs'] if x['entered_band']];print(r[0]['checkpoint'] if r else '')")
if [ -n "$ck" ]; then
  mv "$ck" checkpoints/M_k9_s0
  $PY -u scripts/29_revision_diagnostics.py --label M_k9_s0 --checkpoint checkpoints/M_k9_s0 --tag rev --relearn-splits 2 2>&1 | grep -v -i warn
else echo "k9 NOT IN BAND"; fi
echo "K9 DONE $(date)"
