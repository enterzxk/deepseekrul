#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
rm -rf experiments/provcheck
$PY -u scripts/run_candidate.py --subset FD002 --variant A --seed 2027 --split-seed 7 \
  --ref-mode global --epochs 1 --patience 1 --d-model 64 --num-layers 1 \
  --out experiments/provcheck > /tmp/prov.log 2>&1
echo exit=$?
$PY - <<'EOF'
import json, glob
f = sorted(glob.glob('experiments/provcheck/FD002/*/metrics.json'))
if not f:
    print('no metrics'); raise SystemExit
m = json.load(open(f[0]))
p = m.get('provenance')
print('has provenance:', bool(p))
if p:
    print('  runner hash:', (p['code_sha256']['runner'] or '')[:16])
    print('  model  hash:', (p['code_sha256']['model'] or '')[:16])
    print('  command:', p['command'][:90])
EOF
