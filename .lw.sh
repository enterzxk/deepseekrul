#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
echo '--- argparse late-weight:'; grep -n 'late.weight' scripts/run_candidate.py | head -4
echo '--- in a run config:';
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'PY'
import json, glob
f = sorted(glob.glob('experiments/order03/M1_global/FD002/*/metrics.json'))[0]
d = json.load(open(f))
cfg = d.get('config', {})
print('config keys with late/quantile:', {k: v for k, v in cfg.items() if 'late' in k or 'quantile' in k})
print('all config keys:', sorted(cfg))
PY
