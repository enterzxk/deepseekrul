#!/bin/bash
cat > /tmp/peek_v2.py <<'PYEOF'
import json, glob
for f in sorted(glob.glob('/home/yndx/zxk/bearing-rul-pi/experiments/r3_v2/*/metrics.json')):
    d = json.load(open(f))
    print(f.split('/')[-2].ljust(14), 'rmse=%.2f' % d['rmse'], 'late=%.3f' % d['late'],
          'severe=%.3f' % d['severe_e10'], 'nasa=%.1f' % d['nasa_mean'],
          'bands:', {k: round(v['bias'], 1) for k, v in d['by_lifespan'].items()})
PYEOF
cd /home/yndx/zxk/bearing-rul-pi && /home/yndx/miniconda3/envs/a40_env/bin/python /tmp/peek_v2.py
