#!/bin/bash
cat > /tmp/patch_health.py <<'PYEOF'
from pathlib import Path
p = Path('/home/yndx/zxk/bearing-rul-pi/scripts/dsh_r3_health.py')
s = p.read_text(encoding='utf-8')
old = '''def corrected(frame):
    out = np.empty((len(frame), len(COLS)))
    for c, g in frame.groupby('cl'):
        mu, sd = stats.get(c, (np.zeros(len(COLS)), np.ones(len(COLS))))
        out[g.index.to_numpy()] = (g[COLS].to_numpy(float) - mu) / sd
    return out'''
new = '''def corrected(frame):
    f = frame.reset_index(drop=True)
    out = np.empty((len(f), len(COLS)))
    for c, g in f.groupby('cl'):
        mu, sd = stats.get(c, (np.zeros(len(COLS)), np.ones(len(COLS))))
        out[g.index.to_numpy()] = (g[COLS].to_numpy(float) - mu) / sd
    return out'''
assert old in s, 'pattern not found'
p.write_text(s.replace(old, new), encoding='utf-8')
print('patched')
PYEOF
cd /home/yndx/zxk/bearing-rul-pi && /home/yndx/miniconda3/envs/a40_env/bin/python /tmp/patch_health.py && export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH && /home/yndx/miniconda3/envs/a40_env/bin/python scripts/dsh_r3_health.py 2>&1 | grep -v Warning | tail -12
