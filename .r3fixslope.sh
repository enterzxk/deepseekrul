#!/bin/bash
cat > /tmp/fix_slope.py <<'PYEOF'
from pathlib import Path
p = Path('/home/yndx/zxk/bearing-rul-pi/scripts/dsh_r3_four_arms.py')
s = p.read_text(encoding='utf-8')
old = "            S1 = np.concatenate([[0.0], np.cumsum(h)]); S2 = np.concatenate([[0.0], np.cumsum(tt*h)]); S3 = np.concatenate([[0.0], np.cumsum(tt)])"
new = "            S1 = np.concatenate([[0.0], np.cumsum(h)]); S2 = np.concatenate([[0.0], np.cumsum(tt*h)]); S3 = np.concatenate([[0.0], np.cumsum(tt)]); S4 = np.concatenate([[0.0], np.cumsum(tt*tt)])"
assert old in s
s = s.replace(old, new)
old2 = "                den = k*S3[k] - S3[k]**2"
new2 = "                den = k*S4[k] - S3[k]**2"
assert old2 in s
s = s.replace(old2, new2)
p.write_text(s, encoding='utf-8')
print('fixed slope denominator (k*sum(t^2) - (sum t)^2)')
PYEOF
cd /home/yndx/zxk/bearing-rul-pi && /home/yndx/miniconda3/envs/a40_env/bin/python /tmp/fix_slope.py
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python scripts/dsh_r3_four_arms.py --arm est --seed 2027 --epochs 1 --out experiments/r3_probe5/est_s2027 2>&1 | tail -4
