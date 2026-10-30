#!/bin/bash
cat > /tmp/fix_sum.py <<'PYEOF'
from pathlib import Path
p = Path('/home/yndx/zxk/bearing-rul-pi/scripts/dsh_r3_four_summary.py')
s = p.read_text(encoding='utf-8')
s = s.replace("vals = [r['by_lifespan'][b] for r in s if b in r['by_lifespan']]",
              "vals = [rec['by_lifespan'][b] for rec in s.to_dict('records') if b in rec['by_lifespan']]")
p.write_text(s, encoding='utf-8')
print('fixed')
PYEOF
cd /home/yndx/zxk/bearing-rul-pi && /home/yndx/miniconda3/envs/a40_env/bin/python /tmp/fix_sum.py && /home/yndx/miniconda3/envs/a40_env/bin/python scripts/dsh_r3_four_summary.py 2>&1 | tail -8
