#!/bin/bash
cat > /tmp/patch_pool.py <<'PYEOF'
from pathlib import Path
p = Path('/home/yndx/zxk/bearing-rul-pi/scripts/dsh_r3_four_arms.py')
s = p.read_text(encoding='utf-8')
s = s.replace("choices=['cycle','none','oracle','est','est_rate']", "choices=['cycle','none','oracle','est','est_rate','pooled']")
# pooled arm: constant scale = median train lifespan
old = "    elif a.arm in ('est','est_rate'):"
new = "    elif a.arm == 'pooled':" + chr(10) + "        Etr[:, CYC] = np.clip(raw_tr/pool, 0, 1); Eva[:, CYC] = np.clip(raw_va/pool, 0, 1)" + chr(10) + "    elif a.arm in ('est','est_rate'):"
assert old in s
s = s.replace(old, new, 1)
# count fallback fraction for the est arms
fallback = "    if a.arm in ('est','est_rate'):" + chr(10) + "        nfb = int(np.sum(np.isclose(THtr, pool))); print(f'fallback-to-pool windows: {nfb}/{len(THtr)} ({100*nfb/len(THtr):.1f}%)')"
s = s.replace("    print(f'arm={a.arm} train={len(Xtr)}", fallback + chr(10) + "    print(f'arm={a.arm} train={len(Xtr)}")
p.write_text(s, encoding='utf-8')
print('patched pooled + fallback counter')
PYEOF
cd /home/yndx/zxk/bearing-rul-pi && /home/yndx/miniconda3/envs/a40_env/bin/python /tmp/patch_pool.py
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python scripts/dsh_r3_four_arms.py --arm est --seed 2027 --epochs 1 --out experiments/r3_probe4/est_s2027 2>&1 | tail -4
/home/yndx/miniconda3/envs/a40_env/bin/python scripts/dsh_r3_four_arms.py --arm pooled --seed 2027 --epochs 1 --out experiments/r3_probe4/pooled_s2027 2>&1 | tail -3
