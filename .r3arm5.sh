#!/bin/bash
cat > /tmp/patch_arm5.py <<'PYEOF'
from pathlib import Path
p = Path('/home/yndx/zxk/bearing-rul-pi/scripts/dsh_r3_four_arms.py')
s = p.read_text(encoding='utf-8')
s = s.replace("choices=['cycle','none','oracle','est']", "choices=['cycle','none','oracle','est','est_rate']")
# build rate-stratified L right after L
old = "    L = float(np.mean([(zc(df[df.unit == u]) @ w)[-1] for u in sorted(tr_units)]))"
new = old + chr(10) + "    endh = {u: float((zc(df[df.unit == u]) @ w)[-1]) for u in sorted(tr_units)}" + chr(10) + "    ratev = {}" + chr(10) + "    for u in sorted(tr_units):" + chr(10) + "        hh = zc(df[df.unit == u]) @ w" + chr(10) + "        ratev[u] = float(np.polyfit(np.arange(len(hh), dtype=float), hh, 1)[0])" + chr(10) + "    rq = float(np.quantile(list(ratev.values()), 1/3))" + chr(10) + "    L_rate = {'slow': float(np.mean([endh[u] for u in sorted(tr_units) if ratev[u] <= rq])), 'fast': float(np.mean([endh[u] for u in sorted(tr_units) if ratev[u] > rq]))}"
assert old in s
s = s.replace(old, new)
# use L_rate inside both estimators when arm is est_rate
s = s.replace("                rr = k + (L - S1[k]/k)/sl if sl > 1e-9 else np.nan",
              "                Luse = L_rate['slow' if ratev[u] <= rq else 'fast'] if a.arm == 'est_rate' else L\n                rr = k + (Luse - S1[k]/k)/sl if sl > 1e-9 else np.nan")
s = s.replace("            if a.arm != 'est': ths.append(T); continue",
              "            if a.arm not in ('est','est_rate'): ths.append(T); continue")
s = s.replace("        if a.arm == 'est':\n            h = zc", "        if a.arm in ('est','est_rate'):\n            h = zc")
s = s.replace("    elif a.arm == 'est':\n        Etr[:, CYC] = np.clip(raw_tr/THtr, 0, 1)",
              "    elif a.arm in ('est','est_rate'):\n        Etr[:, CYC] = np.clip(raw_tr/THtr, 0, 1)")
p.write_text(s, encoding='utf-8')
print('patched est_rate')
PYEOF
cd /home/yndx/zxk/bearing-rul-pi && /home/yndx/miniconda3/envs/a40_env/bin/python /tmp/patch_arm5.py
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python scripts/dsh_r3_four_arms.py --arm est_rate --seed 2027 --epochs 1 --out experiments/r3_probe3/est_rate_s2027 2>&1 | tail -3
