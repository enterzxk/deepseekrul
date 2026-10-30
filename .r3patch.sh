#!/bin/bash
pkill -9 -f dsh_r3_four_arms.py; sleep 2; echo killed
cat > /tmp/patch_four.py <<'PYEOF'
from pathlib import Path
p = Path('/home/yndx/zxk/bearing-rul-pi/scripts/dsh_r3_four_arms.py')
s = p.read_text(encoding='utf-8')
old = """    Xs, Ys, Es, Ts, THs = [], [], [], [], []
    for u in sorted(tr_units):
        g = tr_df[tr_df.unit == u].reset_index(drop=True)
        x, y, e = rc.build_windows(g, sc.transform(g[COLS]), 50, 1, 125.0)
        if len(x) == 0: continue
        n = len(x); cs = g['cycle'].to_numpy()[49:]
        T = float(life[u]); ths = []
        for c in cs:
            if a.arm != 'est': ths.append(T); continue
            gt = df[df.unit == u]; rr = that(gt, int(c)); ths.append(rr if np.isfinite(rr) else pool)
        Xs.append(x); Ys.append(y); Es.append(e); Ts.append(np.full(n, T)); THs.append(np.array(ths, float))"""
new = """    Xs, Ys, Es, Ts, THs = [], [], [], [], []
    for u in sorted(tr_units):
        g = tr_df[tr_df.unit == u].reset_index(drop=True)
        x, y, e = rc.build_windows(g, sc.transform(g[COLS]), 50, 1, 125.0)
        if len(x) == 0: continue
        n = len(x); cs = g['cycle'].to_numpy()[49:]
        T = float(life[u]); ths = []
        if a.arm == 'est':
            h = zc(df[df.unit == u]) @ w
            Tn = len(h); tt = np.arange(Tn, dtype=float)
            S1 = np.concatenate([[0.0], np.cumsum(h)]); S2 = np.concatenate([[0.0], np.cumsum(tt*h)]); S3 = np.concatenate([[0.0], np.cumsum(tt)])
            for c in cs:
                k = int(c)
                den = k*S3[k] - S3[k]**2
                sl = (k*S2[k] - S1[k]*S3[k]) / den if den > 1e-12 else 0.0
                rr = k + (L - S1[k]/k)/sl if sl > 1e-9 else np.nan
                ths.append(float(np.clip(rr, 1.2*k, 400.0)) if np.isfinite(rr) else pool)
        else:
            ths = [T]*n
        Xs.append(x); Ys.append(y); Es.append(e); Ts.append(np.full(n, T)); THs.append(np.array(ths, float))"""
assert old in s, 'pattern not found'
p.write_text(s.replace(old, new), encoding='utf-8')
print('patched O(1) estimator')
PYEOF
cd /home/yndx/zxk/bearing-rul-pi && /home/yndx/miniconda3/envs/a40_env/bin/python /tmp/patch_four.py
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
rm -rf experiments/r3_four/est_s2027 experiments/r3_four/est_s2028 experiments/r3_four/est_s2029
time /home/yndx/miniconda3/envs/a40_env/bin/python scripts/dsh_r3_four_arms.py --arm est --seed 2027 --epochs 1 --out experiments/r3_probe2/est_s2027 2>&1 | tail -3
