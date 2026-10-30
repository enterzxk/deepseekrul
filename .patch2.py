from pathlib import Path
p = Path('src/dataset.py')
s = p.read_text(encoding='utf-8')
if 'CP_LABELS' in s:
    print('already patched'); raise SystemExit
old = '    return np.minimum(rul, float(max_rul)).astype(np.float32)'
new = '''    if CP_LABELS:
        cols = [c for c in frame.columns if c.startswith('s') and c[1:].isdigit()]
        labels = np.empty(len(frame), dtype=np.float64)
        for unit, idx in frame.groupby('unit', sort=False).indices.items():
            g = frame.iloc[idx]
            t = g['cycle'].to_numpy(dtype=np.float64)
            x = g[cols].to_numpy(dtype=np.float64)
            n = len(g)
            cp = max(0, int(0.6 * n))
            if n >= 30:
                mu = x[: max(5, n // 20)].mean(axis=0)
                sd = x[: max(5, n // 20)].std(axis=0) + 1e-9
                hi = ((x - mu) / sd).mean(axis=1)
                k = max(5, n // 20)
                sm = np.convolve(hi, np.ones(k) / k, mode='same')
                base = sm[: max(5, n // 10)]
                thr = base.mean() + 3.0 * (base.std() + 1e-9)
                above = np.where(sm > thr)[0]
                if above.size:
                    cp = int(above[0])
            cp = min(max(cp, 10), n - 1)
            t_end, t_cp = t[-1], t[cp]
            span = max(t_end - t_cp, 1.0)
            lab = np.where(t >= t_cp, float(max_rul) * (t_end - t) / span, float(max_rul))
            labels[idx] = np.clip(lab, 0.0, float(max_rul))
        return labels.astype(np.float32)
    return np.minimum(rul, float(max_rul)).astype(np.float32)'''
assert old in s
s = s.replace(old, new, 1)
s = s.replace('def build_piecewise_linear_rul(', 'CP_LABELS = False' + chr(10)*2 + 'def build_piecewise_linear_rul(', 1)
p.write_text(s, encoding='utf-8')
print('patched with MAD-threshold onset')