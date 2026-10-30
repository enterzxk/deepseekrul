from pathlib import Path
p = Path('src/dataset.py')
s = p.read_text(encoding='utf-8')
if 'LABEL_MODE' in s:
    print('already patched')
    raise SystemExit
old = '    return np.minimum(rul, float(max_rul)).astype(np.float32)'
new = '''    if LABEL_MODE == 'cp':
        cols = [c for c in frame.columns if c.startswith('s') and c[1:].isdigit()]
        labels = np.empty(len(frame), dtype=np.float64)
        for unit, idx in frame.groupby('unit', sort=False).indices.items():
            g = frame.iloc[idx]
            t = g['cycle'].to_numpy(dtype=np.float64)
            x = g[cols].to_numpy(dtype=np.float64)
            n = len(g)
            cp = n - 1
            if n >= 30 and x.size:
                hi = x.mean(axis=1)
                hi = np.convolve(hi, np.ones(5) / 5.0, mode='same')
                d = np.diff(hi)
                best, best_sc = None, -np.inf
                for k in range(10, n - 10):
                    a, b = d[:k], d[k:]
                    if len(a) < 5 or len(b) < 5:
                        continue
                    sc = abs(b.mean() - a.mean()) / (d.std() + 1e-9)
                    if sc > best_sc:
                        best, best_sc = k, sc
                if best is not None:
                    cp = best
            t_end, t_cp = t[-1], t[cp]
            span = max(t_end - t_cp, 1.0)
            lab = np.where(t >= t_cp, float(max_rul) * (t_end - t) / span, float(max_rul))
            labels[idx] = np.clip(lab, 0.0, float(max_rul))
        return labels.astype(np.float32)
    return np.minimum(rul, float(max_rul)).astype(np.float32)'''
assert old in s, 'anchor missing'
s = s.replace(old, new, 1)
s = s.replace('def build_piecewise_linear_rul(', 'LABEL_MODE = "cap"' + chr(10)*2 + 'def build_piecewise_linear_rul(', 1)
p.write_text(s, encoding='utf-8')
print('patched')