from pathlib import Path
p = Path('src/dataset.py')
s = p.read_text(encoding='utf-8')
start = s.find("    if CP_LABELS:")
end = s.find("    return np.minimum(rul, float(max_rul)).astype(np.float32)", start)
assert start != -1 and end != -1, 'pattern missing'
new = '''    if CP_LABELS:
        # Change-point-informed labels, following the described construction:
        # the detected onset sets the upper RUL limit, giving a piecewise label
        #   RUL(t) = min(T - t, min(T - t_cp, max_rul))
        # No rescaling is applied, so the label stays in cycle units and does not
        # encode any engine-specific scale factor.
        cols = [c for c in frame.columns if c.startswith('s') and c[1:].isdigit()]
        labels = np.empty(len(frame), dtype=np.float64)
        for unit, idx in frame.groupby('unit', sort=False).indices.items():
            g = frame.iloc[idx]
            t = g['cycle'].to_numpy(dtype=np.float64)
            x = g[cols].to_numpy(dtype=np.float64)
            n = len(g)
            cp = max(0, int(0.6 * n))
            if n >= 30 and x.size:
                w = max(5, n // 20)
                mu = x[:w].mean(axis=0)
                sd = x[:w].std(axis=0) + 1e-9
                hi = ((x - mu) / sd).mean(axis=1)
                k = max(5, n // 20)
                sm = np.convolve(hi, np.ones(k) / k, mode='same')
                base = sm[: max(5, n // 10)]
                thr = base.mean() + 3.0 * (base.std() + 1e-9)
                above = np.where(sm > thr)[0]
                if above.size:
                    cp = int(above[0])
            cp = min(max(cp, 10), n - 1)
            plateau = float(min(t[-1] - t[cp], float(max_rul)))
            lab = np.minimum(t[-1] - t, plateau)
            labels[idx] = np.clip(np.nan_to_num(lab, nan=float(max_rul)), 0.0, float(max_rul))
        return labels.astype(np.float32)
'''.rstrip() + chr(10)
s = s[:start] + new + s[end:]
p.write_text(s, encoding='utf-8')
import ast; ast.parse(s); print('rewritten to paper formula; syntax ok')