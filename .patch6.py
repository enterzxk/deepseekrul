from pathlib import Path
p = Path('src/dataset.py')
s = p.read_text(encoding='utf-8')
start = s.find("    if CP_LABELS:")
end = s.find("    return np.minimum(rul, float(max_rul)).astype(np.float32)", start)
assert start != -1 and end != -1
new = '''    if CP_LABELS:
        # Change-point-informed labels: the detected onset sets the upper RUL limit,
        #   RUL(t) = min(T - t, min(T - t_cp, max_rul))
        # If the detector does not fire, the standard cap is kept (no artificial
        # fallback), so an unfired engine gets exactly the standard label.
        cols = [c for c in SELECTED_SENSOR_COLUMNS if c in frame.columns]
        labels = np.empty(len(frame), dtype=np.float64)
        for unit, idx in frame.groupby('unit', sort=False).indices.items():
            g = frame.iloc[idx]
            t = g['cycle'].to_numpy(dtype=np.float64)
            plateau = float(max_rul)
            if cols:
                x = g[cols].to_numpy(dtype=np.float64)
                n = len(g)
                if n >= 30:
                    w = max(5, n // 20)
                    mu = x[:w].mean(axis=0)
                    sd = x[:w].std(axis=0) + 1e-9
                    hi = ((x - mu) / sd).mean(axis=1)
                    k = max(5, n // 20)
                    smv = np.convolve(hi, np.ones(k) / k, mode='valid')
                    pad = n - len(smv)
                    sm = np.concatenate([np.full(pad // 2, smv[0]), smv,
                                         np.full(pad - pad // 2, smv[-1])])
                    base = sm[: max(5, n // 10)]
                    thr = base.mean() + 3.0 * (base.std() + 1e-9)
                    above = np.where(sm > thr)[0]
                    if above.size:
                        cp = int(min(max(above[0], 5), n - 1))
                        plateau = float(min(t[-1] - t[cp], float(max_rul)))
            lab = np.minimum(t[-1] - t, plateau)
            labels[idx] = np.clip(np.nan_to_num(lab, nan=float(max_rul)), 0.0, float(max_rul))
        return labels.astype(np.float32)
'''
s = s[:start] + new + s[end:]
p.write_text(s, encoding='utf-8')
import ast; ast.parse(s); print('dataset.py CP block rewritten; syntax ok')