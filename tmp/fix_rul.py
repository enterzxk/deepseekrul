from pathlib import Path
p=Path('scripts/run_dspt.py'); s=p.read_text()
old='''def rul_curve(n, max_rul=125.0):
    r = max_rul - np.arange(n, dtype=np.float64)
    return np.clip(r, 0.0, max_rul)'''
new='''def rul_curve(n, max_rul=125.0):
    """True RUL at step t is n-t (cycles remaining), capped at max_rul."""
    r = float(n) - np.arange(n, dtype=np.float64)
    return np.clip(r, 0.0, max_rul)'''
assert old in s, 'anchor'
s=s.replace(old,new,1)
p.write_text(s); print('fixed rul curve')