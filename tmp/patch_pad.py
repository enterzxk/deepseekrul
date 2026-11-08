from pathlib import Path
p=Path('scripts/run_candidate.py'); s=p.read_text()
old='            window_vals = seq[max(0, end - window) : end]'
new=('            window_vals = seq[max(0, end - window) : end]\n'
     '            if window_vals.shape[0] < window:\n'
     '                window_vals = np.concatenate([np.repeat(window_vals[:1], window - window_vals.shape[0], axis=0), window_vals], axis=0)')
assert old in s, 'anchor'
s=s.replace(old,new,1)
p.write_text(s); print('patched padding')