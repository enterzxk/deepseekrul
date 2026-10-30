from pathlib import Path
p = Path('src/dataset.py')
s = p.read_text(encoding='utf-8')
old = "            labels[idx] = np.clip(lab, 0.0, float(max_rul))"
new = ("            lab = np.nan_to_num(lab, nan=float(max_rul), posinf=float(max_rul), neginf=0.0)" + chr(10) +
       "            labels[idx] = np.clip(lab, 0.0, float(max_rul))")
assert old in s
s = s.replace(old, new, 1)
p.write_text(s, encoding='utf-8')
print('nan guard added')