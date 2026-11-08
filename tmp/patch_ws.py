from pathlib import Path
NL = chr(10)
p = Path('/home/user/rulb/scripts/run_dspt.py')
s = p.read_text()
if 'wshift' in s:
    print('already patched'); raise SystemExit

# 1) CLI flag
a1 = "    ap.add_argument('--instnorm', action='store_true')"
assert s.count(a1) == 1, ('a1', s.count(a1))
s = s.replace(a1, a1 + NL + "    ap.add_argument('--wshift', action='store_true')")

# 2) build sample weights after Y is available; insert right before 'net = build('
a2 = '    net = build(14, a.d_model, a.layers, a.dropout'
assert s.count(a2) == 1, ('a2', s.count(a2))
wcode = (
  '    SW = None' + NL +
  '    if a.wshift:' + NL +
  '        CAPW = 125.0' + NL +
  '        lens = np.array([len(g) for _, g in frame.groupby(\"unit\")], float)' + NL +
  '        ys = np.arange(0, int(CAPW) + 1)' + NL +
  '        p_te = np.array([min((lens > y).sum(), (lens > CAPW).sum()) for y in ys], float)' + NL +
  '        p_te = p_te / p_te.sum()' + NL +
  '        Yr = np.round(Y).astype(int)' + NL +
  '        p_tr = np.bincount(np.clip(Yr, 0, int(CAPW)), minlength=int(CAPW) + 1).astype(float)' + NL +
  '        p_tr = p_tr / p_tr.sum()' + NL +
  '        wv = np.where(p_tr > 1e-12, p_te / np.maximum(p_tr, 1e-12), 0.0)' + NL +
  '        wv = wv / (p_tr * wv).sum()' + NL +
  '        SW = torch.from_numpy(wv[np.clip(Yr, 0, int(CAPW))].astype(np.float32)).to(dev)' + NL +
  '        print(\"  wshift: weight range [%.3f, %.3f] ESS %.1f%%\" % (float(wv.min()), float(wv.max()), 100*float((p_tr*wv).sum()**2/((p_tr*wv**2).sum()))), flush=True)' + NL
)
s = s.replace(a2, wcode + a2)

# 3) weighted losses
old = ('            loss = e.square().mean() + 0.5 * torch.where(e > 0, 1.8, 1.0).mul(e.square()).mean()' + NL +
       "            eA = out['y_A'] - y; eB = out['y_B'] - y" + NL +
       "            loss = loss + 0.5 * (eA.square().mean() + eB.square().mean()) / 2")
assert s.count(old) == 1, ('loss', s.count(old))
new = ('            sw = None if SW is None else SW[idx].reshape(-1, 1)' + NL +
       '            def wm(v):' + NL +
       '                return v.mean() if sw is None else (v * sw).sum() / sw.sum()' + NL +
       '            loss = wm(e.square()) + 0.5 * wm(torch.where(e > 0, 1.8, 1.0).mul(e.square()))' + NL +
       "            eA = out['y_A'] - y; eB = out['y_B'] - y" + NL +
       '            loss = loss + 0.5 * (wm(eA.square()) + wm(eB.square())) / 2')
s = s.replace(old, new)
p.write_text(s)
import ast; ast.parse(s)
print('patched wshift, syntax ok')