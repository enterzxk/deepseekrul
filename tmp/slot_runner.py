from pathlib import Path
p = Path('scripts/run_dspt.py')
s = p.read_text()
old = "choices=['dual','short','long','repeat']"
new = "choices=['dual','short','long','repeat','slot']"
assert old in s, 'choices anchor'
s = s.replace(old, new, 1)
old2 = 'out = net(X[idx], mask=MK[idx], baseline=BS[idx], cycle=CY[idx])'
new2 = "out = net(X[idx], mask=MK[idx], baseline=BS[idx], cycle=CY[idx], mode=a.mode)"
assert old2 in s, 'call anchor'
s = s.replace(old2, new2, 1)
old3 = "o = net(torch.from_numpy(VA[0][i:i+512]).to(dev),"
new3 = "o = net(torch.from_numpy(VA[0][i:i+512]).to(dev), mode=a.mode,"
assert old3 in s, 'ev anchor'
s = s.replace(old3, new3, 1)
p.write_text(s)
print('runner supports slot')