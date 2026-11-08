from pathlib import Path
p=Path('scripts/run_dspt.py'); s=p.read_text()
old="    X, Y, MK, BS, CY = [torch.from_numpy(v) for v in TR]"
new="    X, Y, MK, BS, CY = [torch.from_numpy(v).to(dev) for v in TR]"
assert old in s
s=s.replace(old,new,1)
s=s.replace("out = net(X[idx].to(dev), mask=MK[idx].to(dev), baseline=BS[idx].to(dev), cycle=CY[idx].to(dev))",
            "out = net(X[idx], mask=MK[idx], baseline=BS[idx], cycle=CY[idx])")
s=s.replace("y = Y[idx].to(dev).reshape(-1, 1)", "y = Y[idx].reshape(-1, 1)")
p.write_text(s); print('fixed device')