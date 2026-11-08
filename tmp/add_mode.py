from pathlib import Path
p=Path('scripts/run_dspt.py'); s=p.read_text()
s=s.replace("    ap.add_argument('--tag', required=True)",
            "    ap.add_argument('--mode', default='dual', choices=['dual','short','long','repeat'])\n    ap.add_argument('--tag', required=True)",1)
old="            out = net(X[idx], mask=MK[idx], baseline=BS[idx], cycle=CY[idx])"
new=('            out = net(X[idx], mask=MK[idx], baseline=BS[idx], cycle=CY[idx])\n'
     '            if a.mode == "short":\n'
     '                out["prediction"] = out["y_A"]\n'
     '            elif a.mode == "long":\n'
     '                out["prediction"] = out["y_B"]\n'
     '            elif a.mode == "repeat":\n'
     '                out["y_B"] = out["y_A"]\n'
     '                out["prediction"] = out["y_A"]')
assert old in s, 'loop'
s=s.replace(old,new,1)
old2="    vr, pv = evaluate(net, VA, dev)"
new2=('    def ev(mode):\n'
      '        net.eval(); x,y,mk,b,c = VA; ps=[]\n'
      '        with torch.no_grad():\n'
      '            for i in range(0,len(x),512):\n'
      '                o = net(x[i:i+512].to(dev), mask=mk[i:i+512].to(dev), baseline=b[i:i+512].to(dev), cycle=c[i:i+512].to(dev))\n'
      '                p = o["y_A"] if mode=="short" else (o["y_B"] if mode=="long" else (o["y_A"] if mode=="repeat" else o["prediction"]))\n'
      '                ps.append(p.reshape(-1).cpu().numpy())\n'
      '        return np.concatenate(ps)\n'
      '    pv = ev(a.mode); vr = float(np.sqrt(np.mean((pv - VA[1])**2)))')
assert old2 in s, 'eval'
s=s.replace(old2,new2,1)
p.write_text(s); print('added modes')