from pathlib import Path
p=Path('scripts/run_dspt.py'); s=p.read_text()
old="        vr, _ = evaluate(net, VA, dev)"
new=('        vr, _ = evaluate(net, VA, dev)\n'
     '        if ep % 5 == 0 or ep == 1:\n'
     '            print("ep%03d val=%.4f best=%.4f@%d" % (ep, vr, best, best_ep), flush=True)')
assert old in s, 'anchor'
s=s.replace(old,new,1)
p.write_text(s); print('added epoch log')