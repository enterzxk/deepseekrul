from pathlib import Path
p=Path('scripts/run_dspt.py'); s=p.read_text()
old="    out_dir = ROOT/'experiments'/a.tag/a.subset"
new=('    out_dir = ROOT/\'experiments\'/a.tag/a.subset\n'
     '    out_dir.mkdir(parents=True, exist_ok=True)\n'
     '    torch.save(net.state_dict(), out_dir/f\'dspt_s{a.seed}.pt\')')
assert old in s, 'anchor'
s=s.replace(old,new,1)
p.write_text(s); print('added ckpt save')