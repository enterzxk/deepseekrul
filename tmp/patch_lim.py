from pathlib import Path
p=Path('PAPER_DISCUSSION.md'); s=p.read_text()
old = 'the FD004, FD001 and FD003 matrices are reported at a single seed, and the three-seed\nrequirement is met only on FD002; this is stated wherever those numbers appear.'
new = ('the FD004, FD001 and FD003 matrices are reported at a single seed, and the three-seed\n'
       'requirement is met only on FD002; this is stated wherever those numbers appear. The\n'
       'training script initialises the network before resetting the random seed, so a single\n'
       'run is not bit-reproducible: the same fold can differ by about 3% between runs, and the\n'
       'per-fold spread across seeds reaches 1.2 cycles on FD002. The multi-seed means are\n'
       'stable (3.41% with sd 0.30%), so the conclusions rest on them, but single-fold numbers\n'
       'should not be read as reproducible point estimates.')
if old in s:
    s = s.replace(old, new); p.write_text(s); print('patched')
else:
    print('anchor missing')