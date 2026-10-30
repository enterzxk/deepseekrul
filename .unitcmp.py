import json, glob
def units(root, tag):
    fs = sorted(glob.glob(f'experiments/{root}/{tag}/FD002/*/metrics.json'))
    if not fs: return None
    return {m['model_seed']: sorted(m.get('val_units', [])) for m in (json.load(open(f)) for f in fs)}
a = units('levelb','M4_racdr'); b = units('order03','M4_racdr')
print('levelb seeds:', list(a) if a else None)
print('order03 seeds:', list(b) if b else None)
if a and b:
    s1 = next(iter(a)); s2 = next(iter(b))
    A, B = set(a[s1]), set(b[s2])
    print('n levelb:', len(A), 'n order03:', len(B), 'overlap:', len(A & B))
    print('same set:', A == B)
