from pathlib import Path
p=Path('run_v3b.py'); s=p.read_text()
reps=[('def train_v3(arm, net, tr_raw, va_raw, target_path, meta, seed, epochs=80,',
       'def train_v3(arm, net, tr_raw, va_raw, target_path, meta_tr, meta_va, seed, epochs=80,'),
      ('partner, gap = build_pairs(meta, 10, seed)','partner, gap = build_pairs(meta_tr, 10, seed)'),
      ('frame = meta.assign(','frame = meta_va.assign('),
      ('tgt, mtr_df, a.seed, a.epochs','tgt, mtr_df, meta, a.seed, a.epochs')]
for a,b in reps:
    assert a in s, a
    s=s.replace(a,b)
p.write_text(s); print('patched')