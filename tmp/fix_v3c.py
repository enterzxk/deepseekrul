from pathlib import Path
p=Path('run_v3.py'); s=p.read_text()
s=s.replace('def train_v3(arm, net, tr_raw, va_raw, target_path, meta, seed, epochs=80,',
            'def train_v3(arm, net, tr_raw, va_raw, target_path, meta_tr, meta_va, seed, epochs=80,')
s=s.replace('partner, gap = build_pairs(meta, 10, seed)','partner, gap = build_pairs(meta_tr, 10, seed)')
s=s.replace('frame = meta.assign(y_true=va_raw[1]','frame = meta_va.assign(y_true=va_raw[1]')
s=s.replace('                  tgt, mtr_df, a.seed, a.epochs','                  tgt, mtr_df, meta, a.seed, a.epochs')
p.write_text(s)
import re
print('meta_tr' in s, 'meta_va' in s, s.count('mtr_df, meta'))