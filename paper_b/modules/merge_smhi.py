"""Merge SMHI per-cut predictions into the LTAG training/validation CSVs."""
import glob
from pathlib import Path

import pandas as pd

from paper_b.modules.learned_gate import canon

BASE = Path.home() / 'sprint24/experiments/hc/cata_gate'


def main():
    for split, src, dst, smhi_split in [
        ('train', BASE / 'train', BASE / 'train_shi', 'train'),
        ('val', BASE / 'preds', BASE / 'preds_shi', 'val'),
    ]:
        dst.mkdir(parents=True, exist_ok=True)
        for p in sorted(glob.glob(str(src / '*.csv'))):
            tag = Path(p).stem
            sm = pd.read_csv(BASE / 'smhi_preds' / (tag + '.csv'))
            sm = sm[sm['split'] == smhi_split][['engine', 'cut', 'p_smhi']]
            df = canon(pd.read_csv(p))
            m = df.merge(sm, on=['engine', 'cut'], how='left')
            if m['p_smhi'].isna().any():
                raise SystemExit('%s: %d rows missing SMHI' % (tag, int(m['p_smhi'].isna().sum())))
            m['p_smhi_minus_k1'] = m['p_smhi'] - m['p_k1']
            m['p_smhi_minus_med'] = m['p_smhi'] - m['p_med']
            m.to_csv(dst / (tag + '.csv'), index=False)
            print('wrote', dst / (tag + '.csv'), len(m))


if __name__ == '__main__':
    main()
