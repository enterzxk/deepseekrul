"""Identify which sensor pairs change correlation most across operating conditions."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_b.training.data import _prepare_split, SENSOR_IDS


def main():
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else 'data/CMAPSS')
    for subset in ('FD001', 'FD002', 'FD003', 'FD004'):
        s = _prepare_split(subset, data_dir, 7, 0, 0.2, 6)
        Z, lab = s['tr_s'], s['lt']
        mats = []
        for c in range(6):
            m = lab == c
            if m.sum() >= 200:
                mats.append(np.corrcoef(Z[m].T))
        if len(mats) < 2:
            print(subset, 'single condition'); continue
        iu = np.triu_indices(14, k=1)
        d = np.zeros(len(iu[0]))
        for i in range(len(mats)):
            for j in range(i+1, len(mats)):
                d += np.abs(mats[i][iu] - mats[j][iu])
        d /= (len(mats) * (len(mats) - 1) / 2)
        order = np.argsort(-d)[:6]
        print(f'== {subset} clusters={len(mats)} mean|dcorr|={d.mean():.3f}')
        for k in order:
            i, j = iu[0][k], iu[1][k]
            print(f'  s{SENSOR_IDS[i]:>2}-s{SENSOR_IDS[j]:>2}: {d[k]:.3f}')


if __name__ == '__main__':
    main()
