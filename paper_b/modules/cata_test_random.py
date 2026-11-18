"""Post-hoc matched-rate random-gate control on the frozen test features."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper_b.modules.cata_test_eval import rmse
from paper_b.modules.learned_gate import Gate, canon
from paper_b.modules.learned_gate_global import design


def main():
    models_path = sys.argv[1]
    features_glob = sys.argv[2]
    out_path = sys.argv[3]
    ck = torch.load(models_path, map_location='cpu', weights_only=False)
    feats = ck['feats']
    mu, sd = ck['mu'], ck['sd']
    models = []
    for sdct in ck['models']:
        m = Gate(len(feats) + 2 + (4 if ck['use_subset_id'] else 0) + 1, hidden=16)
        m.load_state_dict(sdct)
        m.eval()
        models.append(m)
    out = {}
    for p in sorted(glob.glob(features_glob)):
        df = canon(pd.read_csv(p))
        if 'subset' not in df.columns:
            df['subset'] = Path(p).name.split('_')[0]
        X = design(df, ck['use_subset_id'])
        X = (X - mu) / sd
        g = np.mean([m(torch.tensor(X, dtype=torch.float32)).detach().numpy()
                     for m in models], axis=0)
        p_k1 = df['p_k1'].to_numpy(float)
        p_med = df['p_med'].to_numpy(float)
        y = df['y_cap'].to_numpy(float)
        pred = g * p_med + (1.0 - g) * p_k1
        learned = rmse(pred, y)
        rate = float(g.mean())
        rnd = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            gr = (rng.random(len(g)) <= rate).astype(float)
            rnd.append(rmse(gr * p_med + (1.0 - gr) * p_k1, y))
        out[Path(p).stem] = dict(learned=learned, random_mean=float(np.mean(rnd)),
                                 random_sd=float(np.std(rnd, ddof=1)),
                                 learned_minus_random=float(np.mean(rnd) - learned),
                                 gate_rate=rate)
        print('%-10s learned=%.4f random=%.4f (sd %.4f) learned_better=%+.4f'
              % (Path(p).stem, learned, np.mean(rnd), np.std(rnd, ddof=1),
                 np.mean(rnd) - learned))
    Path(out_path).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
