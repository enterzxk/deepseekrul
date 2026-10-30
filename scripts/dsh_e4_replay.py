"""E4 part 1: exact replay of a stored run from best_model.pt.

Rebuilds variant A with the current code, loads the saved state dict, re-evaluates
the matched-cutoff set and compares against the stored predictions.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import run_candidate as rc
from candidates.model import CandidateModel

CFG = rc.CONFIGS['A']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='experiments/e3/A_s2027/FD002/A_s2027')
    ap.add_argument('--ref-mode', default='global')
    ap.add_argument('--out', default='results/dsh/e4/replay.json')
    a = ap.parse_args()

    prep = rc.prepare('FD002', 7, 50, 125.0, condition_aware=True, n_cond=6,
                      ref_mode=a.ref_mode)
    matched_set, matched_units = prep[7], prep[8]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CandidateModel(
        num_sensors=14, window_size=50, d_model=64, num_layers=1, dropout=0.15,
        use_baseline=CFG['use_baseline'], use_ema_features=CFG['use_ema_features'],
        use_cycle=CFG['use_cycle'], use_pe=False, use_graph=CFG['use_graph'],
        use_dual_head=CFG['use_dual_head'],
    ).to(device)
    state = torch.load(Path(a.run) / 'best_model.pt', map_location='cpu')
    model.load_state_dict(state, strict=True)

    loader = DataLoader(matched_set, batch_size=256, shuffle=False, num_workers=0)
    rmse, pred, true = rc.evaluate(model, loader, device, baseline=False)

    ref = pd.read_csv(Path(a.run) / 'matched_cutoff_predictions.csv').sort_values('unit_id')
    got = pd.DataFrame({'unit_id': matched_units, 'y_pred': pred, 'y_true': true}).sort_values('unit_id')
    merged = ref.merge(got, on='unit_id', suffixes=('_ref', '_new'))
    max_dp = float(np.abs(merged['y_pred_ref'] - merged['y_pred_new']).max())
    max_dt = float(np.abs(merged['y_true_ref'] - merged['y_true_new']).max())
    out = dict(run=a.run, n=int(len(merged)), rmse_replay=float(rmse),
               rmse_stored=float(json.load(open(Path(a.run) / 'metrics.json'))['matched_cutoff_rmse']),
               max_abs_pred_diff=max_dp, max_abs_true_diff=max_dt,
               pass_=bool(max_dp <= 1e-3 and max_dt < 1e-9))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
