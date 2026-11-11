"""Official-test evaluation for a frozen Paper B run.

Refuses to run unless the run's tag is listed in a freeze manifest, so the
official test set cannot be used for model selection by accident.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from ..training.data import prepare_test
from ..training.metrics import summarize
from ..training.train import _build, predict


def evaluate_run(run_dir, data_dir='data/CMAPSS', device='cuda:0'):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / 'config.json').read_text())
    ds = prepare_test(cfg['subset'], Path(data_dir), span=cfg['span'],
                      split_seed=cfg['split_seed'], fold=cfg['fold'],
                      val_frac=cfg['val_frac'], rul_cap=cfg['rul_cap'], n_cond=cfg['n_cond'])
    te = ds['test']
    dev = torch.device(device)
    net = _build(cfg).to(dev)
    net.load_state_dict(torch.load(run_dir / 'model.pt', map_location='cpu', weights_only=True))
    p = predict(net, te, dev, cfg['mode'])
    m = summarize(p, te['y'])
    m.update(subset=cfg['subset'], seed=cfg['seed'], tag=cfg['tag'], n=int(len(p)),
             params=int(sum(q.numel() for q in net.parameters())))
    pd.DataFrame(dict(unit=te['unit'], y_true=te['y'], y_pred=p)).to_csv(
        run_dir / 'test_pred.csv', index=False)
    (run_dir / 'test_metrics.json').write_text(json.dumps(m, indent=1))
    print('TEST %s %s s%d RMSE %.4f NASA %.1f n=%d' % (
        cfg['tag'], cfg['subset'], cfg['seed'], m['rmse'], m['nasa_sum'], m['n']), flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--data-dir', default='data/CMAPSS')
    ap.add_argument('--manifest', default='paper_b/results/FREEZE.json')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    cfg = json.loads((run_dir / 'config.json').read_text())
    man = json.loads(Path(a.manifest).read_text()) if Path(a.manifest).exists() else None
    if not a.force:
        if man is None or not man.get('frozen'):
            raise SystemExit('no frozen manifest: test evaluation is blocked')
        allowed = {(r['tag'], r['subset'], int(r['seed'])) for r in man.get('runs', [])}
        if (cfg['tag'], cfg['subset'], int(cfg['seed'])) not in allowed:
            raise SystemExit(f"run {(cfg['tag'], cfg['subset'], cfg['seed'])} not in freeze manifest")
    else:
        print('WARNING: --force bypasses the freeze manifest (diagnostic only)', flush=True)
    evaluate_run(run_dir, a.data_dir, a.device)


if __name__ == '__main__':
    main()
