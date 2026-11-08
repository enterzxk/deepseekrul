#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
echo "=== calib_frac 0.5 ==="
$PY scripts/run_punet_stage.py --subset FD002 --fold 0 --seed 42 --stage heads --reuse-ckpt   --prior-dir experiments/ida/prior_boot --out experiments/punet_boot --calib-strata regime_sigma   --calib-frac 0.5 2>&1 | tail -1
echo "=== difficulty diagnostic ==="
$PY - <<'PYEOF'
import sys, torch, numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
import run_punet_stage as S
from src.punet.conformal import engine_calib_split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = S.build('FD002', 0, 42, 50, 'experiments/ida/prior_boot')
model = S.make_model(device)
blob = torch.load('experiments/punet_boot/FD002/f0_s42/backbone.pt', map_location=device, weights_only=False)
model.load_state_dict(blob['state_dict'])
h, yb, y = S.extract(model, data['val_loader'], device)
units = np.asarray(data['meta_va'])[:,0]
err = np.abs(yb.numpy()-y.numpy())
cal, tst = engine_calib_split(units, calib_frac=0.3, seed=42)
import pandas as pd
per = pd.DataFrame({'u':units,'e':err}).groupby('u').e.mean()
print('calib engines mean |err| =', round(float(per[per.index.isin(units[cal])].mean()),3))
print('test  engines mean |err| =', round(float(per[per.index.isin(units[tst])].mean()),3))
print('per-engine |err| min/med/max =', round(float(per.min()),2), round(float(per.median()),2), round(float(per.max()),2))
PYEOF
