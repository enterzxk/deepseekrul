#!/bin/bash
cd ~/deepseek-rul
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
$PY - <<'PYEOF'
import sys
sys.path.insert(0, '.')
from src.punet import PUNetModel
from src.lifetime_prior import nested_prior
print('imports ok')
PYEOF
ls data/CMAPSS | head -3
