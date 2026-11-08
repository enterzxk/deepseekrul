#!/bin/bash
set -e
mkdir -p ~/deepseek-rul
tar -xzf ~/repo_sync.tgz -C ~/deepseek-rul
cd ~/deepseek-rul
ls
echo "--- imports ---"
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'PY'
import sys
sys.path.insert(0, '.')
from src.punet import PUNetModel
from src.lifetime_prior import nested_prior
print('punet ok')
print('prior module ok')
PY
echo "--- data ---"
ls data/CMAPSS | head -3
