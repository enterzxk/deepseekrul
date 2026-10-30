#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python .revert_labels.py
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import ast
src = open("src/dataset.py", encoding="utf-8").read()
ast.parse(src)
print("dataset.py syntax ok")
print("LABEL_MODE present:", "LABEL_MODE" in src)
EOF
