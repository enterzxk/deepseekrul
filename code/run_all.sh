#!/bin/bash
# 从项目根目录运行：bash code/run_all.sh
# 作用：建立 work/，链接 data/results 的必要文件，复算 HOF/EOL/汇总。
set -euo pipefail
PY="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$ROOT/work"
mkdir -p "$WORK"
cd "$WORK"

# 数据
for f in ba_clean.csv ba_cell_summary.csv celllist.json; do
  ln -sf "$ROOT/data/$f" "$f"
done
# 结果（final_metrics 需要）
for f in hof_eval_clean.csv hof_eol_clean.csv eol_compare_clean.csv \
         unified_percell.csv gru_pred.csv resgru_pred.csv \
         losource_hof_pred.csv; do
  ln -sf "$ROOT/results/$f" "$f"
done
# 可选：CALCE 原始审计的输入
ln -sfn "$ROOT/data/calce_parsed" parsed2 2>/dev/null || true

echo "[1/3] HOF SOH/EOL 评估"
"$PY" "$ROOT/code/hof_eval_clean.py"

echo "[2/3] EOL 多模型对比"
"$PY" "$ROOT/code/eol_compare_clean.py"

echo "[3/3] 汇总主指标"
"$PY" "$ROOT/code/final_metrics.py"

echo "DONE. 输出在 $WORK/ 与 $ROOT/results/（注意：脚本默认写当前目录；如需覆盖 results，请手动复制）"
