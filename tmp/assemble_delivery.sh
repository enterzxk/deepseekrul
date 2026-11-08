#!/bin/bash
set -uo pipefail
PY=/home/user/miniconda3/envs/pmv/bin/python
cd /home/user/rulb/tim_20261105/full_20261105
echo '== 1. lock reference ==============================='
$PY lock_reference.py
echo '== 2. spec ========================================'
$PY make_spec.py
echo '== 3. results ====================================='
$PY build_results.py --spec spec.json --out-prefix RESULTS || true
echo '== 4. ablations ==================================='
$PY make_ablation_table.py || true
echo '== 5. per-engine / per-fold ======================='
$PY per_engine.py || true
$PY per_fold_table.py || true
$PY cross_subset.py || true
echo '== 6. scope (FD001/FD003) ========================='
$PY scope_check.py || true
echo '== 7. observation-span schedule ==================='
$PY schedule_results.py || true
echo '== 8. paper assets ================================'
$PY make_paper_assets.py || true
$PY make_extra_figures.py || true
$PY per_engine_fig.py || true
$PY make_tex.py || true
echo '== 9. measurement conditions ======================'
$PY mc_aggregate.py || true
echo '== 10. official reproducibility ==================='
$PY repro_check.py | tail -2
echo '== 11. verification / manifest ===================='
$PY verify_delivery.py || true
find . -maxdepth 2 -type f \( -name '*.py' -o -name '*.md' -o -name '*.json' -o -name '*.csv' -o -name '*.sh' -o -name '*.txt' -o -name '*.png' -o -name '*.pdf' -o -name '*.tex' \) ! -path './runs/*' ! -path './runs2/*' ! -path './runs3*/*' ! -path './runs_*/*' ! -path './measurement/*' -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256
wc -l MANIFEST.sha256
echo '== done ==========================================='
