
cd /home/yndx/zxk/bearing-rul-pi
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
for f in tests/test_audit_fixes.py tests/test_stage2_boundaries.py tests/test_sensor_permutation.py tests/test_extras_schema.py; do
  echo "== $f"; $PY "$f" 2>&1 | tail -n 2
done
