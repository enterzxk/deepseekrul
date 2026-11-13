#!/bin/bash
# Run the mechanism probe over every sprint arm that has a checkpoint.
#
# One command for whoever has the 3090 checkpoints: the verdict for M3B / M7 /
# M4 / M2 all depends on the same two numbers (same-cycle ranking accuracy and
# sensor-permutation degradation), and running them one at a time invites
# different flags and therefore non-comparable records.
#
# Usage:  bash docs/paper_b_hc/probes/run_mechanism_all.sh [ROOT] [SUBSET] [DEVICE]
#   ROOT    where the run directories live (default experiments/hc/d80)
#   SUBSET  default FD001
#   DEVICE  default cuda:0
set -u
cd /media/sangfor/vdb/user/zxk/deepseek-rul || exit 1
export PYTHONPATH=.
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH:-}
P=/home/yndx/miniconda3/envs/a40_env/bin/python
ROOT=${1:-experiments/hc/d80}
SUBSET=${2:-FD001}
DEVICE=${3:-cuda:0}

# arm:run_dir suffix. The id already carries arm/subset/seed.
ARMS="B0 B2 B3 M3B M7 M4 M2"
for a in $ARMS; do
  d="$ROOT/HC-D-$a-$SUBSET-s101-D1"
  if [ ! -f "$d/ckpt/best.pt" ]; then
    echo "SKIP $a (no checkpoint at $d)"
    continue
  fi
  echo "=== $a ==="
  $P docs/paper_b_hc/probes/d_mechanism_m3b.py --arm "$a" --subset "$SUBSET" \
     --run-dir "$d" --device "$DEVICE" 2>&1 | grep -e accuracy -e agree \
     -e degradation -e val_engine_mean_rmse -e "record ->"
done
echo "--- summary ---"
$P - <<'PY'
import glob, json
rows = []
for p in sorted(glob.glob('experiments/hc/d/mech_*_FD001.json')):
    r = json.load(open(p))
    rk = r.get('ranking_exact_cycle') or {}
    rows.append((r['arm'], r['val_engine_mean_rmse'], rk.get('accuracy'),
                 rk.get('agree', 0) + rk.get('disagree', 0),
                 r.get('permuted_mean'), r.get('degradation'),
                 r.get('degradation_pct')))
print('%-5s %9s %9s %8s %10s %10s %9s'
      % ('arm', 'rmse', 'rank_acc', 'pairs', 'perm_rmse', 'degrad', 'degrad%'))
for a, rm, ra, np_, pm, dg, dp in rows:
    print('%-5s %9.3f %9s %8d %10.3f %10.3f %9.2f'
          % (a, rm, 'n/a' if ra is None else '%.4f' % ra, np_, pm, dg, dp))
print('\nreference: AGE degrades by exactly 0 (never reads sensors);')
print('a model that only follows age must show ~0 degradation and no')
print('comparable same-cycle pairs.')
PY
