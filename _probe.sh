cd /home/user/pb_sensor || exit 1
echo "--- E0 config ---"
grep -E '"(data_dir|span|split_seed|fp32|sa_mode|use_cycle|hi_source|instnorm|batch)"' \
  experiments/paper_b_sensor_generalization/runs/E0-FD001-s101-4ec72f00/FD001/s101/config.json
echo "--- torch ---"
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
echo "--- h8 ---"
python3 - <<'PY'
import hashlib
from pathlib import Path
R = Path('/home/user/pb_sensor')
F = ('paper_b/models/dspt.py','paper_b/training/train.py','paper_b/training/data.py',
     'paper_b/training/metrics.py','paper_b/sensor_aug/augmentation.py',
     'paper_b/sensor_aug/noise_stats.py','paper_b/sensor_aug/evaluation.py')
def h(p):
    x = hashlib.sha256()
    for r in p:
        x.update((R/r).read_bytes())
    return x.hexdigest()[:12]
print('h7', h(F), 'h8', h(F + ('scripts/run_sensor_aug.py',)))
PY
echo "--- running procs ---"
pgrep -af run_sensor_aug | head -5
