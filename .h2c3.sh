#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
echo trainers=$(pgrep -c -f '[r]un_candidate')
cat /home/yndx/p8/h2.state 2>/dev/null
echo s8=$(ls experiments/split8/base/FD002/*/metrics.json 2>/dev/null | wc -l) s9=$(ls experiments/split9/base/FD002/*/metrics.json 2>/dev/null | wc -l)
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import glob, json
for f in sorted(glob.glob('experiments/split*/base/FD002/*/metrics.json')):
    m = json.load(open(f))
    print(f.split('/')[1], 'seed', m['model_seed'], 'val', round(m['best_val_rmse'],2), 'matched', round(m.get('matched_cutoff_rmse',0),2), 'hash', (m.get('provenance') or {}).get('code_sha256',{}).get('model','')[:8])
EOF
