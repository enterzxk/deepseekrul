import glob, json, os, time, csv
rows = []
for mf in sorted(glob.glob('experiments/*/*/*/*/metrics.json')):
    root = mf.split('/')[1]
    try: d = json.load(open(mf))
    except Exception: d = {}
    st = os.stat(mf)
    prov = d.get('provenance') or {}
    rows.append({
        'run_id': mf.replace('/metrics.json','').replace('experiments/',''),
        'root': root,
        'end_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(st.st_mtime)),
        'start_utc': 'unknown',
        'duration_s': 'unknown',
        'code_sha256': prov.get('code_sha256', 'unknown'),
        'model_seed': d.get('model_seed', 'unknown'),
        'split_seed': d.get('split_seed', 'unknown'),
        'epochs_ran': d.get('epochs_ran', 'unknown'),
        'epochs_requested': (d.get('run_hyperparameters') or {}).get('epochs_requested', 'unknown'),
        'params': d.get('params', 'unknown'),
        'best_val_rmse': d.get('best_val_rmse', 'unknown'),
        'weights': d.get('weights', 'unknown'),
        'concurrency': 'unknown',
        'basis': 'filesystem mtime + fields inside metrics.json',
    })
with open('results/dsh/review_r2/A1_runs.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('rows:', len(rows))
known = sum(1 for r in rows if r['code_sha256'] != 'unknown')
print('runs with known code hash:', known)
print('runs with known model_seed:', sum(1 for r in rows if r['model_seed'] != 'unknown'))
print('runs with known epochs:', sum(1 for r in rows if r['epochs_ran'] != 'unknown'))
print('runs with epochs_ran==0:', sum(1 for r in rows if r['epochs_ran'] == 0))
print()
print('sample:'); [print(' ', r['run_id'], r['end_utc'], r['model_seed'], r['epochs_ran']) for r in rows[:3]]
