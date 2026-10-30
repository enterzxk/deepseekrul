import json, glob
for f in sorted(glob.glob('experiments/r3_cycle/*/FD002/*/metrics.json')):
    d = json.load(open(f))
    k = f.split('/')[2]
    print(k, 'val=%.4f' % d['best_val_rmse'], 'ep=%s' % d['epochs_ran'], 'params=%s' % d['params'],
          'pseudo_mean=%.4f' % d.get('pseudo_test_rmse_mean', float('nan')))
