"""smoke-test the E10 analyser on fold 0 only (no bootstrap, fast)."""
import importlib.util, sys
spec = importlib.util.spec_from_file_location('e10', 'scripts/dsh_e10_cv_analyze.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
units, d = m.load_fold(0)
print('fold0 engines', len(units), 'A shape', d['A'].shape)
delta, ra, rp = m.delta_rmse(d)
print('RMSE_A %.3f  RMSE_PE %.3f  delta %+.3f' % (ra, rp, delta))
# cross-check against the archived E3 numbers
print('expected ~12.676 / ~11.768 / ~-0.908')
