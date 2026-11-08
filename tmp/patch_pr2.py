from pathlib import Path
p=Path('PAPER_RESULTS.md'); s=p.read_text()
old = "## H. Metric behaviour"
new = """## H. Matched-fold multi-seed check

On the four folds where seeds 101 and 202 are both complete, the sensor-history prior
improves the pooled RMSE by 3.20% (seed 101) and 2.27% (seed 202); the two-seed mean is
2.73%, below the 3% threshold. The larger single-seed figure is partly due to a favourable
initialisation on fold 1 (12.04 against 13.28 cycles). Any claim above 3% must therefore
rest on the three-seed mean, not on a single seed.

## I. Metric behaviour"""
if old in s and 'H. Matched-fold' not in s:
    s = s.replace(old, new); p.write_text(s); print('patched')
else:
    print('skip')
