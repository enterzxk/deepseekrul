"""Generate the experiment grids from the frozen protocol constants."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = json.loads((ROOT / 'configs' / 'protocol.json').read_text())
SUBSETS = P['subsets']
SEEDS = P['seeds']
G = ROOT / 'experiments' / 'grids'

COMMON = dict(epochs=P['epochs'], patience=P['patience'], batch=P['batch'], lr=P['lr'],
              weight_decay=P['weight_decay'], d_model=P['d_model'], layers=P['layers'],
              dropout=P['dropout'], n_cond=P['n_cond'], split_seed=P['split_seed'],
              fold=P['fold'], val_frac=P['val_frac'], cuts=P['val_cuts'],
              rul_cap=P['rul_cap'])

CFG8 = {
    'base': dict(ms_mode='off', sa_mode='off', hi_source='none'),
    'ms':   dict(ms_mode='full', sa_mode='off', hi_source='none'),
    'sa':   dict(ms_mode='off', sa_mode='basic', hi_source='none'),
    'hi':   dict(ms_mode='off', sa_mode='off', hi_source='hf', hi_mono='correct'),
    'mssa': dict(ms_mode='full', sa_mode='basic', hi_source='none'),
    'mshi': dict(ms_mode='full', sa_mode='off', hi_source='hf', hi_mono='correct'),
    'sahi': dict(ms_mode='off', sa_mode='basic', hi_source='hf', hi_mono='correct'),
    'full': dict(ms_mode='full', sa_mode='basic', hi_source='hf', hi_mono='correct'),
}


def dspt(tag, subset, seed, **kw):
    c = dict(COMMON, model='dspt', tag=tag, subset=subset, seed=seed, span=P['span'],
             mode='dual', use_cycle=True, use_pad_mask=True, aux_dual=True,
             hi_lambda=P['hi_lambda'], hi_mono_coef=P['hi_mono_coef'], hi_mono='none',
             ms_mode='off', sa_mode='off', hi_source='none', instnorm=False)
    c.update(kw)
    return c


def van(tag, subset, seed, **kw):
    c = dict(COMMON, model='vanilla', tag=tag, subset=subset, seed=seed, span=P['span'],
             mode='dual', input_mode='raw', use_cycle=False)
    c.update(kw)
    return c


def dump(name, runs):
    (G / name).write_text(json.dumps(runs, indent=1))
    print(f'{name}: {len(runs)} runs')


# A. core 8-config DSPT ablation (L=70, 3 seeds, 4 subsets)
runs = []
for cfg, flags in CFG8.items():
    for subset in SUBSETS:
        for seed in SEEDS:
            runs.append(dspt(f'A8_{cfg}_L70', subset, seed, **flags))
dump('ablation8.json', runs)

# B. independent Vanilla Transformer baseline + feature controls
runs = []
for variant, kw in {
    'raw': dict(input_mode='raw', use_cycle=False),
    'dsp_cycle': dict(input_mode='dsp', use_cycle=True),
}.items():
    for subset in SUBSETS:
        for seed in SEEDS:
            runs.append(van(f'VAN_{variant}_L70', subset, seed, **kw))
dump('vanilla.json', runs)

# C. cycle control for DSPT
runs = []
for cfg in ('base',):
    for subset in SUBSETS:
        for seed in SEEDS:
            runs.append(dspt(f'NC_{cfg}_L70', subset, seed, use_cycle=False, **CFG8[cfg]))
dump('nocycle.json', runs)

# D. MS mechanism: isolate projection vs multi-scale vs capacity
runs = []
for mode in ('single_proj', 'multi_noproj', 'multi3'):
    for subset in ('FD001', 'FD004'):
        for seed in SEEDS:
            runs.append(dspt(f'MSM_{mode}_L70', subset, seed, ms_mode=mode, sa_mode='off',
                             hi_source='none'))
dump('ms_mech.json', runs)

# E. SA mechanism: sensor identity / operating-condition context
runs = []
for mode in ('identity', 'cond'):
    for subset in SUBSETS:
        for seed in SEEDS:
            runs.append(dspt(f'SAM_{mode}_L70', subset, seed, ms_mode='off', sa_mode=mode,
                             hi_source='none'))
dump('sa_mech.json', runs)

# F. HI mechanism: source and monotonic sign
runs = []
for name, kw in {
    'hf_wrong': dict(hi_source='hf', hi_mono='wrong'),
    'hf_nomono': dict(hi_source='hf', hi_mono='none'),
    'hB_correct': dict(hi_source='hB', hi_mono='correct'),
}.items():
    for subset in ('FD001', 'FD004'):
        for seed in SEEDS:
            runs.append(dspt(f'HIM_{name}_L70', subset, seed, ms_mode='off', sa_mode='off', **kw))
dump('hi_mech.json', runs)

# G. historical attribution on FD001 / seed 101 (not paper evidence)
runs = []
for name, kw in {
    'fixed': dict(),
    'legacy_ms_pad': dict(legacy_ms_pad=True),
    'legacy_ema': dict(legacy_ema=True),
    'unmasked': dict(use_pad_mask=False),
    'legacy_rul': dict(legacy_rul=True),
    'global_hi': dict(global_hi=True),
    'legacy_hi_wrong': dict(hi_mono='wrong'),
}.items():
    flags = dict(hi_mono='correct'); flags.update(kw)
    runs.append(dspt(f'ATTR_{name}', 'FD001', 101, ms_mode='full', sa_mode='basic',
                     hi_source='hf', **flags))
dump('attribution.json', runs)

# G2. validation-cut sensitivity (earlier cut covers the high-RUL regime)
runs = []
for cfg in ('base', 'full'):
    for subset in ('FD001', 'FD004'):
        for seed in SEEDS:
            c = dspt(f'SENS_{cfg}', subset, seed, **CFG8[cfg])
            c['cuts'] = [0.3, 0.4, 0.55, 0.7, 0.85, 0.95]
            runs.append(c)
dump('sens_cuts.json', runs)

# J. padding-mask control (the attribution run suggests unmasking helps)
runs = []
for cfg, tag in (('base', 'NM_base'), ('full', 'NM_full')):
    for subset in SUBSETS:
        for seed in SEEDS:
            runs.append(dspt(tag, subset, seed, use_pad_mask=False, **CFG8[cfg]))
dump('nomask.json', runs)

# K. 2x2x2 factorial ablation (replacement-based, seed 101, all subsets)
runs = []
for ms in ('single_proj', 'full'):
    for sa in ('mean', 'basic'):
        for hi in (0.0, 0.5):
            for subset in SUBSETS:
                runs.append(dspt(f'F_{ms}_{sa}_hi{int(hi > 0)}', subset, 101,
                                 ms_mode=ms, sa_mode=sa, hi_source='hf',
                                 hi_lambda=hi, hi_mono='correct'))
for tag, kw in {
    'C_linear_mean_hi0': dict(ms_mode='linear_only', sa_mode='mean',
                              hi_source='hf', hi_lambda=0.0, hi_mono='correct'),
    'C_single_static_hi0': dict(ms_mode='single_proj', sa_mode='static',
                                hi_source='hf', hi_lambda=0.0, hi_mono='correct'),
}.items():
    for subset in SUBSETS:
        runs.append(dspt(tag, subset, 101, **kw))
dump('factorial.json', runs)

# I. same-machine reproducibility check (duplicate of two runs)
runs = [dspt('REPRO_base', 'FD001', 101, **CFG8['base']),
        dspt('REPRO_full', 'FD001', 101, **CFG8['full'])]
dump('repro.json', runs)

# H. window search is generated after the ablation decides the final config.
if not (G / 'window.json').exists():
    dump('window.json', [])
