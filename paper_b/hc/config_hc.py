"""DSPT-HC configuration: frozen protocol + Stage C cycle variants.

Everything here is derived from ``paper_b/configs/protocol.json`` (read-only).
No protocol constant is redefined: the file is loaded and asserted, so a
divergence between the task book and the frozen protocol fails loudly instead
of silently training under a different budget.
"""
import json
from pathlib import Path

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'protocol.json'


def load_protocol(path=PROTOCOL_PATH):
    return json.loads(Path(path).read_text())


PROTOCOL = load_protocol()

# Frozen protocol constants (read from the file, never retyped).
SPLIT_SEED = PROTOCOL['split_seed']
FOLD = PROTOCOL['fold']
VAL_FRAC = PROTOCOL['val_frac']
VAL_CUTS = tuple(PROTOCOL['val_cuts'])
RUL_CAP = float(PROTOCOL['rul_cap'])
SPAN = int(PROTOCOL['span'])
SEEDS = tuple(PROTOCOL['seeds'])
SUBSETS = tuple(PROTOCOL['subsets'])
EPOCHS = int(PROTOCOL['epochs'])
PATIENCE = int(PROTOCOL['patience'])
BATCH = int(PROTOCOL['batch'])
LR = float(PROTOCOL['lr'])
WEIGHT_DECAY = float(PROTOCOL['weight_decay'])
D_MODEL = int(PROTOCOL['d_model'])
LAYERS = int(PROTOCOL['layers'])
DROPOUT = float(PROTOCOL['dropout'])
N_COND = int(PROTOCOL['n_cond'])

# Stage C backbone (X2 ruling, 2026-11-11): the task book §5 requires the full
# module set -- multi-scale convolution + cross-sensor attention + learned
# sensor pooling -- so Stage C runs on ms_mode='full', sa_mode='basic', with no
# HI supervision (that is Stage D's object, not this one). The factorial
# candidate backbone (single_proj/basic/hf) drops the multi-scale convolution
# and is NOT used here; if ever reported it must be a separate table labelled
# "cycle effect on the candidate backbone" and never mixed with V0.
BACKBONE = dict(ms_mode='full', sa_mode='basic', hi_source='none',
                hi_mono='none', hi_lambda=0.0, hi_mono_coef=0.0)

# V0 triples (X3 ruling). V0-NoHI is the frozen-injection, cycle-on, no-HI
# model, i.e. exactly A3p; it is trained once and referenced from both tables.
V0_TRIPLES = {
    'V0C': dict(backbone=dict(ms_mode='full', sa_mode='basic', hi_source='hf'),
                cycle='fourier_frozen'),
    'V0N': dict(backbone=dict(ms_mode='full', sa_mode='basic', hi_source='hf'),
                cycle='off'),
    'V0H': dict(backbone=dict(ms_mode='full', sa_mode='basic', hi_source='none'),
                cycle='fourier_frozen'),   # == A3p
}

# ---------------------------------------------------------------------------
# Stage C cycle variants (task book §6)
# ---------------------------------------------------------------------------
# cycle_mode decides what the model receives:
#   none      - no cycle information at all (A1)
#   raw       - scalar cycle, linear projection (A2)
#   fourier   - original DSPT Fourier encoding (A3a/b/c differ by the scalar fed in)
#   log       - log1p(cycle), linear projection (A4)
#   spline    - monotone I-spline age basis, linear projection (A5)
# cycle_source decides the scalar:
#   clipped   - cycle / 250 clipped at 1.0  (exactly what data.py produces)
#   raw       - unclipped cycle in cycles
#   scaled500 - cycle / 500, unclipped (never saturates in the observed range)

VARIANTS = {
    'A0': dict(label='cycle-only baseline', cycle_mode='none', cycle_source='raw',
               cycle_only=True),
    'A1': dict(label='sensor-only DSPT', cycle_mode='none', cycle_source='raw',
               cycle_only=False),
    'A2': dict(label='sensor + raw cycle', cycle_mode='raw', cycle_source='raw',
               cycle_only=False),
    'A3a': dict(label='sensor + Fourier cycle (clipped at 250)', cycle_mode='fourier',
                cycle_source='clipped', cycle_only=False),
    'A3b': dict(label='sensor + Fourier cycle (unclipped)', cycle_mode='fourier',
                cycle_source='raw', cycle_only=False),
    'A3c': dict(label='sensor + Fourier cycle (wide period bank, no phase wrap)',
                cycle_mode='fourier', cycle_source='raw', period_bank='wide',
                cycle_only=False),
    'A4': dict(label='sensor + log cycle', cycle_mode='log', cycle_source='raw',
               cycle_only=False),
    'A5': dict(label='sensor + monotone spline age', cycle_mode='spline',
               cycle_source='raw', cycle_only=False),
    # Injection-point control (A2 ruling). Same Fourier form and same clipped
    # scalar as A3a, but added inside DSPT.encode() before the final LayerNorm,
    # exactly where the frozen model adds it. Without this arm the Gate A
    # comparison cannot separate "Fourier helps" from "post-norm injection
    # helps". A3p is also the V0-NoHI triple, so it is trained once and shared.
    'A3p': dict(label='sensor + Fourier cycle at the frozen injection point',
                cycle_mode='none', cycle_source='clipped', cycle_only=False,
                frozen_injection=True),
    # G2 scale-alignment arms (Codex ruling 2026-11-12). Same encoders as A2 and
    # A3b; the only change is that the cycle term is multiplied by an init-RMS
    # calibrated gain so that ||term||/||h|| == rho in every arm. `scale_mode`
    # selects DSPT_HC_Scaled in scale_align.build_hc_scaled; the gain itself is
    # measured per run and stored in calibration.json, never guessed here.
    'A2cal': dict(label='sensor + raw cycle (untruncated), init-RMS calibrated',
                  cycle_mode='raw', cycle_source='raw', cycle_only=False,
                  scale_mode='cal', rho=1.0),
    'A3bcal': dict(label='sensor + Fourier cycle (unclipped), init-RMS calibrated',
                   cycle_mode='fourier', cycle_source='raw', cycle_only=False,
                   scale_mode='cal', rho=1.0),
    # Pipeline check only. Same path as A3a; never evidence. Registered under
    # its own variant name so a smoke run can never be mistaken for a real cell.
    'SMOKE': dict(label='smoke: A3a path, 1 epoch, not evidence',
                  cycle_mode='fourier', cycle_source='clipped', cycle_only=False,
                  smoke_of='A3a'),
}

# Validation-split drift guard (audit R6). Definition:
#   md5(",".join(str(u) for u in sorted(val_units)))[:16]
# The reference values were recomputed independently by the results
# administrator. A mismatch means the split moved and every comparison against
# the frozen baseline is void, so run_hc refuses to start.
SPLIT_HASH_REF = {
    'FD001': 'f81d7f2496c494ca',
    'FD002': 'cacf4ea1d7cf96b6',
    'FD003': 'f81d7f2496c494ca',
    'FD004': 'eb1b0f00047d393e',
}


def split_hash(units):
    """R6 fingerprint of a sorted engine-id list."""
    import hashlib
    ids = ','.join(str(int(u)) for u in sorted(int(x) for x in units))
    return hashlib.md5(ids.encode()).hexdigest()[:16]


# Gate A pre-registered decision rules (see METHOD_SPECIFICATION.md §5).
GATE_A = dict(
    # (i) Fourier must beat the simplest age encoding (A2 raw cycle) on
    #     validation by more than the seed spread, otherwise the encoding is
    #     not a mechanism.
    fourier_min_gain_pct=1.0,
    # (ii) cycle-only must be materially worse than sensor-only, otherwise the
    #      benchmark is solvable from age alone and no cycle claim is possible.
    cycle_only_min_gap_pct=10.0,
    # (iii) an encoding counts as "helps understand degradation" only if the
    #       prediction responds to cycle while sensor evidence is held fixed,
    #       and the response direction is the physically correct one.
    intervention_min_abs_delta=0.5,
)


def variant(name):
    if name not in VARIANTS:
        raise KeyError(f'unknown variant {name!r}; known: {sorted(VARIANTS)}')
    return dict(VARIANTS[name], name=name)


def hc_config(variant_name, subset, seed, span=SPAN):
    """Full effective config for one run. Written to experiments/hc/<run_id>/."""
    v = variant(variant_name)
    frozen_inj = bool(v.get('frozen_injection', False))
    return dict(
        stage='C', variant=variant_name, label=v['label'],
        subset=subset, seed=int(seed), span=int(span),
        cycle_mode=v['cycle_mode'], cycle_source=v['cycle_source'],
        # A3p stores cycle_mode='none' because the base DSPT class carries the
        # Fourier term itself; the *effective* path is the frozen one. Both
        # fields are written so an auditor reading config.json cannot file A3p
        # as a no-cycle arm (P2-2).
        effective_cycle_mode='fourier_frozen' if frozen_inj else v['cycle_mode'],
        injection='frozen_pre_norm' if frozen_inj else (
            'post_norm' if v['cycle_mode'] != 'none' else 'none'),
        period_bank=v.get('period_bank', 'frozen'),
        frozen_injection=frozen_inj,
        # 'raw' = historical arms (no calibration); 'cal' = init-RMS calibrated.
        # The gain is not a config constant: it is measured per run (see
        # scale_align.calibrate_init_rms) and recorded in calibration.json.
        scale_mode=v.get('scale_mode', 'raw'),
        rho=float(v.get('rho', 1.0)),
        cycle_only=bool(v['cycle_only']),
        split_seed=SPLIT_SEED, fold=FOLD, val_frac=VAL_FRAC, val_cuts=list(VAL_CUTS),
        rul_cap=RUL_CAP, n_cond=N_COND,
        epochs=EPOCHS, patience=PATIENCE, batch=BATCH, lr=LR,
        weight_decay=WEIGHT_DECAY, d_model=D_MODEL, layers=LAYERS, dropout=DROPOUT,
        backbone=BACKBONE,
        aux_dual=True, mode='dual',
    )


def assert_protocol_matches_taskbook():
    """Cheap guard against editing protocol.json by accident."""
    expected = dict(split_seed=7, fold=0, val_frac=0.2, rul_cap=125.0, span=70,
                    epochs=40, patience=12, batch=512, lr=1e-3,
                    weight_decay=1e-4, d_model=64, layers=2, dropout=0.2, n_cond=6)
    for k, v in expected.items():
        got = PROTOCOL[k]
        if isinstance(v, float):
            ok = abs(float(got) - v) < 1e-12
        else:
            ok = got == v
        if not ok:
            raise AssertionError(f'protocol {k}={got!r} != task book {v!r}')
    if tuple(PROTOCOL['seeds']) != (101, 202, 303):
        raise AssertionError('seed list drifted from 101/202/303')
    if tuple(PROTOCOL['val_cuts']) != (0.4, 0.55, 0.7, 0.85, 0.95):
        raise AssertionError('validation cut points drifted')
    return True
