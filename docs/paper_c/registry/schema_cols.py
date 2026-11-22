"""Single source for the runs.csv column order (52 columns, v2).

Imported by gen_queue.py (to write the empty header) and check_registry.py (to
assert runs.csv still matches). Do not add a column to runs.csv without adding
it here - the checker rejects a header that differs from this list, and that is
the point: the ledger's shape must not drift silently.
"""

COLS = [
    'run_id',              # 1  see SCHEMA.md §2 (pattern depends on the arm)
    'wave',                # 2  W1
    'stage',               # 3  P2 / Terminal (only Terminal unlocks TEST)
    'segment',             # 4  A = V0 reproduction, B = +CADF
    'arm',                 # 5  v0 / cadf
    'base_model',          # 6  aidgn / two_phase / dual_mixer / ffve / fcstgnn
    'cadf_variant',        # 7  k0p / k0a / eng / none
    'cadf_k0_mode',        # 8  prop / abs / none
    'cadf_k0_value',       # 9  resolved K0 (this column IS k0_resolved)
    'subset',              # 10 FD001 / FD004
    'seed',                # 11 101 / 202 / 303
    'span',                # 12 window length for this (model, subset)
    'split_protocol',      # 13
    'split_seed',          # 14
    'gate',                # 15 experiment type only (P2_cross_arch)
    'in_gate_table',       # 16 boolean, independent of `gate`
    'hypothesis',          # 17
    'host',                # 18 lab3090 / lab-a40
    'git_sha',             # 19 the code that actually ran
    'code_md5',            # 20 JSON string of the five training files
    'config_sha256',       # 21
    'launched_utc',        # 22
    'closed_utc',          # 23
    'status',              # 24 ok / failed / smoke / running / superseded
    'epochs_run',          # 25
    'best_epoch',          # 26
    'budget_truncated',    # 27
    'val_engine_mean_rmse',# 28
    'val_pooled_rmse',     # 29
    'val_mae',             # 30
    'val_bias',            # 31
    'eol95_mae',           # 32
    'val_nasa',            # 33
    'test_rmse',           # 34 SEALED
    'test_score',          # 35 SEALED
    'ckpt_path',           # 36
    'log_path',            # 37
    'preds_path',          # 38
    'gate_alpha_trace',    # 39 per-epoch gate/alpha norm trace (mandatory for arm=cadf)
    'gate_open_epoch',     # 40 first epoch with mean(gate) > 0.01, or 'none'
    'gate_norm_last',      # 41 ||gate|| at the last epoch
    'alpha_last',          # 42 |alpha| at the last epoch
    'baseline_source',     # 43 frozen_ledger (A) / local_v0 (B)
    'baseline_ref',        # 44 physical_id (A) / run_id of the segment-A cell (B)
    'baseline_val_engine_mean_rmse',  # 45
    'delta_val_rmse',      # 46 baseline - this cell, positive = better
    'paired_win_rate',     # 47 per-engine win rate vs the paired baseline
    'registry_note',       # 48
    'audited_by',          # 49 people only
    'audit_verdict',       # 50 pass / fail / n/a
    'determinism_only',    # 51 boolean; never a verdict
    'baseline_version',    # 52
]
