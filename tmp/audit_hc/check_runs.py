"""Read-only consistency check for docs/paper_b_hc/registry/runs.csv.

Rules (test-lead, 2026-11-11):
  R-a  status/audit_verdict must form one of the legal pairs:
         ok + pass | ok + fail | failed + n/a | smoke + n/a
       Anything else is a ledger inconsistency, because downstream counts
       ("how many usable runs") read these two columns independently.
  R-b  a row carrying a numeric val_rmse must say which aggregation it is
       (val_metric_kind). The frozen trainer's ``val_rmse`` used to be the
       POOLED value; the hc trainer writes the ENGINE-MEAN value. One column
       holding both is not aggregatable.
  R-c  rows produced before the R2 metric change (git_sha outside the frozen
       baseline set) are legacy-口径 and must not be pooled with hc rows.

Usage: python tmp/audit_hc/check_runs.py [path]
Exit code 0 when clean, 1 when anything is flagged.
"""
import sys

import pandas as pd

DEFAULT = r'C:\Users\20562\Desktop\deepseek-rul\docs\paper_b_hc\registry\runs.csv'
# experiment-registry rule 3: status only ok/failed/smoke/running/superseded;
# verdict only pass/fail/n/a. determinism_only is a boolean column, never a verdict.
LEGAL = {('ok', 'pass'), ('ok', 'fail'), ('failed', 'n/a'), ('smoke', 'n/a'),
         ('superseded', 'n/a'), ('running', 'n/a')}
# Commits whose runs may claim the engine-mean protocol.
ENGINE_MEAN_SHAS = {'f314268', 'f83e27e', 'a203f4b', '0280cb8'}


def main(path=DEFAULT):
    df = pd.read_csv(path)
    df['audit_verdict'] = df['audit_verdict'].fillna('n/a')
    flags = []

    for _, r in df.iterrows():
        pair = (r['status'], r['audit_verdict'])
        if pair not in LEGAL:
            flags.append(('R-a status/verdict', r['run_id'], f'{pair[0]} / {pair[1]}'))

    if 'val_metric_kind' not in df.columns:
        n = df['val_rmse'].notna().sum()
        flags.append(('R-b metric kind', '-', f'no val_metric_kind column; {n} rows carry val_rmse'))
    else:
        bad = df[df['val_rmse'].notna() & df['val_metric_kind'].isna()]
        for _, r in bad.iterrows():
            flags.append(('R-b metric kind', r['run_id'], 'val_rmse without kind'))

    # R-c: rows produced before the R2 metric change are legacy-口径 and must not be pooled.
    # 2026-11-12 amendment: two legitimate cases were being flagged as legacy:
    #   (a) the 24 Gate B rows are legacy ON PURPOSE -- they carry legacy_metric=true, so
    #       flagging them tells us nothing new; they are already excluded downstream.
    #   (b) hc runs on a40 have git_sha=WORKTREE by design (a40 has no git); their version
    #       binding lives in baseline_version + the 15-blob code_md5, not in a commit sha.
    legacy = df[(df['status'] == 'ok')
                & (~df['git_sha'].astype(str).isin(ENGINE_MEAN_SHAS))
                & (df.get('legacy_metric', 'false').astype(str).str.lower() != 'true')
                & ~((df['git_sha'].astype(str) == 'WORKTREE')
                    & (df.get('baseline_version', '').astype(str) != ''))]
    for _, r in legacy.iterrows():
        flags.append(('R-c legacy protocol', r['run_id'],
                      f"git_sha={r['git_sha']} predates the engine-mean metric"))

    print(f'rows={len(df)}  flags={len(flags)}')
    for rule, rid, why in flags:
        print(f'  [{rule}] {rid}: {why}')
    return 0 if not flags else 1


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
