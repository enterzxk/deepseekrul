"""`vote_b.py` 的自测：三条条件逐条打穿，以及 SC 不得被当成 Full。

合成数据的构造是精确的：每个发动机的误差向量取 [+a, -a, +a, -a]，于是该发动机的
RMSE 恰好等于 a。于是「臂 A 的 RMSE = base + δ/2、臂 B = base − δ/2」给出逐发动机
Δ = δ，既能控制效应量，也能控制发动机间的异质性（bootstrap 宽窄）。
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vote_b  # noqa: E402

SEEDS = (101, 202, 303)
SUBSETS = ('FD001', 'FD004')
# 夹具与真实网格一致：5 个 SOTA（B_min 的 k0p 格不存在）
BACKBONES = ('aidgn', 'dual_mixer', 'fcstgnn', 'ffve', 'two_phase')
N_ENG = 20
BASE = 12.0
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append((name, bool(ok), detail))
    print(f'{name:58} {"OK" if ok else "MISMATCH"}   {detail}')


def _cell(root, b, u, seed, per_engine_rmse):
    d = root / b / u / f's{seed}'
    d.mkdir(parents=True, exist_ok=True)
    with (d / 'validation_predictions.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['unit', 'cut', 'y_true', 'y_pred'])
        for unit, rmse in sorted(per_engine_rmse.items()):
            for c, sg in enumerate((1.0, -1.0, 1.0, -1.0), 1):
                y = 100.0
                w.writerow([unit, c, y, y - sg * rmse])


def build(root, spec):
    """spec[(b,u)] = dict(delta=, mode='uniform'|'mixed'|'seedflip'|'sub_threshold')"""
    for b in BACKBONES:
        for u in SUBSETS:
            sp = spec.get((b, u), dict(delta=0.0, mode='uniform'))
            for s in SEEDS:
                d = sp['delta']
                if sp['mode'] == 'seedflip' and s == SEEDS[0]:
                    d = -abs(sp['delta'])
                if sp['mode'] == 'sub_threshold':
                    d = 0.44
                r0, r1 = {}, {}
                for k in range(N_ENG):
                    base = BASE + 0.5 * (k % 5)          # engine difficulty, same both arms
                    if sp['mode'] in ('mixed', 'winrate'):
                        if sp['mode'] == 'mixed':
                            off = 2.0 * d if k < 12 else -1.5 * d
                        else:
                            off = 1.5 * d if k < 11 else -0.5 * d
                        r0[k] = base + off / 2 + abs(off) / 2
                        r1[k] = base - off / 2 + abs(off) / 2
                    else:
                        r0[k] = base + d / 2
                        r1[k] = base - d / 2
                _cell(root / 'v0', b, u, s, r0)
                _cell(root / 'v1', b, u, s, r1)


def run_case(name, spec, want_verdict, want_k_full=None, extra_checks=None):
    td = Path(tempfile.mkdtemp())
    try:
        build(td, spec)
        rec = vote_b.judge(td / 'v0', td / 'v1', SEEDS, allow_missing_eol=True)
        ok = rec['verdict'] == want_verdict
        if want_k_full is not None:
            ok = ok and rec['k_full'] == want_k_full
        if extra_checks:
            ok = ok and extra_checks(rec)
        record(name, ok, f'verdict={rec["verdict"]} k_full={rec["k_full"]} k_sc={rec["k_sc"]}')
        return rec
    finally:
        shutil.rmtree(td, ignore_errors=True)


def main():
    allpass = {(b, u): dict(delta=1.0, mode='uniform') for b in BACKBONES for u in SUBSETS}

    rec = run_case('all five pass both subsets -> FULL', allpass, 'full', 5)
    record('Full 档 claim_scope 写跨工况',
           rec['claim_scope'].startswith('cross-condition') and not rec['forbidden_claims'],
           rec['claim_scope'])

    # S1: 一个 seed 反向 -> 3/3 同号失败 -> 该骨干不计票
    s1 = dict(allpass)
    s1[('aidgn', 'FD001')] = dict(delta=1.0, mode='seedflip')
    rec = run_case('S1 one seed negative -> no vote', s1, 'full', 4)
    u = [x for x in rec['units'] if x['backbone'] == 'aidgn' and x['subset'] == 'FD001'][0]
    record('S1 单测：same_sign=False 且 passes=False',
           u['s1_same_sign'] is False and u['passes'] is False and u['within_noise'],
           f's1={u["s1_same_sign"]} passes={u["passes"]}')

    # S2: 效应量 0.44 < 0.45
    s2 = dict(allpass)
    s2[('ffve', 'FD001')] = dict(delta=1.0, mode='sub_threshold')
    rec = run_case('S2 effect 0.44 < 0.45 -> no vote', s2, 'full', 4)
    u = [x for x in rec['units'] if x['backbone'] == 'ffve' and x['subset'] == 'FD001'][0]
    record('S2 单测：s2_effect=False 且 dbar=0.44',
           u['s2_effect'] is False and abs(u['delta_mean'] - 0.44) < 1e-9,
           f's2={u["s2_effect"]} dbar={u["delta_mean"]:.4f}')

    # S3（v7 三分支）：CI 跨 0 但 MDE > 0.45 -> 照计票，标 underpowered 进限制节
    s3 = dict(allpass)
    s3[('two_phase', 'FD001')] = dict(delta=1.0, mode='mixed')
    rec = run_case('S3 CI crosses 0 + underpowered -> not counted', s3, 'full', 4)
    u = [x for x in rec['units'] if x['backbone'] == 'two_phase' and x['subset'] == 'FD001'][0]
    record('S3 单测：underpowered 不计票，且标 evaluable=False',
           u['s3_branch'] == 'underpowered_but_counted' and u['passes'] is False
           and u['underpowered'] is True and u['evaluable'] is False
           and u['boot_ci'][0] <= 0.0 and u['mde'] > vote_b.EFFECT_MIN,
           f'branch={u["s3_branch"]} ci_lo={u["boot_ci"][0]:+.3f} '
           f'mde={u["mde"]:.3f} evaluable={u["evaluable"]}')
    record('underpowered 骨干进记录的 underpowered 名单',
           'two_phase' in rec['underpowered'], f'{rec["underpowered"]}')

    # 代数结论：Δ̄ >= 0.45 时「降级不计」分支不可达（MDE<=0.45 => SE<=0.161
    # => CI 半宽<=0.315 => lo>=0.135>0），所以条件 3 只贴标签、不拦票。
    unreachable, worst = True, 9.9
    for dbar in (0.45, 0.6, 1.0, 2.0, 5.0):
        for sd in (0.01, 0.1, 0.5, 1.0, 2.0, 5.0):
            se = sd / (N_ENG ** 0.5)
            lo = dbar - 1.96 * se
            if 2.80 * se <= vote_b.EFFECT_MIN and lo <= 0.0:
                unreachable = False
                worst = min(worst, lo)
    record('powered_but_null 在 Δ̄>=0.45 下不可达', unreachable,
           f'min lo over grid={worst if worst < 9 else "n/a"}')

    # S4: 胜率 0.55 < 0.60 -> 不计票。这条是牢莫抓出来的死参数：WIN_RATE_MIN 原先
    # 只写进记录、没有进 passes，§4.1 第 2 条的胜率条件被静默丢掉。
    s4 = dict(allpass)
    s4[('dual_mixer', 'FD001')] = dict(delta=1.0, mode='winrate')
    rec = run_case('S4 win rate 0.55 < 0.60 -> no vote', s4, 'full', 4)
    u = [x for x in rec['units'] if x['backbone'] == 'dual_mixer' and x['subset'] == 'FD001'][0]
    record('S4 单测：只有 S4 失败，S1/S2/S3 都过',
           u['s4_win_rate'] is False and abs(u['engine_win_rate'] - 0.55) < 1e-9
           and u['s1_same_sign'] and u['s2_effect'] and u['s3_branch'] == 'consistent'
           and not u['passes'],
           f'win={u["engine_win_rate"]:.2f} s4={u["s4_win_rate"]} '
           f's1/s2/s3={int(u["s1_same_sign"])}/{int(u["s2_effect"])}/{u["s3_branch"]}')

    # ② b_min 没有 k0p 格：显式要求它却没有输入时必须 exit 2，不能静默给绿
    td = Path(tempfile.mkdtemp())
    try:
        build(td, allpass)
        try:
            vote_b.judge(td / 'v0', td / 'v1', SEEDS, backbones=('b_min',),
                         allow_missing_eol=True)
            ok, detail = False, 'no exit'
        except SystemExit as e:
            ok, detail = e.code == vote_b.EXIT_ABORT, f'exit={e.code}'
        record('b_min without k0p cells -> ABORT', ok, detail)

        # ④ 单子集调用不得凭空声称跨工况
        rec = vote_b.judge(td / 'v0', td / 'v1', SEEDS, subsets=('FD001',),
                           allow_missing_eol=True)
        record('single-subset call never claims FD004',
               'FD001' in rec['claim_scope'] and 'FD004' not in rec['claim_scope']
               and rec['verdict'] != 'full',
               f'verdict={rec["verdict"]} scope={rec["claim_scope"]}')

        # P3: --eol-json 缺 evaluated 不能崩（旧代码 KeyError）
        try:
            r2 = vote_b.judge(td / 'v0', td / 'v1', SEEDS,
                              eol=dict(delta_mean=-0.02), allow_missing_eol=True)
            ok = r2['eol']['evaluated'] is True and r2['verdict'] in ('full', 'sc')
            detail = f'verdict={r2["verdict"]} evaluated={r2["eol"]["evaluated"]}'
        except Exception as e:                                    # noqa: BLE001
            ok, detail = False, f'{type(e).__name__}: {e}'
        record('EOL json without `evaluated` does not crash', ok, detail)
    finally:
        shutil.rmtree(td, ignore_errors=True)

    # P1: 只有 3/5 过线 -> 差一票，B_min 补上就是 4/6 -> PENDING，不能判 FAIL
    three = {(b, u): dict(delta=1.0, mode='uniform') for b in BACKBONES[:3] for u in SUBSETS}
    for b in BACKBONES[3:]:
        for u in SUBSETS:
            three[(b, u)] = dict(delta=-0.6, mode='uniform')
    rec = run_case('3/5 pass + 2 determinate fail -> FAIL', three, 'fail', 3)
    record('确定失败档：full_pass=False 且五票都可评估',
           rec['full_pass'] is False and rec['k_full_evaluable'] == 5,
           f'k_full={rec["k_full"]} eval={rec["k_full_evaluable"]}')

    # 牢莫的洞：3 过 + 1 underpowered + 1 败 -> k_pass + n_und = 4 >= 4，不能判 FAIL
    hole = {}
    for i, b in enumerate(BACKBONES):
        for u in SUBSETS:
            if i < 3:
                hole[(b, u)] = dict(delta=1.0, mode='uniform')
            elif i == 3:
                hole[(b, u)] = dict(delta=1.0, mode='mixed')
            else:
                hole[(b, u)] = dict(delta=-0.6, mode='uniform')
    rec = run_case('3 pass + 1 undetermined -> INCONCLUSIVE (not FAIL)', hole,
                   'inconclusive', 3)
    record('未定票可补足门槛时不判 FAIL',
           rec['k_full'] + rec['n_undetermined']['full'] >= 4,
           f'k_pass={rec["k_full"]} n_und={rec["n_undetermined"]["full"]}')

    # dual_mixer 那类协议偏差：既不进分子也不进分母
    dev = {}
    for i, b in enumerate(BACKBONES):
        for u in SUBSETS:
            dev[(b, u)] = (dict(delta=1.0, mode='uniform') if i < 3
                           else dict(delta=-0.6, mode='uniform'))
    td2 = Path(tempfile.mkdtemp())
    try:
        build(td2, dev)
        r = vote_b.judge(td2 / 'v0', td2 / 'v1', SEEDS, allow_missing_eol=True,
                         deviant=('dual_mixer',))
        record('protocol_deviant 既不进分子也不进分母',
               r['per_backbone']['dual_mixer']['protocol_deviant'] is True
               and r['per_backbone']['dual_mixer']['votes_full'] is False
               and r['k_full'] == 2 and r['k_full_evaluable'] == 4
               and r['verdict'] == 'fail',
               f'k_full={r["k_full"]} eval={r["k_full_evaluable"]} verdict={r["verdict"]}')
    finally:
        shutil.rmtree(td2, ignore_errors=True)

    # 经理：underpowered 剔完后可评估票 <4 且不差一票 -> INCONCLUSIVE
    inc = {}
    for i, b in enumerate(BACKBONES):
        for u in SUBSETS:
            if i < 2:
                inc[(b, u)] = dict(delta=1.0, mode='uniform')
            elif i < 4:
                inc[(b, u)] = dict(delta=1.0, mode='mixed')      # underpowered
            else:
                inc[(b, u)] = dict(delta=-0.6, mode='uniform')   # fail
    rec = run_case('underpowered 剔完可评估<4 -> INCONCLUSIVE', inc, 'inconclusive', 2)
    record('INCONCLUSIVE 报出可评估票数',
           rec['k_full_evaluable'] < 4 and rec['k_sc_evaluable'] < 4,
           f'k_full={rec["k_full"]} eval={rec["k_full_evaluable"]}/{rec["k_sc_evaluable"]}')

    # SC: 四骨干 FD001 过线、FD004 无害但不过线 -> 只能是 SC
    sc = {}
    for b in BACKBONES:
        sc[(b, 'FD001')] = dict(delta=1.0, mode='uniform')
        sc[(b, 'FD004')] = (dict(delta=1.0, mode='uniform') if b in ('aidgn', 'dual_mixer')
                            else dict(delta=-0.05, mode='uniform'))
    rec = run_case('4 backbones FD001-only -> SC (not Full)', sc, 'sc', 2)
    record('SC 不得被当成 Full：full_pass=False 且 forbidden_claims 非空',
           rec['full_pass'] is False and rec['sc_pass'] is True
           and '跨工况通用' in rec['forbidden_claims'],
           f'full={rec["full_pass"]} sc={rec["sc_pass"]} forbidden={rec["forbidden_claims"]}')
    record('SC claim_scope 写「单工况前端」',
           rec['claim_scope'].startswith('single-condition'), rec['claim_scope'])

    # FD004 有害（dbar = -0.5 < -0.10）-> 连 SC 都不算
    bad = {}
    for b in BACKBONES:
        bad[(b, 'FD001')] = dict(delta=1.0, mode='uniform')
        bad[(b, 'FD004')] = dict(delta=-0.5, mode='uniform')
    rec = run_case('FD004 harmful -> FAIL (not even SC)', bad, 'fail', 0)

    # 缺格 -> exit 2，不许插补
    td = Path(tempfile.mkdtemp())
    try:
        build(td, allpass)
        (td / 'v1' / 'aidgn' / 'FD001' / 's202' / 'validation_predictions.csv').unlink()
        try:
            vote_b.judge(td / 'v0', td / 'v1', SEEDS)
            ok, detail = False, 'no exit'
        except SystemExit as e:
            ok, detail = e.code == vote_b.EXIT_ABORT, f'exit={e.code}'
        record('missing cell -> ABORT (never interpolated)', ok, detail)
    finally:
        shutil.rmtree(td, ignore_errors=True)

    # EOL 否决
    td = Path(tempfile.mkdtemp())
    try:
        build(td, allpass)
        rec = vote_b.judge(td / 'v0', td / 'v1', SEEDS,
                           eol=dict(evaluated=True, delta_mean=-0.20))
        record('EOL veto -> FAIL', rec['verdict'] == 'fail' and 'veto' in rec['eol'],
               f'verdict={rec["verdict"]}')
    finally:
        shutil.rmtree(td, ignore_errors=True)

    bad = [r for r in RESULTS if not r[1]]
    print()
    if bad:
        print(f'SELFTEST FAILED  ({len(RESULTS) - len(bad)}/{len(RESULTS)} checks)')
        for n, _, d in bad:
            print(f'  FAILED: {n}  {d}')
        return 1
    print(f'SELFTEST PASSED  ({len(RESULTS)}/{len(RESULTS)} checks)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
