"""`vote_b.py` 的自测：三条条件逐条打穿，以及 SC 不得被当成 Full。

合成数据的构造是精确的：每个发动机的误差向量取 [+a, -a, +a, -a]，于是该发动机的
RMSE 恰好等于 a。于是「臂 A 的 RMSE = base + δ/2、臂 B = base − δ/2」给出逐发动机
Δ = δ，既能控制效应量，也能控制发动机间的异质性（bootstrap 宽窄）。
"""
from __future__ import annotations

import contextlib
import csv
import io
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
                    if sp['mode'] in ('mixed', 'winrate', 'null'):
                        if sp['mode'] == 'mixed':
                            off = 2.0 * d if k < 12 else -1.5 * d
                        elif sp['mode'] == 'winrate':
                            off = 1.5 * d if k < 11 else -0.5 * d
                        else:
                            off = 0.5 * d if k < 10 else -0.3 * d
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

    # 经理点名（v7.4）：**Δ̄ 不达标但 CI 跨 0** -> 确定 FAIL，不是 underpowered。
    # 这是「CI 先分」那个 bug 的判据性反例：同一格 ci_branch 是 underpowered
    # （MDE>0.45），但按固定顺序它在第 2 步就被判成 effect_size_fail，**留在分母**。
    s3 = dict(allpass)
    s3[('two_phase', 'FD001')] = dict(delta=0.5, mode='mixed')
    rec = run_case('dbar=0.30 < 0.45 with wide CI -> determinate FAIL', s3, 'full', 4)
    u = [x for x in rec['units'] if x['backbone'] == 'two_phase' and x['subset'] == 'FD001'][0]
    record('S3 单测：同号成立、效应量不足、CI 跨 0 -> effect_size_fail 且留在分母',
           u['s3_branch'] == 'effect_size_fail' and u['ci_branch'] == 'underpowered'
           and u['evaluable'] is True and u['determinate_fail'] is True
           and u['passes'] is False and u['mde'] > vote_b.EFFECT_MIN
           and u['boot_ci'][0] <= 0.0,
           f'branch={u["s3_branch"]} ci_branch={u["ci_branch"]} '
           f'dbar={u["delta_mean"]:+.3f} ci_lo={u["boot_ci"][0]:+.3f} mde={u["mde"]:.3f}')
    record('CI/MDE 仍写进记录（报告项）',
           'boot_ci' in u and 'mde' in u and 'required_n_engines' in u
           and 'ci_detectable_delta' in u and 'effective_threshold' not in u,
           f'mde={u["mde"]:.3f} required_n={u["required_n_engines"]}')
    record('p_null 与 MDE 并排写进记录（每表必报）',
           'p_null' in u and 0.0 <= u['p_null'] < 0.5
           and 'p_null' in rec['power']['per_unit'][0],
           f'p_null={u["p_null"]:.4f}')

    # 真实数据形态（aidgn/FD001：三个 Δ 全负、Δ̄ −0.36、CI 跨 0、MDE 1.1）：
    # 全负 -> same_sign 不成立 -> sign_fail，同样**留在分母**，不是「测不出」。
    s3b = dict(allpass)
    s3b[('aidgn', 'FD001')] = dict(delta=-0.5, mode='mixed')
    rec = run_case('all-negative seeds with wide CI -> sign_fail, stays', s3b, 'full', 4)
    u = [x for x in rec['units'] if x['backbone'] == 'aidgn' and x['subset'] == 'FD001'][0]
    record('S3b 单测：全负 + 宽 CI -> sign_fail（不是 underpowered）',
           u['s3_branch'] == 'sign_fail' and u['ci_branch'] == 'underpowered'
           and u['evaluable'] is True and u['determinate_fail'] is True,
           f'branch={u["s3_branch"]} ci_branch={u["ci_branch"]} '
           f'dbar={u["delta_mean"]:+.3f}')

    # 真正的 underpowered：三样筛条件都过（Δ̄ 0.60、胜率 0.60）但 CI 跨 0、MDE 1.1。
    # 只有这一支能离开分母，所以 k_full 从 5 掉到 4，而 n_und 记 1。
    up = dict(allpass)
    up[('two_phase', 'FD001')] = dict(delta=1.0, mode='mixed')
    rec = run_case('genuine underpowered -> leaves the denominator', up, 'full', 4)
    u = [x for x in rec['units'] if x['backbone'] == 'two_phase' and x['subset'] == 'FD001'][0]
    record('underpowered 是唯一离开分子分母的一支',
           u['s3_branch'] == 'underpowered' and u['evaluable'] is False
           and u['determinate_fail'] is False and u['passes'] is False
           and rec['k_full'] == 4 and rec['n_undetermined']['full'] == 1
           and rec['k_full_evaluable'] == 4,
           f'branch={u["s3_branch"]} eval={u["evaluable"]} k_full={rec["k_full"]} '
           f'n_und={rec["n_undetermined"]["full"]}')

    # 代数结论：走到 CI 分支意味着 Δ̄ >= 0.45；此时 MDE <= 0.45 蕴含 lo >= 0.135 > 0，
    # 所以 powered_but_null 在 v7.4 顺序下不可达（保留分支只为与冻结文本对应）。
    unreachable, worst = True, 9.9
    for dbar in (0.45, 0.6, 1.0, 2.0, 5.0):
        for sd in (0.01, 0.1, 0.5, 1.0, 2.0, 5.0):
            se = sd / (N_ENG ** 0.5)
            lo = dbar - 1.96 * se
            if 2.80 * se <= vote_b.EFFECT_MIN and lo <= 0.0:
                unreachable = False
                worst = min(worst, lo)
    record('powered_but_null 在 v7.4 顺序下不可达', unreachable,
           f'min lo over grid={worst if worst < 9 else "n/a"}')

    # S4: 胜率 0.55 < 0.60 -> 不计票。这条是牢莫抓出来的死参数：WIN_RATE_MIN 原先
    # 只写进记录、没有进 passes，§4.1 第 2 条的胜率条件被静默丢掉。
    s4 = dict(allpass)
    s4[('dual_mixer', 'FD001')] = dict(delta=1.0, mode='winrate')
    rec = run_case('S4 win rate 0.55 < 0.60 -> no vote', s4, 'full', 4)
    u = [x for x in rec['units'] if x['backbone'] == 'dual_mixer' and x['subset'] == 'FD001'][0]
    record('S4 单测：胜率不足先判死，即使 CI 下界 > 0',
           u['s4_win_rate'] is False and abs(u['engine_win_rate'] - 0.55) < 1e-9
           and u['s1_same_sign'] and u['s2_effect']
           and u['s3_branch'] == 'win_rate_fail' and u['ci_branch'] == 'consistent'
           and u['determinate_fail'] is True and not u['passes'],
           f'win={u["engine_win_rate"]:.2f} s4={u["s4_win_rate"]} '
           f'branch={u["s3_branch"]} ci_branch={u["ci_branch"]}')

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

    # `null` 形态（Δ̄=0.10）：在 v7.4 顺序下先卡在效应量，是 effect_size_fail（确定败票），
    # 不再可能是 powered_but_null——那一支只在「三样筛条件全过」之后才可能被问到，
    # 而那时 MDE<=0.45 蕴含 lo>0，必然先命中 consistent。
    pn = dict(allpass)
    pn[('ffve', 'FD001')] = dict(delta=1.0, mode='null')
    rec = run_case('null shape -> effect_size_fail (not powered_but_null)', pn, 'full', 4)
    u = [x for x in rec['units'] if x['backbone'] == 'ffve' and x['subset'] == 'FD001'][0]
    record('Δ̄=0.10 的格是确定败票，留在分母',
           u['s3_branch'] == 'effect_size_fail' and u['evaluable'] is True
           and u['underpowered'] is False and u['determinate_fail'] is True
           and 'ffve' not in rec['underpowered']
           and rec['k_full_evaluable'] == 5,
           f'branch={u["s3_branch"]} dbar={u["delta_mean"]:+.3f} '
           f'mde={u["mde"]:.3f} eval={rec["k_full_evaluable"]}')

    # 牢莫的洞（回滚后它是活的）：3 过 + 1 underpowered + 1 确定败 -> k_pass + n_und = 4
    # >= k_min，未定票补上就能过门，所以**不能判 FAIL**，只能是 INCONCLUSIVE。
    hole = {}
    for i, b in enumerate(BACKBONES):
        for u in SUBSETS:
            if i < 3:
                hole[(b, u)] = dict(delta=1.0, mode='uniform')
            elif i == 3:
                hole[(b, u)] = dict(delta=1.0, mode='mixed')
            else:
                hole[(b, u)] = dict(delta=-0.6, mode='uniform')
    rec = run_case('3 pass + 1 underpowered + 1 fail -> INCONCLUSIVE', hole,
                   'inconclusive', 3)
    record('n_und 是活的：未定票补上就能到 4 -> INCONCLUSIVE 而不是 FAIL',
           rec['k_full'] == 3 and rec['n_undetermined']['full'] == 1
           and rec['k_full_evaluable'] == 4,
           f'k_pass={rec["k_full"]} n_und={rec["n_undetermined"]["full"]} '
           f'eval={rec["k_full_evaluable"]} verdict={rec["verdict"]}')

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

    # v7.2：underpowered 不再影响分母，所以 INCONCLUSIVE 只由 protocol_deviant 触发
    inc = {}
    for i, b in enumerate(BACKBONES):
        for u in SUBSETS:
            inc[(b, u)] = (dict(delta=1.0, mode='uniform') if i < 3
                           else dict(delta=-0.6, mode='uniform'))
    td3 = Path(tempfile.mkdtemp())
    try:
        build(td3, inc)
        r = vote_b.judge(td3 / 'v0', td3 / 'v1', SEEDS, allow_missing_eol=True,
                         deviant=('ffve', 'two_phase'))
        record('可评估票 <4 -> INCONCLUSIVE（只由 protocol_deviant 触发）',
               r['verdict'] == 'inconclusive' and r['k_full_evaluable'] == 3
               and r['k_full'] == 3,
               f'verdict={r["verdict"]} eval={r["k_full_evaluable"]} k_pass={r["k_full"]}')
    finally:
        shutil.rmtree(td3, ignore_errors=True)

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

    # 牢莫（第五次「判定写对、记录层/输出层没对齐」）：自测全部走 judge()，main() 一次
    # 都没被覆盖，所以 s3_boot 这种不存在的字段能一直活到出 verdict 才炸。这里真调 main()：
    # 断言退出码、stdout 关键字段、--record 落盘。任何字段改名都会在这里红。
    td = Path(tempfile.mkdtemp())
    try:
        build(td, allpass)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = vote_b.main(['--v0', str(td / 'v0'), '--v1', str(td / 'v1'),
                              '--allow-missing-eol', '--record', str(td / 'rec.json')])
        out = buf.getvalue()
        recj = json.loads((td / 'rec.json').read_text(encoding='utf-8'))
        record('main() 端到端：PASS 时 EXIT=0、打印分支名、记录落盘',
               rc == vote_b.EXIT_OK and 'S3=consistent' in out
               and recj['verdict'] == 'full' and recj['k_full'] == 5
               and 'p_null' in recj['units'][0],
               f'exit={rc} verdict={recj["verdict"]} k_full={recj["k_full"]}')
        record('记录里写明读了哪两棵树（不靠目录名推断谁是 V0）',
               recj['inputs']['v0_root'] == str(td / 'v0')
               and recj['inputs']['v1_root'] == str(td / 'v1')
               and str(td / 'v0') in out,
               f'v0={recj["inputs"]["v0_root"]} v1={recj["inputs"]["v1_root"]}')

        fail_spec = {(b, u): dict(delta=-0.6, mode='uniform')
                     for b in BACKBONES for u in SUBSETS}
        build(td, fail_spec)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = vote_b.main(['--v0', str(td / 'v0'), '--v1', str(td / 'v1'),
                              '--allow-missing-eol'])
        record('main() 端到端：FAIL 时 EXIT=1（不能一律 0）',
               rc == vote_b.EXIT_FAIL and 'verdict=FAIL' in buf.getvalue(),
               f'exit={rc}')
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
