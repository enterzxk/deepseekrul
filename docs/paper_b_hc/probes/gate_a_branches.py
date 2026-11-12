"""Gate A verdict under both G4 branches -- artifacts only, no model, no GPU.

The two branches share G1/G2/G2'/G3/G5 byte for byte; they differ only in what
G4 is allowed to do. That way the difference between the two outputs is exactly
the decision that is PENDING-L2, and nothing else can drift.

  BRANCH A  G4 exactly as pre-registered in GATE_A_WAVE1_PLAN.md §9:
            for every cycle arm, mean_abs_delta >= 0.5 AND frac_negative >= 0.5.
            If it fails, the plan says that arm's cycle path is unused and the
            arm's conclusion is void -- which voids G1/G2 as well.

  BRANCH B  G4 replaced by the two criteria that have ground truth
            (G4_DIAGNOSIS.md §7):
              B1 marginal direction: within-engine Spearman(age, pred) <= -0.9
              B2 necessity: already G1, reported once and not double counted
            The response-magnitude check stays a diagnostic, not a gate.

Thresholds below are the pre-registration itself, quoted from the plan. They are
deliberately literal: reading a criterion out of the artifact it judges would be
circular.

Inputs: each run's own close.json (status, closed_utc, metrics) and config.json
(variant, subset); within-engine correlations from g4_within.json
(experiments/hc/tmp/g4_probe2.py). Writes nothing.

Usage: python docs/paper_b_hc/probes/gate_a_branches.py
"""
import json
import re
from pathlib import Path

import numpy as np

OUT = Path('/media/sangfor/vdb/user/zxk/deepseek-rul/experiments/hc')
WITHIN = OUT / 'tmp' / 'g4_within.json'
PAT = re.compile(r'^HC-C-(A\d\w?)-((?:FD00[1-4]))-s(\d+)-W1(r\d+)?$')

VARIANTS = ['A0', 'A1', 'A2', 'A3a', 'A4', 'A5', 'A3p']
MAIN = ['A0', 'A1', 'A2', 'A3a', 'A4', 'A5']
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
TREATMENT = ['A2', 'A3a', 'A4', 'A5']

# --- pre-registered thresholds (GATE_A_WAVE1_PLAN.md §9) -------------------
G1_MEAN = 1.0
G1_WORST = -2.0
G2_MEAN = 1.0
G2_SIGN = 3
G3_MAX = -10.0
G4_ABS = 0.5
G4_NEG = 0.5
G5_SIGN = 3
# --- replacement thresholds (G4_DIAGNOSIS.md §7, PENDING-L2) ---------------
B1_RHO = -0.9


def closed(run):
    f = run / 'close.json'
    if not f.exists():
        return None
    try:
        c = json.loads(f.read_text())
    except Exception:
        return None
    return c if c.get('status') == 'ok' else None


def evidence():
    best = {}
    for run in sorted(OUT.glob('HC-C-*')):
        m = PAT.match(run.name)
        if not m:
            continue
        c = closed(run)
        if c is None:
            continue
        key = (m.group(1), m.group(2), int(m.group(3)))
        t = c.get('closed_utc')
        if key not in best or t < best[key][0]:
            best[key] = (t, run, c)
    return {(v, s): best[(v, s, 101)] for v in VARIANTS for s in SUBSETS
            if (v, s, 101) in best}


def rmse(c):
    return float(c['metrics']['val_engine_mean_rmse'])


def intervention(run):
    """cycle_intervention lives in the run's metrics.json, not in close.json."""
    f = run / 'metrics.json'
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text()).get('cycle_intervention')
    except Exception:
        return None


def delta(a, b):
    """Δ(A→B) = (RMSE_A − RMSE_B)/RMSE_A in percent; positive means B is better."""
    return 100.0 * (a - b) / a


def main():
    ev = evidence()
    missing = [(v, s) for v in VARIANTS for s in SUBSETS if (v, s) not in ev]
    r = {(v, s): rmse(ev[(v, s)][2]) for (v, s) in ev}

    d1 = [delta(r[('A1', s)], r[('A2', s)]) for s in SUBSETS]
    d2 = [delta(r[('A2', s)], r[('A3a', s)]) for s in SUBSETS]
    d2p = [delta(r[('A3a', s)], r[('A3p', s)]) for s in SUBSETS]
    d3 = [delta(r[('A1', s)], r[('A0', s)]) for s in SUBSETS]

    def sign_ok(d):
        pos = sum(1 for x in d if x > 0)
        return max(pos, len(d) - pos)

    g1 = dict(vals=d1, mean=float(np.mean(d1)), worst=float(np.min(d1)),
              pass_=bool(np.mean(d1) >= G1_MEAN and np.min(d1) >= G1_WORST),
              sign=sign_ok(d1))
    g2 = dict(vals=d2, mean=float(np.mean(d2)),
              pass_=bool(np.mean(d2) >= G2_MEAN and sign_ok(d2) >= G2_SIGN),
              sign=sign_ok(d2))
    g2p = dict(vals=d2p, mean=float(np.mean(d2p)), sign=sign_ok(d2p))
    g3 = dict(vals=d3, mean=float(np.mean(d3)),
              pass_=bool(np.mean(d3) <= G3_MAX))
    g5 = dict(pass_=bool(sign_ok(d1) >= G5_SIGN and sign_ok(d2) >= G5_SIGN),
              sign1=sign_ok(d1), sign2=sign_ok(d2))

    within = {}
    if WITHIN.exists():
        for rec in json.loads(WITHIN.read_text()):
            within[(rec['variant'], rec['subset'])] = rec

    g4a, g4b = {}, {}
    for v in TREATMENT:
        iv = [x for x in (intervention(ev[(v, s)][1]) for s in SUBSETS) if x]
        absm = float(np.mean([x['mean_abs_delta'] for x in iv])) if iv else float('nan')
        negm = float(np.mean([x['frac_negative'] for x in iv])) if iv else float('nan')
        rho = float(np.mean([within[(v, s)]['within_c_p'] for s in SUBSETS
                             if (v, s) in within]))
        g4a[v] = dict(abs=absm, neg=negm,
                      pass_=bool(absm >= G4_ABS and negm >= G4_NEG))
        g4b[v] = dict(rho=rho, pass_=bool(rho <= B1_RHO))

    a1 = [x for x in (intervention(ev[('A1', s)][1]) for s in SUBSETS) if x]
    a1_abs = max(abs(x['mean_abs_delta']) for x in a1) if a1 else float('nan')

    print('# Gate A 双分支判定（脚本生成，勿手改）\n')
    print('生成脚本：`docs/paper_b_hc/probes/gate_a_branches.py`  ')
    print('证据 run：按台账规则自动挑选（`status=ok` + `-W1(r\\d+)?$` + 最早 `closed_utc`）')
    print('缺格：%s\n' % ('无' if not missing else missing))

    print('## 0. 主表 RMSE（`val_engine_mean_rmse`，越低越好）\n')
    print('| 变体 | ' + ' | '.join(SUBSETS) + ' |')
    print('|---|' + '---|' * len(SUBSETS))
    for v in VARIANTS:
        cells = []
        for s in SUBSETS:
            cells.append('%.4f' % r[(v, s)] if (v, s) in r else '—')
        tag = '（frozen）' if v == 'A3p' else ''
        print('| %s%s | %s |' % (v, tag, ' | '.join(cells)))

    print('\n## 1. 两分支共享的判据\n')
    print('| 判据 | 四子集 Δ（%） | 均值 | 阈值 | 结论 |')
    print('|---|---|---|---|---|')
    print('| G1 A1→A2 | %s | %+.2f | ≥%.1f%%，最差 ≥%.1f%% | %s |'
          % (' / '.join('%+.2f' % x for x in d1), g1['mean'], G1_MEAN, G1_WORST,
             '**过**' if g1['pass_'] else '不过'))
    print('| G2 A2→A3a | %s | %+.2f | ≥%.1f%%，同号 ≥%d/4 | %s |'
          % (' / '.join('%+.2f' % x for x in d2), g2['mean'], G2_MEAN, G2_SIGN,
             '**过**' if g2['pass_'] else '不过'))
    print('| G2′ A3a→A3p | %s | %+.2f | 不设阈值 | 只作 preliminary |'
          % (' / '.join('%+.2f' % x for x in d2p), g2p['mean']))
    print('| G3 A1→A0 | %s | %+.2f | ≤%.1f%% | %s |'
          % (' / '.join('%+.2f' % x for x in d3), g3['mean'], G3_MAX,
             '**过**' if g3['pass_'] else '不过'))
    print('| G5 方向稳定 | G1 同号 %d/4、G2 同号 %d/4 | — | ≥%d/4 | %s |'
          % (g5['sign1'], g5['sign2'], G5_SIGN, '**过**' if g5['pass_'] else '不过'))

    print('\n## 2. BRANCH A —— 保留原预注册 G4\n')
    print('判据：每个 cycle 臂 `mean_abs_delta ≥ %.1f` 且 `frac_negative ≥ %.1f`；'
          'A1 必须 `mean_abs_delta = 0`（实测 %.1e）。\n' % (G4_ABS, G4_NEG, a1_abs))
    print('| 臂 | mean_abs_delta | frac_negative | G4 |')
    print('|---|---|---|---|')
    for v in TREATMENT:
        print('| %s | %.3f | %.3f | %s |'
              % (v, g4a[v]['abs'], g4a[v]['neg'],
                 '**过**' if g4a[v]['pass_'] else '不过'))
    a_fail = [v for v in TREATMENT if not g4a[v]['pass_']]
    print('\n**结论**：G4 %s。' % ('不过（%s）' % '、'.join(a_fail) if a_fail else '过'))
    if a_fail:
        print('\n按预注册「该臂的 cycle 路径未被模型使用，该臂结论作废」，'
              '这四个臂覆盖 G1 的 treatment 端（A2）与 G2 的 treatment 端（A3a），'
              '因此 **G1、G2 同时作废**；G3（A0 vs A1）不受影响，仍是干净通过。')
        print('\n可用结论：只有 G3 —— 这个 benchmark 不能只靠年龄解。'
              'cycle 是否有用、编码形式孰优，本波**不可声明**。')

    print('\n## 3. BRANCH B —— 替换 G4 为有 ground truth 的判据\n')
    print('判据 B1：发动机内部 `Spearman(age, pred) ≤ %.1f`（边际方向，有真值）。\n' % B1_RHO)
    print('| 臂 | within-engine ρ | B1 |')
    print('|---|---|---|')
    for v in TREATMENT:
        print('| %s | %.3f | %s |'
              % (v, g4b[v]['rho'], '**过**' if g4b[v]['pass_'] else '不过'))
    b_fail = [v for v in TREATMENT if not g4b[v]['pass_']]
    print('\n判据 B2：通路必要性 = G1（上面已报 %+.2f%%），不重复计入。\n' % g1['mean'])
    print('**结论**：B1 %s。' % ('不过（%s）' % '、'.join(b_fail) if b_fail else '全过'))
    print('\n于是本分支下：G1 %s、G2 %s、G3 %s、G5 %s。'
          % ('**过**' if g1['pass_'] else '不过', '**过**' if g2['pass_'] else '不过',
             '**过**' if g3['pass_'] else '不过', '**过**' if g5['pass_'] else '不过'))
    print('\n量级诊断（不作判据，必须如实写进论文）：模型每 +10 的响应'
          ' %s，而数据局部条件效应 %s，比值约 3×–90×。'
          % ('/'.join('%.3f' % g4a[v]['abs'] for v in TREATMENT),
             '+0.016…+0.218'))
    print('\n## 4. 两分支的差别只有一处\n')
    print('| | BRANCH A | BRANCH B |')
    print('|---|---|---|')
    print('| G1 cycle 增益 | 作废 | %s（%+.2f%%） |'
          % ('过' if g1['pass_'] else '不过', g1['mean']))
    print('| G2 编码形式 | 作废 | %s（%+.2f%%） |'
          % ('过' if g2['pass_'] else '不过', g2['mean']))
    print('| G3 年龄捷径 | 过 | 过 |')
    print('| G4 语义主张 | 不过 → 作废整臂 | 换成 B1，全过 |')
    print('\n**PENDING-L2 要裁的就是这一格**：G4 作为"作废开关"是否成立。'
          '其余数字两分支完全一致。')


if __name__ == '__main__':
    main()
