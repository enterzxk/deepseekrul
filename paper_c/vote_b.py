"""B 段（+CADF）每一票的判定：engine-level paired bootstrap + 3/3 同号 + 效应量。

为什么是独立脚本、不并进 `gate1.py`：预注册 §9 把 `gate1.py` 的适用范围写死为
「A 段 = 协议保真硬检查 + 漂移测量」，它判的是「本地重跑是否复现锚点」。B 段判的是
「+CADF 是否比原版更好」，输入、零假设、记录格式都不同，混进一个脚本会让 A 段记录
带上 B 段的判定语义。所以本文件是 §4.1 / §4.4 的独立实现。

判据来源 `docs/P2_CROSS_ARCH_PREREG.md` §4.1 / §4.4，v7 的四条改动：
  1. 删掉 3 点 t(2) 配对 CI（n=3 时没有功效）；
  2. 保留 3/3 同号，但只当**方向一致性过滤**：n=3 时双侧符号检验的最小 p 是
     2x(1/2)^3 = 0.25，它本身不是统计证据，也不能被 bootstrap 替代；
  3. 加 engine-level paired bootstrap（10000 次，95% CI 不含 0）——它测的是
     「提升是不是靠少数几台发动机撑起来的」，与 seed 一致性是两个问题；
  4. 效应量门槛统一 0.45（v6.5.10 的 0.40 作废）；`floor_b` 不再当门，只作敏感度谱报告。

一个 (骨干, 子集) 单元过线需三条同时满足：
  S1 3/3 同号   —— 三个 seed 的 Δ = y_V0 − y_V1 全 > 0（正 = 模块更好）
  S2 效应量     —— dbar ≥ EFFECT_MIN
  S3 bootstrap  —— 按发动机重采样 10000 次，95% CI 不含 0
骨干算一票当且仅当 FD001 与 FD004 都过线。

两档判定（Full 优先，SC 永远不能盖过 Full）：
  Full: k ≥ K_MIN 且这些票在两个子集上同时过线；
  SC  : k ≥ K_MIN 且只在 FD001 过线、FD004 无害（dbar_FD004 ≥ −HARMLESS_TOL）。
        SC 的作用域写死为「单工况前端」，记录里带 claim_scope，论文不得越界。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# §4.1 的网格是「5 个 SOTA × 2 子集 × 3 seed × 2 变体 = 60 格」，B_min 不在其中：
# 它的比较在 5-fold 协议内做（p1_v12.json），没有 k0p 格。所以默认只判 5 个 SOTA，
# B_min 走独立输入、单列；k_SOTA 与 k_all 必须分开报。
BACKBONES = ('aidgn', 'dual_mixer', 'fcstgnn', 'ffve', 'two_phase')
SUBSETS = ('FD001', 'FD004')
DEFAULT_SEEDS = (101, 202, 303)

EFFECT_MIN = 0.45        # v7: 效应量门槛（v6.5.10 的 0.40 作废）
K_MIN = 4                # §4.1 第 1 条：6 个骨干里 ≥4
WIN_RATE_MIN = 0.60      # §4.1 第 2 条：配对胜率
HARMLESS_TOL = 0.10      # SC 档「多工况无害」的容差（v7 需确认数值）
EOL_TOL = -0.05          # §4.1 第 3 条：Δ_EOL_MAE ≥ −0.05
BOOT_N = 10000
# MDE at 80% power, two-sided 5%: (1.96 + 0.84) x SE.
MDE_Z = 2.80
# §4.4.1 的符号约定：正 = 模块更好。zxk 报的 CADF - V0 是反号。
DELTA_CONVENTION = 'delta = V0 - V1 (positive = module better, prereg 4.4.1)'
BOOT_SEED = 20261123
ALPHA = 0.05

EXIT_OK, EXIT_FAIL, EXIT_ABORT = 0, 1, 2


try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

def die(msg):
    print(msg, file=sys.stderr)
    raise SystemExit(EXIT_ABORT)


def _read_predictions(path: Path):
    """unit -> per-engine RMSE, from the frozen CSV columns (unit, y_true, y_pred)."""
    if not path.exists():
        die(f'{path}: missing -- a missing cell is a FAIL, never interpolated (§4.1 第 4 条)')
    per = {}
    with path.open(newline='', encoding='utf-8-sig') as fh:
        rd = csv.DictReader(fh)
        need = {'unit', 'y_true', 'y_pred'}
        if not need.issubset(set(rd.fieldnames or [])):
            die(f'{path}: CSV columns {rd.fieldnames} lack {sorted(need - set(rd.fieldnames or []))} '
                f'-- this is not the frozen prediction format')
        for row in rd:
            u = int(float(row['unit']))
            e = float(row['y_true']) - float(row['y_pred'])
            per.setdefault(u, []).append(e * e)
    return {u: float(np.sqrt(np.mean(v))) for u, v in per.items()}


def load_arm(root: Path, backbone: str, subset: str, seed: int):
    d = root / backbone / subset / f's{seed}'
    return _read_predictions(d / 'validation_predictions.csv')


def paired_engine_bootstrap(rmse0, rmse1, n_boot=BOOT_N, seed=BOOT_SEED, alpha=ALPHA):
    """Resample ENGINES with replacement; return (delta_mean, lo, hi, win_rate).

    Pairing is by engine: the same resampled engine index is used for both arms, so
    engine-level difficulty cancels and the CI is about heterogeneity across engines.
    """
    units = sorted(set(rmse0) & set(rmse1))
    if len(units) < 2:
        die(f'paired bootstrap: only {len(units)} engine(s) in common -- cannot resample')
    a = np.array([rmse0[u] for u in units])
    b = np.array([rmse1[u] for u in units])
    d = a - b                                   # per-engine delta, positive = module better
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(units), size=(n_boot, len(units)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(d.mean()), float(lo), float(hi), float((d > 0).mean()),
            float(means.std(ddof=1)))


def judge_unit(root0: Path, root1: Path, backbone: str, subset: str, seeds):
    """The three conditions for one (backbone, subset)."""
    per_seed = []
    rmse0_avg, rmse1_avg = {}, {}
    for s in seeds:
        r0 = load_arm(root0, backbone, subset, s)
        r1 = load_arm(root1, backbone, subset, s)
        if set(r0) != set(r1):
            die(f'{backbone}/{subset}/s{s}: engine sets differ between arms '
                f'({len(set(r0) ^ set(r1))} symmetric difference) -- not a paired cell')
        d_s = float(np.mean([r0[u] for u in sorted(r0)])) - float(np.mean([r1[u] for u in sorted(r1)]))
        per_seed.append(dict(seed=s, delta=d_s))
        for u in r0:
            rmse0_avg.setdefault(u, []).append(r0[u])
            rmse1_avg.setdefault(u, []).append(r1[u])
    # seed-averaged per-engine RMSE: the bootstrap then asks about ENGINE heterogeneity,
    # which is the question the t(2) CI could not answer.
    a0 = {u: float(np.mean(v)) for u, v in rmse0_avg.items()}
    a1 = {u: float(np.mean(v)) for u, v in rmse1_avg.items()}
    dbar, lo, hi, win, se = paired_engine_bootstrap(a0, a1)
    same_sign = all(p['delta'] > 0 for p in per_seed)
    s4_win = bool(win >= WIN_RATE_MIN)
    mde = MDE_Z * se
    if lo > 0.0:
        s3_branch = 'consistent'
    elif hi < 0.0:
        # CI 整段在 0 以下：确定更差。它是**可评估**的（我们学到了东西），只是不过线；
        # 不能归到 underpowered，否则「确定有害」会被算进未定票。
        s3_branch = 'determinately_worse'
    else:
        # 'MDE <= EFFECT_MIN and CI crosses 0' is arithmetically unreachable while
        # dbar >= EFFECT_MIN (SE <= 0.161 => half-width <= 0.315 => lo >= 0.135),
        # so there is no third branch to implement.
        s3_branch = 'underpowered_but_counted'
    # 经理的护栏：underpowered 既不进分子也不进分母，所以 s3_ok 只认 consistent
    s3_ok = (s3_branch == 'consistent')
    ok = bool(same_sign and dbar >= EFFECT_MIN and s3_ok and s4_win)
    return dict(backbone=backbone, subset=subset, per_seed=per_seed, delta_mean=dbar,
                boot_ci=[lo, hi], boot_se=se, mde=mde, n_engines=len(a0),
                engine_win_rate=win,
                s1_same_sign=same_sign, s2_effect=bool(dbar >= EFFECT_MIN),
                s3_branch=s3_branch, s3_ok=s3_ok, s4_win_rate=s4_win,
                underpowered=bool(s3_branch == 'underpowered_but_counted'),
                evaluable=bool(s3_branch != 'underpowered_but_counted'),
                passes=ok, within_noise=not ok)


def judge(root0: Path, root1: Path, seeds, backbones=BACKBONES, subsets=SUBSETS,
          eol=None, harmless_tol=HARMLESS_TOL, k_min=K_MIN, allow_missing_eol=False,
          deviant=()):
    units, per_backbone = {}, {}
    for b in backbones:
        for u in subsets:
            units[(b, u)] = judge_unit(root0, root1, b, u, seeds)
        no_damage = all(units[(b, u)]['delta_mean'] >= -harmless_tol for u in subsets)
        is_deviant = b in deviant
        both = (all(units[(b, u)]['passes'] for u in subsets) and no_damage
                and not is_deviant)
        first = units[(b, 'FD001')]['passes']
        harmless = all(units[(b, u)]['delta_mean'] >= -harmless_tol
                       for u in subsets if u != 'FD001')
        per_backbone[b] = dict(
            votes_full=both, votes_sc=bool(first and harmless and not is_deviant),
            no_damage=no_damage, protocol_deviant=is_deviant,
            fd001_pass=first,
            other_subsets=[dict(subset=u, passes=units[(b, u)]['passes'],
                                delta_mean=units[(b, u)]['delta_mean'],
                                harmless=bool(units[(b, u)]['delta_mean'] >= -harmless_tol))
                           for u in subsets if u != 'FD001'],
            why=('both subsets pass all three conditions' if both
                 else 'FD001 passes and the other subset is harmless' if (first and harmless)
                 else 'does not meet the three conditions on every subset'))
    k_full = sum(1 for b in backbones if per_backbone[b]['votes_full'])
    k_sc = sum(1 for b in backbones if per_backbone[b]['votes_sc'])
    # 可评估 = 该档要求的每个单元都不是 underpowered。underpowered 的票既不进分子
    # 也不进分母，所以「最多能拿到几票」= 可评估的骨干数。
    k_full_eval = sum(1 for b in backbones if b not in deviant
                      and all(units[(b, u)]['evaluable'] for u in subsets))
    k_sc_eval = sum(1 for b in backbones if b not in deviant
                    and units[(b, subsets[0])]['evaluable'])
    # P2 定 (a)：分档只由 5 个 SOTA 判，B_min 单列报告、不投票，所以没有 PENDING 档。
    # 一条式子（牢莫）：k_pass >= k_min 才是 PASS；k_pass + n_undetermined < k_min 才是
    # 确定的 FAIL；其余一律 INCONCLUSIVE。Full 先判，SC 只在 Full 不成立时才看。
    n_und_full = len(backbones) - k_full_eval
    n_und_sc = len(backbones) - k_sc_eval
    undetermined_reason = None
    if len(subsets) < 2:
        verdict, scope = 'incomplete', f'single subset ({subsets[0]})'
        undetermined_reason = '单子集调用无法产出分档判定：Full 要求两个子集同时过线'
    elif k_full >= k_min:
        verdict, scope = 'full', f'cross-condition ({" + ".join(subsets)})'
    elif k_sc >= k_min:
        verdict, scope = 'sc', f'single-condition front end ({subsets[0]} only)'
    elif k_full + n_und_full < k_min and k_sc + n_und_sc < k_min:
        verdict, scope = 'fail', 'none'
    else:
        verdict, scope = 'inconclusive', 'none'
        undetermined_reason = (
            f'未定票可能改变档位：Full 档 k_pass={k_full} + n_und={n_und_full} = '
            f'{k_full + n_und_full}；SC 档 {k_sc} + {n_und_sc} = {k_sc + n_und_sc}，'
            f'门槛 {k_min}。underpowered / protocol_deviant 的票既不进分子也不进分母，'
            f'本批功效或协议完整性不足以判定')
    if not eol and not allow_missing_eol:
        die('§4.1 第 3 条（EOL）是 PASS 的必要条件，未提供 --eol-json 时不出判定；'
            '若本批确实不评估 EOL，显式加 --allow-missing-eol（记录会标未评估）')
    eol_block = dict(evaluated=False, note='§4.1 第 3 条未评估：显式 --allow-missing-eol')
    if eol:
        eol_block = dict(eol)
        eol_block.setdefault('evaluated', True)
        if eol.get('delta_mean') is not None and eol['delta_mean'] < EOL_TOL:
            verdict, scope = 'fail', 'none'
            eol_block['veto'] = f'Δ_EOL_MAE {eol["delta_mean"]:+.4f} < {EOL_TOL}'
    return dict(
        verdict=verdict, claim_scope=scope, k_full=k_full, k_sc=k_sc, k_min=k_min,
        k_basis='5 SOTA only -- B_min 的 5-fold 输入未接入；§4.1 的 ≥4/6 需补 B_min 一票',
        k_all=None,
        effect_min=EFFECT_MIN, harmless_tol=harmless_tol, win_rate_min=WIN_RATE_MIN,
        seeds=list(seeds), boot=dict(n=BOOT_N, seed=BOOT_SEED, alpha=ALPHA,
                                     unit='engine', pairing='same engine index, both arms',
                                     per_seed_averaged=True),
        eol=eol_block,
        full_pass=bool(k_full >= k_min), sc_pass=bool(k_sc >= k_min),
        k_full_evaluable=k_full_eval, k_sc_evaluable=k_sc_eval,
        n_undetermined=dict(full=n_und_full, sc=n_und_sc),
        undetermined_reason=undetermined_reason,
        deviant=sorted(deviant),
        delta_convention=DELTA_CONVENTION,
        underpowered=[b for b in backbones
                      if any(units[(b, u)]['underpowered'] for u in subsets)],
        mde_rule='condition 3 (v7): CI lo > 0 -> consistent; CI crosses 0 and '
                 'MDE <= 0.45 -> downgraded, not counted; CI crosses 0 and '
                 'MDE > 0.45 -> counted, flagged underpowered (goes to limitations). '
                 'MDE = 2.80 x bootstrap SE (80% power, two-sided 5%)',
        mde_degenerate_note='在 Δ̄ >= 0.45 下「downgraded」分支不可达：MDE <= 0.45 '
                            '=> SE <= 0.161 => CI 半宽 <= 0.315 => lo >= 0.135 > 0。'
                            '所以条件 3 只贴标签（consistent / underpowered），'
                            '不拦票；这不是实现选择，是代数结论',
        mde_risk='SE 由 bootstrap 重采样均值给出，包含真实的发动机间异质性；'
                 '「宽 CI = 功效不足」与「宽 CI = 效应本身因发动机而异」在当前数据下'
                 '不可区分，因此 underpowered 标记只能进限制节，不能当成效应异质的证据',
        s1_note=('3/3 同号是方向一致性过滤，不是统计证据：n=3 时双侧符号检验的最小 p = '
                 '2x(1/2)^3 = 0.25，单靠它假阳率 25%；论文里须逐个列出三个 delta 并注明 '
                 'n=3 时 seed 维度无功效'),
        per_backbone=per_backbone,
        units=[units[k] for k in sorted(units)],
        forbidden_claims=(['跨工况通用', '跨架构通用'] if verdict == 'sc' else []),
        note=('SC 档只允许写「单工况前端」，Full 档才是主结论；'
              'SC 不得被表述成 Full（本记录 verdict 字段是唯一判据）'),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description='Judge segment B votes (prereg §4.1/§4.4).')
    ap.add_argument('--v0', required=True, type=Path, help='原版臂根目录 <b>/<u>/s<seed>/')
    ap.add_argument('--v1', required=True, type=Path, help='+CADF 臂根目录')
    ap.add_argument('--seeds', default=','.join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument('--backbones', default=','.join(BACKBONES))
    ap.add_argument('--subsets', default=','.join(SUBSETS))
    ap.add_argument('--eol-json', type=Path, default=None)
    ap.add_argument('--harmless-tol', type=float, default=HARMLESS_TOL)
    ap.add_argument('--k-min', type=int, default=K_MIN)
    ap.add_argument('--deviant', default='',
                    help='协议偏差的骨干（逗号分隔）：既不进分子也不进分母')
    ap.add_argument('--allow-missing-eol', action='store_true',
                    help='§4.1 第 3 条未评估时仍出判定（记录会标未评估）')
    ap.add_argument('--record', type=Path, default=None)
    a = ap.parse_args(argv)
    seeds = tuple(int(x) for x in a.seeds.split(',') if x.strip())
    bb = tuple(x.strip() for x in a.backbones.split(',') if x.strip())
    ss = tuple(x.strip() for x in a.subsets.split(',') if x.strip())
    eol = json.loads(a.eol_json.read_text(encoding='utf-8-sig')) if a.eol_json else None
    dv = tuple(x.strip() for x in a.deviant.split(',') if x.strip())
    rec = judge(a.v0, a.v1, seeds, bb, ss, eol, a.harmless_tol, a.k_min,
                a.allow_missing_eol, dv)
    for b in bb:
        pb = rec['per_backbone'][b]
        u1 = rec['units'][[i for i, x in enumerate(rec['units'])
                           if x['backbone'] == b and x['subset'] == ss[0]][0]]
        print(f'{b:11} FD001 dbar={u1["delta_mean"]:+.4f} CI[{u1["boot_ci"][0]:+.4f},'
              f'{u1["boot_ci"][1]:+.4f}] win={u1["engine_win_rate"]:.1%} '
              f'S1={int(u1["s1_same_sign"])} S2={int(u1["s2_effect"])} S3={int(u1["s3_boot"])} '
              f'-> full_vote={int(pb["votes_full"])} sc_vote={int(pb["votes_sc"])}')
    print(f'k_full={rec["k_full"]}/{len(bb)}  k_sc={rec["k_sc"]}/{len(bb)}  '
          f'verdict={rec["verdict"].upper()}  scope={rec["claim_scope"]}')
    if rec['verdict'] in ('inconclusive', 'incomplete'):
        print(f'NO VERDICT ({rec["verdict"]}): {rec["undetermined_reason"]}',
              file=sys.stderr)
    if rec['verdict'] == 'sc':
        print('SCOPE: SC -- 论文只能写「单工况前端」，禁止「跨工况通用」/「跨架构通用」')
    if not rec['eol']['evaluated']:
        print('WARN: §4.1 第 3 条（EOL）未评估', file=sys.stderr)
    if a.record:
        a.record.parent.mkdir(parents=True, exist_ok=True)
        a.record.write_text(json.dumps(rec, indent=1), encoding='utf-8')
        print(f'wrote {a.record}')
    return EXIT_OK if rec['verdict'] in ('full', 'sc') else EXIT_FAIL


if __name__ == '__main__':
    raise SystemExit(main())
