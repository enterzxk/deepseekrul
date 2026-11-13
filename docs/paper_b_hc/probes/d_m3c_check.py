"""M3-C (same-age ranking) definition check.

M3C is M3B's age-anchored increment plus one loss term:

    L_rank = relu(margin - (D_B - D_A))     r_A > r_B, same cycle t

The claim it has to support is narrow and checkable: the anchored increment is
exactly lambda0 for every engine at step 0, so every same-age pair is tied and
the damage path carries no sensor information at all. If the ranking term is
real, it starts at exactly ``margin`` and its gradient is non-zero; if it is
inert, the arm is M3B under another name.

Part 1 (always) runs on synthetic tensors: direction, margin, the RUL gap filter,
and symmetry. No data, no checkpoint.
Part 2 (--build) builds the real arm from the shared record and checks the
statements that make this a mechanism test rather than a new pipeline.

Usage
  python docs/paper_b_hc/probes/d_m3c_check.py
  python docs/paper_b_hc/probes/d_m3c_check.py --build \
      --data-dir <CMAPSS> --init-from <wave1_init_A1_s101.pt> --root <d80>
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_b.hc import train_d as T  # noqa: E402

RESULTS = []


def check(tag, ok, msg):
    RESULTS.append(bool(ok))
    print('%s %s %s' % ('PASS' if ok else 'FAIL', tag, msg))
    return ok


def fo(D, r):
    return dict(D=torch.tensor(D, dtype=torch.float32),
                r=torch.tensor(r, dtype=torch.float32))


def part1(margin=0.05, gap=10.0):
    print('--- part 1: loss semantics on synthetic tensors ---')
    # T1: the ordering already holds by more than the margin -> no penalty
    a = fo([0.10, 0.20, 0.30], [100.0, 80.0, 60.0])   # healthier, lower damage
    b = fo([0.30, 0.40, 0.50], [50.0, 30.0, 10.0])    # closer to failure, higher
    s, n = T.rank_pair_loss([a, b], margin, gap)
    check('T1 satisfied ordering is free', s is not None and n == 3
          and float(s) == 0.0, 'sum=%.6f points=%d' % (float(s), n))

    # T2: a tie at the same age costs exactly the margin per compared point
    a = fo([0.20, 0.20], [100.0, 80.0])
    b = fo([0.20, 0.20], [50.0, 30.0])
    s, n = T.rank_pair_loss([a, b], margin, gap)
    check('T2 tie costs exactly margin', n == 2
          and abs(float(s) - 2 * margin) < 1e-6,
          'sum=%.6f expected=%.6f points=%d' % (float(s), 2 * margin, n))

    # T3: a partial violation costs the shortfall, not the whole margin
    a = fo([0.20], [100.0])
    b = fo([0.23], [50.0])
    s, n = T.rank_pair_loss([a, b], margin, gap)
    check('T3 shortfall is linear', n == 1 and abs(float(s) - (margin - 0.03)) < 1e-6,
          'sum=%.6f expected=%.6f' % (float(s), margin - 0.03))

    # T4: the RUL gap filter must drop pairs the labels cannot separate
    a = fo([0.20], [100.0])
    b = fo([0.20], [95.0])          # 5 cycles apart, gap is 10
    s, n = T.rank_pair_loss([a, b], margin, gap)
    check('T4 gap filter drops inseparable pairs', s is None and n == 0,
          'sum=%s points=%d (gap_r=%.1f)' % (s, n, gap))

    # T5: orientation -- swapping the two engines must not change the loss
    a = fo([0.10, 0.20], [100.0, 80.0])
    b = fo([0.30, 0.40], [50.0, 30.0])
    s1, n1 = T.rank_pair_loss([a, b], margin, gap)
    s2, n2 = T.rank_pair_loss([b, a], margin, gap)
    check('T5 orientation symmetric', n1 == n2 and abs(float(s1) - float(s2)) < 1e-6,
          'forward=%.6f reverse=%.6f' % (float(s1), float(s2)))

    # T6: the M3C init state -- identical damage on every engine (the age timer)
    # but different true RUL. Every separable pair must cost exactly the margin.
    # (Identical r would give no pairs at all: the labels cannot order them.)
    fos = [fo([0.1, 0.2, 0.3, 0.4],
              [120.0 - 10 * k, 90.0 - 10 * k, 60.0 - 10 * k, 30.0 - 10 * k])
           for k in range(5)]
    s, n = T.rank_pair_loss(fos, margin, gap)
    check('T6 all-tied arm costs margin on every pair', n == 10 * 4
          and abs(float(s) - margin * 40) < 1e-5,
          'sum=%.6f expected=%.6f points=%d' % (float(s), margin * 40, n))


def part2(args):
    print('--- part 2: the real arm ---')
    dev = torch.device(args.device)
    sh = json.loads((Path(args.root) / ('%s_shared.json' % args.subset))
                    .read_text(encoding='utf-8'))
    l_mean = float(sh['l_mean'])
    head_seed = int(sh['head_seed'])

    check('B1 registered', 'M3C' in T.ALL_ARMS and 'M3C' in T.RANK_ARMS,
          'ALL_ARMS=%s RANK_ARMS=%s' % (list(T.ALL_ARMS), list(T.RANK_ARMS)))

    net_b, _ = T.build_d('M3B', args.subset, args.K, l_mean, head_seed, dev,
                         args.init_from, seed=101)
    net_c, _ = T.build_d('M3C', args.subset, args.K, l_mean, head_seed, dev,
                         args.init_from, seed=101)
    pb = sum(p.numel() for p in net_b.parameters())
    pc = sum(p.numel() for p in net_c.parameters())
    check('B2 build_d accepts M3C', type(net_c).__name__ == 'DSPT_HC_Structured'
          and pc == pb, '%s params=%d (M3B %d)' % (type(net_c).__name__, pc, pb))
    check('B3 same head init as M3B',
          net_c.head.fingerprint() == net_b.head.fingerprint(),
          'M3C=%s M3B=%s' % (net_c.head.fingerprint(), net_b.head.fingerprint()))

    # Use the same span/cuts the training run uses, so the engine set and the
    # trajectories are the ones the arm will actually see.
    ebt = T.engine_batch(args.subset, args.data_dir, T.C.SPAN, T.C.VAL_CUTS,
                         split='train')
    ebv = T.engine_batch(args.subset, args.data_dir, T.C.SPAN, T.C.VAL_CUTS,
                         split='val')
    tr_ids = set(int(x) for x in ebt['engine_id'])
    va_ids = set(int(x) for x in ebv['engine_id'])
    check('B4 pairs can only come from train engines', tr_ids.isdisjoint(va_ids),
          'train=%d val=%d overlap=%d' % (len(tr_ids), len(va_ids),
                                          len(tr_ids & va_ids)))

    net_c.eval()
    idx = [int(round(k * (len(ebt['lengths']) - 1) / 7)) for k in range(8)]
    fos, Ds = [], []
    with torch.no_grad():
        for i in idx:
            n = int(ebt['lengths'][i])
            f = T.forward_engine(net_c, 'M3C', ebt, i, n, dev, args.K)
            fos.append(dict(D=f['D'], r=f['r']))
            Ds.append(f['D'])
    m = min(int(d.numel()) for d in Ds)
    spread = max(float((d[:m] - Ds[0][:m]).abs().max()) for d in Ds)
    check('B5 step 0 is a pure age timer (all engines tied)', spread == 0.0,
          'max|D_i - D_0| over %d cycles = %.3e' % (m, spread))

    s, cnt = T.rank_pair_loss(fos, args.rank_margin, args.rank_gap)
    rank0 = float(s) / float(cnt) if s is not None else 0.0
    check('B6 ranking loss at init equals the margin', cnt > 0
          and abs(rank0 - args.rank_margin) < 1e-6,
          'rank0=%.9f margin=%.6f points=%d' % (rank0, args.rank_margin, cnt))

    net_c.train()
    opt = torch.optim.AdamW([p for p in net_c.parameters() if p.requires_grad],
                            lr=args.lr)

    def _rank_backward():
        """One ranking backward on the reference engines (no optimiser step)."""
        opt.zero_grad(set_to_none=True)
        fos = []
        for i in idx:
            n = int(ebt['lengths'][i])
            f = T.forward_engine(net_c, 'M3C', ebt, i, n, dev, args.K)
            fos.append(dict(D=f['D'], r=f['r']))
        s, c = T.rank_pair_loss(fos, args.rank_margin, args.rank_gap)
        s.backward()
        return s, c

    def _grad(prefix):
        return {k: float(p.grad.norm()) for k, p in net_c.named_parameters()
                if p.grad is not None and k.startswith(prefix)}

    # Step 0. The output layer is zero-initialised (hi_d.py:76-79), so u = 0
    # exactly and the clamp is NOT active: d(clamp)/du = 1 at 0. Hence the output
    # layer's gradient is non-zero. If the clamp were saturated instead,
    # d(clamp)/du = 0 and this gradient would be zero as well -- so this single
    # number separates the two explanations.
    _rank_backward()
    g_out = _grad('head.anchor.2')
    g_in = _grad('head.anchor.0')
    check('B7a gradient enters the ranking path at step 0', len(g_out) > 0
          and min(g_out.values()) > 0.0,
          'anchor.2 grads=%s' % {k: '%.3e' % v for k, v in g_out.items()})
    # The first layer is exactly zero at step 0 *by construction*: d(u)/d(W1) is
    # proportional to W2, which is zero. Asserted rather than glossed over, so a
    # zero here is never later read as a broken graph.
    check('B7b first layer is zero at step 0 by design (W2 = 0)', len(g_in) > 0
          and max(g_in.values()) == 0.0,
          'anchor.0 grads=%s (expected 0: d u / d W1 ~ W2 = 0)' % {
              k: '%.3e' % v for k, v in g_in.items()})

    # After one step W2 != 0, so the first layer must start receiving gradient.
    # That is the property the arm needs: from step 1 the whole head moves.
    opt.step()
    _rank_backward()
    g_in2 = _grad('head.anchor.0')
    check('B7c first layer receives gradient from step 1', len(g_in2) > 0
          and min(g_in2.values()) > 0.0,
          'anchor.0 grads after one step=%s' % {
              k: '%.3e' % v for k, v in g_in2.items()})

    # And the arm actually leaves the timer: D must stop being engine-independent.
    for _ in range(max(0, args.steps - 1)):
        opt.step()
        _rank_backward()
    net_c.eval()
    with torch.no_grad():
        Ds = []
        for i in idx:
            n = int(ebt['lengths'][i])
            Ds.append(T.forward_engine(net_c, 'M3C', ebt, i, n, dev, args.K)['D'])
    m2 = min(int(d.numel()) for d in Ds)
    spread2 = max(float((d[:m2] - Ds[0][:m2]).abs().max()) for d in Ds)
    check('B7d ranking steps move D off the age timer', spread2 > 1e-6,
          'max|D_i - D_0| after %d steps = %.3e (was 0 at init)'
          % (args.steps, spread2))

    rid = subprocess.run([sys.executable, '-m', 'paper_b.hc.train_d', 'id',
                          '--arm', 'M3C', '--subset', args.subset, '--seed', '101',
                          '--lam-rank', '0.05', '--rank-gap', '10',
                          '--rank-margin', '0.05'],
                         cwd=str(ROOT), capture_output=True, text=True)
    check('B8 argparse accepts M3C and the new flags', rid.returncode == 0,
          'id -> %s %s' % (rid.stdout.strip(), rid.stderr.strip()[-120:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true',
                    help='also build the real arm (needs torch, data, checkpoint)')
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--K', type=int, default=600)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--rank-margin', type=float, default=0.05)
    ap.add_argument('--rank-gap', type=float, default=10.0)
    ap.add_argument('--lr', type=float, default=1e-3,
                    help='B7c/B7d: optimiser step used to leave the zero-init point')
    ap.add_argument('--steps', type=int, default=3,
                    help='B7d: ranking-only steps before checking D left the timer')
    ap.add_argument('--data-dir',
                    default='/media/sangfor/vdb/user/zxk/deepseek-rul/data/CMAPSS')
    ap.add_argument('--init-from',
                    default='/media/sangfor/vdb/user/zxk/deepseek-rul/experiments/hc/'
                            'tmp/wave1_init_A1_s101.pt')
    ap.add_argument('--root',
                    default='/media/sangfor/vdb/user/zxk/deepseek-rul/experiments/hc/d80')
    a = ap.parse_args()
    part1(a.rank_margin, a.rank_gap)
    if a.build:
        part2(a)
    ok = all(RESULTS)
    print('M3C CHECK %s (%d/%d)' % ('OK' if ok else 'NOT OK',
                                    sum(RESULTS), len(RESULTS)))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
