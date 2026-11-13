"""M3-B mechanism diagnostics: does the HI carry sensor information, or only age?

Two tests, both on the validation split, both cheap (no training):

1. SAME-AGE DAMAGE RANKING. Within a cycle bucket, rank the alive engines by the
   model's damage D_t and compare that ordering with the true remaining-life
   ordering. A model that is only an age timer assigns the SAME damage to every
   engine in the bucket, so every pair is a tie and the accuracy is exactly 0.5.
   Anything above 0.5 is individual information that age alone cannot supply.

2. SENSOR PERMUTATION. Shuffle the time axis inside each 70-cycle sensor window
   (same permutation for all 14 channels, so cross-sensor covariance is kept)
   and leave the cycle channel untouched. A model that reads only age produces
   the identical prediction; a model that reads sensors degrades.

Neither number proves the mechanism is *useful* -- that is still the RMSE
comparison against B0. They only decide whether the HI is a cycle reparameterisation.

Runs on CPU by default: the a40 GPU is busy with the matrix, and both diagnostics
compare a model against itself, so the device does not have to match training.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from paper_b.hc import config_hc as C
from paper_b.hc.data_d import engine_batch
from paper_b.hc.train_d import (build_d, evaluate, fit_age_prior, forward_engine,
                                trainable)

ROOT = '/media/sangfor/vdb/user/zxk/deepseek-rul'
DATA = ROOT + '/data/CMAPSS'
INIT = ROOT + '/experiments/hc/tmp/wave1_init_A1_s101.pt'
SHARED = ROOT + '/experiments/hc/d'


def load_arm(arm, subset, dev, run_dir=None):
    sh = json.loads(open('%s/%s_shared.json' % (SHARED, subset)).read())
    net, _ = build_d(arm, subset, int(sh['K']), float(sh['l_mean']),
                     int(sh['head_seed']), dev, INIT)
    if arm in ('AGE', 'AAR'):
        ebt = engine_batch(subset, DATA, C.SPAN, C.VAL_CUTS, split='train')
        prior = fit_age_prior(ebt, float(C.RUL_CAP))
        net.register_buffer('age_prior', torch.from_numpy(prior).to(dev))
        if arm == 'AAR':
            from paper_b.hc.train_d import ResidualHead
            net.residual = ResidualHead(net.d_model).to(dev)
    if run_dir:
        ck = Path(run_dir) / 'ckpt' / 'best.pt'
        if not ck.exists():
            raise SystemExit('no checkpoint at %s' % ck)
        sd = torch.load(ck, map_location=dev)
        # strict=False used to be here, and it is the same silent fallback that
        # already cost this sprint three days of M4: a checkpoint missing the
        # module under test (cde/csr/dpc/efr/residual) loads cleanly, the probe
        # then measures a randomly initialised head, and the number looks
        # perfectly reasonable. Refuse instead -- a missing tensor is a `not_run`
        # with a reason, never a number.
        missing = sorted(k for k in net.state_dict() if k not in sd)
        if missing:
            raise SystemExit(
                'checkpoint %s lacks %d tensor(s) the arm builds: %s. Refusing to '
                'probe a partly random model; record this arm as not_run.'
                % (ck, len(missing), missing[:8]))
        net.load_state_dict(sd, strict=True)
    trainable(net, arm)
    net.eval()
    return net, sh


def damage_by_cycle(net, arm, eb, dev, K, subset):
    """[(engine_id, cycle, D_t, true_rul)] for every observed cycle.

    For an arm with no damage path (AGE, B0) the age timer is substituted for
    D_t. That is the reference point the ranking test needs: the timer assigns
    the same damage to every engine in a cycle bucket, so every pair is a tie
    and the accuracy is 0.5 by construction. Without it the control has no
    number to compare against.
    """
    rows = []
    l_mean = None
    if arm in ('AGE', 'B0'):
        l_mean = float(json.loads(open('%s/%s_shared.json' % (SHARED, subset)).read())['l_mean'])
    with torch.no_grad():
        for i in range(len(eb['lengths'])):
            n = int(eb['lengths'][i])
            fo = forward_engine(net, arm, eb, i, n, dev, K)
            r = fo['r'].cpu().numpy()
            # Use the model's own output ordering for EVERY arm: -pred. Using D
            # would compare different quantities across arms (B2 has a damage path,
            # B0 does not) and the question is what the model's prediction ranks.
            D = -fo['pred'].detach().cpu().numpy()
            eid = int(eb['engine_id'][i])
            for t in range(n):
                rows.append((eid, t + 1, float(D[t]), float(r[t])))
    return rows


def ranking_accuracy(rows, bucket=20, min_engines=2):
    """Pairwise accuracy of (D ordering == RUL ordering) inside cycle buckets."""
    by_bucket = {}
    for eid, cyc, D, r in rows:
        by_bucket.setdefault((cyc - 1) // bucket, []).append((eid, D, r))
    agree = disagree = ties = 0
    n_buckets = 0
    for b, items in sorted(by_bucket.items()):
        if len({e for e, _, _ in items}) < min_engines:
            continue
        n_buckets += 1
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                _, da, ra = items[i]
                _, db, rb = items[j]
                if abs(ra - rb) < 1e-9 or abs(da - db) < 1e-12:
                    ties += 1
                    continue
                if (da - db) * (ra - rb) < 0:
                    agree += 1
                else:
                    disagree += 1
    n = agree + disagree
    return dict(accuracy=(agree / n if n else None), agree=agree,
                disagree=disagree, ties=ties, n_buckets=n_buckets,
                bucket_cycles=bucket)


def permute_sensors(eb, seed):
    """Time-permute every sensor window; the cycle channel is separate."""
    rng = np.random.default_rng(seed)
    x = np.array(eb['x'], copy=True)
    span = x.shape[2]
    for i in range(x.shape[0]):
        perm = rng.permutation(span)
        x[i] = x[i][:, perm, :]
    out = dict(eb)
    out['x'] = x
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='M3B')
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--run-dir', default=None)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--bucket', type=int, default=20)
    ap.add_argument('--perm-seeds', type=int, nargs='+', default=[0, 1, 2])
    # Paths were hard-coded to the a40 tree, so the probe could not run where the
    # checkpoints now live. Overrides keep the old defaults and make the probe
    # host-independent.
    ap.add_argument('--root', default=None, help='repo root (default: a40 path)')
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--init-from', default=None)
    ap.add_argument('--shared-dir', default=None)
    ap.add_argument('--out-dir', default=None, help='where the mech_*.json record goes')
    a = ap.parse_args()
    global ROOT, DATA, INIT, SHARED
    if a.root:
        ROOT = a.root
    if a.data_dir:
        DATA = a.data_dir
    if a.init_from:
        INIT = a.init_from
    if a.shared_dir:
        SHARED = a.shared_dir
    dev = torch.device(a.device)

    net, sh = load_arm(a.arm, a.subset, dev, a.run_dir)
    K, cap = int(sh['K']), float(C.RUL_CAP)
    eb = engine_batch(a.subset, DATA, C.SPAN, C.VAL_CUTS, split='val')

    rows = damage_by_cycle(net, a.arm, eb, dev, K, a.subset)
    # exact cycle is the primary test: at the SAME cycle the age timer gives every
    # engine the same damage, so accuracy 0.5 by construction. A wide bucket leaks
    # cycle differences into the ordering and inflates the timer's score (measured
    # 0.5688 at bucket=20), so it is reported only as a secondary reading.
    rank_exact = ranking_accuracy(rows, bucket=1)
    rank_wide = ranking_accuracy(rows, bucket=a.bucket)
    base = evaluate(net, a.arm, eb, dev, K, cap)

    perm_rmse = []
    for s in a.perm_seeds:
        ebp = permute_sensors(eb, s)
        ev = evaluate(net, a.arm, ebp, dev, K, cap)
        perm_rmse.append(ev['val_engine_mean_rmse'])

    rec = dict(arm=a.arm, subset=a.subset, device=str(dev),
               run_dir=a.run_dir, K=K,
               ranking_exact_cycle=rank_exact,
               ranking_wide_bucket=rank_wide,
               val_engine_mean_rmse=base['val_engine_mean_rmse'],
               permuted_val_engine_mean_rmse=perm_rmse,
               permuted_mean=float(np.mean(perm_rmse)),
               degradation=float(np.mean(perm_rmse)) - base['val_engine_mean_rmse'],
               degradation_pct=100.0 * (np.mean(perm_rmse) / base['val_engine_mean_rmse'] - 1.0))
    print(json.dumps(rec, indent=1))
    # --out-dir existed but was never used: the path was rebuilt from ROOT, so a
    # caller that passed --root <run root> got
    # <root>/experiments/hc/d/mech_*.json -- a directory that does not exist, and
    # write_text raised FileNotFoundError on all 13 arms. The driver then reported
    # it as a load error because it only kept the last stderr line.
    outdir = Path(a.out_dir) if a.out_dir else (Path(ROOT) / 'experiments/hc/d')
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / ('mech_%s_%s.json' % (a.arm, a.subset))
    out.write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print('record -> %s' % out)


if __name__ == '__main__':
    main()
