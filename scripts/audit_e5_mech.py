"""E5 mechanism checks H1 / H3 / H4, exactly as frozen in E5_PREREG.md §3.

H1  low-rank spectrum: rho = trace(U^T Chat U) / sum(lambda), where
    Chat = (1/D) sum_d C_d and C_d is the 14x14 covariance of the 14-dim
    sensor vector at feature dim d, over all (b,t) of the validation set.
    Pass: rho >= 0.60 AND rho > null 95th percentile (1000 random 4-dim
    subspaces). Additional clause: if the lr14 spectrum does not differ from
    the lr4 spectrum on eigenvalues 5..14 (beyond seed sd), H1 also fails.
H3  basis stability: max principal angle between the U of every seed pair.
    Pass: all three pairs < 30 deg AND below the random-subspace null (5 pct).
H4  condition characterisation: only FD002 / FD004 (single-condition subsets
    are N/A). G(F) pooled over time -> engine-grouped CV linear classifier for
    the operating-condition cluster; pass: balanced accuracy >= 0.50 and above
    the label-shuffled null (95 pct).

H2 is statistical and lives in scripts/audit_e5_arms.py.

Usage:
  python3 scripts/audit_e5_mech.py <runs_root> --data-dir /path/CMAPSS \
      --out /tmp/e5_mech.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from _cells import resolve                                   # noqa: E402
from paper_b.training.data import prepare_train              # noqa: E402
from paper_b.training.train import _build                    # noqa: E402

SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')
SEEDS = (101, 202, 303)
BASE = dict(
    model='dspt', span=70, epochs=40, patience=12, batch=512, lr=1e-3,
    weight_decay=1e-4, d_model=64, layers=2, dropout=0.2, mode='dual',
    ms_mode='full', sa_mode='basic', hi_source='hf', hi_mono='correct',
    hi_lambda=0.5, hi_mono_coef=0.1, use_cycle=True, use_pad_mask=True,
    input_mode='raw', aux_dual=True, instnorm=False, rul_cap=125, fp32=True,
    split_seed=17, fold=0, val_frac=0.2, n_cond=6,
    cuts=[0.4, 0.55, 0.7, 0.85, 0.95],
    legacy_rul=False, global_hi=False, legacy_ms_pad=False, legacy_ema=False,
)


def load_model(cell, subset, seed, lr_mode, dev):
    cfg = {**BASE, 'subset': subset, 'seed': seed, 'lr_mode': lr_mode}
    net = _build(cfg).to(dev)
    net.load_state_dict(torch.load(Path(cell) / 'best.pt', map_location='cpu',
                                   weights_only=True))
    net.eval()
    return net


def capture(net, va, dev, cfg):
    """Run the validation set once; return H at sensor_score, G(F), and the
    branch's relative perturbation ||lr(h)-h|| / ||h||.

    The last quantity is the direct definition of "is the branch awake": a
    zero-initialised last layer makes it exactly 0 at init, and a weight-norm
    proxy cannot tell a dormant branch from an active one (the first Linear in
    g is randomly initialised and its norm is ~4.6 from the start).
    """
    Hs, Gs = [], []
    lr_in, lr_out = [], []
    h_hook = net.sensor_score.register_forward_pre_hook(
        lambda m, i: Hs.append(i[0].detach().float().cpu()))
    g_hook = None
    lr_hook = None
    if getattr(net, 'lr', None) is not None and getattr(net.lr, 'mode', '') == 'lowrank':
        g_hook = net.lr.g.register_forward_hook(
            lambda m, i, o: Gs.append(o.detach().float().cpu()))
    if getattr(net, 'lr', None) is not None:
        def _lr(m, i, o):
            lr_in.append(i[0].detach().float().cpu())
            lr_out.append(o.detach().float().cpu())
        lr_hook = net.lr.register_forward_hook(_lr)
    x = torch.from_numpy(np.ascontiguousarray(va['x'])).to(dev)
    mk = torch.from_numpy(np.ascontiguousarray(va['mask'])).to(dev)
    bs = torch.from_numpy(np.ascontiguousarray(va['baseline'])).to(dev)
    cy = torch.from_numpy(np.ascontiguousarray(va['cycle'])).to(dev)
    cd = torch.from_numpy(np.ascontiguousarray(va['cond'])).to(dev)
    with torch.no_grad():
        for i in range(0, len(x), 256):
            sl = slice(i, i + 256)
            net(x[sl], mask=mk[sl], baseline=bs[sl], cycle=cy[sl], cond=cd[sl],
                mode=cfg['mode'], sensor_keep=None)
    h_hook.remove()
    if g_hook is not None:
        g_hook.remove()
    if lr_hook is not None:
        lr_hook.remove()
    H = torch.cat(Hs, 0) if Hs else None
    G = torch.cat(Gs, 0) if Gs else None
    rel = None
    if lr_in and lr_out:
        hi, ho = torch.cat(lr_in, 0).double(), torch.cat(lr_out, 0).double()
        rel = float((ho - hi).norm() / hi.norm().clamp_min(1e-12))
    return H, G, rel


def chat(H):
    """Chat = (1/D) sum_d C_d, C_d the 14x14 covariance over all (b,t).

    H is (B,L,N,D). For a fixed feature dim d the 14-dim sensor vector is
    X[:, :, d]; its covariance is (1/M) sum_m x_m x_m^T, i.e. einsum
    'mnd,mod->nod' divided by M, then averaged over d.
    """
    H = H.double()
    B, L, N, D = H.shape
    flat = H.reshape(-1, N, D)
    flat = flat - flat.mean(dim=0, keepdim=True)
    M = flat.shape[0]
    Cn = torch.einsum('mnd,mod->nod', flat, flat) / M     # (N,N,D)
    return Cn.mean(dim=2)                                  # (N,N)


def rho_of(C, U):
    U = U.double()
    lam = torch.linalg.eigvalsh(C).clamp_min(0)
    return float((U.T @ C @ U).trace() / lam.sum())


def random_rho(C, r, draws, gen):
    N = C.shape[0]
    vals = []
    for _ in range(draws):
        q, _ = torch.linalg.qr(torch.randn(N, r, generator=gen, dtype=torch.float64))
        vals.append(rho_of(C, q))
    return float(np.percentile(vals, 95)), float(np.mean(vals))


def principal_angle(Ua, Ub):
    s = torch.linalg.svdvals(Ua.double().T @ Ub.double())
    return float(torch.rad2deg(torch.arccos(s.clamp(-1, 1).min())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gen = torch.Generator().manual_seed(20261116)
    rng = np.random.default_rng(20261116)

    out = dict(h1={}, h3={}, h4={}, h4_note='FD001/FD003 single-condition -> N/A')
    for subset in SUBSETS:
        ds = prepare_train(subset, Path(a.data_dir), span=BASE['span'], split_seed=17,
                           fold=0, val_frac=0.2, rul_cap=BASE['rul_cap'],
                           n_cond=BASE['n_cond'], cuts=tuple(BASE['cuts']))
        va = ds['val']
        for lr_mode in ('lr4', 'lr14'):
            prefix = 'E5lr4' if lr_mode == 'lr4' else 'E5lr14'
            per_seed, Us = {}, {}
            for seed in SEEDS:
                cell = resolve(a.root, prefix, subset, seed, 17)
                net = load_model(cell, subset, seed, lr_mode, dev)
                H, G, rel = capture(net, va, dev, {**BASE, 'subset': subset, 'seed': seed})
                C = chat(H)
                U = net.lr.orthonormal_u().detach().float().cpu()
                Us[seed] = U
                lam = torch.linalg.eigvalsh(C.double()).clamp_min(0).flip(0)
                per_seed[seed] = dict(
                    rho=round(rho_of(C, U), 4),
                    best_r4=round(float(lam[:4].sum() / lam.sum()), 4),
                    branch_rel_delta=round(rel, 10),
                    eigenvalues=[round(float(v), 6) for v in lam])
                del net
                torch.cuda.empty_cache()
            null95, nullmean = random_rho(C, 4, a.draws, gen)
            rhos = [v['rho'] for v in per_seed.values()]
            out['h1'][f'{lr_mode}/{subset}'] = dict(
                rho_per_seed={str(k): v['rho'] for k, v in per_seed.items()},
                rho_mean=round(float(np.mean(rhos)), 4),
                rho_sd=round(float(np.std(rhos, ddof=1)), 4) if len(rhos) > 1 else None,
                best_r4={str(k): v['best_r4'] for k, v in per_seed.items()},
                null95=round(null95, 4), null_mean=round(nullmean, 4),
                pass_=(bool(np.mean(rhos) >= 0.60) and bool(np.mean(rhos) > null95)),
                eigenvalues={str(k): v['eigenvalues'] for k, v in per_seed.items()})
            # report-only: is the learned U just the top-4 PCA of H?
            if lr_mode == 'lr4':
                evals, evecs = torch.linalg.eigh(C.double())
                pca4 = evecs[:, -4:]                      # ascending -> top 4 last
                out['h1'][f'lr4/{subset}']['angle_to_pca4_deg'] = round(
                    principal_angle(Us[SEEDS[-1]], pca4), 2)
            if lr_mode == 'lr4':
                pairs = {}
                for i in range(len(SEEDS)):
                    for j in range(i + 1, len(SEEDS)):
                        pairs[f'{SEEDS[i]}-{SEEDS[j]}'] = round(
                            principal_angle(Us[SEEDS[i]], Us[SEEDS[j]]), 2)
                nullang = []
                for _ in range(a.draws):
                    q1, _ = torch.linalg.qr(torch.randn(14, 4, generator=gen, dtype=torch.float64))
                    q2, _ = torch.linalg.qr(torch.randn(14, 4, generator=gen, dtype=torch.float64))
                    nullang.append(principal_angle(q1, q2))
                out['h3'][subset] = dict(
                    pairwise_max_angle=pairs,
                    null5=round(float(np.percentile(nullang, 5)), 2),
                    null_mean=round(float(np.mean(nullang)), 2),
                    pass_=(all(v < 30.0 for v in pairs.values())
                           and all(v < np.percentile(nullang, 5) for v in pairs.values())))
            # also record the `none` arm's spectrum: it is the reference the
            # frozen threshold was calibrated against (E0 == none architecture)
            if lr_mode == 'lr14':
                cell = resolve(a.root, 'E5none', subset, SEEDS[0], 17)
                net = load_model(cell, subset, SEEDS[0], 'none', dev)
                H, _, _rel_none = capture(net, va, dev, {**BASE, 'subset': subset, 'seed': SEEDS[0]})
                Cn = chat(H)
                lamn = torch.linalg.eigvalsh(Cn.double()).clamp_min(0).flip(0)
                out['h1'][f'none/{subset}'] = dict(
                    best_r4=round(float(lamn[:4].sum() / lamn.sum()), 4),
                    null95=round(random_rho(Cn, 4, a.draws, gen)[0], 4),
                    eigenvalues=[round(float(v), 6) for v in lamn])
                del net
                torch.cuda.empty_cache()
            # H4 only for multi-condition subsets, and only for lr4
            if lr_mode == 'lr4' and subset in ('FD002', 'FD004'):
                out['h4'][subset] = run_h4(G, H, va, rng)

        # H1 additional clause (frozen): if lr14 does not differ from lr4 on
        # eigenvalues 5..14 beyond seed sd, the rank constraint did nothing and
        # H1 fails. Computed in float64 throughout: float32 accumulation noise
        # is ~1e-4, the same order as the effect this clause is looking for.
        a4 = out['h1'].get(f'lr4/{subset}')
        a14 = out['h1'].get(f'lr14/{subset}')
        if a4 and a14:
            e4 = np.array([a4['eigenvalues'][str(s)] for s in SEEDS])
            e14 = np.array([a14['eigenvalues'][str(s)] for s in SEEDS])
            t4, t14 = e4[:, 4:], e14[:, 4:]
            diff = float(np.abs(t4.mean(0) - t14.mean(0)).max())
            sd = float(np.sqrt((t4.std(0, ddof=1).mean() ** 2
                                + t14.std(0, ddof=1).mean() ** 2) / 2))
            out['h1'][f'clause5_14/{subset}'] = dict(
                max_abs_diff=round(diff, 8), pooled_seed_sd=round(sd, 8),
                distinguishable=bool(diff > sd),
                dtype='float64',
                note='H1 also fails when this is False (rank constraint inert)')
            print(f'{subset} clause5_14 diff={diff:.3e} sd={sd:.3e} '
                  f'distinguishable={diff > sd}', flush=True)

    txt = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(txt)
    print(txt)


def run_h4(G, H, va, rng):
    """Engine-grouped CV classifier for the operating-condition cluster.

    Report-only addition (does not change the frozen H4 criterion): the same
    classifier on the pooled H, so the G(F) accuracy can be read as an
    increment over what the representation already carries.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import balanced_accuracy_score
    except Exception as exc:                                # pragma: no cover
        return dict(error=f'sklearn unavailable: {exc!r}')
    y = np.asarray(va['cond']).astype(int)
    g = np.asarray(va['unit']).astype(int)

    def feats(T):
        return T.mean(dim=1).reshape(T.shape[0], -1).double().numpy()

    def cv_acc(X, yv):
        n_splits = min(5, len(np.unique(g)))
        accs = []
        for tr, te in GroupKFold(n_splits=n_splits).split(X, yv, g):
            if len(np.unique(yv[tr])) < 2:
                continue
            m = LogisticRegression(max_iter=2000, multi_class='multinomial')
            m.fit(X[tr], yv[tr])
            accs.append(balanced_accuracy_score(yv[te], m.predict(X[te])))
        return float(np.mean(accs)) if accs else float('nan')

    Xg = feats(G)
    obs = cv_acc(Xg, y)
    null = np.array([cv_acc(Xg, rng.permutation(y)) for _ in range(200)])
    res = dict(balanced_accuracy=round(obs, 4),
               null95=round(float(np.percentile(null, 95)), 4),
               null_mean=round(float(null.mean()), 4),
               chance=round(1.0 / len(np.unique(y)), 4),
               pass_=bool(obs >= 0.50 and obs > np.percentile(null, 95)),
               n_windows=int(len(y)), n_engines=int(len(np.unique(g))))
    # report-only: the same classifier on pooled H, and the increment
    Xh = feats(H)
    acc_h = cv_acc(Xh, y)
    res['H_pooled_balanced_accuracy'] = round(acc_h, 4)
    res['G_minus_H'] = round(obs - acc_h, 4)
    return res


if __name__ == '__main__':
    main()
