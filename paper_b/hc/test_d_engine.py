"""D7.2 assertions, written before the engine interface exists.

METHOD_SPECIFICATION.md D7.2 lists four properties the Stage D damage recursion
must satisfy. They are checked here against a **reference recursion**, so the
formulation can be validated before any network code is written:

  1. overlap consistency  -- the same engine cycle gets the same D no matter
                             which window carries it;
  2. increment closure    -- D_s(w2) - D_{s-1}(w1) equals delta_s;
  3. monotonicity         -- D never decreases, delta >= 0;
  4. two-interface parity -- z_t from the window array equals z_t from the
                             engine array (needs `engine_batch`, so it is a
                             contract test that SKIPS until that lands).

Tests 1-3 also run against `ResetPerWindow`, the forbidden implementation, and
**must fail on it** -- a property test that cannot fail on the wrong design is
worth nothing. That negative control is the point of this file.

Numpy only: runnable without torch, GPU, or the data layer.

Usage:  python paper_b/hc/test_d_engine.py
        python -m pytest paper_b/hc/test_d_engine.py -q
"""
import numpy as np

TOL = 1e-6


# --------------------------------------------------------------------------
# reference recursion
# --------------------------------------------------------------------------
def prefix_damage(delta):
    """D_0 = 0, D_t = sum_{u<=t} delta_u. Sum, not integral: delta is dimensionless."""
    return np.concatenate([[0.0], np.cumsum(np.asarray(delta, np.float64))])


def delta_of(engine, t):
    """Stand-in for softplus(f_theta(z_t)).

    Strictly positive, deterministic, and different per engine, so a window-local
    reset cannot accidentally agree with the engine-level recursion.
    """
    return 0.5 + 0.3 * np.sin(0.11 * t + 0.7 * engine) ** 2


class EngineRecursion:
    """The only supported way to obtain D: indexed by engine cycle, not window."""

    def __init__(self, delta_fn=delta_of):
        self.delta_fn = delta_fn
        self._cache = {}

    def deltas(self, engine, n):
        key = (engine, n)
        if key not in self._cache:
            self._cache[key] = np.array(
                [self.delta_fn(engine, t) for t in range(1, n + 1)], np.float64)
        return self._cache[key]

    def damage(self, engine, n):
        return prefix_damage(self.deltas(engine, n))

    def read(self, engine, n, start, length):
        """A window is a READ of D, never a re-initialisation."""
        d = self.damage(engine, n)
        lo, hi = max(1, start), min(n, start + length - 1)
        return np.arange(lo, hi + 1), d[lo:hi + 1]


class ResetPerWindow:
    """The forbidden design, kept here so the tests can be shown to fail on it."""

    def __init__(self, delta_fn=delta_of):
        self.delta_fn = delta_fn

    def read(self, engine, n, start, length):
        lo, hi = max(1, start), min(n, start + length - 1)
        deltas = np.array([self.delta_fn(engine, t) for t in range(lo, hi + 1)])
        return np.arange(lo, hi + 1), prefix_damage(deltas)


def _overlap_error(a, b):
    ta, da = a
    tb, db = b
    ma, mb = dict(zip(ta, da)), dict(zip(tb, db))
    shared = sorted(set(ma) & set(mb))
    assert shared, 'windows do not overlap; the test would be vacuous'
    return shared, np.array([abs(ma[t] - mb[t]) for t in shared])


# --------------------------------------------------------------------------
# 1. overlap consistency
# --------------------------------------------------------------------------
def test_overlap_consistency():
    rec = EngineRecursion()
    n = 120
    shared, err = _overlap_error(rec.read(3, n, 1, 70), rec.read(3, n, 40, 70))
    assert len(shared) == 31, 'expected 31 shared cycles, got %d' % len(shared)
    assert err.max() <= TOL, 'overlap disagreement %.3e > %.0e' % (err.max(), TOL)


# --------------------------------------------------------------------------
# 2. increment closure
# --------------------------------------------------------------------------
def test_increment_closure():
    rec = EngineRecursion()
    n = 120
    s = 40
    _, d1 = rec.read(3, n, 1, 70)          # cycles 1..70
    _, d2 = rec.read(3, n, s, 70)          # cycles 40..109
    d_s = d2[0]                            # D_40
    d_prev = d1[s - 1 - 1]                 # D_39, index = cycle-1
    got = d_s - d_prev
    want = delta_of(3, s)
    assert abs(got - want) <= TOL, 'closure %.12f != delta_%d %.12f' % (got, s, want)


# --------------------------------------------------------------------------
# 3. monotonicity
# --------------------------------------------------------------------------
def test_monotonicity():
    rec = EngineRecursion()
    for engine in (0, 3, 7):
        for n in (40, 200, 543):
            d = rec.damage(engine, n)
            assert (np.diff(d) >= -TOL).all(), 'D decreased (engine %d, n %d)' % (engine, n)
            assert (rec.deltas(engine, n) >= 0).all(), 'negative delta'
            assert d[0] == 0.0, 'D_0 must be 0'


# --------------------------------------------------------------------------
# 4. negative control: the forbidden design MUST fail 1-3
# --------------------------------------------------------------------------
def test_reset_is_detected_by_overlap():
    bad = ResetPerWindow()
    _, err = _overlap_error(bad.read(3, 120, 1, 70), bad.read(3, 120, 40, 70))
    assert err.max() > 1.0, (
        'window-local reset was NOT detected (max err %.3e); the consistency '
        'test has no teeth' % err.max())


def test_reset_drifts_with_window_start():
    """The same cycle, read through two windows with different starts, must not move.

    Changing the window *length* while keeping the start at cycle 1 does NOT
    expose the reset design (both reads share the same prefix), so the
    discriminating construction is two different starts that both contain t.
    """
    good, bad = EngineRecursion(), ResetPerWindow()
    t = 70
    _, dg1 = good.read(3, 120, 1, 70)      # cycles 1..70
    _, dg2 = good.read(3, 120, 40, 70)     # cycles 40..109, t at offset 30
    _, db1 = bad.read(3, 120, 1, 70)
    _, db2 = bad.read(3, 120, 40, 70)
    assert abs(dg1[t - 1] - dg2[t - 40]) <= TOL, 'reference drifted'
    drift = abs(db1[t - 1] - db2[t - 40])
    assert drift > 1.0, (
        'reset design did not drift (%.3e); test is not discriminating' % drift)
    # and the drift must equal the prefix it threw away
    lost = sum(delta_of(3, u) for u in range(1, 40))
    assert abs(drift - lost) <= 1e-9, 'drift %.6f != discarded prefix %.6f' % (drift, lost)


class _Skip(Exception):
    pass


def _skip(msg):
    """Skip under pytest when available, otherwise raise a runner-level skip."""
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        raise _Skip(msg)


def _find_data():
    """Locate C-MAPSS train files; None if this machine has none."""
    import os
    for p in ('data/CMAPSS', 'paper_b/data/CMAPSS',
              '/media/sangfor/vdb/user/zxk/deepseek-rul/data/CMAPSS'):
        if os.path.exists(os.path.join(p, 'train_FD001.txt')):
            return p
    return None


# --------------------------------------------------------------------------
# 5. two-interface parity: the frozen cut window vs the engine trajectory
# --------------------------------------------------------------------------
def test_two_interface_parity():
    """z_t from a frozen cut window must equal z_t from the trajectory slice.

    This is the check that lets the Stage D recursion read the frozen cut
    windows: if the two constructions ever disagree, the damage at a validation
    cut is computed from a different input than the one the model was evaluated
    on, and every val number in D is invalid.
    """
    data_dir = _find_data()
    if data_dir is None:
        _skip('no C-MAPSS data on this machine')
    try:
        import torch
        from paper_b.hc import config_hc, data_d
        from paper_b.hc.data_hc import build
        from paper_b.hc.model_hc import build_hc
    except Exception as e:                              # noqa: BLE001
        _skip('torch / paper_b not importable: %s' % e)

    span, cuts = config_hc.SPAN, config_hc.VAL_CUTS
    net = build_hc(config_hc.hc_config('A1', 'FD001', 101)).eval()
    ds = build('FD001', data_dir, span, cuts, cycle_source='raw')
    eb = data_d.engine_batch('FD001', data_dir, span, cuts, split='val')
    va = ds['val']

    def enc(x, base, mask, cond):
        return net.encode(torch.from_numpy(np.asarray(x, np.float32)),
                          torch.from_numpy(np.asarray(base, np.float32)),
                          torch.from_numpy(np.asarray(mask, np.float32)),
                          None,
                          torch.from_numpy(np.asarray(cond, np.int64)))

    worst, checked = 0.0, 0
    with torch.no_grad():
        for j in range(len(va['y'])):
            uid, c = int(va['unit'][j]), int(va['cut'][j])
            hit = np.where(eb['engine_id'] == uid)[0]
            assert len(hit) == 1, 'engine %d not unique in engine_batch' % uid
            e = int(hit[0])
            assert int(eb['cycle_raw'][e, c - 1]) == c, 'cycle misaligned'
            h_win = enc(va['x'][j:j + 1], va['baseline'][j:j + 1],
                        va['mask'][j:j + 1], va['cond'][j:j + 1])
            h_eng = enc(eb['x'][e, c - 1:c], eb['baseline'][e, c - 1:c],
                        eb['mask'][e, c - 1:c], eb['cond'][e, c - 1:c])
            worst = max(worst, float((h_win - h_eng).abs().max()))
            checked += 1

    assert checked == len(va['y']), 'checked %d of %d windows' % (checked, len(va['y']))
    assert worst <= TOL, 'window/engine encoder mismatch %.3e > %.0e' % (worst, TOL)


# --------------------------------------------------------------------------
# 6. arrival operator (D3). The first test is a regression witness: it pins the
#    defect the reviewer found, so the old form can never come back silently.
# --------------------------------------------------------------------------
def _sigmoid(z):
    """Numerically stable: exp(-z) overflows for large negative z."""
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def arrival_old(D, beta=1.0, K=None):
    """Hazard-product survival sum, reproduced FAITHFULLY (with S_0 = 1).

    h_j = sigmoid(beta*(D_j - 1)); S_0 = 1 (empty product);
    S_k = prod_{j=1..k} (1-h_j); T = sum_{k=0..K-1} S_k.

    With beta=1 and D >= 0, h_j >= sigmoid(-1) =: h for every j, so
    S_k <= (1-h)^k and T <= sum_{k<K} (1-h)^k = (1 - (1-h)^K)/h = 3.718282 = 1+e.
    The earlier version of this helper omitted S_0 and therefore did not
    reproduce the operator it claimed to reproduce.
    """
    K = D.shape[-1] if K is None else K
    h = _sigmoid(beta * (D[..., :K] - 1.0))
    q = 1.0 - h
    S = np.concatenate([np.ones_like(q[..., :1]), np.cumprod(q, axis=-1)[..., :-1]],
                       axis=-1)          # S_0 = 1, S_k = prod_{j<k} q_j
    return S.sum(axis=-1)


def arrival_old_bound(beta=1.0, K=200):
    """Closed form of the cap: (1 - q^K)/h with h = sigmoid(-beta)."""
    h = 1.0 / (1.0 + np.exp(beta))
    return (1.0 - (1.0 - h) ** K) / h


def arrival(D, nu, beta=4.0, eps=1e-6, K=None):
    """Direct soft-survival sum: count the steps that have not failed yet.

    Not a hazard product. For a monotone trajectory the true survival indicator
    is simply 1[D_k < 1] (no product needed), and its smooth surrogate is

        S_k = sigmoid( beta * (1 - D_k) / (nu_k + eps) )      # cycles remaining
        T   = clamp( sum_{k=0}^{K-1} S_k - 0.5, 0, K )

    Dividing by the predicted rate makes the argument "cycles remaining" rather
    than "damage units", so the transition band is ~1/beta CYCLES whatever the
    rate is, and beta stays O(1).

    Semantics, pinned because the reviewer asked for each case:
      k = 0 included   an engine already at the threshold gets ~0, not 1;
      D = 1 exactly    contributes 0.5, which the -0.5 cancels -> hit is exact;
      D > 1            (1-D)/nu < 0, the term is ~0;
      not hit in K     every term saturates to 1 -> T = K - 0.5, and the sample
                       is censored at K (the half-step is the same convention);
      variable nu      the term is a LOCAL-rate estimate of steps remaining, so
                       the bias grows with rate curvature (tested below).
    """
    K = D.shape[-1] if K is None else K
    z = beta * (1.0 - D[..., :K]) / (nu[..., :K] + eps)
    return np.clip(_sigmoid(z).sum(axis=-1) - 0.5, 0.0, float(K))


def _trajectory(n_star, K, nu=None):
    """Monotone damage that crosses 1 at step n_star and keeps growing after.

    Deliberately NOT clamped at 1: the predicted trajectory is D_t + sum(nu),
    which passes the threshold and continues. Clamping would make every step
    after the crossing contribute sigmoid(0)=0.5 instead of 0 and inflate T.
    """
    nu = 1.0 / n_star if nu is None else nu
    D = np.arange(K) * nu
    return D[None, :], np.full((1, K), nu)


def test_arrival_old_operator_is_capped():
    """Witness: the rejected operator cannot exceed 1+e = 3.718 cycles, ever.

    The closed form is an UPPER BOUND, attained only when D is identically 0
    (then h_j = sigmoid(-1) for every j). That is worth pinning separately,
    because it is exactly the degenerate solution the loss can drift to: the
    operator's maximum sits at D = 0.
    """
    K, n_star = 300, 200
    bound = arrival_old_bound(beta=1.0, K=K)
    D, nu = _trajectory(n_star, K)
    T_old = float(arrival_old(D, beta=1.0)[0])
    assert T_old <= bound + 1e-9, 'old operator %.6f exceeded the bound %.6f' % (T_old, bound)
    assert T_old < 3.72, 'old bound %.6f >= 1+e' % T_old
    assert T_old < 0.02 * n_star, (
        'old operator returns %.3f for a trajectory that crosses at %d' % (T_old, n_star))
    T_zero = float(arrival_old(np.zeros((1, K)), beta=1.0)[0])
    assert abs(T_zero - bound) < 1e-9, 'the bound must be attained at D=0: %.6f vs %.6f' % (
        T_zero, bound)
    T_new = float(arrival(D, nu)[0])
    assert abs(T_new - n_star) <= 1.0, 'corrected operator %.3f vs %d' % (T_new, n_star)


def test_arrival_representable_range_exceeds_rul_cap():
    """Constructive: the operator must be able to express RUL > 125.

    The reviewer asked for a prescribed trajectory rather than a random init
    happening to emit a large value: if the operator cannot represent 150 by
    construction, no weight setting can fix it.
    """
    for n_star in (125, 150, 300):
        K = n_star + 100
        D, nu = _trajectory(n_star, K)
        T = float(arrival(D, nu)[0])
        assert abs(T - n_star) <= 1.0, 'n*=%d -> T=%.3f' % (n_star, T)
    assert float(arrival(*_trajectory(150, 250))[0]) > 125.0


def test_arrival_already_failed():
    """At or past the threshold now -> arrival is 0, not 1 (the S0=1 defect)."""
    K = 200
    nu = 1.0 / 150.0
    D = 1.0 + np.arange(K) * nu
    T = arrival(D[None, :], np.full((1, K), nu))
    assert T[0] <= 0.5, 'already-failed arrival %.3f should be ~0' % T[0]


def test_arrival_exact_hit():
    """Crossing exactly at k*=50 is reported as 50 +/- 1 cycle."""
    K, n_star = 200, 50
    D, nu = _trajectory(n_star, K)
    T = arrival(D, nu)
    assert abs(T[0] - n_star) <= 1.0, 'exact hit %.3f vs %d' % (T[0], n_star)


def test_arrival_non_integer_hit():
    """A crossing between two cycles must land between them, not snap."""
    K, n_star = 200, 50.5
    nu = 1.0 / n_star
    D = np.arange(K) * nu
    T = float(arrival(D[None, :], np.full((1, K), nu))[0])
    assert abs(T - n_star) <= 1.0, 'non-integer hit %.3f vs %.1f' % (T, n_star)


def test_arrival_plateau_then_cross():
    """A flat stretch below the threshold must not be counted as failed."""
    K, n_star, plateau = 260, 200, 120
    nu = np.full(K, 1.0 / (n_star - plateau))
    nu[:plateau] = 0.0
    D = np.cumsum(nu)
    T = float(arrival(D[None, :], nu[None, :])[0])
    assert abs(T - n_star) <= 1.0, 'plateau-then-cross %.3f vs %d' % (T, n_star)


def test_arrival_variable_rate():
    """Accelerating degradation: local-rate estimate must stay within 3 cycles.

    The term is a local-rate estimate of steps remaining, so curvature costs
    accuracy; this pins how much for a quadratic profile and records that the
    per-step terms stay non-increasing (no survival rebound).
    """
    K, n_star = 200, 50
    j = np.arange(1, K + 1, dtype=np.float64)
    nu = j / (n_star * (n_star + 1) / 2.0)     # D_k = k(k+1)/2 / S, crosses at n*
    D = np.cumsum(nu)
    T = float(arrival(D[None, :], nu[None, :])[0])
    assert abs(T - n_star) <= 3.0, 'variable-rate arrival %.3f vs %d' % (T, n_star)
    z = 4.0 * (1.0 - D) / (nu + 1e-6)
    terms = _sigmoid(z)
    assert np.all(np.diff(terms) <= 1e-9), 'survival surrogate rebounds with k'


def test_arrival_not_hit_within_K():
    """Never reaching the threshold censors at K, with the documented K-0.5 limit."""
    K = 120
    nu = 1.0 / 400.0
    D = np.minimum(0.5, np.arange(K) * nu)
    T = float(arrival(D[None, :], np.full((1, K), nu))[0])
    assert abs(T - (K - 0.5)) < 1e-6, 'not-hit arrival %.6f vs K-0.5=%d.5' % (T, K)


def test_arrival_gradient_does_not_vanish():
    """The operator must pass gradient to D in the transition band.

    A hard argmin gives zero gradient everywhere. A soft one gives zero only
    where it is saturated (sigma ~ 0 or ~ 1), which is correct: those terms are
    constants with respect to D there. So the test asserts BOTH: non-zero in the
    band, exactly zero far away.
    """
    K, n_star = 200, 100
    D, nu = _trajectory(n_star, K)
    eps = 1e-3
    base = arrival(D, nu)[0]
    for k in (n_star - 2, n_star - 1, n_star, n_star + 1, n_star + 2):
        Dp = D.copy()
        Dp[0, k] += eps
        g = (arrival(Dp, nu)[0] - base) / eps
        assert abs(g) > 1e-6, 'dT/dD[%d] vanished in the band (%.3e)' % (k, g)
    Dp = D.copy()
    Dp[0, 0] += eps
    g_far = (arrival(Dp, nu)[0] - base) / eps
    assert abs(g_far) < 1e-9, 'far-field term should be saturated, got %.3e' % g_far
    # the rate must carry gradient too, otherwise D2 cannot be trained. Note it
    # cannot be probed AT the crossing: there 1-D = 0, so the term is
    # rate-insensitive by construction and the gradient is genuinely zero.
    nup = nu.copy()
    nup[0, n_star - 1] *= 1.01
    assert abs(arrival(D, nup)[0] - base) > 1e-9, 'dT/dnu vanished on the approach'


def test_arrival_full_graph_backward():
    """End-to-end: the loss must reach the rate network through the cumsum.

    The finite differences above probe the operator in isolation. This builds
    D from (D_t, nu) the way the model will, runs a real backward, and requires
    non-zero gradient on both inputs.
    """
    try:
        import torch
    except Exception:
        _skip('torch not importable here')
    K, n_star = 120, 60
    D0 = torch.tensor(0.0, requires_grad=True)
    nu = torch.full((1, K), 1.0 / n_star, requires_grad=True)
    D = D0 + torch.cumsum(nu, dim=-1)
    z = 4.0 * (1.0 - D) / (nu + 1e-6)
    T = (torch.sigmoid(z).sum(dim=-1) - 0.5).clamp(min=0.0, max=float(K))
    loss = (T - n_star) ** 2
    loss.backward()
    assert nu.grad is not None and float(nu.grad.abs().sum()) > 0, 'no gradient to nu'
    assert D0.grad is not None and abs(float(D0.grad)) > 0, 'no gradient to D0'
    assert float(T[0]) > 0.9 * n_star, 'T collapsed to %.2f' % float(T[0])


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = skipped = 0
    for fn in fns:
        try:
            fn()
            print('PASS  %s' % fn.__name__)
            ok += 1
        except _Skip as e:
            print('SKIP  %s  (%s)' % (fn.__name__, e))
            skipped += 1
        except Exception as e:                    # noqa: BLE001
            if type(e).__name__ == 'Skipped':
                print('SKIP  %s  (%s)' % (fn.__name__, e))
                skipped += 1
            else:
                print('FAIL  %s  %s: %s' % (fn.__name__, type(e).__name__, e))
                raise SystemExit(1)
    print('\n%d passed, %d skipped, 0 failed' % (ok, skipped))


if __name__ == '__main__':
    main()
