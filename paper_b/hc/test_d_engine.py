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


def trajectory(D0, nu):
    """D[...,0] = current damage; D[...,k+1] = D[...,k] + nu[...,k].

    One definition shared by every test and by the production operator: the
    reviewer asked for exactly this, because the earlier tests disagreed about
    whether D[0] was "now" or "one step from now".
    """
    return np.concatenate([D0[..., :1], D0[..., :1] + np.cumsum(nu, axis=-1)], axis=-1)


def first_passage(D):
    """Exact first-passage time of a piecewise-linear damage trajectory.

    For each segment [D_k, D_{k+1}] the time spent BELOW the threshold is

        frac_k = clip( (1 - D_k) / (D_{k+1} - D_k), 0, 1 )      if the segment rises
        frac_k = 1[D_k < 1]                                     if it is flat

    and T = sum_k frac_k. This is not a surrogate: for a piecewise-linear
    trajectory it is the exact time at which D crosses 1, so

      already failed (D0 >= 1)     -> 0
      not hit within K             -> K
      constant rate                -> (1 - D0)/nu exactly
      crossing inside a segment    -> interpolated, not snapped

    Why the earlier rate-temperature form had to go: dividing by the LOCAL rate
    lets S rebound when the rate suddenly drops. With D0=0.90, nu0=0.5 and then
    nu=0.01, that form gives S0=0.69 -> S1~1, which is not a survival function.
    """
    lo, hi = D[..., :-1], D[..., 1:]
    d = hi - lo
    rising = d > 0
    safe = np.where(rising, d, 1.0)
    frac = np.where(rising,
                    np.clip((1.0 - lo) / safe, 0.0, 1.0),
                    (lo < 1.0).astype(np.float64))
    return frac.sum(axis=-1)


def arrival(D0, nu):
    """Convenience: T from the current damage and the predicted rate sequence."""
    return first_passage(trajectory(D0, nu))


def _trajectory(n_star, K, nu=None):
    """Constant-rate trajectory crossing 1 at step n_star (D0 = 0)."""
    nu = 1.0 / n_star if nu is None else nu
    return np.zeros((1, 1)), np.full((1, K), nu)


def test_arrival_old_operator_is_capped():
    """Witness: the rejected operator cannot exceed 1+e = 3.718 cycles, ever.

    The closed form is an UPPER BOUND, attained only when D is identically 0
    (then h_j = sigmoid(-1) for every j). That is worth pinning separately,
    because it is exactly the degenerate solution the loss can drift to: the
    operator's maximum sits at D = 0.
    """
    K, n_star = 300, 200
    bound = arrival_old_bound(beta=1.0, K=K)
    D = np.arange(K)[None, :] * (1.0 / n_star)
    T_old = float(arrival_old(D, beta=1.0)[0])
    assert T_old <= bound + 1e-9, 'old operator %.6f exceeded the bound %.6f' % (T_old, bound)
    assert T_old < 3.72, 'old bound %.6f >= 1+e' % T_old
    assert T_old < 0.02 * n_star, (
        'old operator returns %.3f for a trajectory that crosses at %d' % (T_old, n_star))
    T_zero = float(arrival_old(np.zeros((1, K)), beta=1.0)[0])
    assert abs(T_zero - bound) < 1e-9, 'the bound must be attained at D=0: %.6f vs %.6f' % (
        T_zero, bound)
    T_new = float(arrival(*_trajectory(n_star, K))[0])
    assert abs(T_new - n_star) <= 1.0, 'corrected operator %.3f vs %d' % (T_new, n_star)


def test_arrival_deceleration_no_rebound():
    """The reviewer's counterexample: a sudden slowdown must not raise survival.

    D0=0.90, nu=0.5 then 0.01. The rate-temperature form gave S0=0.69 -> S1~1,
    a survival rebound. The piecewise-linear form gives frac=1,1,... (the whole
    segment is below the threshold), which is monotone and correct.
    """
    nu = np.full((1, 21), 0.01)          # the NEXT increment is 0.01 (D1 = 0.91);
    D = trajectory(np.array([[0.90]]), nu)   # 0.5 is the CURRENT rate, a state,
    lo, hi = D[0, :-1], D[0, 1:]             # not the next step
    d = hi - lo
    frac = np.clip((1.0 - lo) / np.where(d > 0, d, 1.0), 0.0, 1.0)
    assert np.all(np.diff(frac) <= 1e-12), 'survival rebounded: %s' % np.diff(frac)[:5]
    assert frac[0] > 0.999, 'first segment is entirely below 1 but scored %.3f' % frac[0]
    T = float(arrival(np.array([[0.90]]), nu)[0])
    # crossing happens 10 cycles later at nu=0.01: (1-0.90)/0.01 = 10
    assert abs(T - 10.0) <= 1.0, 'deceleration arrival %.3f vs 10' % T


def test_arrival_stalled_at_threshold():
    """D already at 1 with a near-zero rate must give 0, not K/2."""
    nu = np.full((1, 100), 1e-6)
    T = float(arrival(np.array([[1.0]]), nu)[0])
    assert T <= 1e-3, 'stalled-at-threshold arrival %.4f should be ~0' % T


def test_arrival_representable_range_exceeds_rul_cap():
    """Constructive: the operator must be able to express RUL > 125."""
    for n_star in (125, 150, 300):
        K = n_star + 100
        T = float(arrival(*_trajectory(n_star, K))[0])
        assert abs(T - n_star) <= 1e-6, 'n*=%d -> T=%.6f' % (n_star, T)
    assert float(arrival(*_trajectory(150, 250))[0]) > 125.0


def test_arrival_already_failed():
    """At or past the threshold now -> arrival is 0 (RUL=0 semantics)."""
    nu = np.full((1, 200), 1.0 / 150.0)
    T = float(arrival(np.array([[1.0]]), nu)[0])
    assert T <= 1e-9, 'already-failed arrival %.6f should be 0' % T


def test_arrival_exact_hit():
    """Crossing exactly at k*=50 is reported as 50."""
    T = float(arrival(*_trajectory(50, 200))[0])
    assert abs(T - 50.0) <= 1e-6, 'exact hit %.6f vs 50' % T


def test_arrival_non_integer_hit():
    """A crossing inside a segment is interpolated, not snapped."""
    T = float(arrival(*_trajectory(50.5, 200))[0])
    assert abs(T - 50.5) <= 1e-6, 'non-integer hit %.6f vs 50.5' % T


def test_arrival_plateau_then_cross():
    """A flat stretch below the threshold is counted in full."""
    K, n_star, plateau = 260, 200, 120
    nu = np.full((1, K), 1.0 / (n_star - plateau))
    nu[0, :plateau] = 0.0
    T = float(arrival(np.zeros((1, 1)), nu)[0])
    assert abs(T - n_star) <= 1e-6, 'plateau-then-cross %.6f vs %d' % (T, n_star)


def test_arrival_variable_rate():
    """Accelerating degradation: the piecewise-linear form is EXACT.

    The earlier rate-temperature form needed a 3-cycle tolerance here; an exact
    operator must meet the same 1-cycle standard as everything else, so the
    tolerance is tightened to 1e-6 rather than widened.
    """
    K, n_star = 200, 50
    j = np.arange(1, K + 1, dtype=np.float64)
    nu = (j / (n_star * (n_star + 1) / 2.0))[None, :]   # D_k = k(k+1)/2 / S
    T = float(arrival(np.zeros((1, 1)), nu)[0])
    assert abs(T - n_star) <= 1e-6, 'variable-rate arrival %.6f vs %d' % (T, n_star)


def test_arrival_not_hit_within_K():
    """Never reaching the threshold returns exactly K (right censoring)."""
    K = 120
    nu = np.full((1, K), 1.0 / 400.0)
    T = float(arrival(np.zeros((1, 1)), nu)[0])
    assert abs(T - K) <= 1e-9, 'not-hit arrival %.6f vs K=%d' % (T, K)


def test_arrival_gradient_structure():
    """Gradient reaches every increment up to the crossing, and nothing after it.

    The reviewer corrected an earlier version of this test: an early increment
    does NOT leave T unchanged. It shifts the cumulative damage, hence the
    crossing position, hence T. With the crossing inside segment j,

        dT/dnu_i = -1/nu[j+1]      for i <= j
        dT/dnu_i =  0              for i > j+1

    (the crossing segment also gets -(1-D_j)/nu[j+1]^2 from the denominator).
    """
    K, n_star = 200, 100.5                    # interior crossing at segment j=100
    j = 100
    D0, nu = _trajectory(n_star, K)           # nu = 1/100.5 everywhere
    base = float(arrival(D0, nu)[0])
    assert abs(base - n_star) <= 1e-6

    # early increment: moves the crossing, magnitude -1/nu[j+1]
    nup = nu.copy()
    dnu = nup[0, 0] * 0.05
    nup[0, 0] += dnu
    d_early = float(arrival(D0, nup)[0]) - base
    expected = -dnu / nu[0, j + 1]
    assert abs(d_early - expected) <= 0.1 * abs(expected), (
        'early increment moved T by %.3e, expected %.3e' % (d_early, expected))

    # increment well after the crossing segment: no effect on T
    nup = nu.copy()
    nup[0, j + 5] *= 1.05
    assert abs(float(arrival(D0, nup)[0]) - base) < 1e-12, (
        'an increment after the crossing must not move T')

    # the crossing segment itself: gradient through the denominator
    nup = nu.copy()
    nup[0, j] *= 1.05
    assert abs(float(arrival(D0, nup)[0]) - base) > 1e-6, 'crossing segment did not move T'


def test_arrival_zero_gradient_trap():
    """If nu*K <= 1 the trajectory never reaches 1: T = K and the gradient is 0.

    The model would then be stuck at the censoring horizon with no signal to
    come back. softplus(0)=0.693 gives nu*K ~ 138 >> 1 at init, which is safe,
    but the trap is asserted so an initialisation change cannot silently open it.
    """
    K = 200
    for nu_val in (1e-3, 1.0 / K * 0.5):
        nu = np.full((1, K), nu_val)
        assert nu_val * K < 1.0, 'this case must be unreachable'
        T = float(arrival(np.zeros((1, 1)), nu)[0])
        assert abs(T - K) <= 1e-9, 'nu=%.2e should never reach the threshold' % nu_val
    # just above the boundary the crossing becomes reachable again
    nu = np.full((1, K), 1.05 / K)
    assert float(arrival(np.zeros((1, 1)), nu)[0]) < K


def test_arrival_requires_monotone_trajectory():
    """The clip sum equals the first passage only for a monotone trajectory.

    D is built from non-negative increments by construction, so this asserts the
    precondition rather than an implementation detail: a non-monotone trajectory
    would turn the sum into "time spent below the threshold", which is a
    different quantity.
    """
    K = 60
    nu = np.full((1, K), 1.0 / 30.0)
    D = trajectory(np.zeros((1, 1)), nu)
    assert np.all(np.diff(D[0]) >= 0), 'trajectory is not monotone'
    nu_bad = np.full((1, K), 1.0 / 30.0)
    nu_bad[0, 20] = -0.5                       # negative increment
    D_bad = trajectory(np.zeros((1, 1)), nu_bad)
    assert np.any(np.diff(D_bad[0]) < 0), 'the negative increment should break monotonicity'


def test_arrival_full_graph_backward():
    """End-to-end: the loss must reach the rate network through the cumsum.

    The crossing must be INTERIOR to a segment. When it falls exactly on a
    boundary every segment clamps to 0 or 1 and the gradient is identically
    zero -- asserted separately below as a documented property, not a bug.
    """
    try:
        import torch
    except Exception:
        _skip('torch not importable here')
    K, n_star = 120, 60.5                    # interior crossing
    D0 = torch.tensor(0.0, requires_grad=True)
    nu = torch.full((1, K), 1.0 / n_star, requires_grad=True)
    D = torch.cat([D0.reshape(1, 1), D0.reshape(1, 1) + torch.cumsum(nu, dim=-1)], dim=-1)
    lo, hi = D[:, :-1], D[:, 1:]
    d = (hi - lo).clamp(min=1e-12)
    frac = ((1.0 - lo) / d).clamp(0.0, 1.0)
    T = frac.sum(dim=-1)
    loss = (T - (n_star + 10.0)) ** 2      # non-zero residual: at the minimum the gradient is 0 by definition
    loss.backward()
    assert nu.grad is not None and float(nu.grad.abs().sum()) > 0, 'no gradient to nu'
    assert D0.grad is not None and abs(float(D0.grad)) > 0, 'no gradient to D0'
    assert abs(float(T[0]) - n_star) <= 1e-4, 'T=%.4f vs %.1f' % (float(T[0]), n_star)


def test_arrival_boundary_hit_is_exact():
    """A crossing exactly on a segment boundary is still reported exactly.

    (An earlier draft asserted zero gradient here; that is wrong. clamp's
    derivative is 1 ON the boundary, so the adjacent segments -- whose ratio is
    exactly 1.0 or 0.0 -- still pass gradient. Measured dT/dD0 = -60 at K=120,
    n*=60. What matters is that the value is exact, which is asserted here.)
    """
    K, n_star = 120, 60
    D0, nu = _trajectory(n_star, K)
    base = float(arrival(D0, nu)[0])
    assert abs(base - n_star) <= 1e-9, 'boundary hit %.9f vs %d' % (base, n_star)
    eps = 1e-3
    g = (float(arrival(D0 + eps, nu)[0]) - base) / eps
    assert abs(g) > 1e-6, 'boundary hit lost its gradient entirely (%.3e)' % g


def test_arrival_not_hit_has_no_gradient():
    """Documented property: censored samples carry no gradient.

    A trajectory that never reaches 1 gives T = K for every setting of D and nu,
    so the loss is flat and the sample is invisible to training. This is the
    censored-hinge semantics, not a defect -- but it must be reported, because
    it means early-life windows do not constrain the damage path at all.
    """
    K = 120
    nu = np.full((1, K), 1.0 / 4000.0)
    base = float(arrival(np.zeros((1, 1)), nu)[0])
    assert abs(base - K) <= 1e-9
    for scale in (1.0, 2.0, 5.0):
        assert abs(float(arrival(np.zeros((1, 1)), nu * scale)[0]) - K) <= 1e-9
    assert abs(float(arrival(np.array([[0.5]]), nu)[0]) - K) <= 1e-9


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
