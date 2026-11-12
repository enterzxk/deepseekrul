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


# --------------------------------------------------------------------------
# 5. contract test: two-interface parity (skipped until engine_batch exists)
# --------------------------------------------------------------------------
def test_two_interface_parity():
    try:
        from paper_b.hc.data_hc import engine_batch  # noqa: F401
    except Exception:
        _skip('engine_batch not implemented yet (D7.1); contract test pending')
    raise AssertionError(
        'engine_batch exists -- implement the z_t parity assertion here before '
        'writing the training loop')


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
