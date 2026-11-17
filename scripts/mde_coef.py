"""Exact 80%-power MDE coefficient for a paired t test (two-sided alpha=0.05).

The plug-in shortcut `t_{0.975,df} + t_{0.80,df}` is a normal-style approximation
and understates the coefficient at small df. The exact statement is:

    find lam such that  P(|T'_{df,lam}| > c) = 0.80 ,  c = t_{0.975,df}
    then MDE80 = lam * SE            (lam = delta*sqrt(n)/sigma)

We evaluate the power by 1-D quadrature over the chi-square mixing variable:

    T' = (Z + lam) / sqrt(V/df),   Z~N(0,1), V~chi2_df
    P(|T'|>c) = E_V[ 1 - Phi(c*sqrt(V/df) - lam) + Phi(-c*sqrt(V/df) - lam) ]

No scipy required.
"""
import math

import numpy as np


def _phi(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _chi2_pdf(v, df):
    k = df / 2.0
    return np.exp((k - 1.0) * np.log(v) - v / 2.0 - k * math.log(2.0) - math.lgamma(k))


def power(lam, df, c):
    """P(|T'_{df,lam}| > c) by quadrature over the chi-square mixing variable."""
    # integrate in log-v so the tail is resolved
    lo, hi = -18.0, math.log(df * 60.0)
    t = np.linspace(lo, hi, 40001)
    v = np.exp(t)
    dens = _chi2_pdf(v, df) * v            # pdf(v) * v  = density in log-v
    s = np.sqrt(v / df)
    p = 1.0 - _phi(c * s - lam) + _phi(-c * s - lam)
    return float(np.trapz(dens * p, t))


def t_quantile_975(df):
    """Two-sided 95% critical value, by bisection on the central t CDF."""
    def cdf(c):
        return 1.0 - power(c, df, 0.0) / 1.0 if False else None
    # central t: P(|T|>c) = 1 - (Phi(c) - Phi(-c)) integrated; use survival = 0.05
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        # survival of |T| under lam=0
        t = np.linspace(-18.0, math.log(df * 60.0), 40001)
        v = np.exp(t)
        dens = _chi2_pdf(v, df) * v
        s = np.sqrt(v / df)
        surv = float(np.trapz(dens * (1.0 - _phi(mid * s) + _phi(-mid * s)), t))
        if surv > 0.05:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mde_coef(df, power_target=0.80):
    """Smallest lam whose power reaches power_target, i.e. MDE80 = lam * SE."""
    c = t_quantile_975(df)
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if power(mid, df, c) < power_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), c


if __name__ == '__main__':
    print('%4s %10s %14s %14s %8s' % ('df', 't_0.975', 'exact coef', 'plug-in', 'ratio'))
    for df in (1, 2, 3, 4, 5, 9):
        coef, c = mde_coef(df)
        plug = c + {1: 1.000, 2: 0.978, 3: 0.941, 4: 0.941, 5: 0.920, 9: 0.883}[df]
        print('%4d %10.4f %14.4f %14.4f %8.3f' % (df, c, coef, plug, coef / plug))
