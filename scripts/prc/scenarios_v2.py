"""Four scenarios with genuinely different delta-sign structures (R2 part 1)."""

BASE = dict(life_lo=40, life_hi=240, sigma_u=6.0, cap=125.0, n_fold=5,
            sigma_fold=2.0, sigma_seed=1.5, short_threshold=60.0, mid_lo=100.0,
            bias_split=80.0,
            protocols=("cap_all", "uncap_all", "short", "mid"))


def _mk(models):
    return {**BASE, "models": models}


SCENARIOS = {
    # A uniformly better: smaller noise in every life segment
    "A_dominant": _mk({"A": dict(bias=0.0, sigma=8.0),
                       "B": dict(bias=0.0, sigma=16.0)}),
    # B uniformly better
    "B_dominant": _mk({"A": dict(bias=0.0, sigma=16.0),
                       "B": dict(bias=0.0, sigma=8.0)}),
    # sign reversal through life-segment precision, not through a mean's sign:
    # A is precise near failure and poor early; B the reverse. Protocols that cover
    # short vs long trajectories therefore disagree in sign.
    # segment-dependent bias: A is unbiased near failure but under-predicts early in
    # life; B is the reverse. Protocols covering short vs mid life then disagree in
    # sign. This is a real failure mode of RUL models (under-prediction early in life)
    # and it cannot be produced by flipping a single mean's sign.
    "sign_flip": _mk({"A": dict(bias=0.0, sigma=6.0, bias_short=0.0, bias_long=-25.0),
                      "B": dict(bias=0.0, sigma=6.0, bias_short=25.0, bias_long=0.0)}),
    # near-zero: everything within a fraction of an RMSE
    "near_zero": _mk({"A": dict(bias=0.0, sigma=10.0),
                      "B": dict(bias=0.0, sigma=10.3)}),
}
