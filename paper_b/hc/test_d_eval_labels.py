"""Actual-path check for the capped-label fix (advisor's admission condition).

The advisor's example: r_raw = 150, prediction = 125. The frozen protocol scores
min(r, 125), so that sample's error is ZERO for the main metric, while the raw
diagnostic must record 25. The earlier `evaluate()` clipped the prediction but
kept the raw `n - t` as the label, so it recorded 25 in the main metric too --
which moved model selection and D-G1.

This calls `metrics_from_arrays`, the same function `evaluate()` uses. A test
that re-implemented the cap would prove nothing about the code that runs.
"""
import numpy as np

from paper_b.hc.train_d import metrics_from_arrays

CAP = 125.0
K = 600


def test_capped_label_scores_zero_at_the_boundary():
    m = metrics_from_arrays([125.0], [150.0], [1], [0], CAP, K)
    assert m['val_pooled_rmse'] == 0.0, m['val_pooled_rmse']
    assert m['rmse_uncapped'] == 25.0, m['rmse_uncapped']
    assert m['frac_capped'] == 1.0
    assert m['boundary_hit'] == 1.0
    assert m['exact_boundary'] == 1.0


def test_prediction_below_cap_is_charged_on_the_capped_label():
    # pred 100 vs r_raw 150: capped label is 125 -> error 25; raw error 50
    m = metrics_from_arrays([100.0], [150.0], [1], [0], CAP, K)
    assert m['val_pooled_rmse'] == 25.0, m['val_pooled_rmse']
    assert m['rmse_uncapped'] == 50.0, m['rmse_uncapped']
    assert m['boundary_hit'] == 0.0


def test_uncensored_sample_is_unaffected_by_the_cap():
    # r_raw = 60 < cap: capped and raw labels agree, so both readings coincide
    m = metrics_from_arrays([70.0], [60.0], [1], [0], CAP, K)
    assert m['val_pooled_rmse'] == 10.0, m['val_pooled_rmse']
    assert m['rmse_uncensored'] == 10.0, m['rmse_uncensored']
    assert m['rmse_uncapped'] == 10.0, m['rmse_uncapped']


def test_engine_mean_uses_capped_labels_per_engine():
    # engine 1: two cuts, capped squared errors 0 and 625 -> RMSE sqrt(312.5)
    # engine 2: one cut, error 0 -> RMSE 0
    # the metric is the MEAN OF PER-ENGINE RMSE (not the root of pooled MSE):
    #   (sqrt(312.5) + 0)/2 = 8.8388...   vs pooled sqrt(625/3) = 14.43
    m = metrics_from_arrays([125.0, 100.0, 125.0], [150.0, 150.0, 150.0],
                            [1, 1, 2], [0, 1, 0], CAP, K)
    assert abs(m['val_engine_mean_rmse'] - np.sqrt(312.5) / 2.0) < 1e-6, \
        m['val_engine_mean_rmse']
    assert abs(m['val_pooled_rmse'] - np.sqrt(625.0 / 3.0)) < 1e-6, m['val_pooled_rmse']


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('PASS %s' % name)
    print('all evaluation-label tests passed')
