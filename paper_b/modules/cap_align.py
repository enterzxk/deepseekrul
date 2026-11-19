"""Cap-aware age alignment for K-window RUL predictions.

A window ending at cycle t predicts the (capped) RUL at t. To align it to the
cutoff c we subtract the age difference d = c - (t+1). When the prediction sits
on the cap plateau (~125), the true RUL did not decrease by d, so the plain
subtraction is biased low. The soft gate below removes most of the subtraction
on the plateau and keeps it on the linear part.
"""
import numpy as np


def align_cap_aware(p, delta, cap=125.0, tau=3.0, margin=3.0):
    p = np.asarray(p, dtype=float)
    delta = np.asarray(delta, dtype=float)
    pi = 1.0 / (1.0 + np.exp((cap - margin - p) / tau))
    return p - delta * pi
