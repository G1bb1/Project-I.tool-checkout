"""Small exact-binomial helpers (no scipy needed)."""
from math import comb


def binom_cdf(k, n, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def upper_bound(k, n, conf=0.95):
    """One-sided exact (Clopper-Pearson) upper bound on a failure rate.
    With k = 0 this is 1 - (1-conf)**(1/n), roughly 3/n (the rule of three)."""
    if n == 0:
        return float("nan")
    if k >= n:
        return 1.0
    alpha = 1 - conf
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def percentile(values, q):
    if not values:
        return float("nan")
    v = sorted(values)
    idx = (len(v) - 1) * q
    lo, hi = int(idx), min(int(idx) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (idx - lo)
