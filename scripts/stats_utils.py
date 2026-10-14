"""
stats_utils.py
================
Statistical-rigor helpers for SGSQ-Net's held-out TEST-SET evaluation.

Why this file exists
---------------------
Every number reported so far in this project (Dice, QWK, accuracy, ...) has
been a single point estimate computed on the VALIDATION split -- the same
split used for early stopping / checkpoint selection. A reviewer at any
serious venue will flag that immediately: reusing the validation set for
both model selection AND final reporting is a well-known source of
optimistic bias, and a single point estimate with no uncertainty band gives
no way to judge whether an observed difference between models (e.g. SGSQ-Net
vs. a baseline) is real or noise.

This module provides two things every "final" number in the paper must go
through before it is reported:

1. `bootstrap_ci`      -- non-parametric bootstrap confidence interval for
                          any metric computed over a set of per-item values
                          (images, for segmentation; test samples, for
                          severity classification).
2. `paired_bootstrap_test` -- paired bootstrap significance test comparing
                          two models evaluated on the SAME held-out items,
                          returning the observed difference, its CI, and a
                          two-sided p-value. This is what justifies a
                          sentence like "SGSQ-Net's QWK was significantly
                          higher than EfficientNet-B0's (p = 0.01)".

Both are seeded, so results are reproducible run-to-run.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Sequence, Tuple, Dict


def bootstrap_ci(
    values: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Non-parametric bootstrap CI for a statistic computed over `values`.

    `values` is an array of per-item values (e.g. per-image Dice scores).
    `statistic_fn` reduces an array of (possibly resampled) values to a
    single number (default use: np.mean).

    Returns dict with point estimate, lower/upper CI bounds, and std error.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "se": float("nan"), "n": 0}

    rng = np.random.default_rng(seed)
    point = float(statistic_fn(values))

    boot_stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_stats[b] = statistic_fn(values[idx])

    alpha = (100.0 - ci) / 2.0
    lo, hi = np.percentile(boot_stats, [alpha, 100.0 - alpha])
    se = float(np.std(boot_stats, ddof=1))
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi), "se": se, "n": n}


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = 5000,
    ci: float = 95.0,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Paired bootstrap test for whether model A's per-item scores are
    significantly different from model B's, on the SAME held-out items
    (e.g. per-test-image Dice, or a per-sample correct/incorrect indicator,
    or a per-sample squared-error-to-true-grade term for QWK-style metrics).

    scores_a, scores_b: 1-D arrays, same length, same item order.

    Returns the observed mean difference (A - B), its bootstrap CI, and a
    two-sided p-value obtained by checking how often the bootstrap
    distribution of the difference crosses zero.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    assert a.shape == b.shape, "paired_bootstrap_test requires equal-length paired arrays"
    n = len(a)
    if n == 0:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan"), "n": 0}

    rng = np.random.default_rng(seed)
    diff = a - b
    observed = float(np.mean(diff))

    boot_diffs = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs[k] = np.mean(diff[idx])

    alpha = (100.0 - ci) / 2.0
    lo, hi = np.percentile(boot_diffs, [alpha, 100.0 - alpha])

    # two-sided p-value: proportion of bootstrap replicates on the opposite
    # side of zero from the observed effect, doubled (standard percentile
    # approach), clipped to [0, 1]
    if observed >= 0:
        p = 2.0 * np.mean(boot_diffs <= 0.0)
    else:
        p = 2.0 * np.mean(boot_diffs >= 0.0)
    p = float(min(1.0, max(0.0, p)))

    return {"diff": observed, "ci_low": float(lo), "ci_high": float(hi), "p_value": p, "n": n}


def qwk_per_sample_terms(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Decompose quadratic weighted kappa into a per-sample quantity so it can
    be bootstrapped and paired-tested like any other per-item score.

    QWK is not naturally a per-sample average metric, so we use the
    standard trick: bootstrap the two confusion-derived quantities that
    make up kappa (observed weighted disagreement O and expected weighted
    disagreement E) by resampling *pairs of samples*, and reconstruct QWK
    per bootstrap resample from those aggregates. To keep this compatible
    with the simple `bootstrap_ci`/`paired_bootstrap_test` helpers above
    (which resample independent scalars), we instead return, per sample,
    the negative squared ordinal error (i.e. -(y_true - y_pred)^2), scaled
    to [0, 1] by the maximum possible squared error. This is NOT QWK
    itself, but it is a proper per-sample decomposition of the "quadratic
    penalty" that QWK is built from, and differences in its bootstrap mean
    track differences in QWK closely while being exactly poolable/pairable.
    Use `compute_qwk_bootstrap` below for a direct, correct bootstrap of
    QWK itself (recommended for the headline number); use this helper only
    for a fast paired significance check.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    max_err2 = (num_classes - 1) ** 2
    return 1.0 - ((y_true - y_pred) ** 2) / max_err2


def _fast_qwk(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """
    Quadratic weighted kappa computed directly from a confusion matrix via
    np.bincount, instead of calling sklearn.metrics.cohen_kappa_score.

    This is mathematically identical to
    `cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=range(num_classes))`
    but 1-2 orders of magnitude faster when called thousands of times inside
    a bootstrap loop (as compute_qwk_bootstrap/paired_qwk_bootstrap_test do),
    because it skips sklearn's per-call input validation and array-type
    dispatch overhead. Verified equal to sklearn's result to float precision
    (see the smoke test in README_TEST_EVAL.md).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = len(y_true)
    if n == 0:
        return float("nan")

    flat = y_true * num_classes + y_pred
    confusion = np.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes).astype(np.float64)

    row_marg = confusion.sum(axis=1)
    col_marg = confusion.sum(axis=0)
    N = confusion.sum()
    if N == 0:
        return float("nan")

    idx = np.arange(num_classes, dtype=np.float64)
    weights = (idx[:, None] - idx[None, :]) ** 2 / max((num_classes - 1) ** 2, 1)

    expected = np.outer(row_marg, col_marg) / N
    denom = float((weights * expected).sum())
    if denom == 0.0:
        return float("nan")
    numer = float((weights * confusion).sum())
    return 1.0 - numer / denom


def compute_qwk_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Direct, correct bootstrap CI for quadratic weighted kappa: resample
    (y_true, y_pred) PAIRS with replacement and recompute QWK from scratch
    on each resample using a fast confusion-matrix-based implementation
    (`_fast_qwk`, mathematically identical to
    `cohen_kappa_score(weights='quadratic')` -- see its docstring).
    This is the number that should be reported as "QWK [95% CI]".
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    point = _fast_qwk(y_true, y_pred, num_classes)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = _fast_qwk(y_true[idx], y_pred[idx], num_classes)
    boots = boots[~np.isnan(boots)]

    alpha = (100.0 - ci) / 2.0
    if len(boots) == 0:
        lo = hi = float("nan")
    else:
        lo, hi = np.percentile(boots, [alpha, 100.0 - alpha])
    return {"point": float(point), "ci_low": float(lo), "ci_high": float(hi), "n": n}


def paired_qwk_bootstrap_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    num_classes: int,
    n_boot: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Paired bootstrap significance test for "is model A's QWK different from
    model B's QWK on the same held-out items". Resamples ROWS (i.e. the
    same resampled index set is applied to y_true, pred_a, pred_b jointly,
    which is what makes this a *paired* test), recomputes both QWKs on
    each resample using the fast confusion-matrix-based `_fast_qwk`, and
    looks at the distribution of (QWK_a - QWK_b).
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    observed = _fast_qwk(y_true, y_pred_a, num_classes) - _fast_qwk(y_true, y_pred_b, num_classes)

    diffs = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[k] = _fast_qwk(y_true[idx], y_pred_a[idx], num_classes) - _fast_qwk(y_true[idx], y_pred_b[idx], num_classes)
    diffs = diffs[~np.isnan(diffs)]

    lo, hi = np.percentile(diffs, [2.5, 97.5]) if len(diffs) else (float("nan"), float("nan"))
    if observed >= 0:
        p = 2.0 * np.mean(diffs <= 0.0) if len(diffs) else float("nan")
    else:
        p = 2.0 * np.mean(diffs >= 0.0) if len(diffs) else float("nan")
    p = float(min(1.0, max(0.0, p))) if not np.isnan(p) else p

    return {"diff": float(observed), "ci_low": float(lo), "ci_high": float(hi), "p_value": p, "n": n}
