"""Numerical data behind friendly statistical diagnostic visualizations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from kiwicalc.probability.distributions import ContinuousDistribution, Normal


def _policy(value):
    if value not in {'raise', 'omit'}:
        raise ValueError("nan_policy must be 'raise' or 'omit'")
    return value


def _sample(data, *, nan_policy='omit', minimum=1):
    nan_policy = _policy(nan_policy)
    try:
        values = np.asarray(data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError('data must contain real numbers') from exc
    if values.ndim == 0:
        values = values.reshape(1)
    else:
        values = values.reshape(-1)
    missing = int(np.count_nonzero(np.isnan(values)))
    if missing and nan_policy == 'raise':
        raise ValueError('data contains NaN values')
    values = values[~np.isnan(values)]
    if np.any(~np.isfinite(values)):
        raise ValueError('data must contain only finite values')
    if values.size < minimum:
        plural = '' if minimum == 1 else 's'
        raise ValueError(f'data must contain at least {minimum} observation{plural}')
    return values, missing


def _reference_distribution(values, distribution):
    if distribution is None or (
            isinstance(distribution, str) and distribution.lower() == 'normal'):
        if values.size < 2:
            raise ValueError('a fitted Normal diagnostic requires at least two observations')
        deviation = float(np.std(values, ddof=1))
        if deviation == 0:
            raise ValueError('a fitted Normal diagnostic requires non-constant data')
        return Normal(float(np.mean(values)), deviation), True
    if isinstance(distribution, str):
        raise ValueError("distribution must be 'normal' or a continuous distribution")
    if not isinstance(distribution, ContinuousDistribution):
        raise TypeError('distribution must be a continuous KiwiCalc distribution')
    return distribution, False


@dataclass(frozen=True)
class ECDFResult:
    """Unique observations and their empirical cumulative probabilities."""

    values: np.ndarray
    probabilities: np.ndarray
    counts: np.ndarray
    sample_size: int
    missing: int = 0

    def plot(self, **kwargs):
        """Plot this ECDF lazily and return its Matplotlib axes."""
        from kiwicalc.plotting.statistics import plot_ecdf
        return plot_ecdf(self, **kwargs)


@dataclass(frozen=True)
class QQData:
    """Theoretical and observed quantiles used by a Q-Q plot."""

    theoretical: np.ndarray
    observed: np.ndarray
    probabilities: np.ndarray
    reference_slope: float
    reference_intercept: float
    distribution: ContinuousDistribution
    fitted: bool
    missing: int = 0


@dataclass(frozen=True)
class PPData:
    """Theoretical and empirical probabilities used by a P-P plot."""

    theoretical: np.ndarray
    empirical: np.ndarray
    observed: np.ndarray
    distribution: ContinuousDistribution
    fitted: bool
    missing: int = 0


@dataclass(frozen=True)
class AssumptionSummary:
    """Descriptive signals relevant to common parametric assumptions.

    The messages are heuristics, not hypothesis tests or pass/fail decisions.
    """

    count: int
    missing: int
    mean: float
    median: float
    standard_deviation: float
    skewness: float
    excess_kurtosis: float
    outlier_count: int
    outlier_fraction: float
    constant: bool
    messages: tuple

    @property
    def has_messages(self):
        return bool(self.messages)

    def as_dict(self):
        return {
            'count': self.count,
            'missing': self.missing,
            'mean': self.mean,
            'median': self.median,
            'standard_deviation': self.standard_deviation,
            'skewness': self.skewness,
            'excess_kurtosis': self.excess_kurtosis,
            'outlier_count': self.outlier_count,
            'outlier_fraction': self.outlier_fraction,
            'constant': self.constant,
            'messages': self.messages,
        }


def ecdf(data, *, nan_policy='omit'):
    """Return the empirical CDF of one-dimensional numerical data."""
    values, missing = _sample(data, nan_policy=nan_policy)
    unique, counts = np.unique(values, return_counts=True)
    probabilities = np.cumsum(counts, dtype=float) / values.size
    return ECDFResult(unique, probabilities, counts, int(values.size), missing)


def qq_data(data, distribution=None, *, nan_policy='omit'):
    """Return data for comparing sample and theoretical quantiles.

    ``distribution=None`` and ``'normal'`` fit a Normal distribution using the
    sample mean and sample standard deviation. Passing a distribution uses its
    existing parameters without fitting.
    """
    values, missing = _sample(data, nan_policy=nan_policy, minimum=2)
    distribution, fitted = _reference_distribution(values, distribution)
    observed = np.sort(values)
    probabilities = (np.arange(values.size, dtype=float) + 0.5) / values.size
    theoretical = np.asarray(distribution.ppf(probabilities), dtype=float)
    if theoretical.shape != observed.shape or np.any(~np.isfinite(theoretical)):
        raise ValueError('distribution returned invalid quantiles for this diagnostic')
    tq = np.quantile(theoretical, [0.25, 0.75])
    oq = np.quantile(observed, [0.25, 0.75])
    width = float(tq[1] - tq[0])
    if width == 0:
        raise ValueError('distribution quantiles do not define a reference line')
    slope = float((oq[1] - oq[0]) / width)
    intercept = float(oq[0] - slope * tq[0])
    return QQData(theoretical, observed, probabilities, slope, intercept,
                  distribution, fitted, missing)


def pp_data(data, distribution=None, *, nan_policy='omit'):
    """Return data for comparing theoretical and empirical probabilities."""
    values, missing = _sample(data, nan_policy=nan_policy, minimum=2)
    distribution, fitted = _reference_distribution(values, distribution)
    observed = np.sort(values)
    empirical = (np.arange(values.size, dtype=float) + 0.5) / values.size
    theoretical = np.asarray(distribution.cdf(observed), dtype=float)
    if theoretical.shape != observed.shape or np.any(~np.isfinite(theoretical)):
        raise ValueError('distribution returned invalid probabilities for this diagnostic')
    if np.any((theoretical < 0) | (theoretical > 1)):
        raise ValueError('distribution probabilities must lie between zero and one')
    return PPData(theoretical, empirical, observed, distribution, fitted, missing)


def assumption_summary(data, *, nan_policy='omit'):
    """Summarize skew, tails, outliers, missingness, and sample-size signals.

    This intentionally avoids claiming that assumptions have passed or failed;
    independence and sampling design cannot be inferred from values alone.
    """
    values, missing = _sample(data, nan_policy=nan_policy)
    count = int(values.size)
    mean = float(np.mean(values))
    median = float(np.median(values))
    deviation = float(np.std(values, ddof=1)) if count > 1 else 0.0
    constant = bool(np.all(values == values[0]))
    if constant or count < 3:
        skew = math.nan
    else:
        centered = values - mean
        second = float(np.mean(centered ** 2))
        skew = float(np.mean(centered ** 3) / second ** 1.5)
    if constant or count < 4:
        kurtosis = math.nan
    else:
        centered = values - mean
        second = float(np.mean(centered ** 2))
        kurtosis = float(np.mean(centered ** 4) / second ** 2 - 3)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    spread = q3 - q1
    outliers = int(np.count_nonzero((values < q1 - 1.5 * spread) |
                                    (values > q3 + 1.5 * spread)))
    messages = []
    if missing:
        messages.append(f'{missing} missing observation(s) were omitted.')
    if count < 30:
        messages.append('The sample is small; graphical diagnostics may be unstable.')
    if constant:
        messages.append('The sample is constant, so spread and shape diagnostics are limited.')
    if math.isfinite(skew) and abs(skew) > 1:
        messages.append('The sample has strong skewness (|skewness| > 1).')
    if math.isfinite(kurtosis) and abs(kurtosis) > 2:
        messages.append('The sample has pronounced tail weight (|excess kurtosis| > 2).')
    if outliers:
        messages.append(f'{outliers} observation(s) fall outside the 1.5×IQR fences.')
    messages.append('Independence and sampling design require subject-matter judgment.')
    return AssumptionSummary(
        count, missing, mean, median, deviation, skew, kurtosis, outliers,
        outliers / count, constant, tuple(messages),
    )


__all__ = [
    'ECDFResult', 'QQData', 'PPData', 'AssumptionSummary',
    'ecdf', 'qq_data', 'pp_data', 'assumption_summary',
]
