"""Statistical inference with a compact, NumPy-only public API."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from statistics import NormalDist

import numpy as np

from kiwicalc.probability.descriptive import ContingencyTable


@dataclass(frozen=True)
class ConfidenceInterval:
    """An estimate and its confidence bounds."""

    lower: object
    upper: object
    confidence: float
    estimate: object
    standard_error: object = None
    method: str = ""

    @property
    def margin_of_error(self):
        return (np.asarray(self.upper) - np.asarray(self.lower)) / 2

    @property
    def width(self):
        return np.asarray(self.upper) - np.asarray(self.lower)

    def contains(self, value):
        result = (np.asarray(value) >= np.asarray(self.lower)) & (
            np.asarray(value) <= np.asarray(self.upper)
        )
        return bool(result) if result.ndim == 0 else result

    def as_tuple(self):
        return self.lower, self.upper

    def __iter__(self):
        return iter(self.as_tuple())


@dataclass(frozen=True)
class TestResult:
    """Unified result returned by KiwiCalc hypothesis tests."""

    # Prevent pytest from mistaking this public result type for a test class
    # when users import it into test modules.
    __test__ = False

    statistic: object
    p_value: object
    method: str
    alternative: str = "two-sided"
    degrees_of_freedom: object = None
    estimate: object = None
    null_value: object = None
    standard_error: object = None
    confidence_interval: ConfidenceInterval = None
    effect_size: object = None
    details: object = None

    @property
    def pvalue(self):
        return self.p_value

    def significant(self, alpha=0.05):
        alpha = _probability(alpha, "alpha", open_interval=True)
        result = np.asarray(self.p_value) < alpha
        return bool(result) if result.ndim == 0 else result

    reject_null = significant

    def as_dict(self):
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "method": self.method,
            "alternative": self.alternative,
            "degrees_of_freedom": self.degrees_of_freedom,
            "estimate": self.estimate,
            "null_value": self.null_value,
            "standard_error": self.standard_error,
            "confidence_interval": self.confidence_interval,
            "effect_size": self.effect_size,
            "details": self.details,
        }


def _probability(value, name, *, open_interval=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    valid = 0 < value < 1 if open_interval else 0 <= value <= 1
    if not math.isfinite(value) or not valid:
        qualifier = "strictly " if open_interval else ""
        raise ValueError(f"{name} must be {qualifier}between zero and one")
    return value


def _alternative(value):
    if not isinstance(value, str):
        raise TypeError("alternative must be text")
    value = value.lower().replace("_", "-")
    aliases = {"two-sided": "two-sided", "two-tailed": "two-sided",
               "less": "less", "lower": "less", "greater": "greater",
               "upper": "greater"}
    if value not in aliases:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    return aliases[value]


def _policy(value):
    if not isinstance(value, str) or value not in {"raise", "omit", "propagate"}:
        raise ValueError("nan_policy must be 'raise', 'omit', or 'propagate'")
    return value


def _numeric(data, name="data"):
    try:
        values = np.asarray(data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc
    if values.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return values


def _sample_stats(data, axis, nan_policy, *, minimum=2, name="data"):
    values = _numeric(data, name)
    if np.any(np.isinf(values)):
        raise ValueError(f"{name} must not contain infinite values")
    policy = _policy(nan_policy)
    if axis is not None:
        if isinstance(axis, bool) or not isinstance(axis, Integral):
            raise TypeError("axis must be an integer or None")
        axis = int(axis)
        if axis < 0:
            axis += values.ndim
        if axis < 0 or axis >= values.ndim:
            raise ValueError(f"axis {axis} is out of bounds for {values.ndim} dimensions")
    if policy == "raise" and np.any(np.isnan(values)):
        raise ValueError(f"{name} contains NaN")
    if policy == "omit":
        count = np.sum(~np.isnan(values), axis=axis)
        if np.any(count < minimum):
            raise ValueError(f"{name} needs at least {minimum} non-missing observations")
        with np.errstate(invalid="ignore"):
            average = np.nanmean(values, axis=axis)
            deviation = np.nanstd(values, axis=axis, ddof=1)
    else:
        count = values.size if axis is None else values.shape[axis]
        if count < minimum:
            raise ValueError(f"{name} needs at least {minimum} observations")
        average = np.mean(values, axis=axis)
        deviation = np.std(values, axis=axis, ddof=1)
    return np.asarray(average), np.asarray(deviation), np.asarray(count), values, axis


def _scalar(value):
    value = np.asarray(value)
    return value.item() if value.ndim == 0 else value


def _safe_ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    )
    result = np.empty(numerator.shape, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    result = np.where(denominator == 0,
                      np.where(numerator == 0, 0.0,
                               np.where(numerator > 0, np.inf, -np.inf)), result)
    return result


def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1 / max(abs(d), 1e-300) * (1 if d >= 0 else -1)
    result = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c
        c = c if abs(c) > 1e-300 else 1e-300
        result *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c
        c = c if abs(c) > 1e-300 else 1e-300
        change = d * c
        result *= change
        if abs(change - 1) < 3e-14:
            break
    return result


def _regularized_beta(x, a, b):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1 - front * _betacf(b, a, 1 - x) / b


def _student_cdf_scalar(value, degrees_of_freedom):
    if math.isnan(value):
        return math.nan
    if value == math.inf:
        return 1.0
    if value == -math.inf:
        return 0.0
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * _regularized_beta(x, degrees_of_freedom / 2, 0.5)
    return 1 - tail if value >= 0 else tail


def _student_cdf(value, degrees_of_freedom):
    values, dfs = np.broadcast_arrays(np.asarray(value, dtype=float),
                                      np.asarray(degrees_of_freedom, dtype=float))
    result = np.fromiter((_student_cdf_scalar(float(x), float(df))
                          for x, df in zip(values.flat, dfs.flat)),
                         dtype=float, count=values.size).reshape(values.shape)
    return _scalar(result)


def _student_ppf_scalar(probability, degrees_of_freedom):
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -_student_ppf_scalar(1 - probability, degrees_of_freedom)
    low, high = 0.0, 1.0
    while _student_cdf_scalar(high, degrees_of_freedom) < probability:
        high *= 2
    for _ in range(80):
        middle = (low + high) / 2
        if _student_cdf_scalar(middle, degrees_of_freedom) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _student_ppf(probability, degrees_of_freedom):
    dfs = np.asarray(degrees_of_freedom, dtype=float)
    result = np.fromiter((_student_ppf_scalar(probability, float(df)) for df in dfs.flat),
                         dtype=float, count=dfs.size).reshape(dfs.shape)
    return _scalar(result)


def _normal_cdf(value):
    values = np.asarray(value, dtype=float)
    normal = NormalDist()
    result = np.fromiter((normal.cdf(float(x)) for x in values.flat),
                         dtype=float, count=values.size).reshape(values.shape)
    return _scalar(result)


def _p_value(statistic, cdf, alternative):
    cumulative = np.asarray(cdf(statistic), dtype=float)
    if alternative == "less":
        result = cumulative
    elif alternative == "greater":
        result = 1 - cumulative
    else:
        result = 2 * np.minimum(cumulative, 1 - cumulative)
    return _scalar(np.clip(result, 0, 1))


def _interval(estimate, standard_error, confidence, critical, alternative, method):
    estimate, standard_error = np.asarray(estimate), np.asarray(standard_error)
    if alternative == "greater":
        lower, upper = estimate - critical * standard_error, np.full_like(estimate, np.inf)
    elif alternative == "less":
        lower, upper = np.full_like(estimate, -np.inf), estimate + critical * standard_error
    else:
        lower = estimate - critical * standard_error
        upper = estimate + critical * standard_error
    return ConfidenceInterval(_scalar(lower), _scalar(upper), confidence,
                              _scalar(estimate), _scalar(standard_error), method)


def mean_confidence_interval(data, confidence=0.95, *, sigma=None, axis=None,
                             nan_policy="propagate"):
    """Return a Student-t interval, or a z interval when ``sigma`` is known."""
    confidence = _probability(confidence, "confidence", open_interval=True)
    average, deviation, count, _, _ = _sample_stats(data, axis, nan_policy)
    if sigma is None:
        standard_error = deviation / np.sqrt(count)
        critical = _student_ppf((1 + confidence) / 2, count - 1)
        method = "Student t confidence interval"
    else:
        sigma = _numeric(sigma, "sigma")
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("sigma must contain positive finite values")
        standard_error = sigma / np.sqrt(count)
        critical = NormalDist().inv_cdf((1 + confidence) / 2)
        method = "Normal z confidence interval"
    return _interval(average, standard_error, confidence, critical, "two-sided", method)


confidence_interval = mean_confidence_interval


def mean_test(data, expected=0, *, sigma=None, alternative="two-sided",
              confidence=0.95, axis=None, nan_policy="propagate"):
    """Test one sample mean against ``expected`` using t or known-sigma z inference."""
    alternative = _alternative(alternative)
    confidence = _probability(confidence, "confidence", open_interval=True)
    average, deviation, count, _, _ = _sample_stats(data, axis, nan_policy)
    expected = _numeric(expected, "expected")
    if np.any(~np.isfinite(expected)):
        raise ValueError("expected must contain finite values")
    if sigma is None:
        standard_error = deviation / np.sqrt(count)
        degrees = count - 1
        statistic = _safe_ratio(average - expected, standard_error)
        p_value = _p_value(statistic, lambda value: _student_cdf(value, degrees), alternative)
        probability = (1 + confidence) / 2 if alternative == "two-sided" else confidence
        critical = _student_ppf(probability, degrees)
        method = "One-sample Student t test"
    else:
        sigma = _numeric(sigma, "sigma")
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("sigma must contain positive finite values")
        standard_error = sigma / np.sqrt(count)
        statistic = (average - expected) / standard_error
        degrees = None
        p_value = _p_value(statistic, _normal_cdf, alternative)
        probability = (1 + confidence) / 2 if alternative == "two-sided" else confidence
        critical = NormalDist().inv_cdf(probability)
        method = "One-sample z test"
    effect = _safe_ratio(average - expected, deviation)
    interval = _interval(average, standard_error, confidence, critical, alternative, method)
    return TestResult(_scalar(statistic), p_value, method, alternative,
                      _scalar(degrees) if degrees is not None else None, _scalar(average),
                      _scalar(expected), _scalar(standard_error), interval, _scalar(effect))


one_sample_t_test = mean_test


def compare_means(first, second, *, paired=False, equal_variance=False,
                  alternative="two-sided", confidence=0.95, axis=None,
                  nan_policy="propagate"):
    """Compare two means with Welch's test by default, pooled or paired on request."""
    alternative = _alternative(alternative)
    confidence = _probability(confidence, "confidence", open_interval=True)
    if not isinstance(paired, bool) or not isinstance(equal_variance, bool):
        raise TypeError("paired and equal_variance must be Boolean")
    if paired:
        first_values, second_values = _numeric(first, "first"), _numeric(second, "second")
        if first_values.shape != second_values.shape:
            raise ValueError("paired samples must have the same shape")
        result = mean_test(first_values - second_values, 0, alternative=alternative,
                           confidence=confidence, axis=axis, nan_policy=nan_policy)
        return TestResult(result.statistic, result.p_value, "Paired Student t test",
                          result.alternative, result.degrees_of_freedom, result.estimate,
                          0.0, result.standard_error, result.confidence_interval,
                          result.effect_size)
    mean1, std1, n1, _, _ = _sample_stats(first, axis, nan_policy, name="first")
    mean2, std2, n2, _, _ = _sample_stats(second, axis, nan_policy, name="second")
    difference = mean1 - mean2
    if equal_variance:
        degrees = n1 + n2 - 2
        pooled_variance = ((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / degrees
        standard_error = np.sqrt(pooled_variance * (1 / n1 + 1 / n2))
        effect_denominator = np.sqrt(pooled_variance)
        method = "Independent pooled Student t test"
    else:
        first_term, second_term = std1 ** 2 / n1, std2 ** 2 / n2
        standard_error = np.sqrt(first_term + second_term)
        degrees_denominator = (first_term ** 2 / (n1 - 1)
                               + second_term ** 2 / (n2 - 1))
        degrees = _safe_ratio((first_term + second_term) ** 2, degrees_denominator)
        degrees = np.where(degrees_denominator == 0, n1 + n2 - 2, degrees)
        effect_denominator = np.sqrt((std1 ** 2 + std2 ** 2) / 2)
        method = "Welch two-sample t test"
    statistic = _safe_ratio(difference, standard_error)
    p_value = _p_value(statistic, lambda value: _student_cdf(value, degrees), alternative)
    probability = (1 + confidence) / 2 if alternative == "two-sided" else confidence
    critical = _student_ppf(probability, degrees)
    interval = _interval(difference, standard_error, confidence, critical, alternative, method)
    effect = _safe_ratio(difference, effect_denominator)
    return TestResult(_scalar(statistic), p_value, method, alternative, _scalar(degrees),
                      _scalar(difference), 0.0, _scalar(standard_error), interval,
                      _scalar(effect))


two_sample_t_test = compare_means


def proportion_confidence_interval(successes, trials, confidence=0.95, *, method="wilson"):
    """Return a Wilson (default) or Wald interval for one or more proportions."""
    confidence = _probability(confidence, "confidence", open_interval=True)
    successes, trials = np.broadcast_arrays(np.asarray(successes), np.asarray(trials))
    if (not np.issubdtype(successes.dtype, np.integer)
            or not np.issubdtype(trials.dtype, np.integer)):
        raise TypeError("successes and trials must be integers")
    if np.any(trials <= 0) or np.any(successes < 0) or np.any(successes > trials):
        raise ValueError("require 0 <= successes <= trials and positive trials")
    estimate = successes / trials
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    method = method.lower() if isinstance(method, str) else method
    if method == "wilson":
        denominator = 1 + z * z / trials
        center = (estimate + z * z / (2 * trials)) / denominator
        margin = z / denominator * np.sqrt(estimate * (1 - estimate) / trials
                                           + z * z / (4 * trials ** 2))
        lower, upper = center - margin, center + margin
        label = "Wilson score interval"
    elif method in {"wald", "normal"}:
        standard_error = np.sqrt(estimate * (1 - estimate) / trials)
        lower, upper = estimate - z * standard_error, estimate + z * standard_error
        lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)
        label = "Wald proportion interval"
    else:
        raise ValueError("method must be 'wilson' or 'wald'")
    standard_error = np.sqrt(estimate * (1 - estimate) / trials)
    return ConfidenceInterval(_scalar(lower), _scalar(upper), confidence, _scalar(estimate),
                              _scalar(standard_error), label)


def proportion_test(successes, trials, expected=0.5, *, alternative="two-sided",
                    confidence=0.95):
    """Test one observed proportion against an expected probability."""
    alternative = _alternative(alternative)
    expected = _probability(expected, "expected", open_interval=True)
    interval = proportion_confidence_interval(successes, trials, confidence)
    estimate = np.asarray(interval.estimate)
    trials_array = np.asarray(trials)
    standard_error = np.sqrt(expected * (1 - expected) / trials_array)
    statistic = (estimate - expected) / standard_error
    p_value = _p_value(statistic, _normal_cdf, alternative)
    return TestResult(_scalar(statistic), p_value, "One-sample proportion z test",
                      alternative, None, _scalar(estimate), expected, _scalar(standard_error),
                      interval, _scalar(estimate - expected))


def compare_proportions(successes1, trials1, successes2, trials2, *,
                        alternative="two-sided", confidence=0.95):
    """Compare two independent proportions with a pooled z test."""
    alternative = _alternative(alternative)
    first = proportion_confidence_interval(successes1, trials1, confidence)
    second = proportion_confidence_interval(successes2, trials2, confidence)
    first_estimate, second_estimate = np.asarray(first.estimate), np.asarray(second.estimate)
    pooled = (np.asarray(successes1) + np.asarray(successes2)) / (
        np.asarray(trials1) + np.asarray(trials2)
    )
    standard_error = np.sqrt(pooled * (1 - pooled) * (1 / np.asarray(trials1)
                                                       + 1 / np.asarray(trials2)))
    difference = first_estimate - second_estimate
    statistic = _safe_ratio(difference, standard_error)
    p_value = _p_value(statistic, _normal_cdf, alternative)
    unpooled_error = np.sqrt(first_estimate * (1 - first_estimate) / np.asarray(trials1)
                             + second_estimate * (1 - second_estimate) / np.asarray(trials2))
    probability = (1 + confidence) / 2 if alternative == "two-sided" else confidence
    interval = _interval(difference, unpooled_error, confidence,
                         NormalDist().inv_cdf(probability), alternative,
                         "Difference in proportions interval")
    return TestResult(_scalar(statistic), p_value, "Two-sample proportion z test",
                      alternative, None, _scalar(difference), 0.0, _scalar(standard_error),
                      interval, _scalar(difference))


def _regularized_gamma_q(a, x):
    if x < 0 or a <= 0:
        raise ValueError("gamma arguments must be positive")
    if x == 0:
        return 1.0
    if x < a + 1:
        term = total = 1 / a
        ap = a
        for _ in range(1, 1001):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        p = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return max(0.0, 1 - p)
    b, c, d, result = x + 1 - a, 1 / 1e-300, 1 / (x + 1 - a), 1 / (x + 1 - a)
    for index in range(1, 1001):
        an = -index * (index - a)
        b += 2
        d = an * d + b
        d = d if abs(d) > 1e-300 else 1e-300
        c = b + an / c
        c = c if abs(c) > 1e-300 else 1e-300
        d = 1 / d
        change = d * c
        result *= change
        if abs(change - 1) < 1e-14:
            break
    return result * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi_square_test(observed, expected=None):
    """Perform a chi-square goodness-of-fit test."""
    observed = _numeric(observed, "observed")
    if (observed.ndim != 1 or observed.size < 2 or np.any(~np.isfinite(observed))
            or np.any(observed < 0)):
        raise ValueError("observed must be a non-negative vector with at least two cells")
    total = float(np.sum(observed))
    if total <= 0:
        raise ValueError("observed counts must have a positive total")
    if expected is None:
        expected = np.full(observed.shape, total / observed.size)
    else:
        expected = _numeric(expected, "expected")
        if (expected.shape != observed.shape or np.any(~np.isfinite(expected))
                or np.any(expected <= 0)):
            raise ValueError("expected must match observed and contain positive values")
        expected_total = float(np.sum(expected))
        if math.isclose(expected_total, 1, abs_tol=1e-12, rel_tol=0):
            expected = expected * total
        elif not math.isclose(expected_total, total, abs_tol=1e-9, rel_tol=1e-9):
            raise ValueError("expected counts must total observed counts (or be probabilities)")
    statistic = float(np.sum((observed - expected) ** 2 / expected))
    degrees = observed.size - 1
    p_value = _regularized_gamma_q(degrees / 2, statistic / 2)
    effect = math.sqrt(statistic / total)
    return TestResult(statistic, p_value, "Chi-square goodness-of-fit test",
                      degrees_of_freedom=degrees, effect_size=effect,
                      details={"observed": observed.copy(), "expected": expected.copy()})


def chi_square_independence(table, *, correction=False):
    """Test independence in a contingency table and report Cramer's V."""
    if isinstance(table, ContingencyTable):
        table = table.counts
    observed = _numeric(table, "table")
    if (observed.ndim != 2 or min(observed.shape) < 2
            or np.any(~np.isfinite(observed)) or np.any(observed < 0)):
        raise ValueError("table must be a non-negative 2D array with at least two rows and columns")
    if not isinstance(correction, bool):
        raise TypeError("correction must be Boolean")
    total = float(np.sum(observed))
    row_totals, column_totals = np.sum(observed, axis=1), np.sum(observed, axis=0)
    if total <= 0 or np.any(row_totals == 0) or np.any(column_totals == 0):
        raise ValueError("every row and column must have a positive total")
    expected = np.outer(row_totals, column_totals) / total
    difference = np.abs(observed - expected)
    if correction and observed.shape == (2, 2):
        difference = np.maximum(0, difference - 0.5)
    statistic = float(np.sum(difference ** 2 / expected))
    degrees = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    p_value = _regularized_gamma_q(degrees / 2, statistic / 2)
    effect = math.sqrt(statistic / (total * min(observed.shape[0] - 1,
                                                 observed.shape[1] - 1)))
    return TestResult(statistic, p_value, "Chi-square test of independence",
                      degrees_of_freedom=degrees, effect_size=effect,
                      details={"observed": observed.copy(), "expected": expected})


def one_way_anova(*groups, nan_policy="propagate"):
    """Compare three or more group means with a one-way ANOVA."""
    if len(groups) == 1 and not isinstance(groups[0], np.ndarray):
        candidate = tuple(groups[0])
        if candidate and all(hasattr(group, "__iter__") for group in candidate):
            groups = candidate
    if len(groups) < 2:
        raise ValueError("provide at least two groups")
    cleaned = []
    for index, group in enumerate(groups):
        values = _numeric(group, f"group {index}").reshape(-1)
        if np.any(np.isinf(values)):
            raise ValueError(f"group {index} must not contain infinite values")
        policy = _policy(nan_policy)
        if np.any(np.isnan(values)):
            if policy == "raise":
                raise ValueError(f"group {index} contains NaN")
            if policy == "propagate":
                return TestResult(math.nan, math.nan, "One-way ANOVA",
                                  degrees_of_freedom=(len(groups) - 1, math.nan))
            values = values[~np.isnan(values)]
        if values.size < 2:
            raise ValueError("each group needs at least two observations")
        cleaned.append(values)
    sizes = np.asarray([group.size for group in cleaned])
    means = np.asarray([np.mean(group) for group in cleaned])
    grand_mean = sum(np.sum(group) for group in cleaned) / np.sum(sizes)
    between = float(np.sum(sizes * (means - grand_mean) ** 2))
    within = float(sum(np.sum((group - mean) ** 2) for group, mean in zip(cleaned, means)))
    df_between, df_within = len(cleaned) - 1, int(np.sum(sizes) - len(cleaned))
    if within == 0:
        statistic = 0.0 if between == 0 else math.inf
    else:
        statistic = (between / df_between) / (within / df_within)
    if statistic == math.inf:
        p_value = 0.0
    else:
        x = df_between * statistic / (df_between * statistic + df_within)
        p_value = 1 - _regularized_beta(x, df_between / 2, df_within / 2)
    total_sum = between + within
    effect = between / total_sum if total_sum else 0.0
    return TestResult(statistic, p_value, "One-way ANOVA", degrees_of_freedom=(
        df_between, df_within), effect_size=effect,
        details={"group_sizes": sizes, "group_means": means, "grand_mean": grand_mean})


anova = one_way_anova


def correlation_test(x, y, *, method="pearson", alternative="two-sided",
                     nan_policy="propagate"):
    """Test a Pearson or Spearman correlation using its standard t approximation."""
    from kiwicalc.probability.descriptive import pearson_correlation, spearman_correlation

    alternative = _alternative(alternative)
    first, second = _numeric(x, "x").reshape(-1), _numeric(y, "y").reshape(-1)
    if np.any(np.isinf(first)) or np.any(np.isinf(second)):
        raise ValueError("x and y must not contain infinite values")
    if first.shape != second.shape:
        raise ValueError("x and y must have the same shape")
    policy = _policy(nan_policy)
    missing = np.isnan(first) | np.isnan(second)
    if np.any(missing):
        if policy == "raise":
            raise ValueError("x and y contain NaN")
        if policy == "omit":
            first, second = first[~missing], second[~missing]
    if first.size < 3:
        raise ValueError("correlation testing needs at least three paired observations")
    if not isinstance(method, str):
        raise TypeError("method must be text")
    method = method.lower()
    if method == "pearson":
        estimate = pearson_correlation(first, second, nan_policy=policy)
        label = "Pearson correlation t test"
    elif method == "spearman":
        estimate = spearman_correlation(first, second, nan_policy=policy)
        label = "Spearman correlation t approximation"
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")
    degrees = first.size - 2
    statistic = estimate * math.sqrt(degrees / max(1e-300, 1 - estimate ** 2))
    p_value = _p_value(statistic, lambda value: _student_cdf(value, degrees), alternative)
    return TestResult(statistic, p_value, label, alternative, degrees, estimate, 0.0,
                      effect_size=estimate)


__all__ = [
    "ConfidenceInterval", "TestResult", "mean_confidence_interval",
    "confidence_interval", "mean_test", "one_sample_t_test", "compare_means",
    "two_sample_t_test", "proportion_confidence_interval", "proportion_test",
    "compare_proportions", "chi_square_test", "chi_square_independence",
    "one_way_anova", "anova", "correlation_test",
]
