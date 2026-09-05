"""Mathematical contracts that every stable probability API must preserve."""

import math

import numpy as np
import pytest

import kiwicalc as kw


def _trapezoid(values, points):
    """Integrate compatibly across NumPy versions supported by KiwiCalc."""
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(values, points)
    return np.trapz(values, points)


def test_inference_result_is_not_collected_as_a_pytest_test_class():
    assert kw.TestResult.__test__ is False


@pytest.mark.parametrize('distribution, bounds', [
    (kw.Uniform(-3, 5), (-3, 5)),
    (kw.Normal(2, 3), (-10, 14)),
    (kw.Exponential(1.7), (0, 10)),
    (kw.distribution('6x(1-x)', between=(0, 1)), (0, 1)),
])
def test_continuous_distribution_shape_invariants(distribution, bounds):
    values = np.linspace(*bounds, 2001)
    density = np.asarray(distribution.pdf(values))
    cumulative = np.asarray(distribution.cdf(values))

    assert np.all(np.isfinite(density))
    assert np.all(density >= 0)
    assert np.all(np.diff(cumulative) >= -1e-12)
    assert np.all((cumulative >= 0) & (cumulative <= 1))
    assert cumulative[0] >= 0
    assert cumulative[-1] <= 1
    assert _trapezoid(density, values) == pytest.approx(
        cumulative[-1] - cumulative[0], abs=3e-4,
    )


@pytest.mark.parametrize('distribution', [
    kw.Uniform(-3, 5),
    kw.Normal(2, 3),
    kw.Exponential(1.7),
    kw.distribution('6x(1-x)', between=(0, 1)),
])
def test_continuous_quantile_inverse_contract(distribution):
    probabilities = np.array([1e-4, 0.01, 0.25, 0.5, 0.9, 0.999])
    quantiles = np.asarray(distribution.ppf(probabilities))

    assert np.all(np.diff(quantiles) >= 0)
    np.testing.assert_allclose(
        distribution.cdf(quantiles), probabilities, rtol=2e-7, atol=2e-8,
    )


@pytest.mark.parametrize('distribution, values', [
    (kw.Bernoulli(0.37), np.arange(0, 2)),
    (kw.Binomial(12, 0.31), np.arange(0, 13)),
    (kw.Hypergeometric(30, 11, 8), np.arange(0, 9)),
    (kw.DiscreteUniform(-4, 7), np.arange(-4, 8)),
    (kw.distribution('x^2', over=[1, 2, 4, 8]), np.array([1, 2, 4, 8])),
])
def test_finite_discrete_distribution_invariants(distribution, values):
    probabilities = np.asarray(distribution.pmf(values))
    cumulative = np.asarray(distribution.cdf(values))

    assert np.all(probabilities >= 0)
    assert np.sum(probabilities) == pytest.approx(1, abs=1e-12)
    assert np.all(np.diff(cumulative) >= -1e-12)
    assert cumulative[-1] == pytest.approx(1)


@pytest.mark.parametrize('distribution', [kw.Geometric(0.17), kw.Poisson(8.4)])
def test_infinite_discrete_distributions_capture_requested_tail(distribution):
    upper = int(distribution.ppf(0.999999))
    lower = int(distribution.support[0])
    values = np.arange(lower, upper + 1)

    total = float(np.sum(distribution.pmf(values)))
    assert 0.999998 <= total <= 1.000000000001
    assert distribution.cdf(upper) == pytest.approx(total, abs=2e-12)


@pytest.mark.parametrize('distribution', [
    kw.Bernoulli(0.37), kw.Binomial(12, 0.31), kw.Geometric(0.17),
    kw.Hypergeometric(30, 11, 8), kw.Poisson(8.4), kw.DiscreteUniform(-4, 7),
])
def test_discrete_quantiles_are_smallest_values_reaching_probability(distribution):
    probabilities = np.array([0.01, 0.2, 0.5, 0.9, 0.999])
    values = np.asarray(distribution.ppf(probabilities), dtype=float)

    assert np.all(distribution.cdf(values) >= probabilities - 1e-14)
    assert np.all(distribution.cdf(values - 1) < probabilities + 1e-14)


def test_descriptive_location_and_scale_invariants():
    values = np.array([-5.0, -1.0, 0.5, 2.0, 9.0])
    shifted = values + 17
    scaled = values * -3

    assert kw.mean(shifted) == pytest.approx(kw.mean(values) + 17)
    assert kw.median(shifted) == pytest.approx(kw.median(values) + 17)
    assert kw.variance(shifted, ddof=1) == pytest.approx(kw.variance(values, ddof=1))
    assert kw.variance(scaled, ddof=1) == pytest.approx(9 * kw.variance(values, ddof=1))
    assert kw.standard_deviation(scaled, ddof=1) == pytest.approx(
        3 * kw.standard_deviation(values, ddof=1)
    )


def test_covariance_and_correlation_matrix_invariants():
    data = np.array([[1, 3, -1], [2, 1, 0], [4, 5, 2], [7, 2, 4]], dtype=float)
    covariance = kw.covariance(data)
    correlation = kw.pearson_correlation(data)

    np.testing.assert_allclose(covariance, covariance.T, atol=1e-14)
    np.testing.assert_allclose(correlation, correlation.T, atol=1e-14)
    np.testing.assert_allclose(np.diag(correlation), 1, atol=1e-14)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-12
    assert np.all(np.abs(correlation) <= 1 + 1e-14)


def test_multivariate_distribution_moment_and_sampling_invariants():
    covariance = np.array([[2.0, 0.7], [0.7, 1.5]])
    distribution = kw.MultivariateNormal([1, -2], covariance)
    samples = distribution.sample(60000, random_state=814)

    np.testing.assert_allclose(np.mean(samples, axis=0), distribution.mean, atol=0.025)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), covariance, atol=0.04)
    assert np.min(np.linalg.eigvalsh(distribution.covariance)) > 0
    assert distribution.pdf([1, -2]) > distribution.pdf([5, 5])


def test_inference_results_obey_probability_and_interval_contracts():
    results = [
        kw.mean_test([1, 2, 3, 4, 5], expected=2),
        kw.compare_means([1, 2, 3], [2, 4, 6]),
        kw.proportion_test(17, 30, expected=0.5),
        kw.compare_proportions(17, 30, 11, 30),
        kw.chi_square_test([12, 18, 20]),
        kw.one_way_anova([1, 2, 3], [3, 4, 5], [7, 8, 9]),
        kw.correlation_test([1, 2, 3, 4], [1, 3, 2, 5]),
    ]

    for result in results:
        assert math.isfinite(float(result.statistic))
        assert 0 <= float(result.p_value) <= 1
        if result.confidence_interval is not None:
            interval = result.confidence_interval
            assert interval.lower <= interval.estimate <= interval.upper


def test_seeded_sampling_contract_is_repeatable_and_shape_preserving():
    distributions = [
        kw.Normal(), kw.Poisson(3), kw.Dirichlet([2, 3, 4]),
        kw.distribution('2x', between=(0, 1)),
        kw.distribution('x', over=[1, 2, 3]),
    ]
    for distribution in distributions:
        first = distribution.sample((3, 4), random_state=917)
        second = distribution.rvs((3, 4), random_state=917)
        np.testing.assert_array_equal(first, second)
