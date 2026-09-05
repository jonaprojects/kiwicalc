"""Independent numerical validation against NumPy, SciPy, and Hypothesis.

These tests use optional development dependencies and never affect KiwiCalc's
runtime dependency footprint.
"""

import numpy as np
import pytest

scipy = pytest.importorskip('scipy')
hypothesis = pytest.importorskip('hypothesis')

from hypothesis import given, settings, strategies as st
from scipy import stats

import kiwicalc as kw


pytestmark = pytest.mark.validation
FINITE_FLOATS = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False,
)


@settings(max_examples=60, deadline=None)
@given(st.lists(FINITE_FLOATS, min_size=2, max_size=50), st.integers(0, 1))
def test_descriptive_location_and_spread_match_numpy(values, ddof):
    array = np.asarray(values, dtype=float)

    assert kw.mean(array) == pytest.approx(np.mean(array), rel=2e-14, abs=1e-12)
    assert kw.median(array) == pytest.approx(np.median(array), rel=2e-14, abs=1e-12)
    assert kw.variance(array, ddof=ddof) == pytest.approx(
        np.var(array, ddof=ddof), rel=5e-13, abs=1e-10,
    )
    assert kw.standard_deviation(array, ddof=ddof) == pytest.approx(
        np.std(array, ddof=ddof), rel=5e-13, abs=1e-10,
    )


@settings(max_examples=50, deadline=None)
@given(
    st.lists(FINITE_FLOATS, min_size=2, max_size=50),
    st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
)
def test_quantiles_match_numpy(values, probability):
    assert kw.quantile(values, probability) == pytest.approx(
        np.quantile(values, probability), rel=2e-13, abs=1e-10,
    )


def test_shape_statistics_match_scipy():
    values = np.array([-3.1, -1.2, -0.4, 0.0, 0.7, 1.1, 2.8, 5.4])

    assert kw.skewness(values) == pytest.approx(stats.skew(values, bias=True), abs=2e-12)
    assert kw.kurtosis(values) == pytest.approx(
        stats.kurtosis(values, fisher=True, bias=True), abs=2e-12,
    )


@pytest.mark.parametrize('distribution, reference, values', [
    (kw.Bernoulli(0.37), stats.bernoulli(0.37), np.arange(-1, 3)),
    (kw.Binomial(17, 0.31), stats.binom(17, 0.31), np.arange(-1, 19)),
    (kw.Geometric(0.23), stats.geom(0.23), np.arange(0, 22)),
    (kw.Hypergeometric(31, 12, 9), stats.hypergeom(31, 12, 9), np.arange(-1, 11)),
    (kw.Poisson(7.4), stats.poisson(7.4), np.arange(-1, 25)),
])
def test_discrete_distributions_match_scipy(distribution, reference, values):
    np.testing.assert_allclose(distribution.pmf(values), reference.pmf(values), rtol=2e-13,
                               atol=2e-15)
    np.testing.assert_allclose(distribution.cdf(values), reference.cdf(values), rtol=2e-13,
                               atol=2e-15)
    probabilities = np.array([1e-6, 0.01, 0.2, 0.5, 0.9, 0.999999])
    np.testing.assert_allclose(distribution.ppf(probabilities), reference.ppf(probabilities),
                               rtol=0, atol=0)


@pytest.mark.parametrize('distribution, reference, values', [
    (kw.Uniform(-2.5, 4.2), stats.uniform(loc=-2.5, scale=6.7),
     np.linspace(-3, 5, 31)),
    (kw.Normal(1.3, 2.7), stats.norm(loc=1.3, scale=2.7),
     np.linspace(-8, 11, 31)),
    (kw.Exponential(1.8), stats.expon(scale=1 / 1.8),
     np.linspace(-1, 6, 31)),
])
def test_continuous_distributions_match_scipy(distribution, reference, values):
    np.testing.assert_allclose(distribution.pdf(values), reference.pdf(values), rtol=3e-13,
                               atol=2e-15)
    np.testing.assert_allclose(distribution.cdf(values), reference.cdf(values), rtol=3e-13,
                               atol=2e-15)
    probabilities = np.array([1e-8, 0.001, 0.2, 0.5, 0.9, 0.999999])
    np.testing.assert_allclose(distribution.ppf(probabilities), reference.ppf(probabilities),
                               rtol=2e-12, atol=2e-12)


def test_multivariate_normal_matches_scipy():
    mean = np.array([0.7, -1.2, 2.4])
    covariance = np.array([[2.0, 0.4, -0.2], [0.4, 1.5, 0.3], [-0.2, 0.3, 1.2]])
    points = np.array([[0.0, 0.0, 0.0], mean, [1.2, -3.1, 1.4]])

    actual = kw.MultivariateNormal(mean, covariance)
    expected = stats.multivariate_normal(mean=mean, cov=covariance)
    np.testing.assert_allclose(actual.pdf(points), expected.pdf(points), rtol=2e-13)
    np.testing.assert_allclose(actual.logpdf(points), expected.logpdf(points), rtol=2e-13)


def test_dirichlet_matches_scipy():
    alpha = np.array([1.7, 2.4, 3.2])
    points = np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]])
    actual = kw.Dirichlet(alpha)
    expected = stats.dirichlet(alpha)

    np.testing.assert_allclose(actual.pdf(points), expected.pdf(points.T), rtol=2e-13)


def test_one_sample_and_two_sample_tests_match_scipy():
    first = np.array([3.2, 4.7, 5.1, 2.9, 6.4, 5.8, 4.1])
    second = np.array([2.1, 3.7, 4.0, 2.6, 4.9, 5.0, 3.3, 3.8])

    actual_one = kw.mean_test(first, expected=4.0)
    expected_one = stats.ttest_1samp(first, popmean=4.0)
    assert actual_one.statistic == pytest.approx(expected_one.statistic, rel=2e-12)
    assert actual_one.p_value == pytest.approx(expected_one.pvalue, rel=2e-11)

    for equal_variance in (False, True):
        actual_two = kw.compare_means(first, second, equal_variance=equal_variance)
        expected_two = stats.ttest_ind(first, second, equal_var=equal_variance)
        assert actual_two.statistic == pytest.approx(expected_two.statistic, rel=2e-12)
        assert actual_two.p_value == pytest.approx(expected_two.pvalue, rel=2e-11)


def test_paired_test_matches_scipy():
    before = np.array([11.0, 10.2, 9.7, 13.1, 12.4, 8.9])
    after = np.array([10.4, 9.8, 9.9, 12.0, 11.7, 8.5])
    actual = kw.compare_means(before, after, paired=True)
    expected = stats.ttest_rel(before, after)

    assert actual.statistic == pytest.approx(expected.statistic, rel=2e-12)
    assert actual.p_value == pytest.approx(expected.pvalue, rel=2e-11)


def test_chi_square_tests_match_scipy():
    observed = np.array([16, 22, 18, 24], dtype=float)
    actual_fit = kw.chi_square_test(observed)
    expected_fit = stats.chisquare(observed)
    assert actual_fit.statistic == pytest.approx(expected_fit.statistic, rel=2e-13)
    assert actual_fit.p_value == pytest.approx(expected_fit.pvalue, rel=2e-11)

    table = np.array([[12, 7, 9], [5, 14, 11], [8, 10, 15]], dtype=float)
    actual_independence = kw.chi_square_independence(table, correction=False)
    expected_independence = stats.chi2_contingency(table, correction=False)
    assert actual_independence.statistic == pytest.approx(expected_independence.statistic,
                                                          rel=2e-13)
    assert actual_independence.p_value == pytest.approx(expected_independence.pvalue,
                                                        rel=2e-11)


def test_anova_and_correlations_match_scipy():
    groups = ([2.1, 2.7, 3.4, 2.9], [4.0, 3.7, 4.6, 4.2], [5.2, 4.8, 5.7, 5.1])
    actual_anova = kw.one_way_anova(*groups)
    expected_anova = stats.f_oneway(*groups)
    assert actual_anova.statistic == pytest.approx(expected_anova.statistic, rel=2e-12)
    assert actual_anova.p_value == pytest.approx(expected_anova.pvalue, rel=2e-11)

    x = np.array([1, 2, 4, 7, 8, 10, 13], dtype=float)
    y = np.array([3, 1, 5, 6, 9, 8, 12], dtype=float)
    for method, oracle in [('pearson', stats.pearsonr), ('spearman', stats.spearmanr)]:
        actual = kw.correlation_test(x, y, method=method)
        expected = oracle(x, y)
        assert actual.estimate == pytest.approx(expected.statistic, rel=2e-12)
        assert actual.p_value == pytest.approx(expected.pvalue, rel=3e-9)


def test_student_confidence_interval_matches_scipy():
    values = np.array([5.2, 4.8, 6.1, 5.9, 4.7, 5.5, 6.3])
    actual = kw.mean_confidence_interval(values, confidence=0.95)
    expected = stats.t.interval(
        0.95, values.size - 1, loc=np.mean(values), scale=stats.sem(values),
    )

    assert actual.lower == pytest.approx(expected[0], rel=2e-12)
    assert actual.upper == pytest.approx(expected[1], rel=2e-12)
