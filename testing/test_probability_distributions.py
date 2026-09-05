import math

import numpy as np
import pytest

import kiwicalc as kw


def assert_reproducible(distribution):
    first = distribution.sample(12, random_state=123)
    second = distribution.rvs(12, random_state=123)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (12,)


def test_bernoulli_distribution():
    distribution = kw.Bernoulli(0.3)
    assert distribution.support == (0, 1)
    assert distribution.mean == 0.3
    assert distribution.variance == pytest.approx(0.21)
    assert distribution.std == pytest.approx(math.sqrt(0.21))
    np.testing.assert_allclose(distribution.pmf([-1, 0, 1, 2]), [0, 0.7, 0.3, 0])
    np.testing.assert_allclose(distribution.cdf([-1, 0, 1]), [0, 0.7, 1])
    np.testing.assert_array_equal(distribution.ppf([0, 0.7, 0.8, 1]), [0, 0, 1, 1])
    assert repr(distribution) == 'Bernoulli(p=0.3)'
    assert_reproducible(distribution)


def test_binomial_distribution():
    distribution = kw.Binomial(5, 0.4)
    assert distribution.support == (0, 5)
    assert distribution.mean == 2
    assert distribution.variance == pytest.approx(1.2)
    assert distribution.pmf(2) == pytest.approx(0.3456)
    assert distribution.pmf(2.5) == 0
    assert distribution.pmf(-1) == 0
    assert distribution.cdf(-1) == 0
    assert distribution.cdf(5) == 1
    assert distribution.cdf(2) == pytest.approx(sum(distribution.pmf(k) for k in range(3)))
    assert distribution.ppf(distribution.cdf(2)) == 2
    assert 'n=5' in repr(distribution)
    assert_reproducible(distribution)


def test_binomial_boundary_probabilities():
    always = kw.Binomial(3, 1)
    never = kw.Binomial(3, 0)
    assert always.pmf(3) == 1
    assert never.pmf(0) == 1
    assert always.ppf(0) == 0
    assert always.ppf(1) == 3


def test_geometric_distribution():
    distribution = kw.Geometric(0.25)
    assert distribution.support == (1, math.inf)
    assert distribution.mean == 4
    assert distribution.variance == 12
    assert distribution.pmf(1) == 0.25
    assert distribution.pmf(3) == pytest.approx(0.75 ** 2 * 0.25)
    assert distribution.pmf(0) == 0
    assert distribution.cdf(3) == pytest.approx(1 - 0.75 ** 3)
    assert distribution.cdf(math.inf) == 1
    assert distribution.ppf(0) == 1
    assert distribution.ppf(1) == math.inf
    assert distribution.cdf(distribution.ppf(0.8)) >= 0.8
    assert_reproducible(distribution)


def test_hypergeometric_distribution():
    distribution = kw.Hypergeometric(population=20, successes=7, draws=5)
    assert distribution.support == (0, 5)
    expected = math.comb(7, 2) * math.comb(13, 3) / math.comb(20, 5)
    assert distribution.pmf(2) == pytest.approx(expected)
    assert distribution.pmf(6) == 0
    assert distribution.mean == pytest.approx(1.75)
    assert distribution.cdf(-1) == 0
    assert distribution.cdf(5) == 1
    assert distribution.ppf(0) == 0
    assert 'population=20' in repr(distribution)
    assert_reproducible(distribution)


def test_degenerate_hypergeometric_variance():
    distribution = kw.Hypergeometric(1, 1, 1)
    assert distribution.variance == 0
    assert distribution.pmf(1) == 1


def test_poisson_distribution():
    distribution = kw.Poisson(3)
    assert distribution.support == (0, math.inf)
    assert distribution.mean == distribution.variance == 3
    assert distribution.pmf(2) == pytest.approx(math.exp(-3) * 9 / 2)
    assert distribution.pmf(-1) == 0
    assert distribution.cdf(-1) == 0
    assert distribution.cdf(2) == pytest.approx(sum(distribution.pmf(k) for k in range(3)))
    assert distribution.cdf(math.inf) == 1
    assert distribution.ppf(0) == 0
    assert distribution.ppf(1) == math.inf
    assert distribution.cdf(distribution.ppf(0.9)) >= 0.9
    assert repr(distribution) == 'Poisson(rate=3.0)'
    assert_reproducible(distribution)


def test_poisson_large_rate_does_not_underflow_around_mean():
    distribution = kw.Poisson(1000)
    assert 0.4 < distribution.cdf(1000) < 0.6
    assert 900 < distribution.ppf(0.5) < 1100


def test_discrete_uniform_distribution():
    distribution = kw.DiscreteUniform(-2, 2)
    assert distribution.support == (-2, 2)
    assert distribution.mean == 0
    assert distribution.variance == 2
    np.testing.assert_allclose(distribution.pmf([-3, -2, 0, 2, 3]), [0, 0.2, 0.2, 0.2, 0])
    assert distribution.cdf(-3) == 0
    assert distribution.cdf(0) == pytest.approx(0.6)
    assert distribution.cdf(2) == 1
    np.testing.assert_array_equal(distribution.ppf([0, 0.2, 0.21, 1]), [-2, -2, -1, 2])
    assert_reproducible(distribution)


def test_categorical_distribution():
    distribution = kw.Categorical({'red': 0.2, 'blue': 0.5, 'green': 0.3})
    assert distribution.support == ('red', 'blue', 'green')
    np.testing.assert_allclose(distribution.pmf(['green', 'missing', 'red']), [0.3, 0, 0.2])
    np.testing.assert_allclose(distribution.cdf(['red', 'blue']), [0.2, 0.7])
    assert distribution.ppf(0.6) == 'blue'
    np.testing.assert_array_equal(distribution.ppf([0, 0.9]), ['red', 'green'])
    assert distribution.probability_between('red', 'blue') == pytest.approx(0.7)
    assert distribution.probability_between(
        'red', 'green', inclusive='neither'
    ) == pytest.approx(0.5)
    assert 'red' in repr(distribution)
    with pytest.raises(TypeError, match='numeric'):
        _ = distribution.mean
    assert_reproducible(distribution)


def test_numeric_categorical_moments_and_normalization():
    distribution = kw.Categorical([2, 3], values=[10, 20], normalize=True)
    assert distribution.probabilities == pytest.approx((0.4, 0.6))
    assert distribution.mean == 16
    assert distribution.variance == 24


def test_continuous_uniform_distribution():
    distribution = kw.Uniform(-1, 3)
    assert isinstance(distribution, kw.ContinuousUniform)
    assert distribution.support == (-1, 3)
    assert distribution.mean == 1
    assert distribution.variance == pytest.approx(4 / 3)
    np.testing.assert_allclose(distribution.pdf([-2, -1, 1, 3, 4]), [0, 0.25, 0.25, 0.25, 0])
    np.testing.assert_allclose(distribution.cdf([-2, -1, 1, 3, 4]), [0, 0, 0.5, 1, 1])
    np.testing.assert_allclose(distribution.ppf([0, 0.5, 1]), [-1, 1, 3])
    assert distribution.probability_between(0, 2) == pytest.approx(0.5)
    assert distribution.probability(0, 2) == pytest.approx(0.5)
    assert_reproducible(distribution)


def test_normal_distribution():
    distribution = kw.Normal(mean=10, std=2)
    assert isinstance(distribution, kw.Gaussian)
    assert distribution.support == (-math.inf, math.inf)
    assert distribution.mean == 10
    assert distribution.variance == 4
    assert distribution.standard_deviation == 2
    assert distribution.pdf(10) == pytest.approx(1 / (2 * math.sqrt(2 * math.pi)))
    assert distribution.cdf(10) == pytest.approx(0.5)
    assert distribution.ppf(0.5) == pytest.approx(10)
    assert distribution.ppf(0) == -math.inf
    assert distribution.ppf(1) == math.inf
    assert distribution.sf(10) == pytest.approx(0.5)
    np.testing.assert_allclose(distribution.z_score([8, 10, 12]), [-1, 0, 1])
    assert distribution.probability_between(8, 12) == pytest.approx(0.682689492, rel=1e-7)
    assert 'mean=10.0' in repr(distribution)
    assert_reproducible(distribution)


def test_exponential_distribution():
    distribution = kw.Exponential(rate=2)
    assert distribution.support == (0, math.inf)
    assert distribution.scale == 0.5
    assert distribution.mean == 0.5
    assert distribution.variance == 0.25
    assert distribution.pdf(-1) == 0
    assert distribution.pdf(0) == 2
    assert distribution.cdf(-1) == 0
    assert distribution.cdf(1) == pytest.approx(1 - math.exp(-2))
    assert distribution.ppf(distribution.cdf(1)) == pytest.approx(1)
    assert distribution.ppf(1) == math.inf
    assert kw.Exponential.from_scale(4).rate == 0.25
    assert repr(distribution) == 'Exponential(rate=2.0)'
    assert_reproducible(distribution)


def test_vectorization_and_nan_propagation():
    distributions = [kw.Binomial(3, 0.5), kw.Geometric(0.5), kw.Poisson(2), kw.Normal()]
    for distribution in distributions:
        values = distribution.cdf(np.array([0.0, 1.0, np.nan]))
        assert values.shape == (3,)
        assert math.isnan(values[-1])


def test_log_probability_helpers():
    discrete = kw.Bernoulli(0.25)
    assert discrete.logpmf(1) == pytest.approx(math.log(0.25))
    assert discrete.logpmf(2) == -math.inf
    continuous = kw.Normal()
    assert continuous.logpdf(0) == pytest.approx(math.log(continuous.pdf(0)))


def test_discrete_interval_probability_endpoint_options():
    distribution = kw.DiscreteUniform(1, 5)
    assert distribution.probability_between(2, 4) == pytest.approx(3 / 5)
    assert distribution.probability_between(2, 4, inclusive='left') == pytest.approx(2 / 5)
    assert distribution.probability_between(2, 4, inclusive='right') == pytest.approx(2 / 5)
    assert distribution.probability_between(2, 4, inclusive='neither') == pytest.approx(1 / 5)
    assert distribution.probability(3) == pytest.approx(1 / 5)


def test_distribution_survival_and_quantile_aliases():
    distribution = kw.Binomial(4, 0.5)
    assert distribution.sf(2) == pytest.approx(1 - distribution.cdf(2))
    assert distribution.quantile(0.5) == distribution.ppf(0.5)


@pytest.mark.parametrize(
    ('constructor', 'error'),
    [
        (lambda: kw.Bernoulli(-0.1), ValueError),
        (lambda: kw.Binomial(-1, 0.5), ValueError),
        (lambda: kw.Binomial(2.5, 0.5), TypeError),
        (lambda: kw.Geometric(0), ValueError),
        (lambda: kw.Hypergeometric(5, 6, 2), ValueError),
        (lambda: kw.Hypergeometric(5, 2, 6), ValueError),
        (lambda: kw.Poisson(0), ValueError),
        (lambda: kw.DiscreteUniform(2, 1), ValueError),
        (lambda: kw.Uniform(1, 1), ValueError),
        (lambda: kw.Normal(std=0), ValueError),
        (lambda: kw.Exponential(-1), ValueError),
    ],
)
def test_invalid_distribution_parameters(constructor, error):
    with pytest.raises(error):
        constructor()


def test_categorical_validation():
    with pytest.raises(ValueError, match='empty'):
        kw.Categorical([])
    with pytest.raises(ValueError, match='same length'):
        kw.Categorical([1], values=['a', 'b'])
    with pytest.raises(ValueError, match='unique'):
        kw.Categorical([0.5, 0.5], values=['a', 'a'])
    with pytest.raises(ValueError, match='negative'):
        kw.Categorical([1.1, -0.1])
    with pytest.raises(ValueError, match='sum to one'):
        kw.Categorical([0.2, 0.2])
    with pytest.raises(ValueError, match='do not pass'):
        kw.Categorical({'a': 1}, values=['a'])
    with pytest.raises(ValueError, match='ordered support'):
        kw.Categorical({'a': 1}).cdf('b')
    with pytest.raises(TypeError, match='Boolean'):
        kw.Categorical([1], normalize=1)
    with pytest.raises(ValueError, match='positive total'):
        kw.Categorical([0, 0])
    with pytest.raises(TypeError, match='hashable'):
        kw.Categorical([0.5, 0.5], values=[[1], [2]])


def test_categorical_interval_endpoint_options_and_validation():
    distribution = kw.Categorical({'a': 0.2, 'b': 0.3, 'c': 0.5})
    assert distribution.probability_between('a', 'c', inclusive='left') == 0.5
    assert distribution.probability_between('a', 'c', inclusive='right') == 0.8
    with pytest.raises(ValueError, match='inclusive'):
        distribution.probability_between('a', 'c', inclusive='yes')
    with pytest.raises(ValueError, match='endpoints'):
        distribution.probability_between('missing', 'c')
    with pytest.raises(ValueError, match='follow'):
        distribution.probability_between('c', 'a')


def test_quantile_validation():
    for value in (-0.1, 1.1, np.nan, math.inf):
        with pytest.raises(ValueError, match='quantiles'):
            kw.Normal().ppf(value)


def test_interval_validation():
    with pytest.raises(ValueError, match='inclusive'):
        kw.Binomial(3, 0.5).probability_between(1, 2, inclusive='yes')
    with pytest.raises(ValueError, match='lower'):
        kw.Binomial(3, 0.5).probability_between(2, 1)
    with pytest.raises(ValueError, match='lower'):
        kw.Normal().probability_between(2, 1)


def test_sampling_size_and_random_state_validation():
    distribution = kw.Normal()
    assert distribution.sample((2, 3), random_state=1).shape == (2, 3)
    with pytest.raises(ValueError, match='size'):
        distribution.sample(-1)
    with pytest.raises(TypeError, match='size'):
        distribution.sample([2, 3])
    with pytest.raises(TypeError, match='random_state'):
        distribution.sample(2, random_state='seed')


def test_distribution_exports():
    for name in (
        'Distribution', 'Bernoulli', 'Binomial', 'Geometric', 'Hypergeometric',
        'Poisson', 'DiscreteUniform', 'Categorical', 'Uniform', 'Normal',
        'Exponential',
    ):
        assert hasattr(kw, name)
