import math

import numpy as np
import pytest

import kiwicalc as kw


def test_joint_discrete_distribution_core_statistics():
    distribution = kw.JointDiscreteDistribution({
        (0, 0): 0.4,
        (0, 1): 0.1,
        (1, 0): 0.1,
        (1, 1): 0.4,
    })
    assert distribution.dimension == 2
    assert distribution.mean == pytest.approx([0.5, 0.5])
    np.testing.assert_allclose(distribution.variance, [0.25, 0.25])
    np.testing.assert_allclose(distribution.covariance, [[0.25, 0.15], [0.15, 0.25]])
    np.testing.assert_allclose(distribution.correlation, [[1, 0.6], [0.6, 1]])
    assert distribution.pmf([1, 1]) == 0.4
    assert distribution.probability([1, 0]) == 0.1
    np.testing.assert_allclose(distribution.pmf([[0, 0], [2, 2]]), [0.4, 0])
    assert distribution.logpmf([2, 2]) == -math.inf
    assert distribution.cdf([0, 1]) == pytest.approx(0.5)
    assert 'JointDiscreteDistribution' in repr(distribution)


def test_joint_discrete_marginal_conditioning_and_events():
    distribution = kw.JointDiscreteDistribution({
        ('sun', 'walk'): 0.4,
        ('sun', 'stay'): 0.1,
        ('rain', 'walk'): 0.1,
        ('rain', 'stay'): 0.4,
    })
    weather = distribution.marginal(0)
    assert isinstance(weather, kw.Categorical)
    assert weather.pmf('sun') == 0.5
    assert distribution.marginal([0, 1]).dimension == 2
    conditioned = distribution.condition({0: 'rain'})
    assert conditioned.pmf(['rain', 'stay']) == pytest.approx(0.8)
    assert distribution.event_probability(lambda outcome: outcome[1] == 'walk') == 0.5


def test_joint_discrete_sampling_is_shaped_and_reproducible():
    distribution = kw.JointDiscreteDistribution([(0, 0), (1, 1)])
    assert distribution.sample(random_state=1).shape == (2,)
    first = distribution.sample((3, 4), random_state=2)
    second = distribution.rvs((3, 4), random_state=2)
    assert first.shape == (3, 4, 2)
    np.testing.assert_array_equal(first, second)


def test_joint_discrete_normalizes_relative_weights():
    distribution = kw.JointDiscreteDistribution([(0, 0), (1, 1)], [2, 3], normalize=True)
    assert distribution.probabilities == pytest.approx((0.4, 0.6))


def test_independent_continuous_product_distribution():
    first = kw.Normal()
    second = kw.Exponential(2)
    distribution = kw.IndependentJointDistribution(first, second)
    assert isinstance(distribution, kw.ProductDistribution)
    assert distribution.dimension == 2
    assert not distribution.is_discrete
    np.testing.assert_allclose(distribution.mean, [0, 0.5])
    np.testing.assert_allclose(distribution.covariance, [[1, 0], [0, 0.25]])
    point = [0, 1]
    assert distribution.pdf(point) == pytest.approx(first.pdf(0) * second.pdf(1))
    assert distribution.cdf(point) == pytest.approx(first.cdf(0) * second.cdf(1))
    expected_box = first.probability_between(-1, 1) * second.probability_between(0, 1)
    assert distribution.probability_box([-1, 0], [1, 1]) == pytest.approx(expected_box)
    assert distribution.marginal(0) is first
    assert distribution.marginal([0, 1]).dimension == 2
    assert distribution.sample(5, random_state=3).shape == (5, 2)
    assert 'IndependentJointDistribution' in repr(distribution)
    with pytest.raises(TypeError, match='pmf'):
        distribution.pmf(point)


def test_independent_discrete_product_supports_named_categories():
    category = kw.Categorical({'red': 0.3, 'blue': 0.7})
    success = kw.Bernoulli(0.4)
    distribution = kw.IndependentJointDistribution([category, success])
    assert distribution.is_discrete
    assert distribution.pmf(['red', 1]) == pytest.approx(0.12)
    assert distribution.cdf(['red', 0]) == pytest.approx(0.3 * 0.6)
    assert distribution.probability_box(['red', 0], ['blue', 1]) == 1
    with pytest.raises(TypeError, match='pdf'):
        distribution.pdf(['red', 1])


def test_multinomial_distribution():
    distribution = kw.Multinomial(4, [0.2, 0.3, 0.5])
    assert distribution.dimension == 3
    np.testing.assert_allclose(distribution.mean, [0.8, 1.2, 2])
    expected_covariance = 4 * (
        np.diag([0.2, 0.3, 0.5]) - np.outer([0.2, 0.3, 0.5], [0.2, 0.3, 0.5])
    )
    np.testing.assert_allclose(distribution.covariance, expected_covariance)
    expected = math.factorial(4) / (math.factorial(1) * math.factorial(1) * math.factorial(2))
    expected *= 0.2 * 0.3 * 0.5 ** 2
    assert distribution.pmf([1, 1, 2]) == pytest.approx(expected)
    assert distribution.pmf([1, 1, 1]) == 0
    assert distribution.logpmf([1, -1, 4]) == -math.inf
    assert isinstance(distribution.marginal(1), kw.Binomial)
    assert distribution.marginal(1).p == 0.3
    assert distribution.sample(10, random_state=4).shape == (10, 3)
    assert 'Multinomial' in repr(distribution)


def test_multinomial_zero_probability_category():
    distribution = kw.Multinomial(2, [1, 0])
    assert distribution.pmf([2, 0]) == 1
    assert distribution.pmf([1, 1]) == 0


def test_dirichlet_distribution():
    distribution = kw.Dirichlet([2, 3, 4])
    assert distribution.dimension == 3
    np.testing.assert_allclose(distribution.mean, [2 / 9, 3 / 9, 4 / 9])
    assert distribution.covariance.shape == (3, 3)
    np.testing.assert_allclose(distribution.variance, np.diag(distribution.covariance))
    np.testing.assert_allclose(distribution.mode, [1 / 6, 2 / 6, 3 / 6])
    point = [0.2, 0.3, 0.5]
    assert distribution.pdf(point) == pytest.approx(math.exp(distribution.logpdf(point)))
    assert distribution.pdf([0.2, 0.3, 0.4]) == 0
    assert distribution.marginal_parameters(1) == (3, 6)
    samples = distribution.sample(20, random_state=5)
    assert samples.shape == (20, 3)
    np.testing.assert_allclose(samples.sum(axis=1), 1)
    assert 'Dirichlet' in repr(distribution)


def test_dirichlet_boundary_density_and_mode_domain():
    assert kw.Dirichlet([1, 1]).pdf([0, 1]) == pytest.approx(1)
    assert kw.Dirichlet([2, 2]).pdf([0, 1]) == 0
    assert kw.Dirichlet([0.5, 2]).pdf([0, 1]) == math.inf
    with pytest.raises(ValueError, match='mode'):
        _ = kw.Dirichlet([1, 2]).mode


def test_multivariate_normal_density_and_geometry():
    distribution = kw.MultivariateNormal([0, 0], [[1, 0], [0, 1]])
    assert distribution.dimension == 2
    np.testing.assert_allclose(distribution.mean, [0, 0])
    np.testing.assert_allclose(distribution.covariance, np.eye(2))
    np.testing.assert_allclose(distribution.correlation, np.eye(2))
    assert distribution.pdf([0, 0]) == pytest.approx(1 / (2 * math.pi))
    assert distribution.logpdf([0, 0]) == pytest.approx(-math.log(2 * math.pi))
    np.testing.assert_allclose(distribution.pdf([[0, 0], [1, 0]]),
                               [1 / (2 * math.pi), math.exp(-0.5) / (2 * math.pi)])
    assert distribution.mahalanobis([3, 4]) == pytest.approx(5)
    assert distribution.sample(random_state=6).shape == (2,)
    assert distribution.sample((2, 3), random_state=6).shape == (2, 3, 2)
    assert 'MultivariateNormal' in repr(distribution)


def test_multivariate_normal_marginals_and_conditionals():
    distribution = kw.MultivariateNormal([1, 2], [[4, 1], [1, 9]])
    marginal = distribution.marginal(0)
    assert isinstance(marginal, kw.Normal)
    assert marginal.mean == 1
    assert marginal.std == 2
    assert distribution.marginal([0, 1]).dimension == 2
    conditional = distribution.conditional(0, [3])
    assert isinstance(conditional, kw.Normal)
    assert conditional.mean == pytest.approx(2.5)
    assert conditional.variance == pytest.approx(8.75)


def test_three_dimensional_normal_conditioning():
    distribution = kw.MultivariateNormal([0, 0, 0], np.eye(3))
    conditional = distribution.conditional([0], [2])
    assert isinstance(conditional, kw.MultivariateNormal)
    np.testing.assert_allclose(conditional.mean, [0, 0])
    np.testing.assert_allclose(conditional.covariance, np.eye(2))


def test_multivariate_normal_probability_estimation_is_explicit():
    distribution = kw.MultivariateNormal([0, 0], np.eye(2))
    estimate = distribution.probability_box([-1, -1], [1, 1], samples=20_000,
                                            random_state=7)
    assert isinstance(estimate, kw.ProbabilityEstimate)
    assert float(estimate) == estimate.probability
    assert estimate.probability == pytest.approx(0.466, abs=0.02)
    assert estimate.standard_error > 0
    low, high = estimate.confidence_interval_95
    assert 0 <= low < estimate.probability < high <= 1
    first = distribution.cdf([[0, 0], [1, 1]], samples=10_000, random_state=8)
    second = distribution.cdf([[0, 0], [1, 1]], samples=10_000, random_state=8)
    np.testing.assert_array_equal(first, second)
    assert first[0] == pytest.approx(0.25, abs=0.02)


def test_zero_variance_correlation_entries_are_nan():
    distribution = kw.Multinomial(2, [1, 0])
    assert np.isnan(distribution.correlation).all()


@pytest.mark.parametrize(
    'constructor',
    [
        lambda: kw.JointDiscreteDistribution([]),
        lambda: kw.JointDiscreteDistribution([(1,), (1,)]),
        lambda: kw.JointDiscreteDistribution([(1,), (1, 2)]),
        lambda: kw.JointDiscreteDistribution([(1,), (2,)], [0.2, 0.2]),
        lambda: kw.Multinomial(-1, [1]),
        lambda: kw.Multinomial(2, [-1, 2]),
        lambda: kw.Dirichlet([1, 0]),
        lambda: kw.MultivariateNormal([0, 0], np.eye(3)),
        lambda: kw.MultivariateNormal([0, 0], [[1, 2], [0, 1]]),
        lambda: kw.MultivariateNormal([0, 0], [[1, 2], [2, 1]]),
    ],
)
def test_invalid_multivariate_parameters(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()


def test_joint_distribution_validation_and_conditioning_errors():
    with pytest.raises(ValueError, match='do not pass'):
        kw.JointDiscreteDistribution({(0,): 1}, [1])
    with pytest.raises(ValueError, match='same length'):
        kw.JointDiscreteDistribution([(0,), (1,)], [1])
    with pytest.raises(TypeError, match='vector-like'):
        kw.JointDiscreteDistribution({1: 1})
    distribution = kw.JointDiscreteDistribution({('a', 0): 1})
    with pytest.raises(TypeError, match='numeric'):
        _ = distribution.mean
    with pytest.raises(TypeError, match='predicate'):
        distribution.event_probability(1)
    with pytest.raises(TypeError, match='non-empty'):
        distribution.condition({})
    with pytest.raises(ValueError, match='zero'):
        distribution.condition({0: 'missing'})


def test_independent_joint_validation():
    with pytest.raises(ValueError, match='at least one'):
        kw.IndependentJointDistribution()
    with pytest.raises(TypeError, match='Distribution'):
        kw.IndependentJointDistribution(kw.Normal(), object())
    with pytest.raises(ValueError, match='all discrete or all continuous'):
        kw.IndependentJointDistribution(kw.Normal(), kw.Bernoulli())
    distribution = kw.IndependentJointDistribution(kw.Normal(), kw.Normal())
    with pytest.raises(ValueError, match='dimension'):
        distribution.pdf([0, 0, 0])
    with pytest.raises(ValueError, match='bounds'):
        distribution.probability_box([0], [1])
    with pytest.raises(ValueError, match='exceed'):
        distribution.probability_box([1, 0], [0, 1])


def test_index_and_conditioning_validation():
    distribution = kw.MultivariateNormal([0, 0], np.eye(2))
    with pytest.raises(ValueError, match='unique'):
        distribution.marginal([0, 0])
    with pytest.raises(ValueError, match='smaller'):
        distribution.marginal(2)
    with pytest.raises(ValueError, match='remain'):
        distribution.conditional([0, 1], [0, 0])
    with pytest.raises(ValueError, match='match'):
        distribution.conditional([0], [0, 1])


def test_multivariate_sampling_validation():
    distribution = kw.Multinomial(2, [0.5, 0.5])
    with pytest.raises(ValueError, match='size'):
        distribution.sample(-1)
    with pytest.raises(TypeError, match='size'):
        distribution.sample([2, 2])
    with pytest.raises(TypeError, match='random_state'):
        distribution.sample(2, random_state='seed')


def test_multivariate_exports():
    for name in (
        'MultivariateDistribution', 'JointDiscreteDistribution',
        'IndependentJointDistribution', 'ProductDistribution', 'Multinomial',
        'Dirichlet', 'MultivariateNormal', 'ProbabilityEstimate',
    ):
        assert hasattr(kw, name)
