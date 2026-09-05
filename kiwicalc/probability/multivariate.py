"""Multidimensional probability distributions.

Observations use the final array axis: one point has shape ``(dimension,)`` and a
batch has shape ``(..., dimension)``. Samples append that same final dimension.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from kiwicalc.probability.distributions import (
    Binomial,
    Categorical,
    ContinuousDistribution,
    DiscreteDistribution,
    Distribution,
    Normal,
)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be an integer')
    value = int(value)
    if value < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    return value


def _vector(values, name, *, positive=False, finite=True):
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f'{name} must contain real numbers') from exc
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f'{name} must be a non-empty one-dimensional sequence')
    if finite and np.any(~np.isfinite(result)):
        raise ValueError(f'{name} must contain finite values')
    if positive and np.any(result <= 0):
        raise ValueError(f'{name} must contain positive values')
    return result


def _probabilities(values, *, normalize=False, tolerance=1e-12):
    probabilities = _vector(values, 'probabilities')
    if np.any(probabilities < 0):
        raise ValueError('probabilities cannot be negative')
    total = float(np.sum(probabilities))
    if total <= 0:
        raise ValueError('probabilities must have a positive total')
    if not isinstance(normalize, bool):
        raise TypeError('normalize must be a Boolean')
    if normalize:
        probabilities = probabilities / total
    elif not math.isclose(total, 1.0, abs_tol=tolerance, rel_tol=0):
        raise ValueError('probabilities must sum to one')
    return probabilities


def _matrix(values, name, dimension):
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f'{name} must contain real numbers') from exc
    if result.shape != (dimension, dimension):
        raise ValueError(f'{name} must have shape ({dimension}, {dimension})')
    if np.any(~np.isfinite(result)):
        raise ValueError(f'{name} must contain finite values')
    return result


def _points(values, dimension, name='points'):
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f'{name} must contain real numbers') from exc
    scalar = array.ndim == 1
    if array.ndim == 0 or array.shape[-1] != dimension:
        raise ValueError(f'{name} must have final dimension {dimension}')
    return array, scalar


def _object_points(values, dimension):
    array = np.asarray(values, dtype=object)
    scalar = array.ndim == 1
    if array.ndim == 0 or array.shape[-1] != dimension:
        raise ValueError(f'points must have final dimension {dimension}')
    return array, scalar


def _result(values, scalar):
    array = np.asarray(values)
    return array.item() if scalar else array


def _generator(random_state=None):
    if random_state is None or (isinstance(random_state, Integral)
                                and not isinstance(random_state, bool)):
        return np.random.default_rng(random_state)
    if isinstance(random_state, np.random.Generator):
        return random_state
    raise TypeError('random_state must be None, an integer seed, or numpy.random.Generator')


def _size(size):
    if size is None:
        return ()
    if isinstance(size, Integral) and not isinstance(size, bool):
        if size < 0:
            raise ValueError('size cannot be negative')
        return (int(size),)
    if isinstance(size, tuple):
        return tuple(_integer(value, 'size dimension') for value in size)
    raise TypeError('size must be None, a non-negative integer, or a tuple of integers')


def _indices(indices, dimension, *, allow_all=False):
    if isinstance(indices, Integral) and not isinstance(indices, bool):
        result = (int(indices),)
    else:
        try:
            result = tuple(indices)
        except TypeError as exc:
            raise TypeError('indices must be an integer or iterable of integers') from exc
    if not result and not allow_all:
        raise ValueError('select at least one index')
    checked = tuple(_integer(index, 'index') for index in result)
    if len(set(checked)) != len(checked):
        raise ValueError('indices must be unique')
    if any(index >= dimension for index in checked):
        raise ValueError(f'indices must be smaller than dimension {dimension}')
    return checked


@dataclass(frozen=True)
class ProbabilityEstimate:
    """Monte Carlo probability estimate with an approximate standard error."""

    probability: float
    standard_error: float
    samples: int

    @property
    def confidence_interval_95(self):
        margin = 1.96 * self.standard_error
        return max(0.0, self.probability - margin), min(1.0, self.probability + margin)

    def __float__(self):
        return self.probability


class MultivariateDistribution:
    """Base interface for distributions over fixed-length vectors."""

    @property
    def dimension(self):
        raise NotImplementedError

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def covariance(self):
        raise NotImplementedError

    @property
    def variance(self):
        return np.diag(self.covariance).copy()

    @property
    def standard_deviation(self):
        return np.sqrt(self.variance)

    @property
    def std(self):
        return self.standard_deviation

    @property
    def correlation(self):
        standard = self.standard_deviation
        denominator = np.outer(standard, standard)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = self.covariance / denominator
        result[denominator == 0] = np.nan
        return result

    def sample(self, size=None, random_state=None):
        raise NotImplementedError

    rvs = sample

    def plot(self, *args, **kwargs):
        """Plot this distribution with KiwiCalc's probability renderer."""
        from kiwicalc.plotting.distributions import plot_distribution
        return plot_distribution(self, *args, **kwargs)

    def scatter(self, *args, **kwargs):
        """Scatter samples from selected dimensions of this distribution."""
        from kiwicalc.plotting.distributions import scatter_distribution
        return scatter_distribution(self, *args, **kwargs)


class MultivariateDiscreteDistribution(MultivariateDistribution):
    def pmf(self, points):
        raise NotImplementedError

    def logpmf(self, points):
        with np.errstate(divide='ignore'):
            result = np.log(np.asarray(self.pmf(points), dtype=float))
        return result.item() if result.ndim == 0 else result

    def probability(self, points):
        return self.pmf(points)


class MultivariateContinuousDistribution(MultivariateDistribution):
    def pdf(self, points):
        raise NotImplementedError

    def logpdf(self, points):
        with np.errstate(divide='ignore'):
            result = np.log(np.asarray(self.pdf(points), dtype=float))
        return result.item() if result.ndim == 0 else result


class JointDiscreteDistribution(MultivariateDiscreteDistribution):
    """An arbitrary finite joint probability mass function."""

    def __init__(self, outcomes, probabilities=None, *, normalize=False):
        if isinstance(outcomes, Mapping):
            if probabilities is not None:
                raise ValueError('do not pass probabilities when outcomes is a mapping')
            try:
                outcome_values = tuple(tuple(outcome) for outcome in outcomes.keys())
            except TypeError as exc:
                raise TypeError('joint outcomes must be vector-like') from exc
            probabilities = tuple(outcomes.values())
        else:
            try:
                outcome_values = tuple(tuple(outcome) for outcome in outcomes)
            except TypeError as exc:
                raise TypeError('outcomes must contain vector-like outcomes') from exc
        if not outcome_values:
            raise ValueError('outcomes cannot be empty')
        dimension = len(outcome_values[0])
        if dimension == 0 or any(len(outcome) != dimension for outcome in outcome_values):
            raise ValueError('all outcomes must have the same positive dimension')
        try:
            if len(set(outcome_values)) != len(outcome_values):
                raise ValueError('joint outcomes must be unique')
        except TypeError as exc:
            raise TypeError('joint outcome values must be hashable') from exc
        if probabilities is None:
            probabilities = (1 / len(outcome_values),) * len(outcome_values)
        probabilities = _probabilities(probabilities, normalize=normalize)
        if probabilities.size != len(outcome_values):
            raise ValueError('outcomes and probabilities must have the same length')
        self._outcomes = outcome_values
        self._probabilities = tuple(float(value) for value in probabilities)
        self._mapping = dict(zip(outcome_values, self._probabilities))
        self._dimension = dimension

    @property
    def dimension(self):
        return self._dimension

    @property
    def outcomes(self):
        return self._outcomes

    @property
    def probabilities(self):
        return self._probabilities

    def _numeric_outcomes(self):
        try:
            return np.asarray(self.outcomes, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError('moments require numeric joint outcomes') from exc

    @property
    def mean(self):
        return np.average(self._numeric_outcomes(), axis=0, weights=self.probabilities)

    @property
    def covariance(self):
        centered = self._numeric_outcomes() - self.mean
        return (centered * np.asarray(self.probabilities)[:, None]).T @ centered

    def pmf(self, points):
        array, scalar = _object_points(points, self.dimension)
        result = np.fromiter(
            (self._mapping.get(tuple(point), 0.0)
             for point in array.reshape(-1, self.dimension)),
            dtype=float,
            count=array.size // self.dimension,
        ).reshape(array.shape[:-1])
        return _result(result, scalar)

    def cdf(self, points):
        numeric = self._numeric_outcomes()
        array, scalar = _points(points, self.dimension)
        results = []
        for point in array.reshape(-1, self.dimension):
            included = np.all(numeric <= point, axis=1)
            results.append(float(np.sum(np.asarray(self.probabilities)[included])))
        return _result(np.asarray(results).reshape(array.shape[:-1]), scalar)

    def event_probability(self, predicate):
        if not callable(predicate):
            raise TypeError('predicate must be callable')
        return float(sum(probability for outcome, probability in self._mapping.items()
                         if predicate(outcome)))

    def marginal(self, indices):
        indices = _indices(indices, self.dimension)
        grouped = OrderedDict()
        for outcome, probability in self._mapping.items():
            key = tuple(outcome[index] for index in indices)
            grouped[key] = grouped.get(key, 0.0) + probability
        if len(indices) == 1:
            return Categorical({key[0]: probability for key, probability in grouped.items()})
        return JointDiscreteDistribution(grouped)

    def condition(self, conditions):
        """Condition on ``{dimension_index: required_value}``, retaining full outcomes."""
        if not isinstance(conditions, Mapping) or not conditions:
            raise TypeError('conditions must be a non-empty index-to-value mapping')
        indices = _indices(conditions.keys(), self.dimension)
        selected = {
            outcome: probability
            for outcome, probability in self._mapping.items()
            if all(outcome[index] == conditions[index] for index in indices)
        }
        total = sum(selected.values())
        if total == 0:
            raise ValueError('conditioning event has zero probability')
        return JointDiscreteDistribution(selected, normalize=True)

    def sample(self, size=None, random_state=None):
        shape = _size(size)
        generator = _generator(random_state)
        count = math.prod(shape) if shape else 1
        indices = generator.choice(len(self.outcomes), size=count, p=self.probabilities)
        values = np.asarray(self.outcomes, dtype=object)[indices]
        result = values.reshape(shape + (self.dimension,))
        return result if shape else result.reshape(self.dimension)

    rvs = sample

    def __repr__(self):
        return f'JointDiscreteDistribution({self._mapping!r})'


class IndependentJointDistribution(MultivariateDistribution):
    """Joint distribution formed from independent one-dimensional components."""

    def __init__(self, *components):
        if len(components) == 1 and isinstance(components[0], Sequence):
            components = tuple(components[0])
        if not components:
            raise ValueError('provide at least one component distribution')
        if not all(isinstance(component, Distribution) for component in components):
            raise TypeError('components must be one-dimensional Distribution objects')
        discrete = all(isinstance(component, DiscreteDistribution) for component in components)
        continuous = all(isinstance(component, ContinuousDistribution) for component in components)
        if not (discrete or continuous):
            raise ValueError('components must be all discrete or all continuous')
        self.components = tuple(components)
        self.is_discrete = discrete

    @property
    def dimension(self):
        return len(self.components)

    @property
    def mean(self):
        return np.asarray([component.mean for component in self.components], dtype=float)

    @property
    def covariance(self):
        return np.diag([component.variance for component in self.components])

    def pmf(self, points):
        if not self.is_discrete:
            raise TypeError('pmf is only available for all-discrete components')
        array, scalar = _object_points(points, self.dimension)
        result = np.ones(array.shape[:-1], dtype=float)
        for index, component in enumerate(self.components):
            result *= component.pmf(array[..., index])
        return _result(result, scalar)

    def pdf(self, points):
        if self.is_discrete:
            raise TypeError('pdf is only available for all-continuous components')
        array, scalar = _points(points, self.dimension)
        result = np.ones(array.shape[:-1], dtype=float)
        for index, component in enumerate(self.components):
            result *= component.pdf(array[..., index])
        return _result(result, scalar)

    def cdf(self, points):
        if self.is_discrete:
            array, scalar = _object_points(points, self.dimension)
        else:
            array, scalar = _points(points, self.dimension)
        result = np.ones(array.shape[:-1], dtype=float)
        for index, component in enumerate(self.components):
            result *= component.cdf(array[..., index])
        return _result(result, scalar)

    def probability_box(self, lower, upper):
        if self.is_discrete:
            lower = np.asarray(lower, dtype=object)
            upper = np.asarray(upper, dtype=object)
            if lower.ndim != 1 or upper.ndim != 1:
                raise ValueError('bounds must be one-dimensional')
        else:
            lower = _vector(lower, 'lower', finite=False)
            upper = _vector(upper, 'upper', finite=False)
        if lower.size != self.dimension or upper.size != self.dimension:
            raise ValueError(f'bounds must have dimension {self.dimension}')
        if not self.is_discrete and np.any(lower > upper):
            raise ValueError('lower bounds cannot exceed upper bounds')
        result = 1.0
        for index, component in enumerate(self.components):
            if self.is_discrete:
                result *= component.probability_between(lower[index], upper[index])
            else:
                result *= component.cdf(upper[index]) - component.cdf(lower[index])
        return float(result)

    def marginal(self, indices):
        indices = _indices(indices, self.dimension)
        selected = tuple(self.components[index] for index in indices)
        return selected[0] if len(selected) == 1 else IndependentJointDistribution(selected)

    def sample(self, size=None, random_state=None):
        shape = _size(size)
        generator = _generator(random_state)
        samples = [np.asarray(component.sample(shape, random_state=generator))
                   for component in self.components]
        return np.stack(samples, axis=-1)

    rvs = sample

    def __repr__(self):
        return f'IndependentJointDistribution({self.components!r})'


ProductDistribution = IndependentJointDistribution


class Multinomial(MultivariateDiscreteDistribution):
    """Counts across categories after ``n`` independent trials."""

    def __init__(self, n, probabilities, *, normalize=False):
        self.n = _integer(n, 'n')
        self.probabilities = _probabilities(probabilities, normalize=normalize)

    @property
    def dimension(self):
        return self.probabilities.size

    @property
    def mean(self):
        return self.n * self.probabilities

    @property
    def covariance(self):
        probabilities = self.probabilities
        return self.n * (np.diag(probabilities) - np.outer(probabilities, probabilities))

    def logpmf(self, points):
        array, scalar = _points(points, self.dimension)
        results = []
        log_coefficient = math.lgamma(self.n + 1)
        for point in array.reshape(-1, self.dimension):
            valid = (np.all(point >= 0) and np.all(point == np.floor(point))
                     and math.isclose(float(np.sum(point)), self.n, abs_tol=1e-12))
            if not valid:
                results.append(-math.inf)
                continue
            value = log_coefficient - math.fsum(math.lgamma(float(count) + 1)
                                                 for count in point)
            impossible = False
            for count, probability in zip(point, self.probabilities):
                if probability == 0:
                    if count > 0:
                        impossible = True
                        break
                else:
                    value += count * math.log(probability)
            results.append(-math.inf if impossible else value)
        return _result(np.asarray(results).reshape(array.shape[:-1]), scalar)

    def pmf(self, points):
        result = np.exp(np.asarray(self.logpmf(points)))
        return result.item() if result.ndim == 0 else result

    probability = pmf

    def marginal(self, index):
        index = _indices(index, self.dimension)[0]
        return Binomial(self.n, float(self.probabilities[index]))

    def sample(self, size=None, random_state=None):
        return _generator(random_state).multinomial(
            self.n, self.probabilities, size=_size(size)
        )

    rvs = sample

    def __repr__(self):
        return f'Multinomial(n={self.n!r}, probabilities={self.probabilities.tolist()!r})'


class Dirichlet(MultivariateContinuousDistribution):
    """Distribution over probability vectors on a simplex."""

    def __init__(self, alpha):
        self.alpha = _vector(alpha, 'alpha', positive=True)
        self._total = float(np.sum(self.alpha))
        self._log_normalizer = math.fsum(math.lgamma(float(value)) for value in self.alpha)
        self._log_normalizer -= math.lgamma(self._total)

    @property
    def dimension(self):
        return self.alpha.size

    @property
    def mean(self):
        return self.alpha / self._total

    @property
    def covariance(self):
        denominator = self._total ** 2 * (self._total + 1)
        result = -np.outer(self.alpha, self.alpha) / denominator
        diagonal = self.alpha * (self._total - self.alpha) / denominator
        np.fill_diagonal(result, diagonal)
        return result

    @property
    def mode(self):
        if np.any(self.alpha <= 1):
            raise ValueError('Dirichlet mode is interior only when every alpha is greater than one')
        return (self.alpha - 1) / (self._total - self.dimension)

    def logpdf(self, points):
        array, scalar = _points(points, self.dimension)
        results = []
        for point in array.reshape(-1, self.dimension):
            if np.any(point < 0) or not math.isclose(float(np.sum(point)), 1.0,
                                                     abs_tol=1e-12, rel_tol=0):
                results.append(-math.inf)
                continue
            value = -self._log_normalizer
            for coordinate, alpha in zip(point, self.alpha):
                if coordinate == 0:
                    if alpha < 1:
                        value = math.inf
                        break
                    if alpha > 1:
                        value = -math.inf
                        break
                else:
                    value += (alpha - 1) * math.log(coordinate)
            results.append(value)
        return _result(np.asarray(results).reshape(array.shape[:-1]), scalar)

    def pdf(self, points):
        result = np.exp(np.asarray(self.logpdf(points)))
        return result.item() if result.ndim == 0 else result

    def marginal_parameters(self, index):
        """Return the two beta parameters for one component's marginal."""
        index = _indices(index, self.dimension)[0]
        return float(self.alpha[index]), float(self._total - self.alpha[index])

    def sample(self, size=None, random_state=None):
        return _generator(random_state).dirichlet(self.alpha, size=_size(size))

    rvs = sample

    def __repr__(self):
        return f'Dirichlet(alpha={self.alpha.tolist()!r})'


class MultivariateNormal(MultivariateContinuousDistribution):
    """Multivariate normal distribution with positive-definite covariance."""

    def __init__(self, mean, covariance):
        self._mean = _vector(mean, 'mean')
        self._covariance = _matrix(covariance, 'covariance', self._mean.size)
        if not np.allclose(self._covariance, self._covariance.T, atol=1e-12, rtol=0):
            raise ValueError('covariance must be symmetric')
        try:
            self._cholesky = np.linalg.cholesky(self._covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError('covariance must be positive definite') from exc
        self._inverse = np.linalg.inv(self._covariance)
        self._log_determinant = 2 * float(np.sum(np.log(np.diag(self._cholesky))))

    @property
    def dimension(self):
        return self._mean.size

    @property
    def mean(self):
        return self._mean.copy()

    @property
    def covariance(self):
        return self._covariance.copy()

    def mahalanobis(self, points):
        array, scalar = _points(points, self.dimension)
        centered = array - self._mean
        squared = np.einsum('...i,ij,...j->...', centered, self._inverse, centered)
        result = np.sqrt(np.maximum(squared, 0))
        return _result(result, scalar)

    def logpdf(self, points):
        array, scalar = _points(points, self.dimension)
        centered = array - self._mean
        quadratic = np.einsum('...i,ij,...j->...', centered, self._inverse, centered)
        result = -0.5 * (
            self.dimension * math.log(2 * math.pi) + self._log_determinant + quadratic
        )
        return _result(result, scalar)

    def pdf(self, points):
        result = np.exp(np.asarray(self.logpdf(points)))
        return result.item() if result.ndim == 0 else result

    def marginal(self, indices):
        indices = _indices(indices, self.dimension)
        mean = self._mean[list(indices)]
        covariance = self._covariance[np.ix_(indices, indices)]
        if len(indices) == 1:
            return Normal(float(mean[0]), math.sqrt(float(covariance[0, 0])))
        return MultivariateNormal(mean, covariance)

    def conditional(self, observed_indices, observed_values):
        """Return the distribution of unobserved dimensions given observations."""
        observed = _indices(observed_indices, self.dimension)
        if len(observed) == self.dimension:
            raise ValueError('at least one dimension must remain unobserved')
        values = _vector(observed_values, 'observed_values')
        if values.size != len(observed):
            raise ValueError('observed_values must match observed_indices')
        remaining = tuple(index for index in range(self.dimension) if index not in observed)
        covariance_oo = self._covariance[np.ix_(observed, observed)]
        covariance_ro = self._covariance[np.ix_(remaining, observed)]
        covariance_or = self._covariance[np.ix_(observed, remaining)]
        covariance_rr = self._covariance[np.ix_(remaining, remaining)]
        adjustment = covariance_ro @ np.linalg.solve(
            covariance_oo, values - self._mean[list(observed)]
        )
        conditional_mean = self._mean[list(remaining)] + adjustment
        conditional_covariance = covariance_rr - (
            covariance_ro @ np.linalg.solve(covariance_oo, covariance_or)
        )
        if len(remaining) == 1:
            return Normal(
                float(conditional_mean[0]),
                math.sqrt(max(0.0, float(conditional_covariance[0, 0]))),
            )
        return MultivariateNormal(conditional_mean, conditional_covariance)

    def probability_box(self, lower, upper, *, samples=100_000, random_state=None):
        """Estimate probability inside an axis-aligned box by Monte Carlo."""
        lower = _vector(lower, 'lower', finite=False)
        upper = _vector(upper, 'upper', finite=False)
        if lower.size != self.dimension or upper.size != self.dimension:
            raise ValueError(f'bounds must have dimension {self.dimension}')
        if np.any(lower > upper):
            raise ValueError('lower bounds cannot exceed upper bounds')
        samples = _integer(samples, 'samples', 1)
        draws = self.sample(samples, random_state=random_state)
        inside = np.all((draws >= lower) & (draws <= upper), axis=-1)
        probability = float(np.mean(inside))
        standard_error = math.sqrt(probability * (1 - probability) / samples)
        return ProbabilityEstimate(probability, standard_error, samples)

    def cdf(self, points, *, samples=100_000, random_state=None):
        """Estimate componentwise CDF values by Monte Carlo."""
        array, scalar = _points(points, self.dimension)
        samples = _integer(samples, 'samples', 1)
        draws = self.sample(samples, random_state=random_state)
        result = np.fromiter(
            (np.mean(np.all(draws <= point, axis=1))
             for point in array.reshape(-1, self.dimension)),
            dtype=float,
            count=array.size // self.dimension,
        ).reshape(array.shape[:-1])
        return _result(result, scalar)

    def sample(self, size=None, random_state=None):
        return _generator(random_state).multivariate_normal(
            self._mean, self._covariance, size=_size(size)
        )

    rvs = sample

    def __repr__(self):
        return (f'MultivariateNormal(mean={self._mean.tolist()!r}, '
                f'covariance={self._covariance.tolist()!r})')


__all__ = [
    'ProbabilityEstimate', 'MultivariateDistribution',
    'MultivariateDiscreteDistribution', 'MultivariateContinuousDistribution',
    'JointDiscreteDistribution', 'IndependentJointDistribution',
    'ProductDistribution', 'Multinomial', 'Dirichlet', 'MultivariateNormal',
]
