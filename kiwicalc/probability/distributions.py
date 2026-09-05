"""Common probability distributions with a small, consistent API."""

from __future__ import annotations

import math
from numbers import Integral, Real
from statistics import NormalDist

import numpy as np


def _real(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if positive and value <= 0:
        raise ValueError(f'{name} must be positive')
    return value


def _probability(value, name='probability', *, open_interval=False):
    value = _real(value, name)
    valid = 0 < value < 1 if open_interval else 0 <= value <= 1
    if not valid:
        interval = 'strictly between zero and one' if open_interval else 'between zero and one'
        raise ValueError(f'{name} must be {interval}')
    return value


def _integer(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be an integer')
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    return value


def _input(values, *, dtype=float):
    try:
        array = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError('values must be numeric') from exc
    return array, array.ndim == 0


def _result(values, scalar):
    array = np.asarray(values)
    return array.item() if scalar else array


def _map_numeric(values, function):
    array, scalar = _input(values)
    flat = np.fromiter((math.nan if math.isnan(float(value)) else function(float(value))
                        for value in array.reshape(-1)),
                       dtype=float, count=array.size)
    return _result(flat.reshape(array.shape), scalar)


def _quantiles(values):
    array, scalar = _input(values)
    if np.any(~np.isfinite(array)) or np.any((array < 0) | (array > 1)):
        raise ValueError('quantiles must be finite and between zero and one')
    return array, scalar


def _generator(random_state=None):
    if random_state is None or isinstance(random_state, (Integral, np.integer)):
        return np.random.default_rng(random_state)
    if isinstance(random_state, np.random.Generator):
        return random_state
    raise TypeError('random_state must be None, an integer seed, or numpy.random.Generator')


def _size(size):
    if size is None:
        return None
    if isinstance(size, Integral) and not isinstance(size, bool):
        if size < 0:
            raise ValueError('size cannot be negative')
        return int(size)
    if isinstance(size, tuple):
        return tuple(_integer(value, 'size dimension', 0) for value in size)
    raise TypeError('size must be None, a non-negative integer, or a tuple of integers')


class Distribution:
    """Shared interface for probability distributions."""

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def variance(self):
        raise NotImplementedError

    @property
    def standard_deviation(self):
        return math.sqrt(self.variance)

    @property
    def std(self):
        return self.standard_deviation

    def cdf(self, value):
        raise NotImplementedError

    def sf(self, value):
        """Survival function, ``P(X > value)``."""
        result = 1 - np.asarray(self.cdf(value))
        return result.item() if result.ndim == 0 else result

    def ppf(self, probability):
        raise NotImplementedError

    quantile = ppf

    def sample(self, size=None, random_state=None):
        raise NotImplementedError

    rvs = sample


class DiscreteDistribution(Distribution):
    """Base interface for integer or categorical probability masses."""

    def pmf(self, value):
        raise NotImplementedError

    def logpmf(self, value):
        probabilities = np.asarray(self.pmf(value), dtype=float)
        with np.errstate(divide='ignore'):
            result = np.log(probabilities)
        return result.item() if result.ndim == 0 else result

    def probability(self, value):
        return self.pmf(value)

    def probability_between(self, lower, upper, *, inclusive='both'):
        """Return interval mass with configurable endpoint inclusion."""
        if inclusive not in {'both', 'left', 'right', 'neither'}:
            raise ValueError("inclusive must be 'both', 'left', 'right', or 'neither'")
        lower = _real(lower, 'lower')
        upper = _real(upper, 'upper')
        if lower > upper:
            raise ValueError('lower cannot exceed upper')
        upper_cutoff = math.floor(upper) if inclusive in {'both', 'right'} else math.ceil(upper) - 1
        lower_cutoff = math.ceil(lower) - 1 if inclusive in {'both', 'left'} else math.floor(lower)
        return max(0.0, float(self.cdf(upper_cutoff) - self.cdf(lower_cutoff)))


class ContinuousDistribution(Distribution):
    """Base interface for continuous probability densities."""

    def pdf(self, value):
        raise NotImplementedError

    def logpdf(self, value):
        densities = np.asarray(self.pdf(value), dtype=float)
        with np.errstate(divide='ignore'):
            result = np.log(densities)
        return result.item() if result.ndim == 0 else result

    def probability_between(self, lower, upper):
        lower = _real(lower, 'lower')
        upper = _real(upper, 'upper')
        if lower > upper:
            raise ValueError('lower cannot exceed upper')
        return max(0.0, float(self.cdf(upper) - self.cdf(lower)))

    probability = probability_between


class Bernoulli(DiscreteDistribution):
    """Bernoulli distribution for one trial, with outcomes zero and one."""

    def __init__(self, p=0.5):
        self.p = _probability(p, 'p')

    @property
    def support(self):
        return 0, 1

    @property
    def mean(self):
        return self.p

    @property
    def variance(self):
        return self.p * (1 - self.p)

    def pmf(self, value):
        def scalar(x):
            if x == 0:
                return 1 - self.p
            if x == 1:
                return self.p
            return 0.0
        return _map_numeric(value, scalar)

    def cdf(self, value):
        return _map_numeric(value, lambda x: 0.0 if x < 0 else (1 - self.p if x < 1 else 1.0))

    def ppf(self, probability):
        values, scalar = _quantiles(probability)
        result = np.where(values <= 1 - self.p, 0.0, 1.0)
        return _result(result, scalar)

    quantile = ppf

    def sample(self, size=None, random_state=None):
        return _generator(random_state).binomial(1, self.p, size=_size(size))

    rvs = sample

    def __repr__(self):
        return f'Bernoulli(p={self.p!r})'


class Binomial(DiscreteDistribution):
    """Number of successes in ``n`` independent Bernoulli trials."""

    def __init__(self, n, p=0.5):
        self.n = _integer(n, 'n', 0)
        self.p = _probability(p, 'p')

    @property
    def support(self):
        return 0, self.n

    @property
    def mean(self):
        return self.n * self.p

    @property
    def variance(self):
        return self.n * self.p * (1 - self.p)

    def _pmf(self, x):
        if not x.is_integer() or x < 0 or x > self.n:
            return 0.0
        k = int(x)
        return math.comb(self.n, k) * self.p ** k * (1 - self.p) ** (self.n - k)

    def pmf(self, value):
        return _map_numeric(value, self._pmf)

    def _cdf(self, x):
        if x < 0:
            return 0.0
        if x >= self.n:
            return 1.0
        return min(1.0, sum(self._pmf(float(k)) for k in range(math.floor(x) + 1)))

    def cdf(self, value):
        return _map_numeric(value, self._cdf)

    def ppf(self, probability):
        values, scalar = _quantiles(probability)
        def inverse(q):
            cumulative = 0.0
            for k in range(self.n + 1):
                cumulative += self._pmf(float(k))
                if cumulative >= q:
                    return float(k)
            return float(self.n)
        result = np.fromiter((inverse(float(q)) for q in values.reshape(-1)),
                             dtype=float, count=values.size).reshape(values.shape)
        return _result(result, scalar)

    quantile = ppf

    def sample(self, size=None, random_state=None):
        return _generator(random_state).binomial(self.n, self.p, size=_size(size))

    rvs = sample

    def __repr__(self):
        return f'Binomial(n={self.n!r}, p={self.p!r})'


class Geometric(DiscreteDistribution):
    """Number of trials up to and including the first success."""

    def __init__(self, p):
        self.p = _probability(p, 'p', open_interval=True)

    @property
    def support(self):
        return 1, math.inf

    @property
    def mean(self):
        return 1 / self.p

    @property
    def variance(self):
        return (1 - self.p) / self.p ** 2

    def pmf(self, value):
        return _map_numeric(
            value,
            lambda x: ((1 - self.p) ** (int(x) - 1) * self.p
                       if x.is_integer() and x >= 1 else 0.0),
        )

    def cdf(self, value):
        return _map_numeric(
            value,
            lambda x: (0.0 if x < 1 else
                       (1.0 if math.isinf(x) else 1 - (1 - self.p) ** math.floor(x))),
        )

    def ppf(self, probability):
        values, scalar = _quantiles(probability)
        def inverse(q):
            if q == 0:
                return 1.0
            if q == 1:
                return math.inf
            return float(math.ceil(math.log1p(-q) / math.log1p(-self.p)))
        result = np.fromiter((inverse(float(q)) for q in values.reshape(-1)),
                             dtype=float, count=values.size).reshape(values.shape)
        return _result(result, scalar)

    quantile = ppf

    def sample(self, size=None, random_state=None):
        return _generator(random_state).geometric(self.p, size=_size(size))

    rvs = sample

    def __repr__(self):
        return f'Geometric(p={self.p!r})'


class Hypergeometric(DiscreteDistribution):
    """Successes drawn without replacement from a finite population."""

    def __init__(self, population, successes, draws):
        self.population = _integer(population, 'population', 1)
        self.successes = _integer(successes, 'successes', 0)
        self.draws = _integer(draws, 'draws', 0)
        if self.successes > self.population:
            raise ValueError('successes cannot exceed population')
        if self.draws > self.population:
            raise ValueError('draws cannot exceed population')

    @property
    def support(self):
        return max(0, self.draws - (self.population - self.successes)), min(
            self.draws, self.successes
        )

    @property
    def mean(self):
        return self.draws * self.successes / self.population

    @property
    def variance(self):
        if self.population == 1:
            return 0.0
        proportion = self.successes / self.population
        return (self.draws * proportion * (1 - proportion)
                * (self.population - self.draws) / (self.population - 1))

    def _pmf(self, x):
        lower, upper = self.support
        if not x.is_integer() or x < lower or x > upper:
            return 0.0
        k = int(x)
        return (math.comb(self.successes, k)
                * math.comb(self.population - self.successes, self.draws - k)
                / math.comb(self.population, self.draws))

    def pmf(self, value):
        return _map_numeric(value, self._pmf)

    def cdf(self, value):
        lower, upper = self.support
        def scalar(x):
            if x < lower:
                return 0.0
            if x >= upper:
                return 1.0
            return min(1.0, sum(self._pmf(float(k)) for k in range(lower, math.floor(x) + 1)))
        return _map_numeric(value, scalar)

    def ppf(self, probability):
        values, scalar = _quantiles(probability)
        lower, upper = self.support
        def inverse(q):
            cumulative = 0.0
            for k in range(lower, upper + 1):
                cumulative += self._pmf(float(k))
                if cumulative >= q:
                    return float(k)
            return float(upper)
        result = np.fromiter((inverse(float(q)) for q in values.reshape(-1)),
                             dtype=float, count=values.size).reshape(values.shape)
        return _result(result, scalar)

    quantile = ppf

    def sample(self, size=None, random_state=None):
        return _generator(random_state).hypergeometric(
            self.successes, self.population - self.successes, self.draws,
            size=_size(size),
        )

    rvs = sample

    def __repr__(self):
        return (f'Hypergeometric(population={self.population!r}, '
                f'successes={self.successes!r}, draws={self.draws!r})')


class Poisson(DiscreteDistribution):
    """Poisson distribution with positive event rate ``rate``."""

    def __init__(self, rate):
        self.rate = _real(rate, 'rate', positive=True)

    @property
    def support(self):
        return 0, math.inf

    @property
    def mean(self):
        return self.rate

    @property
    def variance(self):
        return self.rate

    def _pmf(self, x):
        if not x.is_integer() or x < 0:
            return 0.0
        k = int(x)
        return math.exp(-self.rate + k * math.log(self.rate) - math.lgamma(k + 1))

    def pmf(self, value):
        return _map_numeric(value, self._pmf)

    def _cdf(self, x):
        if x < 0:
            return 0.0
        if math.isinf(x):
            return 1.0
        k_max = math.floor(x)
        return min(1.0, math.fsum(self._pmf(float(k)) for k in range(k_max + 1)))

    def cdf(self, value):
        return _map_numeric(value, self._cdf)

    def ppf(self, probability):
        values, scalar = _quantiles(probability)
        def inverse(q):
            if q == 0:
                return 0.0
            if q == 1:
                return math.inf
            cumulative = 0.0
            k = -1
            while cumulative < q:
                k += 1
                cumulative += self._pmf(float(k))
            return float(k)
        result = np.fromiter((inverse(float(q)) for q in values.reshape(-1)),
                             dtype=float, count=values.size).reshape(values.shape)
        return _result(result, scalar)

    quantile = ppf

    def sample(self, size=None, random_state=None):
        return _generator(random_state).poisson(self.rate, size=_size(size))

    rvs = sample

    def __repr__(self):
        return f'Poisson(rate={self.rate!r})'


class DiscreteUniform(DiscreteDistribution):
    """Uniform distribution over all integers from ``low`` through ``high``."""

    def __init__(self, low, high):
        self.low = _integer(low, 'low')
        self.high = _integer(high, 'high')
        if self.low > self.high:
            raise ValueError('low cannot exceed high')
        self._count = self.high - self.low + 1

    @property
    def support(self):
        return self.low, self.high

    @property
    def mean(self):
        return (self.low + self.high) / 2

    @property
    def variance(self):
        return (self._count ** 2 - 1) / 12

    def pmf(self, value):
        return _map_numeric(
            value,
            lambda x: 1 / self._count if x.is_integer() and self.low <= x <= self.high else 0.0,
        )

    def cdf(self, value):
        def scalar(x):
            if x < self.low:
                return 0.0
            if x >= self.high:
                return 1.0
            return (math.floor(x) - self.low + 1) / self._count
        return _map_numeric(value, scalar)

    def ppf(self, probability):
        values, scalar = _quantiles(probability)
        result = np.where(
            values == 0,
            self.low,
            self.low + np.ceil(values * self._count).astype(int) - 1,
        )
        return _result(result, scalar)

    quantile = ppf

    def sample(self, size=None, random_state=None):
        return _generator(random_state).integers(self.low, self.high + 1, size=_size(size))

    rvs = sample

    def __repr__(self):
        return f'DiscreteUniform(low={self.low!r}, high={self.high!r})'


class Categorical(DiscreteDistribution):
    """Distribution over arbitrary named or numeric categories."""

    def __init__(self, probabilities, values=None, *, normalize=False):
        if isinstance(probabilities, dict):
            if values is not None:
                raise ValueError('do not pass values when probabilities is a mapping')
            values = tuple(probabilities.keys())
            probabilities = tuple(probabilities.values())
        else:
            probabilities = tuple(probabilities)
            values = tuple(range(len(probabilities))) if values is None else tuple(values)
        if not probabilities:
            raise ValueError('probabilities cannot be empty')
        if len(values) != len(probabilities):
            raise ValueError('values and probabilities must have the same length')
        try:
            if len(set(values)) != len(values):
                raise ValueError('categorical values must be unique')
        except TypeError as exc:
            raise TypeError('categorical values must be hashable') from exc
        checked = []
        for index, probability in enumerate(probabilities):
            probability = _real(probability, f'probability {index}')
            if probability < 0:
                raise ValueError('probabilities cannot be negative')
            checked.append(probability)
        total = sum(checked)
        if total <= 0:
            raise ValueError('probabilities must have a positive total')
        if not isinstance(normalize, bool):
            raise TypeError('normalize must be a Boolean')
        if normalize:
            checked = [probability / total for probability in checked]
        elif not math.isclose(total, 1.0, abs_tol=1e-12, rel_tol=0):
            raise ValueError('probabilities must sum to one')
        self.values = values
        self.probabilities = tuple(checked)
        self._mapping = dict(zip(values, checked))

    @property
    def support(self):
        return self.values

    def _numeric_values(self):
        try:
            return np.asarray(self.values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError('mean and variance require numeric categorical values') from exc

    @property
    def mean(self):
        return float(np.dot(self._numeric_values(), self.probabilities))

    @property
    def variance(self):
        values = self._numeric_values()
        return float(np.dot((values - self.mean) ** 2, self.probabilities))

    def pmf(self, value):
        array = np.asarray(value, dtype=object)
        scalar = array.ndim == 0
        result = np.fromiter((self._mapping.get(item, 0.0) for item in array.reshape(-1)),
                             dtype=float, count=array.size).reshape(array.shape)
        return _result(result, scalar)

    def cdf(self, value):
        array = np.asarray(value, dtype=object)
        scalar = array.ndim == 0
        results = []
        for item in array.reshape(-1):
            try:
                index = self.values.index(item)
            except ValueError as exc:
                raise ValueError('categorical cdf requires values in the ordered support') from exc
            results.append(sum(self.probabilities[:index + 1]))
        return _result(np.asarray(results).reshape(array.shape), scalar)

    def probability_between(self, lower, upper, *, inclusive='both'):
        if inclusive not in {'both', 'left', 'right', 'neither'}:
            raise ValueError("inclusive must be 'both', 'left', 'right', or 'neither'")
        try:
            lower_index = self.values.index(lower)
            upper_index = self.values.index(upper)
        except ValueError as exc:
            raise ValueError('categorical interval endpoints must be in the support') from exc
        if lower_index > upper_index:
            raise ValueError('lower cannot follow upper in the ordered support')
        start = lower_index if inclusive in {'both', 'left'} else lower_index + 1
        stop = upper_index + 1 if inclusive in {'both', 'right'} else upper_index
        return float(sum(self.probabilities[start:stop]))

    def ppf(self, probability):
        values, scalar = _quantiles(probability)
        cumulative = np.cumsum(self.probabilities)
        result = np.empty(values.shape, dtype=object)
        for index in np.ndindex(values.shape):
            position = min(int(np.searchsorted(cumulative, values[index], side='left')),
                           len(self.values) - 1)
            result[index] = self.values[position]
        return _result(result, scalar)

    quantile = ppf

    def sample(self, size=None, random_state=None):
        return _generator(random_state).choice(
            np.asarray(self.values, dtype=object), size=_size(size), p=self.probabilities
        )

    rvs = sample

    def __repr__(self):
        return f'Categorical({dict(zip(self.values, self.probabilities))!r})'


class Uniform(ContinuousDistribution):
    """Continuous uniform distribution on ``[low, high]``."""

    def __init__(self, low=0, high=1):
        self.low = _real(low, 'low')
        self.high = _real(high, 'high')
        if self.low >= self.high:
            raise ValueError('low must be smaller than high')

    @property
    def support(self):
        return self.low, self.high

    @property
    def mean(self):
        return (self.low + self.high) / 2

    @property
    def variance(self):
        return (self.high - self.low) ** 2 / 12

    def pdf(self, value):
        density = 1 / (self.high - self.low)
        return _map_numeric(value, lambda x: density if self.low <= x <= self.high else 0.0)

    def cdf(self, value):
        width = self.high - self.low
        return _map_numeric(
            value,
            lambda x: 0.0 if x <= self.low else (1.0 if x >= self.high else (x - self.low) / width),
        )

    def ppf(self, probability):
        values, scalar = _quantiles(probability)
        return _result(self.low + values * (self.high - self.low), scalar)

    quantile = ppf

    def sample(self, size=None, random_state=None):
        return _generator(random_state).uniform(self.low, self.high, size=_size(size))

    rvs = sample

    def __repr__(self):
        return f'Uniform(low={self.low!r}, high={self.high!r})'


class Normal(ContinuousDistribution):
    """Normal distribution parameterized by mean and standard deviation."""

    def __init__(self, mean=0, std=1):
        self._mean = _real(mean, 'mean')
        self._std = _real(std, 'std', positive=True)
        self._normal = NormalDist(mu=self._mean, sigma=self._std)

    @property
    def support(self):
        return -math.inf, math.inf

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._std ** 2

    @property
    def standard_deviation(self):
        return self._std

    def pdf(self, value):
        return _map_numeric(value, self._normal.pdf)

    def cdf(self, value):
        return _map_numeric(value, self._normal.cdf)

    def ppf(self, probability):
        values, scalar = _quantiles(probability)
        def inverse(q):
            if q == 0:
                return -math.inf
            if q == 1:
                return math.inf
            return self._normal.inv_cdf(q)
        result = np.fromiter((inverse(float(q)) for q in values.reshape(-1)),
                             dtype=float, count=values.size).reshape(values.shape)
        return _result(result, scalar)

    quantile = ppf

    def sample(self, size=None, random_state=None):
        return _generator(random_state).normal(self.mean, self._std, size=_size(size))

    rvs = sample

    def z_score(self, value):
        result = (np.asarray(value, dtype=float) - self.mean) / self._std
        return result.item() if result.ndim == 0 else result

    def __repr__(self):
        return f'Normal(mean={self.mean!r}, std={self._std!r})'


class Exponential(ContinuousDistribution):
    """Exponential waiting-time distribution parameterized by event rate."""

    def __init__(self, rate=1):
        self.rate = _real(rate, 'rate', positive=True)

    @classmethod
    def from_scale(cls, scale):
        scale = _real(scale, 'scale', positive=True)
        return cls(rate=1 / scale)

    @property
    def scale(self):
        return 1 / self.rate

    @property
    def support(self):
        return 0.0, math.inf

    @property
    def mean(self):
        return self.scale

    @property
    def variance(self):
        return self.scale ** 2

    def pdf(self, value):
        return _map_numeric(value, lambda x: self.rate * math.exp(-self.rate * x) if x >= 0 else 0.0)

    def cdf(self, value):
        return _map_numeric(value, lambda x: 1 - math.exp(-self.rate * x) if x >= 0 else 0.0)

    def ppf(self, probability):
        values, scalar = _quantiles(probability)
        with np.errstate(divide='ignore'):
            result = -np.log1p(-values) / self.rate
        return _result(result, scalar)

    quantile = ppf

    def sample(self, size=None, random_state=None):
        return _generator(random_state).exponential(self.scale, size=_size(size))

    rvs = sample

    def __repr__(self):
        return f'Exponential(rate={self.rate!r})'


Gaussian = Normal
ContinuousUniform = Uniform


__all__ = [
    'Distribution', 'DiscreteDistribution', 'ContinuousDistribution', 'Bernoulli',
    'Binomial', 'Geometric', 'Hypergeometric', 'Poisson', 'DiscreteUniform',
    'Categorical', 'Uniform', 'ContinuousUniform', 'Normal', 'Gaussian', 'Exponential',
]
