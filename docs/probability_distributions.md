# Probability distributions

KiwiCalc distributions use one predictable interface. Discrete distributions have
`pmf()`, continuous distributions have `pdf()`, and all distributions provide
`cdf()`, `sf()`, `ppf()`/`quantile()`, moments, interval probabilities, and seeded
sampling.

Only NumPy and Python's standard library are used; SciPy is not required.

## Available distributions

| Kind | Distribution | Constructor |
| --- | --- | --- |
| Discrete | Bernoulli | `kw.Bernoulli(p=0.5)` |
| Discrete | Binomial | `kw.Binomial(n, p=0.5)` |
| Discrete | Geometric | `kw.Geometric(p)` |
| Discrete | Hypergeometric | `kw.Hypergeometric(population, successes, draws)` |
| Discrete | Poisson | `kw.Poisson(rate)` |
| Discrete | Integer uniform | `kw.DiscreteUniform(low, high)` |
| Discrete | Categorical | `kw.Categorical(probabilities, values=None)` |
| Continuous | Uniform | `kw.Uniform(low=0, high=1)` |
| Continuous | Normal | `kw.Normal(mean=0, std=1)` |
| Continuous | Exponential | `kw.Exponential(rate=1)` |

`Gaussian` aliases `Normal`, and `ContinuousUniform` aliases `Uniform`.

## A consistent API

```python
import kiwicalc as kw

defects = kw.Binomial(n=20, p=0.04)

defects.mean
defects.variance
defects.standard_deviation
defects.pmf(2)
defects.cdf(2)
defects.sf(2)                       # P(X > 2)
defects.ppf(0.95)                  # 95th percentile
defects.probability_between(1, 3)
defects.sample(1_000, random_state=42)
```

PMFs, PDFs, CDFs, survival functions, and quantiles accept scalars, lists, and
NumPy arrays. Scalar input returns a scalar; array-like input returns an array of
the same shape. Numerical NaN input propagates.

`logpmf()` and `logpdf()` provide log probabilities and densities, including
`-inf` where the corresponding value is zero.

## Continuous distributions

```python
normal = kw.Normal(mean=100, std=15)

normal.pdf([85, 100, 115])
normal.cdf(115)
normal.probability_between(85, 115)
normal.z_score([85, 100, 115])
normal.quantile(0.975)

waiting = kw.Exponential(rate=2)
waiting.scale
waiting.probability_between(0, 1)

# Equivalent scale-first constructor
waiting = kw.Exponential.from_scale(0.5)
```

Continuous interval probabilities are calculated as `cdf(upper) - cdf(lower)`.
Endpoint inclusion does not affect a continuous probability.

## Discrete interval probabilities

```python
roll = kw.DiscreteUniform(1, 6)

roll.probability(4)
roll.probability_between(2, 5)
roll.probability_between(2, 5, inclusive="neither")
```

`inclusive` may be `"both"`, `"left"`, `"right"`, or `"neither"`.

## Categorical outcomes

```python
color = kw.Categorical({
    "green": 0.5,
    "gold": 0.3,
    "blue": 0.2,
})

color.pmf("green")
color.cdf("gold")
color.quantile(0.8)
color.sample(20, random_state=7)
```

Categorical CDFs and quantiles follow insertion order. Means and variances are
available when category values are numeric; named categories raise a clear error
for those operations.

## Reproducible sampling

```python
distribution = kw.Poisson(rate=3)

first = distribution.sample(100, random_state=42)
second = distribution.rvs(100, random_state=42)
```

Both samples are identical. `random_state` accepts an integer seed, an existing
`numpy.random.Generator`, or `None`. Sample size can be an integer or a shape tuple.
