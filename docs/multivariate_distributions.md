# Multidimensional probability distributions

Multidimensional distributions have their own API instead of overloading the
one-dimensional distribution classes. Every observation places its components on
the final NumPy axis:

- one point has shape `(dimension,)`;
- a batch of points has shape `(..., dimension)`;
- generated samples have shape `sample_shape + (dimension,)`.

```python
import kiwicalc as kw

model = kw.MultivariateNormal(
    mean=[10, 20],
    covariance=[[4, 1], [1, 9]],
)

model.pdf([10, 20])
model.pdf([[10, 20], [12, 23]])
model.sample(500, random_state=42).shape   # (500, 2)
```

## Available models

| Model | Purpose |
| --- | --- |
| `JointDiscreteDistribution` | Arbitrary finite vector outcomes |
| `IndependentJointDistribution` | Product of independent 1D distributions |
| `ProductDistribution` | Alias for `IndependentJointDistribution` |
| `Multinomial` | Counts across several categories |
| `Dirichlet` | Continuous probability vectors on a simplex |
| `MultivariateNormal` | Correlated continuous measurements |

All provide `dimension`, vector `mean`, a covariance matrix, component variances
and standard deviations, a correlation matrix, and reproducible sampling.

## Finite joint distributions

```python
weather_activity = kw.JointDiscreteDistribution({
    ("sun", "walk"): 0.40,
    ("sun", "stay"): 0.10,
    ("rain", "walk"): 0.10,
    ("rain", "stay"): 0.40,
})

weather_activity.pmf(["rain", "stay"])
weather_activity.event_probability(lambda outcome: outcome[1] == "walk")

weather = weather_activity.marginal(0)
given_rain = weather_activity.condition({0: "rain"})
```

Marginalizing one component returns a `Categorical` distribution. Selecting
multiple components returns another joint distribution. Conditioning retains the
full outcome vectors and renormalizes their masses.

Means, covariance, and componentwise CDFs require numeric outcomes. Named outcomes
remain fully supported for PMFs, marginals, conditioning, events, and sampling.

## Independent products

```python
independent = kw.IndependentJointDistribution(
    kw.Normal(mean=0, std=1),
    kw.Exponential(rate=2),
)

independent.pdf([0, 1])
independent.cdf([0, 1])
independent.probability_box([-1, 0], [1, 1])
independent.marginal(0)
```

Components must be either all discrete or all continuous. The product model uses
exact products of component PMFs, PDFs, CDFs, and interval probabilities.

## Multinomial and Dirichlet

```python
counts = kw.Multinomial(n=20, probabilities=[0.2, 0.3, 0.5])
counts.pmf([4, 6, 10])
counts.marginal(0)                 # Binomial(20, 0.2)

prior = kw.Dirichlet([2, 3, 5])
prior.pdf([0.2, 0.3, 0.5])
prior.mean
prior.mode
prior.marginal_parameters(0)      # parameters for its beta marginal
```

Multinomial count vectors must contain non-negative integers summing to `n`.
Dirichlet points must be non-negative and sum to one. The interior Dirichlet mode
is available only when every alpha parameter exceeds one.

## Multivariate normal models

```python
measurements = kw.MultivariateNormal(
    mean=[1, 2],
    covariance=[[4, 1], [1, 9]],
)

measurements.mahalanobis([3, 5])
measurements.marginal(0)           # Normal(mean=1, std=2)
measurements.conditional(
    observed_indices=[0],
    observed_values=[3],
)
```

Covariance matrices must be finite, symmetric, and positive definite. Marginals
and conditional distributions use the exact multivariate-normal formulas.

A general multivariate normal CDF has no elementary closed form. KiwiCalc therefore
makes approximation explicit:

```python
estimate = measurements.probability_box(
    lower=[-1, -1],
    upper=[1, 1],
    samples=100_000,
    random_state=42,
)

estimate.probability
estimate.standard_error
estimate.confidence_interval_95
```

`cdf()` is likewise Monte Carlo based and accepts `samples` and `random_state`.
Seeded calls are reproducible. Exact density, marginal, conditional, moment, and
sampling operations do not use Monte Carlo.

## Plotting joint distributions

```python
model.plot()                              # 2D density contours
model.plot("contourf", levels=15)
model.plot("surface", theme="engineering")
model.scatter(size=1_000, random_state=42)
```

For models with more than two variables, choose a pair with
`dimensions=(0, 2)`. Multivariate-normal and independent continuous models plot
the exact selected marginal density, so omitted dimensions are marginalized—not
silently fixed to zero.

Finite joint distributions use a probability heatmap by default and work with
both numeric and named outcomes:

```python
weather_activity.plot(annotate=True)
weather_activity.plot("bubble", colorbar=False)
```

Models whose support is not naturally a rectangular plane, such as Dirichlet and
Multinomial distributions, use a sample cloud by default. Calling `scatter()` is
the explicit equivalent. As with one-dimensional distributions, plotting accepts
existing `fig`/`ax` objects, themes, labels, `show=False`, and Matplotlib style
keywords, and returns the primary artist.
