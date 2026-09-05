# Probability distributions

KiwiCalc distributions use one predictable interface. Discrete distributions have
`pmf()`, continuous distributions have `pdf()`, and all distributions provide
`cdf()`, `sf()`, `ppf()`/`quantile()`, moments, interval probabilities, and seeded
sampling.

Only NumPy and Python's standard library are used; SciPy is not required.
Matplotlib is imported only when a plot is requested.

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
| Discrete | Formula-defined | `kw.distribution(formula, over=values)` |
| Continuous | Uniform | `kw.Uniform(low=0, high=1)` |
| Continuous | Normal | `kw.Normal(mean=0, std=1)` |
| Continuous | Exponential | `kw.Exponential(rate=1)` |
| Continuous | Formula-defined | `kw.distribution(formula, between=(a, b))` |

`Gaussian` aliases `Normal`, and `ContinuousUniform` aliases `Uniform`.

## Define a distribution with a formula

Use `distribution()` when the density or mass is easiest to describe
mathematically. KiwiCalc checks the formula on the requested support and
normalizes it automatically:

```python
x = kw.Var("x")

triangle = kw.distribution(2*x, between=(0, 1))
weighted_die = kw.distribution(x, over=range(1, 7), name="Weighted die")
```

The same formulas can be written as safe strings. Both `2*x` and the
mathematical shorthand `2x` are accepted:

```python
triangle = kw.distribution("2x", between=(0, 1))

parameterized = kw.distribution(
    "a*x",
    variable="x",
    parameters={"a": 2},
    between=(0, 1),
)
```

Exactly one of `between=` and `over=` is required. Continuous bounds must be
finite; discrete support must be a finite collection of distinct real values.
Any symbols other than the inferred distribution variable must be supplied in
`parameters=`.

Formula strings support arithmetic, powers, parentheses, `pi`, `e`, `tau`,
roots, exponentials, logarithms, trigonometric functions, inverse trigonometric
functions, and hyperbolic functions. They are parsed using a restricted
mathematical grammar and are never executed as Python code.

Inspect how KiwiCalc interpreted and normalized the input with:

```python
triangle.formula
triangle.normalized_formula
triangle.variable
triangle.parameters
triangle.normalization_constant
triangle.was_normalized
```

Formula distributions provide the same PDF/PMF, CDF, quantile, moments,
sampling, and plotting API as built-in distributions. Nonnegativity validation
for continuous formulas is numerical; formulas must be finite throughout their
closed bounded support.

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

## Plotting distributions

Every distribution has friendly `plot()` and `scatter()` methods:

```python
normal = kw.Normal(mean=100, std=15)
normal.plot(fill=True, title="Exam score model", theme="classroom")

defects = kw.Binomial(20, 0.04)
defects.plot()                         # PMF bars
defects.plot("cdf")                   # cumulative step plot
defects.scatter(size=200, random_state=42)
```

Continuous plots use a PDF line by default; discrete and categorical plots use
PMF bars. `kind="cdf"` selects a CDF. Finite supports are shown in full, while
infinite supports use central quantiles; override them with `start=` and `stop=`.

All plotting methods accept `fig=`, `ax=`, `show=False`, `title=`, `label=`,
`theme=`, `grid=`, and ordinary Matplotlib styling keywords. They return the
primary Matplotlib artist for further customization. Set `title=""` to suppress
the automatic human-readable title.

The same renderers are available as `kw.plot_distribution(distribution, ...)`
and `kw.scatter_distribution(distribution, ...)` when a functional style fits
better.
