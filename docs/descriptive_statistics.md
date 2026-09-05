# Descriptive statistics

KiwiCalc's descriptive-statistics functions accept Python lists and NumPy arrays.
They use the same optional controls throughout:

```python
import kiwicalc as kw

scores = [[72, 81, 89], [65, 84, 94]]

kw.mean(scores)
kw.mean(scores, axis=1)
kw.mean(scores, axis=1, weights=[1, 1, 2])
kw.describe(scores, axis=1, ddof=1)
```

`axis=None` treats all values as one sample. An integer `axis` calculates one
result per slice. A one-dimensional weight sequence can match the reduction axis;
full-shape and broadcastable weights are also accepted. Weights must be finite and
non-negative, and zero-weight observations are excluded.

## Missing values

Every function accepts a `nan_policy`:

- `"propagate"` (the default) produces `nan` for an affected numerical slice.
- `"omit"` removes missing observations before calculating that slice.
- `"raise"` reports missing input immediately.

If omission leaves no observations, KiwiCalc raises `ValueError`. Frequency and
contingency tables return `None` for an affected slice under `"propagate"`, because
there is no meaningful numeric NaN table.

## Location, spread, and shape

```python
data = [1, 2, 2, 3, 9]

kw.mean(data)
kw.median(data)
kw.mode(data)
kw.min(data), kw.max(data), kw.data_range(data)
kw.quartiles(data)
kw.iqr(data)

kw.population_variance(data)       # ddof=0
kw.sample_variance(data)           # ddof=1
kw.standard_deviation(data, ddof=1)
kw.mean_absolute_deviation(data)
kw.median_absolute_deviation(data)

kw.geometric_mean([1, 4, 16])
kw.harmonic_mean([1, 2, 4])
kw.trimmed_mean(data, proportion_to_cut=0.1)
kw.skewness(data)
kw.kurtosis(data)                   # excess kurtosis
kw.coefficient_of_variation(data)
```

Geometric and harmonic means require positive values. Skewness and kurtosis are
undefined for constant data. The coefficient of variation is undefined when the
mean is zero; these cases raise clear `ValueError` exceptions.

`variance()` and `standard_deviation()` use `ddof=0` by default. Their sample
aliases use `ddof=1`. For weighted variance, the denominator is
`sum(weights) - ddof`. Higher standardized moments use that same denominator,
which keeps weighted and axis-based behavior consistent.

## Quantiles and summaries

```python
kw.quantile(data, [0.1, 0.5, 0.9])
kw.percentile(data, [10, 50, 90])
kw.five_number_summary(data).as_dict()
kw.describe(data).as_dict()

fences = kw.outlier_fences(data)
mask = kw.detect_outliers(data)
```

Unweighted quantiles follow NumPy's linear quantiles. Equal weights produce the
same result. Unequal weights use linear interpolation over the weighted empirical
cumulative distribution.

## Frequencies and categorical data

```python
colors = ["green", "blue", "green", "gold"]

table = kw.frequency_table(colors)
table.to_dict()
kw.proportion_table(colors)

cross = kw.contingency_table(
    ["pass", "pass", "fail"],
    ["A", "B", "A"],
)
cross.to_dict()
```

Tables preserve first-seen category order. Their `counts`, `proportions`, and
category arrays remain available for numerical or plotting workflows.

## Relationships

```python
x = [1, 2, 3, 4]
y = [2, 5, 5, 9]

kw.covariance(x, y)
kw.pearson_correlation(x, y)
kw.spearman_correlation(x, y)
```

If the second variable is omitted from covariance or correlation, a one-dimensional
input returns the corresponding scalar and a two-dimensional input returns a
variable-by-variable matrix. Correlation with a constant variable raises
`ValueError` rather than silently producing a misleading result.
