# Statistical diagnostics

KiwiCalc provides numerical diagnostics and Matplotlib visualizations without
requiring SciPy or Seaborn. Plotting is kept separate from computation, so an
ECDF, Q-Q comparison, or assumption summary can be inspected without creating a
figure.

## One-call diagnostic dashboard

```python
import kiwicalc as kw

sample = [12.1, 11.8, 12.4, 12.0, 13.2, 11.9, 12.3]
fig, axes = kw.diagnostic_plots(sample, theme="classroom")
```

The dashboard contains a density histogram with a fitted Normal curve, an ECDF,
a Normal Q-Q plot, and a Normal P-P plot. Pass an existing continuous
distribution to compare against fixed parameters instead:

```python
kw.diagnostic_plots(sample, distribution=kw.Normal(mean=12, std=0.5))
```

## Empirical CDF

```python
result = kw.ecdf(sample)
result.values
result.probabilities
result.counts
result.plot(theme="publication")

# Equivalent functional plotting API
kw.plot_ecdf(sample)
```

Repeated observations are aggregated. Missing values are omitted by default and
reported through `result.missing`; use `nan_policy="raise"` to reject them.

## Q-Q and P-P diagnostics

```python
qq = kw.qq_data(sample)       # fits a Normal reference
pp = kw.pp_data(sample)

kw.qq_plot(sample)
kw.pp_plot(sample)
```

Passing a distribution uses its existing parameters without fitting:

```python
reference = kw.Uniform(10, 14)
kw.qq_plot(sample, reference)
kw.pp_plot(sample, reference)
```

`QQData` exposes theoretical and observed quantiles plus the robust quartile
reference line. `PPData` exposes theoretical and empirical probabilities.

## Histograms and fitted densities

```python
kw.histogram_plot(sample, bins="auto")
kw.histogram_plot(sample, fit="normal")
kw.histogram_plot(sample, fit=kw.Normal(mean=12, std=0.5))
```

`fit="normal"` estimates the mean and sample standard deviation. A supplied
distribution is only overlaid; KiwiCalc does not silently refit it.

## Assumption summaries

```python
summary = kw.assumption_summary(sample)
summary.skewness
summary.excess_kurtosis
summary.outlier_count
summary.messages
```

The summary reports missingness, spread, skewness, excess kurtosis, and Tukey
1.5×IQR outliers. Its messages are descriptive heuristics, not hypothesis tests
or pass/fail decisions. In particular, independence and sampling design cannot
be inferred from sample values.

## Confidence intervals

```python
control = kw.mean_confidence_interval([10, 11, 9, 12, 10])
treatment = kw.mean_confidence_interval([12, 13, 11, 14, 12])

kw.confidence_interval_plot(
    [control, treatment],
    labels=["Control", "Treatment"],
    reference=10,
)
```

Both horizontal and vertical layouts are supported. Plotting requires finite
bounds; one-sided intervals with an infinite endpoint should be summarized
numerically instead.

Every diagnostic plot accepts `fig=`, `ax=`, `show=False`, `title=`, `theme=`,
and ordinary Matplotlib style options. Individual plot functions return their
axes; `diagnostic_plots()` returns `(figure, axes)`.
