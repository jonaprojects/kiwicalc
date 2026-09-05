# Statistical inference

KiwiCalc's inference API uses two predictable result types:

- `ConfidenceInterval` contains `lower`, `upper`, `estimate`, `standard_error`,
  `confidence`, and `method`.
- `TestResult` contains `statistic`, `p_value`, `method`, `alternative`, degrees
  of freedom, estimates, an optional confidence interval, and effect size.

```python
import kiwicalc as kw

sample = [18.2, 17.9, 18.5, 18.1, 18.4]

interval = kw.confidence_interval(sample)
interval.lower, interval.upper
interval.contains(18)

test = kw.mean_test(sample, expected=18)
test.statistic
test.p_value
test.significant(alpha=0.05)
test.confidence_interval
```

`pvalue` aliases `p_value`, and `reject_null(alpha)` aliases `significant(alpha)`.
Results can be converted with `as_tuple()` or `as_dict()`.

## Means

Without a known population standard deviation, mean inference uses Student's t
distribution:

```python
kw.mean_confidence_interval(sample, confidence=0.95)
kw.mean_test(sample, expected=18, alternative="greater")
```

Pass known `sigma=` to select z inference:

```python
kw.confidence_interval(sample, sigma=0.4)
kw.mean_test(sample, expected=18, sigma=0.4)
```

Two independent samples use Welch's test by default because it does not assume
equal variances:

```python
kw.compare_means(control, treatment)
kw.compare_means(control, treatment, equal_variance=True)
kw.compare_means(before, after, paired=True)
```

`one_sample_t_test` and `two_sample_t_test` are familiar aliases. Mean-test effect
sizes are standardized mean differences. Two-sample estimates and confidence
intervals describe `mean(first) - mean(second)`.

One-sample mean functions accept `axis=` and
`nan_policy="raise" | "omit" | "propagate"`. Two-sample tests accept the same
controls. Paired tests remove a pair together because they operate on paired
differences.

## Proportions

```python
kw.proportion_confidence_interval(successes=63, trials=100)
kw.proportion_test(63, 100, expected=0.5)
kw.compare_proportions(63, 100, 51, 100)
```

The default Wilson score interval behaves substantially better near zero and one
than a plain normal interval. Use `method="wald"` only when that conventional
approximation is specifically required. Counts can be NumPy arrays and are
broadcast normally.

## Categorical inference

Goodness of fit accepts expected counts or probabilities. When omitted, equal
cell probabilities are assumed:

```python
fit = kw.chi_square_test(
    observed=[18, 30, 52],
    expected=[0.2, 0.3, 0.5],
)
```

Independence accepts a count matrix or KiwiCalc `ContingencyTable`:

```python
table = kw.contingency_table(methods, outcomes)
independence = kw.chi_square_independence(table)
independence.details["expected"]
```

`effect_size` is Cohen's w for goodness of fit and Cramer's V for independence.
Yates' correction is opt-in with `correction=True` and applies only to 2-by-2
tables.

## ANOVA and correlation

```python
kw.one_way_anova(group_a, group_b, group_c)
kw.anova([group_a, group_b, group_c])

kw.correlation_test(x, y)
kw.correlation_test(x, y, method="spearman", alternative="greater")
```

One-way ANOVA reports `(between, within)` degrees of freedom and eta-squared as
its effect size. Correlation tests report the correlation itself as both the
estimate and effect size. Spearman significance uses the common t approximation.

## Alternatives and assumptions

Tests accept `alternative="two-sided"`, `"less"`, or `"greater"`; `"lower"`,
`"upper"`, and `"two-tailed"` are readable aliases. One-sided mean and
proportion comparisons return a confidence interval with the unused endpoint at
infinity.

These procedures calculate conventional parametric inferences. Users remain
responsible for sampling design and assumptions such as independence, appropriate
normality, expected chi-square cell sizes, and ANOVA residual behavior. KiwiCalc
uses NumPy plus internal probability calculations, so SciPy is not required.
