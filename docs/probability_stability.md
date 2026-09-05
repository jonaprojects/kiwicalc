# Probability and statistics stability

KiwiCalc validates its probability and statistics package at three levels:

1. Regression tests preserve documented behavior and previously fixed edge cases.
2. Mathematical invariant tests verify normalization, monotonic CDFs, quantile
   inversion, moment identities, covariance structure, and deterministic sampling.
3. Optional reference tests compare results with NumPy and SciPy and use Hypothesis
   to explore generated finite data sets.

## Public behavior

- Public scalar inputs return scalar results; NumPy array inputs preserve their
  broadcast or documented sample shape.
- `random_state=` accepts the existing KiwiCalc seed and generator conventions.
  The same seed produces the same output for the same KiwiCalc version and
  environment. Exact streams should not be assumed to match another library.
- Descriptive functions apply `axis=`, `weights=`, `ddof=`, and `nan_policy=` as
  documented. Invalid, empty, and constant samples raise or return a defined
  result rather than failing incidentally.
- Distribution probabilities are non-negative, CDFs are monotone and bounded by
  zero and one, and quantiles use the smallest supported value whose CDF reaches
  the requested probability for discrete distributions.
- Analytic distributions use closed-form formulas where available. Formula-defined
  continuous distributions use documented numerical integration and root-finding
  tolerances, so comparisons should use an appropriate numerical tolerance.

## Compatibility policy

Names exported from `kiwicalc` or `kiwicalc.probability` are public API. A public
name should not be removed or receive an incompatible signature change without a
deprecation period. Aliases remain useful for friendly discoverability, but the
canonical name shown in the documentation is preferred in new code.

KiwiCalc supports the Python and dependency versions declared in `pyproject.toml`.
The CI matrix includes the oldest supported Python version as well as the primary
Linux and Windows environments.

## Running validation

The normal development suite has no SciPy or Hypothesis requirement:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Install the optional validation tools to run the independent oracle suite:

```bash
python -m pip install -e ".[dev,validation]"
python -m pytest -m validation testing/test_probability_reference_validation.py
```

SciPy and Hypothesis are validation dependencies only; applications using
KiwiCalc do not install them.

## Adding statistical behavior

Every new method should include:

- deterministic reference examples;
- invalid, empty, constant, missing-value, scalar, and array cases where relevant;
- at least one mathematical invariant that does not repeat the implementation;
- an independent comparison when a trustworthy external reference exists;
- seeded tests for stochastic behavior, checking statistical properties separately
  from exact reproducibility.

