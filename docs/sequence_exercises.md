# Sequence and series exercises

KiwiCalc provides deterministic exercises for progressions, recursive sequences,
sigma notation, limits, and introductory convergence tests. Formulas use the
built-in math renderer, so an external LaTeX installation is not required.

## Friendly factory

```python
exercise = kw.sequence_exercise(
    'infinite geometric sum',
    difficulty='hard',
    seed=42,
)
print(exercise.exercise)
print(exercise.solution)
print(exercise.data)
```

`difficulty` accepts `easy`, `medium`, or `hard`. Set `with_solution=False` for
a student-only exercise. Friendly names include `common difference`, `sigma`,
`limit`, `pseries`, `alternating test`, and `telescoping`.

The canonical names in `kw.SEQUENCE_SERIES_EXERCISE_TYPES` are:

- `identify_sequence`, `arithmetic_next_terms`, `arithmetic_nth_term`
- `arithmetic_difference`, `arithmetic_sum`, `arithmetic_missing_term`
- `geometric_next_terms`, `geometric_nth_term`, `geometric_ratio`
- `geometric_sum`, `infinite_geometric_sum`
- `recursive_sequence`, `fibonacci`, `sigma_evaluation`
- `sequence_limit`, `convergence_classification`, `p_series`
- `geometric_series_test`, `alternating_series`, `telescoping_series`
- `elementary_limit`, `euler_limit`, `removable_limit`, `standard_trig_limit`

## Compose a worksheet

```python
sheet = kw.PDFWorksheet('Sequences and Series', theme='academic')
sheet.add_exercise(kw.PDFArithmeticNthTerm(seed=1))
sheet.add_exercise(kw.PDFInfiniteGeometricSum(seed=2))
sheet.add_exercise(kw.PDFAlternatingSeries(difficulty='hard', seed=3))
sheet.end_page()
sheet.create('sequences.pdf')
```

Every canonical type also works with the batch helper:

```python
kw.worksheet(
    'limits.pdf',
    dtype='sequence_limit',
    equations_per_page=10,
    difficulty='medium',
    seed=42,
    theme='assessment',
)
```

## Polynomial-quotient limits

`sequence_limit` generates general polynomial quotients and classifies them by
their dominant powers. Random exercises can have a finite limit, tend to positive
or negative infinity, or genuinely fail to have a limit because of an oscillating
factor. Request a particular teaching case when needed:

```python
kw.sequence_exercise('limit', case='finite_zero', seed=1)
kw.sequence_exercise('limit', case='finite_ratio', seed=2)
kw.sequence_exercise('limit', case='positive_infinity', seed=3)
kw.sequence_exercise('limit', case='negative_infinity', seed=4)
kw.sequence_exercise('limit', case='oscillating', seed=5)
```

Friendly case aliases include `zero`, `finite`, `+infinity`, `-infinity`,
`does not exist`, and `dne`. The metadata distinguishes `exists`, `converges`,
`behavior`, and `limit`; divergence to infinity is not mislabeled as oscillation.

## Deterministic closed-form limits

Direct elementary limits are generated only at points where the selected
function is defined and continuous. The available subtypes are listed by
`kw.PDFElementaryFunctionLimit.FUNCTIONS`:

```python
kw.sequence_exercise('elementary_limit', function='polynomial', seed=1)
kw.sequence_exercise('elementary_limit', function='rational', seed=2)
kw.sequence_exercise('elementary_limit', function='sqrt', seed=3)
kw.sequence_exercise('elementary_limit', function='exp', seed=4)
kw.sequence_exercise('elementary_limit', function='log', seed=5)
kw.sequence_exercise('elementary_limit', function='sin', seed=6)
kw.sequence_exercise('elementary_limit', function='cos', seed=7)
kw.sequence_exercise('elementary_limit', function='abs', seed=8)
```

Polynomial, rational, square-root, and absolute-value answers are exact numeric
values. Exponential, logarithmic, and special-angle trigonometric answers retain
exact symbolic notation and also expose `numeric_result` for checking.

Euler limits use the closed form `(1 + a/n)^(bn) -> e^(ab)`, including rational
values of `a` at harder levels:

```python
kw.sequence_exercise('euler_limit', difficulty='hard', seed=42)
```

`removable_limit` creates a common polynomial factor that cancels at the target
point. Hard exercises show expanded quadratics. `standard_trig_limit` uses scaled
sine, tangent, and one-minus-cosine identities; select one with `form='sine'`,
`form='tangent'`, or `form='cosine'`.

## Exact metadata

Every generated object exposes `kind`, `difficulty`, and `data`. Metadata holds
the generated terms and parameters, the exact answer, and convergence details.
Finite limits and sums use `fractions.Fraction` whenever needed, so educational
apps can check answers without parsing the displayed text or losing precision.
