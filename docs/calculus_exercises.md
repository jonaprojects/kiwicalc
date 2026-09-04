# Calculus and numerical-method exercises

KiwiCalc provides a single, friendly interface for symbolic calculus and
hand-worked numerical-method questions. Every exercise has an exact or
reproducible answer, three difficulty levels, optional solutions, and structured
metadata for answer checking.

## Friendly factory

```python
exercise = kw.calculus_exercise('newton', difficulty='hard', seed=42)
print(exercise.exercise)
print(exercise.solution)
print(exercise.data)
```

`difficulty` accepts `easy`, `medium`, or `hard`. Set `with_solution=False` for
a student copy. Natural aliases such as `first principles`, `integral`,
`central difference`, `trapezoid`, `newton`, `euler`, and `rk4` are accepted.

The canonical names in `kw.CALCULUS_EXERCISE_TYPES` are:

- `difference_quotient`, `derivative`, `tangent_line`
- `critical_points`, `monotonicity`, `concavity`, `optimization`
- `definite_integral`, `area_between`
- `numerical_derivative`, `trapezoidal_rule`, `simpson_rule`
- `newton_iteration`, `euler_method`, `runge_kutta`

## Compose a worksheet

```python
sheet = kw.PDFWorksheet('Calculus and Numerical Methods', theme='academic')
sheet.add_exercise(kw.PDFDerivativeExercise(seed=1))
sheet.add_exercise(kw.PDFNewtonIteration(difficulty='hard', seed=2))
sheet.add_exercise(kw.PDFRungeKuttaMethod(difficulty='hard', seed=3))
sheet.end_page()
sheet.create('calculus.pdf')
```

`end_page()` adds a coordinated solution page. The same exercise names also
work with the batch helper:

```python
kw.worksheet(
    'simpson-practice.pdf',
    dtype='simpson_rule',
    equations_per_page=8,
    difficulty='medium',
    seed=42,
    theme='assessment',
)
```

## Exact metadata

Each object exposes `kind`, `difficulty`, and `data`. Polynomial coefficients,
bounds, sample values, iteration histories, RK4 stages, exact fractions, and
errors are stored there. This makes the exercises usable in notebooks and
interactive answer checkers without parsing display text.

The numerical exercises deliberately teach the algorithm: central differences,
composite trapezoidal and Simpson rules, Newton iteration, Euler's method, and
classical RK4 all expose their intermediate values.
