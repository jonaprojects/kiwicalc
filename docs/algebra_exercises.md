# Core algebra exercises

KiwiCalc's core algebra generators produce render-ready questions, exact
solutions, and stable metadata. They use local random generators, so supplying a
seed never changes Python's global random state.

## Friendly factory

```python
exercise = kw.algebra_exercise('factor', difficulty='hard', seed=42)
print(exercise.exercise)
print(exercise.solution)
print(exercise.data)
```

`difficulty` accepts `easy`, `medium`, or `hard`. Set `with_solution=False` for
a student-only exercise. Friendly aliases such as `factoring`, `evaluate`,
`inequality`, and `rearrange formula` are accepted by `algebra_exercise()`.

The canonical exercise names are:

- `simplify`
- `expand`
- `factor`
- `complete_square`
- `substitution`
- `linear_inequality`
- `absolute_value`
- `exponent_laws`
- `rational`
- `radical`
- `rearrange`

Use `kw.ALGEBRA_EXERCISE_TYPES` when building a menu or iterating over the
catalog.

## Direct classes

Every generator also has a class: `PDFSimplifyExpression`,
`PDFExpandExpression`, `PDFFactorPolynomial`, `PDFCompleteSquare`,
`PDFSubstitution`, `PDFLinearInequality`, `PDFAbsoluteValueEquation`,
`PDFExponentLaws`, `PDFRationalEquation`, `PDFRadicalEquation`, and
`PDFRearrangeFormula`. All derive from `PDFAlgebraExercise` and can be added
directly to `PDFWorksheet`.

```python
sheet = kw.PDFWorksheet('Algebra', theme='classroom')
sheet.add_exercise(kw.PDFLinearInequality(difficulty='hard', seed=4))
sheet.add_exercise(kw.PDFRadicalEquation(seed=5))
sheet.end_page()
sheet.create('algebra.pdf')
```

## Batch worksheets

The canonical names also work as `worksheet()` dtypes:

```python
kw.worksheet(
    'factoring.pdf',
    dtype='factor',
    equations_per_page=12,
    difficulty='hard',
    seed=42,
    get_solutions=True,
    theme='assessment',
)
```

Batch answer pages use the same automatic numbering, mathematical formatting,
themes, and document layout as existing KiwiCalc worksheets.

## Metadata

Each exercise exposes `kind`, `difficulty`, and `data`. The `data` dictionary
contains the generated coefficients, roots, exponents, restrictions, or other
source values used to prove the answer. It is intended for tests, interactive
answer checking, and future non-PDF renderers; callers should not parse the
displayed question text to recover mathematical values.
