# Linear-algebra exercises

KiwiCalc's linear-algebra exercise family combines deterministic generation,
exact answer metadata, and native matrix/vector PDF notation. It does not
require an external LaTeX installation.

## Friendly factory

```python
exercise = kw.linear_algebra_exercise(
    'matrix multiplication',
    difficulty='hard',
    seed=42,
)
print(exercise.exercise)
print(exercise.solution)
print(exercise.data)
```

`difficulty` accepts `easy`, `medium`, or `hard`. Set `with_solution=False` for
a student copy. Common aliases such as `dot`, `norm`, `matmul`, `det`, `inverse`,
`linear system`, `rref`, `eigenvalue`, and `projection` are accepted.

The canonical names in `kw.LINEAR_ALGEBRA_EXERCISE_TYPES` are:

- `vector_arithmetic`, `dot_product`, `vector_magnitude`, `unit_vector`
- `matrix_arithmetic`, `scalar_matrix`, `matrix_multiplication`
- `determinant`, `inverse_matrix`
- `solve_linear_system`, `row_reduction`, `rank`
- `linear_independence`, `basis_coordinates`
- `eigenvalues`, `eigenvector`, `projection`, `linear_transformation`

## Compose a mixed worksheet

```python
sheet = kw.PDFWorksheet('Linear Algebra', theme='academic')
sheet.add_exercise(kw.PDFMatrixMultiplicationExercise(seed=1))
sheet.add_exercise(kw.PDFRowReduction(difficulty='hard', seed=2))
sheet.add_exercise(kw.PDFVectorProjection(seed=3))
sheet.end_page()
sheet.create('linear-algebra.pdf')
```

`end_page()` adds a coordinated solution page. Every canonical name also works
as a batch worksheet type:

```python
kw.worksheet(
    'determinants.pdf',
    dtype='determinant',
    equations_per_page=8,
    difficulty='hard',
    seed=42,
    theme='assessment',
)
```

## Exact metadata

Every exercise exposes `kind`, `difficulty`, and `data`. Depending on the topic,
the metadata includes operands, exact fractions, products, determinants,
augmented matrices, RREFs, pivot columns, eigenvalues, eigenvectors, projection
factors, and transformation results. Applications can check answers without
parsing the formatted question.
