# Geometry and vector exercises

KiwiCalc provides deterministic coordinate-geometry, measurement, solid-
geometry, and vector-geometry exercises with exact solutions and structured
answer metadata. Vector notation uses KiwiCalc's native PDF renderer, so no
external LaTeX installation is required.

## Friendly factory

```python
exercise = kw.geometry_exercise(
    'distance between points',
    difficulty='hard',
    seed=42,
)
print(exercise.exercise)
print(exercise.solution)
print(exercise.data)
```

`difficulty` accepts `easy`, `medium`, or `hard`. Set `with_solution=False` for
a student copy. Friendly aliases include `gradient`, `centroid`, `sector`,
`solid`, `transform point`, `cross`, `vector equation`, and `plane`.

The canonical names in `kw.GEOMETRY_EXERCISE_TYPES` are:

- `distance`, `midpoint`, `slope`, `line_equation`, `point_line_distance`
- `parallel_perpendicular`, `triangle_area`, `triangle_centroid`, `pythagorean`
- `circle_equation`, `arc_sector`, `polygon_angles`, `solid_measurement`
- `coordinate_transformation`, `vector_from_points`, `vector_relationship`
- `vector_angle`, `cross_product`, `vector_line`, `plane_equation`

## Compose a worksheet

```python
sheet = kw.PDFWorksheet('Geometry and Vectors', theme='academic')
sheet.add_exercise(kw.PDFTriangleArea(seed=1))
sheet.add_exercise(kw.PDFVectorAngle(seed=2))
sheet.add_exercise(kw.PDFPlaneEquation(difficulty='hard', seed=3))
sheet.end_page()
sheet.create('geometry.pdf')
```

Every canonical type works with the batch helper too:

```python
kw.worksheet(
    'circles.pdf',
    dtype='circle_equation',
    equations_per_page=10,
    difficulty='medium',
    seed=42,
    theme='assessment',
)
```

## Exact metadata

Every object exposes `kind`, `difficulty`, and `data`. The stored data includes
coordinates, line and plane coefficients, exact fractions, squared distances,
Pythagorean triples, coefficients of pi, transformation names, dot products,
and cross products. Interactive applications can check answers without parsing
the displayed question.
