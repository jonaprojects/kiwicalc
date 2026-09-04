# Matrices and vectors in PDF documents

KiwiCalc renders matrix and vector notation without an external LaTeX
installation. The renderer aligns each column independently and draws brackets
itself, while Matplotlib Mathtext handles individual entries such as fractions
and powers.

## Inline notation

Use `PDFMatrix` and `PDFVector` wherever a `PDFText` part is accepted:

```python
A = kw.Matrix([[2, 1], [-3, 4]])
x = kw.Vector((1, -2))

question = kw.PDFText(
    'Calculate ', kw.PDFMatrix(A), kw.PDFVector(x), '.'
)
```

Ordinary sequences and NumPy arrays work too. A vector is a column by default:

```python
kw.PDFVector([1, 2, 3])
kw.PDFVector([1, 2, 3], orientation='row')
```

## Centered display notation

The worksheet helpers construct and add the corresponding block in one call:

```python
sheet = kw.PDFWorksheet('Linear Algebra', theme='academic')
sheet.add_matrix([[1, 2], [3, 4]])
sheet.add_vector([5, 6])
sheet.create('linear-algebra.pdf')
```

Both helpers return the worksheet, so they can be chained.

## Brackets and entries

Choose `square`, `round`, `determinant`, or `none`:

```python
kw.PDFMatrix(A, brackets='round')
kw.PDFMatrix(A, brackets='determinant')
```

Entries may be integers, finite floats, exact `Fraction` values, complex
numbers, supported KiwiCalc expressions, or explicit Mathtext strings. Matrices
must be rectangular. To preserve readable pages and bounded memory use, display
objects are limited to 30 rows and 30 columns; large matrices should be
summarized before rendering.
