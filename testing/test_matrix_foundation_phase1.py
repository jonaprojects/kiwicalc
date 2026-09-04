import numpy as np
import pytest

import kiwicalc as kw


def test_friendly_matrix_constructors():
    assert kw.Matrix.row([1, 2, 3]) == [[1, 2, 3]]
    assert kw.Matrix.column_vector([1, 2, 3]) == [[1], [2], [3]]
    assert kw.Matrix.zeros((2, 3)) == [[0, 0, 0], [0, 0, 0]]
    assert kw.Matrix.zeros(2, 3) == [[0, 0, 0], [0, 0, 0]]
    assert kw.Matrix.identity(3) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    with pytest.raises(ValueError):
        kw.Matrix.row([])
    with pytest.raises(ValueError):
        kw.Matrix.column_vector([])


def test_numpy_conversion_is_rectangular_and_detached():
    source = np.array([[1.0, 2.0], [3.0, 4.0]])
    matrix = kw.Matrix.from_numpy(source)
    source[0, 0] = 99
    assert matrix == [[1, 2], [3, 4]]

    converted = matrix.to_numpy()
    converted[0, 0] = 88
    assert matrix == [[1, 2], [3, 4]]
    assert matrix.to_numpy(dtype=float).dtype == np.dtype(float)
    with pytest.raises(TypeError, match='numpy.ndarray'):
        kw.Matrix.from_numpy([[1, 2]])
    with pytest.raises(ValueError, match='one- or two-dimensional'):
        kw.Matrix.from_numpy(np.zeros((1, 1, 1)))


def test_list_copy_transpose_shorthand_and_public_copy():
    matrix = kw.Matrix([[1, 2], [3, 4]])
    values = matrix.to_list()
    values[0][0] = 100
    assert matrix[0][0] == 1
    assert matrix.T == [[1, 3], [2, 4]]
    copied = matrix.copy()
    copied[0][0] = 100
    assert matrix[0][0] == 1


def test_rref_is_non_mutating_by_default_and_reports_pivots():
    matrix = kw.Matrix([[1, 2, 1], [2, 4, 2], [0, 1, 3]])
    original = matrix.copy()
    reduced = matrix.rref()
    assert matrix == original
    assert np.asarray(reduced.matrix) == pytest.approx(np.array([[1, 0, -5], [0, 1, 3], [0, 0, 0]]))
    assert matrix.pivot_columns() == (0, 1)
    assert matrix.rank() == 2


def test_rref_copy_false_is_explicitly_mutating():
    matrix = kw.Matrix([[1, 2], [2, 4]])
    returned = matrix.rref(copy=False)
    assert returned is matrix
    assert matrix == [[1, 2], [0, 0]]


def test_explicit_mutators_return_the_current_matrix():
    matrix = kw.Matrix([[1, 2], [3, 4]])
    assert matrix.replace_rows(0, 1) is matrix
    assert matrix.add_and_mul(0, 1, -3) is matrix
    assert matrix.multiply_all(2) is matrix
    assert matrix.divide_all(2) is matrix
    assert matrix.add_to_all(1) is matrix
    assert matrix.subtract_from_all(1) is matrix
    assert matrix.apply_to_all(lambda value: value) is matrix


def test_rref_tolerance_and_partial_pivoting():
    matrix = kw.Matrix([[1e-20, 1], [1, 1]])
    assert np.asarray(matrix.rref().matrix) == pytest.approx(np.eye(2))
    nearly_dependent = kw.Matrix([[1, 1], [1, 1 + 1e-12]])
    assert nearly_dependent.rank(tolerance=1e-10) == 1
    assert nearly_dependent.pivot_columns(tolerance=1e-10) == (0,)


def test_hadamard_is_explicit_and_non_mutating():
    left = kw.Matrix([[1, 2], [3, 4]])
    right = kw.Matrix([[2, 3], [4, 5]])
    assert left.hadamard(right) == [[2, 6], [12, 20]]
    assert left == [[1, 2], [3, 4]]
    assert left.hadamard_inplace(right) is left
    assert left == [[2, 6], [12, 20]]
    with pytest.raises(ValueError, match='different shapes'):
        left.hadamard([[1]])


def test_arithmetic_accepts_rectangular_numpy_arrays():
    matrix = kw.Matrix([[1, 2], [3, 4]])
    other = np.array([[2, 3], [4, 5]])
    assert matrix + other == [[3, 5], [7, 9]]
    assert matrix - other == [[-1, -1], [-1, -1]]
    assert matrix * other == [[2, 6], [12, 20]]
    divided = (matrix / other).to_numpy()
    assert divided == pytest.approx(np.array([[0.5, 2 / 3], [0.75, 0.8]]))
