import math

import numpy as np
import pytest

import kiwicalc as kw
from kiwicalc.linalg.matrix import (
    approximate_jacobian,
    broyden,
    generate_jacobian,
    generate_polynomial_matrix,
)
from kiwicalc.linalg.spaces import copy


def test_matrix_keeps_one_rectangular_storage_model():
    matrix = kw.Matrix([1, 2, 3])
    assert matrix.shape == (1, 3)
    assert matrix.matrix == [[1, 2, 3]]
    assert matrix.sum() == 6
    assert matrix.transpose() == [[1], [2], [3]]


@pytest.mark.parametrize('data', [[], [[]], [[1, 2], [3]]])
def test_matrix_rejects_empty_and_ragged_data(data):
    with pytest.raises(ValueError):
        kw.Matrix(data)


def test_matrix_dimension_validation_is_explicit():
    with pytest.raises(TypeError):
        kw.Matrix(dimensions={2, 3})
    with pytest.raises(ValueError, match='2x3'):
        kw.Matrix(dimensions='2 by 3')
    with pytest.raises(ValueError, match='positive'):
        kw.Matrix(dimensions=(0, 2))
    with pytest.raises(ValueError, match='integers'):
        kw.Matrix(dimensions=(2.5, 3))


def test_matrix_row_operations_are_consistently_zero_based():
    matrix = kw.Matrix([[1, 2], [3, 4]])
    assert matrix.multiply_row(2, 0) is matrix
    assert matrix.divide_row(2, 1) is matrix
    assert matrix == [[2, 4], [1.5, 2]]
    with pytest.raises(IndexError):
        matrix.multiply_row(2, -1)
    with pytest.raises(IndexError):
        matrix.divide_row(2, 2)


def test_matrix_division_accepts_another_matrix():
    assert kw.Matrix([[4, 6]]) / kw.Matrix([[2, 3]]) == [[2, 2]]


def test_matrix_numeric_algorithms_pivot_and_honor_tolerance():
    matrix = kw.Matrix([[1e-320, 1, 0], [1, 1, 0], [0, 0, 1]])
    assert matrix.determinant() == pytest.approx(-1)
    inverse = matrix.inverse()
    assert inverse is not None
    assert np.asarray((matrix @ inverse).matrix) == pytest.approx(np.eye(3))
    nearly_dependent = kw.Matrix([[1, 1], [1, 1 + 1e-12]])
    assert nearly_dependent.get_rank(tolerance=1e-10) == 1


def test_matrix_mutation_preserves_shape_invariants():
    matrix = kw.Matrix([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        matrix[0] = [1]
    del matrix[0]
    assert matrix.shape == (1, 2)
    with pytest.raises(ValueError, match='only row'):
        del matrix[0]
    with pytest.raises(TypeError, match='integers'):
        del matrix[:]


def test_general_and_approximate_jacobians_support_rectangular_inputs():
    symbolic = generate_jacobian([kw.Poly('x+y')], ('x', 'y'))
    assert len(symbolic) == 1 and len(symbolic[0]) == 2
    numeric = approximate_jacobian([lambda x, y: x * y], (2, 3))
    assert numeric[0] == pytest.approx([3, 2], rel=1e-6)
    with pytest.raises(ValueError, match='positive'):
        approximate_jacobian([lambda x: x], [1], h=0)
    with pytest.raises(ValueError, match='At least one'):
        generate_polynomial_matrix(iter(()))


def test_broyden_solves_nontrivial_system_and_validates_failures():
    solution = broyden([lambda x: x * x - 2], [1.0], epsilon=1e-10, nmax=30)
    assert solution == pytest.approx([math.sqrt(2)], rel=1e-9)
    with pytest.raises(ValueError, match='one function per unknown'):
        broyden([lambda x, y: x + y], [1, 1])
    with pytest.raises(ValueError, match='singular'):
        broyden([lambda x: x * x - 1], [0.0])


def test_plane_representation_conversion_and_equality_are_mathematical():
    assert str(kw.Surface((1, 2, 3, 4))) == 'x+2y+3z+4=0'
    assert str(kw.Surface((1, -2, -3, -4))) == 'x-2y-3z-4=0'
    assert kw.Surface((1, 2, 3, 4)) == kw.Surface((2, 4, 6, 8))
    with pytest.raises(ValueError, match='vertical plane'):
        kw.Surface((1, 0, 0, -2)).to_lambda()
    with pytest.raises(ValueError, match='non-zero normal'):
        kw.Surface((0, 0, 0, 1))


def test_plane_vector_intersection_handles_all_geometric_cases():
    plane = kw.Surface((0, 0, 1, 0))
    crossing = kw.Vector((0, 0, -1), start_coordinate=(1, 2, 3))
    assert plane.intersection(crossing) == pytest.approx([1, 2, 0])
    assert plane.intersection(crossing, get_point=True).coordinates == pytest.approx([1, 2, 0])
    parallel = kw.Vector((1, 0, 0), start_coordinate=(0, 0, 1))
    contained = kw.Vector((1, 0, 0), start_coordinate=(0, 0, 0))
    assert plane.intersection(parallel) is None
    assert plane.intersection(contained) is None


def test_copy_context_manager_supports_plain_objects():
    original = object()
    with copy(original) as copied:
        assert copied is not original
