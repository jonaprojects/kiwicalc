import math

import pytest

import kiwicalc as kw


def test_matrix_transpose_and_aggregate_contracts():
    matrix = kw.Matrix([[1, 2, 3], [4, 5, 6]])
    assert matrix.transpose() == [[1, 4], [2, 5], [3, 6]]
    assert matrix.sum() == 21
    assert matrix.average() == pytest.approx(3.5)
    assert matrix.average_in_line(0) == pytest.approx(2)
    assert matrix.average_in_column(1) == pytest.approx(3.5)


def test_matrix_arithmetic_does_not_mutate_operands():
    matrix = kw.Matrix([[1, 2], [3, 4]])
    other = kw.Matrix([[4, 3], [2, 1]])
    assert matrix + other == [[5, 5], [5, 5]]
    assert matrix - other == [[-3, -1], [1, 3]]
    assert matrix - 1 == [[0, 1], [2, 3]]
    assert matrix == [[1, 2], [3, 4]]
    assert other == [[4, 3], [2, 1]]


def test_matrix_product_and_identity():
    matrix = kw.Matrix([[1, 2], [3, 4]])
    identity = kw.Matrix.unit_matrix(2)
    assert matrix @ identity == matrix
    assert identity @ matrix == matrix


def test_matrix_product_rejects_incompatible_shapes():
    with pytest.raises(ValueError):
        kw.Matrix([[1, 2, 3], [4, 5, 6]]) @ kw.Matrix([[1, 2], [3, 4]])


def test_matrix_determinant_inverse_and_rank():
    matrix = kw.Matrix([[1, 2], [3, 4]])
    inverse = matrix.inverse()
    assert matrix.determinant() == pytest.approx(-2)
    assert matrix @ inverse == kw.Matrix.unit_matrix(2)
    assert matrix.get_rank() == 2
    assert kw.Matrix([[1, 2], [2, 4]]).get_rank() == 1


def test_kronecker_product():
    left = kw.Matrix([[1, 2], [3, 4]])
    right = kw.Matrix([[0, 5], [6, 7]])
    assert left.kronecker(right) == [
        [0, 5, 0, 10],
        [6, 7, 12, 14],
        [0, 15, 0, 20],
        [18, 21, 24, 28],
    ]


def test_point_distance_is_symmetric_and_validates_dimensions():
    first = kw.Point2D(0, 0)
    second = kw.Point2D(3, 4)
    assert first.distance(second) == pytest.approx(5)
    assert second.distance(first) == pytest.approx(5)
    with pytest.raises(ValueError):
        first.distance(kw.Point3D(0, 0, 0))


def test_line_properties():
    line = kw.Line2D((0, 1), (2, 5))
    assert line.middle() == kw.Point2D(1, 3)
    assert line.length() == pytest.approx(math.sqrt(20))
    assert line.slope == pytest.approx(2)
    assert line.free_number == pytest.approx(1)
    assert line.to_lambda()(3) == pytest.approx(7)


def test_circle_measurements_and_containment():
    circle = kw.Circle(2, center=(1, 1))
    assert circle.area() == pytest.approx(4 * math.pi)
    assert circle.perimeter() == pytest.approx(4 * math.pi)
    assert circle.point_inside((1, 3))
    assert not circle.point_inside((4, 1))
    assert kw.Circle(1, center=(1, 1)).is_inside(circle)


def test_vector_construction_products_and_direction():
    vector = kw.Vector(start_coordinate=(1, 2), end_coordinate=(4, 6))
    assert vector.direction == [3, 4]
    assert vector.length() == pytest.approx(5)
    assert vector.scalar_product((2, -1)) == 2
    assert vector.equal_direction_ratio(kw.Vector((6, 8)))
    assert not vector.equal_direction_ratio(kw.Vector((6, -8)))
    assert kw.Vector.fill_ones(3).direction == [1, 1, 1]


def test_vector_collection_length_queries():
    shortest = kw.Vector((1, 0))
    middle = kw.Vector((3, 4))
    longest = kw.Vector((6, 8))
    vectors = kw.VectorCollection(middle, longest, shortest)
    assert vectors.shortest() == shortest
    assert vectors.longest() == longest
    assert vectors.nshortest(2) == [shortest, middle]
    assert vectors.nlongest(2) == [longest, middle]
