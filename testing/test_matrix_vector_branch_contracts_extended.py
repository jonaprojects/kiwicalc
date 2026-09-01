import importlib

import matplotlib.pyplot as plt
import pytest

import kiwicalc as kw
from kiwicalc.linalg.matrix import (
    approximate_jacobian,
    broyden,
    generate_jacobian,
    generate_polynomial_matrix,
)


def test_flat_matrix_mapping_copy_and_aggregates():
    matrix = kw.Matrix((3, 1, 2))
    assert matrix.max() == 3
    assert matrix.min() == 1
    matrix.apply_to_all(lambda value: value * 2)
    assert matrix.matrix == [6, 2, 4]
    copied = matrix.__copy__()
    assert copied.matrix == matrix.matrix and copied is not matrix


def test_matrix_scalar_helpers_and_rank_mutation_branch():
    matrix = kw.Matrix([[1, 2], [3, 4]])
    matrix.add_to_all(1)
    matrix.subtract_from_all(1)
    matrix.multiply_all(2)
    matrix.divide_all(2)
    assert matrix == [[1, 2], [3, 4]]
    assert matrix.get_rank(copy=False) == 2
    with pytest.raises(ValueError):
        matrix.divide_all(0)


def test_matrix_gauss_zero_pivot_and_determinant_branches():
    swapped = kw.Matrix([[0, 1], [2, 3]])
    swapped.gauss()
    assert swapped == kw.Matrix.unit_matrix(2)
    singular = kw.Matrix([[0, 0], [0, 0]])
    singular.gauss()
    assert singular.get_rank() == 0
    three = kw.Matrix([[1, 2, 3], [0, 4, 5], [1, 0, 6]])
    assert three.determinant() == pytest.approx(22)
    ranked = three.__copy__()
    assert ranked.determinant(rank=True) == pytest.approx(22)
    assert kw.Matrix([[1, 2, 3], [2, 4, 6], [0, 1, 1]]).determinant() == 0


def test_matrix_arithmetic_scalar_list_and_division_branches():
    matrix = kw.Matrix([[2, 4], [6, 8]])
    assert matrix + 1 == [[3, 5], [7, 9]]
    assert matrix - 1 == [[1, 3], [5, 7]]
    assert matrix * 2 == [[4, 8], [12, 16]]
    assert matrix * [[2, 2], [2, 2]] == [[4, 8], [12, 16]]
    assert matrix / [[2, 4], [3, 2]] == [[1, 1], [2, 4]]
    with pytest.raises(ValueError, match="different shapes"):
        matrix / [[1]]
    with pytest.raises(ZeroDivisionError, match="containing zero"):
        matrix / [[1, 0], [1, 1]]
    with pytest.raises(TypeError):
        matrix / object()


def test_matrix_equality_false_branches():
    matrix = kw.Matrix([[1, 2], [3, 4]])
    assert matrix != [[1, 2]]
    assert matrix != kw.Matrix([[1], [3]])
    assert matrix != [[1, 9], [3, 4]]
    assert matrix == ((1, 2), (3, 4))


def test_matrix_add_subtract_flexible_and_warning_branches():
    assert kw.Matrix([[1, 2]]).add([[3, 4]]) == [[4, 6]]
    assert kw.Matrix([[4, 6]]).subtract(((1, 2),)) == [[3, 4]]
    with pytest.warns(UserWarning, match="different"):
        assert kw.Matrix([[1, 2]]).add([[1]]) is None
    with pytest.warns(UserWarning, match="Expected types"):
        assert kw.Matrix([[1]]).add(object()) is None
    with pytest.warns(UserWarning, match="different"):
        assert kw.Matrix([[1, 2]]).subtract([[1]]) is None
    with pytest.warns(UserWarning, match="Expected types"):
        assert kw.Matrix([[1]]).subtract(object()) is None


def test_matrix_filter_map_foreach_and_ordering_protocols():
    matrix = kw.Matrix([[1, 2], [3, 4]])
    assert matrix.filtered_matrix(lambda value: value % 2 == 0, copy=True, get_list=True) == [[2], [4]]
    assert matrix.filtered_matrix(lambda value: value > 2, copy=False) == [[], [3, 4]]
    assert matrix.mapped_matrix(lambda value: value * 10) == [[10, 20], [30, 40]]
    assert matrix.foreach_item(lambda value: value + 1) == [[2, 3], [4, 5]]
    assert matrix.filter_by_indices(lambda row, column: row == column) == [[2], [5]]
    assert matrix.reversed_columns() == [[3, 2], [5, 4]]
    assert matrix.reversed_rows() == [[4, 5], [2, 3]]
    assert list(matrix.iterate_by_columns()) == [2, 4, 3, 5]
    assert list(matrix.range()) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert reversed(matrix) == [[4, 5], [2, 3]]


def test_matrix_random_shape_and_identity_failure_branches(monkeypatch):
    values = iter((2, 3))
    monkeypatch.setattr("random.randint", lambda start, stop: next(values) if (start, stop) == (1, 5) else start)
    assert kw.Matrix.random_matrix(shape=None).shape == (2, 3)
    assert not kw.Matrix.is_unit_matrix(kw.Matrix([[1, 0, 0], [0, 1, 0]]))
    assert not kw.Matrix.is_unit_matrix(kw.Matrix([[2, 0], [0, 1]]))
    assert not kw.Matrix.is_unit_matrix(kw.Matrix([[1, 2], [0, 1]]))


def test_matrix_inverse_pivot_singular_and_nonsquare_branches():
    assert kw.Matrix([[0, 1], [1, 0]]).inverse() == [[0, 1], [1, 0]]
    assert kw.Matrix([[0, 0], [0, 1]]).inverse() is None
    assert kw.Matrix([[1, 2, 3], [4, 5, 6]]).inverse() is None


class CopyOnly:
    def __init__(self, value):
        self.value = value

    def copy(self):
        return CopyOnly(self.value)


def test_matrix_copy_item_protocol_branches():
    expression = kw.Var("x")
    copy_only = CopyOnly(3)
    matrix = kw.Matrix([[expression, copy_only, 4]])
    copied = matrix.__copy__()
    assert copied[0][0] == expression and copied[0][0] is not expression
    assert copied[0][1].value == 3 and copied[0][1] is not copy_only
    assert copied[0][2] == 4


def test_jacobian_and_polynomial_matrix_branches():
    functions = [kw.Poly("x^2+y"), kw.Poly("x-y^2")]
    jacobian = generate_jacobian(functions, ("x", "y"))
    assert len(jacobian) == 2 and len(jacobian[0]) == 2
    with pytest.raises(ValueError, match="equal number"):
        generate_jacobian(functions, ("x",))
    approximate = approximate_jacobian(
        [lambda x, y: x + y, lambda x, y: x * y], [2.0, 3.0]
    )
    assert approximate[0] == pytest.approx([1, 1], abs=1e-3)
    assert approximate[1] == pytest.approx([3, 2], abs=1e-3)
    assert generate_polynomial_matrix(("x=1", "y=2")).shape == (2, 2)
    assert generate_polynomial_matrix(functions).shape == (2, 2)
    assert broyden((lambda x: x,), [0.0]) == [0.0]


def test_vector_ratio_scalar_and_random_dimension_branches(monkeypatch):
    assert not kw.Vector([]).equal_direction_ratio(kw.Vector([]))
    assert kw.Vector((2,)).equal_direction_ratio(kw.Vector((2,)))
    assert not kw.Vector((0, 1)).equal_direction_ratio(kw.Vector((1, 1)))
    with pytest.raises(TypeError):
        kw.Vector((1, 2)).scalar_product(object())
    values = iter((4, 1, 2, 3, 4, 5, 6, 7, 8))
    monkeypatch.setattr("random.randint", lambda start, stop: next(values))
    assert len(kw.Vector.random_vector((0, 9)).direction) == 4


def test_vector_intersection_all_dispatch_branches(capsys):
    first = kw.Vector((1, 1), start_coordinate=(0, 0))
    parallel = kw.Vector((1, 1), start_coordinate=(1, 0))
    assert first.intersection(parallel) is None
    assert "same directions" in capsys.readouterr().out
    second = kw.Vector((1, -1), start_coordinate=(0, 1))
    assert first.intersection(second) == pytest.approx([0.5, 0.5])
    point = first.intersection(second, get_points=True)
    assert point.coordinates == pytest.approx([0.5, 0.5])
    collection = kw.VectorCollection(parallel, second)
    assert first.intersection(collection)
    with pytest.raises(TypeError):
        first.intersection(object())


def test_vector_equality_collection_and_arithmetic_dispatch():
    vector = kw.Vector((1, 2))
    assert vector != None
    assert vector == kw.VectorCollection(kw.Vector((1, 2)))
    assert vector != kw.VectorCollection(kw.Vector((1, 2)), kw.Vector((1, 2)))
    assert vector + 2 == kw.Vector((3, 4))
    assert vector - 1 == kw.Vector((0, 1))
    assert (kw.VectorCollection(kw.Vector((3, 4))) + vector).num_of_vectors == 2
    assert (vector + kw.VectorCollection(kw.Vector((3, 4)))).num_of_vectors == 2
    assert (vector - kw.VectorCollection(kw.Vector((3, 4)))).num_of_vectors == 2
    assert (3 - vector) == kw.Vector((2, 1))
    assert vector.power_by_vector(iter((2, 3))) == [[1, 1], [4, 8]]


def test_vector_collection_nested_plot_validation_and_equality(monkeypatch):
    nested = kw.VectorCollection(kw.Vector((1, 2)))
    collection = kw.VectorCollection()
    collection.append([nested, kw.Vector((3, 4))])
    assert collection.num_of_vectors == 2
    collection.vectors = [(5, 6), (7, 8)]
    assert collection.num_of_vectors == 2
    with pytest.raises(TypeError):
        kw.VectorCollection(object())
    with pytest.raises(ValueError):
        collection.nlongest(3)
    with pytest.raises(ValueError):
        collection.nshortest(-1)
    assert collection != None
    assert collection != kw.VectorCollection(kw.Vector((5, 6)))
    assert collection != kw.VectorCollection(kw.Vector((5, 6)), kw.Vector((9, 9)))
    assert not (collection == [[object()]])
    with pytest.raises(TypeError):
        _ = collection == object()
    with pytest.raises(TypeError):
        _ = collection == kw.Vector((1, 2))
    assert (object() in collection) is False
    called = []
    monkeypatch.setattr(plt, "show", lambda: called.append(True))
    kw.VectorCollection().plot()
    kw.VectorCollection(kw.Vector((1, 2))).plot()
    kw.VectorCollection(kw.Vector((1, 2, 3))).plot()
    assert called == [True, True, True]
    plt.close("all")
