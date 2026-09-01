import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

import kiwicalc as kw


def test_expression_interface_numeric_helpers_and_reverse_power(tmp_path):
    polynomial = kw.Poly("x^2-4")
    assert polynomial.reinman(0, 1, 1000) == pytest.approx(-11 / 3, abs=0.01)
    assert polynomial.trapz(0, 1, 1000) == pytest.approx(-11 / 3, abs=0.01)
    assert polynomial.simpson(0, 1, 1001) == pytest.approx(-11 / 3, abs=0.01)
    assert polynomial.secant(3, 2.5) == pytest.approx(2, abs=1e-5)
    assert polynomial.bisection(0, 3) == pytest.approx(2, abs=1e-5)
    assert abs(kw.Var("x")) == kw.Abs(kw.Var("x"))
    assert 2 ** kw.Mono(3) == 8
    symbolic_power = 2 ** kw.Var("x")
    assert symbolic_power.when(x=3).try_evaluate() == 8

    output = tmp_path / "poly.json"
    polynomial.export_json(output)
    assert json.loads(output.read_text())["type"] == "Poly"
    assert polynomial.to_Function()(3) == 5


def test_expression_interface_plot_and_scatter_dimension_dispatch():
    kw.Poly("x^2").plot(values=[-1, 0, 1], show=False)
    kw.Poly("x^2").scatter(values=[-1, 0, 1], show=False)
    kw.IExpression.plot(kw.Poly("x+y"), meshgrid=np.meshgrid([0, 1], [0, 1]), show=False)
    kw.IExpression.scatter(kw.Poly("x+y"), start=0, stop=1, step=1, show=False)
    with pytest.raises(ValueError):
        kw.IExpression.plot(kw.Poly("3"), show=False)
    with pytest.raises(ValueError):
        kw.Poly("3").scatter(show=False)
    with pytest.raises(ValueError):
        kw.IExpression.plot(kw.Poly("x+y+z"), show=False)
    with pytest.raises(ValueError):
        kw.Poly("x+y+z").scatter(show=False)
    plt.close("all")


def test_matrix_construction_shapes_and_representations():
    assert kw.Matrix("2x3") == [[0, 0, 0], [0, 0, 0]]
    assert kw.Matrix(dimensions="2,3").shape == (2, 3)
    assert kw.Matrix((1, 2, 3)).shape == (1, 3)
    matrix = kw.Matrix([[1, 2], [3, 4]])
    assert matrix.num_of_rows == 2
    assert matrix.num_of_columns == 2
    assert "| 1.0 2.0" in str(matrix)
    assert repr(matrix) == "Matrix(matrix=[[1, 2], [3, 4]])"
    assert len(matrix) == 2


def test_matrix_row_operations_and_validation():
    matrix = kw.Matrix([[1, 2], [3, 4]])
    matrix.add_and_mul(0, 1, 2)
    assert matrix[0] == [7, 10]
    matrix.replace_rows(0, 1)
    assert matrix == [[3, 4], [7, 10]]
    matrix.divide_row(2, 0)
    assert matrix[0] == [1.5, 2]
    matrix.multiply_row(3, 1)
    assert matrix[1] == [21, 30]
    with pytest.raises(IndexError):
        matrix.add_and_mul(-1, 0, 1)
    with pytest.raises(IndexError):
        matrix.replace_rows(0, 3)
    with pytest.raises(ZeroDivisionError):
        matrix.divide_row(0, 0)
    with pytest.raises(IndexError):
        matrix.multiply_row(2, 0)


def test_matrix_rank_determinant_and_inverse():
    matrix = kw.Matrix([[1, 2], [3, 4]])
    assert matrix.get_rank() == 2
    assert matrix.determinant() == -2
    numpy_inverse = matrix.inverseWithNumpy()
    assert list(numpy_inverse.yield_items()) == pytest.approx([-2, 1, 1.5, -0.5])
    assert matrix.inverse() == kw.Matrix([[-2, 1], [1.5, -0.5]])
    assert kw.Matrix([[1, 2], [2, 4]]).inverseWithNumpy() is None
    with pytest.warns(UserWarning):
        assert kw.Matrix([[1, 2], [2, 4]]).inverseWithNumpy(verbose=True) is None
    with pytest.raises(ValueError):
        kw.Matrix([[1, 2, 3], [4, 5, 6]]).determinant()


def test_matrix_aggregations_transpose_and_kronecker():
    matrix = kw.Matrix([[1, 2], [3, 4]])
    assert list(matrix.yield_items()) == [1, 2, 3, 4]
    assert matrix.transpose() == [[1, 3], [2, 4]]
    assert matrix.sum() == 10
    assert matrix.max() == 4
    assert matrix.min() == 1
    assert matrix.average() == 2.5
    assert matrix.average_in_line(0) == 1.5
    assert matrix.average_in_column(1) == 3
    assert matrix.kronecker(kw.Matrix([[0, 5], [6, 7]])) == [
        [0, 5, 0, 10], [6, 7, 12, 14], [0, 15, 0, 20], [18, 21, 24, 28]
    ]


def test_matrix_arithmetic_and_shape_errors():
    matrix = kw.Matrix([[1, 2], [3, 4]])
    other = kw.Matrix([[2, 3], [4, 5]])
    assert matrix + other == [[3, 5], [7, 9]]
    assert matrix - other == [[-1, -1], [-1, -1]]
    assert matrix * other == [[2, 6], [12, 20]]
    assert matrix @ other == [[10, 13], [22, 29]]
    assert matrix / 2 == [[0.5, 1], [1.5, 2]]
    with pytest.raises(ValueError):
        matrix + kw.Matrix([[1]])
    with pytest.raises(ValueError):
        matrix - kw.Matrix([[1]])
    with pytest.raises(ValueError):
        matrix * kw.Matrix([[1]])
    with pytest.raises(TypeError):
        matrix + object()
    with pytest.raises(TypeError):
        matrix - object()
    with pytest.raises(TypeError):
        matrix * object()
    with pytest.raises(TypeError):
        matrix == object()


def test_matrix_indexing_columns_and_mutation():
    matrix = kw.Matrix([[1, 2], [3, 4], [5, 6]])
    assert matrix[lambda value: value > 2] == [[], [3, 4], [5, 6]]
    assert matrix[[0, 2]] == [[1, 2], [5, 6]]
    assert list(matrix.columns()) == [[1, 3, 5], [2, 4, 6]]
    assert matrix.column(1) == [2, 4, 6]
    matrix[0] = [7, 8]
    del matrix[1]
    assert matrix.matrix == [[7, 8], [5, 6]]
    with pytest.raises(TypeError):
        matrix[object()]


def test_matrix_random_unit_and_copy(monkeypatch):
    monkeypatch.setattr("random.randint", lambda a, b: a)
    assert kw.Matrix.random_matrix((2, 2), values=(3, 9)) == [[3, 3], [3, 3]]
    monkeypatch.setattr("random.uniform", lambda a, b: 1.5)
    assert kw.Matrix.random_matrix((1, 2), dtype="float") == [[1.5, 1.5]]
    unit = kw.Matrix.unit_matrix(2)
    assert kw.Matrix.is_unit_matrix(unit)
    assert not kw.Matrix.is_unit_matrix(kw.Matrix([[1, 0], [1, 1]]))
    copied = unit.__copy__()
    assert copied == unit and copied is not unit
