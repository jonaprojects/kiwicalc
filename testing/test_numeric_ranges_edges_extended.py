import math

import pytest

import kiwicalc as kw
from kiwicalc.numeric.roots import extract_possible_solutions, get_bounds, get_factors


@pytest.mark.parametrize("method", ["central", "forward", "backward"])
def test_numerical_derivative_methods(method):
    assert kw.numerical_diff(lambda x: x**2, 3, method=method, h=1e-5) == pytest.approx(6, rel=1e-4)


def test_numeric_calculus_validation_and_optimization():
    with pytest.raises(ValueError):
        kw.reinman(lambda x: x, 0, 1, 1)
    with pytest.raises(ValueError):
        kw.trapz(lambda x: x, 0, 1, 0)
    with pytest.raises(ValueError):
        kw.simpson(lambda x: x, 0, 1, 2)
    with pytest.raises(ValueError):
        kw.numerical_diff(lambda x: x, 0, method="sideways")

    assert kw.gradient_descent(lambda x: 2 * (x - 3), 10) == pytest.approx(3, abs=1e-4)
    assert kw.gradient_ascent(lambda x: -2 * (x - 3), -5) == pytest.approx(3, abs=1e-4)


def test_single_root_algorithms_converge():
    function = lambda x: x * x - 2
    first = lambda x: 2 * x
    second = lambda x: 2
    expected = math.sqrt(2)

    assert kw.newton_raphson(function, first, 1) == pytest.approx(expected, abs=1e-5)
    assert kw.halleys_method(function, first, second, 1) == pytest.approx(expected, abs=1e-5)
    assert kw.secant_method(function, 2, 1) == pytest.approx(expected, abs=1e-5)
    assert kw.inverse_interpolation(function, 0, 1, 2) == pytest.approx(expected, abs=1e-5)
    assert kw.laguerre_method(function, first, second, 1, 2) == pytest.approx(expected, abs=1e-5)


def test_polynomial_root_helpers():
    assert get_factors(0) == {}
    assert get_factors(6) == {-6, -3, -2, -1, 1, 2, 3, 6}
    assert extract_possible_solutions(2, 3) == {-3, -1.5, -1, -0.5, 0.5, 1, 1.5, 3}
    upper, lower = get_bounds(2, [2, -3, 1])
    assert upper == pytest.approx(4)
    assert lower == pytest.approx(0.4)


def test_open_closed_and_unbounded_ranges():
    bounded = kw.create_range("0<x<=2")
    assert bounded.evaluate_when(x=0) is False
    assert bounded.evaluate_when(x=1) is True
    assert bounded.evaluate_when(x=2) is True
    assert bounded.evaluate_when(x=3) is False
    assert str(bounded) == "0<x<=2"

    upper = kw.create_range("x<5")
    lower = kw.create_range("x>=5")
    assert upper.evaluate_when(x=4) is True
    assert lower.evaluate_when(x=4) is False
    assert kw.RangeOR((upper, lower)).try_evaluate() is None
    assert isinstance(bounded.__copy__(), kw.Range)


def test_range_collection_chaining_and_validation():
    first = kw.create_range("x>0")
    second = kw.create_range("x<10")
    collection = kw.RangeCollection((first,)).chain(second, copy=True)
    assert len(collection.ranges) == 2
    assert str(collection) == "0<x, x<10"
    assert isinstance(collection.__copy__(), kw.RangeCollection)
    with pytest.raises(TypeError):
        collection.chain("x<3")
    with pytest.raises(ValueError):
        kw.create_range("x")


def test_fraction_variables_include_both_sides():
    fraction = kw.Fraction(kw.Var("x") + kw.Var("y"), kw.Var("z") + 1)
    assert fraction.variables == {"x", "y", "z"}


def test_matrix_validation_paths():
    with pytest.raises(ValueError):
        kw.Matrix()
    with pytest.raises(TypeError):
        kw.Matrix(dimensions=object())
    with pytest.raises(ValueError):
        kw.Matrix.random_matrix((1, 1), dtype="decimal")
    with pytest.raises(ZeroDivisionError):
        kw.Matrix([[1]]) / 0
    with pytest.raises(ValueError):
        kw.Matrix([[1, 2]]) @ kw.Matrix([[1, 2]])
    with pytest.raises(IndexError):
        kw.Matrix([[1, 2]]).column(2)
