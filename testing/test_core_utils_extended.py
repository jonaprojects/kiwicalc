import math
import importlib

import matplotlib.pyplot as plt
import pytest

import kiwicalc as kw

utils = importlib.import_module("kiwicalc.core.utils")


@pytest.mark.parametrize(
    ("function", "argument", "expected"),
    [
        (utils.cot, math.pi / 4, 1),
        (utils.sec, 0, 1),
        (utils.csc, math.pi / 2, 1),
        (utils.asec, 2, math.acos(0.5)),
        (utils.acsc, 2, math.asin(0.5)),
        (utils.ln, math.e, 1),
    ],
)
def test_reciprocal_trigonometry_and_log(function, argument, expected):
    assert function(argument) == pytest.approx(expected)


def test_lambda_and_decimal_range_helpers():
    assert utils.is_lambda(lambda value: value)
    assert not utils.is_lambda(abs)
    assert list(utils.decimal_range(0, 0.3, 0.1)) == pytest.approx([0, 0.1, 0.2])
    assert list(utils.decimal_range(3, 1, -1)) == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [("", 1), ("+", 1), ("-", -1), ("2.5", 2.5), ("-3", -3)],
)
def test_extract_coefficient(source, expected):
    assert utils.extract_coefficient(source) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, ""), (-1, "-"), (3, "3"), (2.5, "2.5")],
)
def test_format_coefficient(value, expected):
    assert utils.format_coefficient(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, ""), (3, "+3"), (-3, "-3"), (2.5, "+2.5")],
)
def test_format_free_number(value, expected):
    assert utils.format_free_number(value) == expected


def test_regression_and_interpolation_helpers():
    regression = utils.linear_regression([0, 1, 2], [1, 3, 5])
    assert regression(10) == pytest.approx(21)
    assert utils.linear_regression([0, 1, 2], [1, 3, 5], get_values=True) == pytest.approx((2, 1))

    polynomial = utils.lagrange_polynomial([0, 1, 2], [1, 2, 5])
    for x, y in zip([0, 1, 2], [1, 2, 5]):
        assert polynomial.when(x=x).try_evaluate() == pytest.approx(y)


def test_apply_gcd_and_copy_helpers():
    assert list(utils.apply_on(lambda value: value * 2, [1, 2, 3])) == [2, 4, 6]
    assert utils.gcd([1.5, 3, 4.5]) == pytest.approx(1.5)
    expression = kw.Poly("x+1")
    copied = utils.copy_expression(expression)
    assert copied == expression
    assert copied is not expression


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("x+1", "(x+1)"),
        ("x", "x"),
        ("(x+1)", "((x+1))"),
    ],
)
def test_parenthesis_helpers(source, expected):
    assert utils.apply_parenthesis(source) == expected


def test_formatted_expression_and_lambda_support_math_syntax():
    assert utils.formatted_expression("2x^2", ("x",)) == "2*x**2"
    function = utils.to_lambda("sin(x)^2+cos(x)^2", ("x",))
    assert function(0.37) == pytest.approx(1)
    assert utils.to_lambda("|x|+3!", ("x",), format_abs=True, format_factorial=True)(-2) == 8


@pytest.mark.parametrize(
    ("coefficients", "expected"),
    [([3], 0), ([3, 2], 3), ([3, 2, 1], [6, 2])],
)
def test_derivative_coefficients(coefficients, expected):
    assert utils.derivative(coefficients) == expected


def test_integral_coefficients_and_string_conversion():
    assert utils.integral([6], c=4) == [6, 4]
    assert utils.integral([6, 2], c=4) == [3, 2, 4]
    assert utils.derivative([3, 2, 1], get_string=True) == "6x+2"
    assert utils.integral([6, 2], c=4, get_string=True) == "3x^2+2x+4"
    with pytest.raises(ValueError):
        utils.derivative([])
    with pytest.raises(ValueError):
        utils.integral([])


def test_grid_axis_and_basic_string_predicates():
    fig, ax = utils.create_grid()
    utils.draw_axis(ax)
    assert ax.spines["top"].get_visible() is False
    assert utils.contains_from_list(["sin", "cos"], "sin(x)")
    assert utils.clean_from_spaces(" x + 1 ") == "x+1"
    assert utils.is_evaluatable("1+2")
    assert not utils.is_evaluatable("not valid (")
    assert utils.is_number("2.5")
    assert not utils.is_number("x")
    plt.close(fig)


@pytest.mark.parametrize(
    ("source", "expected"),
    [("abc123", True), ("-123", True), ("", False), (None, False), ("a+b", True)],
)
def test_only_numbers_letters(source, expected):
    assert utils.only_numbers_letters(source) is expected


def test_collection_and_formatting_helpers():
    assert utils.equal_ignore_order([1, 2, 2], [2, 1, 2])
    assert not utils.equal_ignore_order([1, 2], [1, 1])
    assert utils.handle_abs("|x+1|") == "abs(x+1)"
    assert utils.handle_factorial("5!") == "120.0"
    assert utils.format_matplot_polynomial("x^2+3*x") == "x^{2}+3*x"
    assert utils.format_linear_dict({"x": 2, "y": -1, "number": 3}) == "2x-y+3"
    assert utils.format_poly_dict({"x**2": 1, "x**1": -2, "number": 1})
    assert utils.max_power([kw.Mono("x^2"), kw.Mono("3x^5")]) == kw.Mono("3x^5")


def test_recursive_lambda_helper():
    function, indices = utils.lambda_from_recursive("a_n = 2a_{n-1} + a_{n-2}")
    assert indices == ["n-2", "n-1"]
    assert function(4, 3, 5) == 10
