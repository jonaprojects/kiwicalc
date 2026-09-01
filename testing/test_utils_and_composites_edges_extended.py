import math
import importlib

import pytest

import kiwicalc as kw

utils = importlib.import_module("kiwicalc.core.utils")


def test_parenthesis_copy_and_collection_fallbacks():
    assert utils.handle_parenthesis("2(x+1)(x-1)3") == "2*(x+1)*(x-1)*3"
    assert utils.apply_on(lambda item: item + 1, (1, 2)) == [2, 3]
    original = {"x": 1}
    copied = utils.copy_expression(original)
    assert copied == original and copied is not original
    assert utils.copy_expression(3) == 3
    assert utils.float_gcd(0, 4) == 0


def test_coefficient_calculus_accepts_strings_and_mutation_mode():
    assert utils.derivative("3x^2+2x+1") == [6, 2]
    coefficients = [6, 2]
    assert utils.integral(coefficients, c=4, modify_original=True) is coefficients
    assert coefficients == [3, 2, 4]


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (0, 0, "0"),
        (kw.Poly("x+1"), 0, "(x+1)^2"),
        (0, kw.Poly("x-1"), "(x-1)^2"),
        (kw.Var("x"), 2, "(x-2)^2"),
    ],
)
def test_format_minus_contract(first, second, expected):
    assert utils._format_minus(first, second) == expected


def test_object_processing_and_order_helpers():
    converted = utils.process_object(3, "Thing", "method", "value")
    assert isinstance(converted, kw.Mono) and converted.try_evaluate() == 3
    x = kw.Var("x")
    assert utils.process_object(x, "Thing", "method", "value") == x
    with pytest.raises(TypeError, match="paramater 'value'"):
        utils.process_object(object(), "Thing", "method", "value")
    assert not utils.equal_ignore_order(None, [])
    assert not utils.equal_ignore_order([1], [1, 2])
    assert not utils.equal_ignore_order([1, 3], [1, 2])
    assert [str(item) for item in utils.sorted_expressions([kw.Mono("x"), kw.Mono("x^3")])] == ["x^3", "x"]


def test_absolute_and_factorial_formatting_edges():
    assert utils.handle_abs("2|3|") == "2*3"
    assert utils.handle_abs("x|y|") == "x*abs(y)"
    assert utils.handle_factorial("x+1") == "x+1"
    assert utils.handle_factorial("2(3)!") == "2*6.0"
    assert utils.handle_factorial("x!") == "factorial(x)"


def test_trigonometric_string_calculation_contracts():
    assert utils.handle_trigo_calculation("sin(0)") == 0
    assert utils.handle_trigo_calculation("-cos(0)") == -1
    assert utils.handle_trigo_calculation("2sin(0.5)") == pytest.approx(2 * math.sin(0.5), abs=1e-5)
    assert utils.handle_trigo_expression("sin(0)+cos(0)") == 1


def test_miscellaneous_formatting_edges():
    assert utils.format_matplot("x**12-x^2") == "x^{12}-x^{2}"
    with pytest.raises(NotImplementedError):
        utils.format_matplot_function("sin(x)")
    assert utils.format_linear_dict({}) == ""
    assert utils.format_linear_dict({"x": 1, "y": -1, "number": 0}) == "x-y"
    assert utils.format_linear_dict({"x": 1.25}, round_coefficients=False) == "1.25x"


def test_expression_mul_edge_protocols():
    x = kw.Var("x")
    empty = kw.ExpressionMul([])
    assert empty.python_syntax() == "1"
    assert empty.derivative() == 0
    assert str(empty) == "1"
    assert kw.ExpressionMul([0, x]).expressions == []
    with pytest.raises(NotImplementedError):
        kw.ExpressionMul("x")
    with pytest.raises(NotImplementedError):
        kw.ExpressionMul(["x"])
    with pytest.raises(TypeError):
        kw.ExpressionMul([object()])
    with pytest.raises(TypeError):
        empty *= object()


def test_expression_mul_arithmetic_equality_and_serialization_errors():
    x = kw.Var("x")
    product = kw.ExpressionMul([2, kw.Sin(x)])
    product *= kw.Mono(3)
    assert product.when(x=math.pi / 2).try_evaluate() == pytest.approx(6)
    combined = kw.ExpressionMul([kw.Sin(x)])
    combined *= kw.ExpressionMul([2, kw.Abs(x)])
    assert combined.when(x=-1).try_evaluate() == pytest.approx(-2 * math.sin(1))
    assert isinstance(product / 2, kw.Fraction)
    assert isinstance(2 / product, kw.Fraction)
    assert isinstance(2 ** product, kw.Exponent)
    assert kw.ExpressionMul([x]) == x
    assert kw.ExpressionMul([x]) != kw.ExpressionMul([x, x])
    with pytest.raises(TypeError):
        _ = product == object()
    with pytest.raises(ValueError, match="serialization payload"):
        kw.ExpressionMul.from_dict({"type": "Poly"})


def test_expression_mul_recursive_product_rule_and_string_rendering():
    x = kw.Var("x")
    product = kw.ExpressionMul([2, x, kw.Sin(x), kw.Abs(x)])
    derivative = product.derivative().when(x=1).try_evaluate()
    expected = 4 * math.sin(1) + 2 * math.cos(1)
    assert derivative == pytest.approx(expected)
    assert product.python_syntax().count("*") >= 3
    assert "*" in str(product)
    raised = product ** 2
    assert raised.when(x=1).try_evaluate() == pytest.approx((2 * math.sin(1)) ** 2)


def test_expression_sum_distribution_and_numeric_edges():
    x = kw.Var("x")
    product = kw.ExpressionSum([x, 1]) * kw.ExpressionSum([x, 1])
    assert product.when(x=3).try_evaluate() == 16
    assert product == kw.Poly("x^2+2x+1")

    evaluated = kw.ExpressionSum([kw.Mono(2), kw.Mono(3)])
    assert evaluated + 4 == 9
    assert evaluated - 4 == 1
    assert (10 - evaluated).try_evaluate() == 5
    assert kw.ExpressionSum([]) ** 2 is None
    assert kw.ExpressionSum([x]) ** 0 == 1
    with pytest.raises(ValueError, match="divide"):
        kw.ExpressionSum([x]) / 0
    with pytest.raises(TypeError, match="Invalid type"):
        kw.ExpressionSum([x]) / object()


def test_taylor_and_linear_regression_validation():
    approximation = utils.taylor_polynomial(kw.Poly("x^3"), n=2, a=1)
    assert approximation.when(x=1.1).try_evaluate() == pytest.approx(1.33)
    with pytest.raises(ValueError, match="corresponding y value"):
        utils.linear_regression([0, 1], [1])
