import json

import pytest

import kiwicalc as kw
from kiwicalc.core.utils import (
    format_poly_dict,
    handle_trigo_calculation,
    handle_trigo_expression,
    integral as coefficient_integral,
    taylor_polynomial,
)


def test_expression_sum_constructor_flatten_copy_and_empty_branches():
    x = kw.Var("x")
    nested = kw.ExpressionSum([kw.ExpressionSum([x, 1]), 2])
    assert len(nested) == 3
    assert str(kw.ExpressionSum()) == "0"
    assert kw.ExpressionSum().python_syntax() == "0"
    direct = kw.ExpressionSum([x], copy=False)
    assert direct[0] is x
    direct.append(kw.Mono(2))
    assert list(direct) == [x, kw.Mono(2)]


def test_expression_sum_assignment_and_evaluation_branches():
    x, y = kw.Var("x"), kw.Var("y")
    expression = kw.ExpressionSum([kw.Sin(x), kw.Abs(y), 2])
    assert expression.try_evaluate() is None
    assigned = expression.when_all(x=0, y=-3)
    assert assigned.try_evaluate() == 5
    expression.assign_to_all(x=0, y=-3)
    assert expression.try_evaluate() == 5
    assert expression.variables == set()


def test_expression_sum_add_subtract_evaluation_branches():
    x = kw.Var("x")
    constants = kw.ExpressionSum([1, 2])
    assert constants + 3 == 6
    assert constants - 1 == 2
    assert constants + kw.Mono(4) == 7
    assert constants - kw.Mono(4) == -1
    symbolic = kw.ExpressionSum([kw.Sin(x), kw.Abs(x)])
    assert isinstance(symbolic + 2, kw.ExpressionSum)
    assert isinstance(symbolic - 2, kw.ExpressionSum)
    assert isinstance(symbolic + kw.Mono(2), kw.ExpressionSum)
    assert isinstance(symbolic - kw.Mono(2), kw.ExpressionSum)
    assert (3 - symbolic).when(x=0).try_evaluate() == 3

    # A fully evaluatable sum combined with a symbolic expression must retain
    # the evaluated value on the correct side of subtraction.
    evaluated = kw.ExpressionSum([1, 2])
    assert (evaluated + kw.Sin(x)).when(x=0).try_evaluate() == 3
    assert (evaluated - kw.Sin(x)).when(x=0).try_evaluate() == 3


def test_expression_sum_symbolic_add_subtract_branches():
    x, y = kw.Var("x"), kw.Var("y")
    left = kw.ExpressionSum([kw.Sin(x), kw.Abs(x)])
    right = kw.ExpressionSum([kw.Cos(y), kw.Abs(y)])
    added = left + right
    subtracted = left - right
    assert added.when_all(x=0, y=0).try_evaluate() == 1
    assert subtracted.when_all(x=0, y=0).try_evaluate() == -1
    assert isinstance(left + kw.Sin(y), kw.ExpressionSum)
    assert isinstance(left - kw.Sin(y), kw.ExpressionSum)


def test_expression_sum_multiplication_and_poly_conversion_branches():
    x = kw.Var("x")
    polynomial = kw.ExpressionSum([x, 1])
    assert polynomial.is_poly()
    assert polynomial.to_poly() == x + 1
    product = polynomial * kw.ExpressionSum([x, -1])
    assert product == x**2 - 1
    assert polynomial * 2 == 2 * x + 2
    non_polynomial = kw.ExpressionSum([kw.Sin(x), 1])
    assert not non_polynomial.is_poly()
    assert non_polynomial.to_poly() is None
    assert isinstance(non_polynomial * 2, kw.ExpressionSum)


def test_expression_sum_power_branches():
    x = kw.Var("x")
    empty = kw.ExpressionSum()
    assert empty**0 == 1
    assert empty**2 == 0
    assert kw.ExpressionSum([x]) ** 2 == x**2
    pair = kw.ExpressionSum([x, 1])
    assert pair**2 == x**2 + 2 * x + 1
    assert pair**3 == x**3 + 3 * x**2 + 3 * x + 1
    assert isinstance(pair**-2, kw.Fraction)
    assert isinstance(pair**0.5, kw.Exponent)
    assert isinstance(pair ** kw.Var("n"), kw.Exponent)
    assert pair ** kw.Mono(2) == x**2 + 2 * x + 1
    with pytest.raises(TypeError):
        pair ** object()


def test_expression_sum_division_branches():
    x = kw.Var("x")
    expression = kw.ExpressionSum([kw.Sin(x), kw.Abs(x)])
    with pytest.raises(ValueError):
        expression / 0
    with pytest.raises(TypeError):
        expression / object()
    assert (kw.ExpressionSum([2, 4]) / 2) == 3
    assert (expression / kw.Mono(2)).when(x=1).try_evaluate() == pytest.approx(
        (kw.Sin(1).try_evaluate() + 1) / 2
    )
    assert isinstance(expression / kw.ExpressionSum([kw.Cos(x), 1]), kw.Fraction)
    assert isinstance(expression / kw.Sin(x), kw.ExpressionSum)
    assert kw.ExpressionSum([x**2, x]) / x == kw.Poly("x+1")


def test_expression_sum_simplify_render_serialization_and_equality_branches():
    x = kw.Var("x")
    expression = kw.ExpressionSum([kw.Sin(x), 1, 2])
    expression.simplify()
    assert len(expression) == 2
    assert "+" in str(expression)
    assert "+" in expression.python_syntax()
    restored = kw.ExpressionSum.from_dict(expression.to_dict())
    assert restored == expression
    with pytest.raises(ValueError):
        kw.ExpressionSum.from_dict({"type": "Mono", "expressions": []})
    assert kw.ExpressionSum([1, 2]) == 3
    assert kw.ExpressionSum([kw.Sin(x)]) == kw.Sin(x)
    assert kw.ExpressionSum([kw.Sin(x), kw.Cos(x)]) != kw.Sin(x)
    assert kw.ExpressionSum([kw.Sin(x)]) != kw.ExpressionSum([kw.Sin(x), kw.Cos(x)])
    assert str(kw.ExpressionSum([-kw.Sin(x), kw.Cos(x)])).startswith("-")
    assert kw.ExpressionSum([-kw.Sin(x)]).python_syntax().startswith("-")
    assert kw.ExpressionSum([kw.Sin(x), kw.Sin(x)]) != kw.ExpressionSum([kw.Sin(x), kw.Cos(x)])
    assert not (kw.ExpressionSum([kw.Sin(x)]) == object())


def test_remaining_core_utils_branch_contracts():
    x = kw.Var("x")
    assert taylor_polynomial(lambda value: value**2, 0, 3) == 9
    assert taylor_polynomial(x**2, 1, 2) == kw.Poly("4x-4")
    assert coefficient_integral("2x+2", c=3) == [1, 2, 3]
    assert coefficient_integral([4], c=2) == [4, 2]
    with pytest.raises(ValueError):
        coefficient_integral([])
    assert handle_trigo_calculation("sin(0)") == 0
    assert handle_trigo_calculation("-cos(0)") == -1
    assert handle_trigo_calculation("2cos(0)") == 2
    assert handle_trigo_expression("sin(0)+cos(0)") == 1
    assert format_poly_dict({"x**1": 2.0, "y**2": 3.5, "z**0": 4, "number": -1})
    assert format_poly_dict({"x**2": 0, "number": 2}).startswith("+2")


def test_mono_constructor_metadata_and_setter_branches():
    with pytest.raises(TypeError):
        kw.Mono(object())
    parsed = kw.Mono("-2x^3y")
    assert parsed.coefficient == -2
    assert parsed.variables == {"x", "y"}
    assert parsed.num_of_variables == 2
    assert parsed.highest_power() == 3
    constant = kw.Mono(3)
    assert constant.highest_power() == 0
    parsed.coefficient = 0
    assert parsed.variables_dict is None


def test_mono_add_subtract_dispatch_branches():
    x = kw.Mono("x")
    assert x + 0 == x
    assert isinstance(x + 2, kw.Poly)
    assert isinstance(x - 2, kw.Poly)
    assert kw.Mono(2) + 3 == 5
    assert kw.Mono(2) - 3 == -1
    assert x + kw.Mono(0) == x
    assert x - kw.Mono(0) == x
    assert x + kw.Mono("2x") == kw.Mono("3x")
    assert x - kw.Mono("x") == 0
    assert isinstance(x + kw.Mono("y"), kw.Poly)
    assert isinstance(x - kw.Mono("y"), kw.Poly)
    assert x + "x" == kw.Mono("2x")
    assert isinstance(x + kw.Sin(kw.Var("x")), kw.ExpressionSum)
    with pytest.raises(TypeError):
        x.__copy__().__iadd__(object())
    with pytest.raises(TypeError):
        x.__copy__().__isub__(object())


def test_mono_multiply_divide_power_branches():
    x, y = kw.Mono("x"), kw.Mono("y")
    assert x * 0 == 0
    assert x * 3 == kw.Mono("3x")
    assert x * y == kw.Mono("xy")
    assert x * "y" == kw.Mono("xy")
    assert x * kw.Poly("x+1") == kw.Poly("x^2+x")
    with pytest.raises(ZeroDivisionError):
        x.__copy__().divide_by_number(0)
    with pytest.raises(ZeroDivisionError):
        x / 0
    assert kw.Mono("4x") / 2 == kw.Mono("2x")
    assert kw.Mono("x") / "x" == 1
    assert isinstance(x / kw.Poly("x+1"), kw.PolyFraction)
    assert isinstance(x / y, kw.PolyFraction)
    assert x**0 == 1
    assert x**2 == kw.Mono("x^2")
    assert isinstance(x ** kw.Var("n"), kw.Exponent)
    assert x ** kw.Mono(2) == kw.Mono("x^2")
    with pytest.raises(TypeError):
        x / object()


def test_mono_equality_sort_format_assignment_and_calculus_branches(tmp_path):
    x = kw.Mono("x")
    assert x != None  # noqa: E711
    assert x == "x"
    assert x == kw.Poly("x")
    assert x != kw.Poly("x+1")
    with pytest.raises(TypeError):
        x == object()
    assert kw.Mono(2) < 3
    assert str(kw.Mono("-x")) == "-x"
    assert "x^{2.0}" in kw.Mono("2x^2").latex()
    assert x.contains_variable("x")
    assert not kw.Mono(2).contains_variable("x")
    assert kw.Mono(2).is_number()
    assigned = kw.Mono("2x^2y")
    assigned.assign(x=3)
    assert assigned == kw.Mono("18y")
    assert kw.Mono(4).derivative() == 0
    assert kw.Mono("3x").derivative() == 3
    assert kw.Mono("3x^2").derivative() == kw.Mono("6x")
    assert kw.Mono("x^-1").derivative() == kw.Mono("-x^-2")
    assert kw.Mono("2xy").partial_derivative(["x"]) == kw.Mono("2y")
    assert kw.Mono("2xy").partial_derivative(["z"]) == 0
    assert kw.Mono(3).integral("t") == kw.Mono("3t")
    assert kw.Mono("2x").integral() == kw.Mono("x^2")
    assert kw.Mono("x^-2").integral() == kw.Mono("-x^-1")
    logarithmic_integral = kw.Mono("x^-1").integral()
    assert isinstance(logarithmic_integral, kw.Log)
    assert logarithmic_integral.all_bases() == {kw.e}
    payload = json.dumps(kw.Mono("2x").to_dict())
    assert kw.Mono.from_json(payload) == kw.Mono("2x")
    with pytest.raises(ValueError):
        kw.Mono.from_json('{"type":"Poly","coefficient":1,"variables_dict":null}')
