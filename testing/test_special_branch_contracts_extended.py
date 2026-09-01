import math

import pytest

import kiwicalc as kw


def test_factorial_empty_and_evaluation_branches():
    empty = kw.Factorial(None, coefficient=3)
    assert empty.try_evaluate() == 3
    assert str(empty) == "3"
    assert empty.python_syntax() == "3"
    assert (-empty).try_evaluate() == -3
    assert kw.Factorial(3, coefficient=0).try_evaluate() == 0
    assert kw.Factorial(kw.Var("x"), coefficient=kw.Var("c")).try_evaluate() is None


def test_factorial_add_subtract_unmatched_branches():
    x = kw.Var("x")
    assert kw.Factorial(x) + 0 == kw.Factorial(x)
    assert kw.Factorial(x) - 0 == kw.Factorial(x)
    assert isinstance(kw.Factorial(x) + 2, kw.ExpressionSum)
    difference = kw.Factorial(x) - 2
    assert difference.when(x=3).try_evaluate() == 4
    assert isinstance(kw.Factorial(x) + kw.Factorial(x + 1), kw.ExpressionSum)
    assert (kw.Factorial(x, coefficient=3) - kw.Factorial(x)).coefficient == 2


def test_factorial_multiplication_branches():
    x = kw.Var("x")
    assert kw.Factorial(x) * (x + 1) == kw.Factorial(x + 1)
    assert kw.Factorial(4) * 2 == 48
    assert isinstance(kw.Factorial(x) * kw.Factorial(x + 1), kw.ExpressionMul)
    combined = kw.Factorial(x, coefficient=2, power=2) * kw.Factorial(x, coefficient=3, power=4)
    assert combined.coefficient == 6 and combined.power == 6
    assert isinstance(kw.Factorial(x) * kw.Sin(x), kw.Factorial)
    with pytest.raises(TypeError):
        kw.Factorial(x) * object()


def test_factorial_division_branches():
    x = kw.Var("x")
    with pytest.raises(ZeroDivisionError):
        kw.Factorial(x) / 0
    reduced = kw.Factorial(x) / x
    assert reduced.expression == x - 1
    assert (kw.Factorial(x, coefficient=x) / x).coefficient == 1
    assert kw.Factorial(5) / kw.Mono(5) == 24
    assert kw.Factorial(x, coefficient=4) / kw.Mono(2) == kw.Factorial(x, coefficient=2)
    assert kw.Factorial(x, power=3) / kw.Factorial(x) == kw.Factorial(x, power=2)
    assert isinstance(kw.Factorial(x) / kw.Sin(x), kw.Fraction)
    assert isinstance(2 / kw.Factorial(x), kw.Fraction)
    with pytest.raises(TypeError):
        object() / kw.Factorial(x)


def test_factorial_assignment_simplify_render_and_equality_branches():
    x = kw.Var("x")
    expression = kw.Factorial(x + 1, coefficient=2, power=2)
    assert expression.when(x=2).try_evaluate() == 72
    assert "factorial" in expression.python_syntax()
    assert "**2" in str(expression)
    zero = kw.Factorial(x, coefficient=0)
    zero.simplify()
    assert zero.expression is None and zero.power == 1
    assert kw.Factorial(x) != None  # noqa: E711
    assert kw.Factorial(x) != kw.Factorial(x + 1)
    assert kw.Factorial(x) != kw.Sin(x)


@pytest.mark.parametrize("argument", [object(), "x"])
def test_abs_constructor_type_errors(argument):
    with pytest.raises(TypeError):
        kw.Abs(argument)


def test_abs_constructor_power_and_coefficient_errors():
    x = kw.Var("x")
    with pytest.raises(TypeError):
        kw.Abs(x, power=object())
    with pytest.raises(TypeError):
        kw.Abs(x, coefficient=object())


def test_abs_add_subtract_dispatch_branches():
    x, y = kw.Var("x"), kw.Var("y")
    assert kw.Abs(-2) + 3 == 5
    assert kw.Abs(-2) - 3 == -1
    assert isinstance(kw.Abs(x) + 2, kw.ExpressionSum)
    assert (kw.Abs(x) - 2).when(x=-3).try_evaluate() == 1
    assert kw.Abs(x, coefficient=2) + kw.Abs(-x, coefficient=3) == kw.Abs(x, coefficient=5)
    assert kw.Abs(x, coefficient=2) - kw.Abs(x) == kw.Abs(x)
    difference = kw.Abs(x) - kw.Abs(y)
    assert difference.when(x=-3, y=-2).try_evaluate() == 1


def test_abs_multiplication_dispatch_branches():
    x, y = kw.Var("x"), kw.Var("y")
    with pytest.raises(TypeError):
        kw.Abs(x) * object()
    assert (kw.Abs(x) * 3).coefficient == 3
    assert kw.Abs(-2) * kw.Mono(3) == 6
    assert (kw.Abs(x) * kw.Mono(3)).coefficient == 3
    combined = kw.Abs(x, power=2, coefficient=2) * kw.Abs(-x, power=3, coefficient=4)
    assert combined.power == 5 and combined.coefficient == 8
    assert isinstance(kw.Abs(x) * kw.Abs(y), kw.ExpressionMul)


def test_abs_division_dispatch_branches():
    x, y, p = kw.Var("x"), kw.Var("y"), kw.Var("p")
    with pytest.raises(TypeError):
        kw.Abs(x) / object()
    with pytest.raises(ValueError):
        kw.Abs(x) / 0
    with pytest.raises(ValueError):
        kw.Abs(x) / kw.Mono(0)
    assert (kw.Abs(x, coefficient=4) / 2).coefficient == 2
    assert kw.Abs(-4) / kw.Mono(2) == 2
    assert (kw.Abs(x, coefficient=4) / kw.Mono(2)).coefficient == 2
    assert isinstance(kw.Abs(x) / kw.Sin(x), kw.Abs)
    assert isinstance(kw.Abs(x, power=p) / kw.Abs(x), kw.Exponent)
    assert isinstance(kw.Abs(x, power=3) / kw.Abs(x), kw.Abs)
    assert kw.Abs(x) / kw.Abs(x) == 1
    assert isinstance(kw.Abs(x) / kw.Abs(x, power=2), kw.Fraction)
    assert isinstance(kw.Abs(x) / kw.Abs(y), kw.Fraction)


def test_abs_power_assignment_equality_and_render_branches():
    x, p = kw.Var("x"), kw.Var("p")
    with pytest.raises(TypeError):
        kw.Abs(x) ** object()
    assert (kw.Abs(x, coefficient=2) ** 2).coefficient == 4
    assert isinstance(kw.Abs(x) ** p, kw.Exponent)
    assert (kw.Abs(x) ** kw.Mono(2)).power == 2
    assigned = kw.Abs(x, power=p).when(x=-3, p=2)
    assert assigned.try_evaluate() == 9
    assert str(kw.Abs(x, coefficient=0)) == "0"
    assert str(kw.Abs(x, power=0, coefficient=3)) == "3"
    assert str(kw.Abs(x, coefficient=-1)).startswith("-")
    assert kw.Abs(x) == kw.Abs(x)
    assert kw.Abs(x) != kw.Abs(x + 1)
    assert kw.Abs(x) != object()


def test_exponent_constructor_type_branches():
    with pytest.raises(TypeError):
        kw.Exponent(object(), 2)
    with pytest.raises(TypeError):
        kw.Exponent(2, object())
    with pytest.raises(TypeError):
        kw.Exponent(2, 3, coefficient=object())


def test_exponent_add_subtract_dispatch_branches():
    x = kw.Var("x")
    expression = kw.Exponent(2, x)
    assert expression + 0 == expression
    assert isinstance(expression + 2, kw.ExpressionSum)
    assert isinstance(expression - 2, kw.ExpressionSum)
    assert kw.Exponent(2, 3) + 2 == 10
    assert kw.Exponent(2, 3) - 2 == 6
    assert isinstance(expression + kw.Mono(2), kw.ExpressionSum)
    assert isinstance(expression - kw.Sin(x), kw.ExpressionSum)
    assert expression + kw.Exponent(2, x) == kw.Exponent(2, x, coefficient=2)
    assert expression - kw.Exponent(3, x) != 0


def test_exponent_multiplication_division_and_power_branches():
    x = kw.Var("x")
    expression = kw.Exponent(2, x)
    assert expression * 0 == 0
    assert expression * 2 == kw.Exponent(2, x, coefficient=2)
    assert expression * kw.Mono(2) == kw.Exponent(2, x, coefficient=2)
    assert expression * 2 == expression * kw.Exponent(2, 1)
    assert kw.Exponent(2, x) * kw.Exponent(3, x) == kw.Exponent(6, x)
    assert isinstance(kw.Exponent(2, x) * kw.Exponent(3, x + 1), kw.ExpressionMul)
    assert isinstance(expression * kw.Sin(x), kw.ExpressionMul)
    with pytest.raises(TypeError):
        expression * object()
    with pytest.raises(ZeroDivisionError):
        expression.__copy__().divide_by_number(0)
    assert isinstance(expression / 2, kw.Fraction)
    assert expression**0 == 1
    assert (expression**2).power == 2 * x


def test_exponent_derivative_and_evaluation_branches():
    x, y = kw.Var("x"), kw.Var("y")
    assert kw.Exponent(2, 3).derivative() == 0
    assert kw.Exponent(x, 0, coefficient=2).derivative() == 0
    polynomial = kw.Exponent(x, 3, coefficient=2)
    assert polynomial.derivative().when(x=2).try_evaluate() == 24
    exponential = kw.Exponent(2, x, coefficient=3)
    assert exponential.derivative().when(x=2).try_evaluate() == pytest.approx(12 * math.log(2))
    with pytest.warns(UserWarning):
        assert kw.Exponent(-2, x).derivative() is None
    with pytest.raises(ValueError):
        kw.Exponent(x, y).derivative()
    assert kw.Exponent(2, x).try_evaluate() is None
    assert kw.Exponent(2, 0, coefficient=3).try_evaluate() == 3


def test_exponent_equality_render_serialization_and_unsupported_branches():
    x = kw.Var("x")
    expression = kw.Exponent(2, x, coefficient=3)
    assert kw.Exponent.from_dict(expression.to_dict()) == expression
    assert expression != None  # noqa: E711
    assert expression != kw.Exponent(2, x, coefficient=4)
    assert kw.Exponent(x, 2, coefficient=x) == kw.Exponent(x, 3)
    assert str(kw.Exponent(2, x, coefficient=0)) == "0"
    assert str(kw.Exponent(2, 0, coefficient=3)) == "3"
    assert str(kw.Exponent(2, x, coefficient=-1)).startswith("-")
    with pytest.raises(NotImplementedError):
        expression.partial_derivative()
    with pytest.raises(NotImplementedError):
        expression.integral()
