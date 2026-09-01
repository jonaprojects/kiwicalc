import pytest

import kiwicalc as kw


def test_log_symbolic_components_copy_variables_and_simplify():
    coefficient = kw.Var("c")
    copied = kw.Log(kw.Var("x"), coefficient=coefficient, gen_copies=True)
    shared = kw.Log(kw.Var("x"), coefficient=coefficient, gen_copies=False)
    assert copied.coefficient is not coefficient
    assert shared.coefficient is coefficient

    expression = kw.Log([[kw.Var("x"), kw.Var("b"), kw.Var("p")]])
    assert expression.variables == {"x", "b", "p"}
    expression.simplify()


def test_log_addition_subtraction_and_append_paths():
    symbolic = kw.Log(kw.Var("x"), base=10)
    assert isinstance(symbolic + kw.Log(10), kw.Log)
    assert isinstance(kw.Log(10) + symbolic, kw.Log)
    combined = kw.Log(kw.Var("x"), base=2) + kw.Log(kw.Var("y"), base=3)
    assert isinstance(combined, kw.Log) and len(combined._expressions) == 2
    assert kw.Log(kw.Var("x"), coefficient=2) - kw.Log(kw.Var("x")) == kw.Log(kw.Var("x"))
    assert isinstance(kw.Log(kw.Var("x"), base=2) - kw.Log(kw.Var("y"), base=3), kw.ExpressionSum)


def test_log_division_evaluation_python_syntax_and_equality():
    assert kw.Log(100) / kw.Mono(2) == kw.Mono(1)
    assert kw.Log(kw.Var("x")) / kw.Mono(2) == kw.Log(kw.Var("x"), coefficient=0.5)
    assert isinstance(kw.Log(kw.Var("x")) / kw.Var("y"), kw.Fraction)
    symbolic_coefficient = kw.Log(kw.Var("x"), coefficient=kw.Var("c"))
    assert symbolic_coefficient.try_evaluate() is None
    assert kw.Log(kw.Var("x"), coefficient=-1).python_syntax().startswith("-")
    assert kw.Log(10).python_syntax()
    assert kw.Log(kw.Var("x"), coefficient=2) == kw.Log(kw.Poly("x^2"))
    assert kw.Log(kw.Var("x"), base=2) != kw.Log(kw.Var("x"), base=3)


def test_factorial_expression_arithmetic_and_none_assignment_edges():
    x = kw.Var("x")
    assert isinstance(kw.Factorial(x) * kw.Factorial(kw.Var("y")), kw.ExpressionMul)
    assert kw.Factorial(3) * kw.Mono(2) == kw.Mono(12)
    assert (kw.Factorial(x) * kw.Var("y")).variables == {"x", "y"}
    assert kw.Factorial(3) / kw.Mono(3) == kw.Factorial(2)
    with pytest.raises(ZeroDivisionError):
        kw.Factorial(x).__itruediv__(kw.Abs(0))
    assert isinstance(kw.Factorial(x) / kw.Var("y"), kw.Fraction)
    assert isinstance(kw.Var("y") / kw.Factorial(x), kw.Fraction)

    no_expression = kw.Factorial(None, coefficient=kw.Var("c"))
    no_expression.assign(c=2)
    assert no_expression.try_evaluate() == 2


def test_abs_arithmetic_derivative_and_zero_divisor_edges():
    x, y = kw.Var("x"), kw.Var("y")
    assert kw.Abs(2) + kw.Abs(3) == 5
    assert kw.Abs(2) - kw.Abs(3) == -1
    assert isinstance(kw.Abs(x) + kw.Abs(y), kw.ExpressionSum)
    assert isinstance(kw.Abs(x) - kw.Abs(y), kw.ExpressionSum)
    with pytest.raises(ValueError):
        kw.Abs(x).__itruediv__(kw.Abs(0))
    with pytest.warns(UserWarning):
        assert kw.Abs(3).derivative() == 0
    with pytest.warns(UserWarning):
        assert kw.Abs(x, coefficient=0).derivative() == 0
    with pytest.warns(UserWarning):
        pair = kw.Abs(x).derivative(get_derivatives=True)
    assert pair[1] == -pair[0]
    assert str(kw.Abs(x, coefficient=-1)).startswith("-")


def test_exponent_add_multiply_divide_derivative_and_equality_edges():
    x, y = kw.Var("x"), kw.Var("y")
    first = kw.Exponent(x, y)
    assert isinstance(first + kw.Abs(2), kw.ExpressionSum)
    assert isinstance(first - kw.Abs(2), kw.ExpressionSum)
    assert isinstance(first + kw.Abs(y), kw.ExpressionSum)
    assert isinstance(first * kw.Exponent(y, x), kw.ExpressionMul)
    assert first * kw.Mono(2) == kw.Exponent(x, y, coefficient=2)
    with pytest.raises(TypeError):
        first.__imul__(object())
    with pytest.raises(ZeroDivisionError):
        first.divide_by_number(0)

    assert kw.Exponent(x, 2).derivative() == kw.Mono("2x")
    assert kw.Exponent(2, x).derivative().variables == {"x"}
    with pytest.warns(UserWarning):
        assert kw.Exponent(-2, x).derivative() is None
    assert kw.Exponent(x, y, coefficient=x) == kw.Exponent(x, y + 1)
    assert str(kw.Exponent(x, y, coefficient=-1)).startswith("-")
