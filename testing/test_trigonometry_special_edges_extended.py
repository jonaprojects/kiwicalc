import math

import pytest

import kiwicalc as kw


@pytest.mark.parametrize(
    ("factory", "point", "expected"),
    [
        (kw.Sin, 0.4, math.cos(0.4)),
        (kw.Cos, 0.4, -math.sin(0.4)),
        (kw.Tan, 0.4, 1 / math.cos(0.4) ** 2),
        (kw.Cot, 0.4, -1 / math.sin(0.4) ** 2),
        (kw.Sec, 0.4, (1 / math.cos(0.4)) * math.tan(0.4)),
        (kw.Csc, 0.4, -(1 / math.sin(0.4)) * (1 / math.tan(0.4))),
        (kw.Asin, 0.4, 1 / math.sqrt(1 - 0.4**2)),
        (kw.Acos, 0.4, -1 / math.sqrt(1 - 0.4**2)),
        (kw.Atan, 0.4, 1 / (1 + 0.4**2)),
        (kw.Acot, 0.4, -1 / (1 + 0.4**2)),
    ],
)
def test_trigonometric_derivatives(factory, point, expected):
    derivative = factory(kw.Var("x")).derivative()
    assert derivative is not None
    assert derivative.when(x=point).try_evaluate() == pytest.approx(expected)


@pytest.mark.parametrize("factory", [kw.Sin, kw.Cos, kw.Tan, kw.Cot, kw.Sec, kw.Csc])
def test_trigonometric_scaling_power_and_sign(factory):
    expression = factory(kw.Var("x"))
    point = 0.7
    value = expression.when(x=point).try_evaluate()
    assert (expression * 3).when(x=point).try_evaluate() == pytest.approx(3 * value)
    assert (-expression).when(x=point).try_evaluate() == pytest.approx(-value)
    assert (expression**2).when(x=point).try_evaluate() == pytest.approx(value**2)
    assert expression / 2


def test_trigonometric_sum_protocol_and_roundtrip():
    expression = kw.TrigoExprs([kw.Sin(kw.Var("x")), kw.Cos(kw.Var("x")), 2])
    assert expression.variables == {"x"}
    assert expression.when(x=0).try_evaluate() == pytest.approx(3)
    assert expression.to_lambda()(math.pi / 2) == pytest.approx(3)
    assert kw.TrigoExprs.from_dict(expression.to_dict()) == expression
    assert expression.__copy__() == expression
    assert (-expression).when(x=0).try_evaluate() == pytest.approx(-3)
    assert (expression * 2).when(x=0).try_evaluate() == pytest.approx(6)


def test_trigonometric_integral_supported_cases():
    sine = kw.Sin(kw.Var("x"))
    cosine = kw.Cos(kw.Var("x"))
    with pytest.warns(UserWarning):
        sine.integral()
    with pytest.warns(UserWarning):
        cosine.integral()
    assert sine.when(x=0).try_evaluate() == pytest.approx(-1)
    assert cosine.when(x=math.pi / 2).try_evaluate() == pytest.approx(1)


def test_factorial_numeric_and_symbolic_contracts():
    assert kw.Factorial(0).try_evaluate() == 1
    assert kw.Factorial(4).try_evaluate() == 24
    assert kw.Factorial(0.5).try_evaluate() == pytest.approx(math.gamma(0.5) * 0.5)
    assert kw.Factorial(-1).try_evaluate() is None
    assert kw.Factorial(kw.Var("x")).try_evaluate() is None
    assert kw.Factorial(kw.Var("x"), power=0).try_evaluate() == 1

    expression = kw.Factorial(kw.Var("x"), coefficient=2, power=3)
    assert expression.variables == {"x"}
    assert expression.__copy__() == expression
    assert expression != None
    assert expression.python_syntax()
    assert "!" in str(expression)


def test_factorial_arithmetic_and_errors():
    x = kw.Var("x")
    assert kw.Factorial(x) + kw.Factorial(x) == kw.Factorial(x, coefficient=2)
    assert kw.Factorial(x) - kw.Factorial(x) == kw.Factorial(x, coefficient=0)
    assert (kw.Factorial(x) * 3).coefficient == 3
    assert (kw.Factorial(x) ** 2).power == 2
    assert kw.Factorial(4) / 2 == 12
    assert 48 / kw.Factorial(4) == 2
    with pytest.raises(ZeroDivisionError):
        kw.Factorial(x) / 0
    with pytest.raises(TypeError):
        kw.Factorial(x) * object()
    with pytest.raises(TypeError):
        kw.Factorial(x) == object()


def test_root_numeric_edge_paths():
    assert kw.Root(8, 3, 2).try_evaluate() == pytest.approx(4)
    assert isinstance(kw.Root(8, 0).try_evaluate(), ValueError)
    assert kw.Root(8, 3, 0).try_evaluate() == 0
    assert kw.Root(16) ** 2 == 16
    assert kw.Root(8, 3) ** 0 == 1
    assert kw.Root(16) / kw.Root(16) == 1
    assert isinstance(kw.Root(kw.Var("x")) / kw.Var("y"), kw.Fraction)


def test_root_symbolic_arithmetic_and_validation():
    x = kw.Var("x")
    first = kw.Root(x, 2, 2)
    second = kw.Root(x, 2, 3)
    assert first + second == kw.Root(x, 2, 5)
    assert first - first == 0
    assert first * second
    assert first / 2 == kw.Root(x, 2, 1)
    assert first != kw.Root(x, 3, 2)
    assert first != None
    assert kw.Root.dependant_roots(kw.Root(x, 2), kw.Root(x, 3)) is None
    with pytest.raises(TypeError):
        first * object()
