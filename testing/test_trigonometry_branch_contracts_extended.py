import math
import importlib

import pytest

import kiwicalc as kw
from kiwicalc.core.operators import TrigoMethods


def trig(method, inside, power=1, coefficient=1):
    return kw.TrigoExpr(coefficient, expressions=((method, inside, power),))


def test_trigoexpr_constructor_copy_simplify_and_metadata_branches():
    x = kw.Var("x")
    original = trig(TrigoMethods.SIN, x, coefficient=2)
    copied = kw.TrigoExpr(original)
    assert copied == original and copied is not original
    assert kw.TrigoExpr("2sin(x)").coefficient == 2
    assert kw.TrigoExpr(kw.Mono(3), expressions=()).try_evaluate() == 3
    assert kw.TrigoExpr(kw.Poly("x+1"), expressions=()).variables == {"x"}
    zero = trig(TrigoMethods.SIN, x, coefficient=0)
    zero.simplify()
    assert zero.try_evaluate() == 0
    power_zero = trig(TrigoMethods.COS, x, power=0)
    power_zero.simplify()
    assert power_zero.expressions == []


def test_trigoexpr_add_subtract_dispatch_branches():
    x = kw.Var("x")
    sine = kw.Sin(x)
    assert sine + 0 == sine
    assert isinstance(sine + 2, kw.TrigoExprs)
    assert isinstance(sine - 2, kw.TrigoExprs)
    assert kw.Sin(kw.Mono(0)) + 2 == 2
    assert kw.Sin(kw.Mono(0)) - 2 == -2
    assert sine + kw.Sin(x) == trig(TrigoMethods.SIN, x, coefficient=2)
    assert sine - kw.Sin(x) == 0
    assert isinstance(sine + kw.Cos(x), kw.TrigoExprs)
    assert isinstance(sine - kw.Cos(x), kw.TrigoExprs)
    assert isinstance(sine + "cos(x)", kw.TrigoExprs)
    assert isinstance(sine + kw.Poly("x"), kw.ExpressionSum)
    assert isinstance(sine - kw.Poly("x"), kw.ExpressionSum)
    with pytest.raises(TypeError):
        sine += object()


def test_trigoexpr_multiplication_dispatch_branches():
    x = kw.Var("x")
    sine = kw.Sin(x)
    assert (sine * 0).try_evaluate() == 0
    assert (sine * 3).coefficient == 3
    assert (sine * kw.Mono(2)).coefficient == 2
    squared = sine * kw.Sin(x)
    assert squared.expressions[0][2] == 2
    assert (sine * trig(TrigoMethods.COS, x)).expressions[-1][0] is TrigoMethods.COS
    assert (sine * trig(TrigoMethods.SIN, x, coefficient=0)).try_evaluate() == 0
    assert (sine * kw.TrigoExprs([kw.Cos(x)])).when(x=0.4).try_evaluate() == pytest.approx(
        math.sin(0.4) * math.cos(0.4)
    )
    assert isinstance(sine * kw.TrigoExprs([kw.Cos(x), kw.Tan(x)]), kw.TrigoExprs)
    assert isinstance(sine * kw.Abs(x), kw.ExpressionMul)
    assert isinstance(sine * "cos(x)", kw.TrigoExpr)
    with pytest.raises(TypeError):
        sine *= object()


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected_method"),
    [
        (TrigoMethods.SIN, TrigoMethods.COS, TrigoMethods.TAN),
        (TrigoMethods.COS, TrigoMethods.SIN, TrigoMethods.COT),
    ],
)
def test_trigoexpr_division_identity_equal_power(numerator, denominator, expected_method):
    x = kw.Var("x")
    result = trig(numerator, x) / trig(denominator, x)
    assert result.expressions[0][0] is expected_method


@pytest.mark.parametrize("first_power,second_power", [(3, 1), (1, 3)])
@pytest.mark.parametrize(
    ("numerator", "denominator"),
    [(TrigoMethods.SIN, TrigoMethods.COS), (TrigoMethods.COS, TrigoMethods.SIN)],
)
def test_trigoexpr_division_identity_unequal_powers(first_power, second_power, numerator, denominator):
    x = kw.Var("x")
    result = trig(numerator, x, first_power) / trig(denominator, x, second_power)
    if first_power > second_power:
        assert isinstance(result, kw.TrigoExpr)
    else:
        assert isinstance(result, kw.Fraction)


def test_trigoexpr_division_general_dispatch_branches():
    x = kw.Var("x")
    sine = kw.Sin(x)
    assert sine / sine == 1
    with pytest.raises(ZeroDivisionError):
        sine / trig(TrigoMethods.SIN, x, coefficient=0)
    assert (trig(TrigoMethods.SIN, x, 3) / trig(TrigoMethods.SIN, x)).expressions[0][2] == 2
    assert isinstance(sine / kw.Cos(kw.Var("y")), kw.Fraction)
    assert isinstance(sine / kw.TrigoExprs([kw.Sin(x), kw.Cos(x)]), kw.Fraction)
    assert isinstance(sine / kw.Poly("x"), kw.Fraction)


def test_trigoexpr_assignment_power_derivative_integral_branches():
    x = kw.Var("x")
    assigned_zero = kw.Sin(x)
    assigned_zero.assign(x=0)
    assert assigned_zero.try_evaluate() == 0
    assert kw.TrigoExpr(2, expressions=()).derivative() == 0
    assert trig(TrigoMethods.SIN, x, power=0).derivative() == 1
    doubled = kw.TrigoExpr(1, expressions=((TrigoMethods.SIN, x, 1), (TrigoMethods.SIN, x, 1)))
    assert doubled.derivative().when(x=0.4).try_evaluate() == pytest.approx(2 * math.sin(0.4) * math.cos(0.4))
    assert trig(TrigoMethods.ASEC, x).derivative() is None
    assert trig(TrigoMethods.ACSC, x).derivative() is None
    with pytest.warns(UserWarning):
        assert kw.Tan(x).integral() is None
    with pytest.warns(UserWarning):
        assert kw.TrigoExpr(1, expressions=()).integral() == 0


def test_trigoexpr_string_and_python_syntax_branches():
    x = kw.Var("x")
    assert str(trig(TrigoMethods.SIN, x, coefficient=0)) == "0"
    assert str(kw.TrigoExpr(2, expressions=())) == "2"
    assert str(-kw.Sin(x)).startswith("-")
    powered = trig(TrigoMethods.COS, x, power=2, coefficient=3)
    assert "^2" in str(powered)
    assert "**2" in powered.python_syntax()
    assert kw.TrigoExpr(3, expressions=()).python_syntax() == "3"
    raw = kw.TrigoExpr(1, expressions=((None, kw.Mono(2), 1),))
    assert str(raw) == "2"


@pytest.mark.parametrize(
    ("method", "first", "second", "coef1", "coef2", "expected"),
    [
        (TrigoMethods.SIN, 0, 0, 0, 0, True),
        (TrigoMethods.SIN, 0, 2 * math.pi, 2, 2, True),
        (TrigoMethods.SIN, 1, -1, 2, -2, True),
        (TrigoMethods.COS, 1, -1, 2, 2, True),
        (TrigoMethods.TAN, 0, math.pi, 2, 2, True),
        (TrigoMethods.SIN, 0, 1, 2, 2, False),
    ],
)
def test_equal_subexpressions_identity_branches(method, first, second, coef1, coef2, expected):
    result = kw.TrigoExpr.equal_subexpressions(
        kw.Mono(coef1), (method, kw.Poly(first), 1),
        kw.Mono(coef2), (method, kw.Poly(second), 1),
    )
    assert result is expected


def test_equal_subexpressions_cross_method_branches():
    assert kw.TrigoExpr.equal_subexpressions(
        kw.Mono(1), (TrigoMethods.SIN, kw.Poly(math.pi / 2), 1),
        kw.Mono(1), (TrigoMethods.COS, kw.Poly(0), 1),
    )
    assert not kw.TrigoExpr.equal_subexpressions(
        kw.Mono(1), (TrigoMethods.SIN, kw.Poly(0), 1),
        kw.Mono(2), (TrigoMethods.COS, kw.Poly(0), 1),
    )


def test_trigoexpr_equality_multi_expression_branches():
    x = kw.Var("x")
    first = kw.TrigoExpr(2, expressions=((TrigoMethods.SIN, x, 1), (TrigoMethods.COS, x, 1)))
    same = kw.TrigoExpr(2, expressions=((TrigoMethods.COS, x, 1), (TrigoMethods.SIN, x, 1)))
    assert first == same
    assert first != kw.TrigoExpr(3, expressions=((TrigoMethods.SIN, x, 1), (TrigoMethods.COS, x, 1)))
    assert first != kw.TrigoExpr(2, expressions=((TrigoMethods.SIN, x, 1), (TrigoMethods.TAN, x, 1)))
    assert kw.TrigoExpr(3, expressions=()) == 3
    assert kw.Sin(kw.Mono(0)) == 0
    assert kw.Sin(x) != kw.Poly("x")


def test_conversion_wrapper_and_string_constructor_branches():
    x = kw.Var("x")
    converted = kw.Sin(x).to_cos()
    assert converted.when(x=0).try_evaluate() == pytest.approx(math.sin(0))
    squared = kw.Sin(x)
    squared.expressions[0][2] = 2
    assert squared.to_cos() is not None
    multi = kw.Sin(x) * kw.Cos(x)
    nested = kw.Sin(multi)
    assert nested.to_cos().when(x=0.4).try_evaluate() == pytest.approx(
        math.sin(math.sin(0.4) * math.cos(0.4))
    )
    for factory in (kw.Sin, kw.Asin, kw.Cos, kw.Acos, kw.Tan, kw.Atan, kw.Cot, kw.Sec, kw.Acot, kw.ASec, kw.Csc, kw.ACsc):
        expression = factory("x")
        assert expression.variables == {"x"}


def test_trigoexprs_constructor_arithmetic_and_error_branches():
    x = kw.Var("x")
    combined = kw.TrigoExprs([kw.Sin(x), kw.Sin(x), kw.Cos(x), 2])
    assert len(combined.expressions) == 3
    assert isinstance(combined + kw.Tan(x), kw.TrigoExprs)
    assert isinstance(combined + kw.TrigoExprs([kw.Tan(x)]), kw.TrigoExprs)
    assert isinstance(combined + kw.Poly("x"), kw.ExpressionSum)
    assert isinstance(combined - kw.Tan(x), kw.TrigoExprs)
    assert isinstance(combined - kw.TrigoExprs([kw.Tan(x), kw.Cot(x)]), kw.TrigoExprs)
    assert isinstance(combined - kw.Poly("x"), kw.ExpressionSum)
    assert isinstance(combined + 3, kw.ExpressionSum)
    assert isinstance(3 - combined, kw.ExpressionSum)
    with pytest.raises(TypeError):
        kw.TrigoExprs(object())
    with pytest.raises(TypeError):
        kw.TrigoExprs([object()])
    with pytest.raises(TypeError):
        combined += object()
    with pytest.raises(TypeError):
        combined -= object()
    with pytest.raises(TypeError):
        _ = object() - combined


def test_trigoexprs_multiply_divide_and_power_branches():
    x = kw.Var("x")
    pair = kw.TrigoExprs([kw.Sin(x), kw.Cos(x)])
    assert isinstance(pair * kw.TrigoExprs([kw.Tan(x), kw.Cot(x)]), kw.TrigoExprs)
    assert (pair * kw.Mono(2)).when(x=0).try_evaluate() == 2
    with pytest.raises(TypeError):
        pair *= object()
    with pytest.raises(ZeroDivisionError):
        pair / 0
    assert isinstance(pair / 2, kw.TrigoExprs)
    assert isinstance(kw.TrigoExprs([kw.Sin(kw.Mono(0)), 2]) / 2, kw.Mono)
    assert isinstance(pair / kw.Mono(2), kw.TrigoExprs)
    assert kw.TrigoExprs([kw.Sin(x)]) / kw.Sin(x) == 1
    assert pair / pair == 1
    assert isinstance(pair / kw.TrigoExprs([kw.Tan(x)]), kw.Fraction)
    assert isinstance(pair / kw.Poly("x"), kw.Fraction)
    with pytest.raises(TypeError):
        pair /= object()
    assert pair**0 == 1
    assert pair**1 == pair
    assert isinstance(pair**2, kw.TrigoExprs)
    assert isinstance(pair ** kw.Var("n"), kw.Exponent)
    assert isinstance(pair ** kw.Mono(2), kw.TrigoExprs)
    assert isinstance(kw.TrigoExprs([kw.Sin(x)]) ** 2, kw.TrigoExpr)
    with pytest.raises(ValueError, match="EMPTY"):
        kw.TrigoExprs([]) ** 2


def test_trigonometric_plot_dimension_dispatch(monkeypatch):
    plots = importlib.import_module("kiwicalc.plotting.plots")
    calls = []
    monkeypatch.setattr(plots, "plot_function", lambda *args, **kwargs: calls.append("2d"))
    monkeypatch.setattr(plots, "plot_function_3d", lambda *args, **kwargs: calls.append("3d"))
    kw.Sin(kw.Var("x")).plot(show=False)
    trig(TrigoMethods.SIN, kw.Poly("x+y")).plot(show=False)
    kw.TrigoExprs([kw.Sin(kw.Var("x"))]).plot(show=False)
    kw.TrigoExprs([trig(TrigoMethods.SIN, kw.Poly("x+y"))]).plot(show=False)
    assert calls == ["2d", "3d", "2d", "3d"]
    with pytest.raises(ValueError):
        kw.Sin(kw.Poly(1)).plot(show=False)
    with pytest.raises(ValueError):
        kw.TrigoExprs([kw.TrigoExpr(1, expressions=())]).plot(show=False)
