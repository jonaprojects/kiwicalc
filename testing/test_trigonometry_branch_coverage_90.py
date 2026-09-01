import math

import pytest

import kiwicalc as kw
from kiwicalc.core.operators import TrigoMethods


def trig(method, inside, power=1, coefficient=1):
    return kw.TrigoExpr(coefficient, expressions=((method, inside, power),))


def test_trigo_conversion_coefficient_and_evaluated_arithmetic_edges():
    x = kw.Var("x")
    multiple = kw.Sin(x)
    multiple.expressions.append([TrigoMethods.COS, x, 1])
    with pytest.raises(ValueError):
        multiple.to_cos()
    coefficient = kw.Poly("x+1")
    expression = kw.TrigoExpr(coefficient, expressions=((TrigoMethods.SIN, x, 1),))
    assert expression.coefficient == coefficient and expression.coefficient is not coefficient

    constant = kw.Sin(kw.Mono(0))
    assert constant + 2 == 2
    assert constant - 2 == -2
    assert constant + kw.Abs(3) == 3
    assert constant - kw.Abs(3) == -3


def test_trigo_direct_division_assignment_and_derivative_edges():
    x = kw.Var("x")
    zero = kw.TrigoExpr(0, expressions=())
    assert isinstance(kw.Sin(x).divide_by_trigo(zero), ZeroDivisionError)
    assert kw.TrigoExpr(2, expressions=()).divide_by_trigo(kw.TrigoExpr(3, expressions=())) is True

    assigned = trig(TrigoMethods.SIN, x)
    assigned.assign(x=math.pi / 2)
    assert assigned.try_evaluate() == pytest.approx(1)

    unknown = kw.TrigoExpr(1, expressions=((None, x, 1),))
    with pytest.raises(ValueError):
        unknown.derivative()
    doubled = kw.TrigoExpr(2, expressions=((TrigoMethods.SIN, x, 1), (TrigoMethods.SIN, x, 2)))
    assert doubled.derivative().variables == {"x"}
    unlike = kw.TrigoExpr(1, expressions=((TrigoMethods.SIN, x, 1), (TrigoMethods.COS, x, 1)))
    assert unlike.derivative() is None
    unlike.partial_derivative(("x", "y"))


@pytest.mark.parametrize(
    "method,first,second,coef1,coef2",
    [
        (TrigoMethods.SIN, lambda x: x + kw.pi, lambda x: x, 1, -1),
        (TrigoMethods.SIN, lambda x: x, lambda x: kw.pi - x, 1, 1),
        (TrigoMethods.SIN, lambda x: x, lambda x: -x, 1, -1),
        (TrigoMethods.COS, lambda x: x, lambda x: -x, 1, 1),
        (TrigoMethods.COS, lambda x: x, lambda x: kw.pi - x, 1, -1),
        (TrigoMethods.TAN, lambda x: x + kw.pi, lambda x: x, 1, 1),
        (TrigoMethods.TAN, lambda x: x, lambda x: -x, 1, -1),
        (TrigoMethods.TAN, lambda x: x, lambda x: kw.pi - x, 1, -1),
    ],
)
def test_equal_subexpressions_symbolic_same_method_identities(method, first, second, coef1, coef2):
    x = kw.Var("x")
    assert kw.TrigoExpr.equal_subexpressions(
        kw.Mono(coef1), (method, first(x), 1),
        kw.Mono(coef2), (method, second(x), 1),
    )


def test_equal_subexpressions_symbolic_cross_method_identities():
    x = kw.Var("x")
    assert kw.TrigoExpr.equal_subexpressions(
        kw.Mono(1), (TrigoMethods.SIN, x + kw.pi / 2, 1),
        kw.Mono(1), (TrigoMethods.COS, x, 1),
    )
    assert kw.TrigoExpr.equal_subexpressions(
        kw.Mono(-1), (TrigoMethods.SIN, x - kw.pi / 2, 1),
        kw.Mono(1), (TrigoMethods.COS, x, 1),
    )


def test_trigoexprs_string_evaluation_reverse_subtraction_and_division_edges():
    x = kw.Var("x")
    assert kw.TrigoExprs("sin(0)") + "cos(0)" == 1
    assert kw.TrigoExprs("sin(0)") - "cos(0)" == -1
    symbolic = kw.TrigoExprs([kw.Sin(x), kw.Cos(x)])
    assert isinstance(kw.Abs(x) - symbolic, kw.ExpressionSum)
    assert kw.TrigoExprs([kw.Sin(kw.Mono(0))]) / kw.Abs(2) == 0
    assert isinstance(kw.TrigoExprs([kw.Sin(kw.Mono(0))]) / kw.Var("y"), kw.Fraction)

    divisible = kw.TrigoExprs([
        trig(TrigoMethods.SIN, x, 2),
        trig(TrigoMethods.SIN, x, 1) * kw.Cos(x),
    ])
    result = divisible / kw.Sin(x)
    assert isinstance(result, kw.TrigoExprs)
    assert isinstance(symbolic / kw.Tan(x), kw.Fraction)


def test_trigoexprs_three_item_power_loop_and_string_inverse_constructors():
    x = kw.Var("x")
    triple = kw.TrigoExprs([kw.Sin(x), kw.Cos(x), kw.Tan(x)])
    assert isinstance(triple**2, kw.TrigoExprs)
    assert kw.ASec("x").variables == {"x"}
    assert kw.ACsc("x").variables == {"x"}
