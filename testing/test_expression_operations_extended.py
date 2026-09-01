import json
import math

import matplotlib.pyplot as plt
import pytest

import kiwicalc as kw


def test_fastpoly_full_contract(tmp_path):
    polynomial = kw.FastPoly("x^2-3x+2")
    assert polynomial.variables == ["x"]
    assert polynomial.num_of_variables == 1
    assert polynomial.degree == 2
    assert not polynomial.is_free_number
    assert polynomial.try_evaluate() is None
    assert polynomial.to_lambda()(3) == pytest.approx(2)
    assert -polynomial == kw.FastPoly("-x^2+3x-2")
    copied = polynomial.__copy__()
    assert copied == polynomial and copied is not polynomial
    assert polynomial.to_lambda()(4) == pytest.approx(6)
    assert polynomial.python_syntax() == "x**2-3*x+2"

    fig, ax = plt.subplots()
    polynomial.plot(values=[-1, 0, 1], show=False, fig=fig, ax=ax)
    assert ax.lines
    plt.close(fig)


def test_fastpoly_serialization_contract(tmp_path):
    polynomial = kw.FastPoly("x^2-3x+2")
    payload = polynomial.to_dict()
    assert kw.FastPoly.from_dict(payload) == polynomial
    assert kw.create_from_dict(payload) == polynomial
    assert kw.FastPoly.from_json(json.dumps(payload)) == polynomial
    output = tmp_path / "fastpoly.json"
    output.write_text(json.dumps(payload))
    assert kw.FastPoly.import_json(output) == polynomial


def test_fastpoly_calculus_contract():
    polynomial = kw.FastPoly("x^2-3x+2")
    assert polynomial.derivative().to_lambda()(3) == pytest.approx(3)
    assert polynomial.integral().derivative() == polynomial
    assert sorted(round(complex(root).real, 6) for root in polynomial.roots()) == [1, 2]


def test_fastpoly_arithmetic_contract():
    polynomial = kw.FastPoly("x^2-3x+2")
    assert polynomial + kw.FastPoly("x") == kw.FastPoly("x^2-2x+2")


def test_poly_analysis_and_numerical_methods():
    polynomial = kw.Poly("x^3-3x")
    assert polynomial.contains_variable("x")
    assert len(polynomial) == 2
    assert list(polynomial) == polynomial.expressions
    assert polynomial[0] in polynomial.expressions
    assert polynomial.partial_derivative(("x",)) == polynomial.derivative()
    assert polynomial.integral().derivative() == polynomial
    assert polynomial.discriminant() == 108
    assert polynomial.sorted() == polynomial
    polynomial.sort()


def test_poly_gcd_contract():
    polynomial = kw.Poly("x^3-3x")
    assert polynomial.gcd() == kw.Mono("x")
    assert polynomial.divide_by_gcd() == kw.Poly("x^2-3")


def test_poly_numerical_methods_contract():
    polynomial = kw.Poly("x^3-3x")
    roots = polynomial.roots()
    assert sorted(round(complex(root).real, 6) for root in roots) == pytest.approx(
        [-math.sqrt(3), 0, math.sqrt(3)], abs=1e-5
    )


def test_poly_report_contract():
    polynomial = kw.Poly("x^3-3x")
    data = polynomial.data(no_roots=True)
    assert data["derivative"] == polynomial.derivative()


def test_expression_sum_protocol_and_serialization():
    x = kw.Var("x")
    expression = kw.ExpressionSum([kw.Sin(x), kw.Abs(x), kw.Mono(2)])
    assert len(expression) == 3
    assert list(expression) == expression.expressions
    assert expression[0] == kw.Sin(x)
    assert expression.variables == {"x"}
    assert expression.when_all(x=-2).try_evaluate() == pytest.approx(math.sin(-2) + 4)
    assert expression.python_syntax()
    assert expression.__copy__() == expression
    assert expression != kw.ExpressionSum([kw.Sin(x)])

    poly_sum = kw.ExpressionSum([x, 2 * x, kw.Mono(3)])
    assert poly_sum.is_poly()
    assert poly_sum.to_poly() == kw.Poly("3x+3")


def test_expression_sum_assignment_contract():
    x = kw.Var("x")
    poly_sum = kw.ExpressionSum([x, 2 * x, kw.Mono(3)])
    poly_sum.assign_to_all(x=2)
    assert poly_sum.try_evaluate() == 9


def test_expression_sum_serialization_contract():
    x = kw.Var("x")
    expression = kw.ExpressionSum([kw.Sin(x), kw.Abs(x), kw.Mono(2)])
    assert kw.ExpressionSum.from_dict(expression.to_dict()) == expression


def test_expression_sum_derivative_contract():
    x = kw.Var("x")
    expression = kw.ExpressionSum([kw.Sin(x), kw.Abs(x), kw.Mono(2)])
    assert expression.derivative().when(x=2).try_evaluate() == pytest.approx(math.cos(2) + 1)


def test_expression_mul_protocol_and_serialization():
    x = kw.Var("x")
    expression = kw.ExpressionMul([kw.Sin(x), kw.Abs(x)])
    assert expression.variables == {"x"}
    assert expression.__copy__() == expression


def test_expression_mul_serialization_contract():
    x = kw.Var("x")
    expression = kw.ExpressionMul([kw.Sin(x), kw.Abs(x)])
    assert expression.python_syntax()
    assert kw.ExpressionMul.from_dict(expression.to_dict()) == expression


def test_expression_mul_arithmetic_contract():
    x = kw.Var("x")
    expression = kw.ExpressionMul([kw.Sin(x), kw.Abs(x)])
    assert -expression != expression
    assert expression + 2
    assert expression * 3


def test_expression_mul_scaled_derivative_and_zero_factor():
    x = kw.Var("x")
    expression = kw.ExpressionMul([3, x, x])
    assert expression.when(x=2).try_evaluate() == 12
    assert expression.derivative().when(x=2).try_evaluate() == 12
    assert kw.ExpressionMul([0, x]).when(x=4).try_evaluate() == 0


def test_expression_mul_assignment_contract():
    x = kw.Var("x")
    expression = kw.ExpressionMul([kw.Sin(x), kw.Abs(x)])
    assert expression.when(x=-2).try_evaluate() == pytest.approx(math.sin(-2) * 2)


def test_root_properties_arithmetic_and_calculus():
    x = kw.Var("x")
    root = kw.Root(x, root_by=3, coefficient=2)
    assert root.inside == x
    assert root.root == 3
    assert root.coefficient == 2
    assert root.variables == {"x"}
    assert root.when(x=8).try_evaluate() == pytest.approx(4)
    assert kw.Root.from_dict(root.to_dict()) == root
    assert root.__copy__() == root
    assert -root == -1 * root
    assert root + root == 2 * root
    assert root - root == 0
    assert root * 2 == 2 * root
    assert kw.Root(x, 3, 2) ** 3
    assert kw.Root(x, 3, 2) / 2 == kw.Root(x, 3)
    assert "**" in root.python_syntax()


def test_root_derivative_contract():
    derivative = kw.Sqrt(kw.Var("x")).derivative()
    assert derivative.when(x=4).try_evaluate() == pytest.approx(0.25)


def test_factorial_abs_and_exponent_metadata_serialization():
    x = kw.Var("x")
    factorial = kw.Factorial(x + 1, coefficient=2, power=2)
    assert factorial.variables == {"x"}
    assert factorial.when(x=2).try_evaluate() == 72
    assert kw.Factorial.from_dict(factorial.to_dict()) == factorial
    assert factorial.__copy__() == factorial
    assert factorial.python_syntax()

    absolute = kw.Abs(x - 2, coefficient=3, power=2)
    assert absolute.variables == {"x"}
    assert absolute.when(x=0).try_evaluate() == 12
    assert kw.Abs.from_dict(absolute.to_dict()) == absolute
    assert absolute.__copy__() == absolute

    exponent = kw.Exponent(2, x, coefficient=3)
    assert exponent.base == 2
    assert exponent.power == x
    assert exponent.variables == {"x"}
    assert exponent.when(x=3).try_evaluate() == 24
    assert kw.Exponent.from_dict(exponent.to_dict()) == exponent
    assert exponent.__copy__() == exponent
    assert (exponent * 2).when(x=3).try_evaluate() == 48
    assert (exponent / 3).when(x=3).try_evaluate() == 8


def test_absolute_derivative_contract():
    x = kw.Var("x")
    absolute = kw.Abs(x - 2, coefficient=3, power=2)
    assert absolute.derivative().when(x=4).try_evaluate() == pytest.approx(12)
    assert absolute.derivative().when(x=0).try_evaluate() == pytest.approx(-12)


def test_exponent_derivative_contract():
    x = kw.Var("x")
    exponent = kw.Exponent(2, x, coefficient=3)
    assert exponent.derivative().when(x=2).try_evaluate() == pytest.approx(12 * math.log(2))
    polynomial_power = kw.Exponent(x, 2, coefficient=3)
    assert polynomial_power.derivative().when(x=4).try_evaluate() == pytest.approx(24)


@pytest.mark.parametrize(
    ("factory", "point", "expected"),
    [
        (kw.Sin, math.pi / 2, 1),
        (kw.Cos, 0, 1),
        (kw.Tan, math.pi / 4, 1),
        (kw.Asin, 1, math.pi / 2),
        (kw.Acos, 1, 0),
        (kw.Atan, 1, math.pi / 4),
    ],
)
def test_trigonometric_factories_evaluate_and_round_trip(factory, point, expected):
    expression = factory(kw.Var("x"))
    assert expression.when(x=point).try_evaluate() == pytest.approx(expected)
    assert expression.variables == {"x"}
    assert kw.create_from_dict(expression.to_dict()) == expression
    assert expression.__copy__() == expression


@pytest.mark.parametrize(
    ("factory", "point", "expected"),
    [(kw.Cot, math.pi / 4, 1), (kw.Sec, 0, 1), (kw.Csc, math.pi / 2, 1)],
)
def test_reciprocal_trigonometric_factory_contract(factory, point, expected):
    assert factory(kw.Var("x")).when(x=point).try_evaluate() == pytest.approx(expected)
