import numpy as np
import pytest

import kiwicalc as kw
from kiwicalc.parsing import parse_equation, parse_expression


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("x+2-y", ["x", "+2", "-y"]),
        ("x+(y-2)-3", ["x", "+(y-2)", "-3"]),
        ("-x", ["-x"]),
    ],
)
def test_split_expression_preserves_nested_groups(source, expected):
    assert parse_expression.split_expression(source) == expected


def test_basic_expression_parsers():
    coefficient, variables = parse_expression.mono_from_str("-3x^2*y", get_tuple=True)
    assert coefficient == -3
    assert variables == {"x": 2, "y": 1}
    assert parse_expression.poly_from_str("x^2+2x+1") == kw.Poly("x^2+2x+1")
    assert str(parse_expression.monic_poly_from_coefficients([1, -3, 2])) == "x^2-3x+2x^0"
    numerator, denominator = parse_expression.poly_frac_from_str("x+1/x-1", get_tuple=True)
    assert numerator == kw.Poly("x+1")
    assert denominator == kw.Poly("x-1")
    assert parse_expression.extract_variables_from_expression("3x+2y") == {"x", "y"}


def test_trigonometric_log_and_surface_parsers():
    coefficient, expressions = parse_expression.TrigoExpr_from_str("2sin(x)^2", get_tuple=True)
    assert coefficient.try_evaluate() == 2
    assert len(expressions) == 1
    parsed_sum = parse_expression.TrigoExprs_from_str("sin(x)+cos(x)")
    assert parsed_sum.when(x=0).try_evaluate() == pytest.approx(1)
    coefficient, inside, base, power = parse_expression.log_from_str("3log(x,2)^2", get_tuple=True)
    assert coefficient == 3
    assert inside == kw.Poly("x")
    assert base == 2
    assert power == 2
    assert parse_expression.surface_from_str("2x+3y+4z-5=0", get_coefficients=True) == [2, 3, 4, -5]


def test_parse_and_unparse_linear_expression():
    parsed = parse_expression.ParseExpression.parse_linear("2x-3y+5", ("x", "y"))
    assert parsed == {"x": 2, "y": -3, "free": 5}
    assert parse_expression.ParseExpression.unparse_linear({"x": 2, "y": -3, "free": 5}) == "2x-3y+5"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("x^2+2x+1", {"x": [1, 2], "free": 1}),
        ("2x^2+0x-8", {"x": [2, 0], "free": -8}),
    ],
)
def test_parse_quadratic(source, expected):
    assert parse_expression.ParseExpression.parse_quadratic(source, ("x",)) == expected


def test_polynomial_coefficient_round_trip():
    parser = parse_expression.ParseExpression
    coefficients, variable = parser.to_coefficients("2t^4-3t^2+t-7", get_variable=True)
    assert coefficients == [2, 0, -3, 1, -7]
    assert variable == "t"
    assert parser.coefficients_to_str(coefficients, variable) == "2t^4-3t^2+t-7"
    assert parser.parse_polynomial("2x^3-x+4", get_variables=True)[1] == ["x"]
    assert np.asarray(parser.parse_polynomial("x^2+1", numpy_array=True)).tolist() == [1, 0, 1]


def test_equation_dictionary_operations_and_normalization():
    assert parse_equation.extract_dict_from_equation("2x+1=x+4") == {"x": 0, "number": 0}
    assert parse_equation.add_or_sub_coefficients([1, 2], [3, 4]) == [4, 6]
    first = {"x": 2, "number": 1}
    second = {"x": 1, "y": 3, "number": -2}
    with pytest.warns(UserWarning):
        assert parse_equation.subtract_dicts(first, second) == {"x": 1, "number": 3, "y": -3}
    assert parse_equation.equation_to_one_side("2x+1=x+4") == "2x+1-x-4"
    assert set(parse_equation.get_equation_variables("2x+y=3")) == {"x", "y"}


def test_equation_parsers_for_polynomial_degrees():
    assert parse_equation.ParseEquation.parse_polynomial("x^3-6x^2+11x=6") == [1, -6, 11, -6]
    assert parse_equation.ParseEquation.parse_quadratic("x^2-5x+6=0") == [1, -5, 6]
    assert [str(item) for item in parse_equation.coefficients_to_expressions([1, -3, 2])] == ["x^2", "-3x", "2x^0"]
