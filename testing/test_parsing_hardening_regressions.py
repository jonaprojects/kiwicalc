import copy
import math

import pytest

import kiwicalc as kw
from kiwicalc.parsing import parse_equation, parse_expression


@pytest.mark.parametrize(
    "source",
    ("x+1", "=1", "x=", "x=1=2", "", 42),
)
def test_two_sided_parsers_reject_malformed_equations(source):
    with pytest.raises((TypeError, ValueError)):
        parse_equation.extract_dict_from_equation(source)
    with pytest.raises((TypeError, ValueError)):
        parse_equation.equation_to_one_side(source)


def test_equation_variable_order_is_deterministic():
    assert parse_equation.get_equation_variables("z+x+y=1") == ["x", "y", "z"]
    assert list(parse_equation.extract_dict_from_equation("z+x+y=1")) == [
        "x", "y", "z", "number"
    ]


def test_dictionary_subtraction_does_not_mutate_inputs():
    first = {"x": 2, "number": 1}
    second = {"x": 1, "y": 3, "number": -2}
    original_first, original_second = copy.deepcopy(first), copy.deepcopy(second)
    with pytest.warns(UserWarning):
        result = parse_equation.subtract_dicts(first, second)
    assert result == {"x": 1, "number": 3, "y": -3}
    assert first == original_first
    assert second == original_second


def test_equation_normalization_preserves_scientific_notation_signs():
    assert parse_equation.equation_to_one_side("x=1e-3-2e-4") == "x-1e-3+2e-4"
    assert parse_equation.ParseEquation.parse_polynomial("1e-3x=2e-3") == pytest.approx(
        [0.001, -0.002]
    )
    assert kw.solve_linear("1e-3x=2e-3") == pytest.approx(2)


def test_split_expression_respects_groups_and_numeric_exponents():
    split = parse_expression.split_expression
    assert split("x+(y-1)-2") == ["x", "+(y-1)", "-2"]
    assert split("1e-3x+2") == ["1e-3x", "+2"]
    assert split("x^-2+1") == ["x^-2", "+1"]
    with pytest.raises(ValueError, match="Unclosed"):
        split("x+(y-1")
    with pytest.raises(ValueError, match="Unmatched"):
        split("x+y)")


def test_polynomial_parser_accepts_explicit_multi_character_variable():
    parsed = parse_expression.ParseExpression.parse_polynomial(
        "2theta^2-theta+1", variables=("theta",)
    )
    assert parsed == {"theta": [2, -1], "free": 1}


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("xy+1", "monomial"),
        ("x^1.5+1", "non-negative integers"),
        ("x^-1+1", "non-negative integers"),
        ("2*xjunk+1", "Unknown variable"),
    ],
)
def test_polynomial_parser_rejects_ambiguous_or_unsupported_terms(source, message):
    with pytest.raises(ValueError, match=message):
        parse_expression.ParseExpression.parse_polynomial(source, variables=("x", "y"))


def test_polynomial_parser_supports_explicit_multiplication():
    assert parse_expression.ParseExpression.to_coefficients("2*x^2-3*x+1", "x") == [2, -3, 1]


def test_equation_whitespace_is_normalized_at_the_boundary():
    assert parse_equation.equation_to_one_side(" 2x + 1 = x + 4 ") == "2x+1-x-4"
    assert parse_equation.ParseEquation.parse_quadratic(" x^2 = 1 ") == [1, 0, -1]


def test_nested_parentheses_and_constants_are_preserved_inside_logarithms():
    expression = kw.Log("log(sin(x)*cos(x)+7,5)", dtype="trigo")
    value = expression.when(x=0.2).try_evaluate()
    assert value == pytest.approx(
        math.log(math.sin(0.2) * math.cos(0.2) + 7, 5), abs=1e-5
    )
