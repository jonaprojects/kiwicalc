import numpy as np
import pytest

from kiwicalc.parsing import parse_expression as parsing


def test_sparse_strict_degree_parsers_accept_valid_polynomials():
    parser = parsing.ParseExpression
    assert parser.parse_quadratic("x**2+2", ("x",)) == {"x": [1, 0], "free": 2}
    assert parser.parse_cubic("x^3-8", ("x",)) == {"x": [1, 0, 0], "free": -8}
    assert parser.parse_cubic("2x^3-3x^2+4x-5", ("x",)) == {
        "x": [2, -3, 4], "free": -5
    }
    assert parser.parse_quartic("x**4+2", ("x",)) == {
        "x": [1, 0, 0, 0], "free": 2
    }
    assert parser.parse_quartic("2x^4-3x^3+4x^2-5x+6", ("x",)) == {
        "x": [2, -3, 4, -5], "free": 6
    }


@pytest.mark.parametrize(
    ("method", "source", "variables", "message"),
    [
        ("parse_quadratic", "x+1", ("x",), "quadratic term"),
        ("parse_cubic", "x^2+1", ("x",), "cubic term"),
        ("parse_quartic", "x^3+1", ("x",), "quartic term"),
        ("parse_quadratic", "x^2+y", ("x", "y"), "exactly 1 variable"),
        ("parse_cubic", "x^3+y", ("x", "y"), "exactly 1 variable"),
        ("parse_quartic", "x^4+y", ("x", "y"), "exactly 1 variable"),
    ],
)
def test_strict_degree_parsers_reject_wrong_shape(method, source, variables, message):
    with pytest.raises(ValueError, match=message):
        getattr(parsing.ParseExpression, method)(source, variables)


def test_non_strict_degree_parsers_delegate_to_general_parser():
    parser = parsing.ParseExpression
    expected = {"x": [3], "free": 2}
    assert parser.parse_quadratic("3x+2", ("x",), strict_syntax=False) == expected
    assert parser.parse_cubic("3x+2", ("x",), strict_syntax=False) == expected
    assert parser.parse_quartic("3x+2", ("x",), strict_syntax=False) == expected


def test_polynomial_parser_numpy_and_error_edges():
    parser = parsing.ParseExpression
    values, variables = parser.parse_polynomial(
        "2t^3-t+4", numpy_array=True, get_variables=True
    )
    assert isinstance(values, np.ndarray)
    assert values.tolist() == [2, 0, -1, 4]
    assert variables == ["t"]
    assert parser.to_coefficients("12") == [12]
    with pytest.raises(ValueError, match="1 variable"):
        parser.to_coefficients("x+y")
    with pytest.raises(ValueError, match="free number"):
        parser.parse_polynomial("not-a-number", variables=())
    with pytest.raises(ValueError, match="invalid coefficient"):
        parser.parse_polynomial("badx", variables=("x",))
    with pytest.raises(ValueError, match="invalid power"):
        parser.parse_polynomial("2x^bad", variables=("x",))


def test_unparsing_edge_contracts():
    parser = parsing.ParseExpression
    assert parser.unparse_linear({"x": 0, "free": 0}) == "0"
    assert parser.unparse_linear({"x": (2, -1), "free": 0}) == "2x-x"
    with pytest.warns(UserWarning, match="Unrecognized syntax"):
        assert parser.unparse_polynomial({"x": [1, 0], "free": 1}, "latex") == "x^2+1"
    with pytest.warns(UserWarning, match="Unrecognized syntax"):
        assert parser.coefficients_to_str([1, 0, 1], "x", "latex") == "x^2+1"
    with pytest.raises(ValueError, match="At least 1 coefficient"):
        parser.coefficients_to_str([])
    assert parser.coefficients_to_str([7]) == "7"
    assert parser.coefficients_to_str([2, -1, 3], syntax="pythonic") == "2*x**2-x+3"


def test_small_parser_helpers_and_validation():
    assert parsing.fetch_variable({}) is None
    assert parsing.fetch_variable({"z": 3}) == "z"
    assert parsing.fetch_power({"z": 3}) == 3
    assert parsing.poly_from_str("x+2", get_list=True)[-1].try_evaluate() == 2
    assert parsing.poly_frac_from_str("x+1/x-1").when(x=2).try_evaluate() == 3
    assert len(parsing.TrigoExprs_from_str("sin(x)-cos(x)", get_list=True)) == 2
    with pytest.raises(ValueError, match="opening parenthesis"):
        parsing.log_from_str("logx")
    with pytest.raises(ValueError, match=r"contain log\(\) or ln\(\)"):
        parsing.log_from_str("x+1")


def test_log_parser_variants_and_errors():
    coefficient, inside, base, power = parsing.log_from_str("-ln(x)^2", get_tuple=True)
    assert coefficient == -1
    assert base == "e"
    assert power == 2
    assert inside.variables == {"x"}
    assert parsing.log_from_str("log(x)", get_tuple=True)[2] == 10
    with pytest.raises(ValueError, match="Invalid _coefficient"):
        parsing.log_from_str("badlog(x)")
    with pytest.raises(ValueError, match="ending parenthesis"):
        parsing.log_from_str("log(x")
