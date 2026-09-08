import inspect
import math
import random
import copy

import numpy as np
import pytest

import kiwicalc as kw


PUBLIC_SIGNATURES = {
    "solve_quadratic_from_str": ("expression", "real", "strict_syntax"),
    "solve_quadratic": ("a", "b", "c"),
    "solve_quadratic_real": ("a", "b", "c"),
    "solve_quadratic_params": ("a", "b", "c"),
    "solve_cubic": ("a", "b", "c", "d"),
    "solve_cubic_real": ("a", "b", "c", "d"),
    "solve_quartic": ("a", "b", "c", "d", "e"),
    "solve_polynomial": ("coefficients", "epsilon", "nmax"),
    "solve_poly_by_factoring": ("coefficients",),
    "solve_linear": ("equation", "variables", "get_dict", "get_json"),
    "solve_linear_inequality": ("equation", "variables"),
    "solve_linear_system": ("equations", "variables"),
    "solve_poly_system": ("equations", "initial_vals", "epsilon", "nmax", "show_steps"),
    "random_linear": ("coefs_range", "digits_after", "variable", "get_solution", "get_coefficients"),
    "random_polynomial": ("degree", "solutions_range", "digits_after", "variable", "python_syntax", "get_solutions"),
    "random_polynomial2": ("degree", "values", "digits_after", "variable", "python_syntax"),
    "random_linear_system": ("variables", "solutions_range", "coefficients_range", "digits_after", "get_solutions"),
    "random_poly_system": ("variables",),
}


def _relative_residual(coefficients, root):
    coefficients = np.asarray(coefficients, dtype=complex)
    scale = max(abs(coefficients))
    coefficients = coefficients / scale
    degree = len(coefficients) - 1
    denominator = sum(
        abs(coefficient) * max(1.0, abs(root)) ** (degree - index)
        for index, coefficient in enumerate(coefficients)
    )
    return abs(np.polyval(coefficients, root)) / max(1.0, denominator)


def test_public_solver_signatures_and_exports_are_frozen():
    for name, parameters in PUBLIC_SIGNATURES.items():
        assert name in kw.__all__
        assert tuple(inspect.signature(getattr(kw, name)).parameters) == parameters


def test_parse_error_types_are_public_value_errors():
    for error_type in (
        kw.EquationParseError,
        kw.UnsupportedExpressionError,
        kw.AmbiguousVariableError,
    ):
        assert issubclass(error_type, ValueError)
        assert error_type.__name__ in kw.__all__


def test_legacy_return_containers_are_preserved():
    assert isinstance(kw.solve_quadratic(1, -3, 2), tuple)
    assert isinstance(kw.solve_cubic(1, -6, 11, -6), list)
    assert isinstance(kw.solve_quartic(1, 0, -5, 0, 4), list)
    assert isinstance(kw.solve_polynomial([1, 0, 0, 0, 0, -1]), set)
    assert isinstance(kw.solve_linear("x=1"), float)
    assert kw.solve_linear("x=x") == np.inf
    assert kw.solve_linear("1=2") is None


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("2(x+1)", [2, 2]),
        ("(x-1)(x+1)", [1, 0, -1]),
        ("((x+1)^2-1)", [1, 2, 0]),
        ("-x^2+2x-1", [-1, 2, -1]),
        ("2*x**2-3*x+1", [2, -3, 1]),
    ],
)
def test_recursive_polynomial_parser(source, expected):
    assert kw.ParseExpression.to_coefficients(source, "x") == pytest.approx(expected)


def test_legacy_adjacent_variables_remain_multiplication_for_poly_objects():
    expression = kw.poly_from_str("2xy+x")
    assert expression.when(x=3, y=4).try_evaluate() == 27
    with pytest.raises(kw.UnsupportedExpressionError, match="mixed monomial"):
        kw.ParseExpression.parse_polynomial("2xy+x", variables=("x", "y"))


def test_explicit_multi_character_equation_variable():
    equation = kw.QuadraticEquation("theta^2=1", variables=("theta",))
    assert equation.coefficients() == [1, 0, -1]
    assert equation.solve() == pytest.approx((1, -1))


def test_explicit_multi_character_poly_equation_survives_copy_and_reverse():
    equation = kw.PolyEquation("theta^2=1", variables=("theta",))
    assert sorted(equation.solve(), key=lambda root: root.real) == pytest.approx([-1, 1])
    for derived in (copy.copy(equation), reversed(equation)):
        assert isinstance(derived, kw.PolyEquation)
        assert derived.variables == ["theta"]
        assert sorted(derived.solve(), key=lambda root: root.real) == pytest.approx([-1, 1])


def test_quadratic_scaling_avoids_overflow_and_underflow():
    for scale in (1e-300, 1e300):
        roots = kw.solve_quadratic(scale, -scale, -scale)
        assert sorted(root.real for root in roots) == pytest.approx(
            [(1 - math.sqrt(5)) / 2, (1 + math.sqrt(5)) / 2]
        )


def test_fixed_degree_scaling_vieta_and_conjugate_symmetry():
    base = [1, -2, 3, -4, 5]
    for scale in (1e-150, 1.0, 1e150):
        roots = kw.solve_quartic(*(np.asarray(base) * scale))
        assert all(_relative_residual(base, root) < 1e-11 for root in roots)
        assert sum(roots) == pytest.approx(2, abs=1e-10)
        for root in roots:
            if abs(root.imag) > 1e-10:
                assert any(candidate == pytest.approx(root.conjugate(), abs=1e-10) for candidate in roots)


def test_nearby_distinct_roots_are_not_consolidated():
    expected = [1, 1.001, 2, 3]
    coefficients = np.poly(expected)
    actual = kw.solve_quartic(*coefficients)
    assert len(actual) == 4
    assert sorted(root.real for root in actual) == pytest.approx(expected, abs=1e-8)


def test_rectangular_linear_system_contracts():
    assert kw.solve_linear_system(("x+y=3", "2x+2y=6", "x-y=1")) == pytest.approx(
        {"x": 2, "y": 1}
    )
    with pytest.raises(ValueError, match="unique"):
        kw.solve_linear_system(("x+y+z=3", "x-y=1"))
    with pytest.raises(ValueError, match="inconsistent"):
        kw.solve_linear_system(("x+y=3", "2x+2y=7", "x-y=1"))


def test_polynomial_system_uses_structural_parenthesis_parsing():
    result = kw.solve_poly_system(
        ("(x+y)-3=0", "2(x-y)=2"), initial_vals={"x": 0.0, "y": 0.0}
    )
    assert result == pytest.approx({"x": 2, "y": 1})


def test_polynomial_system_damping_escapes_newton_cycle():
    result = kw.solve_poly_system(
        ("x^3-2x+2=0",), initial_vals={"x": 0.0}, nmax=100
    )
    assert result["x"] == pytest.approx(-1.7692923542386314, abs=1e-8)
    assert abs(result["x"] ** 3 - 2 * result["x"] + 2) < 1e-8


def test_polynomial_system_requires_exact_initial_variable_set():
    with pytest.raises(ValueError, match="initial_vals"):
        kw.solve_poly_system(
            ("x+y=2", "x-y=0"), initial_vals={"x": 0.0, "z": 0.0}
        )


def test_generated_polynomial_matrix_uses_structural_parsing():
    matrix = kw.generate_polynomial_matrix(("(x+y)-3=0", "2(x-y)=2"))
    evaluated = [
        sum(entry.when(x=2, y=1).try_evaluate() for entry in row)
        for row in matrix.matrix
    ]
    assert evaluated == pytest.approx([0, 0])


def test_linear_equation_simplify_invalidates_cached_solution():
    equation = kw.LinearEquation("0.333333x=1", calc_now=True)
    previous = equation.solution
    equation.simplify(round_coefficients=True)
    assert equation._solution is None
    assert equation.solution != previous


@pytest.mark.parametrize(
    "call",
    [
        lambda: kw.random_linear((0, 0)),
        lambda: kw.random_polynomial(2, solutions_range=(0, 0)),
        lambda: kw.random_polynomial2(2, values=(0, 0)),
        lambda: kw.random_linear(digits_after=-1),
        lambda: kw.random_polynomial2(2, variable=""),
        lambda: kw.random_linear_system(("x", "x")),
        lambda: kw.random_linear_system(("x",), coefficients_range=(0, 0)),
    ],
)
def test_random_generators_reject_impossible_or_invalid_requests(call):
    with pytest.raises((TypeError, ValueError)):
        call()


def test_generated_equations_match_their_advertised_solutions():
    random_state = random.getstate()
    try:
        for seed in range(20):
            random.seed(seed)
            expression, solution = kw.random_linear(get_solution=True)
            assert kw.solve_linear(f"{expression}=0") == pytest.approx(solution, abs=1e-5)
            polynomial, roots = kw.random_polynomial(4, get_solutions=True)
            coefficients = kw.ParseExpression.to_coefficients(polynomial, "x")
            assert all(_relative_residual(coefficients, root) < 1e-12 for root in roots)
    finally:
        random.setstate(random_state)
