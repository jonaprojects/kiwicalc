import json

import numpy as np
import pytest

import kiwicalc as kw


def test_quadratic_solver_string_and_degenerate_contracts():
    assert kw.solve_quadratic_from_str("5=5") == ()
    with pytest.raises(ValueError):
        kw.solve_quadratic_from_str("x^2+y^2=1")
    assert kw.solve_quadratic_real(1, -2, 1) == 1
    assert kw.solve_quadratic_real("x^2-2x+1=0", None, None) == 1
    assert kw.solve_cubic(0, 1, -3, 2) == kw.solve_quadratic(1, -3, 2)
    assert kw.solve_quartic(0, 1, -6, 11, -6) == kw.solve_cubic(1, -6, 11, -6)


def test_parametric_quadratic_numeric_and_symbolic():
    assert kw.solve_quadratic_params(1, -3, 2) == pytest.approx((2, 1))
    assert kw.solve_quadratic_params(kw.Mono(1), kw.Mono(-3), kw.Mono(2)) == pytest.approx((2, 1))
    roots = kw.solve_quadratic_params(kw.Var("a"), 2, 1)
    assert len(roots) == 2
    assert all(root.variables == {"a"} for root in roots)


@pytest.mark.parametrize(
    ("coefficients", "expected"),
    [([4], None), ([2, -6], [3]), ([1, -3, 2], (2 + 0j, 1 + 0j))],
)
def test_polynomial_solver_degree_dispatch(coefficients, expected):
    result = kw.solve_polynomial(coefficients)
    if expected is None:
        assert result is None
    elif len(coefficients) == 3:
        assert result == pytest.approx(expected)
    else:
        assert result == expected


def test_polynomial_solver_string_and_high_degree():
    roots = kw.solve_polynomial("x^5-x=0")
    assert len(roots) == 5
    for root in roots:
        assert root**5 - root == pytest.approx(0, abs=1e-4)
    assert kw.solve_poly_by_factoring(None) == {}
    assert kw.solve_poly_by_factoring([1, -3, 2]) == pytest.approx((2, 1))


def test_linear_solver_special_outputs():
    assert kw.solve_linear("x=x") == np.inf
    assert kw.solve_linear("x=x+1") is None
    assert kw.solve_linear("2x=8", get_dict=True) == {"x": 4}
    assert json.loads(kw.solve_linear("2x=8", get_json=True)) == {"variable": "x", "result": 4}
    with pytest.raises(ValueError):
        kw.solve_linear("x+y=2")


@pytest.mark.parametrize(
    ("source", "expected"),
    [("2x<8", "x<4"), ("2x>8", "x>4"), ("2x<=8", "x<=4"), ("2x>=8", "x>=4")],
)
def test_linear_inequality_operator_dispatch(source, expected):
    assert kw.solve_linear_inequality(source) == expected


def test_linear_inequality_validation():
    with pytest.raises(ValueError):
        kw.solve_linear_inequality("2x+1")
    with pytest.raises(ValueError):
        kw.solve_linear_inequality("x<2<3")


def test_random_linear_return_modes(monkeypatch):
    values = iter([2.0, -4.0])
    monkeypatch.setattr("random.uniform", lambda a, b: next(values))
    expression, solution, coefficients = kw.random_linear(get_solution=True, get_coefficients=True)
    assert expression == "2x-4"
    assert solution == 2
    assert coefficients == (2, -4)

    values = iter([3.0, 6.0])
    monkeypatch.setattr("random.uniform", lambda a, b: next(values))
    assert kw.random_linear(get_coefficients=True) == ("3x+6", (3, 6))


def test_equation_explicit_variables_and_lazy_solution():
    equation = kw.LinearEquation("3t+2=11", variables=("t",), calc_now=True)
    assert equation.variables == ["t"]
    assert equation.variables_dict == {"t": 0, "number": 0}
    assert equation.solution == 3
    assert equation.__copy__() is not equation


def test_quadratic_coefficients_modes_and_copy():
    equation = kw.QuadraticEquation("x^2-3x+2=0")
    assert equation.coefficients() == [1, -3, 2]
    assert equation.solve("parametric") == pytest.approx((2, 1))
    assert equation.__copy__().coefficients() == equation.coefficients()
    assert "variables" in repr(equation)
    with pytest.warns(UserWarning):
        assert equation.solve("unknown") is None


def test_cubic_and_quartic_coefficients_and_copies():
    cubic = kw.CubicEquation("x^3-6x^2+11x-6=0")
    quartic = kw.QuarticEquation("x^4-5x^2+4=0")
    assert cubic.coefficients() == [1, -6, 11, -6]
    assert quartic.coefficients() == [1, 0, -5, 0, 4]
    assert cubic.__copy__().coefficients() == cubic.coefficients()
    assert quartic.__copy__().coefficients() == quartic.coefficients()


def test_polyequation_constructor_forms_and_validation():
    from_string = kw.PolyEquation("x^2=4")
    from_objects = kw.PolyEquation(kw.Poly("x^2"), kw.Mono(4))
    assert from_string.to_PolyExpr() == from_objects.to_PolyExpr()
    assert sorted(round(complex(root).real, 6) for root in from_string.solution) == [-2, 2]
    assert from_string.solution is from_string.solution
    assert from_string.__copy__().to_PolyExpr() == from_string.to_PolyExpr()
    with pytest.raises(TypeError):
        kw.PolyEquation(None)
    with pytest.raises(TypeError):
        kw.PolyEquation(object(), object())


def test_polyequation_random_expression_branches(monkeypatch):
    monkeypatch.setattr("random.randint", lambda a, b: a)
    assert isinstance(kw.PolyEquation.random_expression(of_order=1, variable="t"), str)
    expression = kw.PolyEquation.random_expression(values=(1, 2), of_order=3, variable="t", all_powers=True)
    assert "t^3" in expression and "t^2" in expression and "t" in expression
    assert "= 0" in kw.PolyEquation.random_quadratic(values=(1, 2), variable="t", all_powers=True)


def test_polynomial_equation_worksheet_smoke(tmp_path):
    single = tmp_path / "single.pdf"
    pages = tmp_path / "pages.pdf"
    assert kw.PolyEquation.random_worksheet(path=single, num_of_equations=1, degrees_range=(2, 2)) is not False
    kw.PolyEquation.random_worksheets(path=pages, num_of_pages=1, equations_per_page=1, degrees_range=(2, 2))
    assert single.read_bytes().startswith(b"%PDF")
    assert pages.read_bytes().startswith(b"%PDF")
