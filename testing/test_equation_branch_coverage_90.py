import importlib

import pytest

import kiwicalc as kw


single = importlib.import_module("kiwicalc.equations.single")


def test_equation_solver_dispatch_and_degenerate_branches(monkeypatch, capsys):
    with pytest.raises(TypeError):
        kw.solve_quadratic_params("x^2=1", 0, 0)
    assert "implemented" in capsys.readouterr().out
    roots = kw.solve_quadratic_params(kw.Var("a"), kw.Var("b"), kw.Var("c"))
    assert len(roots) == 2
    assert kw.solve_cubic(1, 0, 0, 0) == [0]
    monkeypatch.setattr(single, "solve_cubic", lambda *args: [])
    assert single.solve_cubic_real(1, 0, 0, 0) == []


@pytest.mark.parametrize(
    "cubic_roots,expected",
    [([0], [0]), ([1, 0, 0], [0]), ([1, 4, 9], None)],
)
def test_quartic_internal_root_selection(monkeypatch, cubic_roots, expected):
    monkeypatch.setattr(single, "solve_cubic", lambda *args: cubic_roots)
    result = single.solve_quartic(2, 0, 0, 0, 0)
    if expected is not None:
        assert result == expected
    else:
        assert len(result) == 4


def test_polynomial_degree_five_and_explicit_inequality_variables():
    roots = kw.solve_polynomial([1, 0, 0, 0, 0, -1])
    assert len(roots) == 5
    assert kw.solve_polynomial([1, 0, 0, 0, -1])
    assert kw.solve_linear_inequality("2x<8", {"x": 0, "number": 0}) == "x<4"


def test_linear_show_steps_constant_and_variable_outcomes():
    same = kw.LinearEquation("1=1", variables=())
    assert "infinite" in same.show_steps().lower()
    different = kw.LinearEquation("1=2", variables=())
    assert "always false" in different.show_steps().lower()
    assert different.solution is None
    with pytest.raises(NotImplementedError):
        kw.LinearEquation("x+y=2").show_steps()

    assert "infinite" in kw.LinearEquation("x=x").show_steps().lower()
    assert "no solution" in kw.LinearEquation("x=x+1").show_steps().lower()
    assert "final step" in kw.LinearEquation("x-2=0").show_steps().lower()
    assert "final step" in kw.LinearEquation("x+2=0").show_steps().lower()
    equation = kw.LinearEquation("2x=4")
    assert equation.solve() == equation.solve() == 2


def test_linear_plot_solution_numeric_and_unsolved_paths(monkeypatch):
    calls = []
    plots = importlib.import_module("kiwicalc.plotting.plots")
    monkeypatch.setattr(plots, "plot_functions", lambda *args, **kwargs: calls.append(kwargs["title"]))
    monkeypatch.setattr(single.plt, "show", lambda: calls.append("show"))
    assert kw.LinearEquation("2=x").plot_solution(title="constant", show=True) == (2, 2)
    assert kw.LinearEquation("x=2").plot_solution(show=False) == (2, 2)
    assert kw.LinearEquation("2=3").plot_solution(show=False) is None
    assert calls == ["constant", "show", "x=2", "2=3"]


def test_random_linear_zero_retry_and_plain_mode(monkeypatch):
    values = iter([0, 2, -3])
    monkeypatch.setattr(single.random, "uniform", lambda *args: next(values))
    assert kw.random_linear() == "2x-3"
    values = iter([2, 3])
    monkeypatch.setattr(single.random, "uniform", lambda *args: next(values))
    assert kw.random_linear(get_solution=True) == ("2x+3", -1.5)


def test_random_polynomial_default_and_retry_branches(monkeypatch):
    monkeypatch.setattr(single.random, "randint", lambda *args: 2)
    values = iter([0, 1, 0, 0])
    monkeypatch.setattr(single.random, "uniform", lambda *args: next(values))
    assert isinstance(kw.random_polynomial(), str)

    values = iter([0, 2, 0, 3, 0])
    monkeypatch.setattr(single.random, "uniform", lambda *args: next(values))
    assert isinstance(kw.random_polynomial2(3, python_syntax=True), str)


@pytest.mark.parametrize(
    "get_solution,get_variable,expected_length",
    [(True, True, 3), (True, False, 2), (False, True, 2), (False, False, None)],
)
def test_random_linear_equation_fallback_return_shapes(monkeypatch, get_solution, get_variable, expected_length):
    monkeypatch.setattr(single.LinearEquation, "random_expression", lambda **kwargs: "3x")
    monkeypatch.setattr(single.LinearEquation, "solve", lambda self: 1 / 3)
    result = single.LinearEquation.random_equation(
        items_per_side=(1, 1), digits_after=0, variable="x",
        get_solution=get_solution, get_variable=get_variable,
    )
    if expected_length is None:
        assert isinstance(result, str)
    else:
        assert len(result) == expected_length


def test_linear_worksheet_branch_matrix(monkeypatch):
    pages = []
    worksheet = importlib.import_module("kiwicalc.pdf.worksheet")
    monkeypatch.setattr(worksheet, "create_pages", lambda **kwargs: pages.append(kwargs))
    monkeypatch.setattr(
        single.LinearEquation, "random_equation",
        lambda *args, **kwargs: ("x=1", 1, "x") if kwargs.get("get_solution") else "x=1",
    )
    single.LinearEquation.random_worksheets("out", num_of_pages=1, equations_per_page=1, get_solutions=True)
    single.LinearEquation.random_worksheets("out", num_of_pages=0, equations_per_page=0, get_solutions=True, titles=[])
    single.LinearEquation.random_worksheets("out", num_of_pages=1, equations_per_page=1, get_solutions=False)
    single.LinearEquation.random_worksheets("out", num_of_pages=0, equations_per_page=0, get_solutions=False, titles=[])
    assert len(pages) == 4


def test_manual_worksheet_success_and_failure(monkeypatch):
    answers = iter(["name", "title", "x=1", "stop"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    monkeypatch.setattr(single.LinearEquation, "adjusted_worksheet", lambda **kwargs: kwargs["equations"] == ["x=1"])
    assert single.LinearEquation.manual_worksheet()
    monkeypatch.setattr("builtins.input", lambda prompt: (_ for _ in ()).throw(RuntimeError()))
    with pytest.warns(UserWarning):
        assert not single.LinearEquation.manual_worksheet()


def test_quadratic_zero_variable_and_random_branches(monkeypatch):
    constant = kw.QuadraticEquation("2=0")
    with pytest.warns(UserWarning):
        assert constant.solve() is None
    assert constant.coefficients() == [2]
    with pytest.raises(ValueError):
        kw.QuadraticEquation("x^2+y^2=0").simplified_str()
    assert isinstance(kw.QuadraticEquation("x^2+y^2=0").coefficients(), dict)
    with pytest.raises(NotImplementedError):
        kw.QuadraticEquation.random(strict_syntax=False)

    integers = iter([0, 1])
    floats = iter([0, 1, -1])
    monkeypatch.setattr(single.random, "randint", lambda *args: next(integers))
    monkeypatch.setattr(single.random, "uniform", lambda *args: next(floats))
    expression, solutions = kw.QuadraticEquation.random(get_solutions=True)
    assert "x^2" in expression and solutions == (-1, 1)


def test_polynomial_equation_random_helper_and_worksheet_branches(monkeypatch):
    helper = single.PolyEquation._PolyEquation__random_monomial
    values = iter([0, -1, 2])
    monkeypatch.setattr(single.random, "randint", lambda *args: next(values))
    assert helper(power=2) == "0"
    assert helper(power=0) == "-"
    assert helper(power=2) == "2x^2"

    monkeypatch.setattr(single.random, "randint", lambda *args: 2)
    assert isinstance(single.PolyEquation.random_expression(of_order=None), str)

    calls = []
    monkeypatch.setattr(single, "random_polynomial", lambda *args, **kwargs: ("x^2-1", [-1, 1]) if kwargs.get("get_solutions") else "x^2-1")
    worksheet = importlib.import_module("kiwicalc.pdf.worksheet")
    monkeypatch.setattr(worksheet, "create_pdf", lambda **kwargs: calls.append(("pdf", kwargs)))
    monkeypatch.setattr(worksheet, "create_pages", lambda *args, **kwargs: calls.append(("pages", args, kwargs)))
    single.PolyEquation.random_worksheet("out", num_of_equations=1, get_solutions=True)
    single.PolyEquation.random_worksheet("out", num_of_equations=1, get_solutions=False)
    single.PolyEquation.random_worksheets("out", num_of_pages=1, equations_per_page=1, get_solutions=True)
    single.PolyEquation.random_worksheets("out", num_of_pages=0, equations_per_page=0, get_solutions=True, titles=[])
    single.PolyEquation.random_worksheets("out", num_of_pages=1, equations_per_page=1, get_solutions=False)
    single.PolyEquation.random_worksheets("out", num_of_pages=0, equations_per_page=0, get_solutions=False, titles=[])
    assert len(calls) == 6


def test_quadratic_cubic_and_quartic_worksheet_delegation(monkeypatch):
    calls = []
    monkeypatch.setattr(single.PolyEquation, "random_worksheet", lambda **kwargs: calls.append(("one", kwargs)))
    monkeypatch.setattr(single.PolyEquation, "random_worksheets", lambda **kwargs: calls.append(("many", kwargs)))
    for equation_type in (single.CubicEquation, single.QuarticEquation):
        assert isinstance(equation_type.random(get_solutions=False), str)
        equation_type.random_worksheet(num_of_equations=0)
        equation_type.random_worksheets(num_of_pages=0, get_solutions=True)
        equation_type.random_worksheets(num_of_pages=0, get_solutions=False)
        equation_type.random_worksheets(num_of_pages=0, get_solutions=True, titles=[])
    assert len(calls) == 8
