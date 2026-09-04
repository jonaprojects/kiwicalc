import importlib
import json

import numpy as np
import pytest

import kiwicalc as kw
from kiwicalc.core.interfaces import IExpression


numeric_roots = importlib.import_module("kiwicalc.numeric.roots")


def test_expression_interface_convenience_methods(tmp_path):
    x = kw.Mono("x")
    assert abs(x) == kw.Abs(x)
    assert 2 ** kw.Mono(3) == 8
    assert isinstance(2 ** x, kw.Exponent)
    assert "x" in x.python_syntax()
    assert x.to_lambda()(3) == 3
    assert kw.Mono("x").reinman(0, 1, 100) == pytest.approx(0.5, abs=0.02)
    assert kw.Mono("x").trapz(0, 1, 100) == pytest.approx(0.5, abs=0.02)
    assert kw.Mono("x").simpson(0, 1, 100) == pytest.approx(0.5, abs=0.02)
    assert kw.Poly("x^2-4").secant(3, 1) == pytest.approx(2, abs=1e-4)
    assert kw.Poly("x^2-4").bisection(0, 3) == pytest.approx(2, abs=1e-4)
    assert json.loads(x.to_json())["type"] == "Mono"
    output = tmp_path / "mono.json"
    x.export_json(output)
    assert json.loads(output.read_text())["coefficient"] == 1
    assert x.to_Function()(4) == 4


def test_expression_interface_plot_and_scatter_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr("kiwicalc.plotting.plots.plot_function", lambda *args, **kwargs: calls.append("plot2d"))
    monkeypatch.setattr("kiwicalc.plotting.plots.plot_function_3d", lambda *args, **kwargs: calls.append("plot3d"))
    monkeypatch.setattr("kiwicalc.plotting.plots.scatter_function", lambda *args, **kwargs: calls.append("scatter2d"))
    monkeypatch.setattr("kiwicalc.plotting.plots.scatter_function_3d", lambda *args, **kwargs: calls.append("scatter3d"))

    IExpression.plot(kw.Abs(kw.Var("x")), show=False)
    IExpression.plot(kw.Abs(kw.Poly("x+y")), show=False)
    with pytest.raises(ValueError):
        IExpression.plot(kw.Abs(2), show=False)
    IExpression.scatter(kw.Abs(kw.Var("x")), show=False)
    IExpression.scatter(kw.Abs(kw.Poly("x+y")), show=False)
    with pytest.raises(ValueError):
        IExpression.scatter(kw.Abs(2), show=False)
    with pytest.raises(ValueError):
        IExpression.scatter(kw.Abs(kw.Poly("x+y+z")), show=False)
    assert calls == ["plot2d", "plot3d", "scatter2d", "scatter3d"]


def test_factor_and_possible_solution_paths():
    assert numeric_roots.get_factors(0) == {}
    assert numeric_roots.get_factors(6) == {-6, -3, -2, -1, 1, 2, 3, 6}
    assert numeric_roots.extract_possible_solutions(2, 4) == {-4, -2, -1, -0.5, 0.5, 1, 2, 4}
    assert numeric_roots.extract_possible_solutions(2, 0) == {}


def test_newton_halley_secant_and_inverse_paths():
    assert numeric_roots.newton_raphson(lambda x: x - 1, lambda x: 0 if x == 0 else 1, initial_value=0) == pytest.approx(1)
    with pytest.warns(UserWarning):
        numeric_roots.newton_raphson(lambda x: 1, lambda x: 1, nmax=1)
    assert numeric_roots.halleys_method(lambda x: x**2 - 4, lambda x: 2 * x, lambda x: 2, 3) == pytest.approx(2, abs=1e-4)
    assert numeric_roots.secant_method(lambda x: x**2 - 4, 3, 1) == pytest.approx(2, abs=1e-4)
    assert numeric_roots.inverse_interpolation(lambda x: x**2 - 2, 1, 1.5, 2) == pytest.approx(2**0.5, abs=1e-4)
    with pytest.warns(UserWarning):
        numeric_roots.inverse_interpolation(lambda x: x**2 + 1, 0, 1, 2, nmax=1)


def test_laguerre_bounds_and_multi_root_methods():
    assert numeric_roots.laguerre_method(lambda x: x**2 - 4, lambda x: 2 * x, lambda x: 2, 3, 2) == pytest.approx(2, abs=1e-4)
    with pytest.warns(UserWarning):
        numeric_roots.laguerre_method(lambda x: x**2 + 1, lambda x: 2 * x, lambda x: 2, 2, 2, nmax=1)
    upper, lower = numeric_roots.get_bounds(2, [1, 0, -4])
    assert upper > lower > 0
    roots = numeric_roots.durand_kerner(lambda x: x**2 - 1, [1, 0, -1])
    assert roots == {complex(-1, 0), complex(1, 0)}
    roots_non_monic = numeric_roots.durand_kerner(lambda x: 2 * x**2 - 2, [2, 0, -2])
    assert roots_non_monic == {complex(-1, 0), complex(1, 0)}
    assert numeric_roots.durand_kerner2([2, 0, -2]) == {complex(-1, 0), complex(1, 0)}
    assert numeric_roots.negligible_complex(complex(1e-8, -1e-8), 1e-6)
    assert not numeric_roots.negligible_complex(complex(1, 0), 1e-6)


def test_ostrowski_chebyshev_aberth_and_steffensen_paths(monkeypatch):
    assert numeric_roots.ostrowski_method(lambda x: x**2 - 4, lambda x: 0 if x == 0 else 2 * x, 0) == pytest.approx(2, abs=1e-4)
    assert numeric_roots.chebychevs_method(lambda x: x**2 - 4, lambda x: 2 * x, lambda x: 2, 3) == pytest.approx(2, abs=1e-4)
    with pytest.warns(UserWarning):
        numeric_roots.chebychevs_method(lambda x: 1, lambda x: 1, lambda x: 0, 0, nmax=1)
    aberth = numeric_roots.aberth_method(lambda x: x**2 - 1, lambda x: 2 * x, [1, 0, -1])
    assert sorted(aberth, key=lambda z: z.real) == pytest.approx([-1, 1], abs=1e-6)
    monkeypatch.setattr(numeric_roots, "__aberth_approximations", lambda coefficients: (_ for _ in ()).throw(ValueError("bad coefficients")))
    with pytest.raises(ValueError):
        numeric_roots.aberth_method(lambda x: x, lambda x: 1, [0, 0])
    assert numeric_roots.steffensen_method(lambda x: x**2 - 4, 3) == pytest.approx(2, abs=1e-4)
    with pytest.warns(UserWarning):
        numeric_roots.steffensen_method(lambda x: 1, 0)


def test_bisection_validation_and_terminal_paths():
    assert numeric_roots.bisection_method(lambda x: x - 2, 3, 0) == pytest.approx(2, abs=1e-4)
    with pytest.raises(ValueError):
        numeric_roots.bisection_method(lambda x: x, 1, 1)
    with pytest.raises(ValueError):
        numeric_roots.bisection_method(lambda x: x**2 + 1, -1, 1)
    assert numeric_roots.bisection_method(lambda x: x - 2, 0, 3, nmax=0) is None
    assert sorted(numeric_roots.bairstow_method([1, 0, -1])) == pytest.approx([-1, 1])
