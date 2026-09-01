import json

import pytest

import kiwicalc as kw
from kiwicalc.expressions import poly as poly_module


def test_fastpoly_constructor_and_metadata_branches():
    assert kw.FastPoly(3).is_free_number
    assert kw.FastPoly([3]).variables == []
    assert kw.FastPoly([1, 2, 3]).degree == 2
    assert kw.FastPoly({"x": [1, 2], "y": [3], "free": 4}).degree == {"x": 2, "y": 1}
    assert kw.FastPoly("x^2+2x+1").num_of_variables == 1
    with pytest.raises(KeyError):
        kw.FastPoly({"x": [1]})
    with pytest.raises(ValueError):
        kw.FastPoly([])
    with pytest.raises(ValueError):
        kw.FastPoly([1, 2], variables=["x", "y"])
    with pytest.raises(TypeError):
        kw.FastPoly(object())


def test_fastpoly_derivative_integral_and_extremum_branches():
    assert kw.FastPoly(4).derivative() == kw.FastPoly(0)
    assert kw.FastPoly([2, 3], variables=["t"]).derivative() == kw.FastPoly(2)
    assert kw.FastPoly([3, 2, 1]).derivative() == kw.FastPoly([6, 2])
    with pytest.raises(ValueError):
        kw.FastPoly({"x": [1], "y": [1], "free": 0}).derivative()
    assert kw.FastPoly(3).partial_derivative(["x"]) is None
    assert kw.FastPoly(3).extremums() is None
    assert kw.FastPoly([2, 1]).extremums() is None
    extrema = kw.FastPoly([1, 0, 0]).extremums()
    assert len(extrema.points) == 1
    assert kw.FastPoly({"x": [1], "y": [1], "free": 0}).extremums() is None
    assert kw.FastPoly(3).integral(c=2, variable="t") == kw.FastPoly({"t": [3], "free": 2})
    assert kw.FastPoly([2, 1]).integral(c=4) == kw.FastPoly([1, 1, 4])
    with pytest.raises(ValueError):
        kw.FastPoly({"x": [1], "y": [1], "free": 0}).integral()


def test_fastpoly_root_methods_and_arithmetic_dispatch():
    polynomial = kw.FastPoly([1, 0, -4])
    assert polynomial.newton(initial=3) == pytest.approx(2)
    assert polynomial.halley(initial=3) == pytest.approx(2, abs=3e-6)
    assert kw.FastPoly(0).roots() == "Infinite"
    assert kw.FastPoly(2).roots() is None
    assert set(round(root.real) for root in polynomial.roots()) == {-2, 2}
    with pytest.raises(ValueError):
        kw.FastPoly({"x": [1], "y": [1], "free": 0}).roots()

    left = kw.FastPoly({"x": [1], "free": 1})
    assert left + 0 == left
    assert left + 2 == kw.FastPoly({"x": [1], "free": 3})
    assert left - 2 == kw.FastPoly({"x": [1], "free": -1})
    assert left + kw.Mono(2) == kw.FastPoly({"x": [1], "free": 3})
    assert left - kw.Mono(2) == kw.FastPoly({"x": [1], "free": -1})
    assert left + kw.FastPoly({"y": [2], "free": 3}) == kw.FastPoly({"x": [1], "y": [2], "free": 4})
    assert left - kw.FastPoly({"y": [2], "free": 3}) == kw.FastPoly({"x": [1], "y": [-2], "free": -2})
    assert isinstance(left + kw.Sin(kw.Var("x")), kw.ExpressionSum)
    assert isinstance(left - kw.Sin(kw.Var("x")), kw.ExpressionSum)
    with pytest.raises(TypeError):
        left += object()
    with pytest.raises(TypeError):
        left -= object()
    assert left.__imul__(2) is None
    assert left.__itruediv__(2) is None
    assert left.__ipow__(2) is None


def test_fastpoly_assignment_equality_serialization_and_render(tmp_path):
    expression = kw.FastPoly({"x": [2, 3], "y": [4], "free": 1})
    expression.assign(z=10, x=2)
    assert expression == kw.FastPoly({"y": [4], "free": 15})
    with pytest.warns(UserWarning):
        expression.simplify()
    assert expression.try_evaluate() is None
    expression.assign(y=1)
    assert expression.try_evaluate() == 19
    assert expression != None  # noqa: E711
    assert expression == kw.Poly(19)
    assert kw.FastPoly("x") != kw.Poly("x")
    with pytest.raises(TypeError):
        expression == object()
    assert -kw.FastPoly({"x": [1, -2], "free": 3}) == kw.FastPoly({"x": [-1, 2], "free": -3})
    restored = kw.FastPoly.from_dict(expression.to_dict())
    assert restored == expression
    with pytest.raises(ValueError):
        kw.FastPoly.from_dict({"type": "Poly", "data": {"free": 1}})
    payload = json.dumps(expression.to_dict())
    assert kw.FastPoly.from_json(payload) == expression
    with pytest.raises(ValueError):
        kw.FastPoly.from_json('{"type":"Poly","data":{"free":1}}')
    path = tmp_path / "fastpoly.json"
    path.write_text(payload)
    assert kw.FastPoly.import_json(path) == expression
    assert kw.FastPoly("x^2+1").python_syntax()
    assert "x" in str(kw.FastPoly("x+1"))


def test_fastpoly_plot_dimension_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr("kiwicalc.plotting.plots.plot_function", lambda *args, **kwargs: calls.append("2d"))
    monkeypatch.setattr("kiwicalc.plotting.plots.plot_function_3d", lambda *args, **kwargs: calls.append("3d"))
    with pytest.raises(ValueError):
        kw.FastPoly(2).plot(show=False)
    kw.FastPoly("x+1").plot(show=False)
    kw.FastPoly({"x": [1], "y": [1], "free": 0}).plot(show=False)
    with pytest.raises(ValueError):
        kw.FastPoly({"x": [1], "y": [1], "z": [1], "free": 0}).plot(show=False)
    assert calls == ["2d", "3d"]


def test_poly_constructor_iterable_and_arithmetic_edge_paths():
    with pytest.warns(UserWarning):
        expression = kw.Poly([kw.Mono("x"), "2x", kw.Poly("y"), 3, object()])
    assert expression == kw.Poly("3x+y+3")
    with pytest.raises(TypeError):
        kw.Poly(object())
    expression.expressions = [kw.Mono("x"), kw.Mono(1)]
    assert expression == kw.Poly("x+1")
    assert expression + "x-1" == kw.Poly("2x")
    assert expression - "x+1" == 0
    assert isinstance(expression + kw.Sin(kw.Var("x")), kw.ExpressionSum)
    assert isinstance(expression - kw.Sin(kw.Var("x")), kw.ExpressionSum)
    with pytest.raises(TypeError):
        expression += object()
    with pytest.raises(TypeError):
        expression -= object()


def test_poly_multiplication_collection_and_division_paths():
    x = kw.Var("x")
    assert kw.Poly("x+1") * 0 == 0
    assert kw.Poly("x+1") * 2 == kw.Poly("2x+2")
    assert kw.Poly("x+1") * kw.Mono("x") == kw.Poly("x^2+x")
    assert kw.Poly("x+1") * kw.Poly("x-1") == kw.Poly("x^2-1")
    trig_product = kw.Poly("x+1") * kw.Sin(x)
    assert trig_product.when(x=0.5).try_evaluate() == pytest.approx(1.5 * __import__("math").sin(0.5))
    assert kw.Poly("x+1") * [1, 2] == [kw.Poly("x+1"), kw.Poly("2x+2")]
    with pytest.raises(NotImplementedError):
        kw.Poly("x") * kw.Vector([1, 2])

    with pytest.raises(ZeroDivisionError):
        kw.Poly("x") / 0
    assert kw.Poly("4x+2") / 2 == kw.Poly("2x+1")
    assert kw.Poly(4) / 2 == 2
    assert kw.Poly("x^2-1").divide_by_poly(kw.Poly("x-1")) == kw.Poly("x+1")
    quotient, remainder = kw.Poly("x^2+1").divide_by_poly(kw.Poly("x+1"), get_remainder=True)
    assert quotient == kw.Poly("x-1") and remainder == 2
    with pytest.raises(ZeroDivisionError):
        kw.Poly("x").divide_by_poly(kw.Mono(0))
    assert isinstance(kw.Poly("x") / kw.Sin(x), kw.Fraction)
    with pytest.raises(TypeError):
        kw.Poly("x") / object()


def test_poly_power_evaluation_iteration_and_calculus_paths():
    x = kw.Var("x")
    assert kw.Poly("x+1") ** 0 == 1
    assert kw.Poly("x+1") ** 1 == kw.Poly("x+1")
    assert kw.Poly(3) ** 2 == 9
    assert kw.Poly("x+1") ** 3 == kw.Poly("x^3+3x^2+3x+1")
    assert kw.Poly("x+y+1") ** 2 == kw.Poly("x^2+2xy+2x+y^2+2y+1")
    assert kw.Poly("x+1") ** kw.Mono(2) == kw.Poly("x^2+2x+1")
    assert kw.Poly("x+1") ** kw.Poly(2) == kw.Poly("x^2+2x+1")
    with pytest.raises(ValueError):
        kw.Poly("x") ** kw.Mono("y")
    with pytest.raises(ValueError):
        kw.Poly("x") ** kw.Poly("y+1")
    assert kw.Poly([]).try_evaluate() == 0
    assert kw.Poly([1, 2]).try_evaluate() == 3
    assert list(kw.Poly("x+1")) == kw.Poly("x+1").expressions
    assert kw.Poly("x^2+y").partial_derivative(["x"]) == kw.Poly("2x")
    assert kw.Poly("x^2+2x").integral() == kw.Poly("0.33333x^3+x^2")
    assert kw.Poly("x").integral(add_c=True) == kw.Poly("0.5x^2+c")
    with pytest.raises(ValueError):
        kw.Poly("xy").integral()


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("3", 0),
        ("2x+1", 1),
        ("x^2+2x+1", 0),
        ("x^3-1", -27),
        ("x^4-1", -256),
    ],
)
def test_poly_discriminant_degree_paths(expression, expected):
    assert kw.Poly(expression).discriminant() == expected


def test_poly_coefficients_metadata_report_and_gcd_paths(capsys):
    assert kw.Poly([]).coefficients() == [0]
    assert kw.Poly(3).coefficients() == [3]
    with pytest.raises(ValueError):
        kw.Poly("x+y").coefficients()
    with pytest.raises(ValueError):
        kw.Poly("x^5+1").discriminant()
    assert kw.Poly("6x^2+9x").gcd() == kw.Mono("3x")
    assert kw.Poly("6x^2+9").gcd() == 3
    assert kw.Poly("6x^2+9x").divide_by_gcd() == kw.Poly("2x+3")
    assert kw.Poly("x+1").contains_variable("x")
    assert not kw.Poly("x+1").contains_variable("y")
    assert "x" in kw.Poly("x+1")
    assert kw.Poly("x") in kw.Poly("x+1")
    with pytest.raises(TypeError):
        object() in kw.Poly("x")
    assert kw.Poly("x+1").sorted() == kw.Poly("x+1")
    kw.Poly("x^2-1").print_report()
    assert "string" in capsys.readouterr().out


def test_poly_data_and_report_format_paths(monkeypatch):
    constant = kw.Poly(3).data()
    assert constant["coefficients"] == [3] and constant["roots"] == []
    zero = kw.Poly(0).data()
    assert zero["roots"] == float("inf")
    multivariate = kw.Poly("x+y").data()
    assert multivariate["plotDimensions"] == 3
    quadratic = kw.Poly("x^2-1")
    data = quadratic.data()
    assert data["coefficients"] == [1, 0, -1]
    assert "string" in quadratic.get_report(colored=True)
    assert quadratic.get_report(colored=False)
    formatted = quadratic._format_report(data)
    assert any("coefficients" in line for line in formatted)


def test_poly_serialization_import_and_plot_dispatch(tmp_path, monkeypatch):
    expression = kw.Poly("x^2+1")
    assert kw.Poly.from_dict(expression.to_dict()) == expression
    assert kw.Poly.from_json(json.dumps(expression.to_dict())) == expression
    assert isinstance(kw.Poly.from_json('{"type":"Mono","data":[]}'), ValueError)
    path = tmp_path / "poly.json"
    path.write_text(json.dumps(expression.to_dict()))
    assert kw.Poly.import_json(path) == expression
    assert "**" in expression.python_syntax()
    assert expression.to_Function()(2) == 5

    calls = []
    monkeypatch.setattr("kiwicalc.plotting.plots.plot_function", lambda *args, **kwargs: calls.append("2d"))
    monkeypatch.setattr("kiwicalc.plotting.plots.plot_function_3d", lambda *args, **kwargs: calls.append("3d"))
    with pytest.raises(ValueError):
        kw.Poly(2).plot(show=False)
    kw.Poly("x+1").plot(show=False)
    kw.Poly("x+y").plot(show=False)
    with pytest.raises(ValueError):
        kw.Poly("x+y+z").plot(show=False)
    assert calls == ["2d", "3d"]


def test_synthetic_division_paths():
    assert poly_module.synthetic_division([1, 0, -1], 1) == ([1, 1], 0)
    assert poly_module.synthetic_division([1, 0, 1], 1) == ([1, 1, 2], 2)
