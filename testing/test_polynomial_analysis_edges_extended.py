import json

import pytest

import kiwicalc as kw


def test_fastpoly_constructor_forms_and_validation():
    assert kw.FastPoly(5).try_evaluate() == 5
    assert kw.FastPoly([1, -3, 2]).variables_dict == {"x": [1, -3], "free": 2}
    assert kw.FastPoly({"x": [2, 1], "free": -4}).to_lambda()(3) == 17
    with pytest.raises(KeyError):
        kw.FastPoly({"x": [1, 2]})
    with pytest.raises(ValueError):
        kw.FastPoly([])
    with pytest.raises(ValueError):
        kw.FastPoly([1, 2], variables=("x", "y"))
    with pytest.raises(TypeError):
        kw.FastPoly(object())


def test_fastpoly_assignment_updates_metadata():
    polynomial = kw.FastPoly("2x^2+3y-4")
    polynomial.assign(x=2)
    assert polynomial.variables == ["y"]
    assert polynomial.to_lambda()(5) == 19
    polynomial.assign(y=5)
    assert polynomial.variables == []
    assert polynomial.try_evaluate() == 19


def test_fastpoly_constant_calculus_and_roots():
    constant = kw.FastPoly(4)
    assert constant.degree == 0
    assert constant.is_free_number
    assert constant.derivative() == kw.FastPoly(0)
    assert constant.integral(c=2) == kw.FastPoly("4x+2")
    assert kw.FastPoly(0).roots() == "Infinite"
    assert kw.FastPoly(4).roots() is None
    assert kw.FastPoly("2x+1").extremums() is None


def test_fastpoly_numerical_roots_and_extremum():
    polynomial = kw.FastPoly("x^2-4")
    assert polynomial.newton(3) == pytest.approx(2)
    assert polynomial.halley(3) == pytest.approx(2, abs=1e-5)
    extremums = polynomial.extremums()
    assert extremums.points[0] == kw.Point2D(0, -4)


def test_fastpoly_arithmetic_and_type_errors():
    first = kw.FastPoly("x^2+2x+1")
    second = kw.FastPoly("x^2-1")
    assert first + second == kw.FastPoly("2x^2+2x")
    assert first - second == kw.FastPoly("2x+2")
    assert first + 3 == kw.FastPoly("x^2+2x+4")
    assert first - 3 == kw.FastPoly("x^2+2x-2")
    with pytest.raises(TypeError):
        first + object()
    with pytest.raises(TypeError):
        first - object()
    with pytest.raises(TypeError):
        first == object()


def test_fastpoly_serialization_rejects_wrong_payloads():
    with pytest.raises(ValueError):
        kw.FastPoly.from_dict({"type": "Poly", "data": {"free": 1}})
    with pytest.raises(ValueError):
        kw.FastPoly.from_json(json.dumps({"type": "Poly", "data": {"free": 1}}))


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("7", 0),
        ("2x+1", 1),
        ("x^2-5x+6", 1),
        ("x^3-3x+1", 81),
        ("x^4-5x^2+4", 5184),
    ],
)
def test_polynomial_discriminant_degrees(source, expected):
    assert kw.Poly(source).discriminant() == expected


def test_polynomial_discriminant_and_coefficients_validation():
    with pytest.raises(ValueError):
        kw.Poly("x^5+1").discriminant()
    with pytest.raises(ValueError):
        kw.Poly("x+y").coefficients()
    assert kw.Poly([]).coefficients() == [0]
    assert kw.Poly("7").coefficients() == [7]


def test_polynomial_extrema_axes_and_monotonic_ranges():
    polynomial = kw.Poly("x^2-4x+3")
    assert polynomial.extremums_axes() == pytest.approx([2])
    assert polynomial.extremums().points[0] == kw.Point2D(2, -1)
    up, down = polynomial.up_and_down()
    assert up.try_evaluate() is None
    assert down.try_evaluate() is None

    increasing, decreasing = kw.Poly("2x+1").up_and_down()
    assert increasing is not None
    assert decreasing is None


def test_polynomial_data_for_constant_and_multivariate():
    constant = kw.Poly("4").data()
    assert constant["roots"] == []
    assert constant["y_intersection"] == 4
    zero = kw.Poly("0").data()
    assert zero["roots"] == pytest.approx(float("inf"))
    multivariate = kw.Poly("x+y").data()
    assert multivariate["plotDimensions"] == 3
    assert multivariate["variables"] == {"x", "y"}


def test_polynomial_reports_and_function_conversion(capsys):
    polynomial = kw.Poly("x^2-4")
    report = polynomial.get_report(colored=False)
    assert "x^2-4" in report
    assert polynomial.to_Function()(3) == 5
    polynomial.print_report()
    assert "coefficients" in capsys.readouterr().out


def test_polynomial_numerical_methods():
    polynomial = kw.Poly("x^2-4")
    assert polynomial.newton(3) == pytest.approx(2)
    assert polynomial.halleys(3) == pytest.approx(2, abs=1e-5)
    assert polynomial.ostrowski(3) == pytest.approx(2)
    assert polynomial.laguerres(3) == pytest.approx(2)
    roots = polynomial.durand_kerner()
    assert sorted(round(complex(root).real, 5) for root in roots) == [-2, 2]


def test_polynomial_integral_and_partial_derivative_errors():
    with pytest.raises(ValueError):
        kw.Poly("xy").derivative()
    with pytest.raises(ValueError):
        kw.Poly("xy").integral()
    assert kw.Poly("x^2+y^2").partial_derivative(("x",)) == kw.Poly("2x")
    assert kw.Poly("x").integral(add_c=True).variables == {"x", "c"}
