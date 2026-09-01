import importlib

import pytest

import kiwicalc as kw


poly_module = importlib.import_module("kiwicalc.expressions.poly")


def test_fastpoly_subtraction_equality_and_plot_edges(monkeypatch):
    first = kw.FastPoly({"x": [1], "free": 2})
    first -= kw.FastPoly({"y": [3], "free": 4})
    assert first.variables_dict == {"x": [1], "y": [-3], "free": -2}
    assert kw.FastPoly(2) == kw.Mono(2)
    assert kw.FastPoly("x") != kw.Mono(2)

    plots = importlib.import_module("kiwicalc.plotting.plots")
    calls = []
    monkeypatch.setattr(plots, "plot_function", lambda *args, **kwargs: calls.append(kwargs["title"]))
    kw.FastPoly("x").plot(text="custom", show=False)
    assert calls == ["custom"]


def test_poly_copy_and_addition_zero_normalization():
    original = kw.Poly("x+1")
    copied = kw.Poly(original)
    assert copied == original and copied is not original
    assert original + 0 == original
    assert kw.Poly(1) + -1 == 0
    assert kw.Poly("x") + "-x" == 0
    assert kw.Poly("x") + kw.Poly("-x") == 0
    assert kw.Poly("x") + kw.Mono("-x") == 0


def test_poly_reverse_and_subtraction_dispatch_edges():
    polynomial = kw.Poly("x+1")
    assert 3 - polynomial == kw.Poly("2-x")
    assert "sin" in str(polynomial.__rsub__(kw.Sin(kw.Var("x"))))
    with pytest.raises(TypeError):
        polynomial.__rsub__(object())
    assert kw.Poly(1) - 1 == 0
    assert kw.Poly("x") - "x" == 0
    assert kw.Poly("x") - kw.Poly("x") == 0


def test_poly_multiplication_collection_and_zero_edges():
    manually_zero = kw.Poly(1)
    manually_zero.expressions = [kw.Mono(0), kw.Mono(0)]
    assert manually_zero.__imul__(2) == 0
    assert kw.Poly("x+1").__imul__([2, 3]) == [kw.Poly("2x+2"), kw.Poly("3x+3")]
    with pytest.raises(NotImplementedError):
        kw.Poly("x").__imul__(kw.Vector([1, 2]))


def test_poly_division_remainder_and_timeout_branches():
    quotient, remainder = kw.Poly("x^2-1").__truediv__(kw.Mono("x"), get_remainder=True)
    assert remainder == 0
    assert quotient.when(x=2).try_evaluate() == pytest.approx(1.5)
    assert kw.Poly("x^2-1").__truediv__(2, get_remainder=True)[1] == 0
    assert kw.Poly(6).__truediv__(2, get_remainder=True) == (kw.Mono(3), 0)
    assert kw.Poly(6).__truediv__(kw.Mono(2), get_remainder=True) == (kw.Mono(3), 0)
    assert kw.Poly("2x").__truediv__(kw.Mono(2), get_remainder=True)[1] == 0
    with pytest.raises(ZeroDivisionError):
        kw.Poly("x").__truediv__(kw.Mono(0))

    exact, remainder = kw.Poly("x^2-1").divide_by_poly(kw.Poly("x-1"), get_remainder=True)
    assert exact == kw.Poly("x+1") and remainder == 0
    quotient, remainder = kw.Poly("x^2+1").divide_by_poly(kw.Poly("x+1"), get_remainder=True)
    assert remainder == 2
    with pytest.warns(UserWarning, match="timed out"):
        assert isinstance(kw.Poly("x^2+1").divide_by_poly(kw.Poly("x+1"), nmax=0), kw.PolyFraction)


def test_poly_power_and_reverse_power_edges():
    assert kw.Poly("x+1") ** 2.0 == kw.Poly("x^2+2x+1")
    assert kw.Poly("x+1") ** "2" == kw.Poly("x^2+2x+1")
    assert kw.Poly("x+1") ** kw.Poly(2) == kw.Poly("x^2+2x+1")
    with pytest.raises(ValueError):
        kw.Poly("x+1") ** kw.Poly("x")
    assert kw.Poly(2).__rpow__(3, None) == kw.Poly(9)
    assert isinstance(kw.Poly("x").__rpow__(2, None), kw.Exponent)


def test_poly_evaluation_equality_and_coefficients_edges():
    constants = kw.Poly(0)
    constants.expressions = [kw.Mono(1), kw.Mono(2)]
    assert constants.try_evaluate() == 3

    empty = kw.Poly(0)
    empty.expressions = []
    assert empty.coefficients() is None
    constants.expressions = [kw.Mono(1), kw.Mono(2)]
    assert constants.coefficients() == [3]

    assert kw.Poly("x") != kw.Poly("x+y")
    first = kw.Poly("x+y")
    second = kw.Poly("x-y")
    assert first != second
    assert kw.Poly("x+y") == kw.Poly("y+x")


def test_poly_calculus_extrema_and_monotonicity_edges():
    assert kw.Poly(4).extremums() is None
    assert kw.Poly("2x+1").extremums() is None
    assert kw.Poly(4).extremums_axes() is None
    assert kw.Poly("2x+1").extremums_axes() is None
    assert kw.Poly(3).up_and_down() == (None, None)
    assert kw.Poly("2x+1").up_and_down()[1] is None
    assert kw.Poly("-2x+1").up_and_down()[0] is None
    up, down = kw.Poly("x^2").up_and_down()
    assert up is not None and down is not None


def test_poly_report_format_serialization_contains_and_gcd_edges():
    polynomial = kw.Poly("x^2+1")
    data = polynomial.data()
    lines = polynomial._format_report(data)
    assert any("roots" in line.lower() for line in lines)
    assert polynomial._format_report({"string": "x+y", "variables": ["x", "y"]}) == [
        "Function: x+y", "variables: x, y"
    ]

    empty = kw.Poly(0)
    empty.expressions = []
    assert empty.to_dict() == {"type": "Poly", "data": None}
    assert kw.Poly("x") in kw.Poly("x+y")
    assert kw.Poly("x+y+1") not in kw.Poly("x+y")
    assert kw.Poly("6x^2y+9xy^2").gcd() == kw.Mono("3xy")
    assert kw.Poly("6x+9").gcd() == kw.Mono(3)


def test_synthetic_division_nonzero_remainder():
    assert kw.synthetic_division([1, 0, 1], 1) == ([1, 1, 2], 2)
