import importlib

import pytest

import kiwicalc as kw


roots_module = importlib.import_module("kiwicalc.expressions.roots")


@pytest.mark.parametrize(
    "results,expected",
    [
        ([kw.Fraction(1, kw.Var("x"))], None),
        ([(kw.Mono(2), 0)], (kw.Mono(2), "first")),
        ([(kw.Mono(2), 1)], None),
        ([kw.Mono(2), kw.Fraction(1, kw.Var("x"))], None),
        ([kw.Mono(2), (kw.Mono(3), 0)], (kw.Mono(3), "second")),
        ([kw.Mono(2), (kw.Mono(3), 1)], None),
    ],
)
def test_root_dependency_division_result_branches(monkeypatch, results, expected):
    iterator = iter(results)
    monkeypatch.setattr(kw.Mono, "__truediv__", lambda self, other: next(iterator))
    assert kw.Root.dependant_roots(kw.Root(kw.Mono("x")), kw.Root(kw.Mono("y"))) == expected


def test_root_dependency_direction_addition_and_basic_dispatch(monkeypatch):
    first, second = kw.Root(kw.Var("x")), kw.Root(kw.Var("y"))
    monkeypatch.setattr(kw.Root, "dependant_roots", staticmethod(lambda a, b: (kw.Mono(4), "first")))
    assert (first + second).variables
    monkeypatch.setattr(kw.Root, "dependant_roots", staticmethod(lambda a, b: (kw.Mono(4), "second")))
    assert (first + second).variables
    assert first - 0 == first
    with pytest.raises(TypeError):
        first.__imul__("x")


def test_root_power_division_equality_and_derivative_edges():
    x = kw.Var("x")
    symbolic_power = kw.Root(x, root_by=kw.Var("n")) ** kw.Var("p")
    assert isinstance(symbolic_power, kw.Root)
    assert kw.Root(x) ** 4 == kw.Poly("x^2")
    assert kw.Root(x) ** 2 == x
    assert kw.Root(x, root_by=4) / kw.Root(x, root_by=2) == kw.Root(x, root_by=-4)
    assert isinstance(kw.Root(x) / kw.Root(kw.Var("y")), kw.Fraction)
    assert not (kw.Root(x) == kw.Mono("x"))

    no_inside = kw.Root(x)
    no_inside._inside = None
    assert no_inside.derivative() == 0
    with pytest.warns(UserWarning):
        assert kw.Root(4, coefficient=-1).derivative() == 0
    assert kw.Root(x, coefficient=0).derivative() == 0
    assert kw.Root(x, root_by=0.5).derivative() == kw.Mono("2x")
    assert isinstance(kw.Root(x, root_by=3).derivative(), kw.Fraction)


def test_fraction_add_multiply_and_equality_branches():
    symbolic = kw.Fraction(kw.Var("x"), 2)
    assert isinstance(symbolic + 1, kw.ExpressionSum)
    assert kw.Fraction(1, 2) + kw.Mono(1) == kw.Mono(1.5)
    expression_sum = kw.ExpressionSum([kw.Var("x"), 1])
    assert isinstance(symbolic + expression_sum, kw.ExpressionSum)
    assert isinstance(symbolic + kw.Fraction(kw.Var("y"), 3), kw.ExpressionSum)
    assert isinstance(symbolic + kw.Var("y"), kw.ExpressionSum)

    reduced = kw.Fraction(kw.Var("x"), kw.Var("x")) * kw.Var("x")
    assert reduced == kw.Var("x")
    assert kw.Fraction(2, 1) * 1 == kw.Mono(2)
    assert kw.Fraction(kw.Var("x"), 2) == kw.Fraction(kw.Mono("2x"), 4)
    assert kw.Fraction(kw.Var("x"), 2).__eq__(kw.Var("y")) is None


def test_polyfraction_addition_and_subtraction_denominator_relations():
    assert kw.PolyFraction("x/2") + 0 == kw.PolyFraction("x/2")
    same = kw.PolyFraction("x/2") + kw.PolyFraction("1/2")
    assert same == kw.PolyFraction(kw.Poly("x+1"), kw.Poly(2))

    smaller = kw.PolyFraction(kw.Poly("x"), kw.Poly("x^2"))
    larger = kw.PolyFraction(kw.Poly(1), kw.Poly("x"))
    assert isinstance(smaller + larger, kw.PolyFraction)
    assert isinstance(larger + smaller, kw.PolyFraction)
    assert isinstance(smaller - larger, kw.PolyFraction)
    assert isinstance(larger - smaller, kw.PolyFraction)
    with pytest.raises(NotImplementedError):
        smaller.__iadd__(1)
    with pytest.raises(NotImplementedError):
        smaller.__isub__(1)
