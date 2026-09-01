import pytest

import kiwicalc as kw


def test_fraction_constructor_defaults_and_validation():
    assert kw.Fraction(3).denominator == 1
    assert kw.Fraction(3).numerator == 3
    with pytest.raises(TypeError):
        kw.Fraction(object(), 2)
    with pytest.raises(TypeError):
        kw.Fraction(1, object())


def test_fraction_numeric_arithmetic_collapses_to_values():
    fraction = kw.Fraction(1, 2)
    assert fraction + 2 == 2.5
    assert fraction - 2 == -1.5
    assert fraction * 4 == 2
    assert 4 * fraction == 2
    assert fraction / 2 == 0.25
    assert fraction**2 == 0.25
    assert -fraction == -0.5


def test_fraction_symbolic_addition_and_subtraction():
    x = kw.Var("x")
    same = kw.Fraction(x, 2) + kw.Fraction(x + 1, 2)
    assert same.when(x=3).try_evaluate() == pytest.approx(3.5)
    different = kw.Fraction(x, 2) + kw.Fraction(x, 3)
    assert isinstance(different, kw.ExpressionSum)
    assert different.when(x=6).try_evaluate() == 5
    subtracted = kw.Fraction(x, 2) - kw.Fraction(x, 3)
    assert subtracted.when(x=6).try_evaluate() == 1


def test_fraction_symbolic_multiplication_division_and_power():
    x, y = kw.Var("x"), kw.Var("y")
    product = kw.Fraction(x, 2) * kw.Fraction(y, 3)
    assert product.when(x=2, y=3).try_evaluate() == 1
    quotient = kw.Fraction(x, 2) / kw.Fraction(y, 3)
    assert quotient.when(x=2, y=3).try_evaluate() == 1
    squared = kw.Fraction(x, 2) ** 2
    assert squared.when(x=4).try_evaluate() == 4
    assert isinstance(2 ** kw.Fraction(x, 2), kw.Exponent)


def test_fraction_equality_and_type_errors():
    x = kw.Var("x")
    assert kw.Fraction(1, 2) == kw.Fraction(2, 4)
    assert kw.Fraction(x, 2) == kw.Fraction(x, 2)
    assert kw.Fraction(x, 2) != None
    assert kw.Fraction(0, x) == 0
    with pytest.raises(ValueError):
        kw.Fraction(1, 0) == 1
    with pytest.raises(TypeError):
        kw.Fraction(x, 2) + object()


def test_polyfraction_constructor_forms_and_metadata():
    assert kw.PolyFraction(3).try_evaluate() == 3
    original = kw.PolyFraction("x+1/x-1")
    copied = kw.PolyFraction(original)
    assert copied == original and copied is not original
    assert repr(original) == "PolyFraction(x+1,x-1)"
    with pytest.raises(TypeError):
        kw.PolyFraction(object())
    with pytest.raises(TypeError):
        kw.PolyFraction(object(), 2)
    with pytest.raises(TypeError):
        kw.PolyFraction(1, object())


def test_polyfraction_equal_denominator_arithmetic():
    first = kw.PolyFraction("x/x+1")
    second = kw.PolyFraction("1/x+1")
    assert first + second == 1
    assert first - second == kw.PolyFraction("x-1/x+1")
    product = first * second
    product.assign(x=2)
    assert product.try_evaluate() == pytest.approx(2 / 9)
    assert 2 * first == kw.PolyFraction("2x/x+1")


def test_polyfraction_reciprocal_and_reverse_division():
    fraction = kw.PolyFraction("x+1/x-1")
    assert fraction.reciprocal() == kw.PolyFraction("x-1/x+1")
    assert 2 / fraction == kw.PolyFraction("2x-2/x+1")
    copied = fraction.__copy__()
    assert copied == fraction and copied is not fraction


def test_polyfraction_asymptote_branches():
    assert kw.PolyFraction("x^3+1/x^2+1").horizontal_asymptote() == ()
    assert kw.PolyFraction("1/2").horizontal_asymptote() == ()
    assert kw.PolyFraction("x/x^2+1").horizontal_asymptote() == 0
    assert kw.PolyFraction("2x+1/3x-1").horizontal_asymptote() == (1,)


def test_polyfraction_unsupported_arithmetic_raises():
    fraction = kw.PolyFraction("x/x+1")
    with pytest.raises(TypeError):
        fraction * object()
