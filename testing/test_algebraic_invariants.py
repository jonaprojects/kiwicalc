import math
import warnings

import pytest

import kiwicalc as kw


def test_factoring_solver_keeps_quadratic_fallback_roots_without_recovery_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solutions = kw.solve_poly_by_factoring([1, -6, 11, -6])

    assert sorted(solutions) == pytest.approx([1, 2, 3])
    assert not any("solutions might be missing" in str(item.message) for item in caught)


@pytest.mark.parametrize(
    ("dividend_text", "divisor_text"),
    [
        ("x^3-1", "x-1"),
        ("2x^3+3x^2-5x+7", "2x-1"),
        ("x^4-5x^2+4", "x^2-1"),
        ("3x^5-2x^3+x-9", "x^2+1"),
    ],
)
def test_polynomial_division_reconstructs_dividend_and_preserves_operands(
    dividend_text, divisor_text
):
    dividend = kw.Poly(dividend_text)
    divisor = kw.Poly(divisor_text)
    dividend_before = dividend.__copy__()
    divisor_before = divisor.__copy__()

    quotient, remainder = dividend.divide_by_poly(divisor, get_remainder=True)

    assert quotient * divisor + remainder == dividend
    assert dividend == dividend_before
    assert divisor == divisor_before


@pytest.mark.parametrize("constant", [-3, -0.5, 2, 7.25])
def test_polynomial_scalar_round_trip_preserves_value_and_source(constant):
    polynomial = kw.Poly("2x^3-4x+6")
    original = polynomial.__copy__()

    result = polynomial / constant * constant

    assert result == original
    assert polynomial == original


def test_matrix_determinant_is_multiplicative_and_operands_are_unchanged():
    left = kw.Matrix([[2, 3], [1, 4]])
    right = kw.Matrix([[5, -1], [2, 3]])
    left_before = left.__copy__()
    right_before = right.__copy__()

    product = left @ right

    assert product.determinant() == pytest.approx(left.determinant() * right.determinant())
    assert left == left_before
    assert right == right_before


def test_matrix_inverse_is_two_sided_and_does_not_mutate_source():
    matrix = kw.Matrix([[4, 7], [2, 6]])
    original = matrix.__copy__()
    inverse = matrix.inverse()

    identity_values = [1, 0, 0, 1]
    assert list((matrix @ inverse).yield_items()) == pytest.approx(identity_values)
    assert list((inverse @ matrix).yield_items()) == pytest.approx(identity_values)
    assert matrix == original


def test_singular_matrix_has_no_inverse_and_source_is_preserved():
    matrix = kw.Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    original = matrix.__copy__()

    assert matrix.inverse() is None
    assert matrix == original


def test_vector_dot_product_is_distributive():
    left = kw.Vector((2, -1, 3))
    first = kw.Vector((4, 0, -2))
    second = kw.Vector((-1, 5, 1))

    assert left * (first + second) == pytest.approx(left * first + left * second)


@pytest.mark.parametrize("angle", [-2.3, -0.75, 0, 0.4, 1.8, math.pi])
def test_fundamental_trigonometric_identity(angle):
    x = kw.Var("x")
    sine = kw.Sin(x)
    cosine = kw.Cos(x)

    sine.assign(x=angle)
    cosine.assign(x=angle)

    assert sine.try_evaluate() ** 2 + cosine.try_evaluate() ** 2 == pytest.approx(1)


@pytest.mark.parametrize("angle", [-1.1, -0.2, 0.3, 1.25])
def test_tangent_matches_sine_over_cosine_away_from_poles(angle):
    x = kw.Var("x")
    tangent = kw.Tan(x)
    sine = kw.Sin(x)
    cosine = kw.Cos(x)
    for expression in (tangent, sine, cosine):
        expression.assign(x=angle)

    assert tangent.try_evaluate() == pytest.approx(
        sine.try_evaluate() / cosine.try_evaluate()
    )
