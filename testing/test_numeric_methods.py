import math

import pytest

import kiwicalc as kw


@pytest.mark.parametrize(
    ('method', 'expected'),
    [
        (lambda: kw.newton_raphson(lambda x: x*x - 2, lambda x: 2*x, 1), math.sqrt(2)),
        (lambda: kw.halleys_method(lambda x: x*x - 2, lambda x: 2*x, lambda x: 2, 1), math.sqrt(2)),
        (lambda: kw.secant_method(lambda x: x*x - 2, 1, 2), math.sqrt(2)),
        (lambda: kw.ostrowski_method(lambda x: x*x - 2, lambda x: 2*x, 1), math.sqrt(2)),
        (lambda: kw.chebychevs_method(lambda x: x*x - 2, lambda x: 2*x, lambda x: 2, 1), math.sqrt(2)),
        (lambda: kw.steffensen_method(lambda x: x*x - 2, 1), math.sqrt(2)),
        (lambda: kw.bisection_method(lambda x: x*x - 2, 1, 2), math.sqrt(2)),
    ],
)
def test_root_finders_converge_on_square_root_of_two(method, expected):
    assert method() == pytest.approx(expected, rel=1e-4)


def test_bisection_accepts_reversed_bounds():
    assert kw.bisection_method(lambda x: x - 3, 5, 0) == pytest.approx(3, abs=1e-4)


@pytest.mark.parametrize(
    'bounds',
    [(1, 1), (2, 3)],
)
def test_bisection_rejects_invalid_bounds(bounds):
    with pytest.raises(ValueError):
        kw.bisection_method(lambda x: x*x - 2, *bounds)


@pytest.mark.parametrize(
    ('method', 'expected'),
    [
        (lambda: kw.reinman(lambda x: x, 0, 1, 1001), 0.5005),
        (lambda: kw.trapz(lambda x: x*x, 0, 1, 1000), 1 / 3),
        (lambda: kw.simpson(math.sin, 0, math.pi, 1001), 2),
        (lambda: kw.numerical_diff(math.sin, 0), 1),
    ],
)
def test_calculus_methods_against_analytic_results(method, expected):
    assert method() == pytest.approx(expected, rel=1e-4, abs=1e-4)


def test_calculus_methods_validate_sample_counts():
    with pytest.raises(ValueError):
        kw.reinman(lambda x: x, 0, 1, 1)
    with pytest.raises(ValueError):
        kw.trapz(lambda x: x, 0, 1, 0)
    with pytest.raises(ValueError):
        kw.simpson(lambda x: x, 0, 1, 2)


def test_gradient_descent_finds_quadratic_minimum():
    result = kw.gradient_descent(lambda x: 2 * (x - 4), initial_value=0, learning_rate=0.1)
    assert result == pytest.approx(4, abs=1e-4)


def test_gradient_ascent_finds_concave_quadratic_maximum():
    result = kw.gradient_ascent(lambda x: -2 * (x - 4), initial_value=0, learning_rate=0.1)
    assert result == pytest.approx(4, abs=1e-4)
