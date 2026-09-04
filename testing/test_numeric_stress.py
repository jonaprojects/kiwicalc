"""Accuracy checks and regressions for repaired numeric defects.

There are deliberately no wall-clock performance assertions here.
"""
import numpy as np
import pytest

import kiwicalc as kw
from scripts.benchmark_numeric import root_error


@pytest.mark.parametrize('degree', [3, 4, 6, 8, 10])
@pytest.mark.parametrize('seed', range(4))
def test_bairstow_separated_roots_and_coefficient_reconstruction(degree, seed):
    rng = np.random.default_rng(seed)
    expected = np.linspace(-3, 3, degree) + rng.uniform(-.08, .08, degree)
    coefficients = np.poly(expected)
    actual = kw.bairstow_method(coefficients)
    assert root_error(actual, expected) < 1e-6
    assert np.allclose(np.poly(actual), coefficients, rtol=1e-7, atol=1e-7)
    assert root_error(actual, np.roots(coefficients)) < 1e-6


def test_bairstow_is_independent_of_numpy_roots_and_global_random_state(monkeypatch):
    state = np.random.get_state()
    monkeypatch.setattr(np, 'roots', lambda *args: pytest.fail('Unexpected fallback to NumPy roots'))
    result = kw.bairstow_method([1, 0, 0, 0, 1], r=0, s=0)
    after = np.random.get_state()
    assert np.array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    assert max(abs(np.polyval([1, 0, 0, 0, 1], result))) < 1e-9


@pytest.mark.parametrize('method', [
    lambda f, df, ddf: kw.newton_raphson(f, df, 1.5),
    lambda f, df, ddf: kw.halleys_method(f, df, ddf, 1.5),
    lambda f, df, ddf: kw.secant_method(f, 1, 2),
    lambda f, df, ddf: kw.inverse_interpolation(f, 1, 1.5, 2),
    lambda f, df, ddf: kw.laguerre_method(f, df, ddf, 1.5, 2),
    lambda f, df, ddf: kw.ostrowski_method(f, df, 1.5),
    lambda f, df, ddf: kw.chebychevs_method(f, df, ddf, 1.5),
    lambda f, df, ddf: kw.steffensen_method(f, 1.5),
    lambda f, df, ddf: kw.bisection_method(f, 1, 2),
])
def test_scalar_root_methods_agree_with_analytic_and_numpy_root(method):
    result = method(lambda x: x*x-2, lambda x: 2*x, lambda x: 2)
    expected = max(np.roots([1, 0, -2]))
    assert abs(result - expected) < 2e-5


@pytest.mark.parametrize('intervals', [2, 10, 101])
def test_trapezoid_matches_numpy_for_same_grid(intervals):
    trapezoid = getattr(np, 'trapezoid', None) or np.trapz
    x = np.linspace(-1, 2, intervals+1)
    assert kw.trapz(lambda x: x*x, -1, 2, intervals) == pytest.approx(trapezoid(x*x, x))


@pytest.mark.parametrize('samples', [3, 5, 11, 101])
def test_simpson_odd_sample_counts_integrate_cubics_exactly(samples):
    assert kw.simpson(lambda x: x**3, 0, 2, samples) == pytest.approx(4)


@pytest.mark.parametrize('scheme', ['central', 'forward', 'backward'])
def test_finite_difference_known_derivative(scheme):
    assert kw.numerical_diff(np.sin, .5, method=scheme, h=1e-6) == pytest.approx(np.cos(.5), abs=1e-6)


def test_optimization_converges_to_known_stationary_points():
    assert kw.gradient_descent(lambda x: 2*(x-3), 0) == pytest.approx(3, abs=1e-4)
    assert kw.gradient_ascent(lambda x: -2*(x+2), 0) == pytest.approx(-2, abs=1e-4)


def test_riemann_integrates_constant_exactly():
    assert kw.reinman(lambda x: 1, 0, 1, 11) == pytest.approx(1)


@pytest.mark.parametrize('samples', [4, 10, 100])
def test_simpson_even_sample_counts_cover_the_complete_interval(samples):
    assert kw.simpson(lambda x: 1, 0, 1, samples) == pytest.approx(1)


def test_aberth_small_coefficient_scale_preserves_distinct_roots():
    coefficients = np.array([1., -2., -5., 6.]) * 1e-100
    actual = kw.aberth_method(lambda x: np.polyval(coefficients, x),
                             lambda x: np.polyval(np.polyder(coefficients), x),
                             coefficients, epsilon=1e-12, nmax=1000)
    assert len(actual) == 3
    assert root_error(actual, [-2, 1, 3]) < 1e-6


def test_aberth_preserves_distinct_clustered_roots():
    expected = [1, 1.001, 1.002, 1.003]
    coefficients = np.poly(expected)
    actual = kw.aberth_method(lambda x: np.polyval(coefficients, x),
                             lambda x: np.polyval(np.polyder(coefficients), x),
                             coefficients, epsilon=1e-12, nmax=1000)
    assert len(actual) == 4
    assert root_error(actual, expected) < 1e-4


def test_benchmark_matching_detects_multiplicity_and_bad_roots():
    assert root_error([1, 2], [2, 1]) == 0
    assert root_error([1], [1, 1]) is None
    assert root_error([np.nan], [1]) is None
    assert root_error([1, 1], [1, 2]) == pytest.approx(.5)
