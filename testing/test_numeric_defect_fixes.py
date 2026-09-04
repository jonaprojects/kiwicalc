import math

import numpy as np
import pytest

import kiwicalc as kw
from scripts.benchmark_numeric import root_error


@pytest.mark.parametrize('method', [kw.reinman, kw.simpson])
@pytest.mark.parametrize('count', [True, 2.5, -1, 0, np.nan])
def test_integration_rejects_invalid_counts(method, count):
    with pytest.raises(ValueError, match='N must be an integer'):
        method(lambda x: x, 0, 1, count)


@pytest.mark.parametrize('method', [kw.reinman, kw.simpson])
def test_integration_reversed_and_zero_width_bounds(method):
    assert method(lambda x: 2*x+1, 2, -1, 10) == pytest.approx(-6)
    assert method(math.sin, 1, 1, 10) == 0


def test_simpson_evaluates_each_grid_sample_only_once():
    samples = []
    def f(x):
        samples.append(x)
        return x**3
    assert kw.simpson(f, 0, 1, 10) == pytest.approx(.25)
    assert len(samples) == len(set(samples)) == 11
    assert min(samples) == 0 and max(samples) == 1


@pytest.mark.parametrize('scale', [1e-100, 1., -3., 1e100])
def test_aberth_scale_independent_roots(scale):
    coefficients = np.array([1., -2., -5., 6.]) * scale
    original = coefficients.copy()
    roots = kw.aberth_method(lambda z: np.polyval(coefficients, z),
                             lambda z: np.polyval(np.polyder(coefficients), z),
                             coefficients, epsilon=1e-12, nmax=1000)
    assert root_error(roots, [-2, 1, 3]) < 1e-8
    assert np.array_equal(coefficients, original)


def test_aberth_keeps_nearby_distinct_roots_without_fixed_distance_merging():
    # Expanded coefficients of a quadratic; roots are closer than the old 1e-4 filter.
    a, b = 1., 1.00005
    coefficients = [1., -(a+b), a*b]
    roots = kw.aberth_method(lambda z: (z-a)*(z-b), lambda z: 2*z-a-b,
                             coefficients, epsilon=1e-12, nmax=1000)
    assert len(roots) == 2
    assert root_error(roots, [a, b]) < 1e-9


def test_aberth_complex_roots_and_silent_initialization(capsys):
    roots = kw.aberth_method(lambda z: z**4+1, lambda z: 4*z**3,
                             [1, 0, 0, 0, 1], epsilon=1e-12, nmax=1000)
    assert root_error(roots, np.roots([1, 0, 0, 0, 1])) < 1e-9
    assert capsys.readouterr().out == ''


def test_aberth_failure_is_explicit():
    with pytest.raises(RuntimeError, match='did not converge'):
        kw.aberth_method(lambda z: z**3-1, lambda z: 3*z*z, [1, 0, 0, -1], nmax=1)
    with pytest.raises(RuntimeError, match='non-finite'):
        kw.aberth_method(lambda z: np.nan, lambda z: 1, [1, 0, -1])


@pytest.mark.parametrize('coefficients', [[], [0, 0], [1, np.inf], [[1, 0], [0, 1]]])
def test_aberth_invalid_polynomial(coefficients):
    with pytest.raises(ValueError):
        kw.aberth_method(lambda z: z, lambda z: 1, coefficients)
