"""Numerical and public API regressions for Bairstow quadratic deflation."""
import numpy as np
import pytest

import kiwicalc as kw
from kiwicalc.numeric.roots import _bairstow_division


def assert_roots(actual, expected, atol=1e-6):
    remaining = list(map(complex, actual))
    assert len(remaining) == len(expected)
    for root in expected:
        index = min(range(len(remaining)), key=lambda i: abs(remaining[i] - root))
        assert abs(remaining.pop(index) - root) <= atol * max(1, abs(root))


@pytest.mark.parametrize('expected', [
    [-3, -1, 2, 4], [-2, 1, 3], [1j, -1j, 2j, -2j],
    [0, 0, 2, 1j, -1j], [1, 1, -1, -1], [2, 2, 2, 2],
    [1, 2, 3, 4, 5, 6], [-2, 1+2j, 1-2j, -3j, 3j],
])
def test_known_real_complex_and_repeated_roots(expected):
    coefficients = np.real_if_close(np.poly(expected)).tolist()
    original = coefficients.copy()
    actual = kw.bairstow_method(coefficients)
    assert_roots(actual, expected, atol=2e-3 if len(set(expected)) < len(expected) else 1e-6)
    assert coefficients == original
    assert np.allclose(np.poly(actual), coefficients, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize('scale', [1e-100, -7.0, 1e100])
def test_coefficient_scaling_does_not_change_roots(scale):
    assert_roots(kw.bairstow_method(np.array([1, -2, -5, 6]) * scale), [-2, 1, 3])


def test_low_degrees_leading_zeros_and_generators():
    assert kw.bairstow_method([7]) == []
    assert kw.bairstow_method([0, 0, 7]) == []
    assert_roots(kw.bairstow_method(iter([0, 2, -6])), [3])
    assert_roots(kw.bairstow_method([1, -2, 1]), [1, 1])
    assert_roots(kw.bairstow_method([1, 0, 1]), [1j, -1j])
    assert kw.bairstow_method([3, 0, 0, 0]) == [0.0, 0.0, 0.0]
    assert_roots(kw.bairstow_method([1, -1e8, 1]), [1e8, 1e-8], atol=1e-12)


@pytest.mark.parametrize('coefficients', [[], [[1, 2]], [0, 0], [1, np.nan], [np.inf, 1], [1+1j, 2]])
def test_rejects_invalid_polynomials(coefficients):
    with pytest.raises(ValueError):
        kw.bairstow_method(coefficients)


@pytest.mark.parametrize('kwargs', [
    {'epsilon': 0}, {'epsilon': -1}, {'epsilon': np.nan},
    {'nmax': 0}, {'nmax': 1.5}, {'nmax': True}, {'r': np.inf}, {'s': 1j},
])
def test_rejects_invalid_solver_options(kwargs):
    with pytest.raises(ValueError):
        kw.bairstow_method([1, 0, -1], **kwargs)


def test_failure_is_explicit_and_restarts_are_deterministic():
    with pytest.raises(RuntimeError, match='did not converge'):
        kw.bairstow_method([1, -6, 11, -6], nmax=1, r=10, s=10)
    coefficients = [1, 0, 0, 0, 1]
    first = kw.bairstow_method(coefficients, r=0, s=0)
    assert first == kw.bairstow_method(coefficients, r=0, s=0)
    assert_roots(first, np.roots(coefficients))


def test_synthetic_jacobian_matches_finite_differences():
    coefficients = np.array([1., -2., 3., -4., 5.])
    r, s, h = .3, -.7, 1e-6
    _, jacobian = _bairstow_division(coefficients, r, s)
    for column, delta in enumerate([(h, 0), (0, h)]):
        plus, _ = _bairstow_division(coefficients, r + delta[0], s + delta[1])
        minus, _ = _bairstow_division(coefficients, r - delta[0], s - delta[1])
        assert np.allclose(jacobian[:, column], (plus[-2:] - minus[-2:]) / (2*h))


@pytest.mark.parametrize('seed', range(10))
def test_seeded_polynomials_agree_with_numpy(seed):
    rng = np.random.default_rng(seed)
    roots = list(rng.uniform(-3, 3, 3)) + [1+1j, 1-1j]
    coefficients = np.real_if_close(np.poly(roots))
    assert_roots(kw.bairstow_method(coefficients), np.roots(coefficients), atol=1e-5)
