import math

import numpy as np
import pytest

import kiwicalc as kw


@pytest.mark.parametrize('method', ['central', 'forward', 'backward', 'richardson'])
@pytest.mark.parametrize('vectorized', [False, True])
def test_array_differentiation(method, vectorized):
    x = np.array([[.1, .5], [1., 2.]])
    original = x.copy()
    calls = []
    def f(v):
        calls.append(v)
        return np.sin(v)
    result = kw.differentiate(f, x, method=method, vectorized=vectorized, return_info=True)
    np.testing.assert_allclose(result.value, np.cos(x), atol=1e-7)
    np.testing.assert_array_equal(x, original)
    assert result.function_calls == len(calls) == (4 if method == 'richardson' else 2) * (1 if vectorized else x.size)


def test_richardson_improves_accuracy_and_has_diagnostics():
    plain = kw.differentiate(math.exp, 1., step=.1)
    rich = kw.differentiate(math.exp, 1., step=.1, method='richardson', return_info=True)
    assert abs(rich.value - math.e) < abs(plain - math.e) / 100
    assert rich.estimated_error > 0
    assert rich.converged is None
    np.testing.assert_allclose(kw.differentiate(kw.Poly('x^3'), [1, 2], method='richardson'), [3, 12])


def test_array_step_broadcast_and_callback_errors():
    np.testing.assert_allclose(kw.differentiate(lambda x: x*x, [[1, 2], [3, 4]], step=[.01, .02]), [[2, 4], [6, 8]])
    with pytest.raises(ValueError, match='broadcast'):
        kw.differentiate(math.sin, [1, 2], step=[1, 2, 3])
    with pytest.raises(ValueError, match='shape'):
        kw.differentiate(lambda x: 1., [1, 2], vectorized=True)
    with pytest.raises(ValueError, match='empty'):
        kw.differentiate(math.sin, [])
    calls = []
    def broken(x):
        calls.append(x)
        raise TypeError('callback failed')
    with pytest.raises(TypeError, match='callback failed'):
        kw.differentiate(broken, [1, 2], vectorized=True)
    assert len(calls) == 1


@pytest.mark.parametrize('f,bracket,root', [
    (lambda x: x*x-2, (0, 2), math.sqrt(2)),
    (lambda x: math.cos(x)-x, (0, 1), .7390851332151607),
    (lambda x: x**3-1, (10, 0), 1),
    (lambda x: math.exp(x)-1000, (0, 10), math.log(1000)),
    (lambda x: x-2, (2, 3), 2),
    (lambda x: x-2, (1, 2), 2),
    (lambda x: 1e100*(x-1.25), (0, 2), 1.25),
])
def test_brent_known_roots(f, bracket, root):
    info = kw.find_root(f, bracket=bracket, method='brent', return_info=True, tolerance=1e-9)
    assert info.converged and info.residual <= 1e-9
    assert info.value == pytest.approx(root, abs=1e-8)
    assert info.iterations < 100


def test_brent_budget_bracket_discontinuity_and_legacy_selection():
    with pytest.raises(ValueError, match='opposite signs'):
        kw.find_root(lambda x: x*x+1, bracket=(-1, 1))
    info = kw.find_root(lambda x: x*x-2, bracket=(0, 2), max_iterations=1, return_info=True)
    assert not info.converged and info.iterations == 1
    with pytest.raises(RuntimeError):
        kw.find_root(lambda x: -1 if x < .123 else 1, bracket=(0, 1))
    assert kw.find_root(lambda x: x-1, bracket=(0, 2), method='bisection') == 1


@pytest.mark.parametrize('f,a,b,expected', [
    (math.sin, 0, math.pi, 2.),
    (math.exp, 1, 0, 1-math.e),
    (lambda x: x**4, -1, 2, 33/5),
    (lambda x: 1/(1+100*x*x), -1, 1, math.atan(10)/5),
])
def test_adaptive_simpson_accuracy_and_counts(f, a, b, expected):
    calls = []
    def counted(x):
        calls.append(x)
        return f(x)
    info = kw.integrate(counted, a, b, method='adaptive_simpson', tolerance=1e-10, return_info=True)
    assert info.value == pytest.approx(expected, abs=1e-9)
    assert info.converged and info.estimated_error <= 1e-10
    assert info.function_calls == len(calls) == len(set(calls))


def test_adaptive_limits_zero_and_errors():
    assert kw.integrate(lambda x: pytest.fail(), 1, 1, method='adaptive_simpson') == 0
    for options in ({'max_evaluations': 5}, {'max_depth': 1}):
        info = kw.integrate(math.exp, 0, 1, method='adaptive_simpson', tolerance=1e-15, return_info=True, **options)
        assert not info.converged
        with pytest.raises(RuntimeError, match='exhausted'):
            kw.integrate(math.exp, 0, 1, method='adaptive_simpson', tolerance=1e-15, **options)
    with pytest.raises(ValueError):
        kw.integrate(math.exp, 0, 1, method='adaptive_simpson', max_evaluations=4)
    with pytest.raises(ValueError, match='finite'):
        kw.integrate(lambda x: np.nan, 0, 1, method='adaptive_simpson')


@pytest.mark.parametrize('reverse', [False, True])
def test_sampled_nonuniform_quadratic_and_cumulative_linear(reverse):
    x = np.array([0., .2, .7, 1.4, 2.])
    if reverse:
        x = x[::-1]
    np.testing.assert_allclose(kw.differentiate_samples(x*x, x), 2*x, atol=1e-12)
    np.testing.assert_allclose(kw.cumulative_integrate(2*x, x, initial=3), 3+x*x-x[0]**2, atol=1e-12)


def test_sampled_axes_spacing_two_points_and_exports():
    x = np.arange(4)*.5
    values = np.stack([x*x, 3*x*x], axis=1)
    np.testing.assert_allclose(kw.differentiate_samples(values, spacing=.5, axis=0), np.stack([2*x, 6*x], axis=1), atol=1e-12)
    np.testing.assert_allclose(kw.cumulative_integrate(np.ones((4, 2)), spacing=.5, axis=0), np.stack([x, x], axis=1))
    np.testing.assert_allclose(kw.differentiate_samples([2, 4], spacing=2), [1, 1])
    for name in ('differentiate_samples', 'cumulative_integrate'):
        assert name in kw.__all__


@pytest.mark.parametrize('options', [
    {'x': [0, 0, 1]}, {'x': [0, 2, 1]}, {'x': [0, 1]},
    {'x': [0, 1, np.inf]}, {'axis': 2}, {'spacing': 0},
    {'x': [0, 1, 2], 'spacing': 2},
])
def test_sampled_validation(options):
    for method in (kw.differentiate_samples, kw.cumulative_integrate):
        with pytest.raises((ValueError, TypeError)):
            method([1, 2, 3], **options)
