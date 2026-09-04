"""Regression coverage for explicit numerical-domain and budget failures."""
import math

import numpy as np
import pytest

import kiwicalc as kw


@pytest.mark.parametrize('call', [
    lambda: kw.differentiate(np.sin, [1], vectorized=1),
    lambda: kw.differentiate(np.sin, [True]),
    lambda: kw.differentiate_samples(1),
    lambda: kw.differentiate_samples([1, 2], axis=True),
    lambda: kw.differentiate_samples([1]),
    lambda: kw.differentiate_samples([1, 2], edge_order=3),
    lambda: kw.gradient(lambda x: x, [1], variables=['x', 'y']),
    lambda: kw.gradient(lambda x: x, [1], variables='x'),
    lambda: kw.gradient(kw.Poly('x+y'), [1]),
    lambda: kw.gradient(kw.Function('f(x,y)=x+y'), [1, 2], variables=['a', 'b']),
    lambda: kw.gradient(kw.Poly('x'), [1], variables=['y']),
    lambda: kw.gradient(1, [1]),
    lambda: kw.jacobian([], [1]),
    lambda: kw.gradient(lambda x: x, [1e308], step=1e308),
    lambda: kw.solve_system(lambda x: [x], [[1]]),
    lambda: kw.solve_system(lambda x: [x], [1], jacobian=3),
    lambda: kw.integrate_nd(lambda x: x, [(0, 1)], intervals=[1, 2]),
    lambda: kw.integrate_nd(lambda x: x, [0, 1]),
    lambda: kw.integrate_nd(lambda x: x, [(-1e308, 1e308)]),
    lambda: kw.find_root(lambda x: x, bracket=(-1e308, 1e308)),
    lambda: kw.find_root(lambda x: x, x0=1, method='newton', x1=2),
    lambda: kw.find_root(lambda x: x, bracket=(-1, 1), derivative=lambda x: 1),
    lambda: kw.find_root(lambda x: x, x0=1, second_derivative=lambda x: 0),
    lambda: kw.integrate(lambda x: x, -1e308, 1e308),
])
def test_explicit_invalid_inputs(call):
    with np.errstate(over='ignore'):
        with pytest.raises((ValueError, TypeError)):
            call()


def test_vector_matrix_interoperability_and_zero_partial():
    np.testing.assert_allclose(kw.jacobian([kw.Poly('x'), kw.Poly('x^2')], [2]), [[1], [4]])
    np.testing.assert_allclose(kw.jacobian(lambda x: kw.Matrix([[x]]).to_numpy().ravel(), [2]), [[1]])
    info = kw.solve_system(lambda x: [x-1], [2], jacobian=lambda x: kw.Matrix([[1]]), return_info=True)
    assert info.converged
    np.testing.assert_allclose(kw.gradient(kw.Poly('y'), [1, 2], variables=['x', 'y']), [0, 1])
    np.testing.assert_allclose(kw.gradient(lambda x: x*x, [2], method='forward', step=[.01]), [4.01])


def test_system_failure_diagnostics_and_precision_limit():
    result = kw.solve_system(lambda x: [1.], [0], jacobian=lambda x: [[1]], return_info=True)
    assert not result.converged and 'stalled' in result.message
    with np.errstate(over='ignore', invalid='ignore'):
        result = kw.solve_system(lambda x: [1e308], [0], jacobian=lambda x: [[1e-308]], return_info=True)
    assert not result.converged and 'finite' in result.message
    result = kw.integrate(math.exp, 1, np.nextafter(1., 2.), method='adaptive_simpson', return_info=True)
    assert not result.converged


@pytest.mark.parametrize('options', [
    {'coefficients': '12'}, {'coefficients': [1, 2], 'epsilon': 0},
    {'coefficients': [1, 2], 'nmax': 0},
])
def test_polynomial_solver_input_validation(options):
    with pytest.raises(ValueError):
        kw.aberth_method(lambda x: x, lambda x: 1, **options)


def test_polynomial_solver_constant_linear_and_nonfinite():
    with pytest.raises(TypeError):
        kw.aberth_method(1, lambda x: 1, [1, 2])
    assert kw.aberth_method(lambda x: 2, lambda x: 0, [2]) == set()
    assert kw.aberth_method(lambda x: 2*x+4, lambda x: 2, [2, 4]) == {-2+0j}
    with pytest.raises(RuntimeError, match='non-finite'):
        kw.aberth_method(lambda x: np.inf, lambda x: 1, [1, 0, 1])


def test_legacy_solver_budgets_and_one_sided_internal_difference():
    from kiwicalc.numeric.api import _difference
    assert _difference(lambda x: x*x, 2, 'forward', .01) == pytest.approx(4.01)
    with pytest.raises(ValueError, match='distinct'):
        _difference(math.sin, 1, 'central', 1e-30)
    for method in (kw.gradient_descent, kw.gradient_ascent):
        with pytest.warns(UserWarning):
            assert math.isfinite(method(lambda x: 1, 1, nmax=1))
    assert kw.ostrowski_method(lambda x: x-1, lambda x: 1, 0, nmax=1) == pytest.approx(1)
    with pytest.warns(UserWarning):
        kw.chebychevs_method(lambda x: x-1, lambda x: 1, lambda x: 0, 0, nmax=1)


def test_polynomial_legacy_normalization_and_bairstow_validation():
    for method in (lambda: kw.durand_kerner(lambda x: 2*x*x-2, [2, 0, -2]),
                   lambda: kw.durand_kerner2([2, 0, -2]),
                   lambda: kw.durand_kerner2([1, 0, -1])):
        result = method()
        assert len(result) == 2
        assert all(abs(z*z-1) < 1e-3 for z in result)
    with pytest.raises(TypeError):
        kw.bairstow_method(2)
    with pytest.raises(TypeError):
        kw.bairstow_method('123')
    with pytest.raises(TypeError):
        kw.bairstow_method(['not a coefficient'])
    with pytest.raises(ValueError):
        kw.bairstow_method([1, 2], r='invalid')


def test_legacy_zero_derivative_and_small_secant_step():
    root = kw.chebychevs_method(lambda x: x*x-1, lambda x: 2*x, lambda x: 2, 0)
    assert abs(root*root-1) < 1e-5
    assert kw.secant_method(lambda x: 1e10*x-1, 0, 1, epsilon=1e-5) == pytest.approx(1e-10, abs=1e-15)


def test_wrapper_handles_legacy_solver_without_candidate(monkeypatch):
    from kiwicalc.numeric import roots
    monkeypatch.setattr(roots, 'newton_raphson', lambda *args, **kwargs: None)
    result = kw.find_root(lambda x: x*x-2, x0=1, return_info=True)
    assert not result.converged and result.value == 1


def test_exact_bairstow_factor_and_zero_quadratic():
    from kiwicalc.numeric.roots import _bairstow_quadratic
    assert _bairstow_quadratic(1., 0., 0.) == [0., 0.]
    roots = kw.bairstow_method([1, 0, 0, 0, -1], r=0, s=1, nmax=1)
    assert len(roots) == 4
    assert all(abs(root**4-1) < 1e-12 for root in roots)


def test_system_line_search_skips_overflowing_candidates():
    with np.errstate(over='ignore'):
        info = kw.solve_system(lambda x: [1e308], [1.7e308],
                               jacobian=lambda x: [[-1.]], return_info=True)
    assert not info.converged and 'stalled' in info.message
