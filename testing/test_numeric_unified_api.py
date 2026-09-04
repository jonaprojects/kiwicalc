import math
import warnings

import numpy as np
import pytest

import kiwicalc as kw


def test_public_exports():
    for name in ('differentiate', 'integrate', 'find_root', 'NumericalResult'):
        assert name in kw.__all__
        assert hasattr(kw, name)


@pytest.mark.parametrize('f', [lambda x: x*x, kw.Function('f(x)=x^2'), kw.Poly('x^2'), kw.Mono('x^2')])
def test_math_objects_share_the_same_scalar_interface(f):
    assert kw.differentiate(f, at=2) == pytest.approx(4, abs=1e-7)
    assert kw.integrate(f, 0, 1) == pytest.approx(1/3)
    assert isinstance(kw.differentiate(f, at=2), float)


def test_constant_and_non_x_expressions():
    assert kw.integrate(kw.Mono(3), 0, 2) == pytest.approx(6)
    assert kw.differentiate(kw.Poly('t^2'), at=3) == pytest.approx(6)


@pytest.mark.parametrize('method', ['central', 'forward', 'backward'])
def test_differentiation_methods_and_diagnostics(method):
    calls = []
    def f(x):
        calls.append(x)
        return math.sin(x)
    result = kw.differentiate(f, .5, method=method, step=1e-6, return_info=True)
    assert result.value == pytest.approx(math.cos(.5), abs=1e-6)
    assert result.function_calls == len(calls) == 2
    assert result.method == method
    assert result.converged is None and result.residual is None


@pytest.mark.parametrize('method', ['simpson', 'trapezoid', 'midpoint'])
def test_integration_interval_convention_reversed_bounds_and_info(method):
    result = kw.integrate(lambda x: x, 2, -1, method=method, intervals=7, return_info=True)
    assert result.value == pytest.approx(-1.5)
    assert result.method == method
    assert result.function_calls > 0
    assert result.converged is None
    assert kw.integrate(lambda x: pytest.fail('must not evaluate'), 1, 1, method=method) == 0


@pytest.mark.parametrize('options,method', [
    ({'bracket': (0, 2)}, 'brent'),
    ({'x0': 1}, 'newton'),
    ({'x0': 1, 'derivative': lambda x: 2*x}, 'newton'),
    ({'x0': 1, 'x1': 2}, 'secant'),
    ({'x0': 1, 'method': 'halley', 'derivative': lambda x: 2*x, 'second_derivative': lambda x: 2}, 'halley'),
    ({'x0': 1.5, 'method': 'steffensen'}, 'steffensen'),
])
def test_root_dispatch_and_residual_contract(options, method):
    info = kw.find_root(lambda x: x*x-2, tolerance=1e-10, return_info=True, **options)
    assert info.method == method
    assert info.converged
    assert info.value == pytest.approx(math.sqrt(2), abs=1e-9)
    assert info.residual <= 1e-10
    assert info.function_calls > 0


def test_root_accepts_expressions_and_function_objects():
    assert kw.find_root(kw.Poly('x^2-2'), bracket=(0, 2)) == pytest.approx(math.sqrt(2))
    assert kw.find_root(kw.Function('f(t)=t-3'), x0=1) == pytest.approx(3)


def test_root_checks_initial_root_before_zero_derivative_and_bracket_endpoints():
    assert kw.find_root(lambda x: x*x, x0=0, derivative=lambda x: 0) == 0
    assert kw.find_root(lambda x: x-2, bracket=(3, 2)) == 2


def test_root_residual_check_does_not_accept_discontinuity_or_stalled_iterations():
    discontinuous = lambda x: -1 if x < .3 else 1
    with pytest.raises(RuntimeError, match='residual tolerance'):
        kw.find_root(discontinuous, bracket=(0, 1), max_iterations=8)
    result = kw.find_root(discontinuous, bracket=(0, 1), max_iterations=8, return_info=True)
    assert not result.converged and result.residual == 1
    with pytest.warns(UserWarning):
        with pytest.raises(RuntimeError):
            kw.find_root(lambda x: 1., x0=0, x1=1)


def test_function_call_counts_include_supplied_derivatives():
    calls = []
    def f(x):
        calls.append('f')
        return x*x-2
    def df(x):
        calls.append('df')
        return 2*x
    info = kw.find_root(f, x0=1, derivative=df, return_info=True)
    assert info.function_calls == len(calls)


def test_user_type_error_propagates_without_retry():
    calls = []
    def broken(x):
        calls.append(x)
        raise TypeError('failure inside callback')
    with pytest.raises(TypeError, match='failure inside callback'):
        kw.integrate(broken, 0, 1)
    assert len(calls) == 1


@pytest.mark.parametrize('output', [[1, 2], np.array([1]), 1j, True, '1'])
def test_scalar_real_output_contract(output):
    with pytest.raises(TypeError, match='scalar real'):
        kw.differentiate(lambda x: output, at=1)


@pytest.mark.parametrize('output', [np.nan, np.inf])
def test_nonfinite_output_is_not_a_convergence_failure(output):
    with pytest.raises(ValueError, match='finite'):
        kw.find_root(lambda x: output, bracket=(0, 1), return_info=True)


@pytest.mark.parametrize('f', ['x^2', 3, kw.Function('f(x,y)=x+y'), kw.Poly('x+y')])
def test_nonfunction_and_multivariable_inputs_are_rejected(f):
    with pytest.raises((TypeError, ValueError)):
        kw.integrate(f, 0, 1)


@pytest.mark.parametrize('options', [
    {}, {'method': 'brent', 'bracket': (0, 1)}, {'method': 'bisection'},
    {'method': 'secant', 'x0': 1}, {'x0': 1, 'x1': 1},
    {'method': 'halley', 'x0': 1}, {'bracket': (1, 1)},
    {'bracket': (1, 2, 3)}, {'bracket': (0, 2), 'x0': 1},
    {'method': 'newton', 'bracket': (0, 2)},
    {'x0': 1, 'tolerance': 0}, {'x0': 1, 'max_iterations': 0},
])
def test_root_option_validation(options):
    with pytest.raises((ValueError, TypeError)):
        kw.find_root(lambda x: x*x-2, **options)


def test_derivative_step_and_integration_count_validation():
    for step in (0, -1, np.inf):
        with pytest.raises(ValueError):
            kw.differentiate(math.sin, 1, step=step)
    with pytest.raises(ValueError, match='distinct'):
        kw.differentiate(math.sin, 1, step=1e-30)
    for count in (True, 1.2, 0):
        with pytest.raises(ValueError):
            kw.integrate(math.sin, 0, 1, intervals=count)
    assert np.allclose(kw.differentiate(math.sin, at=np.array([1, 2])), np.cos([1, 2]))


def test_legacy_methods_remain_available_and_unchanged():
    assert kw.numerical_diff(lambda x: x*x, 2) == pytest.approx(4)
    assert kw.simpson(lambda x: x*x, 0, 1, 101) == pytest.approx(1/3)
    assert kw.bisection_method(lambda x: x-1, 0, 2) == 1
