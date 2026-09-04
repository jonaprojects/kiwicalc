"""Friendly real-valued differentiation, integration, and root-finding APIs."""

from dataclasses import dataclass
from numbers import Real
from typing import Optional, Union

import numpy as np

from kiwicalc.numeric import calculus, roots


@dataclass(frozen=True)
class NumericalResult:
    """Optional numerical diagnostics.

    Fixed-grid integration and differentiation have no convergence test:
    ``converged`` and ``residual`` are None for those methods. Richardson's
    ``estimated_error`` is the extrapolation correction magnitude, not a bound.
    Adaptive Simpson reports a heuristic error estimate and convergence status.
    ``function_calls`` counts evaluations including supplied derivative calls.
    ``value`` is a float or ndarray; newer iterative methods report ``iterations``.
    """

    value: Union[float, np.ndarray]
    method: str
    function_calls: int
    converged: Optional[bool] = None
    residual: Optional[float] = None
    message: str = ''
    iterations: Optional[int] = None
    estimated_error: Optional[Union[float, np.ndarray]] = None


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a scalar real number')
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _positive(value, name):
    value = _real(value, name)
    if value <= 0:
        raise ValueError(f'{name} must be positive')
    return value


def _count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _method(value, choices):
    if not isinstance(value, str) or value not in choices:
        raise ValueError('method must be one of: ' + ', '.join(choices))
    return value


class _ScalarFunction:
    """Adapt existing math objects once; do not retry failing callbacks."""

    def __init__(self, function):
        from kiwicalc.core.interfaces import IExpression
        from kiwicalc.functions.function import Function

        if isinstance(function, IExpression):
            variables = list(function.variables)
            if len(variables) > 1:
                raise ValueError('Expected a single-variable expression')
            if variables:
                function = function.to_lambda()
            else:
                constant = _real(function.try_evaluate(), 'expression value')
                function = lambda x: constant
        elif isinstance(function, Function):
            if function.num_of_variables != 1:
                raise ValueError('Expected a single-variable Function')
        if not callable(function):
            raise TypeError('f must be callable, a Function, or a single-variable expression')
        self.function = function
        self.calls = 0

    def __call__(self, x):
        x = _real(x, 'evaluation point')
        self.calls += 1
        # User exceptions propagate unchanged, without arity probing or retries.
        return _real(self.function(x), 'function output')


def _difference(f, at, method, step):
    if step is None:
        power = 1 / 3 if method == 'central' else 1 / 2
        step = np.finfo(float).eps**power * max(1, abs(at))
    else:
        step = _positive(step, 'step')
    if ((method != 'backward' and (not np.isfinite(at + step) or at + step == at)) or
            (method != 'forward' and (not np.isfinite(at - step) or at - step == at))):
        raise ValueError('step must produce distinct finite sample points')
    return _real(calculus.numerical_diff(f, at, method=method, h=step), 'derivative')


def differentiate(f, at, *, method='central', step=None, vectorized=False, return_info=False):
    """Estimate a first derivative at a real point or array of independent points.

    Methods: central (default), forward, backward, richardson. The latter
    extrapolates central differences at step and step/2 to fourth order for
    smooth functions. step is scalar or broadcasts to at; None is scale-aware.
    By default callbacks receive scalar points. vectorized=True explicitly
    passes arrays and requires matching output shape, without probing/retries.
    Returns a float for scalar at, otherwise a shape-preserving ndarray;
    return_info=True adds diagnostics. This is not a multivariable gradient.
    """
    from kiwicalc.numeric.advanced import _differentiate
    return _differentiate(f, at, method, step, vectorized, return_info)


def integrate(f, a, b, *, method='simpson', intervals=1000, tolerance=1e-8,
              max_evaluations=10000, max_depth=30, return_info=False,
              explain=False, trace_limit=1000):
    """Integrate a scalar real function over finite bounds.

    Methods: simpson (default), trapezoid, midpoint. ``intervals`` always means
    subintervals, unlike the different N conventions of the legacy methods.
    Simpson rounds odd interval counts up to the next even count. Reversed
    bounds reverse the sign; equal bounds return zero without evaluation.
    Fixed resolution: no error estimate or convergence guarantee is implied.
    Alternatively adaptive_simpson refines panels until estimated absolute
    error <= tolerance, limited by max_evaluations and max_depth. intervals is
    unused by this method. Its error estimate is heuristic, not a rigorous bound;
    unresolved peaks/discontinuities can be missed. Failure raises RuntimeError
    unless return_info=True; callback errors always propagate.
    explain=True returns a NumericalExplanation for fixed-grid methods. It
    records up to trace_limit panels and provides lazy plots/animation. Normal
    calls use the original solver loops without tracing checks or allocations.
    """
    method = _method(method, ('simpson', 'trapezoid', 'midpoint', 'adaptive_simpson'))
    a, b = _real(a, 'a'), _real(b, 'b')
    intervals = _count(intervals, 'intervals')
    if not np.isfinite(b - a):
        raise ValueError('The integration interval width must be finite')
    function = _ScalarFunction(f)
    if explain:
        from kiwicalc.numeric.explanations import trace_integral
        return trace_integral(function, a, b, method, intervals, trace_limit)
    if method == 'adaptive_simpson':
        from kiwicalc.numeric.advanced import _adaptive_simpson
        return _adaptive_simpson(function, a, b, tolerance, max_evaluations, max_depth, return_info)
    if a == b:
        value = 0.0
    elif method == 'simpson':
        intervals += intervals % 2
        value = calculus.simpson(function, a, b, intervals + 1)
    elif method == 'trapezoid':
        value = calculus.trapz(function, a, b, intervals)
    else:
        value = calculus.reinman(function, a, b, intervals + 1)
    result = NumericalResult(_real(value, 'integral'), method, function.calls)
    return result if return_info else result.value


class _RootFound(Exception):
    def __init__(self, value, residual):
        self.value = value
        self.residual = residual


def find_root(f, *, bracket=None, x0=None, x1=None, derivative=None,
              second_derivative=None, method='auto', tolerance=1e-8,
              max_iterations=1000, return_info=False, explain=False, trace_limit=1000):
    """Find one real root with a unified absolute-residual stopping criterion.

    auto selects Brent's method for a bracket, secant for x0 and x1, otherwise
    Newton for x0 (using central differences if derivative is omitted).
    Explicit methods: brent, bisection, newton, secant, halley, steffensen.
    A bracket requires continuity; no routine can guarantee finding every root.

    A float is returned only when abs(f(root)) <= tolerance. Non-convergence
    raises RuntimeError, or returns converged=False with return_info=True.
    Input/domain errors always propagate. Legacy solver warnings are retained.
    function_calls includes function and explicit derivative evaluations.
    explain=True returns a NumericalExplanation even on non-convergence.
    Supported teaching methods: bisection, newton, secant. Select bisection
    explicitly for a bracket (auto still selects Brent, which is not traced).
    trace_limit bounds saved records, not computation. return_info is redundant
    when explain=True. Input/callback errors still propagate.
    """
    method = _method(method, ('auto', 'brent', 'bisection', 'newton', 'secant', 'halley', 'steffensen'))
    tolerance = _positive(tolerance, 'tolerance')
    max_iterations = _count(max_iterations, 'max_iterations')
    if method == 'auto':
        method = 'brent' if bracket is not None else 'secant' if x1 is not None else 'newton'
    if bracket is not None and method not in ('bisection', 'brent'):
        raise ValueError('bracket is supported only by bisection or brent')
    if method in ('bisection', 'brent'):
        if bracket is None:
            raise ValueError(f'{method} requires bracket=(a, b)')
        try:
            a, b = bracket
        except (TypeError, ValueError) as exc:
            raise ValueError('bracket must contain exactly two bounds') from exc
        a, b = _real(a, 'bracket lower bound'), _real(b, 'bracket upper bound')
        if a == b:
            raise ValueError('bracket bounds must be different')
        if not np.isfinite(b-a) or not np.isfinite(a+b):
            raise ValueError('bracket bounds are too large for bisection')
        if x0 is not None or x1 is not None:
            raise ValueError('Use bracket or starting guesses, not both')
    else:
        if x0 is None:
            raise ValueError(f'{method} requires x0')
        x0 = _real(x0, 'x0')
    if method == 'secant':
        if x1 is None:
            raise ValueError('secant requires x0 and x1')
        x1 = _real(x1, 'x1')
        if x0 == x1:
            raise ValueError('x0 and x1 must be different')
    elif x1 is not None:
        raise ValueError('x1 is supported only by the secant method')
    if method not in ('newton', 'halley') and (derivative is not None or second_derivative is not None):
        raise ValueError('Derivative arguments require newton or halley')
    if second_derivative is not None and method != 'halley':
        raise ValueError('second_derivative is supported only by halley')
    if method == 'halley' and (derivative is None or second_derivative is None):
        raise ValueError('halley requires derivative and second_derivative')
    function = _ScalarFunction(f)
    if explain:
        from kiwicalc.numeric.explanations import trace_root
        return trace_root(function, derivative, method,
                          (a, b) if method in ('bisection', 'brent') else None,
                          x0, x1, tolerance, max_iterations, trace_limit)
    if method == 'brent':
        from kiwicalc.numeric.advanced import _brent
        return _brent(function, a, b, tolerance, max_iterations, return_info)
    first = _ScalarFunction(derivative) if derivative is not None else None
    second = _ScalarFunction(second_derivative) if second_derivative is not None else None
    last = [None, None]

    def evaluate(x):
        value = function(x)
        last[:] = [float(x), abs(value)]
        if abs(value) <= tolerance:
            # All legacy solvers now obey the same residual stopping criterion,
            # including bisection (whose native epsilon is an interval tolerance).
            raise _RootFound(float(x), abs(value))
        return value

    df = first if first is not None else lambda x: _difference(function, x, 'central', None)
    converged = False
    try:
        # Check exact starting roots before Newton's legacy zero-derivative adjustment.
        if method != 'bisection':
            evaluate(x0)
        if method == 'bisection':
            candidate = roots.bisection_method(evaluate, a, b, epsilon=0, nmax=max_iterations)
        elif method == 'newton':
            candidate = roots.newton_raphson(evaluate, df, x0, epsilon=0, nmax=max_iterations)
        elif method == 'secant':
            candidate = roots.secant_method(evaluate, x0, x1, epsilon=0, nmax=max_iterations)
        elif method == 'halley':
            candidate = roots.halleys_method(evaluate, df, second, x0, epsilon=0, nmax=max_iterations)
        else:
            candidate = roots.steffensen_method(evaluate, x0, epsilon=0, nmax=max_iterations)
        if candidate is not None:
            evaluate(candidate)
    except _RootFound as found:
        last[:] = [found.value, found.residual]
        converged = True
    calls = function.calls + (first.calls if first else 0) + (second.calls if second else 0)
    message = 'Residual tolerance satisfied' if converged else 'Root solver did not satisfy the residual tolerance'
    result = NumericalResult(last[0], method, calls, converged, last[1], message)
    if not converged and not return_info:
        raise RuntimeError(message)
    return result if return_info else result.value


__all__ = ['NumericalResult', 'differentiate', 'integrate', 'find_root']
