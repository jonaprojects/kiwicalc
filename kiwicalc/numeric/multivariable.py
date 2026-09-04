"""Friendly finite-difference, integration, and system-solving interfaces.

Points use the last axis for coordinates. Callbacks are evaluated one point at
a time; unpacked f(x, y) and explicit vector-style f(point) are supported.
"""
from itertools import product
from math import prod

import numpy as np

from kiwicalc.numeric.api import NumericalResult, _count, _method, _positive, _real


def _array(value, name):
    if hasattr(value, 'to_numpy'):
        value = value.to_numpy()
    array = np.asarray(value)
    if array.dtype.kind not in 'iuf':
        raise TypeError(f'{name} must contain real numbers (not booleans or complex values)')
    array = array.astype(float, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must contain finite values')
    return array


def _points(at):
    points = _array(at, 'at')
    if points.ndim < 1 or any(size == 0 for size in points.shape):
        raise ValueError('at must have shape (variables,) or (..., variables), with nonempty axes')
    return points


class _Function:
    """One function or scalar component functions, with a fixed output shape."""

    def __init__(self, f, dimension, argument_style, variables, vector_output=False):
        from kiwicalc.core.interfaces import IExpression
        from kiwicalc.functions.function import Function

        self.style = _method(argument_style, ('unpacked', 'vector'))
        self.calls = 0
        self.vector_output = vector_output
        self.output_shape = None
        components = isinstance(f, (list, tuple))
        if components and (not vector_output or not f):
            raise ValueError('A nonempty sequence of component functions is only valid for vector outputs')
        sources = list(f) if components else [f]
        names = []
        for source in sources:
            if isinstance(source, (IExpression, Function)):
                source_names = list(source.variables)
                if isinstance(source, IExpression):
                    source_names = sorted(source_names)
                for name in source_names:
                    if name not in names:
                        names.append(name)
        if variables is not None:
            if isinstance(variables, str):
                raise TypeError('variables must be a sequence of variable names')
            names = list(variables)
            if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
                raise ValueError('variables must contain unique, nonempty names')
            if len(names) != dimension:
                raise ValueError('variables must match the coordinate dimension')
        elif names and len(names) != dimension:
            raise ValueError('Math-object variables do not match the point; pass variables=... explicitly')
        self.names = names
        self.callbacks = []
        for source in sources:
            if isinstance(source, IExpression):
                if not set(source.variables).issubset(names):
                    raise ValueError('variables does not include every expression variable')
                def callback(point, expression=source):
                    return expression.when(**dict(zip(names, point))).try_evaluate()
            elif isinstance(source, Function):
                if not set(source.variables).issubset(names):
                    raise ValueError('variables does not include every Function variable')
                indices = [names.index(name) for name in source.variables]
                def callback(point, function=source, indices=indices):
                    return function(*(point[i] for i in indices))
            elif callable(source):
                if self.style == 'vector':
                    def callback(point, function=source):
                        return function(point.copy())
                else:
                    def callback(point, function=source):
                        return function(*point)
            else:
                raise TypeError('f must be callable, a math object, or a sequence of scalar functions')
            self.callbacks.append(callback)
        self.components = components

    def __call__(self, point):
        outputs = []
        for callback in self.callbacks:
            self.calls += 1
            value = callback(point)  # Preserve callback exceptions; never probe by retrying.
            outputs.append(_real(value, 'component output') if self.components else value)
        result = outputs if self.components else outputs[0]
        if self.vector_output:
            result = _array(result, 'function output')
            if result.ndim != 1 or result.size == 0:
                raise ValueError('Function must return a nonempty one-dimensional vector')
            if self.output_shape is not None and result.shape != self.output_shape:
                raise ValueError('Function output shape must remain constant')
            self.output_shape = result.shape
            return result
        return _real(result, 'function output')


def _steps(point, step, method='central', order=1):
    if step is None:
        exponent = .25 if order == 2 else 1/3 if method == 'central' else .5
        steps = np.finfo(float).eps**exponent * np.maximum(1., abs(point))
    else:
        steps = _array(step, 'step')
        if steps.ndim == 0:
            steps = np.full(point.shape, float(steps))
        if steps.shape != point.shape or np.any(steps <= 0):
            raise ValueError('step must be positive and scalar or have one value per variable')
    with np.errstate(over='ignore'):
        plus, minus = point + steps, point - steps
    if ((method != 'backward' and (not np.isfinite(plus).all() or np.any(plus == point))) or
            (method != 'forward' and (not np.isfinite(minus).all() or np.any(minus == point)))):
        raise ValueError('step must produce distinct finite coordinates')
    return steps


def _jacobian_at(function, point, step, method='central', base=None):
    steps = _steps(point, step, method)
    if base is None:
        base = function(point)
    columns = []
    for i, h in enumerate(steps):
        upper, lower = point.copy(), point.copy()
        upper[i] += h
        lower[i] -= h
        if method == 'central':
            column = (function(upper) - function(lower)) / (2*h)
        elif method == 'forward':
            column = (function(upper) - base) / h
        else:
            column = (base - function(lower)) / h
        columns.append(column)
    return _array(np.stack(columns, axis=-1), 'derivative')


def _derivatives(f, at, method, step, argument_style, variables, vector, return_info):
    method = _method(method, ('central', 'forward', 'backward'))
    points = _points(at)
    dimension = points.shape[-1]
    function = _Function(f, dimension, argument_style, variables, vector)
    results = [_jacobian_at(function, point, step, method)
               for point in points.reshape(-1, dimension)]
    values = np.stack(results).reshape(points.shape[:-1] + results[0].shape)
    result = NumericalResult(values, method, function.calls)
    return result if return_info else values


def gradient(f, at, *, method='central', step=None, argument_style='unpacked',
             variables=None, return_info=False):
    """Gradient of a scalar function at points shaped (..., variables).

    Default f(x, y); use argument_style='vector' for f(point). Returns an
    ndarray shaped (..., variables), never a new Vector abstraction.
    For expressions, variables=... explicitly selects coordinate order;
    otherwise expression names are sorted, or Function declaration order is used.
    """
    return _derivatives(f, at, method, step, argument_style, variables, False, return_info)


def jacobian(f, at, *, method='central', step=None, argument_style='unpacked',
             variables=None, return_info=False):
    """Jacobian shaped (..., outputs, variables).

    Accepts a vector-valued callback or a list/tuple of scalar component
    functions. Batch axes belong to at; each callback sees only one point.
    """
    return _derivatives(f, at, method, step, argument_style, variables, True, return_info)


def hessian(f, at, *, step=None, argument_style='unpacked', variables=None, return_info=False):
    """Central-difference Hessian of a scalar function, shaped (..., n, n)."""
    points = _points(at)
    dimension = points.shape[-1]
    function = _Function(f, dimension, argument_style, variables)
    results = []
    for point in points.reshape(-1, dimension):
        steps = _steps(point, step, order=2)
        base = function(point)
        matrix = np.empty((dimension, dimension))
        for i, hi in enumerate(steps):
            upper, lower = point.copy(), point.copy()
            upper[i] += hi
            lower[i] -= hi
            matrix[i, i] = (function(upper) - 2*base + function(lower)) / hi**2
            for j in range(i):
                hj = steps[j]
                total = 0.
                for si, sj in product((-1, 1), repeat=2):
                    sample = point.copy()
                    sample[i] += si*hi
                    sample[j] += sj*hj
                    total += si*sj*function(sample)
                matrix[i, j] = matrix[j, i] = total / (4*hi*hj)
        results.append(_array(matrix, 'Hessian'))
    values = np.stack(results).reshape(points.shape[:-1] + (dimension, dimension))
    result = NumericalResult(values, 'central', function.calls)
    return result if return_info else values


def solve_system(f, initial, *, jacobian=None, step=None, tolerance=1e-8,
                 max_iterations=100, argument_style='unpacked', variables=None, return_info=False):
    """Solve a square nonlinear system F(x)=0 with damped Newton iterations.

    f returns a vector, or is a sequence of scalar equations (residuals).
    initial is one coordinate vector. An optional Jacobian callback uses the
    same argument style and returns shape (n, n); otherwise central differences
    are used. Convergence is based on the infinity norm of the residual.
    Numerical failure raises RuntimeError, or returns converged=False when
    return_info=True. Invalid inputs and callback exceptions always propagate.
    """
    point = _points(initial)
    if point.ndim != 1:
        raise ValueError('initial must be a single coordinate vector, not a batch')
    tolerance = _positive(tolerance, 'tolerance')
    max_iterations = _count(max_iterations, 'max_iterations')
    _steps(point, step)
    function = _Function(f, len(point), argument_style, variables, True)
    if jacobian is not None and not callable(jacobian):
        raise TypeError('jacobian must be callable')
    residual = function(point)
    if residual.shape != point.shape:
        raise ValueError('solve_system requires one equation per unknown')
    norm = float(np.linalg.norm(residual, ord=np.inf))
    iterations, jacobian_calls = 0, 0
    message = 'Maximum iterations reached'
    while norm > tolerance and iterations < max_iterations:
        if jacobian is None:
            matrix = _jacobian_at(function, point, step, base=residual)
        else:
            jacobian_calls += 1
            matrix = _array(jacobian(point.copy()) if argument_style == 'vector' else jacobian(*point), 'Jacobian')
            if matrix.shape != (len(point), len(point)):
                raise ValueError('Jacobian must have shape (variables, variables)')
        try:
            delta = np.linalg.solve(matrix, -residual)
        except np.linalg.LinAlgError:
            message = 'Jacobian is singular'
            break
        if not np.isfinite(delta).all():
            message = 'Newton step is not finite'
            break
        accepted = False
        for backtrack in range(20):
            candidate = point + delta * .5**backtrack
            if not np.isfinite(candidate).all():
                continue
            next_residual = function(candidate)
            next_norm = float(np.linalg.norm(next_residual, ord=np.inf))
            if next_norm < norm or next_norm <= tolerance:
                point, residual, norm = candidate, next_residual, next_norm
                accepted = True
                break
        iterations += 1
        if not accepted:
            message = 'Line search stalled without reducing the residual'
            break
    converged = norm <= tolerance
    if converged:
        message = 'Residual tolerance satisfied'
    result = NumericalResult(point.copy(), 'damped_newton', function.calls + jacobian_calls,
                             converged, norm, message, iterations)
    if not converged and not return_info:
        raise RuntimeError(message)
    return result if return_info else result.value


def integrate_nd(f, bounds, *, method='midpoint', intervals=20, max_evaluations=100000,
                 argument_style='unpacked', variables=None, return_info=False):
    """Fixed tensor-grid integration over a finite rectangular domain.

    bounds has shape (variables, 2). intervals is a positive integer or one
    count per axis. Methods: midpoint or trapezoid. Reversed bounds change
    orientation. The grid-size guard prevents accidental exponential work;
    no adaptive convergence or error estimate is claimed.
    """
    method = _method(method, ('midpoint', 'trapezoid'))
    bounds = _array(bounds, 'bounds')
    if bounds.ndim != 2 or bounds.shape[0] == 0 or bounds.shape[1] != 2:
        raise ValueError('bounds must have shape (variables, 2)')
    dimension = len(bounds)
    if np.isscalar(intervals):
        counts = [_count(intervals, 'intervals')] * dimension
    else:
        counts = [_count(value, 'intervals') for value in intervals]
        if len(counts) != dimension:
            raise ValueError('intervals must have one count per variable')
    limit = _count(max_evaluations, 'max_evaluations')
    function = _Function(f, dimension, argument_style, variables)
    widths = bounds[:, 1] - bounds[:, 0]
    if not np.isfinite(widths).all():
        raise ValueError('Integration widths must be finite')
    if np.any(widths == 0):
        result = NumericalResult(0., method, 0)
        return result if return_info else result.value
    sizes = [n + (method == 'trapezoid') for n in counts]
    if prod(sizes) > limit:
        raise ValueError('Integration grid exceeds max_evaluations; reduce intervals or explicitly raise the limit')
    steps = widths / np.asarray(counts)
    total = 0.
    for index in product(*(range(size) for size in sizes)):
        weight = 1.
        if method == 'midpoint':
            point = bounds[:, 0] + (np.asarray(index) + .5)*steps
        else:
            point = bounds[:, 0] + np.asarray(index)*steps
            for axis, i in enumerate(index):
                if i == 0 or i == counts[axis]:
                    weight *= .5
        total += weight * function(point)
    value = _real(float(total * np.prod(steps)), 'integral')
    result = NumericalResult(value, method, function.calls)
    return result if return_info else value


__all__ = ['gradient', 'jacobian', 'hessian', 'solve_system', 'integrate_nd']
