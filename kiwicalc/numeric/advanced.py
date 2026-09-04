"""Array differentiation, safeguarded root finding, and adaptive quadrature."""
import heapq
import itertools

import numpy as np

from kiwicalc.numeric.api import NumericalResult, _ScalarFunction, _count, _method, _positive, _real


def _array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'iuf':
        raise TypeError(f'{name} must contain real numbers')
    array = array.astype(float, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite')
    return array


def _differentiate(f, at, method, step, vectorized, return_info):
    method = _method(method, ('central', 'forward', 'backward', 'richardson'))
    if not isinstance(vectorized, bool):
        raise TypeError('vectorized must be a boolean')
    points = _array(at, 'at')
    if points.size == 0:
        raise ValueError('at must not be empty')
    power = .2 if method == 'richardson' else 1/3 if method == 'central' else .5
    steps = np.finfo(float).eps**power * np.maximum(1., abs(points)) if step is None else _array(step, 'step')
    try:
        steps = np.broadcast_to(steps, points.shape)
    except ValueError as exc:
        raise ValueError('step must broadcast to the shape of at') from exc
    if np.any(steps <= 0):
        raise ValueError('step must be positive')
    smallest = steps / 2 if method == 'richardson' else steps
    for sign in ((1,) if method == 'forward' else (-1,) if method == 'backward' else (-1, 1)):
        with np.errstate(over='ignore'):
            sample = points + sign * steps
        if not np.isfinite(sample).all() or np.any(points + sign * smallest == points):
            raise ValueError('step must produce distinct finite sample points')
    function = _ScalarFunction(f)
    calls = 0

    def evaluate(x):
        nonlocal calls
        if not vectorized:
            return np.asarray([function(float(v)) for v in x.flat]).reshape(x.shape)
        calls += 1
        value = _array(function.function(x.copy()), 'function output')
        if value.shape != x.shape:
            raise ValueError('vectorized function output must match the shape of at')
        return value

    if method in ('central', 'richardson'):
        coarse = (evaluate(points + steps) - evaluate(points - steps)) / (2 * steps)
        if method == 'richardson':
            fine = (evaluate(points + steps/2) - evaluate(points - steps/2)) / steps
            correction = (fine - coarse) / 3
            value, error = fine + correction, abs(correction)
        else:
            value, error = coarse, None
    elif method == 'forward':
        value, error = (evaluate(points + steps) - evaluate(points)) / steps, None
    else:
        value, error = (evaluate(points) - evaluate(points - steps)) / steps, None
    value = _array(value, 'derivative')
    if value.ndim == 0:
        value = float(value)
        error = None if error is None else float(error)
    result = NumericalResult(value, method, function.calls + calls, estimated_error=error)
    return result if return_info else value


def _brent(f, a, b, tolerance, max_iterations, return_info):
    """Brent-Dekker: inverse interpolation safeguarded by bisection."""
    fa, fb = f(a), f(b)
    if abs(fa) <= tolerance:
        b, fb = a, fa
    elif abs(fb) > tolerance and np.signbit(fa) == np.signbit(fb):
        raise ValueError('bracket endpoints must have opposite signs')
    if abs(fa) < abs(fb):
        a, b, fa, fb = b, a, fb, fa
    c, fc, d = a, fa, a
    bisected = True
    iterations = 0
    while abs(fb) > tolerance and iterations < max_iterations:
        scale = max(abs(fa), abs(fb), abs(fc))
        ya, yb, yc = fa/scale, fb/scale, fc/scale
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            if ya != yc and yb != yc and ya != yb:
                s = (a * (yb/(ya-yb)) * (yc/(ya-yc))
                     + b * (ya/(yb-ya)) * (yc/(yb-yc))
                     + c * (ya/(yc-ya)) * (yb/(yc-yb)))
            else:
                s = b - yb * ((b-a)/(yb-ya)) if yb != ya else np.nan
        bound = a + .25*(b-a)
        minimum = np.finfo(float).eps * max(1., abs(b))
        if (not np.isfinite(s) or not min(bound, b) < s < max(bound, b)
                or abs(s-b) >= abs(b-c if bisected else c-d)/2
                or abs(b-c if bisected else c-d) < minimum):
            s, bisected = a + (b-a)/2, True
        else:
            bisected = False
        if s == a or s == b:
            break
        fs = f(s)
        iterations += 1
        d, c, fc = c, b, fb
        if np.signbit(fa) != np.signbit(fs):
            b, fb = s, fs
        else:
            a, fa = s, fs
        if abs(fa) < abs(fb):
            a, b, fa, fb = b, a, fb, fa
    converged = abs(fb) <= tolerance
    message = 'Residual tolerance satisfied' if converged else 'Root solver did not satisfy the residual tolerance'
    result = NumericalResult(b, 'brent', f.calls, converged, abs(fb), message, iterations)
    if not converged and not return_info:
        raise RuntimeError(message)
    return result if return_info else b


def _adaptive_simpson(f, a, b, tolerance, max_evaluations, max_depth, return_info):
    tolerance = _positive(tolerance, 'tolerance')
    limit = _count(max_evaluations, 'max_evaluations')
    depth_limit = _count(max_depth, 'max_depth')
    if limit < 5:
        raise ValueError('max_evaluations must be at least 5')
    if a == b:
        result = NumericalResult(0., 'adaptive_simpson', 0, True, estimated_error=0.)
        return result if return_info else 0.
    orientation = 1 if b > a else -1
    a, b = min(a, b), max(a, b)
    serial = itertools.count()

    def panel(left, right, fl, fm, fr, depth):
        middle = left + (right-left)/2
        lmid, rmid = left + (middle-left)/2, middle + (right-middle)/2
        if lmid in (left, middle) or rmid in (middle, right):
            coarse = (right-left)/6 * (fl + 4*fm + fr)
            return (float('inf'), coarse, (left, right, fl, fm, fr, None, None, depth))
        fq1, fq3 = f(lmid), f(rmid)
        coarse = (right-left)/6 * (fl + 4*fm + fr)
        fine = (right-left)/12 * (fl + 4*fq1 + 2*fm + 4*fq3 + fr)
        correction = (fine-coarse)/15
        value = _real(fine + correction, 'integral')
        return (abs(correction), value, (left, right, fl, fm, fr, fq1, fq3, depth))

    middle = a + (b-a)/2
    initial = panel(a, b, f(a), f(middle), f(b), 0)
    heap = [(-initial[0], next(serial), initial)]
    total, error, iterations = initial[1], initial[0], 0
    message = 'Estimated error tolerance satisfied'
    while error > tolerance:
        _, _, worst = heap[0]
        old_error, old_value, data = worst
        left, right, fl, fm, fr, fq1, fq3, depth = data
        if depth >= depth_limit or fq1 is None or f.calls + 4 > limit:
            message = 'Adaptive Simpson exhausted its evaluation, depth, or floating-point resolution limit'
            break
        heapq.heappop(heap)
        middle = left + (right-left)/2
        children = [panel(left, middle, fl, fq1, fm, depth+1),
                    panel(middle, right, fm, fq3, fr, depth+1)]
        total += sum(child[1] for child in children) - old_value
        error = max(0., error - old_error + sum(child[0] for child in children))
        for child in children:
            heapq.heappush(heap, (-child[0], next(serial), child))
        iterations += 1
    converged = error <= tolerance
    result = NumericalResult(_real(orientation*total, 'integral'), 'adaptive_simpson', f.calls,
                             converged, message=message, iterations=iterations, estimated_error=error)
    if not converged and not return_info:
        raise RuntimeError(message)
    return result if return_info else result.value


def _samples(values, x, spacing, axis):
    values = _array(values, 'values')
    if values.ndim == 0:
        raise ValueError('values must have a sample axis')
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise TypeError('axis must be an integer')
    if not -values.ndim <= axis < values.ndim:
        raise ValueError('axis is out of range')
    axis %= values.ndim
    n = values.shape[axis]
    if n < 2:
        raise ValueError('At least two samples are required')
    spacing = _positive(spacing, 'spacing')
    if x is None:
        x = np.arange(n, dtype=float) * spacing
    else:
        if spacing != 1.:
            raise ValueError('Specify x or spacing, not both')
        x = _array(x, 'x')
        if x.shape != (n,):
            raise ValueError('x must be one-dimensional and match the sample axis')
    differences = np.diff(x)
    if not np.isfinite(differences).all() or not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError('Sample coordinates must be finite and strictly monotonic')
    return values, x, axis


def differentiate_samples(values, x=None, *, spacing=1., axis=-1, edge_order=2):
    """Differentiate sampled data, including nonuniform or decreasing x.

    Uses three-point interior differences. With only two samples, both results
    are the secant slope. Otherwise edge_order is 1 or 2. Shape is preserved.
    """
    values, x, axis = _samples(values, x, spacing, axis)
    if isinstance(edge_order, bool) or edge_order not in (1, 2):
        raise ValueError('edge_order must be 1 or 2')
    result = np.gradient(values, x, axis=axis, edge_order=min(edge_order, values.shape[axis]-1))
    return _array(result, 'derivative')


def cumulative_integrate(values, x=None, *, spacing=1., axis=-1, initial=0.):
    """Cumulative trapezoidal integral with the same shape as values.

    The first output is initial (an additive integration constant). Supports
    nonuniform/decreasing coordinates and multidimensional sampled signals.
    """
    values, x, axis = _samples(values, x, spacing, axis)
    initial = _real(initial, 'initial')
    moved = np.moveaxis(values, axis, -1)
    areas = (moved[..., :-1]/2 + moved[..., 1:]/2) * np.diff(x)
    result = np.empty_like(moved)
    result[..., 0] = initial
    result[..., 1:] = initial + np.cumsum(areas, axis=-1)
    return _array(np.moveaxis(result, -1, axis), 'integral')


__all__ = ['differentiate_samples', 'cumulative_integrate']
