from __future__ import annotations
import math
import warnings
from typing import Union, Tuple, List, Optional, Any, Callable
import numpy as np

def reinman(f: Callable, a, b, N: int):
    """Midpoint Riemann sum over N-1 equal intervals (legacy spelling).

    N denotes the number of boundary points, as in the original API.
    Each interval is counted once; its function value is sampled at its midpoint.
    """
    _integration_count(N, 2)
    dx = (b - a) / (N - 1)
    return dx * sum(f(a + (i + 0.5) * dx) for i in range(N - 1))

def trapz(f: Callable, a, b, N: int):
    if N == 0:
        raise ValueError('Trapz(): N cannot be 0')
    dx = (b - a) / N
    return 0.5 * dx * sum((f(a + i * dx) + f(a + (i - 1) * dx) for i in range(1, int(N) + 1)))

def simpson(f: Callable, a, b, N: int):
    """Composite Simpson integration using N sample points.

    Even N is increased by one before computing the grid, ensuring an even
    number of intervals and coverage of the complete integration domain.
    Each sample is evaluated only once; scalar callables remain supported.
    """
    _integration_count(N, 3)
    if N % 2 == 0:
        N += 1
    dx = (b - a) / (N - 1)
    odd = sum(f(a + i * dx) for i in range(1, N - 1, 2))
    even = sum(f(a + i * dx) for i in range(2, N - 1, 2))
    return dx / 3 * (f(a) + 4 * odd + 2 * even + f(b))


def _integration_count(value, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'N must be an integer >= {minimum}')

def numerical_diff(f, a, method='central', h=0.01):
    if method == 'central':
        return (f(a + h) - f(a - h)) / (2 * h)
    elif method == 'forward':
        return (f(a + h) - f(a)) / h
    elif method == 'backward':
        return (f(a) - f(a - h)) / h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")

def gradient_descent(f_tag: Callable, initial_value, learning_rate: float=0.01, precision: float=1e-06, nmax=10000):
    previous_step_size = 1
    for i in range(nmax):
        if previous_step_size <= precision:
            return initial_value
        new_value = initial_value - learning_rate * f_tag(initial_value)
        previous_step_size = abs(new_value - initial_value)
        initial_value = new_value
    warnings.warn('Reached maximum limit of iterations! Result might be inaccurate!')
    return initial_value

def gradient_ascent(f_tag: Callable, initial_value, learning_rate: float=0.01, precision: float=1e-06, nmax=10000):
    previous_step_size = 1
    for i in range(nmax):
        if previous_step_size <= precision:
            return initial_value
        new_value = initial_value + learning_rate * f_tag(initial_value)
        previous_step_size = abs(new_value - initial_value)
        initial_value = new_value
    warnings.warn('Reached maximum limit of iterations! Result might be inaccurate!')
    return initial_value
