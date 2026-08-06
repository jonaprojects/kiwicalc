from __future__ import annotations
import math
import warnings
from typing import Union, Tuple, List, Optional, Any, Callable
import numpy as np

def reinman(f: Callable, a, b, N: int):
    if N < 2:
        raise ValueError('The method requires N >= 2')
    return sum(((b - a) / (N - 1) * f(value) for value in np.linspace(a, b, N)))

def trapz(f: Callable, a, b, N: int):
    if N == 0:
        raise ValueError('Trapz(): N cannot be 0')
    dx = (b - a) / N
    return 0.5 * dx * sum((f(a + i * dx) + f(a + (i - 1) * dx) for i in range(1, int(N) + 1)))

def simpson(f: Callable, a, b, N: int):
    if N <= 2:
        raise ValueError('The method requires N >= 2')
    dx = (b - a) / (N - 1)
    if N % 2 != 0:
        N += 1
    return dx / 3 * sum((f(a + (2 * i - 2) * dx) + 4 * f(a + (2 * i - 1) * dx) + f(a + 2 * i * dx) for i in range(1, int(N / 2))))

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
