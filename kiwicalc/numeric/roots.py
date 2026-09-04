from __future__ import annotations
import math
from math import sin, cos, pi, e
import cmath
import warnings
from sys import exc_info
from functools import reduce
from typing import Union, Tuple, List, Optional, Any, Callable, Iterator
import numpy as np

from kiwicalc.core.utils import is_lambda, round_decimal, max_power
from kiwicalc.expressions.poly import Poly, synthetic_division
from kiwicalc.parsing.parse_expression import monic_poly_from_coefficients

def get_factors(n):
    if n == 0:
        return {}
    factors = set(reduce(list.__add__, ([i, n // i] for i in range(1, int(abs(n) ** 0.5) + 1) if n % i == 0)))
    return factors.union({-number for number in factors})

def extract_possible_solutions(most_significant_coef: float, free_number: float):
    p_factors = get_factors(free_number)
    q_factors = get_factors(most_significant_coef)
    possible_solutions = []
    for p_factor in p_factors:
        for q_factor in q_factors:
            if q_factor != 0:
                possible_solutions.append(p_factor / q_factor)
    if possible_solutions:
        return set(possible_solutions)
    return {}

def __find_solutions(coefficients, possible_solutions):
    from kiwicalc.equations.single import solve_polynomial, solve_quadratic_real
    solutions = set()
    copy = coefficients.copy()
    if copy[-1] == 0:
        solutions.add(0)
        for i in range(len(copy) - 1, 0, -1):
            if copy[i] != 0:
                break
            del copy[i]
        solutions.add(solve_polynomial(copy))
    for solution in possible_solutions:
        division_result, remainder = synthetic_division(coefficients, solution)
        if remainder != 0:
            continue
        solutions.add(solution)
        if len(division_result) >= 4:
            for sol in solve_polynomial(division_result):
                solutions.add(sol)
        elif len(division_result) == 3:
            try:
                values = solve_quadratic_real(division_result[0], division_result[1], division_result[2])
                solutions.add(values[0])
                solutions.add(values[1])
            except Exception as e:
                warnings.warn(f'Due to an {e.__class__} error in line {exc_info()[-1].tb_lineno}, some solutions might be missing ! ')
        else:
            print('Whoops! it seems something went wrong')
    return solutions

def newton_raphson(f_0: Callable, f_1: Callable, initial_value: float=0, epsilon=1e-05, nmax: int=100000):
    """
    The Newton-Raphson method is a root-finding algorithm, introduced in the 17th century and named after
    Almighty god Issac Newton and the mathematician Joseph Raphson.
    In each iteration, the functions provides a better approximation for the root.


    :param f_0: The origin function. Must be callable and return a float

    :param f_1: The derivative function, Must be callable and return a float

    :param initial_value: The initial value to start the approximation with. Different initial values may lead to different results. For example, if a function intersects with the x axis in (5,0) and (-2,0), an initial_value of 4 will lead to 5, while an initial value of -1 will lead to -2.

    :param epsilon: We want to find out an x value, that its y is near 0. Epsilon determines the difference in which y is considered as 0. For example, if the y value is 0.0000001, in most cases it's negligible

    :return: Returns the closest root of the function to the initial value ( if the function has any roots)
    :rtype: float
    """
    if f_1(initial_value) == 0:
        initial_value += 0.1
    for i in range(nmax):
        f_x = f_0(initial_value)
        if abs(f_x) <= epsilon:
            return initial_value
        f_tag = f_1(initial_value)
        if f_tag == 0:
            warnings.warn('Newton-Raphson failed because the derivative is zero')
            return initial_value
        initial_value -= f_x / f_tag
    warnings.warn('The solution might have not converged properly')
    return initial_value

def halleys_method(f_0: Callable, f_1: Callable, f_2: Callable, initial_value: float, epsilon: float=1e-05, nmax: int=100000):
    """
    Halleys method is a root-finding algorithm which is derived from Newton's method. Unlike newton's method,
    it requires also the second derivative of the function, in addition the first derivative. However, it usually
    converges to the root faster. This method finds only 1 root in each call, depending on the initial value.

    :param f_0: The function. f(x)
    :param f_1: The first derivative. f'(x)
    :param f_2: The second derivative f''(x)
    :param initial_value: The initial guess for the approximation.
    :param epsilon: Epsilon determines how close can a dot be to the x axis, to be considered a root.
    :return: Returns the approximation of the root.
    """
    current_x = initial_value
    for i in range(nmax):
        f = f_0(current_x)
        if abs(f) < epsilon:
            return current_x
        f_prime = f_1(current_x)
        f_double_prime = f_2(current_x)
        denominator = 2 * f_prime ** 2 - f * f_double_prime
        if denominator == 0:
            warnings.warn("Halley's method failed because its update denominator is zero")
            return current_x
        current_x = current_x - 2 * f * f_prime / denominator
    warnings.warn('The solution might have not converged properly')
    return current_x

def secant_method(f: Callable, n_0: float=1, n_1: float=0, epsilon: float=1e-05, nmax=100000):
    """
    The secant method is a root-finding algorithm.

    :param f:
    :param n_0:
    :param n_1:
    :param epsilon:
    :return:
    """
    f_0 = f(n_0)
    if abs(f_0) <= epsilon:
        return n_0
    f_1 = f(n_1)
    if abs(f_1) <= epsilon:
        return n_1
    for _ in range(nmax):
        denominator = f_0 - f_1
        if denominator == 0:
            warnings.warn('The secant method failed because the function values are equal')
            return n_0
        d = (n_0 - n_1) / denominator * f_0
        n_1 = n_0
        n_0 -= d
        if abs(d) <= epsilon:
            return n_0
        f_1 = f_0
        f_0 = f(n_0)
        if abs(f_0) <= epsilon:
            return n_0
    warnings.warn('The solution might have not converged properly')
    return n_0

def inverse_interpolation(f: Callable, x0: float, x1: float, x2: float, epsilon: float=1e-05, nmax: int=100000):
    """
    Quadratic Inverse Interpolation is a root-finding algorithm, that requires a function and 3 arguments.
    Unlike other methods, like Newton-Raphson, and Halley's method, it does not require computing
    the derivative of the function.

    :param f:
    :param x0:
    :param x1:
    :param x2:
    :param epsilon:
    :return:
    """
    for _ in range(nmax):
        if abs(f(x2)) <= epsilon:
            return x2
        x3 = f(x2) * f(x1) / ((f(x0) - f(x1)) * (f(x0) - f(x2))) * x0
        x3 += f(x0) * f(x2) / ((f(x1) - f(x0)) * (f(x1) - f(x2))) * x1
        x3 += f(x0) * f(x1) / ((f(x2) - f(x0)) * (f(x2) - f(x1))) * x2
        x0 = x1
        x1 = x2
        x2 = x3
    warnings.warn('The result might be inaccurate. Try entering different parameters or using different methods')
    return x2

def laguerre_method(f_0: Callable, f_1: Callable, f_2: Callable, x0: float, n: float, epsilon: float=1e-05, nmax=100000):
    """
    Laguerre's method is a root-finding algorithm,

    :param f_0: The polynomial function.
    :param f_1: The first derivative of the function
    :param f_2: The second derivative of the function
    :param x0: An initial value
    :param n: The degree of the polynomial
    :param epsilon: Determines when a y value of the approximation is small enough to be rounded to 0 and thus considered as a root.
    :param nmax:
    :return: An approximation of a single root of the function.
    """
    xk = x0
    for _ in range(nmax):
        if abs(f_0(xk)) <= epsilon:
            return xk
        G = f_1(xk) / f_0(xk)
        H = G ** 2 - f_2(xk) / f_0(xk)
        root = cmath.sqrt((n - 1) * (n * H - G ** 2))
        d = max((G + root, G - root), key=abs)
        a = n / d
        xk -= a
    warnings.warn('The solution might be inaccurate due to insufficient convergence.')
    return xk

def get_bounds(degree: int, coefficients):
    upper = 1 + 1 / abs(coefficients[-1]) * max((abs(coefficients[x]) for x in range(degree)))
    lower = abs(coefficients[0]) / (abs(coefficients[0]) + max((abs(coefficients[x]) for x in range(1, degree + 1))))
    return (upper, lower)

def __aberth_approximations(coefficients):
    n = len(coefficients) - 1
    radius = 1 + max(abs(value / coefficients[0]) for value in coefficients[1:])
    return np.asarray([radius * np.exp(2j * pi * (index + .37) / n)
                       for index in range(n)], dtype=np.clongdouble)

def __durandKerner_approximations(coefficients):
    n = len(coefficients) - 1
    if coefficients[0] == 0:
        return [0 for _ in range(n)]
    radius = 1 + max((abs(coefficient) for coefficient in coefficients))
    return [complex(radius * cos(angle), radius * sin(angle)) for angle in np.linspace(0, 2 * pi, n)]

def durand_kerner(f_0: Callable, coefficients, epsilon=1e-05, nmax=5000):
    """
    The Durand-Kerner method, also known as the Weierstrass method is an iterative approach for finding all of the
    real and complex roots of a polynomial.
    It was first discovered by the German mathematician Karl Weierstrass in 1891, and was later discovered by
    Durand(1960) and Kerner (1966). This method requires the function and a collection of its coefficients.
    If you wish to enter only the coefficients, import and use the method durand_kerner2().

    :param f_0: The function.
    :param coefficients: A Sized and Iterable collection of the coefficients of the function
    :param epsilon:
    :param nmax: the max number of iterations allowed. default is 5000, but it can be changed manually.
    :return: Returns a set of the approximations of the root of the function.
    """
    if coefficients[0] != 1:
        coefficients = [coefficient / coefficients[0] for coefficient in coefficients]
        f_0 = monic_poly_from_coefficients(coefficients).to_lambda()
    else:
        coefficients = [coefficient for coefficient in coefficients]
    current_guesses = __durandKerner_approximations(coefficients)
    for i in range(nmax):
        if all((abs(f_0(current_guess)) < epsilon for current_guess in current_guesses)):
            return {complex(round_decimal(c.real), round_decimal(c.imag)) for c in current_guesses}
        for index in range(len(current_guesses)):
            numerator = f_0(current_guesses[index])
            other_guesses = (guess for j, guess in enumerate(current_guesses) if j != index)
            denominator = reduce(lambda a, b: a * b, (current_guesses[index] - guess for guess in other_guesses))
            current_guesses[index] -= numerator / denominator
    return {complex(round_decimal(c.real), round_decimal(c.imag)) for c in current_guesses}

def durand_kerner2(coefficients, epsilon=0.0001, nmax=5000):
    if coefficients[0] != 1:
        coefficients = [coefficient / coefficients[0] for coefficient in coefficients]
    else:
        coefficients = [coefficient for coefficient in coefficients]
    executable_lambda = monic_poly_from_coefficients(coefficients).to_lambda()
    return durand_kerner(executable_lambda, coefficients, epsilon, nmax)

def negligible_complex(expression: complex, epsilon) -> bool:
    return abs(expression.real) < epsilon and abs(expression.imag) < epsilon

def ostrowski_method(f_0: Callable, f_1: Callable, initial_value, epsilon: float=1e-05, nmax: int=100000):
    """ A root finding algorithm with a convergence rate of 3. Finds a single real root."""
    if f_1(initial_value) == 0:
        initial_value += 0.1
    for i in range(nmax):
        f_x = f_0(initial_value)
        if abs(f_x) < epsilon:
            return initial_value
        f_tag = f_1(initial_value)
        y = initial_value - f_x / f_tag
        f_y = f_0(y)
        initial_value = y - f_y * (y - initial_value) / (2 * f_y - f_x)
    return initial_value

def chebychevs_method(f_0: Callable, f_1: Callable, f_2: Callable, initial_value, epsilon: float=1e-05, nmax: int=100000):
    if f_1(initial_value) == 0:
        initial_value += 0.1
    for i in range(nmax):
        f_x = f_0(initial_value)
        if abs(f_x) < epsilon:
            return initial_value
        f_tag = f_1(initial_value)
        f_tag_tag = f_2(initial_value)
        initial_value -= f_x / f_tag * (1 + f_x * f_tag_tag / (2 * f_tag ** 2))
    warnings.warn('The solution might have not converged properly')
    return initial_value

def aberth_method(f_0: Callable, f_1: Callable, coefficients, epsilon: float=1e-06, nmax: int=100000) -> set:
    """
    Aberth-Erlich method is a root-finding algorithm, developed in 1967 Oliver Aberth, and later improved
    in the seventies by Louis W. Ehrlich.
    It finds all of the roots of a function - both real and complex, except some special cases.
    Callbacks must evaluate the polynomial and derivative at complex inputs.
    Convergence requires relative updates and coefficient-scaled residuals.
    Root accuracy can be much poorer for repeated or clustered roots.
    Invalid coefficients raise ValueError; exhausted iterations or numerical
    breakdown raise RuntimeError rather than silently returning incomplete roots.
    Results are not rounded or merged by distance. The historical set return
    type is retained, so multiplicity is not represented reliably.

    :param f_0: The origin function. f(x).
    :param f_1: The first derivative. f'(x)
    :param coefficients: Polynomial coefficients from highest power to constant.
    :return: Returns a set of all of the different solutions.
    """
    if not callable(f_0) or not callable(f_1):
        raise TypeError('f_0 and f_1 must be callable')
    if isinstance(coefficients, (str, bytes)):
        raise ValueError('coefficients must be a finite one-dimensional sequence')
    values = np.asarray(list(coefficients), dtype=np.clongdouble)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('coefficients must be a finite one-dimensional sequence')
    nonzero = np.flatnonzero(values)
    if not len(nonzero):
        raise ValueError('The zero polynomial has infinitely many roots')
    values = values[nonzero[0]:]
    if not np.isscalar(epsilon) or not np.isreal(epsilon) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be positive and finite')
    if isinstance(nmax, (bool, np.bool_)) or not isinstance(nmax, (int, np.integer)) or nmax <= 0:
        raise ValueError('nmax must be a positive integer')
    if len(values) == 1:
        return set()
    if len(values) == 2:
        return {complex(-values[1] / values[0])}
    scale = max(abs(values))
    normalized = values / scale
    guesses = np.asarray(__aberth_approximations(normalized), dtype=np.clongdouble)
    for _ in range(nmax):
        offsets = np.zeros_like(guesses)
        for k, z in enumerate(guesses):
            f = f_0(z) / scale
            derivative = f_1(z) / scale
            if not np.isfinite(f) or not np.isfinite(derivative):
                raise RuntimeError('Aberth encountered a non-finite function or derivative value')
            if f == 0:
                continue
            differences = z - np.delete(guesses, k)
            if np.any(differences == 0):
                raise RuntimeError('Aberth iterates collided; polynomial roots may be ill-conditioned')
            denominator = derivative - f * np.sum(1 / differences)
            if denominator == 0:
                offsets[k] = epsilon * (1 + abs(z)) * (1+1j)
            else:
                offsets[k] = f / denominator
        guesses -= offsets
        if not np.isfinite(guesses).all():
            raise RuntimeError('Aberth produced non-finite root approximations')
        if np.all(abs(offsets) <= epsilon * np.maximum(1, abs(guesses))):
            residuals = np.asarray([abs(f_0(z) / scale) for z in guesses])
            bounds = np.polyval(abs(normalized), abs(guesses))
            if np.all(residuals <= epsilon * bounds):
                return {complex(z) for z in guesses}
    raise RuntimeError('Aberth did not converge; increase nmax or use another polynomial solver')

def steffensen_method(f: Callable, initial: float, epsilon: float=1e-06, nmax=100000):
    """
    The Steffensen method is a root-finding algorithm, named after the Danish mathematician Johan Frederik Steffensen.
    It is considered similar to the Newton-Raphson method, and in some implementations it achieves quadratic
    convergence. Unlike many other methods, the Steffensen method doesn't require more than one initial value nor
    computing derivatives. This might be an advantage if it's difficult to compute a derivative of a function.


    :param f: The origin function. Every suitable callable will be accepted, including lambda expressions.
    :param initial: The initial guess. Should be very close to the actual root.
    :param epsilon:
    :return: returns an approximation of the root.
    """
    x = initial
    for _ in range(nmax):
        fx = f(x)
        if abs(fx) < epsilon:
            break
        gx = f(x + fx) / fx - 1
        if gx == 0:
            warnings.warn('Failed using the steffensen method!')
            return x
        x -= fx / gx
    else:
        warnings.warn('The solution might have not converged properly')
    return x

def bisection_method(f: Callable, a: float, b: float, epsilon: float=1e-05, nmax: int=10000):
    """
    The bisection method is a root-finding algorithm, namely, its purpose is to find the zeros of a function.
    For it to work, the function must be continuous, and it must receive two different x values, that their y values
    have opposite signs.

    For example, For the function f(x) = x^2 - 5*x :
    We can choose for example the values 3 and 10.

    f(3) = 3^2 - 5*3 = -6 (The sign is NEGATIVE)
    f(10) =  10^2 - 5*10 = 50 ( The sign is POSITIVE )

    When ran, the bisection will find the root 5.0 ( approximately ) .

    This implementation only supports real roots. See Durand-Kerner / Aberth method for complex

    values as well.
    :param f: The function entered
    :param a:  x value of the function
    :param b: another x value of the function, that its corresponding y value has a different sign than the former.
    :param epsilon:
    :param nmax: The maximum number of iterations
    :return: Returns an approximation of a root of the function, if successful.
    """
    if a > b:
        a, b = (b, a)
    elif a == b:
        raise ValueError('a and b cannot be equal! a must be smaller than b')
    fa, fb = (f(a), f(b))
    if abs(fa) <= epsilon:
        return a
    if abs(fb) <= epsilon:
        return b
    if not (fa < 0 < fb or fb < 0 < fa):
        raise ValueError('a and b must be of opposite signs')
    for i in range(nmax):
        c = (a + b) / 2
        fc = f(c)
        if fc == 0 or (b - a) / 2 < epsilon:
            return c
        if fc * fa > 0:
            a = c
            fa = fc
        else:
            b = c
    return None

def bairstow_method(coefficients, epsilon=1e-12, nmax=1000, *, r=0.0, s=-1.0):
    """Find the real and complex roots of a real polynomial.

    ``coefficients`` are ordered from highest power to constant, e.g.
    ``[1, 0, -1]`` represents ``x**2 - 1``. The returned list preserves
    multiplicity (its order is unspecified). Leading zeros are ignored; a
    nonzero constant has no roots and the zero polynomial is rejected.

    Bairstow iteration refines factors ``x**2 - r*x - s`` using real
    synthetic division and a two-variable Newton update, then deflates the
    polynomial. ``epsilon`` is a relative coefficient-remainder tolerance,
    not a guarantee on root accuracy, especially for repeated roots.
    ``nmax`` bounds Newton iterations per quadratic factor, including
    deterministic restarts. Failure raises RuntimeError rather than returning
    an incomplete set of roots. Inputs are never modified.

    Reference: https://dlmf.nist.gov/3.8#iv (Bairstow's method).
    """
    if isinstance(coefficients, (str, bytes)):
        raise TypeError('coefficients must be an iterable of real numbers')
    try:
        raw = np.asarray(list(coefficients))
    except TypeError as exc:
        raise TypeError('coefficients must be an iterable of real numbers') from exc
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError('coefficients must be a nonempty one-dimensional sequence')
    if np.iscomplexobj(raw):
        raise ValueError('Bairstow requires real coefficients')
    try:
        values = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError('coefficients must contain real numbers') from exc
    if not np.isfinite(values).all():
        raise ValueError('coefficients must be finite')
    for name, value in (('epsilon', epsilon), ('r', r), ('s', s)):
        if isinstance(value, (bool, complex)) or not np.isscalar(value):
            raise ValueError(f'{name} must be a finite real number')
        try:
            finite = np.isfinite(float(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{name} must be a finite real number') from exc
        if not finite:
            raise ValueError(f'{name} must be a finite real number')
    epsilon, r, s = float(epsilon), float(r), float(s)
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    if isinstance(nmax, bool) or not isinstance(nmax, (int, np.integer)) or nmax <= 0:
        raise ValueError('nmax must be a positive integer')
    nonzero = np.flatnonzero(values)
    if not len(nonzero):
        raise ValueError('The zero polynomial has infinitely many roots')
    values = values[nonzero[0]:].copy()
    roots = []
    while len(values) > 1 and values[-1] == 0:
        roots.append(0.0)
        values = values[:-1]
    if len(values) == 1:
        return roots
    values /= np.max(np.abs(values))
    rng = np.random.default_rng(0)
    while len(values) > 3:
        # Several short Newton runs avoid stagnation at singular Jacobians.
        radius = max(1.0, abs(values[-1] / values[0]) ** (1 / (len(values) - 1)))
        current_r, current_s = r, s
        run_length = max(1, nmax // 10)
        converged = False
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            for iteration in range(nmax):
                if iteration and iteration % run_length == 0:
                    current_r, current_s = rng.uniform(-2, 2, 2) * (radius, radius**2)
                b, jacobian = _bairstow_division(values, current_r, current_s)
                error = np.max(np.abs(b[-2:]))
                if np.isfinite(error) and error <= epsilon * np.max(np.abs(values)):
                    converged = True
                    break
                try:
                    correction = np.linalg.solve(jacobian, -b[-2:])
                except np.linalg.LinAlgError:
                    correction = np.full(2, np.nan)
                accepted = False
                if np.isfinite(correction).all():
                    for backtrack in range(16):
                        trial_r, trial_s = np.array((current_r, current_s)) + correction * 0.5**backtrack
                        trial_b, _ = _bairstow_division(values, trial_r, trial_s)
                        trial_error = np.max(np.abs(trial_b[-2:]))
                        if np.isfinite(trial_error) and trial_error < error:
                            current_r, current_s = trial_r, trial_s
                            accepted = True
                            break
                if not accepted:
                    current_r, current_s = rng.uniform(-2, 2, 2) * (radius, radius**2)
            # Permit convergence on the last allowed update.
            if not converged:
                b, _ = _bairstow_division(values, current_r, current_s)
                converged = np.isfinite(b).all() and np.max(np.abs(b[-2:])) <= epsilon * np.max(np.abs(values))
        if not converged:
            raise RuntimeError(
                f'Bairstow did not converge for the remaining degree-{len(values) - 1} '
                'polynomial; increase nmax or try different r and s guesses'
            )
        roots.extend(_bairstow_quadratic(1.0, -current_r, -current_s))
        values = b[:-2]
        values = values / np.max(np.abs(values))
    if len(values) == 3:
        roots.extend(_bairstow_quadratic(*values))
    elif len(values) == 2:
        roots.append(float(-values[1] / values[0]))
    return roots


def _bairstow_division(coefficients, r, s):
    """Synthetic coefficients and derivatives of the last two coefficients."""
    b = np.zeros(len(coefficients))
    dr, ds = np.zeros_like(b), np.zeros_like(b)
    for i, coefficient in enumerate(coefficients):
        b[i] = coefficient
        if i >= 1:
            b[i] += r * b[i - 1]
            dr[i] += b[i - 1] + r * dr[i - 1]
            ds[i] += r * ds[i - 1]
        if i >= 2:
            b[i] += s * b[i - 2]
            dr[i] += s * dr[i - 2]
            ds[i] += b[i - 2] + s * ds[i - 2]
    return b, np.column_stack((dr[-2:], ds[-2:]))


def _bairstow_quadratic(a, b, c):
    """Cancellation-resistant quadratic roots, retaining multiplicity."""
    scale = max(abs(a), abs(b), abs(c))
    a, b, c = a / scale, b / scale, c / scale
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        imaginary = math.sqrt(-discriminant) / (2 * a)
        real = -b / (2 * a)
        return [complex(real, imaginary), complex(real, -imaginary)]
    q = -0.5 * (b + math.copysign(math.sqrt(discriminant), b))
    if q == 0:
        return [0.0, 0.0]
    return [float(q / a), float(c / q)]
