"""
Numerical methods for approximating solutions to equations and system of equations
"""

# Built in imports
from typing import Callable
import cmath
import warnings
import functools
# Kiwicalc imports
from auxiliary import round_decimal


def newton_raphson(f_0: Callable, f_1: Callable, initial_value: float = 0, epsilon=0.00001, nmax: int = 100000):
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
    if f_1(initial_value) == 0:  # Avoiding zero division error
        initial_value += 0.1
    for i in range(nmax):
        f_x = f_0(initial_value)
        if abs(f_x) <= epsilon:
            return initial_value
        f_tag = f_1(initial_value)
        initial_value -= f_x / f_tag
    warnings.warn("The solution might have not converged properly")
    return initial_value


def halleys_method(f_0: Callable, f_1: Callable, f_2: Callable, initial_value: float, epsilon: float = 0.00001,
                   nmax: int = 100000):
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
        # Calculate function values
        f = f_0(current_x)
        if abs(f) < epsilon:
            return current_x
        f_prime = f_1(current_x)
        f_double_prime = f_2(current_x)

        # Update the value of the variable as long as the threshold has not been met
        current_x = current_x - (2 * f * f_prime) / \
            (2 * f_prime ** 2 - f * f_double_prime)


def secant_method(f: Callable, n_0: float = 1, n_1: float = 0, epsilon: float = 0.00001, nmax=100000):
    """
    The secant method is a root-finding algorithm.

    :param f:
    :param n_0:
    :param n_1:
    :param epsilon:
    :return:
    """
    d = (n_0 - n_1) / (f(n_0) - f(n_1)) * f(n_0)
    for i in range(100000):
        if abs(d) <= epsilon:
            break
        n_1 = n_0
        n_0 -= d
        d = (n_0 - n_1) / (f(n_0) - f(n_1)) * f(n_0)
    return n_0


def inverse_interpolation(f: Callable, x0: float, x1: float, x2: float, epsilon: float = 0.00001, nmax: int = 100000):
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
        x3 = (f(x2) * f(x1)) / ((f(x0) - f(x1)) * (f(x0) - f(x2))) * x0
        x3 += (f(x0) * f(x2)) / ((f(x1) - f(x0)) * (f(x1) - f(x2))) * x1
        x3 += (f(x0) * f(x1)) / ((f(x2) - f(x0)) * (f(x2) - f(x1))) * x2
        x0 = x1
        x1 = x2
        x2 = x3
    warnings.warn(
        "The result might be inaccurate. Try entering different parameters or using different methods")
    return x2


def laguerre_method(f_0: Callable, f_1: Callable, f_2: Callable, x0: float, n: float, epsilon: float = 0.00001,
                    nmax=100000):
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
    warnings.warn(
        "The solution might be inaccurate due to insufficient convergence.")
    return xk


def get_bounds(degree: int, coefficients):
    upper = 1 + 1 / abs(coefficients[-1]) * \
        max(abs(coefficients[x]) for x in range(degree))
    lower = abs(coefficients[0]) / (abs(coefficients[0]) +
                                    max(abs(coefficients[x]) for x in range(1, degree + 1)))
    return upper, lower


def __aberth_approximations(coefficients):
    n = len(coefficients) - 1
    if coefficients[-1] == 0:
        return __durandKerner_approximations(coefficients)

    radius = abs(coefficients[-1] / coefficients[0]) ** (1 / n)
    print(f"radius:{radius}")
    return [complex(radius * cos(angle), radius * sin(angle)) for angle in np.linspace(0, 2 * pi, n)]


def __durandKerner_approximations(coefficients):
    n = len(coefficients) - 1
    if coefficients[0] == 0:
        return [0 for _ in range(n)]
    radius = 1 + max(abs(coefficient) for coefficient in coefficients)
    return [complex(radius * cos(angle), radius * sin(angle)) for angle in np.linspace(0, 2 * pi, n)]


def durand_kerner(f_0: Callable, coefficients, epsilon=0.00001, nmax=5000):
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
        coefficients = [coefficient / coefficients[0]
                        for coefficient in coefficients]
        f_0 = monic_poly_from_coefficients(coefficients).to_lambda()
    else:
        coefficients = [coefficient for coefficient in coefficients]
    current_guesses = __durandKerner_approximations(coefficients)
    for i in range(nmax):
        if all(abs(f_0(current_guess)) < epsilon for current_guess in current_guesses):
            return {complex(round_decimal(c.real), round_decimal(c.imag)) for c in current_guesses}
        for index in range(len(current_guesses)):
            numerator = f_0(current_guesses[index])
            other_guesses = (guess for j, guess in enumerate(
                current_guesses) if j != index)
            denominator = functools.reduce(
                lambda a, b: a * b, (current_guesses[index] - guess for guess in other_guesses))
            current_guesses[index] -= numerator / \
                denominator  # Updating each guess
    return {complex(round_decimal(c.real), round_decimal(c.imag)) for c in current_guesses}


def durand_kerner2(coefficients, epsilon=0.0001, nmax=5000):
    if coefficients[0] != 1:
        coefficients = [coefficient / coefficients[0]
                        for coefficient in coefficients]
    else:
        coefficients = [coefficient for coefficient in coefficients]
    executable_lambda = monic_poly_from_coefficients(coefficients).to_lambda()
    return durand_kerner(executable_lambda, coefficients, epsilon, nmax)


def negligible_complex(expression: complex, epsilon) -> bool:
    return abs(expression.real) < epsilon and abs(expression.imag) < epsilon


def ostrowski_method(f_0: Callable, f_1: Callable, initial_value, epsilon: float = 0.00001, nmax: int = 100000):
    """ A root finding algorithm with a convergence rate of 3. Finds a single real root."""
    if f_1(initial_value) == 0:  # avoid zero division error, when the guess is the zero of the derivative
        initial_value += 0.1
    for i in range(nmax):
        f_x = f_0(initial_value)
        if abs(f_x) < epsilon:
            return initial_value
        f_tag = f_1(initial_value)
        y = initial_value - f_x / f_tag  # risk of zero division error
        f_y = f_0(y)
        initial_value = y - (f_y * (y - initial_value)) / (2 * f_y - f_x)
    return initial_value


def chebychevs_method(f_0: Callable, f_1: Callable, f_2: Callable, initial_value, epsilon: float = 0.00001,
                      nmax: int = 100000):
    if f_1(initial_value) == 0:  # avoid zero division error, when the guess is the zero of the derivative
        initial_value += 0.1
    for i in range(nmax):
        f_x = f_0(initial_value)
        if abs(f_x) < epsilon:
            return initial_value
        f_tag = f_1(initial_value)
        f_tag_tag = f_2(initial_value)
        initial_value -= (f_x / f_tag) * \
            (1 + (f_x * f_tag_tag) / (2 * f_tag ** 2))
    warnings.warn("The solution might have not converged properly")
    return initial_value


def aberth_method(f_0: Callable, f_1: Callable, coefficients, epsilon: float = 0.000001, nmax: int = 100000) -> set:
    """
    Aberth-Erlich method is a root-finding algorithm, developed in 1967 Oliver Aberth, and later improved
    in the seventies by Louis W. Ehrlich.
    It finds all of the roots of a function - both real and complex, except some special cases.
    It is considered more efficient than other multi-root finding methods such as durand-kerner,
    since it converges faster to the roots.

    :param f_0: The origin function. f(x).
    :param f_1: The first derivative. f'(x)
    :param coefficients: A collection of the coefficients of the function.
    :return: Returns a set of all of the different solutions.
    """
    try:
        random_guesses = __aberth_approximations(coefficients)
        for n in range(nmax):
            offsets = []
            for k, zk in enumerate(random_guesses):
                m = f_0(zk) / f_1(zk)
                sigma = sum(1 / (zk - zj)
                            for j, zj in enumerate(random_guesses) if k != j and zk != zj)
                denominator = 1 - m * sigma
                offsets.append(m / denominator)
            random_guesses = [
                approximation - offset for approximation, offset in zip(random_guesses, offsets)]
            if all(negligible_complex(f_0(guess), epsilon) for guess in random_guesses):
                break
        solutions = [complex(round_decimal(result.real), round_decimal(
            result.imag)) for result in random_guesses]
        delete_indices = []
        for index, solution in enumerate(solutions):
            for i in range(index + 1, len(solutions)):
                if i in delete_indices:
                    continue
                suspect = solutions[i]
                if abs(solution.real - suspect.real) < 0.0001 and abs(solution.imag - suspect.imag) < 0.0001:
                    delete_indices.append(i)
        return {solutions[i] for i in range(len(solutions)) if i not in delete_indices}
    except ValueError:
        return set()


def steffensen_method(f: Callable, initial: float, epsilon: float = 0.000001, nmax=100000):
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
    for i in range(100000):
        fx = f(x)
        if abs(fx) < epsilon:
            break
        gx = (f(x + fx)) / fx - 1
        if gx == 0:
            warnings.warn("Failed using the steffensen method!")
            break
        x -= fx / gx
    return x


def bisection_method(f: Callable, a: float, b: float, epsilon: float = 0.00001, nmax: int = 10000):
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
        a, b = b, a
    elif a == b:
        raise ValueError("a and b cannot be equal! a must be smaller than b")
    fa, fb = f(a), f(b)
    if not (fa < 0 < fb or fb < 0 < fa):
        raise ValueError("a and b must be of opposite signs")
    for i in range(nmax):
        c = (a + b) / 2
        fc = f(c)
        if fc == 0 or (b - a) / 2 < epsilon:
            return c
        if fc * f(a) > 0:
            a = c
        else:
            b = c
    # Didn't find the solution !
    return None




def bairstow_method():  # TODO: implement this already ...
    pass

