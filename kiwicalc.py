# STANDARD LIBRARY IMPORTS
from sys import exc_info
from math import sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, sqrt, e, pi, floor, ceil, log, \
    log10, log2, exp, erf, erfc, gamma, lgamma, tau, comb, degrees, radians
from enum import Enum
import string
import random
import warnings
from functools import reduce
import json
import operator
import re
import inspect
import cmath
from itertools import permutations, combinations, cycle
from abc import ABC, abstractmethod
from collections import Counter, namedtuple
from typing import Callable, Any, Optional, Iterable, Iterator, List, Union, Tuple, Set
from contextlib import contextmanager
import os

# THIRD PARTY IMPORTS
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv, LinAlgError
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.units import cm
from reportlab.lib import utils
from reportlab.platypus import Frame, Image
from anytree import Node, ZigZagGroupIter, PreOrderIter
from defusedxml.ElementTree import parse
from anytree import RenderTree
from googletrans import Translator

# TODOS:
# TODO: Implement multiple horizontal asymptotes?
# TODO: keep improving the derivatives and integrals of functions and expressions  [ HARD ] [ IN PROGRESS ]
# TODO: implement Fraction and Root fully as base classes and change the child classes [ IN PROGRESS ]
# TODO: fix polynomial division, and try to implement polynomial sub-expression sorting more efficiently. [HARD]
# TODO: finish the documentation ... [IN PROGRESS]
# TODO: add in the documentation the part of generating random equations too [ IN PROGRESS ]
# TODO: finish doing the unit testing for subclasses of IExpression and for the whole program eventually. [IN PROGRESS]
# TODO: working with ExpressionSum and Matrices together: Multiplication for start
# TODO: add a generic algorithm  thingy ???
# TODO: create plot2d and plot3d methods as separate methods as well
# TODO: add reports to IExpression objects.
# TODO: simplify logarithm division!
# TODO: add try to mono or poly to the exponent object.

# NEXT VERSIONS:
# TODO: arithmetic progression and geometric series from strings
# TODO: ExpressionSum could be imported and exported in XML too?
# TODO: work with trigonometric expressions with different units: Radians, Degrees, Gradians
# TODO: Create a method that factors a polynomial  [ HARD ]
# TODO: TRY TO ENHANCE PERFORMANCE WITH CTYPES


# GLOBAL VARIABLES


def cot(number):
    return cos(number) / sin(number)


def sec(number):
    return 1 / cos(number)


def asec(number):
    return acos(1 / number)


def factorial(number: Union[int, float]):
    return gamma(number + 1)


def csc(number):
    return 1 / sin(number)


def acsc(number):
    return asin(1 / number)


class TrigoMethods(Enum):
    SIN = sin,
    ASIN = asin,
    COS = cos,
    ACOS = acos,
    TAN = tan,
    ATAN = atan,
    COT = atan,
    SEC = sec
    CSC = csc
    ASEC = asec
    ACSC = acsc


def _TrigoMethodFromString(method_string: str):
    """ Method for internal use. DO NOT USE IT IF YOU'RE NOT IN THE KIWICALC DEVELOPERS TEAM"""
    try:
        method_string = method_string.strip().upper()
        return operator.attrgetter(method_string)(TrigoMethods)
    except AttributeError:
        raise AttributeError(
            f"Unsupported trigonometric method:'{method_string}'")


def get_image(path, width=1 * cm):
    """Utility method for building images for PDF files. Only for internal use."""
    img = utils.ImageReader(path)
    iw, ih = img.getSize()
    aspect = ih / float(iw)
    return Image(path, width=width, height=(width * aspect))


def ln(x) -> float:
    """ ln(x) = log_e(x)"""
    return log(x, e)


def lagrange_polynomial(axes, y_values):
    """
    Get a collection of corresponding x and y values, and return a polynomial that passes through these dots

    :param axes: A collection of x values
    :param y_values: A collection of corresponding y values
    :return: A polynomial that passes through of all of the dots
    """
    x = Var('x')
    result = Poly(0)
    for i, xi in enumerate(axes):
        numerator, denominator = Poly(1), 1
        for j, xj in enumerate(axes):
            if xi != xj:
                numerator *= (x - xj)
                denominator *= (xi - xj)
        result += (numerator / denominator) * y_values[i]

    result.simplify()
    return result


def taylor_polynomial(func: "Union[Function, Poly, Mono]", n: int, a: float, var: str = 'x'):
    """This feature is under testing and development at the moment."""
    mono_expressions = [func(a)]
    current_var = Var(var)
    ith_derivative = func
    for i in range(n):
        ith_derivative = ith_derivative.derivative()
        expression = ith_derivative(
            a) / factorial(i+1) * (current_var - a) ** (i+1)
        mono_expressions.append(expression)
    return Poly(mono_expressions)


def generate_jacobian(functions, variables):  # TODO: add more specific type hints
    if len(functions) != len(variables):
        raise ValueError("The Jacobian matrix must be nxn, so you need to enter an equal number of functions "
                         "and variables_dict")

    return [[func.partial_derivative(variable) for variable in variables] for func in functions]


def approximate_jacobian(functions, values, h=0.001):
    result_jacobian = []
    for f in functions:
        temp = f(*values)
        new_list = []
        for index, variable in enumerate(values):
            new_list.append(
                (f(*(values[:index] + [values[index] + h] + values[index + 1:])) - temp) / h)
        result_jacobian.append(new_list)
    return result_jacobian


def equation_to_function(equation: str, variables: Iterable[str] = None) -> "Function":
    """ Convert an equation to a Function object"""

    function_string = equation_to_one_side(equation)
    function_signature = f"f({','.join(variables)})"
    return Function(f"{function_signature} = {function_string}")


def generate_polynomial_matrix(
        equations: "Union[Iterable[Union[str,Poly,Mono]],Iterable[Union[str, Poly, Mono]]]") -> "Matrix":
    """Creating a matrix of polynomials from a collection of equations"""
    if isinstance(equations[0], str):
        return Matrix(matrix=[poly_from_str(equation_to_one_side(equation)) for equation in equations])
    return Matrix(matrix=equations)


def add_or_sub_coefficients(first_coefficients, second_coefficients, mode='add', copy_first=True):
    first_coefficients = list(
        first_coefficients) if copy_first else first_coefficients
    second_coefficients = list(second_coefficients)
    my_variables_length = len(first_coefficients)
    other_variables_length = len(second_coefficients)
    # Make sure the two lists are in the same length
    if my_variables_length > other_variables_length:
        for _ in range(my_variables_length - other_variables_length):
            second_coefficients.insert(0, 0)
    elif my_variables_length < other_variables_length:
        for _ in range(other_variables_length - my_variables_length):
            first_coefficients.insert(0, 0)
    if mode == 'add':
        for index in range(len(first_coefficients)):
            first_coefficients[index] += second_coefficients[index]
    elif mode == 'sub':
        for index in range(len(first_coefficients)):
            first_coefficients[index] -= second_coefficients[index]
    while first_coefficients[0] == 0:  # delete spare zeros
        del first_coefficients[0]

    return first_coefficients


def sorted_expressions(expressions: "Iterable[Union[Poly,Mono]]"):
    # TODO: add feature for handling free numbers as well ?
    assert all(
        expression.variables_dict is not None for expression in expressions), "This method cannot accept free numbers"
    return sorted(expressions, key=lambda item: max(item.variables_dict.values()),
                  reverse=True)  # sort by the power


def fetch_power(variables: dict):
    return variables[next(iter(variables))]


def fetch_variable(variables: dict):
    """ Brings the first variable in a dictionary of variables_dict and their values """
    try:
        return f'{next(iter(variables))}'
    except (IndexError, StopIteration):
        return None


def process_object(expression: Union[IExpression, int, float], class_name: str, method_name: str, param_name: str):
    if isinstance(expression, (int, float)):
        return Mono(expression)
    elif isinstance(expression, IExpression):
        return expression.__copy__()
    raise TypeError(f"Invalid type '{type(expression)}' of paramater '{param_name}' in method {method_name} in class"
                    f" {class_name}")


def conversion_wrapper(given_func: Callable):
    def inner(self):
        if not len(self._expressions) == 1:
            raise ValueError(
                "expression must contain one item only for cosine conversion: For example, sin(3x)")
        return given_func(self)

    return inner


class Asin(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Asin, self).__init__(
                coefficient=f"asin{expression}", dtype=dtype)
        super(Asin, self).__init__(1, expressions=(
            (TrigoMethods.ASIN, expression, 1),))


class Acos(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Acos, self).__init__(
                coefficient=f"acos({expression})", dtype=dtype)
        else:
            super(Acos, self).__init__(1, expressions=(
                (TrigoMethods.ACOS, expression, 1),))


class Atan(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Atan, self).__init__(
                coefficient=f"atan{expression}", dtype=dtype)
        else:
            super(Atan, self).__init__(1, expressions=(
                (TrigoMethods.ATAN, expression, 1),))


class Cot(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Cot, self).__init__(
                coefficient=f"cot{expression}", dtype=dtype)
        else:
            super(Cot, self).__init__(1, expressions=(
                (TrigoMethods.COT, expression, 1),))

    @conversion_wrapper
    def to_sin(self) -> "Sin":
        pass

    @conversion_wrapper
    def to_cos(self) -> "Cos":
        pass

    @conversion_wrapper
    def to_tan(self) -> "Tan":
        pass

    @conversion_wrapper
    def to_sec(self) -> "Sec":
        pass

    @conversion_wrapper
    def to_csc(self) -> "Csc":
        pass


class Sec(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Sec, self).__init__(coefficient=f"sec({expression})")
        else:
            super(Sec, self).__init__(1, expressions=(
                (TrigoMethods.SEC, expression, 1),))

    @conversion_wrapper
    def to_sin(self) -> "Sin":
        pass

    @conversion_wrapper
    def to_cos(self):
        return Fraction(1, Sin(self.expressions[0][1]))

    @conversion_wrapper
    def to_tan(self) -> "Tan":
        pass

    @conversion_wrapper
    def to_cot(self) -> "Cot":
        pass

    @conversion_wrapper
    def to_csc(self) -> "Csc":
        pass


# TODO: add ACOT TO THE SUPPORTED METHODS
class Acot(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Acot, self).__init__(
                coefficient=f"asec({expression})", dtype=dtype)
        else:
            super(Acot, self).__init__(1, expressions=(
                (TrigoMethods.ACOT, expression, 1),))


class ASec(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(ASec, self).__init__(
                coefficient=f"asec({expression})", dtype=dtype)
        else:
            super(ASec, self).__init__(1, expressions=(
                (TrigoMethods.ASEC, expression, 1),))


class Csc(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Csc, self).__init__(
                coefficient=f"csc({expression})", dtype=dtype)
        else:
            super(Csc, self).__init__(1, expressions=(
                (TrigoMethods.CSC, expression, 1),))

    @conversion_wrapper
    def to_sin(self) -> "Sin":
        pass

    @conversion_wrapper
    def to_cos(self) -> "Cos":
        pass

    @conversion_wrapper
    def to_tan(self) -> "Tan":
        pass

    @conversion_wrapper
    def to_cot(self) -> "Cot":
        pass

    @conversion_wrapper
    def to_sec(self) -> "Sec":
        pass


class ACsc:
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(ACsc, self).__init__(
                coefficient=f"acsc({expression})", dtype=dtype)
        else:
            super(ACsc, self).__init__(1, expressions=(
                (TrigoMethods.ACSC, expression, 1),))





def max_power(expressions):
    return max(expressions, key=lambda expression: max(expression.variables_dict.values()))


def numerical_diff(f, a, method='central', h=0.01):
    if method == 'central':
        return (f(a + h) - f(a - h)) / (2 * h)
    elif method == 'forward':
        return (f(a + h) - f(a)) / h
    elif method == 'backward':
        return (f(a) - f(a - h)) / h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")


def get_equation_variables(equation: str) -> List[Optional[str]]:
    return list({character for character in equation if character.isalpha()})


# TODO: implement these methods !
def format_matplot_function(expression: str):
    raise NotImplementedError


def format_matplot(expression: str):
    return format_matplot_polynomial(expression)


def _format_minus(expression1, expression2):
    """
    Internal method. Not for outside use !!!
    For formatting strings in the format (x-a)^2
    """
    expression1_str, expression2_str = expression1.__str__(), expression2.__str__()
    if "+" in expression1_str or "-" in expression1_str:
        expression1_str = F"({expression1})"
    else:
        expression1_str = f"{expression1}"
    if "-" in expression2_str or "+" in expression2_str:
        expression2_str = f"({expression2})"
    else:
        expression2_str = f"{expression2}"
    if expression2 == 0:
        if expression1 == 0:
            return "0"
        return f"{expression1_str}^2"
    elif expression1 == 0:
        return f"{expression2_str}^2"  # because (0-a)^2 equals a^2 for example

    return f"({expression1_str}-{expression2_str})^2"


def _get_limits_vectors_2d(vectors):
    """Internal method: find the edge values for the scope of the 2d frame"""
    min_x = min(min(
        vector.start_coordinate[0], vector.end_coordinate[0]) for vector in vectors) * 1.05
    max_x = max(max(
        vector.start_coordinate[0], vector.end_coordinate[0]) for vector in vectors) * 1.05
    min_y = min(min(
        vector.start_coordinate[1], vector.end_coordinate[1]) for vector in vectors) * 1.05
    max_y = max(max(
        vector.start_coordinate[1], vector.end_coordinate[1]) for vector in vectors) * 1.05
    return min_x, max_x, min_y, max_y


def _get_limits_vectors_3d(vectors):
    """Internal method: find the edge values for the scope of the 3d frame"""
    min_x = min(
        min(vector.start_coordinate[0], vector.end_coordinate[0]) for vector in vectors)
    max_x = max(
        max(vector.start_coordinate[0], vector.end_coordinate[0]) for vector in vectors)
    min_y = min(
        min(vector.start_coordinate[1], vector.end_coordinate[1]) for vector in vectors)
    max_y = max(
        max(vector.start_coordinate[1], vector.end_coordinate[1]) for vector in vectors)
    min_z = min(
        min(vector.start_coordinate[2], vector.end_coordinate[2]) for vector in vectors)
    max_z = max(
        max(vector.start_coordinate[2], vector.end_coordinate[2]) for vector in vectors)
    return min_x, max_x, min_y, max_y, min_z, max_z


def main():
    """ main  method """
    pass


if __name__ == '__main__':
    main()
