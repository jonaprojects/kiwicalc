# STANDARD LIBRARY IMPORTS
from sys import exc_info

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



def main():
    """ main  method """
    pass


if __name__ == '__main__':
    main()
