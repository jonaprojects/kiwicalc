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
# TODO: in register page, redirect if the user is authenticated [ CHECK IF DONE]
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
        raise AttributeError(f"Unsupported trigonometric method:'{method_string}'")


class Operator:
    __slots__ = ['__sign', '__method']

    def __init__(self, sign: str, method: Callable):
        self.__sign = sign
        self.__method = method

    @property
    def sign(self) -> str:
        return self.__sign

    @property
    def method(self):
        return self.__method

    def __str__(self):
        return self.__sign


class GreaterThan(Operator):
    def __init__(self):
        super(GreaterThan, self).__init__(">", operator.gt)


class LessThan(Operator):
    def __init__(self):
        super(LessThan, self).__init__("<", operator.lt)


class GreaterOrEqual(Operator):
    def __init__(self):
        super(GreaterOrEqual, self).__init__(">=", operator.ge)


class LessOrEqual(Operator):
    def __init__(self):
        super(LessOrEqual, self).__init__("<=", operator.le)


GREATER_THAN, GREATER_OR_EQUAL, LESS_THAN, LESS_OR_EQUAL = GreaterThan(), GreaterOrEqual(), LessThan(), LessOrEqual()
ptn = re.compile(r"a_(?:n|{n-\d})")  # a_n a_{n-5}
number_pattern = r"\d+[.,]?\d*"  # 3.14159265358979323466



def get_image(path, width=1 * cm):
    """Utility method for building images for PDF files. Only for internal use."""
    img = utils.ImageReader(path)
    iw, ih = img.getSize()
    aspect = ih / float(iw)
    return Image(path, width=width, height=(width * aspect))


def ln(x) -> float:
    """ ln(x) = log_e(x)"""
    return log(x, e)


def is_lambda(v) -> bool:
    """ Returns True whether an expression is a lambda expression, otherwise False"""
    sample_lambda = lambda: 0
    try:
        return isinstance(v, type(sample_lambda)) and v.__name__ == sample_lambda.__name__
    except:
        return False


def format_coefficient(coefficient: "Union[int, float, IExpression]") -> str:
    if coefficient == 1:
        return ""
    if coefficient == -1:
        return "-"
    return coefficient.__str__()


def format_free_number(free_number: Union[int, float]):
    if free_number == 0:
        return ""
    if free_number < 0:
        return f"{round_decimal(free_number)}"

    return f"+{round_decimal(free_number)}"


def linear_regression(axes, y_values, get_values: bool = False):
    """
    Receives a collection of x values, and a collection of their corresponding y values, and builds a fitting
    linear line from then. If the parameter "get_values" is set to True, a tuple will be returned : (slope,free_number)
    otherwise, a lambda equation of the form - lambda x : a*x + b, namely, f(x) = ax+b , will be returned.
    """
    if len(axes) != len(y_values):
        raise ValueError(f"Each x must have a corresponding y value ( Got {len(axes)} x values and {len(y_values)} y "
                         f"values ).")
    n = len(axes)
    sum_x, sum_y = sum(axes), sum(y_values)
    sum_x_2, sum_xy = sum(x ** 2 for x in axes), sum(x * y for x, y in zip(axes, y_values))
    denominator = n * sum_x_2 - sum_x ** 2
    b = (sum_y * sum_x_2 - sum_x * sum_xy) / denominator
    a = (n * sum_xy - sum_x * sum_y) / denominator
    if get_values:
        return a, b
    return lambda x: a * x + b


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
        expression = ith_derivative(a) / factorial(i+1) * (current_var - a) ** (i+1)
        mono_expressions.append(expression)
    return Poly(mono_expressions)


def apply_on(func: Callable, collection: Iterable) -> Iterable:
    """Apply a certain given function on a collection of items"""

    if isinstance(collection, (list, set)):  # modify the given collection
        for index, value in enumerate(collection):
            collection[index] = func(value)
        return collection
    return [func(item) for item in collection]


def float_gcd(a: float, b: float, rtol: float = 1e-05, atol: float = 1e-08):
    """ finding the greatest common divisor for 2 float numbers"""
    if a == 0 or b == 0:
        return 0
    t = min((abs(a), abs(b)))
    while abs(b) > rtol * t + atol:
        a, b = b, a % b
    return round_decimal(a)


def gcd(decimal_numbers: Iterable):
    """
    Finding the greatest common divisor. For example, for the tuple (2.5,3.5,1.5) the result would be 0.5.

    :param decimal_numbers: a list or tuple with decimal numbers
    :return: the greatest common divisor of those numbers
    """
    return reduce(lambda a, b: float_gcd(a, b), decimal_numbers)








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
            new_list.append((f(*(values[:index] + [values[index] + h] + values[index + 1:])) - temp) / h)
        result_jacobian.append(new_list)
    return result_jacobian


def equation_to_one_side(equation: str) -> str:
    """ Move all of the items of the equation to one side"""

    equal_sign = equation.find("=")
    if equal_sign == -1:
        raise ValueError("Invalid equation - an equation must have two sides, separated by '=' ")
    first_side, second_side = equation[:equal_sign], equation[equal_sign + 1:]
    second_side = "".join(
        ('+' if character == '-' else ('-' if character == '+' else character)) for character in second_side)
    second_side = f'-{second_side}' if second_side[0] not in ('+', '-') else second_side
    if second_side[0] in ('+', '-'):
        return first_side + second_side
    return first_side + second_side


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


def broyden(functions, initial_values, h: float = 0.0001, epsilon: float = 0.00001, nmax: int = 10000):
    # move this to numpy implementation
    if all(abs(initial_value) <= epsilon for initial_value in initial_values):
        return initial_values
    initial_jacobian = Matrix(approximate_jacobian(functions, initial_values, h))
    initial_inverse = initial_jacobian.inverse()
    for _ in range(1):
        if all(abs(initial_value) <= epsilon for initial_value in initial_values):
            return initial_values
        F_x0 = [[f(*initial_values)] for f in functions]
        F_x0_matrix = Matrix(F_x0)
        initial_values_matrix = Matrix(initial_values)
        current_values = initial_values_matrix - initial_inverse * F_x0
        F_x1 = [f(*current_values) for f in functions]
        y1 = [item1 - item2 for (item1, item2) in zip(F_x1, F_x0)]
        s1 = [item1 - item2 for (item1, item2) in zip(initial_values, current_values)]
        temp = s1 * initial_inverse * y1
        print(temp)


def add_or_sub_coefficients(first_coefficients, second_coefficients, mode='add', copy_first=True):
    first_coefficients = list(first_coefficients) if copy_first else first_coefficients
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


def derivative(coefficients, get_string=False) -> Union[int, float, list]:
    """ receives the coefficients of a polynomial or a string
     and returns the derivative ( either list, float, or integer) """
    if isinstance(coefficients, str):
        coefficients = ParseExpression.to_coefficients(coefficients)

    num_of_coefficients = len(coefficients)
    if num_of_coefficients == 0:
        raise ValueError("At least one coefficient is required")
    elif num_of_coefficients == 1:  # Derivative of a free number is 0
        return 0
    elif num_of_coefficients == 2:  # f(x) = 2x, f'(x) = 2
        return coefficients[0]
    result = [coefficients[index] * (num_of_coefficients - index - 1) for index in range(num_of_coefficients - 1)]
    if get_string:
        return ParseExpression.coefficients_to_str(result)
    return result


def integral(coefficients, c=0, modify_original=False, get_string=False):
    """ receives the coefficients of a polynomial or a string
     and returns the integral ( either list, float, or integer) """

    if isinstance(coefficients, str):
        coefficients = ParseExpression.to_coefficients(coefficients)
    num_of_coefficients = len(coefficients)
    if num_of_coefficients == 0:
        raise ValueError("At least one coefficient is required")
    elif num_of_coefficients == 1:
        return [coefficients[0], c]
    else:
        coefficients = coefficients if modify_original and not isinstance(coefficients, (tuple, set)) else list(
            coefficients)
        coefficients.insert(0, coefficients[0] / num_of_coefficients)
        for i in range(1, num_of_coefficients):  # num_of_coefficient is now like len(coefficients)-1
            coefficients[i] = coefficients[i + 1] / (num_of_coefficients - i)
        coefficients[-1] = c
        if get_string:
            return ParseExpression.coefficients_to_str(coefficients)
        return coefficients


class ParseExpression:

    @staticmethod
    def parse_linear(expression, variables):
        pass

    @staticmethod
    def unparse_linear(variables_dict: dict, free_number: float):
        accumulator = []
        for variable, coefficients in variables_dict.items():
            for coefficient in coefficients:
                if coefficient == 0:
                    continue
                coefficient_str = format_coefficient(coefficient)
                if coefficient > 0:
                    accumulator.append(f"+{coefficient_str}{variable}")
                else:
                    accumulator.append(f"{coefficient_str}{variable}")
        accumulator.append(format_free_number(free_number))
        result = "".join(accumulator)
        if not result:
            return "0"
        if result[0] == '+':
            return result[1:]
        return result

    @staticmethod
    def parse_quadratic(expression: str, variables=None, strict_syntax=True):
        expression = expression.replace(" ", "")
        if variables is None:
            variables = get_equation_variables(expression)
        if strict_syntax:
            if len(variables) != 1:
                raise ValueError(f"Strict quadratic syntax must contain exactly 1 variable, found {len(variables)}")
            variable = variables[0]
            if '**' in expression:
                expression = expression.replace('**', '^')
            a_expression_index: int = expression.find(f'{variable}^2')
            if a_expression_index == -1:
                raise ValueError(f"Didn't find expression containing '{variable}^2' ")
            elif a_expression_index == 0:
                a = 1
            else:
                a = extract_coefficient(expression[:a_expression_index])
            b_expression_index = expression.rfind(variable)
            if b_expression_index == -1:
                b = 0
                c_str = expression[a_expression_index + 3:]
            else:
                b = extract_coefficient(expression[a_expression_index + 3:b_expression_index])
                c_str = expression[b_expression_index + 1:]
            if c_str == '':
                c = 0
            else:
                c = extract_coefficient(c_str)
            return {variable: [a, b], 'free': c}
        else:
            return ParseExpression.parse_polynomial(expression, variables, strict_syntax)

    @staticmethod
    def parse_cubic(expression: str, variables, strict_syntax=True):
        expression = expression.replace(" ", "")
        if strict_syntax:
            print("reached here for cubic")
            if len(variables) != 1:
                raise ValueError(f"Strict cubic syntax must contain exactly 1 variable, found {len(variables)}")
            variable = variables[0]
            if '**' in expression:
                expression = expression.replace('**', '^')
            a_expression_index: int = expression.find(f'{variable}^3')
            if a_expression_index == -1:
                raise ValueError(f"Didn't find expression containing '{variable}^3' ")
            elif a_expression_index == 0:
                a = 1
            else:
                a = extract_coefficient(expression[:a_expression_index])
            b_expression_index = expression.find(f'{variable}^2')
            if b_expression_index == -1:
                b_expression_index = a_expression_index
                b = 0
            else:
                b = extract_coefficient(expression[a_expression_index + 3:b_expression_index])

            c_expression_index = expression.rfind(variable)
            if c_expression_index == -1:
                c = 0
                c_expression_index = b_expression_index
            else:
                c = extract_coefficient(expression[b_expression_index + 3:c_expression_index])
            d_str = expression[c_expression_index + 1:]
            if d_str == '':
                d = 0
            else:
                d = extract_coefficient(d_str)
            return {variable: [a, b, c], 'free': d}
        else:
            return ParseExpression.parse_polynomial(expression, variables, strict_syntax)

    @staticmethod
    def parse_quartic(expression: str, variables, strict_syntax=True):
        expression = expression.replace(" ", "")
        if strict_syntax:
            if len(variables) != 1:
                raise ValueError(f"Strict quadratic syntax must contain exactly 1 variable, found {len(variables)}")
            variable = variables[0]
            if '**' in expression:
                expression = expression.replace('**', '^')
            a_expression_index: int = expression.find(f'{variable}^4')
            if a_expression_index == -1:
                raise ValueError(f"Didn't find expression containing '{variable}^4' ")
            elif a_expression_index == 0:
                a = 1
            else:
                a = extract_coefficient(expression[:a_expression_index])
            b_expression_index = expression.find(f'{variable}^3')
            if b_expression_index == -1:
                b_expression_index = a_expression_index
                b = 0
            else:
                b = extract_coefficient(expression[a_expression_index + 3:b_expression_index])

            c_expression_index = expression.find(f'{variable}^2')
            if c_expression_index == -1:
                c = 0
                c_expression_index = b_expression_index
            else:
                c = extract_coefficient(expression[b_expression_index + 3:c_expression_index])

            d_expression_index = expression.rfind(variable)
            if d_expression_index == -1:
                d = 0
                d_expression_index = c_expression_index
            else:
                d = extract_coefficient(expression[c_expression_index + 3:d_expression_index])

            e_str = expression[d_expression_index + 1:]
            if e_str == '':
                e = 0
            else:
                e = extract_coefficient(e_str)

            return {variable: [a, b, c, d], 'free': e}
        else:
            return ParseExpression.parse_polynomial(expression, variables)

    @staticmethod
    def parse_polynomial(expression: str, variables=None, strict_syntax=True, numpy_array=False, get_variables=False):
        if variables is None:
            variables = list({character for character in expression if character.isalpha()})
        expression = clean_spaces
        mono_expressions = split_expression(expression)
        if numpy_array:
            variables_dict = {variable: np.array([], dtype='float64') for variable in variables}
        else:
            variables_dict = {variable: [] for variable in variables}
        variables_dict['free'] = 0
        for mono in mono_expressions:
            coefficient, variable, power = ParseExpression._parse_monomial(mono, variables)
            if power == 0:
                variables_dict['free'] += coefficient
            else:
                coefficient_list = variables_dict[variable]
                if power > len(coefficient_list):
                    zeros_to_add = int(power) - len(coefficient_list) - 1
                    if numpy_array:
                        coefficient_list = np.pad(coefficient_list, (zeros_to_add, 0), 'constant', constant_values=(0,))
                        variables_dict[variable] = np.insert(coefficient_list, 0, coefficient)
                    else:
                        for _ in range(zeros_to_add):
                            coefficient_list.insert(0, 0)
                        coefficient_list.insert(0, coefficient)
                else:
                    coefficient_list[len(coefficient_list) - int(power)] += coefficient
        if not get_variables:
            return variables_dict
        return variables_dict, variables

    @staticmethod
    def unparse_polynomial(parsed_dict: dict, syntax=""):
        """Taking a parsed polynomial and returning a string from it"""
        accumulator = []
        if syntax not in ("", "pythonic"):
            warnings.warn(f"Unrecognized syntax: {syntax}. Either use the default or 'pythonic' ")
        for variable, coefficients in parsed_dict.items():
            if variable == 'free':
                continue
            sub_accumulator, num_of_coefficients = [], len(coefficients)
            for index, coefficient in enumerate(coefficients):
                if coefficient != 0:
                    coefficient_str = format_coefficient(round_decimal(coefficient))
                    if coefficient_str not in ('', '-') and syntax == 'pythonic':
                        coefficient_str += '*'
                    power = len(coefficients) - index
                    sign = '' if coefficient < 0 or (not accumulator and not sub_accumulator) else '+'
                    if power == 1:
                        sub_accumulator.append(f"{sign}{coefficient_str}{variable}")
                    else:
                        if syntax == 'pythonic':
                            sub_accumulator.append(f"{sign}{coefficient_str}{variable}**{power}")
                        else:
                            sub_accumulator.append(f"{sign}{coefficient_str}{variable}^{power}")
            accumulator.extend(sub_accumulator)
        free_number = parsed_dict['free']
        if free_number != 0 or not accumulator:
            sign = '' if free_number < 0 or not accumulator else '+'
            accumulator.append(f"{sign}{round_decimal(free_number)}")
        return "".join(accumulator)

    @staticmethod
    def _parse_monomial(expression: str, variables):
        """ Extracting the coefficient an power from a monomial, this method is used while parsing polynomials"""
        # Check which variable appears in the expression
        print(expression)
        variable_index = -1
        for suspect_variable in variables:
            suspect_variable_index = expression.find(suspect_variable)
            if suspect_variable_index != -1:
                variable_index = suspect_variable_index
                break
        if variable_index == -1:
            # If we haven't found any variable, that means that the expression is a free number
            try:
                return float(expression), 'free', 0
            except ValueError:
                raise ValueError("Couldn't parse the expression! Found no variables, but the free number isn't valid.")
        else:
            variable = expression[variable_index]
            try:
                coefficient = extract_coefficient(expression[:variable_index])
            except ValueError:
                raise ValueError(f"Encountered an invalid coefficient '{expression[:variable_index]}' while"
                                 f"parsing the monomial '{expression}'")
            power_index = expression.find('^')
            if power_index == -1:
                return coefficient, variable, 1
            try:
                power = float(expression[power_index + 1:])
                return coefficient, variable, power
            except ValueError:
                raise ValueError(f"encountered an invalid power '{expression[power_index + 1:]} while parsing the"
                                 f"monomial '{expression}'")

    @staticmethod
    def to_coefficients(expression: str, variable=None, strict_syntax=True, get_variable=False):
        expression = clean_spacesession)
        if variable is None:
            variables = get_equation_variables(expression)
            num_of_variables = len(variables)
            if num_of_variables == 0:
                return [float(expression)]
            elif num_of_variables != 1:
                raise ValueError(f"Can only parse polynomials with 1 variable, but got {num_of_variables}")
            variable = variables[0]
        mono_expressions = split_expression(expression)
        coefficients_list = [0]
        for mono in mono_expressions:
            coefficient, variable, power = ParseExpression._parse_monomial(mono, (variable,), strict_syntax)
            if power == 0:
                coefficients_list[-1] += coefficient
            else:
                if power > len(coefficients_list) - 1:
                    zeros_to_add = int(power) - len(coefficients_list)
                    for _ in range(zeros_to_add):
                        coefficients_list.insert(0, 0)
                    coefficients_list.insert(0, coefficient)
                else:
                    coefficients_list[len(coefficients_list) - int(power) - 1] += coefficient
        if not get_variable:
            return coefficients_list
        return coefficients_list, variable

    @staticmethod
    def coefficients_to_str(coefficients, variable='x', syntax=""):
        """Taking a parsed polynomial and returning a string from it"""
        accumulator = []
        if syntax not in ("", "pythonic"):
            warnings.warn(f"Unrecognized syntax: {syntax}. Either use the default or 'pythonic' ")
        num_of_coefficients = len(coefficients)
        if num_of_coefficients == 0:
            raise ValueError("At least 1 coefficient is required")
        elif num_of_coefficients == 1:
            return f"{coefficients[0]}"
        for index in range(num_of_coefficients - 1):
            coefficient = coefficients[index]
            if coefficient != 0:
                coefficient_str = format_coefficient(round_decimal(coefficient))
                if coefficient_str not in ('', '-') and syntax == 'pythonic':
                    coefficient_str += '*'
                power = len(coefficients) - index - 1
                sign = '' if coefficient < 0 or not accumulator else '+'
                if power == 1:
                    accumulator.append(f"{sign}{coefficient_str}{variable}")
                else:
                    if syntax == 'pythonic':
                        accumulator.append(f"{sign}{coefficient_str}{variable}**{power}")
                    else:
                        accumulator.append(f"{sign}{coefficient_str}{variable}^{power}")
        free_number = coefficients[-1]
        if free_number != 0 or not accumulator:
            sign = '' if free_number < 0 or not accumulator else '+'
            accumulator.append(f"{sign}{round_decimal(free_number)}")
        return "".join(accumulator)


class ParseEquation:

    @staticmethod
    def parse_polynomial(equation: str):
        variables = get_equation_variables(equation)
        if len(variables) != 1:
            raise ValueError("can only parse quadratic equations with 1 variable")
        variable = variables[0]
        first_side, second_side = equation.split("=")
        first_dict = ParseExpression.parse_polynomial(first_side, variables=variables)
        second_dict = ParseExpression.parse_polynomial(second_side, variables=variables)
        add_or_sub_coefficients(first_dict[variable], second_dict[variable], copy_first=False, mode='sub')
        return first_dict[variable] + [first_dict['free'] - second_dict['free']]

    @staticmethod
    def parse_quadratic(equation: str, strict_syntax=False):  # TODO: check and fix this
        if strict_syntax:
            return ParseExpression.parse_quadratic(equation, strict_syntax=True)
        return ParseEquation.parse_polynomial(equation)


def range_operator_from_string(operator_str: str):
    if operator_str == '>':
        return GREATER_THAN
    if operator_str == '<':
        return LESS_THAN
    if operator_str == '>=':
        return GREATER_OR_EQUAL
    if operator_str == '<=':
        return LESS_OR_EQUAL
    raise ValueError(f"Invalid operator: {operator_str}. Expected: '>', '<', '>=' ,'<='")


def create_range(expression: str, min_dtype: str = 'poly', expression_dtype: str = 'poly', max_dtype='poly',
                 get_tuple=False):
    exprs = re.split('(<=|>=|>|<)', expression)
    num_of_expressions = len(exprs)
    if num_of_expressions == 5:  # 3 < x < 6
        limits = create(exprs[0], dtype=min_dtype), create(exprs[4], dtype=max_dtype)
        middle = create(exprs[2], dtype=expression_dtype)
        min_operator, max_operator = range_operator_from_string(exprs[1]), range_operator_from_string(exprs[3])
    elif num_of_expressions == 3:
        middle = create(exprs[0], dtype=min_dtype)
        my_operator = exprs[1]
        if '>' in my_operator:
            my_operator = my_operator.replace('>', '<')
            min_operator, max_operator = range_operator_from_string(my_operator), None
            limits = (create(exprs[2], dtype=min_dtype), None)
        elif '<' in my_operator:
            min_operator, max_operator = None, range_operator_from_string(my_operator)
            limits = (None, create(exprs[2], dtype=max_dtype))
        else:
            raise ValueError(f"Invalid operator: {my_operator}. Expected: '>', '<', '>=' ,'<='")
    else:
        raise ValueError(f"Invalid string for creating a Range expression: {expression}. Expected expressions"
                         f" such as '3<x<5', 'x^2 > 16', etc..")

    if get_tuple:
        return middle, limits, (min_operator, max_operator)

    return Range(expression=middle, limits=limits, operators=(min_operator, max_operator))


class Range:
    __slots__ = ['__expression', '__minimum', '__maximum', '__min_operator', '__max_operator']

    def __init__(self, expression: "Union[str,IExpression, Function, int, float]",
                 limits: Union[set, list, tuple] = None,
                 operators: Union[set, list, tuple] = None, dtype='poly', copy: bool = True):
        # Handle the expression parameter
        if isinstance(expression, str):
            self.__expression, (self.__minimum, self.__maximum), (
                self.__min_operator, self.__max_operator) = create_range(expression, get_tuple=True)
            return
        elif isinstance(expression, (IExpression, Function)):
            self.__expression = expression.__copy__() if copy else expression
        elif isinstance(expression, (int, float)):
            self.__expression = Mono(expression)
        else:
            raise TypeError(
                f"Range.__init__(): Invalid type of expression: {type(expression)}.Expected types 'IExpression', 'Function', "
                f"or str.")

        # check whether limits is valid
        if not isinstance(limits, (set, list, tuple)):
            raise TypeError(
                f"Range.__init__(): Invalid type of limits: {type(limits)}. Expected types 'list', 'tuple', 'set'.")
        if len(limits) != 2:
            raise ValueError("The length")

        # handle the minimum
        if limits[0] in (np.inf, -np.inf):
            self.__minimum = limits[0]
        elif isinstance(limits[0], (int, float)):
            self.__minimum = Mono(limits[0])
        elif isinstance(limits[0], (IExpression, Function)):
            self.__minimum = limits[0].__copy__() if copy else limits[0]
        elif limits[0] is None:
            self.__minimum = -np.inf

        else:
            raise TypeError("Minimum of the range must be of type 'IExpression', 'Function', None, and inf ")

        # handle the maximum
        if limits[1] in (np.inf, -np.inf):
            self.__maximum = limits[1]
        elif isinstance(limits[1], (int, float)):
            self.__maximum = Mono(limits[1])
        elif isinstance(limits[1], (IExpression, Function)):
            self.__maximum = limits[1].__copy__() if copy else limits[1]
        elif limits[1] is None:
            self.__maximum = -np.inf

        else:
            raise TypeError("Maximum of the range must be of type 'IExpression', 'Function', None, and inf ")

        # handle the operators
        if not isinstance(operators, (list, set, tuple)):
            raise TypeError(
                f"Range.__init__(): Invalid type of operators: {type(limits)}. Expected types 'list', 'tuple', 'set'.")

        if not len(operators) == 2:
            raise ValueError(f"Range.__init__(): The length of the operators must be 2.")

        if copy:
            self.__min_operator = operators[0].__copy__() if hasattr(operators[0], "__copy__") else operators[0]
            self.__max_operator = operators[1].__copy__() if hasattr(operators[1], "__copy__") else operators[1]
        else:
            self.__min_operator, self.__max_operator = operators

    @property
    def expression(self):
        return self.__expression

    @property
    def min_limit(self):
        return self.__minimum

    @property
    def max_limit(self):
        return self.__maximum

    @property
    def min_operator(self):
        return self.__min_operator

    @property
    def max_operator(self):
        return self.__max_operator

    def try_evaluate(self):
        return self.__evaluate()

    def evaluate_when(self, **kwargs):
        if isinstance(self.__minimum, IExpression):
            min_eval = self.__minimum.when(**kwargs).try_evaluate()
        else:
            min_eval = None

        expression_eval = self.__expression.when(**kwargs).try_evaluate()

        if isinstance(self.__maximum, IExpression):
            max_eval = self.__maximum.when(**kwargs).try_evaluate()
        else:
            max_eval = None

        return self.__evaluate(min_eval, expression_eval, max_eval)

    def __evaluate(self, min_eval: float = None, expression_eval: float = None, max_eval: float = None) -> Optional[
        bool]:
        if self.__minimum == np.inf or self.__maximum == -np.inf:
            return False
        expression_eval = self.__expression.try_evaluate() if expression_eval is None else expression_eval
        if self.__minimum != -np.inf:
            minimum_evaluation = self.__minimum.try_evaluate() if min_eval is None else min_eval
            if self.__maximum != np.inf:
                maximum_evaluation = self.__maximum.try_evaluate() if max_eval is None else max_eval
                if None not in (minimum_evaluation, maximum_evaluation):
                    if maximum_evaluation < minimum_evaluation:
                        return False

                if None not in (maximum_evaluation, expression_eval):
                    if not self.__max_operator.method(expression_eval, maximum_evaluation):
                        return False

            if None not in (minimum_evaluation, expression_eval):
                return self.__min_operator.method(minimum_evaluation, expression_eval)
            return None
        else:
            maximum_evaluation = self.__maximum.try_evaluate() if max_eval is None else max_eval
            if None not in (maximum_evaluation, expression_eval):
                return self.__max_operator.method(expression_eval, maximum_evaluation)
            return None

    def __str__(self):
        if self.__minimum == -np.inf and self.__maximum == np.inf:
            return f"-∞{self.__min_operator}{self.__expression}{self.__max_operator}∞"
        if self.__minimum == -np.inf:
            minimum_str = ""
        else:
            minimum_str = f"{self.__minimum}{self.__min_operator}"

        if self.__maximum == np.inf:
            maximum_str = ""
        else:
            maximum_str = f"{self.__max_operator}{self.__maximum}"
        return f"{minimum_str}{self.__expression}{maximum_str}"

    def __copy__(self):
        return Range(self.__expression, (self.__minimum, self.__maximum), (self.__min_operator, self.__max_operator),
                     copy=True)


class RangeCollection:
    __slots__ = ['_ranges']

    def __init__(self, ranges: "Iterable[Range, RangeCollection]", copy=False):
        if copy:
            self._ranges = [my_range.__copy__() for my_range in ranges]
        else:
            self._ranges = [my_range for my_range in ranges]

    @property
    def ranges(self):
        return self._ranges

    def chain(self, range_obj: Range, copy=False):
        if not isinstance(range_obj, Range):
            return TypeError(f"Invalid type {type(range_obj)} for chaining Ranges. Expected type: 'Range' ")
        self._ranges.append((range_obj.__copy__() if copy else range_obj))
        return self

    def __or__(self, other: Range):
        return RangeOR((self, other))

    def __and__(self, other):
        return RangeAND((self, other))

    def __copy__(self):
        return RangeCollection(ranges=self._ranges, copy=True)

    def __str__(self):
        return ", ".join(
            (f"({my_range.__str__()})" if isinstance(my_range, RangeCollection) else my_range.__str__()) for my_range in
            self._ranges)


class RangeOR(RangeCollection):
    """
    This class represents several ranges or collection of ranges with the OR method.
    For instance:
    (x^2 > 25) or (x^2 < 9)
    Or a more complicated example:
    (5<x<6 and x^2>26) or x<7 or (sin(x)>=0 or sin(x) < 0.5)
    """

    def __init__(self, ranges: "Iterable[Range, RangeCollection]", copy=False):
        super(RangeOR, self).__init__(ranges)

    def try_evaluate(self):
        pass

    def simplify(self) -> Optional[Union[Range, RangeCollection]]:
        pass

    def __str__(self):
        return " or ".join(
            (f"({my_range.__str__()})" if isinstance(my_range, RangeCollection) else my_range.__str__()) for my_range in
            self._ranges)

    def __copy__(self):
        return RangeOR(self._ranges, copy=True)


class RangeAND(RangeCollection):
    """
    This class represents several ranges or collection of ranges with the AND method.
    For instance:
    (x^2 > 25) and (x>0)
    Or a more complicated example:
    (5<x<6 and x^2>26) and x<7 and (sin(x)>=0 or sin(x) < 0.5)
    """

    def __init__(self, ranges: "Iterable[Range, RangeCollection]", copy=False):
        super(RangeAND, self).__init__(ranges)

    def try_evaluate(self):
        pass

    def simplify(self) -> Optional[Union[Range, RangeCollection]]:
        pass

    def __str__(self):
        return " and ".join(
            (f"({my_range.__str__()})" if isinstance(my_range, RangeCollection) else my_range.__str__()) for my_range in
            self._ranges)

    def __copy__(self):
        return RangeOR(self._ranges, copy=True)


def solve_poly_system(
        equations: "Union[Iterable[Union[str,Poly,Mono]],Iterable[Union[str, Poly, Mono]]]",
        initial_vals: dict = None, epsilon: float = 0.00001, nmax: int = 10000, show_steps=False):
    """
    This method solves for all the real solutions of a system of polynomial equations.
    :param equations: A collection of equations; each equation must be an equation (of type 'str')  or a polynomial.
    :param initial_vals: A dictionary with some initial approximations to the solutions. For example: {'x':1,'y':2}
    :param epsilon: negligible y value to be considered as 0: for example: 0.001, 0.000001. The smaller epsilon, the more accurate the result and more iterations are required.
    :param nmax: the maximum number of iterations
    :param show_steps: True / False - Whether to show the steps of the solution while solving.
    :return: returns the results of the equation system, a dictionary with variables_dict as keys and their values.
    """
    if initial_vals is None:
        variables = {extract_variables_from_expression(equation) for equation in equations}
        initial_vals = Matrix(matrix=[0 for _ in range(len(variables))])
    variables, initial_values = list(initial_vals.keys()), Matrix(matrix=initial_vals.values())
    polynomials = [(poly_from_str(equation_to_one_side(equation)) if isinstance(equation, str) else equation) for
                   equation in equations]
    jacobian_matrix = Matrix(
        matrix=generate_jacobian(polynomials, variables))  # Generating a corresponding jacobian matrix
    current_values_matrix = Matrix(matrix=[[current_value] for current_value in list(initial_vals.values())])
    for i in range(nmax):
        assignment_dictionary = dict(zip(variables, [row[0] for row in current_values_matrix.matrix]))
        assigned_jacobian = jacobian_matrix.mapped_matrix(
            lambda polynomial: polynomial.when(**assignment_dictionary).try_evaluate())
        jacobian_inverse = assigned_jacobian.inverse()
        assigned_polynomials = Matrix(matrix=[[polynomial.when(**assignment_dictionary).try_evaluate()] for
                                              polynomial in polynomials])
        if all(abs(row[0]) < epsilon for row in assigned_polynomials.matrix):
            return {variables[index]: row[0] for index, row in enumerate(current_values_matrix)}
        interval_matrix = jacobian_inverse @ assigned_polynomials
        current_values_matrix -= interval_matrix


def values_in_range(func: Callable, start: float, end: float, step: float, round_results: bool = False):
    """
    fetches all the valid values of a function in the specified range
    :param func: A callable function, that accepts one parameter and returns a single result
    :param start: the beginning of the range
    :param end: the end of the range
    :param step: the interval between each item in the range
    :return: returns the values in the range, and their valid results
    """
    if round_results:
        values = [round_decimal(i) for i in decimal_range(start, end, step)]
        results = [round_decimal(func(i)) for i in values]
    else:
        values = [_ for _ in decimal_range(start, end, step)]
        results = [func(i) for i in values]
    for index, result in enumerate(results):
        if result is None:
            del results[index]
            del values[index]
        elif isinstance(result, bool):
            results[index] = float(result)
    return values, results









def linear_from_points_exercise(get_solution=True, variable='x', lang="en"):
    if lang != 'en':
        translator = Translator()
    first_point = (random.randint(-15, 15), random.randint(-15, 15))
    second_point = (random.randint(-15, 15), random.randint(-15, 15))
    if first_point[1] == second_point[1]:
        first_point = first_point[0], first_point[1] + random.randint(1, 3)
    a = round_decimal((second_point[1] - first_point[1]) / (second_point[0] - first_point[0]))
    b = round_decimal(first_point[1] - a * first_point[0])
    exercise = f""" a)    Find the linear function that passes through the points {first_point} and {second_point}.
           b)    Is the function increasing or decreasing?
           c)    Bonus: Sketch the function.
           """
    if lang != 'en':
        translation = translator.translate(exercise, dest=lang)
        exercise = translation.text
    a_str = format_coefficient(round_decimal(a))
    b_str = format_free_number(b)
    if a > 0:
        answer_for_b = f"Increasing, because the slope of the function is positive"
    else:
        answer_for_b = f"Decreasing, because the slope of the function is negative"

    if get_solution:
        solution = f"""    a)     y = {a_str}{variable}{b_str}
        b.    {answer_for_b}
        c. Sketching isn't supported yet
        """
        if lang != 'en':
            translation = translator.translate(solution, dest=lang)
            solution = translation.text

        return exercise, solution

    return exercise


def linearFromPointAndSlope_exercise(get_solution=True, variable='x', lang="en"):
    if lang != 'en':
        translator = Translator()
    my_point = (random.randint(-15, 15), random.randint(-15, 15))
    my_slope = random.randint(-15, 15)
    while my_slope == 0:
        my_slope = random.randint(-15, 15)

    exercise = f"""The linear function f(x) passes through the point {my_point} and has a slope of {my_slope}.
           a)    Find f(x).
           b)    Find where the function intersects with the x axis.
           c)    Bonus: Sketch the function.
           """
    if lang != 'en':
        translation = translator.translate(exercise, dest=lang)
        exercise = translation.text
    a_str = format_coefficient(my_slope)
    b_str = format_free_number(my_point[1] - my_slope * my_point[0])
    if get_solution:
        solution = f"""    a)     y = {a_str}{variable}{b_str}
        b.  {round_decimal(-my_point[1] / my_slope), 0}
        c. Sketching isn't supported yet
        """
        if lang != 'en':
            translation = translator.translate(solution, dest=lang)
            solution = translation.text

        return exercise, solution

    return exercise


def linear_intersection_exercise(get_solution=True, variable='x', lang='en'):
    pass


def linear_system_exercise(variables, get_solution=True, digits_after: int = 0, lang='en'):
    if lang != 'en':
        translator = Translator()
    if get_solution:
        equations, solutions = random_linear_system(variables, get_solutions=get_solution, digits_after=digits_after)
    else:
        equations = random_linear_system(variables, get_solutions=get_solution, digits_after=digits_after)

    exercise = """Solve the system of equations:\n""" + "\n".join(f"     {equation}" for equation in equations)
    if lang != 'en':
        exercise = translator.translate(exercise, dest=lang).text

    if get_solution:
        solution = ", ".join(f"{variable}={round_decimal(value)}" for variable, value in zip(variables, solutions))
        return exercise, solution
    return exercise


def generate_pdf_path() -> str:
    path = f"worksheet1.pdf"
    index = 1
    while os.path.isfile(path):
        index += 1
        path = f"worksheet{index}.pdf"

    return path


def worksheet(path: str = None, dtype='linear', num_of_pages: int = 1, equations_per_page: int = 20, get_solutions=True,
              digits_after=0, titles=None):
    if path is None:
        path = generate_pdf_path()
    if dtype == 'linear':
        LinearEquation.random_worksheets(path=path, num_of_pages=num_of_pages, equations_per_page=equations_per_page,
                                         after_point=digits_after, get_solutions=get_solutions, titles=titles)

    elif dtype == 'quadratic':
        QuadraticEquation.random_worksheets(path=path, num_of_pages=num_of_pages, equations_per_page=equations_per_page,
                                            digits_after=digits_after, get_solutions=get_solutions, titles=titles)
    elif dtype == 'cubic':
        CubicEquation.random_worksheets(path=path, num_of_pages=num_of_pages, equations_per_page=equations_per_page,
                                        digits_after=digits_after, get_solutions=get_solutions, titles=titles)
    elif dtype == 'quartic':
        QuarticEquation.random_worksheets(path=path, num_of_pages=num_of_pages, equations_per_page=equations_per_page,
                                          digits_after=digits_after, get_solutions=get_solutions, titles=titles)

    elif dtype == 'polynomial':
        PolyEquation.random_worksheets(path=path, titles=titles, equations_per_page=equations_per_page,
                                       num_of_pages=num_of_pages, digits_after=digits_after,
                                       get_solutions=get_solutions)

    elif dtype == 'trigo':
        # TrigoEquation.random_worksheets(path=path, num_of_pages=num_of_pages, equations_per_page=equations_per_page,
        #                                after_point=after_point, get_solutions=get_solutions, titles=titles)
        pass
    elif dtype == 'log':
        pass
        # LogEquation.random_worksheets(path=path, num_of_pages=num_of_pages, equations_per_page=equations_per_page,
        #                                 after_point=after_point, get_solutions=get_solutions, titles=titles)
    else:
        raise ValueError(f"worksheet(): unknown dtype {dtype}: expected 'linear', 'quadratic', 'cubic', "
                         f"'quartic', 'polynomial', 'trigo', 'log' ")


class PDFExercise:
    """
    This class represents an exercise in a PDF page.
    """
    __slots__ = ['__exercise', '__exercise_type', '__dtype', '__solution', '__number', '__lang']

    def __init__(self, exercise: str, exercise_type: str, dtype: str, solution=None, number=None, lang="en"):
        self.__exercise = exercise
        self.__exercise_type = exercise_type
        self.__dtype = dtype
        self.__solution = solution
        self.__number = number
        self.__lang = lang

    @property
    def exercise(self):
        return self.__exercise

    @property
    def number(self):
        return self.__number

    @number.setter
    def number(self, number: int):
        self.__number = number

    @property
    def dtype(self):
        return self.__dtype

    @property
    def solution(self):
        return self.__solution

    @property
    def has_solution(self):
        return self.__solution is not None

    @property
    def lang(self):
        return self.__lang

    def __str__(self):
        return self.__exercise


class PDFCalculusExercise(PDFExercise):
    def __init__(self, exercise, dtype, solution=None, lang="en"):
        super(PDFCalculusExercise, self).__init__(exercise, "calculus", dtype, solution, lang=lang)


class PDFAnalyzeFunction(PDFCalculusExercise):
    def __init__(self, exercise, dtype: str, solution=None, lang="en"):
        super(PDFAnalyzeFunction, self).__init__(exercise, dtype=dtype, solution=solution, lang=lang)


class PDFLinearFunction(PDFAnalyzeFunction):
    def __init__(self, with_solution: bool = True, lang: str = 'en'):
        if lang != 'en':  # Translate the exercise for other languages if needed!
            translator = Translator()
        my_linear, solution, coefficients = random_linear(get_solution=True, get_coefficients=True)
        random_function = f"f(x) = {my_linear}"
        exercise = f""" The linear function {random_function} is given.
            a) Where does the function intersect with the x axis?
            b) Where does the function intersect with the y axis?
            c) Is the function increasing or decreasing?
            d) What is the derivative of the function?
            e) Sketch the function.
        """

        if lang != 'en':  # Translate the exercise for other languages if needed!
            translation = translator.translate(exercise, dest=lang)
            exercise = translation.text
            print(exercise)
        if with_solution:
            if coefficients[0] > 0:
                answer_for_c = f"Increasing, because the slope of the function is positive"
            else:
                answer_for_c = f"Decreasing, because the slope of the function is negative"
            solution = f"""    a)    ({solution}, 0)
            b)    (0, {coefficients[1]})
            c) {answer_for_c}
            d) f'(x) = {coefficients[0]}
            e) Sketch not supported yet!
             """
            if lang != 'en':
                translation = translator.translate(solution, dest=lang)
                solution = translation.text
        else:
            solution = None

        super(PDFAnalyzeFunction, self).__init__(exercise, dtype='linear', solution=solution, lang=lang)


class PDFLinearIntersection(PDFExercise):
    def __init__(self, with_solution=True, lang='en'):
        pass


class PDFLinearSystem(PDFExercise):
    def __init__(self, with_solution=True, lang='en', num_of_equations=None, digits_after: int = 0):
        if num_of_equations is None:
            num_of_equations = random.randint(2, 3)

        variables = ['x', 'y', 'z', 'm', 'n', 't', 'a', 'b']
        num_of_variables = num_of_equations
        if num_of_variables <= len(variables):
            variables = variables[:num_of_variables]
        elif num_of_variables <= 26:
            variables = string.ascii_lowercase[:num_of_variables]
        else:
            raise ValueError("The system does not support systems of equations with more than 26 equations.")
        result = linear_system_exercise(variables, get_solution=with_solution, digits_after=digits_after)
        if with_solution:
            exercise, solution = result
        else:
            exercise, solution = result, None
        super(PDFLinearSystem, self).__init__(exercise, exercise_type="system of equations", dtype='linear',
                                              solution=solution, lang=lang)


class PDFLinearFromPoints(PDFAnalyzeFunction):
    def __init__(self, with_solution: bool = True, lang: str = "en"):
        result = linear_from_points_exercise(get_solution=with_solution, lang=lang)
        if with_solution:
            exercise, solution = result
        else:
            exercise, solution = result, None

        super(PDFLinearFromPoints, self).__init__(exercise, dtype='linear', solution=solution, lang=lang)


class PDFLinearFromPointAndSlope(PDFAnalyzeFunction):
    def __init__(self, with_solution: bool = True, lang: str = "en"):
        result = linearFromPointAndSlope_exercise(get_solution=with_solution, lang=lang)
        if with_solution:
            exercise, solution = result
        else:
            exercise, solution = result, None

        super(PDFLinearFromPointAndSlope, self).__init__(exercise, dtype='linear', solution=solution, lang=lang)


class PDFPolyFunction(PDFAnalyzeFunction):
    def __init__(self, with_solution: bool = True, degree: int = None, lang: str = 'en'):
        if lang != 'en':  # Translate the exercise for other languages if needed!
            translator = Translator()
        if degree is None:
            degree = random.randint(2, 5)
        random_poly, solutions = random_polynomial(degree=degree, get_solutions=True)
        random_function = f"f(x) = {random_poly}"
        exercise = f""" The function {random_function} is given.
            a) What is the domain of the function?
            b) What is the derivative of the function?
            c) What are the extremums of the function?
            d) When is the function increasing, and when is it decreasing?
            e) Find the horizontal asymptotes of the function (if there are any).
            f) sketch the function.
        """
        if lang != 'en':  # Translate the exercise for other languages if needed!
            translation = translator.translate(exercise, dest=lang)
            exercise = translation.text
            print(exercise)
        if with_solution:
            my_poly = Poly(random_poly)
            data = my_poly.data(no_roots=True)
            data['roots'] = solutions
            extremums_string = ", ".join(extremum.__str__() for extremum in data['extremums'])
            if not extremums_string:
                extremums_string = 'None'
            solution = f"""
            a. Domain: all 
            b. Derivative: {data['derivative']}
            c. Extremums: {extremums_string}
            d. Increase & Decrease: Increase: {data['up']}, Decrease: {data['down']}
            e. Horizontal Asymptotes: Not Supported yet
            f. Sketch: Not supported yet in this format.
             """
            if lang != 'en':
                translation = translator.translate(solution, dest=lang)
                solution = translation.text
        else:
            solution = None
        super(PDFPolyFunction, self).__init__(exercise, dtype='poly', solution=solution, lang=lang)


class PDFQuadraticFunction(PDFPolyFunction):
    def __init__(self, with_solution: bool = True, lang: str = "en"):
        super(PDFQuadraticFunction, self).__init__(with_solution=with_solution, degree=2, lang=lang)


class PDFCubicFunction(PDFPolyFunction):
    def __init__(self, with_solution: bool = True, lang: str = "en"):
        super(PDFCubicFunction, self).__init__(with_solution=with_solution, degree=3, lang=lang)


class PDFQuarticFunction(PDFPolyFunction):
    def __init__(self, with_solution: bool = True, lang: str = "en"):
        super(PDFQuarticFunction, self).__init__(with_solution=with_solution, degree=4, lang=lang)


class PDFEquationExercise(PDFExercise):
    def __init__(self, exercise: str, dtype: str, solution=None, number: int = None):
        super(PDFEquationExercise, self).__init__(exercise, "equation", dtype, solution, number)


class PDFLinearEquation(PDFEquationExercise):
    def __init__(self, with_solution=True, number: int = None):
        if with_solution:
            equation, solution = LinearEquation.random_equation(digits_after=1, get_solution=True)
        else:
            equation, solution = LinearEquation.random_equation(digits_after=1, get_solution=False), None

        super(PDFLinearEquation, self).__init__(equation, dtype='linear', solution=solution, number=number)


class PDFQuadraticEquation(PDFEquationExercise):
    def __init__(self, with_solution=True, number: int = None):
        if with_solution:
            equation, solutions = random_polynomial(degree=2, get_solutions=True)
        else:
            equation, solutions = random_polynomial(degree=2), None
        equation += " = 0"
        super(PDFQuadraticEquation, self).__init__(equation, dtype='quadratic', solution=solutions, number=number)


class PDFCubicEquation(PDFEquationExercise):
    def __init__(self, with_solution=True, number: int = None):
        if with_solution:
            equation, solutions = random_polynomial(degree=3, get_solutions=True)
        else:
            equation, solutions = random_polynomial(degree=3), None
        equation += " = 0"
        super(PDFCubicEquation, self).__init__(equation, dtype='cubic', solution=solutions, number=number)


class PDFQuarticEquation(PDFEquationExercise):
    def __init__(self, with_solution=True, number: int = None):
        if with_solution:
            equation, solutions = random_polynomial(degree=4, get_solutions=True)
        else:
            equation, solutions = random_polynomial(degree=4), None
        equation += " = 0"
        super(PDFQuarticEquation, self).__init__(equation, dtype='quartic', solution=solutions, number=number)


class PDFPolyEquation(PDFEquationExercise):
    def __init__(self, with_solution=True, number: int = None):
        if with_solution:
            equation, solutions = random_polynomial(degree=random.randint(2, 5), get_solutions=True)
        else:
            equation, solutions = random_polynomial(degree=random.randint(2, 5)), None
        equation += " = 0"
        super(PDFPolyEquation, self).__init__(equation, dtype='poly', solution=solutions, number=number)


class PDFPage:
    def __init__(self, title="Worksheet", exercises=None):
        self.__title = title
        if exercises is None:
            self.__exercises = []
        else:
            self.__exercises = exercises

    @property
    def exercises(self):
        return self.__exercises

    @property
    def title(self):
        return self.__title

    def add(self, exercise):
        self.__exercises.append(exercise)

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):
        if self.__index >= len(self.__exercises):
            raise StopIteration
        temp = self.__index
        self.__index += 1
        return self.__exercises[temp]


class PDFWorksheet:
    __slots__ = ['__pages', '__ordered', '__current_page', '__lines', '__title', '__num_of_exercises']

    def __init__(self, title="Worksheet", ordered=True):
        self.__pages = [PDFPage(title=title)]
        self.__ordered = ordered
        self.__current_page = self.__pages[0]
        self.__lines = [[]]
        self.__title = title
        self.__num_of_exercises = 0

    @property
    def num_of_pages(self):
        return len(self.__pages)

    @property
    def pages(self):
        return self.__pages

    def del_last_page(self):
        if len(self.__pages):
            del self.__pages[-1]

    @property
    def current_page(self):
        return self.__current_page

    def add_exercise(self, exercise):
        self.__num_of_exercises += 1
        self.__current_page.add(exercise)
        if '\n' in exercise.__str__():
            lines = exercise.__str__().split('\n')
            if self.__ordered:
                exercise.number = self.__num_of_exercises
                self.__lines[-1].append(f"{exercise.number}.    {lines[0]}")
            else:
                self.__lines[-1].append(lines[0])

            for i in range(1, len(lines)):
                self.__lines[-1].append(lines[i])

            self.__lines[-1].append("")  # separator
        else:
            if self.__ordered:
                exercise.number = self.__num_of_exercises
                self.__lines[-1].append(f"{exercise.number}.    {exercise.__str__()}")
            else:
                self.__lines[-1].append(exercise.__str__())

    def end_page(self):
        if any(exercise.has_solution for exercise in self.__current_page.exercises):
            solutions_string = []
            for index, exercise in enumerate(self.__current_page.exercises):
                if exercise.solution is None:
                    continue
                if not isinstance(exercise.solution, (int, float, str)) and isinstance(exercise.solution, Iterable):
                    str_solution = ",".join(str(solution) for solution in exercise.solution)
                    if self.__ordered:
                        solutions_string.append(f"{exercise.number}.    {str_solution}")
                    else:
                        solutions_string.append(str_solution)
                else:
                    if not isinstance(exercise.solution, str):
                        str_solution = str(exercise.solution)
                    else:
                        str_solution = exercise.solution

                    if "\n" in str_solution:
                        lines = exercise.solution.split("\n")
                        solutions_string.append(f"{exercise.number}. {lines[0]}" if self.__ordered else f"{lines[0]}")
                        for j in range(1, len(lines)):
                            solutions_string.append(lines[j])
                        solutions_string.append("")
                    else:
                        if self.__ordered:
                            solutions_string.append(f"{exercise.number}.    {exercise.solution}")
                        else:
                            solutions_string.append(f"{exercise.solution}")

            self.__pages.append(PDFPage(title="Solutions", exercises=solutions_string))
            self.__lines.append(solutions_string)

    def next_page(self, title=None):
        if title is None:
            title = self.__title
        self.__pages.append(PDFPage(title))
        self.__current_page = self.__pages[-1]
        self.__lines.append([])

    def create(self, path: str = None):
        if path is None:
            path = generate_pdf_path()
        create_pages(path, self.num_of_pages, [page.title for page in self.__pages], self.__lines)


def create(expression: str, dtype: str = 'poly'):
    if dtype == 'poly':
        return Poly(expression)
    elif dtype == 'log':
        return Log(expression)
    elif dtype == 'ln':
        return Ln(expression)
    elif dtype == 'trigo':
        return TrigoExprs(expression)
    elif dtype == 'root':
        return Root(expression)  # TODO: implement string constructor in root via dtype as well
    elif dtype == 'factorial':
        return Factorial(expression)  # TODO: implement string constructor in root via dtype as well
    else:
        raise ValueError(f"Invalid parameter 'dtype': {dtype}")


def create_from_dict(given_dict: dict):
    if isinstance(given_dict, int):
        return Mono(given_dict)
    expression_type = given_dict['type'].lower()
    if expression_type == 'mono':
        return Mono.from_dict(given_dict)
    elif expression_type == 'poly':
        return Poly.from_dict(given_dict)
    elif expression_type == 'trigoexpr':
        return TrigoExpr.from_dict(given_dict)
    elif expression_type == 'trigoexprs':
        return TrigoExprs.from_dict(given_dict)
    elif expression_type == 'log':
        return Log.from_dict(given_dict)
    elif expression_type == 'factorial':
        return Factorial.from_dict(given_dict)
    elif expression_type == 'root':
        return Root.from_dict(given_dict)
    elif expression_type == 'abs':
        return Abs.from_dict(given_dict)
    elif expression_type == 'iexpressions':
        return ExpressionSum.from_dict(given_dict)
    elif expression_type == 'fraction':
        return Fraction.from_dict(given_dict)
    elif expression_type == 'fastpoly':
        return FastPoly.from_dict(given_dict)
    elif expression_type == 'exponent':
        return Exponent.from_dict(given_dict)
    else:
        raise ValueError(f"Invalid type of expression: {given_dict}")


def mav(func1: Callable, func2: Callable, start: float, stop: float, step: float):
    """ Mean absolute value"""
    my_sum = 0
    num_of_points = 0
    for value in decimal_range(start=start, stop=stop, step=step):
        my_sum += abs(func1(value) - func2(value))
        num_of_points += 1
    if num_of_points == 0:
        raise ZeroDivisionError("Cannot process 0 points")
    return my_sum / num_of_points


def msv(func1: Callable, func2: Callable, start: float, stop: float, step: float):
    """mean square value"""
    my_sum = 0
    num_of_points = 0
    for value in decimal_range(start=start, stop=stop, step=step):
        my_sum += (func1(value) - func2(value)) ** 2
        num_of_points += 1
    if num_of_points == 0:
        raise ZeroDivisionError("Cannot process 0 points")
    return my_sum / num_of_points


def mrv(func1: Callable, func2: Callable, start: float, stop: float, step: float):
    """ mean root value """
    my_sum = 0
    num_of_points = 0
    for value in decimal_range(start=start, stop=stop, step=step):
        my_sum += (func1(value) - func2(value)) ** 2
        num_of_points += 1
    if num_of_points == 0:
        raise ZeroDivisionError("Cannot process 0 points")
    return sqrt(my_sum / num_of_points)


@contextmanager
def copy(expression):  # TODO: how to do the exception handling correctly? is this right ??
    try:
        copy_method = getattr(expression, "__copy__", None)  # check for __copy__() method
        if callable(copy_method):
            copy_of_expression = expression.__copy__()
            yield copy_of_expression
        else:
            copy_method = getattr(expression, "copy", None)  # check for copy() method
            if callable(copy_method):
                copy_of_expression = expression.copy()
                yield copy_of_expression
    finally:
        del copy_of_expression





class ExpressionMul(IExpression, IPlottable, IScatterable):
    __slots__ = ['_coefficient', '_expressions']

    def __init__(self, expressions: Union[Iterable[Union[IExpression, float, int, str]], str], gen_copies=True):
        if isinstance(expressions, str):
            raise NotImplementedError  # TODO: classify expressions types and use string analysis methods
        else:
            self._expressions = list()
            for expression in expressions:
                if isinstance(expression, (float, int)):
                    self._expressions.append(Mono(expression))
                elif isinstance(expression, IExpression):
                    if gen_copies:
                        self._expressions.append(
                            expression.__copy__())  # TODO: copy method could possibly lead to infinite recursion?
                    else:
                        self._expressions.append(expression)
                elif isinstance(expression, str):
                    # TODO: implement a string analysis method
                    raise NotImplementedError
                else:
                    raise TypeError(f"Encountered an invalid type: '{type(expression)}', when creating a new "
                                    f"Expression object.")

    @property
    def expressions(self):
        return self._expressions
    def assign(self, **kwargs):
        for expression in self._expressions:
            expression.assign(**kwargs)

    def python_syntax(self):
        if not self._expressions:
            return self._coefficient.python_syntax()
        accumulator = f"({self._coefficient})*"
        for iexpression in self._expressions:
            accumulator += f"({iexpression.python_syntax()})*"
        return accumulator[:-1]

    def simplify(self):
        if self._coefficient == 0:
            self._expressions = []

    @property
    def variables(self):
        variables = set()
        for expression in self._expressions:
            variables.update(expression.variables)
        return variables

    def try_evaluate(self):
        evaluated_expressions = [expression.try_evaluate() for expression in self._expressions]
        if all(evaluated_expressions):
            return sum(evaluated_expressions)  # TODO: check if this actually works!

    def __split_expressions(self, num_of_expressions:int):
        return ExpressionMul(self._expressions[:num_of_expressions // 2]), ExpressionMul(
            self._expressions[num_of_expressions // 2:])

    def derivative(self):
        print(f"calculating the derivative of {self}, num of expressions: {len(self._expressions)}")
        # Assuming all the expressions can be derived
        num_of_expressions = len(self._expressions)
        if num_of_expressions == 0: # No expressions, then no derivative!
            return None
        if num_of_expressions == 1:
            return self._expressions[0].derivative()
        elif num_of_expressions == 2:
            first, second = self._expressions[0], self._expressions[1]
            return first.derivative() * second + second.derivative() * first
        else: # more than 2 expressions
            expressionMul1, expressionMul2 = self.__split_expressions(num_of_expressions)
            first_derivative, second_derivative = expressionMul1.derivative(), expressionMul2.derivative()
            if isinstance(first_derivative, (int, float)):
                first_derivative = Mono(first_derivative)
            if isinstance(second_derivative, (int, float)):
                second_derivative = Mono(second_derivative)
            return first_derivative * expressionMul2 + second_derivative * expressionMul1

    def __copy__(self):
        return ExpressionMul(self._expressions)

    def __neg__(self):
        copy_of_self = self.__copy__()
        copy_of_self._coefficient *= 1
        return copy_of_self

    def __iadd__(self, other):
        return ExpressionSum((self, other))

    def __isub__(self, other):
        return ExpressionSum((self, other.__neg__()))

    def __imul__(self, other):
        self._expressions.append(other)
        return self

    def __itruediv__(self, other):
        return Fraction(self, other)

    def __rtruediv__(self, other):
        return Fraction(other, self)

    def __ipow__(self, power):
        for index, expression in enumerate(self._expressions):
            self._expressions[index] = expression.__pow__(power)
        return self

    def __rpow__(self, other):  # TODO: Implement exponents for that
        return Exponent(other, self)

    def __str__(self) -> str:
        accumulator = f""
        for index, expression in enumerate(self._expressions):
            content = expression.__str__()
            if index > 0 and not content.startswith('-'):
                content = f"*{content}"
            if not content.endswith(')'):
                content = f'({content})'  # Add parenthesis to clarify the order of actions
            accumulator += content
        return accumulator

    def __eq__(self, other: Union[IExpression, int, float]) -> bool:
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            return my_evaluation is not None and my_evaluation == other
        elif isinstance(other, IExpression):
            # First check equality with the evaluations, if the expressions can be evaluated
            other_evaluation = other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                return my_evaluation == other_evaluation
            if isinstance(other, ExpressionMul):
                pass  # TODO: Use an outside method for this complicated equality checking, like the TrigoExpr
            else:
                if len(self._expressions) == 1 and self._expressions[0] == other:
                    return True
                # Add more checks ?
                return False
        else:
            raise TypeError(f"Invalid type {type(other)} for equality checking in Expression class")

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_dict(self):
        pass

    def from_dict(self):
        pass


class ExpressionSum(IExpression, IPlottable, IScatterable):
    __slots__ = ['_expressions', '_current_index']

    def __init__(self, expressions: Iterable[IExpression] = None, copy=True):
        self._current_index = 0
        if expressions is None:
            self._expressions = []
        else:
            if copy:
                self._expressions = [copy_expression(expression) for expression in expressions]
            else:
                self._expressions = [expression for expression in expressions]
        # Now we need to check for any "ExpressionSum" object to unpack, and for numbers
        expressions_to_add = []
        indices_to_delete = []
        for index, expression in enumerate(self._expressions):
            if isinstance(expression, ExpressionSum):
                expressions_to_add.extend(expression._expressions)  # the expressions should be unpacked
                indices_to_delete.append(index)
            elif isinstance(expression, (int, float)):
                self._expressions[index] = Mono(expression)
        self._expressions = [expression for index, expression in enumerate(self._expressions) if index not in indices_to_delete]
        self._expressions.extend(expressions_to_add)

    @property
    def expressions(self):
        return self._expressions

    def append(self, expression: IExpression):
        self._expressions.append(expression)

    def assign_to_all(self, **kwargs):
        for expression in self._expressions:
            expression.assign(**kwargs)

    def when_all(self, **kwargs):
        return ExpressionSum((expression.when(**kwargs) for expression in self._expressions), copy=False)
        # Prevent unnecessary double copying

    def __add_or_sub(self, other: "Union[IExpression, ExpressionSum]", operation='+'):
        if isinstance(other, (int, float)):
            my_evaluation = self.try_evaluate()
            if my_evaluation is not None:
                if operation == '+':
                    return Mono(my_evaluation + other)
                else:
                    return Mono(my_evaluation - other)
            if operation == '+':
                self._expressions.append(Mono(other))
            else:
                self._expressions.append(Mono(-other))
            self.simplify()
            return self
        elif isinstance(other, IExpression):
            my_evaluation, other_evaluation = self.try_evaluate(), other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                return Mono(my_evaluation + other_evaluation)
            elif my_evaluation is not None:
                if operation == '+':
                    return other.__add__(my_evaluation)
                else:
                    return other.__sub__(my_evaluation)
            elif other_evaluation is not None:
                if operation == '+':
                    self._expressions.append(Mono(other_evaluation))
                else:
                    self._expressions.append(Mono(-other_evaluation))
                self.simplify()
                return self
            else:
                # nothing can be evaluated.
                if operation == '+':
                    self._expressions.append(other.__copy__())
                else:
                    self._expressions.append(other.__neg__())
                self.simplify()
                return self

            if isinstance(other, ExpressionSum):
                if operation == '+':
                    for expression in other._expressions:
                        self._expressions.append(expression)
                else:
                    for expression in other._expressions:
                        self._expressions.append(expression)

                self.simplify()
                return self

        self.simplify()
        return self

    def __iadd__(self, other: "Union[IExpression, ExpressionSum]"):
        return self.__add_or_sub(other, operation='+')

    def __isub__(self, other: "Union[IExpression, int, float, ExpressionSum]"):
        return self.__add_or_sub(other, operation='-')

    def __rsub__(self, other: "Union[IExpression, int, float, ExpressionSum]"):
        return ExpressionSum((other, -self))

    def __neg__(self):
        return ExpressionSum((expression.__neg__() for expression in self._expressions))

    def __imul__(self, other: "Union[IExpression, int, float, ExpressionSum]"):
        if isinstance(other, ExpressionSum):
            final_expressions: List[Optional[IExpression]] = []
            for my_expression in self._expressions:
                for other_expression in other._expressions:
                    final_expressions.append(my_expression * other_expression)
            result = self.to_poly()
            if result is not None:
                return result
            self.simplify()
            return self
        else:
            for index in range(len(self._expressions)):
                self._expressions[index] *= other

            result = self.to_poly()
            if result is not None:
                return result
            self.simplify()
            return self

    def __ipow__(self, power: Union[IExpression, int, float]):
        if isinstance(power, (int, float)):
            length = len(self._expressions)
            if length == 0:  # No expression
                return None
            if power == 0:
                return Mono(1)
            if length == 1:  # Single expression
                self._expressions[0] **= power
                return self
            if length == 2:
                # Binomial
                if 0 < power < 1:  # Root
                    pass
                elif power > 0:
                    pass
                else:  # Negative powers
                    pass
            elif length > 2:  # More than two items
                if 0 < power < 1:  # Root
                    pass
                elif power > 0:
                    copy_of_self = self.__copy__()
                    for index in range(power - 1):
                        self.__imul__(copy_of_self)
                    return self
                else:  # Negative powers
                    return Fraction(1, self.__ipow__(abs(power)))
        elif isinstance(power, IExpression):  # Return an exponent if the expression can't be evaluated into number
            other_evaluation = power.try_evaluate()
            if other_evaluation is None:
                # TODO: return here an exponent object
                pass
            else:
                return self.__ipow__(other_evaluation)
        else:
            raise TypeError(f"Invalid type '{type(power)}' for raising an 'ExpressionSum' object by a power")

    def __pow__(self, power: Union[IExpression, int, float]):
        return self.__copy__().__ipow__(power)

    def __itruediv__(self, other: Union[IExpression, int, float]) -> "Union[ExpressionSum,IExpression]":
        if other == 0:
            raise ValueError("Cannot divide an ExpressionSum object by 0.")
        if isinstance(other, (int, float)):
            for my_expression in self._expressions:
                my_expression /= other
            return self

        other_evaluation = other.try_evaluate()
        if other_evaluation is not None:
            if other == 0:
                raise ValueError("Cannot divide an ExpressionSum object by 0.")
            for my_expression in self._expressions:
                my_expression /= other_evaluation
            return self

        if isinstance(other, (ExpressionSum, Poly, TrigoExprs)):
            # put the expressions in a Fraction object when encountering collections
            return Fraction(self, other)
        if not isinstance(other, (int, float, IExpression)):
            raise TypeError(f"Invalid type {type(other)} for dividing with 'ExpressionSum' class.")
        for my_expression in self._expressions:
            my_expression /= other
        result = self.to_poly()  # try to simplify to a polynomial (there's heavy support for polynomials)
        if result is not None:
            return result
        self.simplify()
        return self

    def assign(self, **kwargs) -> None:
        for expression in self._expressions:
            expression.assign(**kwargs)

    def is_poly(self):
        return all(isinstance(expression, (Mono, Poly)) for expression in self._expressions)

    def to_poly(self) -> "Optional[Poly]":
        """Tries to convert the ExpressionSum object to a Poly object (to a polynomial).
        If not successful, None will be returned.
        """
        if not self.is_poly():
            return None
        my_poly = Poly(0)
        for expression in self._expressions:
            my_poly += expression
        return my_poly

    def simplify(self):
        for expression in self._expressions:
            expression.simplify()
        evaluation_sum: float = 0
        delete_indices = []
        for index, expression in enumerate(self._expressions):
            expression_evaluation = expression.try_evaluate()
            if expression_evaluation is not None:
                evaluation_sum += expression_evaluation
                delete_indices.append(index)
        self._expressions = [expression for index, expression in enumerate(self._expressions) if
                             index not in delete_indices]
        if evaluation_sum:  # if evaluation sum is not 0, add it. ( Because there's no point in adding trailing zeroes)
            self._expressions.append(Mono(evaluation_sum))

    def try_evaluate(self):
        """ Try to evaluate the expressions into float or an int """
        evaluation_sum = 0
        for expression in self._expressions:
            expression_evaluation: Optional[Union[int, float]] = expression.try_evaluate()
            if expression_evaluation is None:
                return None
            evaluation_sum += expression_evaluation
        return evaluation_sum

    @property
    def variables(self):
        variables = set()
        for expression in self._expressions:
            variables.update(variables.union(expression.variables))
        return variables

    def derivative(self):
        warnings.warn("This feature is still experimental, and might not work.")
        if any(not hasattr(expression, 'derivative') for expression in self._expressions):
            raise AttributeError("Not all expressions support derivatives")
        return ExpressionSum([expression.derivative() for expression in self._expressions], copy=False)
        # Prevent unnecessary copies by setting copy to False

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self):
        if self._current_index < len(self._expressions):
            value = self._expressions[self._current_index]
            self._current_index += 1
            return value
        raise StopIteration

    def __getitem__(self, item):
        return self._expressions.__getitem__(item)

    def __len__(self):
        return len(self._expressions)

    def __copy__(self):
        return ExpressionSum(
            (expression.__copy__() for expression in self._expressions))  # Generator for memory saving..

    def __str__(self):
        accumulator = ""
        for expression in self._expressions:
            expression_string: str = expression.__str__()
            if not expression_string.startswith('-'):
                accumulator += "+"
            accumulator += expression_string
        if accumulator[0] == '+':
            return accumulator[1:]
        return accumulator

    def python_syntax(self) -> str:
        accumulator = ""
        for expression in self._expressions:
            expression_string: str = expression.python_syntax()
            if not expression_string.startswith('-'):
                accumulator += "+"
            accumulator += expression_string
        if accumulator[0] == '+':
            return accumulator[1:]
        return accumulator

    def to_dict(self):
        return {
            "type": "ExpressionSum",
            "expressions": [expression.to_dict() for expression in self._expressions]
        }

    def from_dict(self):
        pass

    def __eq__(self, other: Union[IExpression, int, float]):  # TODO: improve this method
        """Tries to figure out whether the expressions are equal. May not apply to special cases such as trigonometric
        identities"""
        if isinstance(other, (int, float)):
            if len(self._expressions) == 1:
                return self._expressions[0] == other
        elif isinstance(other, IExpression):
            if isinstance(other, ExpressionSum):
                if len(self._expressions) != len(other._expressions):
                    return False
                for my_expression in self._expressions:
                    my_count = self._expressions.count(my_expression)
                    other_count = other._expressions.count(my_expression)
                    if my_count != other_count:
                        return False
                return True
            else:  # Equating between ExpressionSum to a single expression
                if len(self._expressions) == 1:
                    return self._expressions[0] == other
                else:
                    other_evaluation = other.try_evaluate()
                    if other_evaluation is None:
                        return False
                    my_evaluation = self.try_evaluate()
                    if my_evaluation is None:
                        return False
                    # If reached here, both expressions can be evaluated
                    return my_evaluation == other_evaluation

    def __ne__(self, other: Union[IExpression, int, float]):
        return not self.__eq__(other)


def __helper_trigo(expression: str) -> Optional[Tuple[int, Optional[float]]]:
    try:
        first_letter_index = expression.find(next(
            (character for character in expression if character.isalpha() and character not in ('e', 'i'))))
        return first_letter_index, float(str(extract_coefficient(expression[:first_letter_index])))
    except (StopIteration, ValueError):
        print(expression)
        return None


def analyze_single_trigo(trigo_expression: str, get_tuple=False, dtype='poly'):
    """
    Generates a TrigoExpr object from a string with a simplified trigonometric expression, such as sin(5x+7), or sin(45)

    :param trigo_expression: the string
    :param get_tuple: if set to True, a tuple of the _coefficient,chosen trigonometric method, and the inside expression will be returned
    :return: a TrigoExpr object corresponding to the string,or a tuple if get_tuple is set to True
    """
    trigo_expression = trigo_expression.strip().replace("**", '^').replace(" ", "")  # To prevent any stupid mistakes
    left_parenthesis_index: int = trigo_expression.find('(')
    right_parenthesis_index: int = trigo_expression.rfind(')')
    first_letter_index, coefficient = __helper_trigo(trigo_expression)
    method_chosen = trigo_expression[first_letter_index:left_parenthesis_index].upper()
    method_chosen = TrigoMethods[method_chosen]
    inside_string = trigo_expression[left_parenthesis_index + 1:right_parenthesis_index]
    inside = create(inside_string, dtype=dtype)
    power_index = trigo_expression.rfind('^')
    if power_index == -1 or power_index < right_parenthesis_index:
        power = 1
    else:
        power = float(trigo_expression[power_index + 1:])
    if get_tuple:
        return coefficient, method_chosen, inside, power
    return TrigoExpr(coefficient, [(method_chosen, inside, power)])


def TrigoExpr_from_str(trigo_expression: str, get_tuple=False,
                       dtype='poly') -> "Union[Tuple[IExpression,List[list]],TrigoExpr]":
    """

    :param trigo_expression:
    :param get_tuple:
    :return:
    """
    trigo_expression = trigo_expression.strip().replace("**", "^")  # Avoid stupid mistakes
    coefficient = Poly(1)
    expressions = [expression for expression in trigo_expression.split('*') if expression.strip() != ""]
    new_expressions = []
    for expression in expressions:
        if is_number(expression):  # Later add support for i and e in the _coefficient detection
            coefficient *= float(expression)
        else:
            new_expressions.append(expression)
    analyzed_generator = (analyze_single_trigo(expression, get_tuple=True, dtype=dtype) for expression in
                          new_expressions)
    analyzed_expressions = []
    for (coef, method_chosen, inside, power) in analyzed_generator:
        analyzed_expressions.append([method_chosen, inside, power])
        coefficient *= coef
    if not analyzed_expressions:
        analyzed_expressions = None
    if get_tuple:
        return coefficient, analyzed_expressions
    return TrigoExpr(coefficient, expressions=analyzed_expressions)


def TrigoExprs_from_str(trigo_expression: str, get_list=False):
    """

    :param trigo_expression:
    :param get_tuple:
    :return:
    """
    trigo_expressions: list = split_expression(trigo_expression)  # TODO: What about the minus in the beginning?
    new_expressions: list = [TrigoExpr_from_str(expression) for expression in trigo_expressions]
    if get_list:
        return new_expressions
    return TrigoExprs(new_expressions)




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
            raise ValueError("expression must contain one item only for cosine conversion: For example, sin(3x)")
        return given_func(self)

    return inner




class Asin(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Asin, self).__init__(coefficient=f"asin{expression}", dtype=dtype)
        super(Asin, self).__init__(1, expressions=((TrigoMethods.ASIN, expression, 1),))


class Cos(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Cos, self).__init__(coefficient=f"cos({expression})", dtype=dtype)
        else:
            super(Cos, self).__init__(1, expressions=((TrigoMethods.COS, expression, 1),))

    @conversion_wrapper
    def to_sin(self) -> "Sin":
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

    @conversion_wrapper
    def to_csc(self) -> "Csc":
        pass


class Acos(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Acos, self).__init__(coefficient=f"acos({expression})", dtype=dtype)
        else:
            super(Acos, self).__init__(1, expressions=((TrigoMethods.ACOS, expression, 1),))


class Tan(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Tan, self).__init__(coefficient=f"tan({expression})", dtype=dtype)
        else:
            super(Tan, self).__init__(1, expressions=((TrigoMethods.TAN, expression, 1),))

    @conversion_wrapper
    def to_sin(self) -> "Sin":
        pass

    @conversion_wrapper
    def to_cos(self) -> "Cos":
        pass

    @conversion_wrapper
    def to_cot(self) -> "Cot":
        pass

    @conversion_wrapper
    def to_sec(self) -> "Sec":
        pass

    @conversion_wrapper
    def to_csc(self) -> "Csc":
        pass


class Atan(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Atan, self).__init__(coefficient=f"atan{expression}", dtype=dtype)
        else:
            super(Atan, self).__init__(1, expressions=((TrigoMethods.ATAN, expression, 1),))


class Cot(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Cot, self).__init__(coefficient=f"cot{expression}", dtype=dtype)
        else:
            super(Cot, self).__init__(1, expressions=((TrigoMethods.COT, expression, 1),))

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
            super(Sec, self).__init__(1, expressions=((TrigoMethods.SEC, expression, 1),))

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
            super(Acot, self).__init__(coefficient=f"asec({expression})", dtype=dtype)
        else:
            super(Acot, self).__init__(1, expressions=((TrigoMethods.ACOT, expression, 1),))


class ASec(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(ASec, self).__init__(coefficient=f"asec({expression})", dtype=dtype)
        else:
            super(ASec, self).__init__(1, expressions=((TrigoMethods.ASEC, expression, 1),))


class Csc(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Csc, self).__init__(coefficient=f"csc({expression})", dtype=dtype)
        else:
            super(Csc, self).__init__(1, expressions=((TrigoMethods.CSC, expression, 1),))

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
            super(ACsc, self).__init__(coefficient=f"acsc({expression})", dtype=dtype)
        else:
            super(ACsc, self).__init__(1, expressions=((TrigoMethods.ACSC, expression, 1),))


def equal_ignore_order(a, b):
    """ Use only when elements are neither hashable nor sortable! """
    if None in (a, b):
        return False
    if len(a) != len(b):
        return False
    unmatched = list(b)
    for element in a:
        try:
            unmatched.remove(element)
        except ValueError:
            return False
    return not unmatched


class TrigoExprs(ExpressionSum, IPlottable, IScatterable):
    def _transform_expression(self, expression, dtype='poly'):
        if isinstance(expression, (int, float)):
            return Mono(expression)
        elif isinstance(expression, str):
            return create(expression, dtype=dtype)
        elif isinstance(expression, (IExpression)):
            return expression.__copy__()
        else:
            raise TypeError(f"Unexpected type {type(expression)} in TrigoExpr.__init__()")

    def __init__(self, expressions: Union[str, Iterable[IExpression]], dtype='poly'):
        if isinstance(expressions, str):
            expressions = TrigoExprs_from_str(expressions, get_list=True)
        if isinstance(expressions, Iterable):
            expressions_list = []
            # print(f"received {[expression.__str__() for expression in expressions]} in init")
            for index, expression in enumerate(expressions):
                try:
                    matching_index = next(index for index, existing in enumerate(expressions_list) if
                                          equal_ignore_order(existing.expressions, expression.expressions))
                    expressions_list[matching_index]._coefficient += expression.coefficient
                except StopIteration:
                    expressions_list.append(expression)
            # print([f"{expr.__str__()}" for expr in expressions_list])
            super(TrigoExprs, self).__init__(
                [self._transform_expression(expression, dtype=dtype) for expression in expressions_list])
        else:
            raise TypeError(
                f"Unexpected  type {type(expressions)} in TrigoExpr.__init__(). Expected an iterable collection or a "
                f"string.")
        # now 

    def __add_TrigoExpr(self, other: TrigoExpr):  # TODO: check this methods
        """Add a TrigoExpr expression"""
        try:
            index = next((index for index, expression in enumerate(self._expressions) if
                          expression.expressions == other.expressions))  # Find the first matching expression
            self._expressions[index]._coefficient += other.coefficient
        except StopIteration:
            # No matching expression
            self._expressions.append(other)

    def __sub_TrigoExpr(self, other: TrigoExpr):
        """ Subtract a TrigoExpr expression"""
        try:
            index = next((index for index, expression in enumerate(self._expressions) if
                          expression.expressions == other.expressions))  # Find the first matching expression
            self._expressions[index]._coefficient -= other.coefficient
        except StopIteration:
            # No matching expression
            self._expressions.append(other)

    def __iadd__(self, other: Union[int, float, IExpression]):
        print(f"adding {other} to {self}")
        if other == 0:
            return self
        if isinstance(other, str):
            other = TrigoExprs(other)
        if isinstance(other, IExpression):
            my_evaluation, other_evaluation = self.try_evaluate(), other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                return Mono(my_evaluation + other_evaluation)

            if isinstance(other, TrigoExpr):
                self.__add_TrigoExpr(other)
                return self
            elif isinstance(other, TrigoExprs):
                for other_expression in other._expressions:
                    self.__add_TrigoExpr(other_expression)
                return self
            else:
                return ExpressionSum((self, other))
        else:
            raise TypeError(f"Invalid type for adding trigonometric expressions: {type(other)}")

    def __isub__(self, other: Union[int, float, IExpression]):
        if isinstance(other, str):
            other = TrigoExprs(other)
        if isinstance(other, IExpression):
            my_evaluation, other_evaluation = self.try_evaluate(), other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                return Mono(my_evaluation - other_evaluation)
            if isinstance(other, TrigoExpr):
                self.__sub_TrigoExpr(other)
                return self
            elif isinstance(other, TrigoExprs):
                for other_expression in other._expressions:
                    self.__sub_TrigoExpr(other_expression)
                    return self
            else:
                return ExpressionSum((self, other))
        else:
            raise TypeError(f"Invalid type for subtracting trigonometric expressions: {type(other)}")

    @property
    def variables(self):
        variables = set()
        for trigo_expression in self._expressions:
            variables.update(trigo_expression.variables)
        return variables

    def flip_signs(self):
        for expression in self._expressions:
            expression *= -1

    def __neg__(self):
        copy_of_self = self.__copy__()
        copy_of_self.flip_signs()
        return copy_of_self

    def __rsub__(self, other):
        if isinstance(other, IExpression):
            return other.__sub__(self)
        elif isinstance(other, (int, float)):
            pass  # TrigoExprs or TrigoExpr
        else:
            raise TypeError(f"Invalid type while subtracting a TrigoExprs object: {type(other)} ")

    def __imul__(self, other: Union[int, float, IExpression]):
        if isinstance(other, (int, float, IExpression)):
            if isinstance(other, IExpression):
                other_evaluation = other.try_evaluate()
                value = other_evaluation if other_evaluation is not None else other
            else:
                value = other
            if isinstance(value, TrigoExprs):
                expressions_list = []
                for other_expression in value.expressions:
                    for my_expression in self._expressions:
                        expressions_list.append(my_expression * other_expression)
                return TrigoExprs(expressions_list)
            else:
                for index in range(len(self._expressions)):
                    self._expressions[index] *= value
                return self
        else:
            raise TypeError(f"Invalid type '{type(other)}' when multiplying a TrigoExprs object.")

    def __mul__(self, other):
        return self.__copy__().__imul__(other)

    def __rmul__(self, other):  # same as mul
        return self.__mul__(other)

    def __itruediv__(self, other: Union[int, float, IExpression]):  # TODO: implement it
        my_evaluation = self.try_evaluate()
        if other == 0:
            raise ZeroDivisionError("Cannot divide a TrigoExprs object by 0")
        if isinstance(other, (int, float)):
            if my_evaluation is None:
                for trigo_expression in self._expressions:
                    trigo_expression /= other
                return self
            else:
                return Mono(coefficient=my_evaluation / other)
        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if other_evaluation is not None:
                if my_evaluation is not None:
                    return Mono(coefficient=my_evaluation / other_evaluation)
                else:
                    for trigo_expression in self._expressions:
                        trigo_expression /= other
                    return self
            elif my_evaluation is not None:  # the current instance represents a number but 'other' does not.
                # Therefore, a case such as 1/x will be created, which can be represented via a Fraction object.
                return Fraction(self, other)
            elif isinstance(other, TrigoExpr):
                if len(self._expressions) == 1:
                    return self._expressions[0] / other
                for [method, inside, power] in other.expressions:
                    for trigo_expression in self._expressions:
                        found = False
                        for [method1, inside1, power1] in trigo_expression.expressions:
                            if (method, inside) == (method1, inside1) and power1 >= power:
                                found = True
                                break

                        if not found:
                            return Fraction(self, other)
                for [method, inside, power] in other.expressions:
                    for trigo_expression in self._expressions:
                        delete_indices = []
                        for index, [method1, inside1, power1] in enumerate(trigo_expression.expressions):
                            if (method, inside) == (method1, inside1) and power1 >= power:
                                trigo_expression.expressions[index][2] -= power
                                if trigo_expression.expressions[index][2] == 0:  # we can cancel results like sin(x)^2
                                    delete_indices.append(index)
                        if delete_indices:
                            trigo_expression._expressions = [item for index, item in
                                                             enumerate(trigo_expression.expressions) if
                                                             index not in delete_indices]

                print("done!")
                return self

            elif isinstance(other, TrigoExprs):
                # First of all, check for equality, then return 1
                if all(trigo_expr in other.expressions for trigo_expr in self._expressions) and all(
                        trigo_expr in self.expressions for trigo_expr in other._expressions):
                    return Mono(1)
                # TODO: Further implement it.
                return Fraction(self, other)

            else:
                return Fraction(self, other)
        else:
            raise TypeError(f"Invalid type '{type(other)}' for dividing a TrigoExprs object")

    def __pow__(self, power: Union[int, float, IExpression]):  # Check if this works
        if isinstance(power, IExpression):
            power_evaluation = power.try_evaluate()
            if power_evaluation is not None:
                power = power_evaluation
            else:
                return Exponent(self, power)
        if power == 0:
            return TrigoExpr(1)
        elif power == 1:
            return self.__copy__()
        items = len(self._expressions)
        if items == 1:
            return self._expressions[0].__copy__().__ipow__()
        elif items == 2:
            expressions = []
            # Binomial theorem
            for k in range(power + 1):
                comb_result = comb(power, k)
                first_power, second_power = power - k, k
                first = self._expressions[0] ** first_power
                first *= comb_result
                first *= self._expressions[1] ** second_power
                expressions.append(first)
            return TrigoExprs(expressions)
        elif items > 2:
            for i in range(power - 1):
                self.__imul__(self)  # Outstanding move!
            return self
        else:
            raise ValueError(f"Cannot raise an EMPTY TrigoExprs object by the power of {power}")

    def assign(self, **kwargs):  # TODO: implement it
        for trigo_expression in self._expressions:
            trigo_expression.assign(**kwargs)
        self.simplify()

    def try_evaluate(self, **kwargs):  # TODO: implement it
        self.simplify()  # Simplify first
        evaluation_sum = 0
        for trigo_expression in self._expressions:
            trigo_evaluation = trigo_expression.try_evaluate()
            if trigo_evaluation is None:
                return None
            evaluation_sum += trigo_evaluation
        return evaluation_sum

    def to_lambda(self):
        return to_lambda(self.python_syntax(), self.variables)

    def plot(self, start: float = -8, stop: float = 8,
             step: float = 0.3, ymin: float = -3, ymax: float = 3, title=None, show_axis=True, show=True,
             fig=None, ax=None, formatText=True, values=None):
        variables = self.variables
        num_of_variables = len(variables)
        if num_of_variables == 1:
            plot_function(self.to_lambda(), start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title,
                          show_axis=show_axis, show=show, fig=fig, ax=ax, formatText=formatText, values=values)
        elif num_of_variables == 2:
            plot_function_3d(given_function=self.to_lambda(), start=start, stop=stop, step=step)
        else:
            raise ValueError(f"Cannot plot a trigonometric expression with {num_of_variables} variables.")

    def to_dict(self):
        return {'type': 'TrigoExprs', 'data': [expression.to_dict() for expression in self._expressions]}

    @staticmethod
    def from_dict(given_dict: dict):
        return TrigoExprs([TrigoExpr.from_dict(sub_dict) for sub_dict in given_dict['expressions']])

    def __eq__(self, other):
        result = super(TrigoExprs, self).__eq__(other)
        if result:
            return result
        # TODO: HANDLE TRIGONOMETRIC IDENTITIES

    def __str__(self):
        return "+".join((expression.__str__() for expression in self._expressions)).replace("+-", "-")

    def __copy__(self):
        return TrigoExprs(self._expressions.copy())


def synthetic_division(coefficients: list, number: float):
    """performs a division in order for polynomial equation solving"""
    new_list = []
    result = 0
    for coefficient in coefficients:
        value = coefficient + result * number
        new_list.append(value)
        result = value
    if new_list[-1] == 0:
        del new_list[-1]
    if new_list is None:
        return []
    return new_list, result


class Factorial(IExpression, IPlottable, IScatterable):
    __slots__ = ['_coefficient', '_expression', '_power']

    def __init__(self, expression: Optional[Union[IExpression, int, float, str]],
                 coefficient: Union[IExpression, int, float] = Mono(1),
                 power: Union[IExpression, int, float] = Mono(1), dtype=''):

        if isinstance(coefficient, (int, float)):
            self._coefficient = Mono(coefficient)
        else:
            self._coefficient = coefficient.__copy__()

        if isinstance(power, (int, float)):
            self._power = Mono(power)
        else:
            self._power = power.__copy__()

        if isinstance(expression, (int, float)):
            self._expression = Mono(expression)
        else:
            self._expression = expression.__copy__()

    @property
    def coefficient(self):
        return self._coefficient

    @property
    def expression(self):
        return self._expression

    @property
    def power(self):
        return self._power

    @property
    def variables(self):
        """ A set of all of the existing variables_dict inside the expression"""
        coefficient_variables: set = self._coefficient.variables
        coefficient_variables.update(self._expression.variables)
        coefficient_variables.update(self._power.variables)
        return coefficient_variables

    def to_dict(self):
        return {
            "type": "Factorial",
            "coefficient": self._coefficient.to_dict(),
            "expression": self._expression.to_dict(),
            "power": self._power.to_dict()
        }

    @staticmethod
    def from_dict(given_dict: dict):
        expression_obj = create_from_dict(given_dict['expression'])
        coefficient_obj = create_from_dict(given_dict['coefficient'])
        power_obj = create_from_dict(given_dict['power'])
        return Factorial(expression=expression_obj, power=power_obj, coefficient=coefficient_obj)

    def __iadd__(self, other: Union[int, float, IExpression]):
        if other == 0:
            return self
        if isinstance(other, (int, float)):
            other = Mono(other)
        if isinstance(other, Factorial):
            if self._expression == other._expression and self._power == other._power:
                self._coefficient += other._coefficient
                return self
        return ExpressionSum((self, other))

    def __isub__(self, other):
        if other == 0:
            return self
        if isinstance(other, (int, float)):
            other = Mono(other)
        if isinstance(other, Factorial):
            if self._expression == other._expression and self._power == other._power:
                self._coefficient -= other._coefficient
                return self
        return ExpressionSum((self, other))

    def __imul__(self, other: Union[IExpression, int, float]):
        if self._expression == other - 1:
            self._expression += 1
            return self
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            if my_evaluation is None:
                self._coefficient *= other
                return self
            return Mono(coefficient=my_evaluation * other)
        elif isinstance(other, IExpression):
            if isinstance(other, Factorial):
                if self._expression == other._expression:
                    self._coefficient *= other._coefficient
                    self._power += other._power
                    return self
                else:
                    return ExpressionSum((self, other))
            other_evaluation = other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                return Mono(coefficient=my_evaluation * other_evaluation)
            else:
                self._coefficient *= other
                return self
        else:
            raise TypeError(f"Invalid type '{type(other)}' when multiplying a factorial object with id:{id(other)}")

    def __mul__(self, other: Union[IExpression, int, float]):
        return self.__copy__().__imul__(other)

    def __itruediv__(self, other: Union[IExpression, int, float]) -> "Optional[Union[Fraction,Factorial]]":
        if other == 0:
            raise ZeroDivisionError("Cannot divide a factorial expression by 0")
        if other == self._expression:  # For example: 8! / 8 = 7!
            if other == self._coefficient:
                self._coefficient = Mono(1)
                return self
            if isinstance(other, IExpression):
                division_with_coefficient = self._coefficient / other
                division_eval = division_with_coefficient.try_evaluate()
                if division_eval is not None:
                    self._coefficient = Mono(division_eval)
                    return self

            self._expression -= 1
            self.simplify()
            return self

        if isinstance(other, (int, float)):
            self._coefficient /= other
            self.simplify()
            return self

        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if other_evaluation is not None:
                # The expression can be evaluated into a float or an int
                if other_evaluation == 0:
                    raise ZeroDivisionError("Cannot divide a factorial expression by 0")
                if other_evaluation == self._expression:
                    self._expression -= 1
                    self.simplify()
                    return self
                else:
                    self._coefficient /= other
                    self.simplify()
                    return self
            elif isinstance(other, Factorial):  # TODO: poorly implemented!
                if self._expression == other._expression:
                    self._coefficient /= other._coefficient
                    self._power -= other._power
                    self.simplify()

            else:  # Just a random IExpression - just return a Fraction ..
                return Fraction(self, other)

        else:
            raise TypeError(f"Invalid type for dividing factorials: '{type(other)}'")

    def __rtruediv__(self, other: Union[int, float, IExpression]):
        my_evaluation = self.try_evaluate()
        if my_evaluation == 0:
            raise ZeroDivisionError("Cannot divide by 0: Tried to divide by a Factorial expression that evaluates"
                                    "to zero")
        if isinstance(other, (int, float)):
            if my_evaluation is not None:
                return Mono(other / my_evaluation)
            return Fraction(other, self)
        elif isinstance(other, IExpression):
            return other.__truediv__(self)
        else:
            raise TypeError("Invalid type for dividing an expression by a Factorial object.")

    def __ipow__(self, other: Union[int, float, IExpression]):
        self._power *= other
        return self

    def __pow__(self, power):
        return self.__copy__().__ipow__(power)

    def __neg__(self):
        if self._expression is None:
            return Factorial(
                coefficient=self._coefficient.__neg__(),
                expression=None,
                power=Mono(1)
            )
        return Factorial(
            coefficient=self._coefficient.__neg__(),
            expression=self._expression.__neg__(),
            power=self._power.__neg__()
        )

    def assign(self, **kwargs):
        self._coefficient.assign(**kwargs)
        self._expression.assign(**kwargs)
        self._power.assign(**kwargs)
        self.simplify()

    def try_evaluate(self) -> Optional[Union[int, float]]:
        if self._coefficient == 0:
            return 0
        coefficient_evaluation = self._coefficient.try_evaluate()
        if self._expression is None:
            if coefficient_evaluation is not None:
                return coefficient_evaluation
            return None
        expression_evaluation = self._expression.try_evaluate()
        power_evaluation = self._power.try_evaluate()
        if None not in (coefficient_evaluation, expression_evaluation, power_evaluation):  # If all can be evaluated
            if expression_evaluation < 0:
                return None  # Cannot evaluate negative factorials
            if expression_evaluation == 0:
                my_factorial = 1
            elif expression_evaluation == int(
                    expression_evaluation):  # If the expression can be evaluated to an integer
                my_factorial = factorial(int(expression_evaluation))
            else:  # Factorials of decimal numbers
                my_factorial = gamma(expression_evaluation) * expression_evaluation

            return coefficient_evaluation * my_factorial ** power_evaluation
        elif power_evaluation == 0 and coefficient_evaluation is not None:
            # Can disregard if the expression can't be evaluated, because power by 0 is 1
            return coefficient_evaluation  # coefficient * (...) ** 0 = coefficient * 1 = coefficient
        return None  # Couldn't evaluate

    def simplify(self):
        """Try to simplify the factorial expression"""
        self._coefficient.simplify()
        if self._coefficient == 0:
            self._expression = None
            self._power = Mono(1)

    def python_syntax(self):
        if self._expression is None:
            return f"{self._coefficient.python_syntax()}"
        return f"{self._coefficient} * factorial({self._expression.python_syntax()}) ** {self._power.python_syntax()}"

    def __str__(self):
        if self._expression is None:
            return f"{self._coefficient}"
        coefficient_str = format_coefficient(self._coefficient)
        if coefficient_str not in ('', '-'):
            coefficient_str += '*'
        power_str = f"**{self._power.__str__()}" if self._power != 1 else ""
        inside_str = self._expression.__str__()
        if '-' in inside_str or '+' in inside_str or '*' in inside_str or '/' in inside_str:
            inside_str = f'({inside_str})'
        expression_str = f"({inside_str}!)" if coefficient_str != "" else f"{inside_str}!"
        if power_str == "":
            return f"{coefficient_str}{expression_str}"
        return f"{coefficient_str}({expression_str}){power_str}"

    def __copy__(self):
        return Factorial(
            coefficient=self._coefficient,
            expression=self._expression,
            power=self._power

        )

    def __eq__(self, other: Union[IExpression, int, float]):
        if other is None:
            return False
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            if my_evaluation is not None:
                return my_evaluation == other
            return False
        elif isinstance(other, IExpression):
            if my_evaluation is not None:  # If self can be evaluated
                other_evaluation = other.try_evaluate()
                return other_evaluation is not None and my_evaluation == other_evaluation
            if isinstance(other, Factorial):
                return self._coefficient == other._coefficient and self._expression == other._expression and self._power == other._power
            return False

        else:
            raise TypeError(f"Invalid type '{type(other)}' for equating with a Factorial expression.")

    def __ne__(self, other: Union[IExpression, int, float]):
        return not self.__eq__(other)


class Abs(IExpression, IPlottable, IScatterable):
    """A class for representing expressions with absolute values. For instance, Abs(x) is the same as |x|."""
    __slots__ = ['_coefficient', '_expression', '_power']

    def __init__(self, expression: Union[IExpression, int, float], power: Union[int, float, IExpression] = 1,
                 coefficient: Union[int, float, IExpression] = 1, gen_copies=True):

        # Handling the expression
        if isinstance(expression, (int, float)):
            self._expression = Mono(expression)
        elif isinstance(expression, IExpression):
            self._expression = expression.__copy__() if gen_copies else expression
        else:
            raise TypeError(f"Invalid type {type(expression)} for inner expression when creating an Abs object.")

        # Handling the power
        if isinstance(power, (int, float)):
            self._power = Mono(power)
        elif isinstance(power, IExpression):  # Allow algebraic powers here?
            self._power = power.__copy__() if gen_copies else power
        else:
            raise TypeError(f"Invalid type {type(power)} for 'power' argument when creating a new Abs object.")

        # Handling the coefficient
        if isinstance(coefficient, (int, float)):
            self._coefficient = Mono(coefficient)
        elif isinstance(coefficient, IExpression):  # Allow algebraic powers here?
            self._coefficient = coefficient.__copy__() if gen_copies else coefficient
        else:
            raise TypeError(
                f"Invalid type {type(coefficient)} for 'coefficient' argument when creating a new Abs object.")

    @property
    def coefficient(self):
        return self._coefficient

    @property
    def expression(self):
        return self._expression

    @property
    def power(self):
        return self._power

    @property
    def variables(self):
        variables = self._coefficient.variables
        variables.update(self._expression.variables)
        variables.update(self._power.variables)
        return variables

    def simplify(self):
        self._coefficient.simplify()
        self._expression.simplify()
        self._power.simplify()

    def assign(self, **kwargs):
        self._coefficient.assign(**kwargs)
        self._expression.assign(**kwargs)
        self._expression.assign(**kwargs)

    def to_dict(self):
        return {
            "type": "Abs",
            "coefficient": self._coefficient.to_dict(),
            "expression": self._expression.to_dict(),
            "power": self._power.to_dict()
        }

    @staticmethod
    def from_dict(given_dict: dict):
        expression_obj = create_from_dict(given_dict['expression'])
        coefficient_obj = create_from_dict(given_dict['coefficient'])
        power_obj = create_from_dict(given_dict['power'])
        return Abs(expression=expression_obj, power=power_obj, coefficient=coefficient_obj)

    def __add_or_sub(self, other, operation: str = '+'):
        if isinstance(other, (int, float)):
            my_evaluation = self.try_evaluate()
            if my_evaluation is not None:
                if operation == '+':
                    return Mono(my_evaluation + other)
                else:
                    return Mono(my_evaluation - other)
            else:
                if operation == '+':
                    return ExpressionSum([self, Mono(other)])
                else:
                    return ExpressionSum([self, Mono(-other)])

        elif isinstance(other, IExpression):
            my_evaluation = self.try_evaluate()
            other_evaluation = other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                if operation == '+':
                    return Mono(my_evaluation + other_evaluation)
                return Mono(my_evaluation - other_evaluation)
            if (my_evaluation, other_evaluation) == (None, None):
                if isinstance(other, Abs):
                    if self._power == other._power:
                        if self._expression == other._expression or self._expression == -other._expression:
                            # |x| = |x|, or |x| = |-x|. Find whether the two expressions are compatible for addition.
                            if operation == '+':
                                self._coefficient += other._coefficient
                            else:
                                self._coefficient -= other._coefficient
                            return self

            return ExpressionSum((self, other))

    def __iadd__(self, other: Union[int, float, IExpression]):
        return self.__add_or_sub(other, operation='+')

    def __isub__(self, other):
        return self.__add_or_sub(other, operation='-')

    def __imul__(self, other: Union[int, float, IExpression]):
        if not isinstance(other, (int, float, IExpression)):
            raise TypeError(f" Invalid type: {type(other)} when multiplying an Abs object."
                            f" Expected types 'int', 'float', 'IExpression'.")
        if isinstance(other, (int, float)):
            self._coefficient *= other
            return self
        my_evaluation = self.try_evaluate()
        other_evaluation = other.try_evaluate()
        if None not in (my_evaluation, other_evaluation):
            return Mono(my_evaluation * other_evaluation)
        if other_evaluation is not None:
            self._coefficient *= other_evaluation
            return self

        if not isinstance(other, Abs):
            self._coefficient *= other
            return self
        # If other is indeed an Abs object:
        # Find whether the two expressions are connected somehow
        if self._expression == other._expression or self._expression == -other._expression:
            self._power += other._power
            self._coefficient *= other._coefficient
            return self
        return ExpressionMul((self, other))  # TODO: implement it later

    def __itruediv__(self, other: Union[int, float, IExpression]):
        if not isinstance(other, (int, float, IExpression)):
            raise TypeError(f" Invalid type: {type(other)} when dividing an Abs object."
                            f" Expected types 'int', 'float', 'IExpression'.")
        if other == 0:
            raise ValueError(f"Cannot divide an Abs object by 0.")
        if isinstance(other, (int, float)):
            self._coefficient /= other
            return self
        my_evaluation, other_evaluation = self.try_evaluate(), other.try_evaluate()
        if other_evaluation == 0:
            raise ValueError(f"Cannot divide an Abs object by 0.")
        if None not in (my_evaluation, other_evaluation):
            return Mono(my_evaluation / other_evaluation)
        if other_evaluation is not None:
            self._coefficient /= other
            return self
        if not isinstance(other, Abs):
            self._coefficient /= other
            return self
        # TODO: revise this solution...
        if self._expression == other._expression or self._expression == -other._expression:
            power_difference = self._power - other._power  # also handle cases such as |x|^x / |x|^(x-1) = |x|
            difference_evaluation = power_difference.try_evaluate()
            if difference_evaluation is None:
                self._coefficient /= other._coefficient
                return Exponent(coefficient=self._coefficient, base=self._expression, power=power_difference)
            else:
                if difference_evaluation > 0:
                    self._power = Mono(difference_evaluation)
                    self._coefficient /= other._coefficient
                    return Abs(coefficient=self._coefficient, power=self._power, expression=self._expression,
                               gen_copies=False)
                elif difference_evaluation == 0:
                    return self._coefficient
                else:
                    return Fraction(self._coefficient / other._coefficient,
                                    Abs(self._expression, -difference_evaluation))
        return Fraction(self, other)  # TODO: implement it later

    def __ipow__(self, power: Union[int, float, IExpression]):
        if not isinstance(power, (int, float, IExpression)):
            raise TypeError(f"Invalid type: {type(power)} when raising by a power an Abs object."
                            f" Expected types 'int', 'float', 'IExpression'.")
        if isinstance(power, (int, float)):
            self._coefficient **= power
            self._power *= power
            return self
        # The power is an algebraic expression
        power_evaluation = power.try_evaluate()
        if power_evaluation is not None:
            self._coefficient **= power
            self._power *= power
            return self
        return Exponent(self, power)

    def __neg__(self):
        return Abs(expression=self._expression, power=self._power, coefficient=self._coefficient.__neg__())

    def __eq__(self, other: Union[IExpression, int, float]):
        if isinstance(other, (int, float)):
            my_evaluation = self.try_evaluate()
            return my_evaluation == other

        if isinstance(other, IExpression):
            my_evaluation = self.try_evaluate()
            other_evaluation = other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                return my_evaluation == other_evaluation
            if (my_evaluation, other_evaluation) == (None, None):
                if isinstance(other, Abs):
                    if self._expression == other._expression:
                        return (self._coefficient, self._power) == (other._coefficient, other._power)
                expression_evaluation = self._expression.try_evaluate()  # computed the second time - waste..
                if expression_evaluation is not None:
                    return self._coefficient * abs(expression_evaluation) ** self._power == other
            return False
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def derivative(self, get_derivatives=False):
        warnings.warn("Derivatives are still experimental, and might not work for other algebraic expressions"
                      "rather than polynomials.")
        num_of_variables = len(self.variables)
        if num_of_variables == 0:
            return lambda x: self.try_evaluate()
        assert num_of_variables == 1, "Use partial derivatives of expressions with several variables."
        positive_expression = self._coefficient * self._expression ** self._power
        try:
            positive_derivative = positive_expression.derivative()
        except:
            return None
        negative_derivative = -positive_derivative
        if get_derivatives:
            return positive_derivative, negative_derivative
        positive_derivative, negative_derivative = positive_derivative.to_lambda(), positive_derivative.to_lambda()
        return lambda x: positive_derivative(x) if x > 0 else (negative_derivative(x) if x < 0 else 0)

    def integral(self, other):
        pass

    def try_evaluate(self) -> Optional[Union[int, float]]:
        coefficient_evaluation = self._coefficient.try_evaluate()
        if coefficient_evaluation is None:
            return None
        if coefficient_evaluation == 0:
            return 0
        expression_evaluation = self._expression.try_evaluate()
        power_evaluation = self._power.try_evaluate()
        if power_evaluation is None:
            return None
        if power_evaluation == 0:
            return coefficient_evaluation
        if expression_evaluation is None:
            return None
        return coefficient_evaluation * abs(expression_evaluation) ** power_evaluation

    def __str__(self):
        if self._coefficient == 0 or self._expression == 0:
            return "0"
        if self._power == 0:
            return self._coefficient.__str__()
        elif self._power == 1:
            power_string = ""
        else:
            power_string = f"**{self._power.python_syntax()}"
        if self._coefficient == 1:
            coefficient_string = f""
        elif self._coefficient == -1:
            coefficient_string = f"-"
        else:
            coefficient_string = f"{self._coefficient.__str__()}*"
        return f"{coefficient_string}|{self._expression}|{power_string}"

    def __copy__(self):
        return Abs(expression=self._expression, power=self._power, coefficient=self._coefficient, gen_copies=True)


def max_power(expressions):
    return max(expressions, key=lambda expression: max(expression.variables_dict.values()))


def get_factors(n):
    if n == 0:
        return {}
    factors = set(reduce(list.__add__,
                         ([i, n // i] for i in range(1, int(abs(n) ** 0.5) + 1) if n % i == 0)))
    return factors.union({-number for number in factors})


def extract_possible_solutions(most_significant_coef: float, free_number: float):
    p_factors = get_factors(free_number)
    q_factors = get_factors(most_significant_coef)
    possible_solutions = []
    for p_factor in p_factors:
        for q_factor in q_factors:
            if q_factor != 0:  # Can't divide by 0..
                possible_solutions.append(p_factor / q_factor)
    if possible_solutions:
        return set(possible_solutions)
    return {}


def __find_solutions(coefficients, possible_solutions):
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
            # Solving a quadratic equation!
            try:
                # find more solutions. right now, this feature only works on real numbers, and when the highest power
                # is 3
                values = solve_quadratic_real(division_result[0], division_result[1], division_result[2])
                solutions.add(values[0])
                solutions.add(values[1])
            except Exception as e:
                warnings.warn(
                    f"Due to an {e.__class__} error in line {exc_info()[-1].tb_lineno}, some solutions might be "
                    f"missing ! ")
        else:
            print("Whoops! it seems something went wrong")
    return solutions


def solve_polynomial(coefficients, epsilon: float = 0.000001, nmax: int = 10_000):
    """
    This method find the roots of a polynomial from a collection of the coefficients of the expression.
    The algorithm chooses the most efficient algorithm in correspondence to the degree of the polynomial.
    for a collection of coefficients of length n, its degree would be n-1.
    For example, [1, -2, 1] represents x^2 - 2x + 1. The length of the list is 3 and the degree of the expression is 2.
    For degrees of 4 or lower, the execution time will be significantly lower, since it is computed via generalized
    formulas or algebra.
    For degrees of 5 or more, there aren't any generalized formulas ( as proven in Abel's impossibility theorem ),
    So instead of a formula, an iterative method is used to approximate the roots ( complex and real ) of
    the polynomial.

    :param coefficients: The coefficients of the polynomial.
    :param epsilon:
    :param nmax: Max number of iterations In case the polynomial is of a degree of 5 or more.Default is 100,000
    :return:
    """
    if isinstance(coefficients, str):
        return solve_polynomial(ParseEquation.parse_polynomial(coefficients))

    if len(coefficients) == 1:
        return None
    if len(coefficients) == 2:
        return [-coefficients[1] / coefficients[0]]
    if len(coefficients) == 3:
        return solve_quadratic(coefficients[0], coefficients[1], coefficients[2])
    if len(coefficients) == 4:
        return solve_cubic(coefficients[0], coefficients[1], coefficients[2], coefficients[3])
    if len(coefficients) == 5:
        return solve_quartic(coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4])
    # Iterative approach via aberth's method, since there are no generalized formulas for polynomial equations
    # of degree 5 or higher, as proven by the Abel–Ruffini theorem.
    polynomial_obj = Poly(coefficients_to_expressions(coefficients))
    poly_derivative = polynomial_obj.derivative().to_lambda()
    return aberth_method(polynomial_obj.to_lambda(), poly_derivative, coefficients, epsilon, nmax)


def solve_poly_by_factoring(coefficients):
    """
    This method attempts to find the roots of a polynomial by synthetic division.
    It won't always return all the solutions, but it is faster than many numerical root finding algorithms, and might
    even be preferable in some cases over these algorithms.
    """
    if len(coefficients) == 3:
        return solve_quadratic_real(coefficients[0], coefficients[1], coefficients[2])
    if coefficients is None:
        return {}
    most_significant = coefficients[0]
    free_number = coefficients[-1]
    possible_solutions = extract_possible_solutions(most_significant, free_number)
    print(possible_solutions)
    solutions = __find_solutions(coefficients, possible_solutions)
    return solutions


def coefficients_to_expressions(coefficients, variable: str = "x"):
    """
    Getting a list of coefficients and the name of the variable, and returns a list of polynomial expressions,
    namely, a list of Mono objects.
    :param coefficients: the coefficients, for example : [ 1,0,2,3] ( the output expression would be x^3+2x+3 for x )
    :param variable: the name of the variable, the default is "x"
    :return: returns a list of polynomials with the corresponding coefficients and powers.
    """
    return [Mono(coefficient=coef, variables_dict={variable: len(coefficients) - 1 - index}) for index, coef in
            enumerate(coefficients) if coef != 0]




def extract_dict_from_equation(equation: str, delimiter="="):  # TODO: improve this shitty method.
    """
    This method should accept an equation, and extract the variable from it. It is still quite basic..

    :param equation: the equation, of type string
    :param delimiter: separator
    :return: returns a dictionary of the __variables and the number. for example, for the equation 3x-y+8 = 6+y+x the
    dictionary returned would be {'x':0,'y':0,'number':0}
    """
    variables = dict()
    first_side, second_side = equation.split(delimiter)
    accumulator = ""
    for expression in split_expression(first_side) + split_expression(second_side):
        start_index = -1
        for index, character in enumerate(expression):
            if character.isalpha():
                start_index = index
                break
        if start_index != -1:
            accumulator = ""
            for character in expression[start_index:]:
                accumulator += character
        variables[accumulator.strip()] = 0
    variables["number"] = 0
    return {key: value for key, value in variables.items() if key != ''}


def linear_expression_to_dict(expression: str, variables: Iterable) -> dict:
    """alternative way to """
    expression = clean_spaces(expression)
    my_dict = dict()
    if expression[-1] != ' ':
        expression += ' '
    matches = list(re.finditer(fr"([-+]?\d+[.,]?\d*)?\*?([a-zA-Z]+)", expression))
    for variable in variables:
        my_dict[variable] = sum(extract_coefficient(match.group(1)) for match in matches if match.group(2) == variable)
    matches = re.finditer(fr"([-+]?\d+[.,]?\d*)[-+\s]", expression)
    numbers_sum = sum(extract_coefficient(match.group(1)) for match in matches)
    my_dict["number"] = numbers_sum
    return my_dict


def simplify_linear_expression(expression: str, variables: Iterable[str], format_abs=False, format_factorial=False) -> dict:
    if format_abs:
        expression = handle_abs(expression)
    if format_factorial:
        expression = handle_abs(expression)
    expr = expression.replace("-", "+-").replace(" ", "")
    expressions = [num for num in expr.split("+") if num != '' and num is not None]
    if isinstance(variables, dict):
        new_dict = variables.copy()
    else:
        new_dict = {variable_name: 0 for variable_name in variables}
    if "number" not in new_dict:
        new_dict["number"] = 0
    for item in expressions:
        if item[-1].isalpha() or contains_from_list(allowed_characters, item):
            if item[-1] in new_dict.keys():
                if len(item) == 1:
                    item = f"1{item}"
                elif len(item) == 2 and item[0] == '-':
                    item = f"-1{item[-1]}"
                new_dict[item[-1]] += float(item[:-1])
            elif not is_number(item):
                raise ValueError(f"Unrecognized expression {item}")
        else:
            new_dict["number"] += float(item)
    return new_dict



def create_pdf(path: str, title="Worksheet", lines=()) -> bool:
    try:
        c = Canvas(path)
        c.setFontSize(22)
        c.drawString(50, 800, title)
        textobject = c.beginText(2 * cm, 26 * cm)
        c.setFontSize(14)
        for index, line in enumerate(lines):
            textobject.textLine(f'{index + 1}. {line.strip()}')
            textobject.textLine('')
        c.drawText(textobject)
        # c.showPage()
        # c.setFontSize(22)
        # c.drawString(50, 800, title)
        c.showPage()
        c.save()
        return True
    except Exception as ex:
        warnings.warn(f"Couldn't create the pdf file due to a {ex.__class__} error")
        return False


def create_pages(path: str, num_of_pages: int, titles, lines):
    c = Canvas(path)
    for i in range(num_of_pages):
        c.setFontSize(22)
        c.drawString(50, 800, titles[i])
        textobject = c.beginText(2 * cm, 26 * cm)
        c.setFontSize(14)
        for index, line in enumerate(lines[i]):
            textobject.textLine(f'{lines[i][index]}')
            textobject.textLine('')
        c.drawText(textobject)
        c.showPage()
    c.save()




def subtract_dicts(dict1: dict, dict2: dict) -> dict:
    """
    each side in the equation is processed into a dictionary. in order to reach a result, it is imperative
    to subtract the two sides, and equate what's left to 0.
    This method is responsible for taking both dictionaries, and subtracting them.
    :param dict1: the first dictionary
    :param dict2: the second dictionary
    :return:
    """
    new_dict = {}
    # making sure the dictionaries have the same keys
    for key in dict2.keys():
        if key not in dict1.keys():
            dict1[key] = 0
            warnings.warn(f"variable {key} wasn't found in the first data structure")
    for key in dict1.keys():
        if key not in dict2.keys():
            dict2[key] = 0
            warnings.warn(f" variable {key} wasn't found in the second data structure")

    for key in dict1.keys():
        new_dict[key] = dict1[key] - dict2[key]

    return new_dict


def reinman(f: Callable, a, b, N: int):
    if N < 2:
        raise ValueError("The method requires N >= 2")
    return sum((b - a) / (N - 1) * f(value) for value in np.linspace(a, b, N))


def trapz(f: Callable, a, b, N: int):
    if N == 0:
        raise ValueError("Trapz(): N cannot be 0")
    dx = (b - a) / N
    return 0.5 * dx * sum((f(a + i * dx) + f(a + (i - 1) * dx)) for i in range(1, int(N) + 1))


def simpson(f: Callable, a, b, N: int):
    if N <= 2:
        raise ValueError("The method requires N >= 2")
    dx = (b - a) / (N - 1)
    if N % 2 != 0:
        N += 1
    return (dx / 3) * sum(
        ((f(a + (2 * i - 2) * dx) + 4 * f(a + (2 * i - 1) * dx) + f(a + 2 * i * dx)) for i in range(1, int(N / 2))))


def numerical_diff(f, a, method='central', h=0.01):
    if method == 'central':
        return (f(a + h) - f(a - h)) / (2 * h)
    elif method == 'forward':
        return (f(a + h) - f(a)) / h
    elif method == 'backward':
        return (f(a) - f(a - h)) / h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")


def gradient_descent(f_tag: Callable, initial_value,
                     learning_rate: float = 0.01, precision: float = 0.000001,
                     nmax=10000):
    previous_step_size = 1
    for i in range(nmax):
        if previous_step_size <= precision:
            return initial_value
        new_value = initial_value - learning_rate * f_tag(initial_value)
        previous_step_size = abs(new_value - initial_value)
        initial_value = new_value
    warnings.warn("Reached maximum limit of iterations! Result might be inaccurate!")
    return initial_value


def gradient_ascent(f_tag: Callable, initial_value, learning_rate: float = 0.01, precision: float = 0.000001,
                    nmax=10000):
    previous_step_size = 1
    for i in range(nmax):
        if previous_step_size <= precision:
            return initial_value
        new_value = initial_value + learning_rate * f_tag(initial_value)
        previous_step_size = abs(new_value - initial_value)
        initial_value = new_value
    warnings.warn("Reached maximum limit of iterations! Result might be inaccurate!")
    return initial_value


def solve_linear(equation: str, variables=None, get_dict=False, get_json=False):
    if variables is None:
        variables = extract_dict_from_equation(equation)
    first_side, second_side = equation.split("=")
    first_dict = simplify_linear_expression(expression=first_side, variables=variables)
    second_dict = simplify_linear_expression(expression=second_side, variables=variables)
    result_dict = {key: value for key, value in subtract_dicts(dict1=first_dict, dict2=second_dict).items() if key}
    if len(result_dict) < 2:
        return None
    elif len(result_dict) == 2:
        if list(result_dict.values())[0] == 0:
            if list(result_dict.values())[1] == 0:
                return np.inf
            return None  # There are no solutions, ( like 0 = 5 )
        solution = -result_dict["number"] / list(result_dict.values())[0]
        if get_dict:
            return {list(variables.keys())[0]: solution}
        elif get_json:
            return json.dumps({"variable": list(variables.keys())[0], "result": solution})
        return solution
    elif len(result_dict) > 2:
        raise ValueError("Invalid equation caused an unexpected error")


def solve_linear_inequality(equation: str, variables=None):
    equal_sign_index = equation.find('=')
    if equal_sign_index == -1:
        bigger_sign_index = equation.find('>')
        if bigger_sign_index == -1:
            smaller_sign_index = equation.find('<')
            if smaller_sign_index == -1:
                raise ValueError("Invalid equation")
            else:
                sign = '<'
        else:
            sign = '>'
    else:
        bigger_sign_index = equation.find('>')
        if bigger_sign_index == -1:
            smaller_sign_index = equation.find('<')
            if smaller_sign_index == -1:
                sign = '<='
            else:
                sign = '='

        else:
            sign = '>='

    expressions = equation.split(sign)
    if len(expressions) != 2:
        raise ValueError(f"Invalid equation")
    if variables is None:
        variables = extract_dict_from_equation(equation, delimiter=sign)
    first_side, second_side = expressions
    first_dict = simplify_linear_expression(first_side, variables)
    second_dict = simplify_linear_expression(second_side, variables)
    result_dict = subtract_dicts(first_dict, second_dict)
    first_key = list(result_dict.keys())[0]
    first_value = list(result_dict.values())[0]
    number_value = result_dict["number"]
    return f"{first_key}{sign}{round_decimal(-number_value / first_value)}"


def solve_linear_system(equations, variables=None):
    """ Solve a system of linear equations via Guass-Elimination Method with matrices"""
    if not variables:
        variables = set()
        for equation in equations:
            variables.update(extract_variables_from_expression(equation))
    values_matrix = []
    for equation in equations:
        equal_index = equation.find('=')
        side1, side2 = equation[:equal_index], equation[equal_index + 1:]
        first_dict = simplify_linear_expression(side1, variables)
        second_dict = simplify_linear_expression(side2, variables)
        result_dict = subtract_dicts(second_dict, first_dict)
        values_matrix.append(list(result_dict.values()))
    matrix_obj = Matrix(matrix=values_matrix)
    matrix_obj.gauss()
    answers = {}
    keys = list(variables)
    i = 0
    for row in matrix_obj.matrix:
        answers[keys[i]] = -round_decimal(row[len(row) - 1])
        i += 1
    return answers


def get_equation_variables(equation: str) -> List[Optional[str]]:
    return list({character for character in equation if character.isalpha()})


def random_linear_system(variables, solutions_range: Tuple[int, int] = (-10, 10),
                         coefficients_range: Tuple[int, int] = (-10, 10),
                         digits_after=0, get_solutions=False):
    equations = []
    num_of_equations = len(variables)
    solutions = [round(random.uniform(solutions_range[0], solutions_range[1]), digits_after) for _ in
                 range(num_of_equations)]
    for _ in range(num_of_equations):
        coefficients_dict = dict()
        equation_sum = 0
        # Create the basics: ax + by + cz ... = equation_sum
        for index, variable in enumerate(variables):
            random_coefficient = random.randint(coefficients_range[0], coefficients_range[1])
            coefficients_dict[variable] = [random_coefficient]
            equation_sum += random_coefficient * solutions[index]
        # Complicate the equation
        free_number = random.randint(coefficients_range[0], coefficients_range[1])
        equation_sum += free_number

        other_side_dict = {variable: [] for variable in variables}
        num_of_operations = random.randint(2, 5)  # TODO: customize these parameters
        for _ in range(num_of_operations):
            operation_index = random.randint(0, 1)
            if operation_index == 0:  # Addition - add an expression such as '3x', '-5y', etc, to both sides
                random_variable = random.choice(variables)
                random_coefficient = random.randint(coefficients_range[0], coefficients_range[1])
                coefficients_dict[random_variable].append(random_coefficient)
                other_side_dict[random_variable].append(random_coefficient)
            else:  # Multiplication - multiply both sides by a number
                random_num = random.randint(1, 3)  # Also 1, to perhaps prevent very large numbers.
                for variable, coefficients in coefficients_dict.items():
                    for index in range(len(coefficients)):
                        coefficients[index] *= random_num
                free_number *= random_num
                for variable, coefficients in other_side_dict.items():
                    for index in range(len(coefficients)):
                        coefficients[index] *= random_num
                equation_sum *= random_num
        equations.append(
            f"{ParseExpression.unparse_linear(coefficients_dict, free_number)}={ParseExpression.unparse_linear(other_side_dict, equation_sum)}")
    if get_solutions:
        return equations, solutions
    return equations


def random_poly_system(variables):
    pass


def random_linear(coefs_range=(-15, 15), digits_after: int = 0, variable='x', get_solution: bool = False,
                  get_coefficients: bool = False):
    """
    Generates a random linear expression in the form ax+b
    :param coefs_range: the range from which the coefficients will be chosen randomly
    :param digits_after: the maximum number of digits after the decimal point for the coefficients
    :param variable: the variable that will appear in the string
    :param get_solution: whether to return also the solution
    :param get_coefficients: whether to return also the coefficients, (a, b)
    :return:
    """
    a = round_decimal(round(random.uniform(coefs_range[0], coefs_range[1]), digits_after))
    while a == 0:
        a = round_decimal(round(random.uniform(coefs_range[0], coefs_range[1]), digits_after))

    b = round_decimal(round(random.uniform(coefs_range[0], coefs_range[1]), digits_after))
    a_str = format_coefficient(round_decimal(a))
    b_str = format_free_number(b)

    if get_solution:
        if get_coefficients:
            return f"{a_str}{variable}{b_str}", round_decimal(-b / a), (a, b)
        return f"{a_str}{variable}{b_str}", round_decimal(-b / a)
    elif get_coefficients:
        return f"{a_str}{variable}{b_str}", (a, b)

    return f"{a_str}{variable}{b_str}"


def random_polynomial(degree: int = None, solutions_range=(-5, 5), digits_after=0, variable='x', python_syntax=False,
                      get_solutions=False):
    if degree is None:
        degree = random.randint(2, 9)
    a = round_decimal(round(random.uniform(solutions_range[0], solutions_range[1]), digits_after))
    while a == 0:
        a = round_decimal(round(random.uniform(solutions_range[0], solutions_range[1]), digits_after))
    accumulator = [f'{format_coefficient(a)}x**{degree}'] if python_syntax else [f'{format_coefficient(a)}x^{degree}']
    solutions = {round_decimal(round(random.uniform(solutions_range[0], solutions_range[1]), digits_after)) for _ in
                 range(degree)}
    permutations_length = 1
    for i in range(degree):
        current_permutations = set(tuple(sorted(per)) for per in permutations(solutions, permutations_length))
        current_sum = 0
        for permutation in current_permutations:
            current_sum += reduce(operator.mul, permutation)
        if current_sum != 0:
            current_power = degree - permutations_length
            coefficient = format_coefficient(
                round_decimal(current_sum * a)) if current_power != 0 else f"{round_decimal(current_sum * a)}"
            if coefficient != "" and coefficient[0] not in ('+', '-'):
                coefficient = f"+{coefficient}"
            if current_power == 0:
                accumulator.append(f"{coefficient}")
            elif current_power == 1:
                accumulator.append(f"{coefficient}{variable}")
            else:
                accumulator.append(f"{coefficient}{variable}^{current_power}")
        permutations_length += 1
    equation = "".join(accumulator)
    if get_solutions:
        return equation, [-solution for solution in solutions]
    return equation


def random_polynomial2(degree: int, values=(-15, 15), digits_after=0, variable='x', python_syntax=False):
    a = round_decimal(round(random.uniform(values[0], values[1]), digits_after))
    while a == 0:
        a = round_decimal(round(random.uniform(values[0], values[1]), digits_after))
    accumulator = []
    while a == 0:
        a = round_decimal(round(random.uniform(values[0], values[1]), digits_after))
    accumulator.append(f"{format_coefficient(a)}{variable}^{degree}")
    for index in range(1, degree - 1):
        m = round_decimal(round(random.uniform(values[0], values[1]), digits_after))
        coef_str = format_coefficient(m)
        if coef_str:
            if coef_str[0] not in ('+', '-'):
                coef_str = f'+{coef_str}'
        power = degree - 1
        power_str = f'^{power}' if power != 1 else f''
        if python_syntax:
            pass
        else:
            accumulator.append(f"{coef_str}{variable}{power_str}")
    m = round_decimal(round(random.uniform(values[0], values[1]), digits_after))
    accumulator.append(f"+{round_decimal(m)}" if m > 0 else f"{m}") if m != 0 else ""
    return "".join(accumulator)




class LinearSystem:
    """
    This class represents a system of linear __equations.
    It solves them via a simple implementation of the Gaussian Elimination technique.
    """

    def __init__(self, equations: Iterable, variables: Iterable = None):
        """
        Creating a new equation system

        :param equations: An iterable collection of equations. Each equation in the collection can be of type
        string or Equation
        :param variables:(Optional) an iterable collection of strings that be converted to a list.
        Each item represents a variable in the equations. For example, ('x','y','z').
        """
        self.__equations, self.__variables = [], list(variables) if variables is not None else []
        self.__variables_dict = dict()
        for equation in equations:
            if isinstance(equation, str):
                self.__equations.append(LinearEquation(equation))
            elif isinstance(equation, LinearEquation):
                self.__equations.append(equation)
            else:
                raise TypeError

    # PROPERTIES
    @property
    def equations(self):
        return self.__equations

    @property
    def variables(self):
        return self.__variables

    def add_equation(self, equation: str):
        self.__equations.append(LinearEquation(equation))

    def __extract_variables(self):
        variables_dict = {}
        for equation in self.__equations:
            if not equation.variables_dict:
                equation.__variables = equation.variables_dict
            for variable in equation.variables_dict:
                if variable not in variables_dict and variable != "number":
                    variables_dict[variable] = 0
        variables_dict["number"] = 0
        self.__variables_dict = variables_dict
        return variables_dict

    def to_matrix(self):
        """
        Converts the equation system to a matrix of _coefficient, so later the Gaussian elimination method wil
        be implemented on it, in order to solve the system.
        :return:
        """
        variables = self.__variables_dict if self.__variables_dict else self.__extract_variables()
        values_matrix = []
        for equation in self.__equations:
            equation.__variables = variables
            equal_index = equation.equation.find('=')
            side1, side2 = equation.equation[:equal_index], equation.equation[equal_index + 1:]
            first_dict = simplify_linear_expression(side1, equation.variables_dict)
            second_dict = simplify_linear_expression(side2, equation.variables_dict)
            result_dict = subtract_dicts(second_dict, first_dict)
            values_matrix.append(list(result_dict.values()))

        return values_matrix

    def to_matrix_and_vector(self):
        pass

    def get_solutions(self):
        """
        fetches the solutions
        :return: returns a dictionary that contains the name of each variable, and it's (real) __solution.
        for example: {'x':6,'y':4}
        This comes handy later since you can access simply the solutions.
        """
        values_matrix = self.to_matrix()
        matrix_obj = Matrix(matrix=values_matrix)
        matrix_obj.gauss()
        answers = {}
        keys = list(self.__variables_dict.keys())
        i = 0
        for row in matrix_obj.matrix:
            answers[keys[i]] = -round_decimal(row[len(row) - 1])
            i += 1
        return answers

    def simplify(self):
        pass

    def print_solutions(self):
        """
        prints out the solutions of the equation.
        :return: None
        """
        solutions = self.get_solutions()
        for key, value in solutions.items():
            print(f'{key} = {value}')


def format_linear_dict(algebraic_dict: dict, round_coefficients: bool = True) -> str:
    """ Receives a dictionary that represents a linear expression and creates a new string from it.
        For instance, the dictionary {'x':2, 'y':-4, 'number':5} represents the expression 2x - 4y + 5
     """
    if not algebraic_dict:
        return ""
    accumulator = ""
    for key, value in algebraic_dict.items():
        if value != 0:
            value = round_decimal(value) if round_coefficients else value
            coef = f"+{value}" if value > 0 else f"{value}"
            if key == 'number':
                accumulator += coef
            elif value == 1:
                accumulator += f"+{key}"
            elif value == -1:
                accumulator += f"-{key}"
            else:
                accumulator += f"{coef}{key}"
    return accumulator[1:] if accumulator[0] == '+' else accumulator


def format_poly_dict(algebraic_dict: dict):
    """
    Internal method: Receives a dictionary that represents a polynomial, in the Equation's class format, and turns it into string.
    *** This method might become redundant / deleted / updated.
    :param algebraic_dict: the dictionary of the expression, for example : {'x**2':3,'x':2,'number':-1} -> 3x^2+2x-1
    :return:
    """
    accumulator = ""
    for expression, coefficient in algebraic_dict.items():
        if expression == 'number':
            if coefficient >= 0:
                accumulator += f'+{round_decimal(coefficient)}'
            else:
                accumulator += f'{round_decimal(coefficient)}'
        else:
            if coefficient != 0:
                power = float(expression[expression.find('**') + 2:])
                if coefficient == int(coefficient):
                    coefficient = int(coefficient)
                if power == int(power):
                    power = int(power)
                if power == 1:
                    accumulator += f"{coefficient}{expression[:expression.find('**')]}"
                elif power == 0:
                    algebraic_dict['number'] += 1
                else:
                    accumulator += f' {coefficient}{expression} +'
    return accumulator.replace("++", "+").replace("+-", "-").replace("--", "-")




def only_numbers_letters(given_string: str):
    """
    checks whether a string contains only letters and numbers.
    :param given_string:
    :return:
    """
    if given_string == "" or given_string is None:
        return False
    char_array = list(given_string)
    if char_array[0] == '-':
        del char_array[0]
    return bool([char for char in char_array if char.isalpha() or char.isdigit()])


def handle_trigo_calculation(expression: str):
    """ getting the result of a single trigonometric operation, e.g : sin(90) -> 1"""
    selected_operation = [op for op in TRIGONOMETRY_CONSTANTS.keys() if op in expression]
    selected_operation = selected_operation[0]
    start_index = expression.find(selected_operation) + len(selected_operation) + 1
    coef = expression[:expression.find(selected_operation)]
    if coef == "" or coef is None or coef == "+":
        coef = 1
    elif coef == '-':
        coef = 1
    else:
        coef = float(coef)
    parameter = expression[start_index] if expression[start_index].isdigit() or expression[
        start_index] == '-' else ""
    for i in range(start_index + 1, expression.rfind(')')):
        parameter += expression[i]
    if is_evaluatable(parameter):
        parameter = float(eval(parameter))
    parameter = -float(parameter) if expression[0] == '-' else float(parameter)
    return round_decimal(coef * TRIGONOMETRY_CONSTANTS[selected_operation](parameter))


def handle_trigo_expression(expression: str):
    """ handles a whole trigonometric expression, for example: 2sin(90)+3sin(60)"""
    expressions = split_expression(expression)
    result = 0
    for expr in expressions:
        result += handle_trigo_calculation(expr)
    return result


class Function(IPlottable, IScatterable):
    arithmetic_operations = ('+', '-')

    class Classification(Enum):
        linear = 1,
        quadratic = 2,
        polynomial = 3,
        trigonometric = 4,
        logarithmic = 5,
        exponent = 6,
        constant = 7,
        command = 8,
        linear_several_parameters = 8,
        non_linear_several_parameters = 9,
        exponent_several_parameters = 10,
        predicate = 11

    def __init__(self, func=None):
        """ creating a new instance of a function
        You can enter a string, such as "f(x) = 3x^2+6x+6sin(2x)

        """
        self.__analyzed = None
        if isinstance(func, str):
            self.__func = clean_spaces(func).replace("^", "**")
            if "lambda" in self.__func and ':' in self.__func:
                # eval it from python lambda syntax
                lambda_index, colon_index = self.__func.find("lambda") + 6, self.__func.find(':')
                self.__func_signature = f"f({self.__func[lambda_index:colon_index]})"
                self.__func_expression = self.__func[colon_index + 1:]
                self.__func = f"{self.__func_signature}={self.__func_expression}"
            elif "=>" in self.__func:
                # eval it in C# or java like lambda expression
                self.__func_signature, self.__func_expression = self.__func.split('=>')
                self.__func = f"f({self.__func_signature})={self.__func_expression}"
            first_equal_index = self.__func.find('=')
            if first_equal_index == -1:  # Get here if there is no declaration of function
                self.__variables = list(extract_variables_from_expression(func))
                self.__func_signature = f'f({",".join(self.__variables)})'
                self.__func_expression = func
                self.__func = self.__func_signature + "=" + self.__func_expression
            else:
                self.__func_signature = clean_spaces(self.__func[:first_equal_index])
                self.__func_expression = clean_spaces(self.__func[first_equal_index + 1:])
                self.__variables = self.__func_signature[
                                   self.__func_signature.find('(') + 1:self.__func_signature.rfind(')')].split(',')
                self.__variables = [x for x in self.__variables if x != ""]
            self.__num_of_variables, self.__classification = len(self.__variables), None
            self.classify_function()
            try:
                self.__lambda_expression = self.__to_lambda()
            except:
                warnings.warn("Couldn't generate an executable lambda function from the input, trying manual execution")
                self.__lambda_expression = None
        elif isinstance(func, (Mono, Poly)):
            self.__variables = list(func.variables)
            self.__num_of_variables = len(self.__variables)
            self.__lambda_expression = func.to_lambda()
            self.__func_expression = func.__str__().replace("^", "**")
            self.__func_signature = f'f({",".join(self.__variables)}'
            self.__func = f"{self.__func_signature})={self.__func_expression}"
            self.classify_function()

        elif not isinstance(func, str):
            if is_lambda(func):
                self.__analyzed = None
                self.__lambda_expression = func
                lambda_str = (inspect.getsourcelines(func)[0][0])
                declaration_start = lambda_str.rfind("lambda") + 6
                declaration_end = lambda_str.find(":")
                inside_signature: str = lambda_str[declaration_start:declaration_end].strip()
                self.__func_signature = f'f({inside_signature})'
                self.__variables = inside_signature.split(',')
                self.__num_of_variables = len(self.__variables)  # DEFAULT
                self.__func_expression: str = lambda_str[declaration_end:lambda_str.rfind(')')]
                self.__func = "".join((self.__func_signature, self.__func_expression))
                self.classify_function()
                # TODO: CHANGE THE EXTRACTION OF DATA FROM LAMBDAS, SINCE IT WON'T WORK WHEN THERE'S MORE THAN 1 LAMBDA
                # TODO: IN A SINGLE LINE, AND ALSO IT WON'T WORK IN CASE THE LAMBDA WAS SAVED IN A VARIABLE ....
            else:
                raise TypeError(f"Function.__init__(). Unexpected type {type(func)}, expected types str,Mono,"
                                f"Poly, or a lambda expression")

        else:
            raise TypeError(f"Function.__init__(). Unexpected type {type(func)}, expected types str,Mono,"
                            f"Poly, or a lambda expression")

    # PROPERTIES - GET ACCESS ONLY

    @property
    def function_string(self):
        return self.__func

    @property
    def function_signature(self):
        return self.__func_signature

    @property
    def function_expression(self):
        return self.__func_expression

    @property
    def lambda_expression(self):
        return self.__lambda_expression

    @property
    def variables(self):
        return self.__variables

    @property
    def num_of_variables(self):
        return self.__num_of_variables

    def determine_power_role(self):  # TODO: wtf is this
        found = False
        for variable in self.__variables:
            if f'{variable}**' in self.__func:
                current_index = self.__func_expression.find(f'{variable}**') + len(variable) + 2
                if current_index > len(self.__func_expression):
                    raise ValueError("Invalid syntax: '**' is misplaced in the string. ")
                else:
                    finish_index = current_index + 1
                    while finish_index < len(self.__func_expression) and only_numbers_letters(
                            self.__func_expression[finish_index].strip()):
                        finish_index += 1
                    power = self.__func_expression[current_index:finish_index]
                    found = True
                    break
        if not found:
            return False
        return "polynomial" if is_number(power) and float(power) > 1 else "exponent"

    def classify_function(self):
        if '==' in self.__func_expression:
            self.__classification = self.Classification.predicate
        elif self.__num_of_variables == 1:
            if contains_from_list(list(TRIGONOMETRY_CONSTANTS.keys()), self.__func_expression):
                self.__classification = self.Classification.trigonometric
            elif f'{self.__variables[0]}**2' in self.__func_expression:
                self.__classification = Function.Classification.quadratic
            elif f'{self.__variables[0]}**' in self.__func_expression:
                power_role = self.determine_power_role()
                if power_role == 'polynomial':
                    self.__classification = self.Classification.polynomial
                elif power_role == 'exponent':
                    self.__classification = self.Classification.exponent
                else:
                    self.__classification = self.Classification.linear
            elif is_evaluatable(self.__func_expression):
                self.__classification = Function.Classification.constant
            else:
                self.__classification = Function.Classification.linear
        elif self.__num_of_variables < 1:
            self.__classification = Function.Classification.command
        else:
            # implement several __variables trigonometric choice
            power_role = self.determine_power_role()
            if power_role == 'polynomial':
                self.__classification = self.Classification.non_linear_several_parameters
            elif power_role == 'exponent':
                self.__classification = self.Classification.exponent_several_parameters
            else:
                self.__classification = self.Classification.linear_several_parameters

    @property
    def classification(self):
        return self.__classification

    def compute_value(self, *parameters):
        """ getting the result of the function for the specified parameters"""
        if self.__lambda_expression is not None:  # If an executable lambda has already been created
            try:
                if len(parameters) == 1:
                    if isinstance(parameters[0], Matrix):
                        matrix_copy = parameters[0].__copy__()
                        matrix_copy.apply_to_all(self.__lambda_expression)
                        return matrix_copy

                return self.__lambda_expression(*parameters)
            except ZeroDivisionError:  # It means that value wasn't valid.
                return None
            except ValueError:
                return None
            except OverflowError or MemoryError:
                warnings.warn(f"The operation on the parameters: '{parameters}'"
                              f" have exceeded python's limitations ( too big )")
        else:
            warnings.warn("Couldn't compute the expression entered! Check for typos and invalid syntax."
                          "Valid working examples: f(x) = x^2 + 8 , g(x,y) = sin(x) + 3sin(y)")
            return None

    def apply_on(self, collection):
        return apply_on(self.lambda_expression, collection)

    def range_gen(self, start: float, end: float, step: float = 1) -> Iterator:
        """
        Yields tuples of (x,y) values of the function within the range and interval specified.
        For example, for f(x) = x^2, in the range 2 and 4, and the step of 1, the function will
        yield (2,4), and then (3,9).
        Currently Available only to functions with one variable!

        :param start: the start value
        :param end: the end value
        :param step: the difference between each value
        :return: yields a (value,result) tuple every time
        """
        if self.__num_of_variables > 1:
            warnings.warn("Cannot give the range of functions with more than one variable!")
        for val in decimal_range(start, end, step):
            yield val, self.compute_value(val)

    def toIExpression(self):
        """
        Try to convert the function into an algebraic expression.
        :return: If successful, an algebraic expression will be returned. Otherwise - None
        """
        try:
            my_type = self.__classification
            poly_types = [self.Classification.linear, self.Classification.quadratic, self.Classification.polynomial
                , self.Classification.linear_several_parameters]
            if my_type in poly_types:
                return Poly(self.__func_expression)
            elif my_type == self.Classification.trigonometric:
                return TrigoExprs(self.__func_expression)
            elif my_type == self.Classification.logarithmic:
                return Log(self.__func_expression)
            elif my_type in (self.Classification.exponent, self.Classification.exponent_several_parameters):
                return Exponent(self.__func_expression)
            else:
                raise ValueError
        except:
            raise ValueError("Cannot convert the function to an algebraic expression! Either the method is "
                             "invalid, or it's not supported yet for this feature. Wait for next versions!")
            return None

    def derivative(self) -> "Optional[Union[Function, int, float]]":
        """
        Try to compute the derivative of the function. If not successful - return None
        :return: a string representation of the derivative of the function
        """
        num_of_variables = self.__num_of_variables
        if num_of_variables == 0:
            return 0
        elif num_of_variables == 1:
            poly_types = [self.Classification.linear, self.Classification.quadratic, self.Classification.polynomial]
            if self.__classification in poly_types:
                poly_string = Poly(self.__func_expression).derivative().__str__()
                return Function(poly_string)
            my_expression = self.toIExpression()
            if my_expression is None or not hasattr(my_expression, "derivative"):
                return None
            return Function(my_expression.derivative())
        else:
            raise ValueError("Use the partial_derivative() method for functions with multiple variables")

    def partial_derivative(self, variables: Iterable) -> "Optional[Union[Function, int, float]]":
        """Experimental Feature: Try to compute the partial derivative of the function."""
        num_of_variables = self.__num_of_variables
        if num_of_variables == 0:
            return 0
        elif num_of_variables == 1:
            return self.derivative()
        else:
            my_expression = self.toIExpression()
            if my_expression is None or not hasattr(my_expression, "partial_derivative"):
                return None
            return Function(my_expression.partial_derivative(variables).__str__())

    def integral(self) -> "Optional[Union[Function, int, float]]":
        """Computing the integral of the function. Currently without adding C"""
        num_of_variables = self.__num_of_variables
        if num_of_variables > 1:
            raise ValueError("Integrals with multiple variables are not supported yet")
        my_expression = self.toIExpression()
        if my_expression is None or not hasattr(my_expression, "integral"):
            return None
        return my_expression.integral()

    def to_dict(self):
        return {
            "type": "function",
            "string": self.__func
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    def export_json(self, path: str):
        with open(path, 'w') as json_file:
            json_file.write(self.to_json())

    @staticmethod
    def from_dict(func_dict: dict):
        return Function(func_dict['string'])

    @staticmethod
    def from_json(json_string: str):
        return Function.from_dict(json.loads(json_string))

    def trapz(self, a: float, b: float, N: int):
        return trapz(self.__lambda_expression, a, b, N)

    def simpson(self, a: float, b: float, N: int):
        return simpson(self.__lambda_expression, a, b, N)

    def reinman(self, a: float, b: float, N: int):
        return reinman(self.__lambda_expression, a, b, N)

    def range(self, start: float, end: float, step: float, round_results: bool = False):
        """
        fetches all the valid values of a function in the specified range
        :param start: the beginning of the range
        :param end: the end of the range
        :param step: the interval between each item in the range
        :return: returns the values in the range, and their valid results
        """
        if round_results:
            values = [round_decimal(i) for i in decimal_range(start, end, step)]
            results = [self.compute_value(i) for i in values]
            for index, result in enumerate(results):
                if result is not None:
                    results[index] = round_decimal(result)
        else:
            values = [i for i in decimal_range(start, end, step)]
            results = [self.compute_value(i) for i in values]
        for index, result in enumerate(results):
            if result is None:
                del results[index]
                del values[index]
            elif isinstance(result, bool):
                results[index] = float(result)
        return values, results

    def range_3d(self, x_start: float, x_end: float, x_step: float, y_start: float, y_end: float, y_step: float,
                 round_results: bool = False):
        x_values, y_values, z_values = [], [], []
        for x in decimal_range(x_start, x_end, x_step):
            for y in decimal_range(y_start, y_end, y_step):
                if round_results:
                    x_values.append(round_decimal(x))
                    y_values.append(round_decimal(y))
                    z = self.compute_value(x, y)
                    z_values.append(round_decimal(z))
                else:
                    x_values.append(x)
                    y_values.append(y)
                    z = self.compute_value(x, y)
                    z_values.append(z)

        return x_values, y_values, z_values

    def random(self, a: int = 1, b: int = 10, custom_values=None, as_point=False, as_tuple=False):
        """ returns a random value from the function"""
        if self.num_of_variables == 1:
            random_number = random.randint(a, b) if custom_values is None else custom_values
            if not as_point:
                if as_tuple:
                    return random_number, self.compute_value(random_number)
                return self.compute_value(random_number)
            return Point2D(random_number, self.compute_value(random_number))
        else:
            values = [random.randint(a, b) for _ in
                      range(self.num_of_variables)] if custom_values is None else custom_values
            if not as_point:
                if as_tuple:
                    return values, self.compute_value(*values)
                return self.compute_value(*values)
            values.append(self.compute_value(*values))
            if len(values) == 3:
                return Point3D(values[0], values[1], values[2])
            return Point(values)

    def plot(self, start: float = -10, stop: float = 10, step: float = 0.05, ymin: float = -10, ymax: float = 10
             , text: str = None, others: "Optional[Iterable[Function]]" = None, fig=None, ax=None
             , show_axis=True, show=True, formatText=True, values=None):
        """ plots the graph of the function using matplotlib.
        currently operational only for 1 parameter functions """
        if self.__num_of_variables == 1:
            if others is None:
                plot_function(func=self.lambda_expression, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax,
                              title=text, show_axis=show_axis, show=show, fig=fig, ax=ax, formatText=formatText,
                              values=values)
            else:

                funcs = [func for func in others]
                funcs.append(self)
                plot_functions(funcs, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax,
                               show_axis=show_axis, title=text, show=show)  # TODO: fix this!!!!!!!
        elif self.__num_of_variables == 2:
            # functions with two variables_dict in the form f(x,y) = .... can plotted in 3D
            plot_function_3d(given_function=self.lambda_expression, start=start, stop=stop, step=step)

    def scatter(self, start: float = -3, stop: float = 3,
                step: float = 0.3, show_axis=True, show=True):
        num_of_variables = len(self.__variables)
        if num_of_variables == 1:
            self.scatter2d(start=start, step=step, show_axis=show_axis, show=show)
        elif num_of_variables == 2:
            self.scatter3d(start=start, stop=stop, step=step, show=show)
        else:
            raise ValueError(f"Cannot scatter a function with {num_of_variables} variables")

    def scatter3d(self, start: float = -3, stop: float = 3,
                  step: float = 0.3,
                  xlabel: str = "X Values",
                  ylabel: str = "Y Values", zlabel: str = "Z Values", show=True, fig=None, ax=None,
                  write_labels=True, meshgrid=None, title=""):
        return scatter_function_3d(
            func=self.__lambda_expression, start=start, stop=stop, step=step, xlabel=xlabel, ylabel=ylabel,
            zlabel=zlabel, show=show, fig=fig, ax=ax, write_labels=write_labels, meshgrid=meshgrid, title=title)

    def scatter2d(self, start: float = -15, stop: float = 15, step: float = 0.3, ymin=-15, ymax=15, show_axis=True,
                  show=True, basic=True):
        if basic:
            scatter_function(self.__lambda_expression, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax,
                             show_axis=show_axis, show=show)
            return
        values, results = self.range(start, stop, step, round_results=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('equal')
        # And a corresponding grid
        ax.grid(which='both')

        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        if show_axis:
            # Set bottom and left spines as x and y axes of coordinate system
            ax.spines['bottom'].set_position('zero')
            ax.spines['left'].set_position('zero')

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Create 'x' and 'y' labels placed at the end of the axes
        plt.title(fr"${format_matplot(self.__func)}$", fontsize=14)

        norm = plt.Normalize(-10, 10)
        cmap = plt.cm.RdYlGn
        colors = [-5 for _ in range(len(results))]
        # plt.plot(values, results, 'o')
        sc = plt.scatter(x=values, y=results, c=colors, s=90, cmap=cmap, norm=norm)
        annotation = ax.annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                                 bbox=dict(boxstyle="round", fc="w"),
                                 arrowprops=dict(arrowstyle="->"))
        annotation.set_visible(False)

        def hover(event):
            vis = annotation.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    pos = sc.get_offsets()[ind["ind"][0]]
                    annotation.xy = pos
                    text = f"{pos[0], pos[1]}"
                    annotation.set_text(text)
                    annotation.get_bbox_patch().set_facecolor(cmap(norm(colors[ind["ind"][0]])))
                    annotation.get_bbox_patch().set_alpha(0.4)
                    annotation.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annotation.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.ylim(ymin, ymax)
        plt.ylim(ymin, ymax)
        if show:
            plt.show()

    @staticmethod
    def plot_all(*functions, show_axis=True, start=-10, end=10, step=0.01, ymin=-20, ymax=20):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('equal')

        # And a corresponding grid
        ax.grid(which='both')

        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        if show_axis:
            # Set bottom and left spines as x and y axes of coordinate system
            ax.spines['bottom'].set_position('zero')
            ax.spines['left'].set_position('zero')

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        for func_object in functions:
            if isinstance(func_object, str) or is_lambda(func_object):
                func_object = Function(func_object)

            if isinstance(func_object, Function):
                values, results = func_object.range_gen(start, end, step)
                plt.plot(values, results)
            else:
                raise TypeError(f"Invalid type for a function: type {type(func_object)} ")
        plt.ylim(ymin, ymax)
        plt.show()

    def search_roots_in_range(self, val_range: tuple, step=0.01, epsilon=0.01, verbose=True):
        """
        Iterates on the function in a certain range, to find the estimated roots in the range.
        This is not a rather efficient method compared to other root-finding algorithms, however, it should work in
        almost all cases and types of functions, depending on the step of each iteration and the epsilon.

        :param val_range: A tuple with the range which the roots will be searched in. For example, (-50,50)
        :param step: The interval between each X value in the iteration.
        :param epsilon: If y of a point is smaller than epsilon, the corresponding x will be considered a root.
        :return:Returns a list with all of the different roots which were found ( these roots were also rounded ).
        """
        if verbose:
            warnings.warn("The use of this method for finding roots is not recommended!")
        if len(val_range) != 2:
            raise IndexError(
                "The range of values must be an iterable containing 2 items: the minimum value, and the maximum."
                "\nFor example, (0,10) "
            )

        values, results = self.range_gen(val_range[0], val_range[1], step)
        matching_values = [(value, result) for value, result in
                           zip(values, results) if values is not None and result is not None and abs(result) < epsilon]
        return [(round_decimal(value), round_decimal(result)) for value, result in matching_values]

    def newton(self, initial_value: float):
        """
        Finds a single root of the function with the Newton-Raphson method.

        :param initial_value: An arbitrary number, preferably close the root.
        :return: The closest root of the function to the given initial value.

        """
        if self.__lambda_expression is not None:
            return newton_raphson(self.__lambda_expression, self.derivative().__lambda_expression, initial_value)
        return newton_raphson(self, self.derivative(), initial_value)

    def finite_integral(self, a, b, N: int, method: str = 'trapz'):
        if not isinstance(method, str):
            raise TypeError(f"Invalid type for param 'method' in method finite_integral() of class Function."
                            f"Expected type 'str'.")
        method = method.lower()
        if method == 'trapz':
            return self.trapz(a, b, N)
        elif method == 'simpson':
            return self.simpson(a, b, N)
        elif method == 'reinman':
            return self.reinman(a, b, N)
        else:
            raise ValueError(f"Invalid method '{method}'. The available methods are 'trapz' and 'simpson'. ")

    def coefficients(self):  # TODO: implement this, or check if analyze can be used for this in someway
        if self.__classification in (self.Classification.polynomial, self.Classification.quadratic,
                                     self.Classification.linear, self.Classification.constant):
            return Poly(self.__func_expression).coefficients()
        else:
            raise ValueError(f"Function's classification ({self.__classification}) doesn't support this feature.")

    # TODO: add more root-finding algorithms here
    def roots(self, epsilon=0.000001, nmax=100000):
        return aberth_method(self.__to_lambda(), self.derivative().__to_lambda(), self.coefficients(), epsilon, nmax)

    def max_and_min(self):
        """
        tries to find the max and min points
        """
        if self.__classification not in (
                self.Classification.quadratic, self.Classification.polynomial, self.Classification.linear):
            raise NotImplementedError
        first_derivative = self.derivative()
        second_derivative = first_derivative.derivative()
        derivative_roots = aberth_method(first_derivative.lambda_expression, second_derivative.lambda_expression,
                                         first_derivative.coefficients())
        derivative_roots = (root for root in derivative_roots if
                            abs(root.imag) < 0.000001)  # Accepting only real solutions for now
        max_points, min_points = [], []
        for derivative_root in list(derivative_roots):
            val = second_derivative(derivative_root.real)  # if 0, it's not min and max
            value = derivative_root.real
            result = round_decimal(self.lambda_expression(value))
            if val.real > 0:
                min_points.append((value, result))
            elif val.real < 0:
                max_points.append((value, result))
        return max_points, min_points

    def incline_and_decline(self):
        return NotImplementedError

    def chain(self, other_func: "Optional[Union[Function,str]]"):
        if isinstance(other_func, Function):
            return FunctionChain(self, other_func)
        else:
            try:
                return FunctionChain(self, Function(other_func))
            except ValueError:
                raise ValueError(f"Invalid value {other_func} when creating a FunctionChain object")
            except TypeError:
                raise TypeError(f"Invalid type {type(other_func)} when creating a FunctionChain object")

    def y_intersection(self):
        """
        Finds the intersection of the function with the y axis

        :return: Returns the y value when x = 0, or None, if function is not defined in x=0, or an error has occurred.
        """
        try:
            return self.compute_value(0)
        except:
            return None

    def __call__(self, *parameters):
        return self.compute_value(*parameters)

    def __str__(self):
        return self.__func

    def __repr__(self):
        return f'Function("{self.__func}")'

    def __getitem__(self, item):
        """
        :param item: a slice object which represents indices range or an int that represent index
        :return: returns the variable name in the index, or the variable names in the indices in case of slicing
        """
        if isinstance(item, slice):  # handling slicing
            start = item.start
            step = 1 if item.step is None else item.step
            return [self.__variables[i] for i in range(start, item.stop, step)]
        elif isinstance(item, int):
            return self.__variables[item]

    def __setitem__(self, key, value):
        return self.__variables.__setitem__(key, value)

    def __delitem__(self, key):
        return self.__variables.__delitem__(key)

    def __eq__(self, other):
        """ when equating between 2 functions, a list of the approximations of their intersections will be returned
        :rtype: list
        """
        if other is None:
            return False
        if isinstance(other, str):
            other = Function(other)
        if isinstance(other, Function):
            if other.__num_of_variables != other.__num_of_variables:
                return False
            for _ in range(3):  # check equality between 3 random values
                values, results = self.random(as_tuple=True)
                other_results = other.random(custom_values=values)
                if results != other_results:
                    return False

            return self.__lambda_expression.__code__.co_code == other.__lambda_expression.__code__.co_code
        elif is_lambda(other):
            other_num_of_variables = other.__code__.co_argcount
            if self.__num_of_variables != other_num_of_variables:
                return False
            for _ in range(3):  # check equality between 3 random values
                values = [random.randint(1, 10) for _ in range(other_num_of_variables)]
                my_results, other_results = self.compute_value(*values), other.compute_value(*values)
                if my_results != other_results:
                    return False

            return self.__lambda_expression.__code__.co_code == other.__lambda_expression.__code__.co_code

        else:
            raise TypeError(f"Invalid type {type(other)} for equating, allowed types: Function,str,lambda expression")

    def __ne__(self, other):
        return not self.__eq__(other)

    def __to_lambda(self):
        """Returns a lambda expression from the function. You should use the lambda_expression property."""
        return to_lambda(self.__func_expression, self.__variables,
                         (list(TRIGONOMETRY_CONSTANTS.keys()) + list(MATHEMATICAL_CONSTANTS.keys())),
                         format_abs=True, format_factorial=True)

    def search_intersections(self, other_func, values_range=(-100, 100), step=0.01, precision=0.01):
        """
        Returns the intersections between the current function to another function
        Currently works only in functions with only one parameter ... """
        if isinstance(other_func, str):
            other_func = Function(other_func)
        elif inspect.isfunction(other_func):  # handle user-defined functions and lambdas
            if is_lambda(other_func):
                other_func = Function(other_func)
            else:
                raise TypeError("Cannot perform intersection! only lambda functions are supported "
                                "so far in this version.")
        if isinstance(other_func, Function):
            intersections = []
            for i in np.arange(values_range[0], values_range[1], step):
                first_result = self.compute_value(i)
                if first_result is None:
                    continue
                first_result = round_decimal(first_result)
                second_result = other_func.compute_value(i)
                if second_result is None:
                    continue
                second_result = round_decimal(second_result)
                if abs(first_result - second_result) <= precision:
                    found = False
                    for x, y in intersections:
                        x_difference = abs(round_decimal(x - i))
                        if x_difference <= precision:
                            found = True
                            break
                    if not found:
                        intersections.append(
                            (round_decimal(i), round_decimal((first_result + second_result) / 2)))  # add the best
                        # approximation for y
            return intersections

        else:
            raise TypeError(f"Invalid type {type(other_func)} for a function. Use types Function ( recommended) or str"
                            f"instead")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

    def __copy__(self):  # Inefficient single the lambda expression will be re-calculated for no reason ...
        return Function(func=self.__func)


class FunctionCollection(IPlottable, IScatterable):
    def __init__(self, *functions, gen_copies=False):
        self._functions = []
        for func in functions:
            if isinstance(func, str) or is_lambda(func):  # lambda expressions might not work here
                self._functions.append(Function(func))
            elif isinstance(func, Function):
                if gen_copies:
                    self._functions.append(func.__copy__())
                else:
                    self._functions.append(func)
            elif isinstance(func, (FunctionChain, FunctionCollection)):
                if all(isinstance(f, Function) for f in func):
                    for f in func:
                        if gen_copies:
                            self._functions.append(f.__copy__())
                        else:
                            self._functions.append(f)
                else:
                    pass  # Recursive Algorithm to break down anything, or raise an Error

    @property
    def functions(self):
        return self._functions

    @property
    def num_of_functions(self):
        return self._functions

    @property
    def variables(self):
        variables_set = set()
        for func in self._functions:
            variables_set.update(func.variables)
        return variables_set

    @property
    def num_of_variables(self):
        if not self._functions:
            return 0
        return len(self.variables)

    def clear(self):
        self._functions = []

    def is_empty(self):
        return not self._functions

    def add_function(self, func: Union[Function, str]):
        if isinstance(func, Function):
            self._functions.append(func)
        elif isinstance(func, str):
            self._functions.append(Function(func))
        else:
            raise TypeError(f"Invalid type {type(func)}. Allowed types for this method are 'str' and 'Function'")

    def extend(self, functions: Iterable[Union[Function, str]]):
        for function in functions:
            if isinstance(function, str):
                function = Function(function)
            self._functions.append(function)

    def values(self, *args, **kwargs):
        pass

    def random_function(self):
        return random.choice(self._functions)

    def random_value(self, a: Union[int, float], b: Union[int, float], mode='int'):
        my_random_function = self.random_function()
        if a > b:
            a, b = b, a
        num_of_variables = my_random_function.num_of_variables
        if mode == 'float':
            parameters = [random.uniform(a, b) for _ in range(num_of_variables)]
        elif mode == 'int':
            parameters = [random.randint(a, b) for _ in range(num_of_variables)]
        else:
            raise ValueError(f"invalid mode {mode}: expected 'int' or 'float'")
        return my_random_function(*parameters)

    def derivatives(self):
        if any(func.num_of_variables != 1 for func in self._functions):
            raise ValueError("All functions must have exactly 1 parameter (For this version)")
        return [func.derivative() for func in self._functions]

    def filter(self, predicate: Callable[[Function], bool]):
        return filter(predicate, self._functions)

    def __len__(self):
        return len(self._functions)

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):
        if self.__index < len(self._functions):
            value = self._functions[self.__index]
            self.__index += 1
            return value
        else:
            raise StopIteration

    def __getitem__(self, item):
        return FunctionCollection(*(self._functions.__getitem__(item)))

    def __str__(self):
        return "\n".join((f"{index + 1}. {function.__str__()}" for index, function in enumerate(self._functions)))

    def plot(self, start: float = -10, stop: float = 10,
             step: float = 0.01, ymin: float = -10, ymax: float = 10, text=None, show_axis=True, show=True):
        plot_functions(self._functions, start, stop, step, ymin, ymax, text, show_axis, show)

    def scatter(self, start: float = -10, stop: float = 10,
                step: float = 0.01, ymin: float = -10, ymax: float = 10, text=None, show_axis=True, show=True):
        scatter_functions(self._functions, start, stop, step, ymin, ymax, text, show_axis, show)


class FunctionChain(FunctionCollection):
    def __init__(self, *functions):
        super(FunctionChain, self).__init__(*functions)

    def execute_all(self, *args):
        """Execute all of the functions consecutively"""
        if not len(self._functions):
            raise ValueError("Cannot execute an empty FunctionChain object!")

        final_x = self._functions[0](*args)
        for index in range(1, len(self._functions)):
            final_x = self._functions[index](*args)
        return final_x

    def execute_reverse(self, *args):
        if not len(self._functions):
            raise ValueError("Cannot execute an empty FunctionChain object!")

        final_x = self._functions[-1](*args)
        for index in range(len(self._functions) - 2, 0, -1):
            final_x = self._functions[index](*args)
        return final_x

    def execute_indices(self, indices, *args):
        if not len(indices):
            raise ValueError("Cannot execute an empty FunctionChain object!")

        final_x = indices[0](*args)
        for index in indices[1:]:
            final_x = self._functions[index](*args)
        return final_x

    def __call__(self, *args):
        return self.execute_all(*args)

    def chain(self, func: "Union[Function, FunctionChain, str]"):
        self.add_function(func)
        return self

    def plot(self, start: float = -10, stop: float = 10,
             step: float = 0.01, ymin: float = -10, ymax: float = 10, title=None, show_axis=True, show=True,
             fig=None, ax=None, formatText=False, values=None):
        if not self._functions:
            raise ValueError("Cannot plot an empty FunctionChain object")
        num_of_variables = self._functions[0].num_of_variables
        if num_of_variables == 0:
            raise ValueError("Cannot plot functions without any variables")

        elif num_of_variables == 1:
            plot_function(
                self.execute_all, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title,
                show_axis=show_axis
                , show=show, fig=fig, ax=ax, formatText=formatText, values=values)

        elif num_of_variables == 2:
            return plot_function_3d(self.execute_all, start=start, stop=stop, step=step, ax=ax, fig=fig,
                                    meshgrid=values)
        else:
            raise ValueError(f"Can only plot functions with one or two variables: found ")

    def scatter(self, start: float = -10, stop: float = 10,
                step: float = 0.01, ymin: float = -10, ymax: float = 10, title=None, show_axis=True, show=True,
                fig=None, ax=None, values=None):
        scatter_function(func=self.execute_all, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title,
                         fig=None, ax=None)

    def __getitem__(self, item):
        return FunctionChain(*(self._functions.__getitem__(item)))





def format_matplot_polynomial(expression: str):
    """
    formats a polynomial expression into matplot's format
    :param expression:
    :return:
    """
    # SPAGHETTI CODE ALERT !!!
    expressions = split_expression(expression)
    for index, expr in enumerate(expressions):
        expr = expr.replace('**', '^')
        accumulator = ""
        skip = 0
        for i in range(len(expr)):
            character = expr[i]
            if skip == 0:
                if character == '^':
                    accumulator = "".join((accumulator, '^{'))
                    j = i + 1
                    while j < len(expr) and expr[j] not in ('^', '*', '+', '-'):
                        accumulator += expr[j]
                        j += 1
                    accumulator = "".join((accumulator, '}'))
                    skip = j - i - 1
                else:
                    accumulator += character
            else:
                skip -= 1
        expressions[index] = rf'{accumulator}'
    return f'{"".join(expressions)}'


# TODO: implement these methods !
def format_matplot_function(expression: str):
    raise NotImplementedError


def format_matplot(expression: str):
    return format_matplot_polynomial(expression)


class Sequence(ABC):

    @property
    @abstractmethod
    def first(self):
        pass

    @abstractmethod
    def in_index(self, index: int) -> float:
        pass

    @abstractmethod
    def index_of(self, item: float) -> float:
        pass

    @abstractmethod
    def sum_first_n(self, n: int) -> float:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    def range(self, start: int, stop: int):
        return (self.in_index(current_index) for current_index in range(start, stop))

    def product_in_range(self, start: int, end: int):
        return reduce(lambda a, b: a * b, (self.in_index(i) for i in range(start, end)))

    def product_first_n(self, end: int):
        return self.product_in_range(0, end)

    def sum_in_range(self, start: int, end: int):  # TODO: improve this
        return sum(self.in_index(current_index) for current_index in range(start, end))

    def __contains__(self, item: float):
        index = self.index_of(item)
        return index > 0

    def __getitem__(self, item):
        if isinstance(item, slice):  # handling slicing
            if item.start == 0:
                warnings.warn("Sequence indices start from 1 and not from 0, skipped to 1")
                start = 1
            else:
                start = item.start
            step = 1 if item.step is None else item.step
            return [self.in_index(i) for i in range(start, item.stop + 1, step)]
        elif isinstance(item, int):
            return self.in_index(item)

    def __generate_data(self, start: int, stop: int, step: int):
        return list(range(start, stop, step)), [self.in_index(index) for index in range(start, stop, step)]

    def plot(self, start: int, stop: int, step: int = 1, show=True):
        axes, y_values = self.__generate_data(start, stop, step)
        plt.plot(axes, y_values)
        if show:
            plt.show()


class GeometricSeq(Sequence, IPlottable):
    """
    A class that represents a geometric sequence, namely, a sequence in which every item can be
    multiplied by a constant (the ratio of the sequence) to reach the next item.
    """

    def __init__(self, first_numbers: Union[tuple, list, set, str, int, float], ratio: float = None):
        """Create a new GeometricSeq object"""

        if isinstance(first_numbers, str):
            if ',' in first_numbers:
                first_numbers = [float(i) for i in tuple(first_numbers.split(','))]
            else:
                first_numbers = [float(i) for i in tuple(first_numbers.split(' '))]
        elif isinstance(first_numbers, (int, float)):
            first_numbers = [first_numbers]

        if isinstance(first_numbers, (tuple, list, set)):
            if not first_numbers:
                raise ValueError("GeometricSeq.__init__(): Cannot accept an empty collection for parameter "
                                 "'first_numbers'")
            if any(number == 0 for number in first_numbers):
                raise ValueError("GeometricSeq.__init__(): Zeroes aren't allowed in geometric sequences")
            self.__first = first_numbers[0]
            if ratio is not None:
                self.__ratio = ratio
                return
            # We get here only if ratio is None
            if len(first_numbers) == 1:
                raise ValueError("GeometricSeq.__init__(): Please Enter more initial values, or specify the ratio of "
                                 "the sequence.")
            self.__ratio = first_numbers[1] / first_numbers[0]

        else:
            raise TypeError(f"GeometricSeq.__init__():"
                            f"Invalid type {type(first_numbers)} for parameter 'first_numbers'. Expected types"
                            f" 'tuple', 'list', 'set', 'str', 'int', 'float' ")

    @property
    def first(self):
        return self.__first

    @property
    def ratio(self):
        return self.__ratio

    def in_index(self, index: int) -> float:
        return self.__first * pow(self.__ratio, (index - 1))

    def index_of(self, item: float) -> float:
        result = log(item / self.__first, self.__ratio) + 1
        if not result.is_integer():
            return -1
        return result

    def sum_first_n(self, n: int) -> float:
        return self.__first * (self.__ratio ** n - 1) / (self.__ratio - 1)

    def __repr__(self):
        return f"Sequence(first_numbers=({self.__first},),ratio={self.__ratio})"

    def __str__(self):
        return f"{self.__first}, {self.in_index(2)}, {self.in_index(3)} ... (ratio = {self.__ratio})"


class ArithmeticProg(Sequence):
    """A class for representing arithmetic progressions. for instance: 2, 4, 6, 8, 10 ..."""

    def __init__(self, first_numbers: Union[tuple, list, set, str, int, float], difference: float = None):
        if isinstance(first_numbers, str):
            if ',' in first_numbers:
                first_numbers = [float(i) for i in tuple(first_numbers.split(','))]
            else:
                first_numbers = [float(i) for i in tuple(first_numbers.split(' '))]
        elif isinstance(first_numbers, (int, float)):
            first_numbers = [first_numbers]

        if isinstance(first_numbers, (tuple, list, set)):
            if not first_numbers:
                raise ValueError("ArithmeticProg.__init__(): Cannot accept an empty collection for parameter "
                                 "'first_numbers'")
            self.__first = first_numbers[0]
            if difference is not None:
                self.__difference = difference
                return
            # We get here only if the difference is None
            if len(first_numbers) == 1:
                raise ValueError("ArithmeticProg.__init__(): Please Enter more initial values,"
                                 " or specify the difference of the sequence.")
            self.__difference = first_numbers[1] - first_numbers[0]

        else:
            raise TypeError(f"ArithmeticProg.__init__():"
                            f"Invalid type {type(first_numbers)} for parameter 'first_numbers'. Expected types"
                            f" 'tuple', 'list', 'set', 'str', 'int', 'float' ")

    @property
    def first(self):
        return self.__difference

    @property
    def difference(self):
        return self.__difference

    def in_index(self, index: int) -> float:
        return self.__first + self.__difference * (index - 1)

    def index_of(self, item: float) -> float:
        result = (item - self.__first) / self.__difference + 1
        if not result.is_integer():
            return -1
        return result

    def sum_first_n(self, n: int) -> float:
        return 0.5 * n * (2 * self.__first + (n - 1) * self.__difference)

    def __str__(self):
        return f"{self.__first}, {self.in_index(2)}, {self.in_index(3)} ... (difference = {self.__difference})"

    def __repr__(self):
        return f"Sequence(first_numbers=({self.__first},),difference={self.__difference})"


def lambda_from_recursive(
        recursive_function: str):  # For now assuming it's of syntax a_n = ....... later on order it with highest power at left?
    elements = set(ptn.findall(recursive_function))
    elements = sorted(elements, key=lambda element: 0 if "{" not in element else float(
        element[element.find('n') + 1:element.find('}')]))
    indices = [(element[element.find('{') + 1:element.find('}')] if '{' in element else 'n') for element in elements]
    new_elements = [element.replace('{', '').replace('}', '').replace('+', 'p').replace('-', 'd').replace('n', 'k') for
                    element in
                    elements]  # that's enough for now
    recursive_function = recursive_function[recursive_function.find('=') + 1:]
    for element, new_element in zip(elements, new_elements):
        recursive_function = recursive_function.replace(element, new_element)
    del new_elements[-1]
    new_elements.append('n')
    lambda_expression = to_lambda(recursive_function, new_elements,
                                  (list(TRIGONOMETRY_CONSTANTS.keys()) + list(MATHEMATICAL_CONSTANTS.keys())))
    del indices[-1]
    return lambda_expression, indices  # Chef's kiss


class RecursiveSeq(Sequence):
    def __init__(self, recursive_function: str, first_values: Iterable):
        """
        Create a new instance of a recursive sequence.

        :param recursive_function:
        :param first_values:
        """
        self.__first_values = {index: value for index, value in enumerate(first_values)}
        self.__recursive_string = recursive_function.strip()
        self.__lambda, self.__indices = lambda_from_recursive(self.__recursive_string)

    @property
    def first(self):
        return self.__first_values[0]

    def in_index(self, n: int, accumulate=True):
        return self.at_n(n, accumulate)

    def index_of(self, item):
        raise NotImplementedError

    def sum_first_n(self, n: int):
        raise NotImplementedError

    def at_n(self, n: int, accumulate=True):
        if n == 0:
            raise ValueError("Sequence indices start from 1, not from 0 - a1,a2,a3....")
        return self.__at_n(n - 1, accumulate)

    def __at_n(self, n: int, accumulate=True):  # if accumulate set to true, these values will be saved in a buffer
        """
        Get the nth element in the series.

        :param n: The place of the desired element. Must be an integer and greater than zero.
        :param accumulate: If set to true, results of computations will be saved to shorten execution time ( on the expense of the allocated memory).
        :return: Returns the nth element of the series.
        """
        if len(self.__indices) > len(self.__first_values) - 1:  # TODO: modify this condition later
            raise ValueError(
                f"Not enough initial values were entered for the series, got {len(self.__first_values)}, expected at least {len(self.__indices)} values")
        if n in self.__first_values:  # if there is already a computed answer for the calculation
            return self.__first_values[n]
        new_indices = [int(eval(old_index.replace('k', str(n)))) for old_index in
                       self.__indices]  # TODO: Later modify this too
        pre_defined_values, undefined_indices = [], []
        for new_index in new_indices:
            if new_index in self.__first_values:
                pre_defined_values.append(self.__first_values[new_index])
            else:
                undefined_indices.append(new_index)

        if undefined_indices:
            pre_defined_values.extend([self.__at_n(index, accumulate) for index in undefined_indices])
        pre_defined_values.append(n + 1)  # Decide what to do about the indices
        result = self.__lambda(*pre_defined_values)  # The item on place N
        if accumulate:
            self.__first_values[n] = result
        return result

    def place_already_found(self, n: int) -> bool:
        """
        Checks if the value in the specified place in the recursive series has already been computed.

        :param n: The place on the series, starting from 1. Must be an integer.
        :return: Returns True if the value has been computed, otherwise, False
        """
        return n in self.__first_values.keys()

    def __str__(self):
        return f"{self.__recursive_string}"


class Point:
    def __init__(self, coordinates: Union[Iterable, int, float]):
        if isinstance(coordinates, Iterable):
            self._coordinates = [coordinate for coordinate in coordinates]
            for index, coordinate in enumerate(self._coordinates):
                if isinstance(coordinate, IExpression):
                    self._coordinates[index] = coordinate.__copy__()
        elif isinstance(coordinates, (int, float)):
            self._coordinates = [coordinates]
        else:
            raise TypeError(f"Invalid type {type(coordinates)} for creating a new Point object")

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates: Iterable):
        self._coordinates = coordinates

    @property
    def dimensions(self):
        return len(self._coordinates)

    def plot(self):
        self.scatter()

    def scatter(self, show=True):  # TODO: create it with a grid and stuff
        if len(self._coordinates) == 1:
            plt.scatter(self._coordinates[0], 0)
        if len(self._coordinates) == 2:
            plt.scatter(self._coordinates[0], self._coordinates[1])
        elif len(self._coordinates) == 3:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter2d(self._coordinates[0], self._coordinates[1], self._coordinates[2])
        if show:
            plt.show()

    def __iadd__(self, other: "Union[Iterable, Point]"):
        if isinstance(other, Iterable):
            self._coordinates = [coord1 + coord2 for coord1, coord2 in zip(self._coordinates, other)]
            return self
        elif isinstance(other, Point):
            self._coordinates = [coord1 + coord2 for coord1, coord2 in zip(self._coordinates, other._coordinates)]
            return self
        else:
            raise TypeError(f"Encountered unexpected type {type(other)} while attempting to add points. Expected types"
                            f"Iterable or Point")

    def __isub__(self, other: "Union[Iterable, Point]"):
        if isinstance(other, Point):
            self._coordinates = [coord1 - coord2 for coord1, coord2 in zip(self._coordinates, other._coordinates)]
            return self
        elif isinstance(other, Iterable):
            self._coordinates = [coord1 - coord2 for coord1, coord2 in zip(self._coordinates, other)]
            return self

        else:
            raise TypeError(f"Encountered unexpected type {type(other)} while attempting to subtract points. Expected"
                            f"types Iterable or Point")

    def __add__(self, other: "Union[Iterable, Point]"):
        return self.__copy__().__iadd__(other)

    def __radd__(self, other: "Union[Iterable, Point]"):
        if isinstance(other, Iterable):
            other = Point(Iterable)

        if isinstance(other, Point):
            return other.__add__(self)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        return self.__copy__().__isub__(other)

    def __rsub__(self, other: "Union[Point,PointCollection]"):
        if isinstance(other, Iterable):
            other = Point(Iterable)

        if isinstance(other, Point):
            return other.__sub__(self)
        else:
            raise NotImplementedError

    def __imul__(self, other: "Union[int, float, Point, PointCollection, IExpression]"):
        if isinstance(other, (int, float, IExpression)):
            if isinstance(other, IExpression):
                other_evaluation = other.try_evaluate()
                if other_evaluation is not None:
                    other = other_evaluation
            for index in range(len(self._coordinates)):
                self._coordinates[index] *= other
                return self
        elif isinstance(other, Point):
            return reduce(lambda tuple1, tuple2: tuple1[0] * tuple2[0] + tuple1[1] * tuple2[1],
                          zip(self._coordinates, other._coordinates))
        elif isinstance(other, PointCollection):
            raise NotImplementedError("This feature isn't implemented yet in this version")

    def __mul__(self, other: "Union[int, float, Point, PointCollection, IExpression]"):
        return self.__copy__().__imul__(other)

    def __rmul__(self, other: "Union[int, float, Point, PointCollection, IExpression]"):
        return self.__copy__().__imul__(other)

    def coord_at(self, index: int):
        return self._coordinates[index]

    def max_coord(self):
        return max(self._coordinates)

    def min_coord(self):
        return min(self._coordinates)

    def sum(self):
        return sum(self._coordinates)

    def distance(self, other_point: "Point"):
        if len(self.coordinates) != len(other_point.coordinates):
            raise ValueError(f"Cannot calculate distance between points in different dimensions.")
        return sqrt(sum((coord1 - coord2) ** 2 for (coord1, coord2) in zip(self.coordinates, other_point.coordinates)))

    def __eq__(self, other: "Union[Point, PointCollection]"):
        if other is None:
            return False
        if isinstance(other, PointCollection):
            if len(other.points) != 1:
                return False
            other = other.points[0]

        if isinstance(other, Point):
            return self._coordinates == other._coordinates
        else:
            raise TypeError(f"Invalid type {type(other)} for comparing with a Point object")

    def __ne__(self, other: "Union[Point, PointCollection]"):
        return not self.__eq__(other)

    def __neg__(self):
        return Point(coordinates=[-coordinate for coordinate in self._coordinates])

    def __repr__(self):
        return f"Point({self._coordinates})"

    def __str__(self):
        if all(isinstance(coordinate, (int, float)) for coordinate in self._coordinates):
            return f"({','.join(str(round(coordinate, 3)) for coordinate in self._coordinates)})"
        return f"({','.join(coordinate.__str__() for coordinate in self._coordinates)})"

    def __copy__(self):
        return Point(self._coordinates)  # New coordinates will be created in the init, so memory won't be shared
        # between different objects

    def __len__(self):
        return len(self._coordinates)


class Point1D(Point, IPlottable):
    def __init__(self, x: Union[int, float, IExpression]):
        super(Point1D, self).__init__((x,))

    @property
    def x(self):
        return self._coordinates[0]


class Point2D(Point, IPlottable):
    def __init__(self, x: Union[int, float, IExpression], y: Union[int, float, IExpression]):
        super(Point2D, self).__init__((x, y))

    @property
    def x(self):
        return self._coordinates[0]

    @property
    def y(self):
        return self._coordinates[1]


class Point3D(Point):
    def __init__(self, x: Union[int, float, IExpression], y: Union[int, float, IExpression],
                 z: Union[int, float, IExpression]):
        super(Point3D, self).__init__((x, y, z))

    @property
    def x(self):
        return self._coordinates[0]

    @property
    def y(self):
        return self._coordinates[1]

    @property
    def z(self):
        return self._coordinates[2]


class Point4D(Point):
    def __init__(self, x: Union[int, float, IExpression], y: Union[int, float, IExpression],
                 z: Union[int, float, IExpression], c: Union[int, float, IExpression]):
        super(Point4D, self).__init__((x, y, z, c))

    @property
    def x(self):
        return self._coordinates[0]

    @property
    def y(self):
        return self._coordinates[1]

    @property
    def z(self):
        return self._coordinates[2]

    @property
    def c(self):
        return self._coordinates[3]


class Line2D(IPlottable):
    def __init__(self, point1: Union[Point2D, Iterable], point2: Union[Point2D, Iterable], gen_copies=True):
        if isinstance(point1, Point2D):
            self._point1 = point1.__copy__() if gen_copies else point1
        elif isinstance(point1, Iterable):
            x, y = point1
            self._point1 = Point2D(x, y)
        else:
            raise TypeError(f"Invalid type for param 'point1' when creating a Line object.")

        if isinstance(point2, Point2D):
            self._point2 = point2.__copy__() if gen_copies else point2
        elif isinstance(point2, Iterable):
            x, y = point2
            self._point2 = Point2D(x, y)
        else:
            raise TypeError(f"Invalid type for param 'point2' when creating a Line object.")

    def middle(self):
        return Point2D((self._point1.x + self._point2) / 2, (self._point1.y + self._point2.y) / 2)

    def length(self):
        inside_root = (self._point1.x - self._point2.x) ** 2 + (self._point1.y - self._point2.y) ** 2
        if isinstance(inside_root, (int, float)):
            return sqrt(inside_root)

    @property
    def slope(self):
        x1, x2 = self._point1.x, self._point2.x
        y1, y2 = self._point1.y, self._point2.y
        numerator, denominator = y2 - y1, x2 - x1
        if denominator is None:
            warnings.warn("There's no slope for a single x value with two y values.")
            return None
        return numerator / denominator

    @property
    def free_number(self):
        m = self.slope
        if m is None:
            warnings.warn("There's no free number for a single x value with two y values.")
            return None
        return self._point1.y - self._point1.x * m

    def equation(self):
        m = self.slope
        if m is None:
            warnings.warn("There's no slope for a single x value with two y values.")
            return None
        b = self._point1.y - self._point1.x * m
        m_str = format_coefficient(m)
        b_str = format_free_number(b)
        return f"{m_str}x{b_str}"

    def to_lambda(self):
        m = self.slope
        if m is None:
            warnings.warn("Cannot generate a lambda expression for a single x value with two y values.")
            return None
        b = self._point1.y - self._point1.x * m
        return lambda x: m * x + b

    def intersection(self):  # TODO: implement it.......
        pass

    def plot(self, start: float = -6, stop: float = 6, step: float = 0.3, ymin: float = -10,
             ymax: float = 10, title: str = None, formatText: bool = False,
             show_axis: bool = True, show: bool = True, fig=None, ax=None, values=None):
        my_lambda = self.to_lambda()
        if my_lambda is None:
            pass  # TODO: implement it.
        plot_function(my_lambda,
                      start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title,
                      show_axis=show_axis, show=show, fig=fig, formatText=formatText, ax=ax,
                      values=values)

    def scatter(self, start: float = -10, stop: float = 10,
                step: float = 0.05, ymin: float = -10, ymax: float = 10, title=None, show_axis=True, show=True,
                fig=None, ax=None, formatText=True, values=None):

        lambda_expression = self.to_lambda()
        if not lambda_expression:
            pass  # TODO: implement it

        if title is None:
            title = self.__str__()

        scatter_function(lambda_expression, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title,
                         show_axis=show_axis, show=show, fig=fig, ax=ax, values=values)


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


class Circle(IPlottable):
    def __init__(self, radius: Union[float, int, IExpression],
                 center: Union[Iterable[Union[int, float, IExpression]], Point] = (0, 0),
                 gen_copies=False):
        # Handling the radius
        if isinstance(radius, (int, float)):
            self._radius = Mono(radius)
        elif isinstance(radius, IExpression):
            if gen_copies:
                self._radius = radius.__copy__()
            else:
                self._radius = radius
        else:
            raise TypeError(f"Invalid type {type(radius)} for radius when creating a Circle object")
        # Handling the Center
        if isinstance(center, Iterable) and not isinstance(center, Point):
            center_list = [coordinate for coordinate in center]
            if any(not isinstance(coordinate, (IExpression, int, float)) for coordinate in center_list):
                raise TypeError(f"Invalid types of coordinates when creating a Circle object")
            for index, coordinate in enumerate(center_list):
                if isinstance(coordinate, (int, float)):
                    center_list[index] = Mono(coordinate)
            center = Point(center_list)

        if isinstance(center, Point):
            if center.dimensions != 2:
                raise ValueError(f"Circle object can only contain a 2D Point as a center ( Got {center.dimensions}D")
            self._center = center.__copy__() if gen_copies else center
        else:
            raise TypeError(f"Invalid type {type(center)} for the center point when creating a Circle object")

    @property
    def radius(self):
        return self._radius

    @property
    def diameter(self):
        return self._radius * 2

    @property
    def center(self) -> Point:
        return self._center

    @property
    def left_edge(self):
        return Point((-self._radius + self.center_x, self.center_y))

    @property
    def right_edge(self):
        return Point((self._radius + self.center_x, self.center_y))

    @property
    def top_edge(self):
        return Point((self.center_x, self._radius + self.center_y))

    @property
    def bottom_edge(self):
        return Point((self.center_x, -self._radius + self.center_y))

    @property
    def center_x(self):
        return self._center.coordinates[0]

    @property
    def center_y(self):
        return self._center.coordinates[1]

    def area(self):
        result = self._radius ** 2 * pi
        if isinstance(result, IExpression):
            result_eval = result.try_evaluate()
            if result_eval is not None:
                return result_eval
            return result
        return result

    def perimeter(self):
        result = self._radius * 2 * pi
        if isinstance(result, IExpression):
            result_eval = result.try_evaluate()
            if result_eval is not None:
                return result_eval
            return result
        return result

    def point_inside(self, point: Union[Point, Iterable], already_evaluated: Tuple[float, float, float] = None) -> bool:
        """
        Checks whether a 2D point is inside the circle

        :param point: the point
        :param already_evaluated: Evaluations of the radius and center point of the circle as floats.
        :return: Returns True if the point is indeed inside the circle or touches it from the inside, otherwise False.
        """
        if isinstance(point, Point):
            x, y = point.coordinates[0], point.coordinates[1]  # TODO: later accept only Point2D objects..
        elif isinstance(point, Iterable):
            coordinates = [coord for coord in point]
            if len(coordinates) != 2:  # TODO: later accept only Point2D objects..
                raise ValueError("Can only accept points with 2 dimensions")
            x, y = coordinates[0], coordinates[1]
        else:
            raise ValueError(f"Invalid type {type(point)} for this method.")
        if already_evaluated is not None:
            radius_eval, center_x_eval, center_y_eval = already_evaluated
        else:
            radius_eval = self._radius.try_evaluate()
            center_x_eval = self.center_x.try_evaluate()
            center_y_eval = self.center_y.try_evaluate()
        if None not in (radius_eval, center_x_eval, center_y_eval):
            # TODO: check for all edges
            if x > center_x_eval + radius_eval:  # After the right edge
                return False
            if x < center_x_eval - radius_eval:  # Before the right edge
                return False
            if y > center_y_eval + radius_eval:
                return False
            if y < center_y_eval - radius_eval:
                return False
            return True

        else:
            raise ValueError("This feature is only supported for Circles without any additional parameters")

    def is_inside(self, other_circle: "Circle") -> bool:
        if not isinstance(other_circle, Circle):
            raise TypeError(f"Invalid type '{type(other_circle)}'. Expected type 'circle'. ")
        my_radius_eval = self._radius.try_evaluate()
        my_center_x_eval = self.center_x.try_evaluate()
        my_center_y_eval = self.center_y.try_evaluate()
        other_radius_eval = other_circle._radius.try_evaluate()
        other_center_x_eval = other_circle.center_x.try_evaluate()
        other_center_y_eval = other_circle.center_y.try_evaluate()

        if None not in (my_radius_eval, my_center_x_eval, my_center_y_eval, other_radius_eval, other_center_x_eval,
                        other_center_y_eval):
            # Check for all edges
            if not other_circle.point_inside(self.top_edge, already_evaluated=(
                    other_radius_eval, other_center_x_eval, other_center_y_eval)):
                return False
            if not other_circle.point_inside(self.bottom_edge, already_evaluated=(
                    other_radius_eval, other_center_x_eval, other_center_y_eval)):
                return False
            if not other_circle.point_inside(self.right_edge, already_evaluated=(
                    other_radius_eval, other_center_x_eval, other_center_y_eval)):
                return False
            if not other_circle.point_inside(self.left_edge, already_evaluated=(
                    other_radius_eval, other_center_x_eval, other_center_y_eval)):
                return False
            return True
        else:
            raise ValueError("Can't determine whether a circle is inside another, when one or more of them "
                             "are expressed via parameters")

    def plot(self, fig=None, ax=None):
        radius_eval = self._radius.try_evaluate()
        center_x_eval = self.center_x.try_evaluate()
        center_y_eval = self.center_y.try_evaluate()
        if None in (radius_eval, center_x_eval, center_y_eval):
            raise ValueError("Can only plot circles with real numbers (and not algebraic expressions)")
        circle1 = plt.Circle((center_x_eval, center_y_eval), radius_eval, color='r', fill=False)
        if None in (fig, ax):
            fig, ax = plt.subplots()  # note we must use plt.subplots, not plt.subplot
        ax.add_patch(circle1)
        ax.set_aspect('equal', adjustable='datalim')
        ax.plot()  # Causes an autoscale update.
        plt.show()

    def to_lambda(self):
        warnings.warn("This is an experimental feature!")
        radius_evaluation = self._radius.try_evaluate()
        center_x_evaluation = self.center_x.try_evaluate()
        center_y_evaluation = self.center_y.try_evaluate()
        if None not in (radius_evaluation, center_x_evaluation, center_y_evaluation):
            # If we're just working with numbers
            return lambda x: (sqrt(abs(radius_evaluation ** 2 - (x - center_x_evaluation) ** 2)) + center_y_evaluation,
                              -sqrt(abs(radius_evaluation ** 2 - (x - center_x_evaluation) ** 2)) + center_y_evaluation)

        return lambda x: (Sqrt(Abs(self._radius ** 2 - (x - self.center_x) ** 2)) + self.center_y,
                          -Sqrt(Abs(self._radius ** 2 - (x - self.center_x) ** 2)) + self.center_y)

    @property
    def equation(self) -> str:
        x_part = _format_minus('x', self.center_x)

        if self.center_y == 0:
            y_part = "y^2"
        elif '+' in self.center_y.__str__() or '-' in self.center_y.__str__():
            y_part = f"(y-({self.center_y}))^2"
        else:
            y_part = f"(y-{self.center_y})^2"

        radius_eval = self._radius.try_evaluate()
        radius_part = f"{self._radius}^2" if radius_eval is None or (
                radius_eval is not None and radius_eval > 100) else f"{radius_eval ** 2}"
        return f"{x_part} + {y_part} = {radius_part}"

    def x_intersection(self):
        pass  # it's like the y intersection ...

    def _expression(self):
        x = Var('x')
        y = Var('y')
        return (x - self.center_x) ** 2 + (y - self.center_y) ** 2 - self._radius ** 2

    def intersection(self, other):  # This will work when poly_from_str() will be updated...
        if isinstance(other, Circle):
            if self.has_parameters() or other.has_parameters():
                raise ValueError("This feature hasn't been implemented yet for Circle equations with additional"
                                 "parameters")
            else:
                initial_x = (self.center_x + other.center_x) / 2  # The initial x is the average
                # between the x coordinates of the centers of the circles.
                initial_y = (self.center_y + other.center_y) / 2
                intersections = solve_poly_system([self._expression(), other._expression()],
                                                  initial_vals={'x': initial_x, 'y': initial_y})
                return intersections

    def has_parameters(self) -> bool:
        coefficient_eval = self._radius.try_evaluate()
        if coefficient_eval is None:
            return True
        center_x_eval = self.center_x.try_evaluate()
        if center_x_eval is None:
            return True
        center_y_eval = self.center_y.try_evaluate()
        if center_y_eval is None:
            return True
        return False

    def y_intersection(self, get_complex=False):
        center_x_eval = self.center_x.try_evaluate()
        center_y_eval = self.center_y.try_evaluate()
        radius_eval = self._radius.try_evaluate()
        if None not in (center_x_eval, radius_eval):
            # If those are numbers, we will be able to simplify the root
            # The equation is : +-sqrt(r**2 - a**2)
            if abs(center_x_eval) > abs(radius_eval):  # Then the inside of the root will be negative
                if get_complex:
                    warnings.warn("Solving the intersections with complex numbers is still experimental..."
                                  "The issue will be resolved in later versions. Sorry!")
                    val = cmath.sqrt(radius_eval ** 2 - center_x_eval ** 2)
                    if center_y_eval is not None:
                        y1, y2 = val + center_y_eval, -val + center_y_eval
                    else:
                        y1, y2 = val + self.center_y, -val + self.center_y  # TODO: create a way to represent these...
                    return Point((0, y1)), Point((0, y2))  # TODO: return a complex point object instead
                return None
            else:  # Then we will find real solutions !
                val = sqrt(radius_eval ** 2 - center_x_eval ** 2)
                if val == 0:
                    if center_y_eval is not None:
                        return Point((0, center_y_eval))
                    else:
                        return Point((0, self.center_y))
                else:
                    if center_y_eval is not None:
                        y1, y2 = val + center_y_eval, -val + center_y_eval
                    else:
                        y1, y2 = val + self.center_y, -val + self.center_y
                    return Point((0, y1)), Point((0, y2))
        else:  # TODO: finish this part
            my_root = f"sqrt({_format_minus(self._radius, 0)} - {_format_minus(self.center_x, 0)})"

    def assign(self, **kwargs):
        self._radius.assign(**kwargs)
        self._center.coordinates[0].assign(**kwargs)
        self._center.coordinates[1].assign(**kwargs)

    def when(self, **kwargs):
        copy_of_self = self.__copy__()
        copy_of_self.assign(**kwargs)
        return copy_of_self

    def __copy__(self):
        return Circle(radius=self._radius.__copy__(), center=self.center.__copy__())

    def __call__(self, x: Union[int, float, IExpression], **kwargs):
        pass

    def __repr__(self):
        return f"Circle(radius={self._radius}, center={self._center})"

    def __str__(self):  # Get the equation or the repr ?
        return f"Circle(radius={self._radius}, center={self._center})"


class PointCollection:
    def __init__(self, points: Iterable = ()):
        self._points = []
        for point in points:
            if isinstance(point, (Iterable, int, float)) and not isinstance(point, Point):
                if isinstance(point, Iterable):
                    num_of_coordinates = len(point)
                    if num_of_coordinates == 2:
                        point = Point2D(point[0], point[1])
                    elif num_of_coordinates == 3:
                        point = Point3D(point[0], point[1], point[2])
                    else:
                        point = Point(point)

            if isinstance(point, Point):
                self._points.append(point)
            else:
                raise TypeError(f"encountered invalid type '{type(point)}' of value {point} when creating a "
                                f"PointCollection object. ")

    @property
    def points(self):
        return self._points

    def coords_at(self, index: int):
        """ returns all of the coordinates of the points in the specified index. For example, for an index of 0,
         a list of all of the x coordinates will be returned.
        """
        try:
            return [point.coordinates[index] for point in self._points]
        except IndexError:
            raise IndexError(f"The PointCollection object doesn't have points with coordinates of index {index}")

    def add_point(self, point: Point):
        self._points.append(point)

    def remove_point(self, index: int):
        del self._points[index]

    def max_coord_at(self, index: int):
        """
        Fetch the biggest coordinate at the specified index. For example, for the index of 0, the biggest
        x value will be returned.
        """
        try:
            return max(self.coords_at(index))
        except IndexError:
            raise IndexError(f"The PointCollection object doesn't have points with coordinates of index {index}")

    def min_coord_at(self, index: int):
        """
        Fetch the smallest coordinate at the specified index. For example, for the index of 0, the smallest
        x value will be returned.
        """
        try:
            return min(self.coords_at(index))
        except IndexError:
            raise IndexError(f"The PointCollection object doesn't have points with coordinates of index {index}")

    def avg_coord_at(self, index: int):
        """Returns the average value for the points' coordinates at the specified index. For example,
        for an index of 0, the average x value of the points will be returned, for an index of 1, the average y value
        of all of the dots will be returned, and so on.
        """
        try:
            coords = self.coords_at(index)
            return sum(coords) / len(coords)
        except IndexError:
            raise IndexError(f"The PointCollection object doesn't have points with coordinates of index {index}")

    def longest_distance(self, get_points=False):
        """ Gets the longest distance between two dots in the collection"""  # TODO: improve with combinations
        if len(self._points) <= 1:
            return 0
        pairs = combinations(self._points, 2)
        p1, p2 = max(pairs, key=lambda p1, p2: sum(
            (coord1 - coord2) ** 2 for (coord1, coord2) in zip(p1.coordinates, p2.coordinates)))
        max_distance = sqrt(sum((coord1 - coord2) ** 2 for (coord1, coord2) in zip(p1.coordinates, p2.coordinates)))
        if get_points:
            return max_distance, (p1, p2)
        return max_distance

    def shortest_distance(self, get_points=False):
        """ Gets the shortest distance between two dots in the collection"""
        if len(self._points) <= 1:
            return 0
        pairs = combinations(self._points, 2)
        p1, p2 = min(pairs, key=lambda p1, p2: sum(
            (coord1 - coord2) ** 2 for (coord1, coord2) in zip(p1.coordinates, p2.coordinates)))
        min_distance = sqrt(sum((coord1 - coord2) ** 2 for (coord1, coord2) in zip(p1.coordinates, p2.coordinates)))
        if get_points:
            return min_distance, (p1, p2)
        return min_distance

    def scatter(self, show=True):  # Add limits of x and y
        dimensions = len(self._points[0].coordinates)
        x_values = self.coords_at(0)
        if dimensions == 1:
            min_value, max_value = min(x_values), max(x_values)
            plt.hlines(0, min_value, max_value)  # Draw a horizontal line
            plt.xlim(0.8 * min_value, 1.2 * max_value)
            plt.ylim(-1, 1)
            y = np.zeros(len(self._points))  # Make all y values the same
            plt.plot(x_values, y, '|', ms=70)  # Plot a line at each location specified in a
            plt.axis('on')
        elif dimensions == 2:
            y_values = self.coords_at(1)
            plt.scatter(x_values, y_values)
        elif dimensions == 3:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.coords_at(0), self.coords_at(1), self.coords_at(2))
        elif dimensions == 4:  # Use a heat map in order to represent 4D dots ( The color is the 4th dimension )
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x, y, z, c = self.coords_at(0), self.coords_at(1), self.coords_at(2), self.coords_at(3)
            img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
            fig.colorbar(img)
        else:
            raise ValueError(f"Can only scatter in 1D, 2D, 3D and 4D dimension, But got {dimensions}D")
        if show:
            plt.show()

    def __str__(self):
        return ", ".join((point.__str__() for point in self._points))

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):
        if self.__index == len(self._points):
            raise StopIteration
        index = self.__index
        self.__index += 1
        return self._points[index]

    def __repr__(self):
        return f"PointCollection(({self.__str__()}))"


class Point1DCollection(PointCollection):
    def __init__(self, points: Iterable = ()):
        points_list = list(points)
        if any(len(point) != 1 for point in points_list):
            raise ValueError("All points must have 1 coordinates.")
        super(Point1DCollection, self).__init__(points_list)


class Point2DCollection(PointCollection):
    def __init__(self, points: Iterable = ()):
        points_list = list(points)
        if any(len(point) != 2 for point in points_list):
            raise ValueError("All points must have 2 coordinates.")
        super(Point2DCollection, self).__init__(points_list)

    def plot_regression(self, show=True):
        if len(self._points[0].coordinates) == 2:
            linear_function = self.linear_regression()  # a lambda expression is returned
            min_x, max_x = self.min_coord_at(0) - 50, self.max_coord_at(0) + 50
            plt.plot((min_x, max_x), (linear_function(min_x), linear_function(max_x)))
            if show:
                plt.show()

    @property
    def x_values(self):
        return self.coords_at(0)

    @property
    def y_values(self):
        return self.coords_at(1)

    def add_point(self, point: Point2D):
        if len(point) != 2:
            raise ValueError("Can only accept 2D points to Point2DCollection")
        self._points.append(point)

    def sum(self):
        return Point2D(sum(self.x_values), sum(self.y_values))

    def scatter(self, show=True):
        x_values, y_values = self.coords_at(0), self.coords_at(1)
        plt.scatter(x_values, y_values)
        if show:
            plt.show()

    def scatter_with_regression(self, show=True):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('equal')
        # And a corresponding grid
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        ax.spines['bottom'].set_position('zero')
        ax.spines['left'].set_position('zero')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self.scatter(show=show)
        self.plot_regression(show=show)

    def linear_regression(self, get_tuple=False):
        if len(self._points[0].coordinates) == 2:
            return linear_regression([point.coordinates[0] for point in self._points],
                                     [point.coordinates[1] for point in
                                      self._points], get_tuple)


class Point3DCollection(PointCollection):
    def __init__(self, points: Iterable = ()):
        points_list = list(points)
        if any(len(point) != 3 for point in points_list):
            raise ValueError("All points must have 3 coordinates.")
        super(Point3DCollection, self).__init__(points_list)

    @property
    def x_values(self):
        return self.coords_at(0)

    @property
    def y_values(self):
        return self.coords_at(1)

    @property
    def z_values(self):
        return self.coords_at(2)

    def sum(self):
        return Point3D(sum(self.x_values), sum(self.y_values), sum(self.z_values))

    def scatter(self, show=True):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x_values, self.y_values, self.z_values)
        if show:
            plt.show()

    def add_point(self, point: Point3D):
        if len(point) != 3:
            raise ValueError("Can only accept 3D points to Point3DCollection")
        self._points.append(point)


class Point4DCollection(PointCollection):
    def __init__(self, points: Iterable = ()):
        points_list = list(points)
        if any(len(point) != 4 for point in points_list):
            raise ValueError("All points must have 4 coordinates.")
        super(Point4DCollection, self).__init__(points_list)

    @property
    def x_values(self):
        return self.coords_at(0)

    @property
    def y_values(self):
        return self.coords_at(1)

    @property
    def z_values(self):
        return self.coords_at(2)

    @property
    def c_values(self):
        return self.coords_at(3)

    def sum(self):
        return Point4D(sum(self.x_values), sum(self.y_values), sum(self.z_values), sum(self.c_values))

    def scatter(self, show=True):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z, c = self.coords_at(0), self.coords_at(1), self.coords_at(2), self.coords_at(3)
        img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
        fig.colorbar(img)
        if show:
            plt.show()





def _get_limits_vectors_2d(vectors):
    """Internal method: find the edge values for the scope of the 2d frame"""
    min_x = min(min(vector.start_coordinate[0], vector.end_coordinate[0]) for vector in vectors) * 1.05
    max_x = max(max(vector.start_coordinate[0], vector.end_coordinate[0]) for vector in vectors) * 1.05
    min_y = min(min(vector.start_coordinate[1], vector.end_coordinate[1]) for vector in vectors) * 1.05
    max_y = max(max(vector.start_coordinate[1], vector.end_coordinate[1]) for vector in vectors) * 1.05
    return min_x, max_x, min_y, max_y


def _get_limits_vectors_3d(vectors):
    """Internal method: find the edge values for the scope of the 3d frame"""
    min_x = min(min(vector.start_coordinate[0], vector.end_coordinate[0]) for vector in vectors)
    max_x = max(max(vector.start_coordinate[0], vector.end_coordinate[0]) for vector in vectors)
    min_y = min(min(vector.start_coordinate[1], vector.end_coordinate[1]) for vector in vectors)
    max_y = max(max(vector.start_coordinate[1], vector.end_coordinate[1]) for vector in vectors)
    min_z = min(min(vector.start_coordinate[2], vector.end_coordinate[2]) for vector in vectors)
    max_z = max(max(vector.start_coordinate[2], vector.end_coordinate[2]) for vector in vectors)
    return min_x, max_x, min_y, max_y, min_z, max_z

def column(matrix, index: int):
    """
    Fetches a column in a matrix

    :param matrix: the matrix from which we fetch the column
    :param index: the index of the column. From 0 to the number of num_of_columns minus 1.
    :return: Returns a list of numbers, that represents the column in the given index
    :raise: Raises index error if the index isn't valid.
    """
    return [row[index] for row in matrix]



def main():
    """ main  method """
    pass

if __name__ == '__main__':
    main()
