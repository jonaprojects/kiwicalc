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
TRIGONOMETRY_CONSTANTS = {
    'sin': lambda x: sin(x),
    'cos': lambda x: cos(x),
    'tan': lambda x: tan(x),
    'cot': lambda x: 1 / tan(x),
    'sec': lambda x: 1 / cos(x),
    'csc': lambda x: 1 / sin(x),
    'cosec': lambda x: 1 / sin(x),
    'asin': lambda x: asin(x),
    'acos': lambda x: acos(x),
    'atan': lambda x: atan(x),
    'sinh': lambda x: sinh(x),
    'cosh': lambda x: cosh(x),
    'tanh': lambda x: tanh(x),
    'asinh': lambda x: asinh(x),
    'acosh': lambda x: acosh(x),
    'atanh': lambda x: atanh(x)
}
MATHEMATICAL_CONSTANTS = {
    'e': e,
    'pi': pi,
    'tau': tau,
    'log': lambda x, base: log(x=x, base=base),
    'log2': lambda x: log2(x),
    'log10': lambda x: log10(x),
    'ln': lambda x: log(x, e),
    'exp': lambda x: exp(x),
    'w': lambda x: NotImplemented,
    '&#8730;': lambda x: sqrt(x),
    'sqrt': lambda x: sqrt(x),
    'erf': lambda x: erf(x),
    'erfc': lambda x: erfc(x),
    'gamma': lambda x: gamma(x),
    'lgamma': lambda x: lgamma(x),

    'lambert': lambda x: NotImplemented
}


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
allowed_characters = list(string.ascii_lowercase)
allowed_characters.remove('e')
allowed_characters.remove('i')


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


def solve_quadratic_from_str(expression, real=False, strict_syntax=False):  # TODO: fix this !!!!
    if isinstance(expression, str):
        variables = get_equation_variables(expression)
        if len(variables) == 0:
            return tuple()
        elif len(variables) == 1:
            variable = variables[0]
            parsed_dict = ParseEquation.parse_quadratic(expression, variables, strict_syntax=strict_syntax)
            solve_method = solve_quadratic_real if real else solve_quadratic
            return solve_method(parsed_dict[variable][0], parsed_dict[variable][0], parsed_dict['free'])
        else:
            raise ValueError("Can't solve a quadratic equation with more than 1 variable")


def solve_quadratic(a: Union[str, float], b: float = None, c: float = None) -> tuple:
    """ Solves a quadratic equation using computations of complex numbers ( utilizing the cmath library)"""
    if isinstance(a, str):
        return solve_quadratic_from_str(a)
    discriminant = b ** 2 - 4 * a * c
    return (-b + cmath.sqrt(discriminant)) / (2 * a), (-b - cmath.sqrt(discriminant)) / (2 * a)


def solve_quadratic_real(a: Union[str, float], b: float, c: float) -> Optional[Union[Tuple[float, float], float]]:
    """returns onlu the real solutions of the quadratic equation"""
    if isinstance(a, str):
        return solve_quadratic_from_str(a, real=True)
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return None
    if discriminant == 0:
        return (-b + sqrt(discriminant)) / (2 * a)
    return (-b + sqrt(discriminant)) / (2 * a), (-b + sqrt(discriminant) / (2 * a))


def solve_quadratic_params(a: "Union[IExpression, int, float,str]", b: "Union[IExpression, int, float]"
                           , c: "Union[IExpression,int,float]"):
    if isinstance(a, str):  # TODO: implement string analysis here
        print("need to be implemented")
    if all(isinstance(coefficient, (int, float)) for coefficient in (a, b, c)):
        return solve_quadratic(a, b, c)
    if isinstance(a, IExpression):
        a_eval = a.try_evaluate()
        if a_eval is not None:
            a = a_eval
    if isinstance(b, IExpression):
        b_eval = b.try_evaluate()
        if b_eval is not None:
            b = b_eval
    if isinstance(c, IExpression):
        c_eval = c.try_evaluate()
        if c_eval is not None:
            c = c_eval
    if all(isinstance(coefficient, (int, float)) for coefficient in (a, b, c)):
        return (-b + sqrt(b ** 2 - 4 * a * c)) / (2 * a), (-b - sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    else:
        return (-b + Sqrt(b ** 2 - 4 * a * c)) / (2 * a), (-b - Sqrt(b ** 2 - 4 * a * c)) / (2 * a)


def decimal_range(start: float, stop: float, step: float = 1):
    while start <= stop:
        yield start
        start += step


def extract_coefficient(coefficient: str) -> float:
    """[method for inside use]"""
    return -1 if coefficient == '-' else 1 if coefficient in ('+', '') else float(coefficient)


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


def copy_expression(expression):
    if isinstance(expression, IExpression) or hasattr(expression, "__copy__"):
        return expression.__copy__()

    if isinstance(expression, (list, set)) or hasattr(expression, "copy"):
        return expression.copy()

    return expression


def apply_parenthesis(given_string: str, delimiters=('+', '-', '*', '**')):
    """put parenthesis on expressions such as x+5, 3*x , etc - if needed."""
    if any(character in delimiters for character in given_string):
        return f"({given_string})"
    return given_string


def handle_parenthesis(expression: str):  # TODO: later replace with the code inside to_lambda
    """
    [INSIDE USE][Needs to be fixed and reviewed] This method processes the appearances of parenthesis in expressions.

    :param expression:
    :return:
    """
    new_expression = ""
    for character_index in range(len(expression)):
        # Handling cases such as y(3x+6) -> y*(3x+6)
        if expression[character_index] == '(' and character_index > 0 and expression[character_index - 1] not in (
                '+', '-', '*', '/', '%'):
            new_expression += '*'
        new_expression += expression[character_index]
        # Handling cases such as (3+x)y -> (3+x)*y
        if expression[character_index] == ')' and character_index < len(expression) - 1 and expression[
            character_index + 1] not in ('+', '-', '*', '/', '%', '!'):
            new_expression += '*'
    return new_expression


def formatted_expression(expression: str, variables, constants=(), format_abs=False, format_factorial=False):
    """
    Formats an expression
    For example: The string "3x^2 + 5x + 6" -> "3*x**2+5*x+6"

    :param expression: The expression entered
    :param variables: The variables_dict appearing in the expression
    :param constants: Constants that appear in the expression

    :return: A new string, with proper pythonic algebraic syntax
    """
    if format_abs:
        expression = handle_abs(expression)

    if format_factorial:
        expression = handle_factorial(expression)

    expressions = split_expression(expression.replace("^", "**"))  # Handling absolute value notations
    # and splitting the expression into sub-expressions.
    modified_variables = list(variables) + list(constants)
    for index, expression in enumerate(expressions):
        new_expression = ""
        occurrences = []
        for variable in modified_variables:
            occurrences += [m.start() for m in re.finditer(variable, expression)]
        for character_index in range(len(expression)):
            # Handling cases such as y(3x+6) -> y*(3x+6)
            """ if expression[character_index] == '(' and character_index > 0 and expression[character_index - 1] not in (
                    '+', '-', '*', '/', '%'):
                new_expression += '*'
            """
            new_expression += expression[character_index]
            # Handling cases such as (3+x)y -> (3+x)*y
            """if expression[character_index] == ')' and character_index < len(expression) - 1 and expression[
                character_index + 1] not in ('+', '-', '*', '/', '%', '!'):
                new_expression += '*'"""
            if character_index + 1 in occurrences and (expression[character_index].isdigit() or expression[
                character_index].isalpha()) and expression[character_index] + expression[
                character_index + 1] not in modified_variables:
                new_expression += '*'
        expressions[index] = new_expression
    return "".join(expressions)


def to_lambda(expression: str, variables, constants=(), format_abs=False, format_factorial=False):
    """
    Generate an executable lambda expression from a string

    :param expression: The string that represents the expression ( focuses on processing algebraic expressions)
    :param variables: The variables_dict of the expression.
    :param constants: Constants or methods to be used in the expression.
    :param format_abs:  Whether to check for absolute values and process them.
    :param format_factorial: Whether to check for factorials and process them.
    :return: Returns a lambda expression corresponding to the expression given.
    """
    modified_expression = formatted_expression(expression, variables, constants, format_abs=format_abs,
                                               format_factorial=format_factorial)
    # print(f'lambda {",".join(variables_dict)}:{modified_expression}')
    return eval(f'lambda {",".join(variables)}:{modified_expression}')


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
        expression = clean_from_spaces(expression)
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
        expression = clean_from_spaces(expression)
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


def round_decimal(number: float):
    """
    Rounds a decimal number, or at least tries to.... Since python is weird with it
    :param number: ugly number we wish to round
    :return: less ugly number
    """
    if number - floor(number) < 0.000001:
        return floor(number)
    elif abs(number - ceil(number)) < 0.000001:
        return ceil(number)
    return round(number, 5)


def create_grid():
    """ Create a grid in matplotlib"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('equal')
    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    return fig, ax


def draw_axis(ax):
    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create 'x' and 'y' labels placed at the end of the axes


def scatter_dots(x_values, y_values, title: str = "", ymin: float = -10, ymax: float = 10, color=None,
                 show_axis=True, show=True, fig=None, ax=None):  # change start and end to be automatic?
    if (length := len(x_values)) != (y_length := len(y_values)):
        raise ValueError(f"You must enter an equal number of x and y values. Got {length} x values and "
                         f"{y_length} y values.")
    if None in (fig, ax):
        fig, ax = create_grid()
    if show_axis:
        draw_axis(ax)
    plt.title(title, fontsize=14)
    plt.ylim(ymin, ymax)
    plt.scatter(x=x_values, y=y_values, s=90, c=color)
    if show:
        plt.show()


def scatter_dots_3d(x_values, y_values, z_values, title: str = "", xlabel: str = "X Values",
                    ylabel: str = "Y Values", zlabel: str = "Z Values", fig=None,
                    ax=None, show=True, write_labels=True):
    if None in (fig, ax):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    if title:
        plt.title(title)
    ax.scatter(x_values, y_values, z_values)

    if write_labels:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

    if show:
        plt.show()


def scatter_function(func: Union[Callable, str], start: float = -10, stop: float = 10,
                     step: float = 0.5, ymin: float = -10, ymax: float = 10, title="", color=None,
                     show_axis=True, show=True, fig=None, ax=None, values=None):
    if isinstance(func, str):
        func = Function(func)
    if values is not None:
        results = [func(value) for value in values]
    else:
        values, results = values_in_range(func, start, stop, step)
    scatter_dots(values, results, title=title, ymin=ymin, ymax=ymax, color=color, show_axis=show_axis, show=show,
                 fig=fig, ax=ax)


def scatter_function_3d(func: "Union[Callable, str, IExpression]", start: float = -3, stop: float = 3,
                        step: float = 0.3,
                        xlabel: str = "X Values",
                        ylabel: str = "Y Values", zlabel: str = "Z Values", show=True, fig=None, ax=None,
                        write_labels=True, meshgrid=None, title=""):
    if isinstance(func, str):
        func = Function(func)

    if meshgrid is None:
        x = y = np.arange(start, stop, step)
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = meshgrid
    zs = np.array([])
    for x_value, y_value in zip(np.ravel(X), np.ravel(Y)):
        try:
            zs = np.append(zs, func(x_value, y_value))
        except:
            zs = np.append(zs, np.nan)
    Z = zs.reshape(X.shape)
    scatter_dots_3d(
        X, Y, Z, fig=fig, ax=ax, title=title, show=show, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
        write_labels=write_labels)


def scatter_functions_3d(functions: "Iterable[Union[Callable, str, IExpression]]", start: float = -5, stop: float = 5,
                         step: float = 0.1,
                         xlabel: str = "X Values",
                         ylabel: str = "Y Values", zlabel: str = "Z Values"):
    pass


def process_to_points(func: Union[Callable, str], start: float = -10, stop: float = 10,
                      step: float = 0.01, ymin: float = -10, ymax: float = 10, values=None):
    if isinstance(func, str):
        func = Function(func)
    if values is None:
        values = list(decimal_range(start, stop, step)) if values is None else values
    results = []
    for index, value in enumerate(values):
        try:
            current_result = func(value)
            results.append(current_result)
        except ValueError:
            results.append(None)

    return values, results


def plot_function(func: Union[Callable, str], start: float = -10, stop: float = 10,
                  step: float = 0.01, ymin: float = -10, ymax: float = 10, title=None, show_axis=True, show=True,
                  fig=None, ax=None, formatText=False, values=None):
    # Create the setup of axis and stuff

    if None in (fig, ax):  # If at least one of those parameters is None, then create a new ones
        fig, ax = create_grid()
    if show_axis:
        draw_axis(ax)
    values, results = process_to_points(func, start, stop, step, ymin, ymax, values)
    if title is not None:
        if formatText:
            plt.title(fr"${format_matplot(title)}$", fontsize=14)
        else:
            plt.title(fr"{title}", fontsize=14)
    plt.ylim(ymin, ymax)
    plt.plot(values, results)

    if show:
        plt.show()


def plot_function_3d(given_function: "Union[Callable, str, IExpression]", start: float = -3, stop: float = 3,
                     step: float = 0.3,
                     xlabel: str = "X Values",
                     ylabel: str = "Y Values", zlabel: str = "Z Values", show=True, fig=None, ax=None,
                     write_labels=True, meshgrid=None):
    if step < 0.1:
        step = 0.3
        warnings.warn("step parameter modified to 0.3 to avoid lag when plotting in 3D")
    if isinstance(given_function, str):
        given_function = Function(given_function)
    elif isinstance(given_function, IExpression):
        num_of_variables = len(given_function.variables)
        if num_of_variables != 2:
            raise ValueError(f"Invalid expression: {given_function}. Found {num_of_variables} variables, expected 2.")
        if hasattr(given_function, "to_lambda"):
            given_function = given_function.to_lambda()
        elif hasattr(given_function, "__call__"):
            pass
        else:
            raise ValueError(f"This type of algebraic expression isn't supported for plotting in 3D!")

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')
    if meshgrid is None:
        x = y = np.arange(start, stop, step)
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = meshgrid

    zs = np.array([])
    for x_value, y_value in zip(np.ravel(X), np.ravel(Y)):
        try:
            result = given_function(x_value, y_value)
            if result is None:
                result = np.nan
            zs = np.append(zs, result)
        except ValueError:
            zs = np.append(zs, np.nan)
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z)
    if write_labels:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
    if show:
        plt.show()


def plot_functions_3d(functions: "Iterable[Union[Callable, str, IExpression]]", start: float = -5, stop: float = 5,
                      step: float = 0.1,
                      xlabel: str = "X Values",
                      ylabel: str = "Y Values", zlabel: str = "Z Values"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(start, stop, step)
    for func in functions:
        plot_function_3d(func, show=False, write_labels=False, fig=fig, ax=ax, meshgrid=np.meshgrid(x, y))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()


def plot_functions(functions, start: float = -10, stop: float = 10, step: float = 0.01, ymin: float = -10,
                   ymax: float = 10, title: str = None, formatText: bool = False,
                   show_axis: bool = True, show: bool = True, with_legend=True):
    fig, ax = create_grid()
    if show_axis:
        draw_axis(ax)
    values = np.arange(start, stop, step)
    plt.ylim(ymin, ymax)
    if title is not None:
        if formatText:
            plt.title(fr"${format_matplot(title)}$", fontsize=14)
        else:
            plt.title(title, fontsize=14)
    for given_function in functions:
        if isinstance(given_function, str):
            label = given_function
            given_function = Function(given_function).lambda_expression
        elif isinstance(given_function, Function):
            label = given_function.function_string
            given_function = given_function.lambda_expression
        elif isinstance(given_function, IExpression):
            label = given_function.__str__()
            if hasattr(given_function, "to_lambda"):
                given_function = given_function.to_lambda()
            else:
                raise ValueError(f"Invalid algebraic expression for plotting: {given_function}")
        else:
            label = None
        plt.plot(values, [given_function(value) for value in values], label=label)
    if with_legend:
        plt.legend()
    if show:
        plt.show()


def scatter_functions(functions, start: float = -10, stop: float = 10, step: float = 0.5, ymin: float = -10,
                      ymax: float = 10, title: str = None,
                      show_axis: bool = True, show: bool = True):
    fig, ax = create_grid()
    cycol = cycle('bgrcmykw')
    values = np.arange(start, stop, step)
    for index, current_function in enumerate(functions):
        scatter_function(func=current_function, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title,
                         color=next(cycol), show_axis=True, show=False, fig=fig, ax=ax, values=values)
    plt.show()


def plot_vector_2d(x_start: float, y_start: float, x_distance: float, y_distance: float, show=True, fig=None, ax=None):
    if None in (fig, ax):
        fig, ax = plt.subplots(figsize=(10, 8))
    ax.arrow(x_start, y_start, x_distance, y_distance,
             head_width=0.1,
             width=0.01)
    if show:
        plt.show()


def plot_vector_3d(starts: Tuple[float, float, float], distances: Tuple[float, float, float], arrow_length_ratio=0.08,
                   show=True, fig=None, ax=None):
    """plot a 3d vector"""
    u, v, w = distances
    start_x, start_y, start_z = starts

    if (fig, ax) == (None, None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([start_x, (start_x + u)])
        ax.set_ylim([start_y, (start_y + v)])
        ax.set_zlim([start_z, (start_z + w)])

    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = fig.add_subplot(111, projection='3d')

    ax.quiver(start_x, start_y, start_z, u, v, w, arrow_length_ratio=arrow_length_ratio)
    if show:
        plt.show()


def plot_complex(*numbers: complex, title: str = "", show=True):
    """
    plot complex numbers on the complex plane

    :param numbers: The complex numbers to be plotted
    :param show: If set to false, the plotted
    :return: fig, ax
    """
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_title(title, va='bottom')
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    plt.title(title)
    max_radius = abs(numbers[0])
    for c in numbers:
        radius = abs(c)
        if radius > max_radius:
            max_radius = radius
        ax.scatter(cmath.phase(c), radius)

    ax.set_rticks(np.linspace(0, int(max_radius) * 2, num=5))  # Less radial
    ax.set_rmax(max_radius * 1.25)
    if show:
        plt.show()
    return fig, ax


def generate_subplot_shape(num_of_functions: int):
    square_root = sqrt(num_of_functions)
    if square_root == int(square_root):  # if an integer square root is found, then we're over
        return int(square_root), int(square_root)
    # Then find the 2 biggest factors
    try:
        result = min([(first, second) for first, second in combinations(range(1, num_of_functions), 2) if
                      first * second == num_of_functions], key=lambda x: abs(x[1] - x[0]))
        if result[0] > result[1]:
            return result[1], result[0]
        return result
    except ValueError:
        return ceil(square_root), ceil(square_root)


def plot_multiple(funcs, shape: Tuple[int, int] = None, start: float = -10, stop: float = 10,
                  step: float = 0.01, ymin: float = -10, ymax: float = 10, title=None, show_axis=True, show=True,
                  values=None):
    num_of_functions = len(funcs)
    if shape is None:
        shape = generate_subplot_shape(num_of_functions)
    fig, ax = plt.subplots(shape[0], shape[1])

    fig.tight_layout()
    func_index = 0
    for i in range(shape[0]):
        if func_index >= num_of_functions:
            break
        for j in range(shape[1]):
            if func_index >= num_of_functions:
                break
            values, results = process_to_points(funcs[func_index], start, stop, step, ymin, ymax, values)
            current_ax = ax[i, j] if shape[0] > 1 else ax[j]
            current_ax.plot(values, results, label=funcs[func_index])
            current_ax.set_title(funcs[func_index])
            if show_axis:
                draw_axis(current_ax)
            func_index += 1

    if title is not None:
        plt.title(title)
    if show:
        try:  # try to plot these in full screen.
            wm = plt.get_current_fig_manager()
            wm.window.state('zoomed')
        except:
            warnings.warn("Couldn't plot in full screen!")
        plt.show()


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


class IPlottable(ABC):
    @abstractmethod
    def plot(self):
        pass


class IScatterable(ABC):
    @abstractmethod
    def scatter(self):
        pass


class IPlottable3D(ABC):
    @abstractmethod
    def plot3d(self):
        pass


class IScatterable3D(ABC):
    @abstractmethod
    def scatter3d(self):
        pass


class IExpression(ABC):

    @abstractmethod
    def assign(self, **kwargs):
        pass

    def when(self, **kwargs):
        copy_of_self = self.__copy__()
        copy_of_self.assign(**kwargs)
        return copy_of_self

    @abstractmethod
    def try_evaluate(self):
        pass

    @property
    @abstractmethod
    def variables(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __iadd__(self, other):
        pass

    def __add__(self, other: "Union[int, float, IExpression]"):
        return self.__copy__().__iadd__(other)

    def __radd__(self, other: "Union[int, float, IExpression]"):
        return self.__copy__().__iadd__(other)

    @abstractmethod
    def __isub__(self, other):
        pass

    def __sub__(self, other: "Union[int, float, IExpression]"):
        return self.__copy__().__isub__(other)

    @abstractmethod
    def __imul__(self, other):
        pass

    def __mul__(self, other: "Union[int, float, IExpression]"):
        return self.__copy__().__imul__(other)

    def __rmul__(self, other: "Union[int, float, IExpression]"):
        return self.__copy__().__imul__(other)

    @abstractmethod
    def __itruediv__(self, other):
        pass

    def __truediv__(self, other: "Union[int, float, IExpression]"):
        return self.__copy__().__itruediv__(other)

    @abstractmethod
    def __ipow__(self, other):
        pass

    def __pow__(self, other: "Union[int, float, IExpression]"):
        return self.__copy__().__ipow__(other)

    @abstractmethod
    def __neg__(self):
        pass

    @abstractmethod
    def __copy__(self):
        pass

    @abstractmethod
    def simplify(self):
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __ne__(self, other) -> bool:
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @staticmethod
    @abstractmethod
    def from_dict(given_dict: dict):
        pass

    def __abs__(self):
        return Abs(self)

    def __rpow__(self, other: "Union[IExpression, int, float]"):
        my_evaluation = self.try_evaluate()
        if my_evaluation is not None:
            return other ** my_evaluation
        return Exponent(other, self)

    def python_syntax(self, format_abs=True, format_factorial=True):
        return formatted_expression(self.__str__(), variables=self.variables, format_abs=format_abs,
                                    format_factorial=format_factorial)

    def to_lambda(self, variables=None, constants=tuple(), format_abs=True, format_factorial=True):
        if variables is None:
            variables = self.variables
        return to_lambda(self.python_syntax(), variables, constants, format_abs=format_abs,
                         format_factorial=format_factorial)

    def reinman(self, a: float, b: float, N: int):
        return reinman(self.to_lambda(), a, b, N)

    def trapz(self, a: float, b: float, N: int):
        return trapz(self.to_lambda(), a, b, N)

    def simpson(self, a: float, b: float, N: int):
        return simpson(self.to_lambda(), a, b, N)

    def secant(self, n_0: float, n_1: float, epsilon: float = 0.00001, nmax: int = 10_000):
        return secant_method(self.to_lambda(), n_0, n_1, epsilon, nmax)

    def bisection(self, a: float, b: float, epsilon: float = 0.00001, nmax=100000):
        return bisection_method(self.to_lambda(), a, b, epsilon, nmax)

    def plot(self, start: float = -6, stop: float = 6, step: float = 0.3, ymin: float = -10,
             ymax: float = 10, title: str = None, formatText: bool = False,
             show_axis: bool = True, show: bool = True, fig=None, ax=None, values=None, meshgrid=None):
        variables = self.variables
        num_of_variables = len(variables)
        if num_of_variables == 1:
            plot_function(self.to_lambda(),
                          start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title,
                          show_axis=show_axis, show=show, fig=fig, formatText=formatText, ax=ax,
                          values=values)
        elif num_of_variables == 2:
            plot_function_3d(given_function=self.to_lambda(), start=start, stop=stop, meshgrid=meshgrid)
        else:
            raise ValueError(f"Cannot plot an expression with {num_of_variables} variables")

    def scatter(self, start: float = -10, stop: float = 10,
                step: float = 0.05, ymin: float = -10, ymax: float = 10, title=None, show_axis=True, show=True,
                fig=None, ax=None, values=None):

        lambda_expression = self.to_lambda()
        num_of_variables = len(self.variables)
        if title is None:
            title = self.__str__()
        if num_of_variables == 0:  # TODO: plot this in a number axis
            raise ValueError("Cannot plot a polynomial with 0 variables_dict")
        elif num_of_variables == 1:
            scatter_function(lambda_expression, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax,
                             show_axis=show_axis, show=show, fig=fig, ax=ax, values=values, title=title)
        elif num_of_variables == 2:
            scatter_function_3d(lambda_expression, start=start, stop=stop, step=step,
                                title=title)  # TODO: update the parameters
        else:
            raise ValueError("Cannot plot a function with more than two variables_dict (As for this version)")

    def to_json(self):
        return json.dumps(self.to_dict())

    def export_json(self, path: str):
        with open(path, 'w') as json_file:
            json_file.write(self.to_json())

    def to_Function(self) -> "Optional[Function]":
        try:
            return Function(self.__str__())
        except:
            return None


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


class Mono(IExpression, IPlottable, IScatterable):
    __slots__ = ['_coefficient', '__variables']

    def __init__(self, coefficient: Union[str, int, float] = None, variables_dict: dict = None):
        if isinstance(coefficient, str):
            self._coefficient, self.__variables = mono_from_str(coefficient, get_tuple=True)
        else:
            if isinstance(coefficient, (int, float)):
                self._coefficient = coefficient
                self.__variables = None if variables_dict is None else variables_dict.copy()
            else:
                raise TypeError(f"Invalid type '{type(coefficient)}' for the coefficient of Mono object")

    @property
    def variables_dict(self) -> Optional[dict]:
        return self.__variables

    @variables_dict.setter
    def variables_dict(self, variables: dict):
        self.__variables = variables

    @property
    def variables(self):
        if self.__variables is None:
            return set()
        return set(self.__variables.keys())

    @property
    def coefficient(self):
        return self._coefficient

    @property
    def num_of_variables(self):
        return len(self.__variables)

    @coefficient.setter
    def coefficient(self, new_coefficient: float):
        self._coefficient = new_coefficient
        if new_coefficient == 0:
            self.__variables = None

    def highest_power(self) -> Union[int, float]:
        if self.__variables is None:
            return 0
        return max(self.__variables.values())

    def __iadd__(self, other: Union[int, float, str, IExpression]):
        if other == 0:
            return self
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            if my_evaluation is not None:
                self._coefficient += other
                return self
            return Poly(expressions=(self, Mono(other)))
        elif isinstance(other, str):
            other = Poly(other)

        if isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if None not in (other_evaluation, my_evaluation):
                self._coefficient = my_evaluation + other_evaluation
                self.__variables = None
                return self
            elif other_evaluation is not None:
                if other_evaluation == 0:  # Adding zero doesn't change anything.
                    return self
                return Poly(expressions=(self, Mono(other_evaluation)))
            if isinstance(other, Mono):
                if other._coefficient == 0:
                    return self
                if not self.__variables and not other.__variables:  # Either they are two free numbers
                    self._coefficient += other._coefficient
                    return self
                if self.__variables == other.__variables:  # Or they have the same variables_dict
                    # Has the same __variables and powers !
                    self._coefficient += other._coefficient
                    if self._coefficient == 0:
                        self.__variables = None
                        return self
                    if self.__variables is not None and other.__variables is not None:  # Or they have different variables_dict
                        self.__variables = {**self.__variables, **other.__variables}
                    else:
                        first_variables = self.__variables if self.__variables is not None else {}  # Or one of them is a number
                        second_variables = other.__variables if other.__variables is not None else {}
                        self.__variables = {**first_variables, **second_variables}

                    return self
                    # All remain unchanged, except the _coefficient which are summed together
                else:
                    return Poly([self, other])
            elif isinstance(other, Poly):
                return other.__add__(self)
            return ExpressionSum((self, other))
        else:
            raise TypeError(
                f"Mono.__add__: invalid type {type(other)}. Expected Mono,Poly, str,int,float."
            )

    def __isub__(self, other: Union[int, float, str, IExpression]):
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            if my_evaluation is not None:
                self._coefficient -= other
                self.simplify()
                return self
            else:
                return Poly((self, Mono(-other)))

        elif isinstance(other, str):
            other = Poly(other)

        if isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                self._coefficient = my_evaluation - other_evaluation
                self.__variables = None
                return self
            elif other_evaluation is not None:
                if other_evaluation == 0:  # subtracting 0 doesn't change anything
                    return self
                return Poly((self, Mono(-other_evaluation)))
            if isinstance(other, Mono):
                if other._coefficient == 0:
                    return self
                if not self.__variables and not other.__variables:  # Either they are two free numbers
                    self._coefficient -= other._coefficient
                    return self
                if self.__variables == other.__variables:  # Or they have the same variables_dict
                    # Has the same __variables and powers !
                    self._coefficient -= other._coefficient
                    if self._coefficient == 0:
                        self.__variables = None
                        return self
                    if self.__variables is not None and other.__variables is not None:  # Or they have different variables_dict
                        self.__variables = {**self.__variables, **other.__variables}
                    else:
                        first_variables = self.__variables if self.__variables is not None else {}  # Or one of them is a number
                        second_variables = other.__variables if other.__variables is not None else {}
                        self.__variables = {**first_variables, **second_variables}
                    self.simplify()
                    return self
                    # All remain unchanged, except the coefficients which are summed together
                else:
                    return Poly([self, -other])
            elif isinstance(other, Poly):
                return other.__neg__().__iadd__(self)
            else:
                return ExpressionSum((self, -other))

        else:
            raise TypeError(
                f"Mono.__add__: invalid type {type(other)}. Expected Mono,Poly, str,int,float."
            )

    def __sub__(self, other: Union[int, float, str, IExpression]) -> "Union[Mono,Poly]":
        return self.__copy__().__isub__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __imul__(self, other: Union[int, float, str, IExpression]):
        if other == 0:
            self.__variables = None
        if isinstance(other, (int, float)):
            self._coefficient *= other
            return self
        elif isinstance(other, str):
            other = Poly(other)
        # If it's not int, float or str, then it must be of type IExpression, due to the type limitations
        if isinstance(other, Mono):
            my_counter = Counter(self.__variables)
            my_counter.update(other.__variables)
            self.__variables = dict(my_counter)
            self.simplify()
            # Filter expressions such as x^0
            if self.__variables == {}:  # counter-measure so that all numbers will have None and not {} as variables_dict.
                self.__variables = None
            self._coefficient *= other._coefficient
            return self
        elif isinstance(other, Poly):
            new_expressions = []
            for poly_expression in other.expressions:
                multiply_result = self * poly_expression
                found = False
                for index, new_expression in enumerate(new_expressions):
                    if new_expression.__variables == multiply_result.__variables:
                        addition_result = new_expression + multiply_result
                        if addition_result._coefficient == 0:
                            del new_expressions[index]
                        else:
                            new_expressions[index] = addition_result
                        found = True
                        break
                if not found:
                    new_expressions.append(multiply_result)
            return Poly(new_expressions)
        else:
            evaluated_other = other.try_evaluate()
            if evaluated_other is not None:  # if the other expression can be evaluated into float or int
                self._coefficient *= evaluated_other
                return self
            if isinstance(other, (ExpressionMul, ExpressionSum)):
                for expression in other.expressions:
                    result = self.__imul__(expression)
                return result
            return other * self

    def __mul__(self, other: Union[IExpression, int, float, str], show_steps=False):
        return self.__copy__().__imul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def multiply_by_number(self, number: Union[int, float]):
        self._coefficient *= number

    def divide_by_number(self, number: Union[int, float]):
        if number == 0:
            raise ZeroDivisionError(f"Cannot divide a Mono object '{self.__str__()}' by 0")
        self._coefficient /= number

    def __itruediv__(self, other: Union[int, float, str, IExpression]):
        if other == 0:
            raise ZeroDivisionError("Can't divide by 0")

        if isinstance(other, (int, float)):
            if other == 0:
                raise ValueError("Cannot divide a Mono object by 0.")
            self._coefficient /= other
            return self

        elif isinstance(other, str):
            other = Mono(other)

        if isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if other_evaluation is not None:
                if other_evaluation == 0:
                    raise ValueError("Cannot divide a Mono object by 0")
                self._coefficient /= other_evaluation
                return self
            if isinstance(other, Poly):
                if len(other.expressions) != 1:
                    return PolyFraction(self, other)
                other = other.expressions[0]

            if isinstance(other, Mono):
                if self._coefficient == 0:
                    self.__variables = None
                    return self
                elif other._coefficient == 0:
                    raise ZeroDivisionError("Cannot divide by 0 !")
                elif other.__variables in ({}, None):  # Handling the case that the other number is a free number
                    self._coefficient /= other._coefficient
                    return self
                if self.num_of_variables < other.num_of_variables:
                    return PolyFraction(self, other)
                my_variables, other_variables = self.variables, other.variables
                if my_variables != other_variables:
                    return PolyFraction(self, other)
                if any(my_value < other_value for my_value, other_value in
                       zip(self.__variables.values(), self.__variables.values())):
                    # That means we should return a fraction - since Monomials don't support negative powers
                    return PolyFraction(self, other)
                my_keys = [] if self.__variables in (None, {}) else list(self.variables)
                keys = my_keys + list(other.variables)
                new_variables = dict.fromkeys(keys, 0)
                if self.__variables is not None:
                    for key, value in self.__variables.items():
                        new_variables[key] += value
                for key, value in other.__variables.items():
                    new_variables[key] -= value
                new_variables = {key: value for (key, value) in new_variables.items() if
                                 value != 0}  # filter zeroes from the result
                if new_variables == {}:
                    new_variables = None

                self._coefficient /= other._coefficient
                self.__variables = new_variables
                return self
            else:
                return Fraction(self, other)
        else:
            raise TypeError(f"Invalid type {type(other)} for dividing a Mono object.")

    def __ipow__(self, power: Union[int, float, IExpression]):
        if isinstance(power, IExpression):
            power_eval = power.try_evaluate()
            if power_eval is None:  # the algebraic expression couldn't be evaluated
                return Exponent(self, power)
            power = power_eval
        if power == 0:
            self._coefficient = 1
            self.__variables = None
            return self

        self._coefficient **= power
        if self.__variables not in (None, {}):
            self.__variables = {variable_name: variable_power * power for (variable_name, variable_power) in
                                self.__variables.items()}
        return self

    def __eq__(self, other: Union[int, float, str, IExpression]):
        if other is None:
            return False
        if isinstance(other, (int, float)):
            return self._coefficient == other and self.__variables is None
        if isinstance(other, str):
            other = Mono(other)
        if isinstance(other, IExpression):
            my_evaluation = self.try_evaluate()
            if my_evaluation is not None:
                other_evaluation = other.try_evaluate()
                if other_evaluation is not None:
                    return my_evaluation == other_evaluation
            if isinstance(other, Poly):
                if len(other.expressions) != 1:
                    return False
                first = other.expressions[0]
                return self._coefficient == first._coefficient and self.__variables == first.__variables
            if isinstance(other, (Mono, Var)):
                return self._coefficient == other._coefficient and self.__variables == other.__variables
        else:
            raise TypeError(f"Can't equate between types Mono and {type(other)}")

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):  # TODO: can be removed?
        """
        Implementation of the '<' operator for sorting purposes.
        :param other:
        :return:
        """
        if isinstance(other, (int, float)):
            if self.__variables is None:
                return self._coefficient < other
            else:
                other = Mono(other)
        if isinstance(other, Mono):
            if self.__variables is None:
                return other
            if other.__variables is None:
                return self
            biggest_power1, biggest_power2 = max(self.__variables.values()), max(other.__variables.values())
            if biggest_power2 < biggest_power1:
                return biggest_power2
            elif biggest_power1 < biggest_power2:
                return biggest_power1
            if len(self.__variables) < len(other.__variables):
                return self
            elif len(other.__variables) < len(self.__variables):
                return other
            if len(self.__variables) == len(other.__variables):
                if len(self.__variables) == 1:
                    first_variable, second_variable = fetch_variable(self.__variables), fetch_variable(
                        other.__variables)
                    return first_variable < second_variable
                else:
                    max_string1 = max(list(self.__variables.keys()))
                    max_string2 = max(list(self.__variables.keys()))
                    return self if max_string1 < max_string2 else other
        raise TypeError(f"Invalid type {type(other)} for __lt__(). Expected type Mono")

    def __str__(self):
        if self.__variables is not None:
            return f"{round_decimal(self._coefficient) if self._coefficient not in (-1, 1) else ('-' if self.coefficient == -1 else '')}" + "*".join(
                [(
                    f"{variable}" if power == 1 else f"{variable}^{round_decimal(power)}")
                    for variable, power in
                    self.__variables.items()])
        result = str(round_decimal(self._coefficient))
        return result

    def contains_variable(self, variable: str) -> bool:
        """Checking whether a given variable appears in the expression """
        if self.__variables in (None, {}):
            return False
        return variable in self.__variables  # Return whether the variable was found

    def is_number(self) -> bool:
        """Checks whether the Mono represents a free number-
        If it is a free number, True will be returned, otherwise - False
        """
        return self.__variables in ({}, None)  # If there no variables_dict, it's a free number !

    def latex(self):
        return f"{self._coefficient}*{'*'.join([(f'{variable}^{{{power}}}' if self._coefficient >= 0 else f'({variable}^{{{power}}})') for variable, power in self.__variables.items()])} "

    def to_dict(self):
        return {'type': 'Mono', 'coefficient': self._coefficient, 'variables_dict': self.__variables}

    @staticmethod
    def from_dict(parsed_dict: dict):
        return Mono(coefficient=parsed_dict['coefficient'], variables_dict=parsed_dict['variables_dict'])

    @staticmethod
    def from_json(json_content):
        """Receives a string in JSON syntax, and returns a new Mono object from it."""
        parsed_dictionary = json.loads(json_content)
        if parsed_dictionary['type'].strip().lower() != 'mono':
            raise ValueError(f"Incompatible type {parsed_dictionary['type']}: Expected 'Mono'")
        return Mono(coefficient=parsed_dictionary['coefficient'], variables_dict=parsed_dictionary['variables_dict'])

    @staticmethod
    def import_json(path):
        """reads the contents of a JSON file with a single Mono object and tries to create a Mono object from it"""
        with open(path) as json_file:
            return Mono.from_json(json_file.read())

    def __copy__(self):
        return Mono(coefficient=self._coefficient,
                    variables_dict=self.__variables.copy() if self.__variables is not None else None)

    def __neg__(self):
        return Mono(coefficient=-self._coefficient, variables_dict=self.__variables)

    def assign(self, **kwargs):
        if self.__variables is None:
            return None
        new_dict = dict()
        for variable in self.__variables:
            if variable in kwargs:
                self._coefficient *= kwargs[variable] ** self.__variables[variable]
            else:
                new_dict[variable] = self.__variables[variable]
        self.__variables = new_dict

    def __call__(self, **kwargs):  # implement with the lambda thing ?
        pass

    def simplify(self):
        if self._coefficient == 0:
            self.__variables = None
        else:
            if self.__variables:
                self.__variables = {key: value for key, value in self.__variables.items() if value != 0}
            else:
                self.__variables = None

    def python_syntax(self) -> str:
        if self.__variables in ({}, None):
            return f"{self._coefficient}"
        formatted_variables = "*".join(
            (f"{variable}" if power == 1 else f"{variable}**{power}") for variable, power in self.__variables.items())
        coefficient = format_coefficient(self._coefficient)
        if coefficient not in ('', '+', '-'):
            coefficient += '*'
        return f"{coefficient}{formatted_variables}"

    def try_evaluate(self) -> Optional[float]:
        """ trying to evaluate the Mono object into a float number """
        if self.__variables in ({}, None):
            return self._coefficient
        return None

    def derivative(self):
        if self.__variables is not None and len(self.__variables) > 1:
            raise ValueError(f"Try using partial_derivative(), for expression with more than one variable ")
        if self.__variables is None:  # the derivative of a free number is 0
            return 0
        power = fetch_power(self.__variables)
        if power == 1:
            return self._coefficient
        elif power == 0:  # Since x^0 = 1, and the derivative of a free number is 0
            return 0
        elif power > 0:
            return Mono(self._coefficient * power, variables_dict={fetch_variable(self.__variables): power - 1})
        elif power < 0:  # Fraction functions. for example : 3x^-1 = 3/x
            return NotImplementedError

    def partial_derivative(self, variables: Iterable):
        # TODO: make a more specific type hint, but that accept generators
        if self.__variables is None:
            return Mono(0)
        derived_expression = self.__copy__()  # Using one copy for all derivatives, to save memory and time!
        for variable in variables:
            # Assuming variable is a one lettered string ( this method is internally used inside the class )
            if variable not in self.__variables:
                return Mono(0)  # Everything else is considered a parameter, so they derive to zero.
            derived_expression._coefficient *= derived_expression.__variables[variable]
            derived_expression.__variables[variable] -= 1
            if derived_expression.__variables[variable] == 0:  # Delete x^0 for example, since it's just 1....
                del derived_expression.__variables[variable]
        derived_expression.simplify()
        return derived_expression

    def integral(self, variable_name='x'):
        """
        Computes the integral of the polynomial expression
        :param variable_name: if the expression is a number, a variable name needs to be specified.
        For example, 6 -> 6x when it is default
        :return: the integral of the expression
        :rtype: Should be of type Mono
        """
        if self.__variables is not None and len(self.__variables) > 1:
            raise ValueError(f"Can only compute the derivative with one variable or less ( got {len(self.__variables)}")
        if self.__variables is None:
            return Mono(self._coefficient, variables_dict={variable_name: 1})
        variable, power = fetch_variable(self.__variables), fetch_power(self.__variables)
        if power == 0:
            return Mono(round_decimal(self._coefficient), variables_dict={variable: 1})
        elif power > 0:
            return Mono(round_decimal(self._coefficient / (power + 1)), variables_dict={variable: power + 1})
        else:  # ( power < 0 )
            return NotImplementedError

    def to_lambda(self):
        """
        Produces a lambda expression from the Mono object.

        :return: Returns a lambda expression corresponding to the Mono object.
        """
        return to_lambda(self.__str__(), self.__variables)

    def to_Function(self):
        """
        Get a function from the Mono object.

        :return: A Function object, corresponding to the  Mono object
        """
        return Function(self.python_syntax())


# SOME STRING ANALYSIS METHODS

def mono_from_str(mono_expression: str, get_tuple=False):
    """
    Analyzes a string, such as "3x^2*y^2" and creates a monomial expression ( of type Mono )
    :param mono_expression: the string that represents the monomial
    :param get_tuple: if set to True, instead of a Mono object, the _coefficient(float) and __variables(dict)
    will be returned.
    :return: The monomial, or if get_tuple=True, then its _coefficient and __variables.
    :rtype: Mono or tuple
    """
    try:
        # Try just converting it to float, if it's a normal number
        mono_expression = clean_from_spaces(mono_expression)
        number = float(mono_expression)
        if get_tuple:
            return number, None
        return Mono(number)
    except (ValueError, TypeError):
        mono_expression: str = mono_expression.strip().replace("**", "^")
        for variable in (character for character in mono_expression if character in allowed_characters):
            occurrences = [m.start() for m in re.finditer(variable, mono_expression)]
        new_expression: str = ''
        for character_index in range(len(mono_expression)):
            new_expression = "".join((new_expression, mono_expression[character_index]))
            if character_index + 1 in occurrences and (mono_expression[character_index].isdigit() or mono_expression[
                character_index].isalpha()):
                new_expression += '*'
        basic_expressions: list = new_expression.split('*')
        final_coefficient, variables_and_powers = 1, dict()
        for basic_expression in basic_expressions:
            variable: str = "".join([character for character in basic_expression if character in allowed_characters])
            current_coefficient, dictionary_item = __data_from_single(basic_expression, variable)
            final_coefficient *= current_coefficient
            if dictionary_item is not None:
                variables_and_powers = {**variables_and_powers, **dictionary_item}
        if get_tuple:
            return final_coefficient, variables_and_powers

        return Mono(coefficient=final_coefficient, variables_dict=variables_and_powers)


def poly_from_str(poly_expression: str, get_list=False) -> "Union[Poly,List]":
    """
    Analyzes a string, such as "3x^2 + 2xy - 7" and generates a polynomial expression
    :param poly_expression:
    :param get_list: if set to True, a list of the monomials ( Mono objects ) will be returned instead
    of a Poly object
    :return: a polynomial corresponding to the string, or a list of monomials.
    :rtype: Poly or list
    """
    poly_expression = clean_from_spaces(poly_expression)
    expressions = (mono_expression for mono_expression in poly_expression.replace('-', '+-').split('+') if
                   mono_expression != "")
    expressions = [mono_from_str(expression) for expression in expressions]
    if get_list:
        return expressions
    return Poly(expressions)


def monic_poly_from_coefficients(coefficients, var_name='x') -> "Poly":
    length = len(coefficients)
    return Poly([Mono(coefficient=coef, variables_dict={var_name: length - 1 - index}) for index, coef in
                 enumerate(coefficients)])


def coefficient_to_float(coefficient: str) -> Optional[float]:  # TODO: verbose ? due to extract_coefficient()
    # TODO: further implement it so it'll support i,e, and pi.
    return float(coefficient)


def __helper_trigo(expression: str) -> Optional[Tuple[int, Optional[float]]]:
    try:
        first_letter_index = expression.find(next(
            (character for character in expression if character.isalpha() and character not in ('e', 'i'))))
        return first_letter_index, coefficient_to_float(str(extract_coefficient(expression[:first_letter_index])))
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


class Var(Mono):
    def __init__(self, variable='x'):
        super().__init__(coefficient=1, variables_dict={variable: 1})

    # Make sure the variables_dict don't change, by creating copies
    def __iadd__(self, other):
        return super().__add__(other)

    def __isub__(self, other):
        return super().__sub__(other)

    def __imul__(self, other):
        return super().__mul__(other)

    def __itruediv__(self, other):
        return super().__truediv__(other)

    def __ipow__(self, other):
        return super().__pow__(other)


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


class FastPoly(IExpression, IPlottable):
    __slots__ = ['__variables', '__variables_dict']

    def __init__(self, polynomial: Union[str, dict, list, tuple, float, int], variables: Iterable = None):

        self.__variables = None if variables is None else list(variables)
        if isinstance(polynomial, (int, float)):
            self.__variables = []
            self.__variables_dict = {'free': polynomial}
        if isinstance(polynomial, str):  # Parse a given string
            self.__variables_dict = ParseExpression.parse_polynomial(polynomial, self.__variables, strict_syntax=True)
            self.__variables_dict = ParseExpression.parse_polynomial(polynomial, self.__variables, strict_syntax=True)
        elif isinstance(polynomial, dict):  # Enter the parsed dictionary
            if "free" not in polynomial.keys():
                raise KeyError(f"Key 'free' must appear in FastPoly.__init__() when entering dict. Its value is the"
                               f"free number of the expression")
            self.__variables_dict = polynomial.copy()
        elif isinstance(polynomial, (list, tuple)):  # Enter the coefficients of a polynomial with 1 variable
            if not polynomial:
                raise ValueError(f"FastPoly.__init__(): At least one coefficient is required.")
            if self.__variables is None:
                self.__variables = ['x']  # Default variable is 'x' for simplicity..
            elif len(self.__variables) > 1:  # More than one variable entered - invalid for this method..
                raise ValueError("FastPoly.__init__(): When entering a list of coefficients, only 1 variable"
                                 f"is accepted, but found {len(self.__variables)}")
            x_coefficients, free_number = polynomial[:-1], polynomial[-1]
            if not x_coefficients:
                self.__variables = []
                self.__variables_dict = {'free': free_number}
            else:
                self.__variables_dict = {self.__variables[0]: x_coefficients, 'free': free_number}

        else:
            raise TypeError(f"Invalid type {type(polynomial)} in FastPoly.__init__(). Expected types 'str' or 'dict'")
        if self.__variables is None:
            self.__variables = [key for key in self.__variables_dict.keys() if key != 'free']

    @property
    def variables(self):
        return self.__variables.copy()

    @property
    def num_of_variables(self):
        return len(self.__variables)

    @property
    def variables_dict(self):
        return self.__variables_dict.copy()

    @property
    def degree(self) -> Union[float, dict]:
        num_of_variables = len(self.__variables)
        if num_of_variables == 0:
            return 0
        elif num_of_variables == 1:
            return len(self.__variables_dict[self.__variables[0]])
        return {variable: len(self.__variables_dict[variable]) for variable in self.__variables}

    @property
    def is_free_number(self):
        return self.num_of_variables == 0 or len(self.__variables_dict.keys()) == 1

    def derivative(self) -> "FastPoly":
        num_of_variables = self.num_of_variables
        if num_of_variables == 0:
            return FastPoly(0)
        elif num_of_variables == 1:
            variable = self.__variables[0]
            derivative_coefficients = derivative(self.__variables_dict[variable] + [self.__variables_dict['free']])
            free_number = derivative_coefficients[-1]
            del derivative_coefficients[-1]
            if isinstance(derivative_coefficients, (int, float)):  # if a number is returned ..
                derivative_coefficients = []
                self.__variables_dict['free'] = derivative_coefficients
            return FastPoly({variable: derivative_coefficients, 'free': free_number})
        else:
            raise ValueError("Please use the partial_derivative() method for polynomials with several "
                             "variables")

    def partial_derivative(self, variables: Iterable[str]):
        pass

    def extremums(self):
        num_of_variables = len(self.__variables)
        if num_of_variables == 0:
            return None
        elif num_of_variables == 1:
            my_lambda = self.to_lambda()
            my_derivative = self.derivative()
            if my_derivative.is_free_number:
                return None
            derivative_roots = my_derivative.roots(nmax=1000)
            myRoots = [Point2D(root.real, my_lambda(root.real)) for root in derivative_roots if root.imag <= 0.00001]
            return PointCollection(myRoots)
        else:
            pass

    def integral(self, c: float = 0, variable='x'):
        num_of_variables = len(self.__variables)
        if num_of_variables == 0:
            return FastPoly({variable: [self.__variables_dict['free']], 'free': c})
        elif num_of_variables != 1:
            raise ValueError("Cannot integrate a PolyFast object with more than 1 variable")
        variables = self.__variables_dict[self.__variables[0]] + [self.__variables_dict['free']]
        result = integral(variables, modify_original=True)
        del result[-1]
        return FastPoly({self.__variables[0]: result, 'free': c})

    def newton(self, initial: float = 0, epsilon: float = 0.00001, nmax=10_000):
        return newton_raphson(self.to_lambda(), self.derivative().to_lambda(), initial, epsilon, nmax)

    def halley(self, initial: float = 0, epsilon: float = 0.00001, nmax=10_000):
        first_derivative = self.derivative()
        second_derivative = first_derivative.derivative()
        return halleys_method(self.to_lambda(), first_derivative.to_lambda(), second_derivative.to_lambda(), initial,
                              epsilon, nmax)

    def __add_or_sub(self, other: "FastPoly", mode: str):
        for variable in other.__variables:
            if variable in self.__variables:
                add_or_sub_coefficients(self.__variables_dict[variable], other.__variables_dict[variable], mode=mode,
                                        copy_first=False)
            else:
                self.__variables.append(variable)
                if mode == 'add':
                    self.__variables_dict[variable] = other.__variables_dict[variable].copy()
                elif mode == 'sub':
                    self.__variables_dict[variable] = [-coef for coef in other.__variables_dict[variable]]

        if mode == 'add':
            self.__variables_dict['free'] += other.__variables_dict['free']
        elif mode == 'sub':
            self.__variables_dict['free'] -= other.__variables_dict['free']

    def __iadd__(self, other: Union[IExpression, int, float]):
        if other == 0:
            return self
        if isinstance(other, (int, float)):
            self.__variables_dict['free'] += other
            return self
        if not isinstance(other, IExpression):
            raise TypeError(f"Invalid type {type(other)} when adding FastPoly objects. Expected types "
                            f"'int', 'float', or 'IExpression'")
        other_evaluation = other.try_evaluate()
        if other_evaluation is not None:
            self.__variables_dict['free'] += other_evaluation
            return self
        if not isinstance(other, FastPoly):
            return ExpressionSum((self, other))
        self.__add_or_sub(other, mode='add')
        return self

    def __isub__(self, other):
        if isinstance(other, (int, float)):
            self.__variables_dict['free'] -= other
            return self
        if not isinstance(other, IExpression):
            raise TypeError(f"Invalid type {type(other)} when subtracting FastPoly objects. Expected types "
                            f"'int', 'float', or 'IExpression'")
        other_evaluation = other.try_evaluate()
        if other_evaluation is not None:
            self.__variables_dict['free'] -= other_evaluation
            return self
        if not isinstance(other, FastPoly):
            return ExpressionSum((self, other))
        self.__add_or_sub(other, mode='sub')
        return self

    def __imul__(self, other):
        pass

    def __itruediv__(self, other):
        pass

    def __ipow__(self, other):
        pass

    def assign(self, **kwargs):
        for variable, value in kwargs.items():
            if variable not in self.__variables_dict:
                continue
            coefficients_length = len(self.__variables_dict[variable])
            for index, coefficient in enumerate(self.__variables_dict[variable]):
                self.__variables_dict['free'] += coefficient * value ** (coefficients_length - index)
            del self.__variables_dict[variable]  # Delete the key value pair as it was evaluated into free numbers

    def simplify(self):
        warnings.warn("FastPoly objects are already simplified. Method is deprecated.")

    def try_evaluate(self) -> Optional[float]:
        if self.num_of_variables == 0:
            return self.__variables_dict['free']
        return None

    def roots(self, epsilon=0.00001, nmax: int = 10000):
        num_of_variables = len(self.__variables)
        if num_of_variables == 0:
            return "Infinite" if self.__variables_dict['free'] == 0 else None
        elif num_of_variables == 1:
            return solve_polynomial(self.__variables_dict[self.__variables[0]] + [self.__variables_dict['free']],
                                    epsilon, nmax)
        else:
            raise ValueError(f"Can only solve polynomials with 1 variable, but found {num_of_variables}")

    def __eq__(self, other: "Union[IExpression, FastPoly]"):
        """Equate between expressions. not fully compatible with the IExpression classes ..."""
        if other is None:
            return False
        if not isinstance(other, IExpression):
            raise TypeError(f"Invalid type {type(other)} for equating FastPoly objects")
        other_evaluation = other.try_evaluate()
        if other_evaluation is not None:
            my_evaluation = self.try_evaluate()
            if my_evaluation is not None:
                return my_evaluation == other_evaluation
        if not isinstance(other, FastPoly):
            return False
        return self.__variables_dict == other.__variables_dict

    def __ne__(self, other: "FastPoly"):
        return not self.__eq__(other)

    def __neg__(self):
        new_dict = {variable: [-coefficient for coefficient in coefficients] for variable, coefficients in
                    self.__variables_dict.items() if variable != 'free'}
        new_dict['free'] = -self.__variables_dict['free']
        return FastPoly(new_dict)

    def __copy__(self):
        return FastPoly(self.__variables_dict)  # the dictionary is later copied so no memory will be shared

    def to_lambda(self):
        return to_lambda(self.__str__(), self.__variables)

    def plot(self, start: float = -10, stop: float = 10,
             step: float = 0.01, ymin: float = -10, ymax: float = 10, text=None, show_axis=True, show=True,
             fig=None, ax=None, formatText=True, values=None):
        lambda_expression = self.to_lambda()
        num_of_variables = self.num_of_variables
        if text is None:
            text = self.__str__()
        if num_of_variables == 0:  # TODO: plot this in a number axis
            raise ValueError("Cannot plot a polynomial with 0 variables_dict")
        elif num_of_variables == 1:
            plot_function(lambda_expression, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=text,
                          show_axis=show_axis, show=show, fig=fig, ax=ax, formatText=formatText, values=values)
        elif num_of_variables == 2:
            plot_function_3d(lambda_expression, start=start, stop=stop, step=step)  # TODO: update the parameters
        else:
            raise ValueError("Cannot plot a function with more than two variables_dict (As for this version)")

    def to_dict(self):
        return {"type": "FastPoly", "data": self.__variables_dict.copy()}

    @staticmethod
    def from_dict(given_dict: dict):
        return FastPoly(given_dict)

    @staticmethod
    def from_json(json_content: str):
        loaded_json = json.loads(json_content)
        if loaded_json['type'].strip().lower() != 'fastpoly':
            raise ValueError(f"Unexpected type '{loaded_json['type']}' when creating a new "
                             f"FastPoly object from JSON (Expected TypePoly).")
        return FastPoly(loaded_json['data'])

    @staticmethod
    def import_json(path):
        with open(path, 'r') as json_file:
            return FastPoly.from_json(json_file.read())

    def python_syntax(self):
        return ParseExpression.unparse_polynomial(parsed_dict=self.__variables_dict, syntax='pythonic')

    def __str__(self):
        return ParseExpression.unparse_polynomial(parsed_dict=self.__variables_dict)


class Poly(IExpression, IPlottable):
    __slots__ = ['_expressions', '__loop_index']

    def __init__(self, expressions):
        self.__loop_index = 0
        if isinstance(expressions, str):
            self._expressions: List[Mono] = poly_from_str(expressions, get_list=True)
            self.simplify()
        elif isinstance(expressions, (int, float)):
            self._expressions = [Mono(expressions)]
        elif isinstance(expressions, Mono):
            self._expressions = [expressions.__copy__()]
        elif isinstance(expressions, Iterable):
            self._expressions = []
            for expression in expressions:
                if isinstance(expression, Mono):
                    self._expressions.append(expression.__copy__())  # avoiding memory sharing by passing by value
                elif isinstance(expression, str):
                    self._expressions += (
                        poly_from_str(expression, get_list=True))
                elif isinstance(expression, Poly):
                    self._expressions.extend(expression.expressions.copy())
                elif isinstance(expression, (int, float)):
                    self._expressions.append(Mono(expression))
                else:
                    warnings.warn(f"Couldn't process expression '{expression} with invalid type {type(expression)}'")
            self.simplify()
        elif isinstance(expressions, Poly):
            self._expressions: List[Mono] = [mono_expression.__copy__() for mono_expression in expressions._expressions]
            self.simplify()

        elif isinstance(expressions, Mono):
            self._expressions: List[Mono] = [expressions.__copy__()]
        else:
            raise TypeError(
                f"Invalid type {type(expressions)} in Poly.__init__(). Allowed types: list,tuple,Mono,"
                f"Poly,str,int,float "
            )

    @property
    def expressions(self):
        return self._expressions

    @expressions.setter
    def expressions(self, expressions):
        self._expressions = expressions

    def __iadd__(self, other: Union[IExpression, int, float, str]):
        if other == 0:
            return self
        if isinstance(other, (int, float)):
            self.__add_monomial(Mono(other))
            if all(expression.coefficient == 0 for expression in self.expressions):
                self.expressions = [Mono(0)]
            self.simplify()
            return self

        elif isinstance(other, str):
            expressions = poly_from_str(other, get_list=True)
            for mono_expression in expressions:
                self.__add_monomial(mono_expression)
            if all(expression.coefficient == 0 for expression in self.expressions):
                self.expressions = [Mono(0)]
            self.simplify()
            return self

        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if other_evaluation is not None:
                self.__add_monomial(Mono(other_evaluation))
                self.simplify()
                return self

            if isinstance(other, Mono):
                self.__add_monomial(other)
                if all(expression.coefficient == 0 for expression in self.expressions):
                    self.expressions = [Mono(0)]
                self.simplify()
                return self

            elif isinstance(other, Poly):
                for mono_expression in other.expressions:
                    self.__add_monomial(mono_expression)
                if all(expression.coefficient == 0 for expression in self.expressions):
                    self.expressions = [Mono(0)]
                self.simplify()
                return self
            else:  # If it's just a random ExpressionSum expression
                return ExpressionSum((self, other))
        else:
            raise TypeError(
                f"__add__ : invalid type '{type(other)}'. Allowed types: str, Mono, Poly, int, or float"
            )

    def __add_monomial(self, other: Mono) -> None:
        self.__filter_zeroes()
        for index, expression in enumerate(self.expressions):
            if expression.variables_dict == other.variables_dict or (not expression.variables and not other.variables):
                # if they can be added
                self._expressions[index] += other
                return
        self._expressions.append(other)

    def __sub_monomial(self, other: Mono) -> None:
        self.__filter_zeroes()
        for index, expression in enumerate(self._expressions):
            if expression.variables_dict == other.variables_dict or (not expression.variables and not other.variables):
                # if they can be subtracted
                self._expressions[index] -= other
                return  # Break out of the function.

        self.expressions.append(-other)

    def __rsub__(self, other: Union[int, float, str, IExpression]):
        if isinstance(other, (int, float, str)):
            other = Poly(other)

        if isinstance(other, Mono):
            return Poly([expression / other for expression in self.expressions])
        elif isinstance(other, Poly):
            other.__isub__(self)
            return other
        elif isinstance(other, IExpression):
            return ExpressionMul((other, -self))
        else:
            raise TypeError(
                f"Poly.__rsub__: Expected types int,float,str,Mono,Poly, but got {type(other)}")

    def __isub__(self, other: Union[int, float, IExpression, str]):
        if isinstance(other, (int, float)):
            self.__sub_monomial(Mono(other))
            if all(expression.coefficient == 0 for expression in self.expressions):  # TODO: check if needed
                self.expressions = [Mono(0)]
            self.simplify()
            return self

        elif isinstance(other, str):
            expressions = poly_from_str(other, get_list=True)
            for mono_expression in expressions:
                self.__sub_monomial(mono_expression)
            if all(expression.coefficient == 0 for expression in self.expressions):
                self.expressions = [Mono(0)]
            self.simplify()
            return self

        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if other_evaluation is not None:
                self.__sub_monomial(Mono(other_evaluation))
                self.simplify()
                return self

            if isinstance(other, Mono):
                self.__sub_monomial(other)
                if all(expression.coefficient == 0 for expression in self._expressions):
                    self.expressions = [Mono(0)]
                self.simplify()
                return self

            elif isinstance(other, Poly):
                for mono_expression in other._expressions:
                    self.__sub_monomial(mono_expression)
                if all(expression.coefficient == 0 for expression in self._expressions):
                    self.expressions = [Mono(0)]
                self.simplify()
                return self
            else:  # If it's just a random IExpression expression
                return ExpressionSum((self, -other))
        else:
            raise TypeError(
                f"Invalid type '{type(other)} while subtracting polynomials.")

    def __neg__(self):
        return Poly([-expression for expression in self.expressions])

    def __imul__(self, other: Union[int, float, IExpression]):  # TODO: try to make it more efficient ..
        if other == 0:
            return Mono(coefficient=0)
        if isinstance(other, (int, float)):
            for index, expression in enumerate(self._expressions):
                self._expressions[index].coefficient *= other
            if all(expression.coefficient == 0 for expression in self._expressions):
                self._expressions = [Mono(0)]
            self.simplify()
            return self
        if isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if other_evaluation is not None:
                for index in range(len(self._expressions)):
                    self._expressions[index] *= other_evaluation
                self.simplify()
                return self
            if isinstance(other, Mono):
                for index, expression in enumerate(self._expressions):
                    self._expressions[index] *= other
                self.simplify()
                return self
            elif isinstance(other, Poly):
                new_expressions = []
                for expression1 in self.expressions:
                    for expression2 in other.expressions:

                        result = expression1 * expression2
                        found = False
                        for index, new_expression in enumerate(
                                new_expressions):  # Checking whether to append or simplify
                            if new_expression.variables_dict == result.variables_dict:
                                addition_result = new_expression + result
                                if addition_result.coefficient == 0:
                                    del new_expressions[index]
                                else:
                                    new_expressions[index] = addition_result
                                found = True
                                break
                        if not found:
                            new_expressions.append(result.__copy__())  # is it necessary ?
                self._expressions = new_expressions
                self.simplify()
                return self
            else:  # Multiply by an unknown IExpression. Could be Root, Fraction, etc.
                return other * self

        elif isinstance(other, Matrix):  # Check if this works
            other.multiply_all(self)
        elif isinstance(other, Vector):
            raise NotImplementedError
        elif isinstance(other, Iterable):
            return [item * self for item in other]

    def __filter_zeroes(self):
        if len(self._expressions) > 1:
            for index, expression in enumerate(self._expressions):
                if expression.coefficient == 0:
                    del self.expressions[index]

    def divide_by_number(self, number: int):
        for mono_expression in self._expressions:
            mono_expression.divide_by_number(number)
        return self

    def divide_by_poly(self, other: "Union[Mono, Poly]", get_remainder=False, nmax=1000):
        # TODO: fix this method ....
        if isinstance(other, Poly) and len(other.expressions) == 1:
            other = other.expressions[0]  # If the polynomial contains only one monomial, turn it to Mono
        if isinstance(other, Mono):
            if other.coefficient == 0:
                raise ZeroDivisionError("cannot divide by an expression whose coefficient is zero")
            other_copy = other.__copy__()
            other_copy.coefficient = 1 / other_copy.coefficient
            if other_copy.variables_dict is not None:
                other_copy.variables_dict = {variable: -value for (variable, value) in
                                             other_copy.variables_dict.items()}
                # dividing by x^3 is equivalent to multiplying by x^-3
            if get_remainder:
                return self.__imul__(other_copy), 0
            return self.__imul__(other_copy)
        elif isinstance(other, Poly):
            new_expression, remainder = Mono(0), 0
            temp_expressions = Poly(self._expressions.copy())
            for i in range(nmax):
                if len(temp_expressions._expressions) == 0:
                    new_expression.simplify()
                    if get_remainder:
                        return new_expression, 0
                    return new_expression
                if len(temp_expressions._expressions) == 1 and temp_expressions.expressions[0].variables_dict is None:
                    if get_remainder:
                        return new_expression, other._expressions[0]
                    return new_expression + other._expressions[0] / other

                first_item = temp_expressions._expressions[0] / other._expressions[0]
                new_expression += first_item.__copy__()
                subtraction_expressions = first_item * other
                temp_expressions -= subtraction_expressions
                if len(temp_expressions.expressions) == 1:
                    if temp_expressions.expressions[0].coefficient == 0:
                        if isinstance(new_expression, Poly):
                            new_expression.simplify()
                        if get_remainder:
                            return new_expression, remainder
                        if remainder == 0:
                            return new_expression
                        new_expression += Fraction(remainder, other)
                        return new_expression
                    elif temp_expressions.expressions[0].variables_dict is None:  # Reached a result with a remainder
                        if isinstance(new_expression, Poly):
                            new_expression.sort()
                        remainder = temp_expressions.expressions[0].coefficient
                        if get_remainder:
                            return new_expression, remainder
                        if remainder == 0:
                            return new_expression
                        new_expression += Fraction(remainder, other)
                        return new_expression
                    else:  # The remainder is algebraic
                        warnings.warn("Got an algebraic remainder when dividing Poly objects")
                        if isinstance(new_expression, Poly):
                            new_expression.sort()
                        if get_remainder:
                            return new_expression, remainder
                        if remainder == 0:
                            return new_expression
                        new_expression += Fraction(remainder, other)
                        return new_expression
            warnings.warn("Division timed out ...")
            return PolyFraction(self, other)

    def __itruediv__(self, other: Union[int, float, IExpression], get_remainder=False):
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("cannot divide by 0")
            if my_evaluation is None:  # if the Poly object can't be evaluated into a free number
                if get_remainder:
                    return self.divide_by_number(other), 0
                return self.divide_by_number(other)
            else:
                if get_remainder:
                    return Mono(coefficient=my_evaluation / other), 0
                return Mono(coefficient=my_evaluation / other)
        if isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if other_evaluation == 0:
                raise ZeroDivisionError(f"Cannot divide a polynomial by the expression {other} which evaluates to 0")
            if None not in (my_evaluation, other_evaluation):
                # Both expressions can be evaluated
                if get_remainder:
                    return Mono(coefficient=my_evaluation / other_evaluation), 0
                return Mono(coefficient=my_evaluation / other_evaluation)
            elif my_evaluation is None and other_evaluation is not None:
                # only the other object can be evaluated into a number
                if get_remainder:
                    return self.divide_by_number(other_evaluation), 0
                return self.divide_by_number(other_evaluation)
            else:  # Both expressions can't be evaluated into numbers apparently
                if isinstance(other, (Poly, Mono)):
                    return self.divide_by_poly(other, get_remainder=get_remainder)
                else:
                    return Fraction(self, other)
        else:
            raise TypeError(f"Invalid type '{type(other)} when dividing Poly objects' ")

    def __truediv__(self, other: Union[int, float, IExpression], get_remainder=False):
        return self.__copy__().__itruediv__(other, get_remainder=get_remainder)

    def __calc_binomial(self, power: int):
        """Internal method for using the newton's binomial in order to speed up calculations in the form (a+b)^2"""
        expressions = []
        first, second = self._expressions[0], self._expressions[1]
        if_number1, if_number2 = first.variables_dict is None, second.variables_dict is None
        for k in range(power + 1):
            comb_result = comb(power, k)
            first_power, second_power = power - k, k
            if if_number1:
                first_expression = Mono(first.coefficient ** first_power * comb_result)
            else:
                first_expression = Mono(first.coefficient ** first_power * comb_result,
                                        {key: value * first_power for (key, value) in
                                         first.variables_dict.items()})
            if if_number2:
                second_expression = Mono(second.coefficient ** second_power)
            else:
                second_expression = Mono(second.coefficient ** second_power,
                                         {key: value * second_power for (key, value) in
                                          second.variables_dict.items()})
            expressions.append(first_expression * second_expression)
        return Poly(expressions)

    def __pow__(self, power: Union[int, float, IExpression, str], modulo=None):
        if isinstance(power, float):  # Power by float is not supported yet ...
            power = int(power)
        if not isinstance(power, int):
            if isinstance(power, str):
                power = Poly(power)
            if isinstance(power, Mono):
                if power.variables_dict is not None:
                    raise ValueError("Cannot perform power with an algebraic exponent on polynomials")
                    # TODO: implement exponents for that
                else:
                    power = power.coefficient
            elif isinstance(power, Poly):
                if len(power._expressions) == 1 and power._expressions[0].variables_dict is None:
                    power = power._expressions[0].coefficient
                else:
                    raise ValueError("Cannot perform power with an algebraic exponent")
        if power == 0:
            return Poly(1)
        elif power == 1:
            return Poly(self._expressions)

        my_evaluation = self.try_evaluate()
        if my_evaluation is not None:
            return Mono(coefficient=my_evaluation ** power)

        if len(self.expressions) == 2:  # FOR TWO ITEMS, COMPUTE THE RESULT WITH THE BINOMIAL THEOREM
            return self.__calc_binomial(power)

        else:  # FOR MORE THAN TWO ITEMS, OR JUST 1, CALCULATE IT AS MULTIPLICATION ( LESS EFFICIENT )
            new_expression = self
            for i in range(power - 1):
                new_expression *= self
            return new_expression

    def __rpow__(self, other, power, modulo=None):  # TODO: for that, basic exponents need to be implemented.
        if len(self._expressions) == 1 and self._expressions[0].variables_dict is None:
            if not isinstance(other, (Mono, Poly)):
                other = Poly(other)
            return other.__pow__(self)

        else:
            return Exponent(self, other)

    def __ipow__(self, other):  # TODO: re-implement it later
        self._expressions = self.__pow__(other)._expressions
        return self

    def is_number(self):
        return all(expression.is_number() for expression in self._expressions)

    def try_evaluate(self) -> Optional[Union[int, float, complex]]:
        if not self._expressions:
            return 0
        if self.is_number() and (length := len(self._expressions)) > 0:
            if length > 1:
                self.simplify()
            return self._expressions[0].coefficient
        return None

    def __eq__(self, other):
        if other is None:
            return False
        if isinstance(other, str):
            other = Poly(other)
        if isinstance(other, (int, float, Mono)):
            if len(self._expressions) != 1:
                return False
            return self._expressions[0].__eq__(other)
        elif isinstance(other, Poly):
            self.simplify()
            other.simplify()
            my_num_of_variables = self.num_of_variables
            other_num_of_variables = other.num_of_variables
            if my_num_of_variables != other_num_of_variables:
                return False
            if my_num_of_variables == 0:  # both expressions don't contain any variable, meaning only free numbers
                if len(self._expressions) != len(other._expressions):
                    return False
                return self._expressions[0] == other._expressions[
                    1]  # After simplification,only one free number is left
            elif my_num_of_variables == 1:  # both expressions have one variable
                return self._expressions == other._expressions  # all items should be in the same place when sorted!
            else:  # more than one variable
                expressions_checked = []
                for expression in self._expressions:
                    if expression not in expressions_checked:
                        instances_in_other = other._expressions.count(expression)
                        instances_in_me = self._expressions.count(expression)
                        if instances_in_other != instances_in_me:
                            return False
                        expressions_checked.append(expression)

                for other_expression in other._expressions:
                    if other_expression not in expressions_checked:
                        instances_in_me = self._expressions.count(other_expression)
                        instances_in_other = other._expressions.count(other_expression)
                        if instances_in_me != instances_in_other:
                            return False
                        expressions_checked.append(expression)
            return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __iter__(self):
        self.__loop_index = 0
        return self

    def __next__(self):
        if self.__loop_index < len(self.expressions):
            result = self._expressions[self.__loop_index]
            self.__loop_index += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, item):
        return self._expressions.__getitem__(item)

    def derivative(self):
        for expression in self._expressions:
            if expression.variables_dict is not None and len(expression.variables_dict) > 1:
                raise ValueError("Try using partial_derivative() for expression with more than variable")
        derived_expression = Poly([expression.derivative() for expression in self._expressions])
        derived_expression.simplify()
        derived_expression.sort()
        return derived_expression

    def is_empty(self) -> bool:
        return not self._expressions

    def partial_derivative(self, variables: Iterable):  # TODO: implement it more efficiently
        derived_expression = Poly((monomial.partial_derivative(variables) for monomial in self._expressions))
        derived_expression.simplify()
        derived_expression.sort()
        if derived_expression.is_empty():
            return Mono(0)
        return derived_expression

    def integral(self, add_c=False):
        for expression in self.expressions:
            if expression.variables_dict is not None and len(expression.variables_dict) > 1:
                raise ValueError(
                    f"IExpression {expression.__str__()}: Can only compute the integral with one variable or "
                    f"less ( got {len(expression.variables_dict)}")
        result = Poly([expression.integral() for expression in self.expressions])  # TODO: fix double values problem.
        if add_c:
            c = Var('c')
            result += c
        return result

    @property
    def variables(self):
        variables = set()
        for expression in self._expressions:
            variables.update(expression.variables)
        return variables

    @property
    def num_of_variables(self):
        return len(self.variables)

    def coefficients(self):
        """
        convert the polynomial expression to a list of coefficients. Currently works only with one variable.
        :return:
        """
        number_of_variables = self.num_of_variables
        if number_of_variables == 0:  # No variables_dict found - free number or empty expression
            num_of_expressions = len(self._expressions)
            if num_of_expressions == 0:
                return None
            elif num_of_expressions == 1:
                return [self._expressions[0].coefficient]
            elif num_of_expressions > 1:
                self.simplify()
                return [self._expressions[0].coefficient]
        elif number_of_variables > 1:  # Expressions with more than one variable aren't valid!
            raise ValueError(
                f"Can only fetch the coefficients of a polynomial with 1 variable, found {number_of_variables}")
        # One variable - expressions such as x^2 + 6
        sorted_exprs = sorted_expressions(
            [expression for expression in self._expressions if expression.variables_dict is not None])
        biggest_power = max_power(sorted_exprs)
        coefficients = [0] * (int(fetch_power(biggest_power.variables_dict)) + 1)
        for index, sorted_expression in enumerate(sorted_exprs):
            coefficients[
                len(coefficients) - int(
                    fetch_power(sorted_expression.variables_dict)) - 1] = sorted_expression.coefficient
        free_numbers = [expression for expression in self._expressions if expression.variables_dict is None]
        free_number = sum((expression.coefficient for expression in free_numbers))
        coefficients[-1] = free_number
        return coefficients

    def assign(self, **kwargs):  # TODO: check if it's even working !
        for expression in self._expressions:
            expression.assign(**kwargs)
        self.simplify()

    def discriminant(self):
        my_coefficients = self.coefficients()
        length = len(my_coefficients)
        if length == 1:  # CONSTANT
            return 0  # There is no common convention for a discriminant of a constant polynomial.
        elif length == 2:  # LINEAR - convention is 1
            return 1
        elif length == 3:  # QUADRATIC
            return my_coefficients[1] ** 2 - 4 * my_coefficients[0] * my_coefficients[2]
        elif length == 4:  # CUBIC
            if my_coefficients[0] == 1 and my_coefficients[1] == 0:  # depressed cubic : x^3 + px + q
                return -4 * my_coefficients[2] ** 3 - 27 * my_coefficients[3] ** 2
        elif length == 5:  # QUARTIC
            a, b, c, d, e = my_coefficients[0], my_coefficients[1], my_coefficients[2], my_coefficients[3], \
                            my_coefficients[4]
            result = 256 * a ** 3 * e ** 3 - 192 * a ** 2 * b * d * e ** 2 - 128 * a ** 2 * c ** 2 * e ** 2 + 144 * a ** 2 * c * d ** 2 * e
            result += -27 * a ** 2 * d ** 4 + 144 * a * b ** 2 * c * e ** 2 - 6 * a * b ** 2 * d ** 2 * e - 80 * a * b * c ** 2 * d * e
            result += 18 * a * b * c * d ** 3 + 16 * a * c ** 4 * e - 4 * a * c ** 3 * d ** 2 - 27 * b ** 4 * e ** 2 + 18 * b ** 3 * c * d * e
            result += -4 * b ** 3 * d ** 3 - 4 * b ** 2 * c ** 3 * e + b ** 2 * c ** 2 * d ** 2
            return result
        else:
            raise ValueError("Discriminants are not supported yet for polynomials with degree 5 or more")

    def roots(self, epsilon=0.000001, nmax=10_000):
        my_coefficients = self.coefficients()
        return solve_polynomial(my_coefficients, epsilon, nmax)

    def real_roots(self):
        pass

    def extremums(self):
        num_of_variables = len(self.variables)
        if num_of_variables == 0:
            return None
        elif num_of_variables == 1:
            my_lambda = self.to_lambda()
            my_derivative = self.derivative()
            if my_derivative.is_number():
                return None
            derivative_roots = my_derivative.roots(nmax=1000)
            myRoots = [Point2D(root.real, my_lambda(root.real)) for root in derivative_roots if root.imag <= 0.00001]
            return PointCollection(myRoots)

    def extremums_axes(self, get_derivative=False):
        num_of_variables = len(self.variables)
        if num_of_variables == 0:
            return None
        elif num_of_variables == 1:
            my_derivative = self.derivative()
            if my_derivative.is_number():
                return None
            my_roots = [root.real for root in my_derivative.roots(nmax=1000) if root.imag <= 0.00001]
            my_roots.sort()
            if get_derivative:
                return my_roots, my_derivative
            return my_roots

    def up_and_down(self):
        extremums_axes, my_derivative = self.extremums_axes(get_derivative=True)
        return self.__up_and_down(extremums_axes, my_derivative)

    def __up_and_down(self, extremums_axes, my_derivative=None):
        x = Var('x')
        coefficients = self.coefficients()
        num_of_coefficients: int = len(coefficients)
        if num_of_coefficients == 1:  # free number
            return None, None  # the function just stays constant
        elif num_of_coefficients == 2:  # linear function
            if coefficients[0] > 0:
                return Range(expression=x, limits=(-np.inf, np.inf), operators=(LESS_THAN, LESS_THAN)), None
            elif coefficients[0] < 0:
                return None, Range(expression=x, limits=(-np.inf, np.inf), operators=(LESS_THAN, LESS_THAN))
        elif num_of_coefficients == 2:  # Quadratic function
            first = Range(expression=x, limits=(-np.inf, extremums_axes[0]), operators=(LESS_THAN, LESS_THAN))
            second = Range(expression=x, limits=(extremums_axes[0], np.inf), operators=(LESS_THAN, LESS_THAN))
            if coefficients[0] > 0:  # Happy parabola
                return second, first
            return first, second  # Sad parabola:

        else:
            num_of_extremums = len(extremums_axes)
            if num_of_extremums == 0:
                print("didn't find any extremums...")

            if my_derivative is None:
                my_derivative = self.derivative()
            derivative_lambda = my_derivative.to_lambda()
            up_ranges, down_ranges = [], []
            derivatives_values = [
                derivative_lambda(random.uniform(extremums_axes[i], extremums_axes[i + 1])) for i in
                range(num_of_extremums - 1)]
            before_value = derivative_lambda(extremums_axes[0] - 1)
            after_value = derivative_lambda(extremums_axes[-1] + 1)
            derivatives_values.append(after_value)
            if before_value > 0:
                up_ranges.append(
                    Range(expression=x, limits=(-np.inf, extremums_axes[0]), operators=(LESS_THAN, LESS_THAN)))
            elif before_value < 0:
                down_ranges.append(
                    Range(expression=x, limits=(-np.inf, extremums_axes[0]), operators=(LESS_THAN, LESS_THAN)))
            else:
                pass

            for i in range(num_of_extremums - 1):
                random_value = derivative_lambda(random.uniform(extremums_axes[i], extremums_axes[i + 1]))
                if random_value > 0:
                    up_ranges.append(
                        Range(expression=x, limits=(extremums_axes[i], extremums_axes[i + 1]),
                              operators=(LESS_THAN, LESS_THAN)))
                elif random_value < 0:
                    down_ranges.append(
                        Range(expression=x, limits=(extremums_axes[i], extremums_axes[i + 1]),
                              operators=(LESS_THAN, LESS_THAN)))
                else:
                    pass

            if after_value > 0:
                up_ranges.append(
                    Range(expression=x, limits=(extremums_axes[-1], np.inf), operators=(LESS_THAN, LESS_THAN)))
            elif after_value < 0:
                down_ranges.append(
                    Range(expression=x, limits=(extremums_axes[-1], np.inf), operators=(LESS_THAN, LESS_THAN)))
            else:
                pass

            return RangeOR(up_ranges), RangeOR(down_ranges)

    def data(self, no_roots=False):
        """
        Get a dictionary that provides information about the polynomial: string, degree, coefficients, roots, extremums, up and down.
        """
        variables = self.variables
        num_of_variables = len(variables)
        my_eval = self.try_evaluate()
        if num_of_variables == 0:
            return {
                "string": self.__str__(),
                "variables": variables,
                "plotDimensions": num_of_variables + 1,
                "coefficients": [my_eval],
                "roots": np.inf if my_eval == 0 else [],
                "y_intersection": my_eval,
                "extremums": [],
                "up": None,
                "down": None,
            }
        elif num_of_variables == 1:
            extremums_axes = self.extremums_axes()
            my_lambda = self.to_lambda()
            my_extremums = [Point2D(x, my_lambda(x)) for x in extremums_axes]
            my_derivative = self.derivative()
            up, down = self.__up_and_down(extremums_axes, my_derivative=my_derivative)
            return {
                "string": self.__str__(),
                "variables": variables,
                "plotDimensions": num_of_variables + 1,
                "coefficients": self.coefficients(),
                "roots": [] if no_roots else self.roots(),
                "y_intersection": my_lambda(0),
                "derivative": my_derivative,
                "extremums": my_extremums,
                "up": up.__str__(),
                "down": down.__str__()
            }
        else:
            return {
                "string": self.__str__(),
                "variables": variables,
                "plotDimensions": num_of_variables + 1,
            }

    def get_report(self, colored=True) -> str:
        if colored:
            accumulator = ""
            for key, value in self.data().items():
                accumulator += f"\033[93m{key}\33[0m: {value.__str__()}\n"
            return accumulator
        return "\n".join(value.__str__() for key, value in self.data().items())

    def _format_report(self, data):
        accumulator = [f"Function: {data['string']}"]
        variables = ", ".join(variable for variable in data["variables"])
        accumulator.append(f"variables: {variables}")
        if len(data['variables']) == 1:
            accumulator.append(f"coefficients: {data['coefficients']}")
            roots = list(data['roots'])
            for index, root in enumerate(roots):
                if isinstance(root, complex):
                    if root.imag < 0.0001:
                        roots[index] = round(root.real, 3)
            roots_string = ", ".join(str(root) for root in roots)
            accumulator.append(f"roots: {roots_string}")
            accumulator.append(f"Intersection with the y axis: {round(data['y_intersection'], 3)}")
            accumulator.append(f"Derivative: {data['derivative']}")
            accumulator.append("Extremums Points:" + ",".join(extremum.__str__() for extremum in data['extremums']))
            accumulator.append(f"Up: {data['up']}")
            accumulator.append(f"Down: {data['down']}")

        return accumulator

    def print_report(self):
        print(self.get_report())

    def export_report(self, path: str, delete_image=True):
        c = Canvas(path)
        c.setFont('Helvetica-Bold', 22)

        c.drawString(50, 800, "Function Report")
        textobject = c.beginText(2 * cm, 26 * cm)
        c.setFont('Helvetica', 16)
        data = self.data()
        variables = ",".join(data['variables'])
        for line in self._format_report(data):
            textobject.textLine(line)
            textobject.textLine("")
        c.drawText(textobject)
        if len(variables) == 1:
            plot_function(f"f({variables}) = {data['string']}", show=False)
        else:
            plot_function_3d(f"f({variables}) = {data['string']}", show=False)
        plt.savefig("tempPlot1146151.png")  # Long path so it won't collide with the user's images accidentally.
        if len(data['variables']) == 1 or len(data['variables']) == 2:
            if len(data['variables']) == 1:
                c.drawInlineImage("tempPlot1146151.png", 50, -215, width=500, preserveAspectRatio=True)
            elif len(data['variables']) == 2:
                c.drawInlineImage("tempPlot1146151.png", 50, 200, width=500, preserveAspectRatio=True)
            if delete_image:
                os.remove("tempPlot1146151.png")
        c.showPage()
        c.save()

    def durand_kerner(self):
        return durand_kerner(self.to_lambda(), self.coefficients())

    def ostrowski(self, initial_value: float, epsilon=0.00001, nmax=10_000):
        return ostrowski_method(self.to_lambda(), self.derivative().to_lambda(), initial_value, epsilon, nmax)

    def laguerres(self, x0: float, epsilon=0.00001, nmax=100000):
        my_derivative = self.derivative()
        second_derivative = self.derivative().to_lambda()
        return laguerre_method(self.to_lambda(), my_derivative.to_lambda(), second_derivative, x0, epsilon, nmax)

    def halleys(self, initial_value=0, epsilon=0.00001, nmax=10_000):  # TODO: check if works
        """
        Halley's method is a root finding method developed by Edmond Halley for functions with continuous second
        derivatives and a single variable.
        :param initial_value:
        :param epsilon:
        :return:
        """
        f_0 = self
        f_1 = f_0.derivative()
        f_2 = f_1.derivative()

        f_0 = self.to_lambda()
        f_1 = f_1.to_lambda()
        f_2 = f_2.to_lambda()
        return halleys_method(f_0, f_1, f_2, initial_value, epsilon, nmax)

    def newton(self, initial_value=0, epsilon=0.00001, nmax=10_000):
        return newton_raphson(self.to_lambda(), self.derivative().to_lambda(), initial_value, epsilon, nmax)

    def __str__(self):
        if len(self._expressions) == 1:
            return self._expressions[0].__str__()
        accumulator = ""
        for index, expression in enumerate(self._expressions):
            accumulator += '+' if expression.coefficient >= 0 and index > 0 else ""
            accumulator += expression.__str__()
        return accumulator

    def to_dict(self):
        if not self._expressions:
            return {'type': 'Poly', 'data': None}
        return {'type': 'Poly', 'data': [item.to_dict() for item in self._expressions]}

    @staticmethod
    def from_dict(given_dict: dict):
        return Poly([Mono.from_dict(sub_dict) for sub_dict in given_dict['data']])

    @staticmethod
    def from_json(json_content: str):  # Check this method
        parsed_dictionary = json.loads(json_content)
        if parsed_dictionary['type'].strip().lower() != "poly":
            return ValueError(f"Invalid type: {parsed_dictionary['type']}. Expected 'Poly'. ")
        return Poly(Mono.from_dict(mono_dict) for mono_dict in parsed_dictionary['data'])

    @staticmethod
    def import_json(path: str):
        with open(path) as json_file:
            return Poly.from_json(json_file.read())

    def python_syntax(self):
        accumulator = ""
        for index, expression in enumerate(self._expressions):
            accumulator += '+' if expression.coefficient >= 0 and index > 0 else ""
            accumulator += expression.python_syntax()
        return accumulator

    def __fetch_variables_set(self) -> set:
        return {json.dumps(mono_expression.variables_dict) for mono_expression in self._expressions}

    def simplify(
            self):  # TODO: create a new way to simplify polynomials, without re-creating it each time with overhead
        """ simplifying a polynomial"""
        if len(self._expressions) == 0:
            self._expressions = [Mono(0)]
            return
        different_variables: set = self.__fetch_variables_set()
        if "{}" in different_variables:  # TODO: find the source of this stupid bug.
            different_variables.remove("{}")
            different_variables.add("null")
        new_expressions = []
        for variable_dictionary in different_variables:
            if variable_dictionary == 'null':
                same_variables = [expression for expression in self._expressions if
                                  json.dumps(expression.variables_dict) in ("null", "{}")]
            else:
                same_variables = [expression for expression in self._expressions if
                                  json.dumps(expression.variables_dict) == variable_dictionary]

            if len(same_variables) > 1:
                # TODO: BUG ? MONO EXPRESSIONS SOMEHOW GOT MONO COEFFICIENT?
                assert all(
                    isinstance(same_variable.coefficient, (int, float)) for same_variable in
                    same_variables), "Bug detected.."
                coefficients_sum: float = sum(same_variable.coefficient for same_variable in same_variables)
                if coefficients_sum != 0:
                    new_expressions.append(
                        Mono(coefficient=coefficients_sum, variables_dict=same_variables[0].variables_dict))
            elif len(same_variables) == 1:
                if same_variables[0].coefficient != 0:
                    new_expressions.append(same_variables[0])
        self._expressions = new_expressions
        self.sort()

    def sorted_expressions_list(self) -> list:
        """

        :return:
        """
        sorted_exprs = sorted_expressions(
            [expression for expression in self._expressions if expression.variables_dict not in (None, {})])
        free_number = sum(
            expression.coefficient for expression in self._expressions if expression.variables_dict in (None, {}))
        if free_number != 0:  # No reason to add a trailing zero
            sorted_exprs.append(Mono(free_number))
        return sorted_exprs

    def sort(self):
        """
        sorts the polynomial's expression by power, for example : 6 + 3x^2 + 2x  -> 3x^2 + 2x + 6

        :return:
        """
        self._expressions = self.sorted_expressions_list()

    def sorted(self):
        """

        :return:
        """
        return Poly(self.sorted_expressions_list())

    def __len__(self):
        return len(self.expressions)

    def contains_variable(self, variable: str) -> bool:
        """
        Checking whether a certain given variable appears in the expression. For example 'x' does appear in 3x^2 + 5

        :param variable: The variable to be looked for ( type str ). For example : 'x', 'y', etc.
        :return: Returns True if the variable appears in the expression, otherwise False.
        """
        for mono_expression in self._expressions:
            if mono_expression.contains_variable(variable):  # It only has to appear in at least 1 Mini-IExpression
                return True
        return False

    def __contains__(self, item):
        """
        Determines whether a Poly contains a certain value. for example, 3x^2+5x+7 contains 5x, but doesn't contain 8.
        :param item: allowed types: int,float,str,Mono,Poly
        :return:
        """
        if isinstance(item, (int, float, str)):
            item = Mono(item)

        if isinstance(item, Mono):
            return item in self._expressions
        elif isinstance(item, Poly):
            if len(self.expressions) < len(item._expressions):  # Meaning it's smaller and thus can't contain the items
                return False
            return all(item in self.expressions for item in item._expressions)  # if it contains all items, return True
        else:
            raise TypeError(
                f"Poly.__contains__(): unexpected type {type(item)}, expected types: int,float,str,Mono,Poly")

    def __copy__(self):
        return Poly([expression.__copy__() for expression in self._expressions])

    def to_lambda(self):
        """ Returns a lambda expression from the Polynomial"""
        return to_lambda(self.__str__(), self.variables)

    def plot(self, start: float = -10, stop: float = 10,
             step: float = 0.01, ymin: float = -10, ymax: float = 10, text=None, show_axis=True, show=True,
             fig=None, ax=None, formatText=True, values=None):
        lambda_expression = self.to_lambda()
        num_of_variables = self.num_of_variables
        if text is None:
            text = self.__str__()
        if num_of_variables == 0:  # TODO: plot this in a number axis
            raise ValueError("Cannot plot a polynomial with 0 variables_dict")
        elif num_of_variables == 1:
            plot_function(lambda_expression, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=text,
                          show_axis=show_axis, show=show, fig=fig, ax=ax, formatText=formatText, values=values)
        elif num_of_variables == 2:
            plot_function_3d(lambda_expression, start=start, stop=stop, step=step)  # TODO: update the parameters
        else:
            raise ValueError("Cannot plot a function with more than two variables_dict (As for this version)")

    def to_Function(self):
        return Function(self.__str__())

    def gcd(self):
        """Greatest common divisor of the expressions: for example, for the expression 3x^2 and 6x,
        the result would be 3x"""
        gcd_coefficient = gcd(self.coefficients())
        if any(not expression.variables_dict for expression in self._expressions):  # If there's a free number
            return Mono(gcd_coefficient)
        gcd_algebraic = Mono(gcd_coefficient)
        my_variables = self.variables
        for variable in my_variables:
            if all(variable in expression.variables_dict for expression in self._expressions):
                powers = [expression.variables_dict[variable] for expression in self._expressions]
                if gcd_algebraic.variables_dict is not None:
                    gcd_algebraic.variables_dict = {**gcd_algebraic.variables_dict, **{variable: min(powers)}}
                else:
                    gcd_algebraic.variables_dict = {variable: min(powers)}
        return gcd_algebraic

    def divide_by_gcd(self):
        return self.__itruediv__(self.gcd())


def poly_frac_from_str(expression: str, get_tuple=False):
    # TODO: implement it better concerning parenthesis and '/' in later versions
    """
    Generates a PolyFraction object from a given string

    :param expression: The given string that represents a polynomial fraction
    :param get_tuple : If set to True, the a tuple of length 2 with the numerator at index 0 and the denoominator at index 1 will be returned.
    :return: Returns a new PolyFraction object, unless get_tuple is True, and then returns the corresponding tuple.
    """
    first_expression, second_expression = expression.split('/')
    if get_tuple:
        return Poly(first_expression), Poly(second_expression)
    return PolyFraction(Poly(first_expression), Poly(second_expression))


class Fraction(IExpression):  # Does everything that inherit from IExpression will be accepted here ?
    __slots__ = ['_numerator', '_denominator']

    def __init__(self, numerator: "Union[IExpression,float,int]",
                 denominator: "Optional[Union[IExpression,float,int]]" = None, gen_copies=True):
        # Handle the numerator
        if isinstance(numerator, (float, int)):
            self._numerator = Mono(numerator)
        elif isinstance(numerator, IExpression):
            self._numerator = numerator.__copy__() if gen_copies else numerator
        else:
            raise TypeError(f"Unexpected type {type(numerator)} in Fraction.__init__."
                            f"Modify the type of the numerator parameter to a valid one.")

        if denominator is None:
            self._denominator = Mono(1)
            return
        # Handle the denominator
        if isinstance(denominator, (float, int)):  # Create a Mono object instead of ints and floats
            self._denominator = Mono(denominator)
        elif isinstance(denominator, IExpression):
            self._denominator = denominator.__copy__() if gen_copies else denominator
        else:
            raise TypeError(f"Unexpected type {type(denominator)} in Fraction.__init__. Modify the type of the"
                            f"denominator parameter to a valid one")

    @property
    def numerator(self):
        return self._numerator

    @property
    def denominator(self):
        return self._denominator

    @property
    def variables(self):
        return self._numerator.variables_dict.union(self._denominator.variables_dict)

    def assign(self, **kwargs):
        self._numerator.assign(**kwargs)
        self._denominator.assign(**kwargs)

    def derivative(self):
        return (self._numerator.derivative() * self._denominator - self._numerator * self._denominator.derivative) \
               / self._denominator ** 2

    def integral(self):  # TODO: got no idea how
        pass

    def simplify(self):  # TODO: try to divide the numerator and denominator and check whether it can be done
        pass  # TODO: how to check whether the division is successful..

    def try_evaluate(self) -> Optional[Union[int, float]]:
        """ try to evaluate the expression into a float or int value, if not successful, return None"""
        numerator_evaluation = self._numerator.try_evaluate()
        denominator_evaluation = self._denominator.try_evaluate()
        if denominator_evaluation is None:
            if self._numerator == 0:
                return 0

        if denominator_evaluation == 0:
            raise ZeroDivisionError(f"Denominator of fraction {self.__str__()} was evaluated into 0. Cannot divide "
                                    f"by 0.")
        if None not in (numerator_evaluation, denominator_evaluation):
            return numerator_evaluation / denominator_evaluation
        division_result = (self._numerator / self._denominator)
        if isinstance(division_result, Fraction):  # bug fix: preventing a recursive endless loop .....
            return None
        division_evaluation = division_result.try_evaluate()
        if division_evaluation is not None:
            return division_evaluation
        return None

    def to_dict(self):
        return {
            "type": "Fraction",
            "numerator": self._numerator.to_dict(),
            "denominator": self._denominator.to_dict() if self._denominator is not None else None
        }

    @staticmethod
    def from_dict(given_dict: dict):
        numerator_obj = create_from_dict(given_dict['numerator'])
        denominator_obj = create_from_dict(given_dict['denominator'])
        return Fraction(numerator=numerator_obj, denominator=denominator_obj)

    def __iadd__(self, other: Union[IExpression, int, float]):
        #  TODO: add simplifications for expressions that can be evaluated like Log(5) for instance
        if isinstance(other, (int, float)):  # If we're adding a number
            my_evaluation = self.try_evaluate()
            if my_evaluation is not None:  # If the fraction can be evaluated into number
                return Mono(coefficient=my_evaluation + other)
            else:
                return ExpressionSum((self, Mono(coefficient=other)))
        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            my_evaluation = self.try_evaluate()
            if None not in (other_evaluation, my_evaluation):  # Both of the expressions can be evaluated into numbers
                return Mono(coefficient=my_evaluation + other_evaluation)
            if isinstance(other, ExpressionSum):
                copy_of_other = other.__copy__()
                copy_of_other += self
                return copy_of_other
            elif isinstance(other, Fraction):  # TODO: try to improve this section with common denominator?
                if self._denominator == other._denominator:  # If the denominators are equal, just add the numerator.
                    self._numerator += other._numerator
                else:
                    return ExpressionSum((self, other))

            else:  # Other types of IExpression that don't need special-case handling.
                return ExpressionSum((self, other))

        else:
            raise TypeError(f"Invalid type '{type(other)}' for addition with fractions")

    def __isub__(self, other: Union[IExpression, int, float]):
        return self.__iadd__(-other)

    def __neg__(self):
        copy_of_self = self.__copy__()
        copy_of_self._numerator *= -1
        return copy_of_self

    def __imul__(self, other: Union[IExpression, int, float]):
        if isinstance(other, Fraction):
            self._numerator *= other._numerator
            self._denominator *= other._denominator
            return self
        if self._denominator == other:
            self._denominator = Mono(1)
            my_evaluation = self.try_evaluate()
            if my_evaluation is not None:
                return Mono(my_evaluation)
            return self._numerator
        self._numerator *= other
        return self

    def __mul__(self, other: Union[IExpression, int, float]):
        return self.__copy__().__imul__(other)

    def __itruediv__(self, other: Union[IExpression, int, float]):
        if isinstance(other, Fraction):
            self._numerator *= other._denominator
            self._denominator *= other._numerator
        else:
            self._denominator *= other
        return self

    def __rmul__(self, other: Union[IExpression, int, float]):
        return self.__copy__().__imul__(other)

    def __ipow__(self, other: Union[IExpression, int, float]):
        self._numerator **= other
        self._denominator **= other
        self.simplify()
        return self

    def __rpow__(self, other):
        return Exponent(self, other)

    def __copy__(self):
        return Fraction(self._numerator, self._denominator)

    def __eq__(self, other: Union[IExpression, int, float]) -> Optional[bool]:
        if other is None:
            return False
        numerator_evaluation = self._numerator.try_evaluate()
        if numerator_evaluation == 0:
            return other == 0
        denominator_evaluation = self._denominator.try_evaluate()
        if denominator_evaluation == 0:  # making sure a ZeroDivisionError won't occur somehow
            raise ValueError(f"Denominator of a fraction cannot be 0.")
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):  # if the parameter is a number
            if my_evaluation is not None:
                return my_evaluation == other
            return None  # Algebraic expression isn't equal to a free number

        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):  # Both expressions can be evaluated into numbers
                return my_evaluation == other_evaluation
            if isinstance(other, Fraction):
                if self._numerator == other._numerator and self._denominator == other._denominator:
                    return True
                # Won't reach here if any of them is zero, so no reason to worry about ZeroDivisionError
                # Check for cases such as 0.2x / y and x / 5y , which are the same.
                numerator_ratio = self._numerator / other._numerator
                denominator_ratio = self._denominator / other._denominator
                return numerator_ratio == denominator_ratio
            else:
                pass  # Implement it ..

    def __ne__(self, other: Union[IExpression, int, float]):
        return not self.__eq__(other)

    def python_syntax(self) -> str:
        return f"({self._numerator.python_syntax()})/({self._denominator.python_syntax()})"

    def __str__(self):
        return f"({self._numerator.__str__()})/({self._denominator.__str__()})"


class PolyFraction(Fraction):
    """
    Creating a new algebraic fraction with a polynomial numerator and denominator.
    In later version, further types of expressions will be allowed in fractions.
    """

    def __init__(self, numerator, denominator=None, gen_copies=True):
        if denominator is None:  # if the user chooses to enter only one parameter in the constructor
            if isinstance(numerator, str):  # in a case of a string in the format : (...) / (...)
                numerator1, denominator1 = poly_frac_from_str(numerator, get_tuple=True)
                super().__init__(numerator1, denominator1)
            elif isinstance(numerator, PolyFraction):
                super().__init__(
                    numerator._numerator.__copy__() if gen_copies else numerator._numerator,
                    numerator._denominator.__copy__() if gen_copies else numerator._denominator)
            elif isinstance(numerator, (int, float, Mono, Poly)):
                super().__init__(Poly(numerator), Mono(1))
            else:
                raise TypeError(f"Invalid type for a numerator in PolyFraction : {type(numerator)}.")

        else:  # if the user chooses to specify both parameter in the constructor.
            if isinstance(numerator, Poly):  # Handling the numerator
                numerator = numerator.__copy__()
            elif isinstance(numerator, (int, float, str, Mono)):
                numerator = Poly(numerator)
            else:
                raise TypeError(f"Invalid type for a numerator in PolyFraction : {type(numerator)}. Expected types "
                                f" Poly, Mono, str , float , int")

            if isinstance(denominator, Poly):  # Handling the denominator
                denominator = denominator.__copy__()
            elif isinstance(denominator, (int, float, str, Mono)):
                denominator = Poly(denominator)
            else:
                raise TypeError(f"Invalid type for a denominator in PolyFraction : {type(denominator)}. Expected types "
                                f" Poly, Mono, str , float , int")
            super().__init__(numerator, denominator)

    def roots(self, epsilon: float = 0.000001, nmax: int = 100000):
        return self._numerator.roots(epsilon, nmax)

    def invalid_values(self):
        """ When the denominator evaluates to 0"""
        return self._denominator.roots()  # TODO: hopefully it works..

    def horizontal_asymptote(self):  # TODO: what about multiple asymptotes ?
        power1, power2 = self._numerator.expressions[0].highest_power(), \
                         self._denominator.expressions[0].highest_power()
        if power1 > power2 or power1 == power2 == 0:
            return tuple()
        if power1 < power2:
            return 0
        return power1 / power2,

    def __str__(self):
        return f"({self._numerator})/({self._denominator})"

    def __repr__(self):
        return f"PolyFraction({self._numerator.__str__()},{self._denominator.__str__()})"

    def __iadd__(self, other):
        if other == 0:
            return self
        if isinstance(other, PolyFraction):
            if self._denominator == other._denominator:
                self._numerator += other._numerator
                return self
            elif (division_result := self._denominator.__truediv__(other._denominator, get_remainder=True))[
                1] == 0:  # My denominator is bigger
                self._numerator += other._numerator * division_result[0]
                return self
            elif (division_result := other._denominator / self._denominator)[1] == 0:  # My denominator is smaller
                self._numerator *= division_result[0]
                self._denominator *= division_result[0]
                self._numerator += other._numerator
                return self
            else:  # There is no linear connection between the two denominators
                raise NotImplemented
        else:
            raise NotImplemented

    def __radd__(self, other):
        new_copy = self.__copy__()
        return new_copy.__iadd__(other)

    def __isub__(self, other):
        if isinstance(other, PolyFraction):
            if self._denominator == other._denominator:
                self._numerator -= other._numerator
                return self
            elif (division_result := self._denominator / other._denominator)[1] == 0:  # My denominator is bigger
                self._numerator -= other._numerator * division_result[0]
                return self
            elif (division_result := other._denominator / self._denominator)[1] == 0:  # My denominator is smaller
                self._numerator *= division_result[0]
                self._denominator *= division_result[0]
                self._numerator -= other._numerator
                return self
            else:  # There is no linear connection between the two denominators
                raise NotImplemented
        else:
            raise NotImplemented

    def __sub__(self, other):
        new_copy = self.__copy__()
        return new_copy.__isub__(other)

    def __rsub__(self, other):  # TODO: does this even work ??
        new_copy = self.__copy__()
        new_copy.__isub__(other)
        new_copy.__imul__(-1)

    def __imul__(self, other):
        if isinstance(other, PolyFraction):
            self._numerator *= other._numerator
            self._denominator *= other._denominator
            return self
        elif isinstance(other, (int, float, Mono, Poly)):
            self._numerator *= other
            return self
        else:
            raise TypeError(f"Invalid type {type(other)} for multiplying PolyFraction objects. Allowed types: "
                            f" PolyFraction, Mono, Poly, int, float")

    def __mul__(self, other):
        new_copy = self.__copy__()
        return new_copy.__imul__(other)

    def __rmul__(self, other):
        new_copy = self.__copy__()
        new_copy.__imul__(other)
        return new_copy

    def __rtruediv__(self, other):
        inverse_fraction: PolyFraction = self.reciprocal()
        return inverse_fraction.__imul__(other)

    def reciprocal(self):
        return PolyFraction(self._denominator, self._numerator)

    def __copy__(self):
        """Create a new copy of the polynomial fraction"""
        return PolyFraction(self._numerator, self._denominator)


class Root(IExpression, IPlottable, IScatterable):
    __slots__ = ['_coefficient', '_inside', '_root']

    def __init__(self, inside: Union[IExpression, float, int], root_by: Union[IExpression, float, int] = 2
                 , coefficient: Union[int, float, IExpression] = Mono(1)):
        self._coefficient = process_object(coefficient,
                                           class_name="Root", method_name="__init__", param_name="coefficient")
        self._inside = process_object(inside,
                                      class_name="Root", method_name="__init__", param_name="inside")
        self._root = process_object(root_by,
                                    class_name="Root", method_name="__init__", param_name="root_by")

    @property
    def coefficient(self):
        return self._coefficient

    @property
    def inside(self):
        return self._inside

    @property
    def root(self):
        return self._root

    def assign(self, **kwargs):
        self._coefficient.assign(**kwargs)
        self._inside.assign(**kwargs)
        self._root.assign(**kwargs)

    def try_evaluate(self) -> Optional[Union[complex, float, ValueError]]:
        coefficient_evaluation = self._coefficient.try_evaluate()
        if coefficient_evaluation == 0:
            return 0
        inside_evaluation = self._inside.try_evaluate()
        root_evaluation = self._root.try_evaluate()
        if None not in (coefficient_evaluation, inside_evaluation, root_evaluation):
            if root_evaluation == 0:
                return ValueError("Cannot compute root by 0")
            return coefficient_evaluation * inside_evaluation ** (1 / root_evaluation)
        return None

    def simplify(self) -> None:
        self._coefficient.simplify()
        self._root.simplify()
        self._inside.simplify()

    @property
    def variables(self):
        variables = self._coefficient.variables
        variables.update(self._inside.variables)
        variables.update(self._root.variables)
        return variables

    def to_dict(self):
        return {
            "type": "Root",
            "coefficient": self._coefficient.to_dict(),
            "inside": self._inside.to_dict(),
            "root_by": self._root.to_dict()
        }

    @staticmethod
    def from_dict(given_dict: dict):
        coefficient_obj = create_from_dict(given_dict['coefficient'])
        inside_obj = create_from_dict(given_dict['inside'])
        root_obj = create_from_dict(given_dict['root_by'])
        return Root(coefficient=coefficient_obj, inside=inside_obj, root_by=root_obj)

    @staticmethod
    def dependant_roots(first_root: "Root", second_root: "Root") -> Optional[Tuple[IExpression, str]]:
        if first_root._root != second_root._root:
            return None
        result = first_root._inside.__truediv__(second_root._inside)  # If the second root is the common denominator
        if isinstance(result, Fraction) or result is None:
            return None
        if isinstance(result, tuple):
            result, remainder = result
            if remainder == 0:
                return result, "first"  # the first is bigger
            return None
        result = second_root._inside.__truediv__(first_root._inside)
        if isinstance(result, Fraction) or result is None:
            return None
        if isinstance(result, tuple):
            result, remainder = result
            if remainder == 0:
                return result, "second"  # the first is bigger
            return None
        return result, "second"

    def __iadd__(self, other: Union[IExpression, float, int, str]):
        if other == 0:
            return self
        if isinstance(other, IExpression):  # If the expression is IExpression
            if isinstance(other, Root):  # if it's a another root
                division_result: Optional[IExpression] = Root.dependant_roots(self, other)
                if division_result is not None:
                    root_evaluation = self._root.try_evaluate()
                    if division_result[1] == "first":
                        other_copy = other.__copy__()
                        other_copy._coefficient = Mono(1)  # later maybe make it common denominator too
                        division_result = division_result[0]
                        if root_evaluation is not None:
                            return other_copy * (division_result ** (
                                    1 / root_evaluation) + other_copy._coefficient * self._coefficient)
                    else:  # The second is bigger
                        division_result = division_result[0]
                        if root_evaluation is not None:
                            self_copy = self.__copy__()
                            self_copy._coefficient = Mono(1)
                            return self * (division_result ** (
                                    1 / root_evaluation) + self._coefficient * other._coefficient)
        if isinstance(other, (int, float)):
            other = Mono(other)
        return ExpressionSum((self, other))  # If it's not another root

    def __isub__(self, other):
        if other == 0:
            return self
        return self.__iadd__(-other)  # TODO: Should I create here too a separate implementation ?

    def multiply_by_root(self, other: "Root"):
        other_evaluation = other.try_evaluate()
        if other_evaluation is not None:  # If the root can be evaluated into a number, such as Root of 2:
            self._coefficient *= other_evaluation
            return self
        if self._root == other._root:
            self._inside *= other._inside
            return self
        else:
            return ExpressionMul((self, other))

    def __imul__(self, other: Union[IExpression, float, int, str]):
        if isinstance(other, (int, float)):
            self._coefficient *= other
            self.simplify()
            return self
        if isinstance(other, str):  # TODO: implement a kind of string-processing method
            pass

        if isinstance(other, IExpression):
            if isinstance(other, Root):
                return self.multiply_by_root(other)
            else:
                self._coefficient *= other
                return self

        return TypeError(f"Invalid type {type(other)} for multiplying roots.")

    def __mul__(self, other: Union[int, float, IExpression]):
        return self.__copy__().__imul__(other)

    def __rmul__(self, other: Union[int, float, IExpression]):
        return self.__copy__().__mul__(other)

    def __neg__(self):
        copy_of_self = self.__copy__()
        copy_of_self._coefficient *= -1
        return copy_of_self

    def __ipow__(self, other: Union[int, float, IExpression]):
        if other == 1:
            return self
        if other == 0:
            pass  # TODO: Return Mono(1) or change the current object somehow?
        root_division = self._root / other  # A good start. Will be developed later
        if isinstance(root_division, IExpression):
            evaluated_division = root_division.try_evaluate()
            if evaluated_division is None:
                self._root = root_division
                return self
        elif isinstance(root_division, (int, float)):
            evaluated_division = root_division
        else:
            raise TypeError(f"Invalid type '{type(other)} when dividing Root objects.'")
        if 0 < evaluated_division < 1:
            return self._inside ** (1 / evaluated_division)
        elif evaluated_division == 1:
            return self._inside
        self._root = evaluated_division
        return self

    def __pow__(self, power):
        return self.__copy__().__ipow__(power)

    def __itruediv__(self, other: Union[int, float, IExpression]):
        if other == 0:
            return ZeroDivisionError("Cannot divide a Root object by 0")
        if isinstance(other, (int, float)):
            self._coefficient /= other
            return self
        else:  # Other is of type IExpression
            other_evaluation = other.try_evaluate()
            if other_evaluation is not None:  # The other argument can be evaluated into an int or a float
                self._coefficient /= other_evaluation
                return self
            if isinstance(other, Root):  # If we're dividing by another root
                if self._root == other._root:  # the other expression has the same root number
                    if self == other:  # The two expressions are equal
                        return Mono(1)
                elif self._inside == other._inside:
                    # Different roots, but same inside expressions and thus it can be evaluated ..
                    my_root_evaluation = self._root.try_evaluate()
                    other_root_evaluation = other._root.try_evaluate()
                    if my_root_evaluation and other_root_evaluation:
                        # Both roots can be evaluated into numbers, and not 0 ( 0 is false )
                        self._coefficient /= other._coefficient
                        power_difference = (1 / my_root_evaluation) - (1 / other_root_evaluation)
                        self._root = 1 / power_difference
                        return self
            else:
                return Fraction(self, other)

            return Fraction(self, other)

    def __copy__(self):
        return Root(
            inside=self._inside,
            root_by=self._root,
            coefficient=self._coefficient
        )

    def __str__(self):
        if self._coefficient == 0:
            return "0"
        if self._coefficient == 1:
            coefficient = ""
        elif self._coefficient == -1:
            coefficient = '-'
        else:
            coefficient = f"{self._coefficient} * "
        root = f"{self._root}^" if self._root != 2 else ""
        return f"{coefficient}{root}√({self._inside})"

    def __eq__(self, other: Union[IExpression, int, float]):
        """ Compare between a Root object and other expressions"""
        if other is None:
            return False
        if isinstance(other, (int, float)):
            my_evaluation = self.try_evaluate()
            print(my_evaluation)
            return my_evaluation == other
        if isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            my_evaluation = self.try_evaluate()
            if None not in (other_evaluation, my_evaluation):  # Both can be evaluated
                return my_evaluation == other_evaluation
            if (my_evaluation, other_evaluation) == (None, None):  # None can be evaluated
                if isinstance(other, Root):  #
                    if self._coefficient == other._coefficient and self._inside == other._inside and self._root == other._root:
                        return True
                    return False  # TODO: handle cases in which it will be true, considering the coefficient and stuff
        return False  # Can be wrong ?

    def __ne__(self, other: Union[IExpression, int, float]):
        return not self.__eq__(other)

    def derivative(self):

        if self._inside is None:
            return 0
        my_evaluation = self.try_evaluate()
        if my_evaluation is not None:  # If the expression can be evaluated into number, the derivative is 0.
            if my_evaluation < 0:
                warnings.warn("Root evaluated to negative result. Complex Analysis is yet to be supported "
                              "in this version")
            return 0
        if self._coefficient == 0:
            return 0  # If the coefficient is 0 then it's gg, the whole expression is 0
        coefficient_evaluation = self._coefficient.try_evaluate()
        root_evaluation = self._root.try_evaluate()
        inside_evaluation = self._inside.try_evaluate()
        if None not in (coefficient_evaluation, root_evaluation, inside_evaluation):  # everything can be evaluated ..
            return 0
        inside_variables = self._inside.variables
        if None not in (coefficient_evaluation, root_evaluation) and len(inside_variables) == 1:
            # if the coefficient and the root can be evaluated into free numbers, and only one variable ...
            new_power = (1 / root_evaluation) - 1
            new_root = 1 / new_power
            inside_derivative = self._inside.derivative()  # Might not always work ..
            if new_power > 1:
                monomial = Mono(coefficient=coefficient_evaluation, variables_dict={inside_variables: new_power})
                monomial *= inside_derivative
                return monomial

            elif new_power == 0:  # then the inside expression is 1, and it's multiplied by the coefficient
                inside_derivative *= coefficient_evaluation
                return inside_derivative
            else:
                if new_root == 1:
                    return coefficient_evaluation * self._inside
                inside_derivative *= coefficient_evaluation
                if new_root < 0:
                    return Fraction(
                        numerator=inside_derivative,
                        denominator=
                        Root(
                            coefficient=1,
                            root_by=abs(new_root),
                            inside=self._inside.__copy__()
                        )
                    )
                else:
                    return Root(
                        coefficient=inside_derivative,
                        root_by=new_root,
                        inside=self._inside.__copy__()
                    )


        else:  # Handling more complex derivatives
            pass

    def integral(self):
        pass

    def python_syntax(self) -> str:  # Create a separate case for the Log class
        """ Returns a string that can be evaluated using the eval() method to actual objects from the class, if
        imported properly
        """
        if isinstance(self._coefficient, Log):
            coefficient_str = self._coefficient.python_syntax()
        else:
            coefficient_str = self._coefficient.__str__()

        if isinstance(self._inside, Log):
            inside_str = self._inside.python_syntax()
        else:
            inside_str = self._inside.__str__()

        if isinstance(self._root, Log):
            root_str = self._root.python_syntax()
        else:
            root_str = self._root.__str__()

        return f"{coefficient_str}*({inside_str}) ** (1/{root_str})"


class Sqrt(Root):  # A class for clarity purposes, so it's clear when someone is using a square root
    def __init__(self, inside: Union[IExpression, float, int], coefficient: Union[int, float, IExpression] = Mono(1)):
        super(Sqrt, self).__init__(inside=inside, root_by=2, coefficient=coefficient)


def log_from_str(expression: str, get_tuple=False, dtype: str = 'poly'):
    expression = expression.strip().lower()  # Make sure the string entered is lower case and without spaces
    if "log" in expression or "ln" in expression:

        # STEP 1 : Extract the _coefficient
        coefficient = expression[:expression.find('l')]
        if coefficient == '':
            coefficient = 1
        elif coefficient == '-':
            coefficient = -1
        else:
            try:
                coefficient = float(coefficient)  # Add support on coefficients such as e and pi here perhaps later
            except ValueError:  # Raise an appropriate message to the user, so he knows his mistake. Is this verbose?
                raise ValueError(f"Invalid _coefficient '{coefficient}' in expression {expression}, while creating"
                                 f"a PolyLog object from a given string.")
        #  STEP 2: Extract the polynomial and the base
        start_parenthesis = expression.find('(')
        if start_parenthesis == -1:
            raise ValueError(F"Invalid string '{expression}' without opening parenthesis for the expression.")
        ending_parenthesis = expression.find(')')
        if ending_parenthesis == -1:
            raise ValueError(f"Invalid string: '{ending_parenthesis} without ending parenthesis for the expression'")
        if "log" in expression:
            inside = expression[start_parenthesis + 1:ending_parenthesis]
            if ',' in inside:
                inside, base = inside.split(',')
                base = float(base)
            else:
                base = 10
            inside = create(inside, dtype=dtype)
        else:
            base = 'e'
            inside = create(expression[start_parenthesis + 1:ending_parenthesis], dtype=dtype)

        # STEP 3: Extract the power
        power_index = expression.find('^')
        if power_index == -1:
            power_index = expression.find('**')
        if power_index == -1:  # In case no sign of power was found, the default is 1
            power = 1
        else:
            close_parenthesis_index = expression.rfind(')')
            if power_index > close_parenthesis_index:
                power = float(expression[power_index + 1:])
            else:
                power = 1

        if get_tuple:
            return coefficient, inside, base, power
        return Log(expression=[[inside, base, power]], coefficient=coefficient)
    else:
        raise ValueError("The string need to contain log() or ln()")


class Log(IExpression, IPlottable, IScatterable):
    __slots__ = ['_coefficient', '_expressions']

    def __init__(self, expression, base: Union[int, float] = 10, coefficient: Union[IExpression, int, float] = 1,
                 dtype='poly', gen_copies=True):

        if isinstance(expression, (int, float)):
            if expression < 0:
                raise ValueError("Negative logarithms are not defined in this program.")
            elif expression == 0 and base > 0:
                raise ValueError(f"{expression}^n cannot be 0 for any real n, Thus the expression isn't defined!")
            else:
                self._coefficient = Mono(log(expression, base))
                self._expressions = []
                return

        elif isinstance(expression, str):
            result = log_from_str(expression, get_tuple=True, dtype=dtype)
            coefficient, expression = result[0], [[result[1], result[2], result[3]]]

        elif isinstance(expression, IExpression):
            if isinstance(coefficient, (int, float)):
                self._coefficient = Mono(coefficient)
            elif isinstance(coefficient, IExpression):
                self._coefficient = coefficient.__copy__() if gen_copies else coefficient
            else:
                raise TypeError(f"Log.__init__(): invalid type for coefficient: {type(coefficient)}")
            self._expressions = [[expression.__copy__() if gen_copies else expression, base, 1]]
            return

        if isinstance(expression, List) and len(expression) and isinstance(expression[0], list):
            self._expressions = []
            for inner_list in expression:
                self._expressions.append([inner_list[0].__copy__(), inner_list[1], inner_list[2]])

        else:
            self._expressions: List[List[IExpression, Union[float, int], Union[int, float]]] = [
                list(expression) for expression in
                expression]  # inside, base, power
        self._coefficient = coefficient.__copy__() if isinstance(coefficient, IExpression) else Mono(coefficient)

    @property
    def coefficient(self):
        return self._coefficient

    def index_of(self, other_list):
        """ """
        if other_list in self._expressions:
            return self._expressions.index(other_list)
        return -1

    def all_bases(self) -> Set[float]:
        return {inner_list[1] for inner_list in self._expressions}

    def biggest_power(self) -> float:
        return max((inner_list[2] for inner_list in self._expressions))

    @property
    def variables(self):
        variables = set()
        for (inside, base, power) in self._expressions:
            variables.update(inside.variables_dict)
        return variables

    def simplify(self):  # TODO: further implementation needed
        self._coefficient.simplify()
        for (inside, base, power) in self._expressions:
            if hasattr(inside, "simplify"):
                inside.simplify()
            if hasattr(base, "simplify"):
                base.simplify()
            if hasattr(power, "simplify"):
                power.simplify()

    def __iadd__(self, other: "Union[int,float,IExpression]"):
        if other == 0:
            return self
        if isinstance(other, (int, float)):
            if not self._expressions:  # Meaning that the current object basically represents a free number
                self._coefficient += other
                return self
            return ExpressionSum(expressions=(self, Mono(other)))
        elif isinstance(other, Log):
            my_eval = self.try_evaluate()
            other_eval = other.try_evaluate()
            if None not in (my_eval, other_eval):
                return Mono(my_eval + other_eval)
            if my_eval is None:
                pass
            elif other_eval is None:
                pass
            if len(self._expressions) == 1 == len(other._expressions) and not isinstance(self._coefficient, Log) \
                    and not isinstance(other._coefficient, Log) and self._expressions[0][1] == other._expressions[0][1]:
                if self._coefficient == other._coefficient:  # The coefficient stays the same
                    self._expressions[0][0] *= other._expressions[0][0]
                    return self
                else:
                    try:
                        self._expressions[0][0] *= other._expressions[0][0] ** other._expressions[0][2]
                    except (TypeError, ValueError):
                        return ExpressionSum((self, other))
            for inner_list in other._expressions:
                self._expressions.append([inner_list[0].__copy__(), inner_list[1], inner_list[2]])  # Appending the
                # other's items to our items
            return self
        else:
            return ExpressionSum((self, other))

    def __radd__(self, other):
        return self.__copy__().__iadd__(other)

    def __isub__(self, other):
        if isinstance(other, (int, float)):
            if not self._expressions:  # Meaning that the current object basically represents a free number
                self._coefficient -= other
                return self
            return ExpressionSum(expressions=(self, Mono(other)))
        elif isinstance(other, Log):
            if len(self._expressions) == 1 == len(other._expressions) and not isinstance(self._coefficient, Log) \
                    and not isinstance(other._coefficient, Log) and self._expressions[0][1] == other._expressions[0][1]:
                if self._coefficient == other._coefficient:  # The coefficient stays the same
                    self._expressions[0][0] /= other._expressions[0][0]
                    return self
                else:
                    try:
                        self._expressions[0][0] /= other._expressions[0][0] ** other._expressions[0][2]
                    except (TypeError, ValueError):
                        return ExpressionSum((self, other))

        return ExpressionSum((self, other.__neg__()))

    def __sub__(self, other):
        return self.__copy__().__isub__(other)

    def __imul__(self, other: "Union[int, float, IExpression]"):
        if isinstance(other, Log):
            other_evaluation = other.try_evaluate()
            if other_evaluation is not None:
                self._coefficient *= other_evaluation
            else:
                self._coefficient *= other._coefficient
                for other_list in other._expressions:
                    existing_appearance: int = self.index_of(other_list)
                    if existing_appearance != -1:
                        self._expressions[existing_appearance][2] += other_list[2]
                    else:
                        self._expressions.append([other_list[0].__copy__(), other_list[1], other_list[2]])
            return self
        else:
            self._coefficient *= other

            return self

    def __mul__(self, other: "Union[int, float, IExpression]"):
        return self.__copy__().__imul__(other)

    def __rmul__(self, other):
        return self.__copy__().__imul__(other)

    def __itruediv__(self, other: "Union[int, float, IExpression]"):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot Divide A logarithm by an expression that evaluates to 0")
            self._coefficient /= other
            return self
        if isinstance(other, IExpression):
            my_eval, other_eval = self.try_evaluate(), other.try_evaluate()
            if None not in (my_eval, other_eval):
                if other_eval == 0:
                    raise ZeroDivisionError("Cannot Divide A logarithm by an expression that evaluates to 0")
                self._expressions = []
                self._coefficient = my_eval / other_eval
                if isinstance(self._coefficient, (int, float)):
                    self._coefficient = Mono(self._coefficient)
                return self
            elif other_eval is not None:
                self._coefficient /= other_eval
                return self
            else:
                # TODO: add here checks to see if the expressions are dependant and thus can be simplified!
                return Fraction(self, other)

    def __ipow__(self, power):
        self._coefficient **= power
        for mini_expression in self._expressions:
            mini_expression[2] *= power
        return self

    def __neg__(self):
        self._coefficient *= -1
        return self

    def assign(self, **kwargs):
        for expression in self._expressions:
            expression[0].assign(**kwargs)  # TODO: can be dangerous, requires further checking

    def try_evaluate(self):
        """ return an int / float evaluation of the expression. If not possible, return None."""
        evaluated_coefficient = self._coefficient.try_evaluate()
        if evaluated_coefficient is None:
            return None
        if not self._expressions:
            return round_decimal(evaluated_coefficient)
        evaluated_inside = self._expressions[0][0].try_evaluate()
        if not isinstance(self._expressions[0][1], (int, float)):
            evaluated_base = self._expressions[0][1].try_evaluate()
        else:
            evaluated_base = self._expressions[0][1]
        if evaluated_inside is None:
            return None
        power = self._expressions[0][2]
        if isinstance(power, IExpression):
            power = power.try_evaluate()
            if power is None:
                return None
        return round_decimal(evaluated_coefficient * log(evaluated_inside, evaluated_base) ** power)

    def _single_log_str(self, inside: IExpression, base, power_by) -> str:

        if power_by == 0:
            return "1"
        if power_by == 1:
            power_by = ""
        else:
            power_by = f"^{round_decimal(power_by)} "
        if abs(base - e) < 0.00001:
            return f"ln({inside.__str__()}){power_by}"
        return f"log{base}({inside.__str__()}){power_by}"

    def python_syntax(self) -> str:
        """ Return a string that represents the expression and can be evaluated into expression using eval()"""
        if isinstance(self._coefficient, IExpression):
            coefficient_str = self._coefficient.python_syntax()
        else:
            coefficient_str = self._coefficient.__str__()
        if coefficient_str == "1":
            coefficient_str = ""
        elif coefficient_str == "-1":
            coefficient_str = "-"
        else:
            coefficient_str += "*"
        expression_str = ""
        for (expression, base, power) in self._expressions:
            if isinstance(power, IExpression):
                power = power.python_syntax()
            if power == 1:
                power = ""
            else:
                power = f"** {power}"
            if isinstance(expression, IExpression):
                expression_str += f"log({expression.python_syntax()},{base}){power}*"
            else:
                expression_str += f"log({expression.__str__()},{base}){power}*"
        if len(expression_str):  # If the string isn't empty because the object represents a free number
            expression_str = expression_str[:-1]  # remove the last star from the string

        return f"{coefficient_str}{expression_str}"

    def to_dict(self):
        return {'type': 'Log', 'data': {'coefficient': (self._coefficient.to_dict() if \
                                                            hasattr(self._coefficient,
                                                                    'to_dict') else self._coefficient),
                                        'expressions': [[(inside.to_dict() if \
                                                              hasattr(inside, 'to_dict') else inside),
                                                         (base.to_dict() if hasattr(base, 'to_dict') else base),
                                                         (power.to_dict() if \
                                                              hasattr('power', 'to_dict') else power)] for
                                                        [inside, base, power] in self._expressions]}}

    @staticmethod
    def from_dict(given_dict: dict):
        coefficient_obj = create_from_dict(given_dict['data']['coefficient'])
        expressions_objs = [
            [create_from_dict(expression[0]), create_from_dict(expression[1]), create_from_dict(expression[2])] for
            expression in given_dict['data']['expressions']]
        return Log(expression=expressions_objs, coefficient=coefficient_obj)

    def __str__(self) -> str:
        if self._coefficient == 0:
            return "0"
        if not self._expressions:
            return f"{self._coefficient}"
        coefficient_str: str = format_coefficient(self._coefficient)
        if coefficient_str not in ("", "-"):
            coefficient_str += '*'

        return coefficient_str + "*".join(
            self._single_log_str(log_list[0], log_list[1], log_list[2])
            for log_list in self._expressions)

    def __copy__(self):
        new_log = Log(self._expressions)
        new_log._coefficient = self._coefficient.__copy__()
        return new_log

    # TODO: implement and use in __eq__
    def _equate_single_logs(self, other):
        pass

    def __eq__(self, other: Union[int, float, IExpression]):
        if other is None:
            return False
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            return my_evaluation is not None and my_evaluation == other  # TODO: check if this applies to all cases
        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if my_evaluation == other_evaluation:
                return True
            if isinstance(other, Log):
                if self._coefficient != other._coefficient:
                    return False
                # Do equating similarly to what is done in TrigoExpr , between the expressions
            else:
                return False
        else:
            raise TypeError(f"Invalid type '{type(other)}' for equating logarithms.")

    def __ne__(self, other: Union[IExpression, int, float]):
        return not self.__eq__(other)


class PolyLog(Log):  # Find out the proper implementation
    def __init__(self, expressions: Union[
        Iterable[Union[Poly, Mono]], Iterable[List[list]], Union[Poly, Mono], str, float, int],
                 base: Union[int, float] = 10, coefficient: Union[int, float] = Poly(1)):
        super(PolyLog, self).__init__(expression=expressions, base=base, coefficient=coefficient)


class Ln(Log):
    def __init__(self, expressions: Union[Iterable[IExpression], Iterable[List[list]], IExpression, str, float, int]):
        super(Ln, self).__init__(expressions, base=e)


class TrigoExpr(IExpression, IPlottable, IScatterable):
    """ This class represents a single trigonometric expression, such as 3sin(2x)cos(x) for example. """
    __slots__ = ['_coefficient', '_expressions']

    def __init__(self, coefficient, dtype='poly',
                 expressions: "Iterable[Iterable[Union[int,float,Mono,Poly,TrigoMethods]]]" = None):
        self._coefficient = None
        self._expressions: list = []
        if isinstance(coefficient, TrigoExpr):  # TODO: delete copy constructor and modify coefficient
            self._coefficient = coefficient._coefficient
            self._expressions = coefficient._expressions.copy()
        if isinstance(coefficient, str):
            self._coefficient, self._expressions = TrigoExpr_from_str(coefficient, get_tuple=True, dtype=dtype)
        else:
            # First handle the _coefficient parameter
            if isinstance(coefficient, (int, float)):
                self._coefficient = Mono(coefficient)
            elif isinstance(coefficient, Mono):
                self._coefficient = coefficient.__copy__()
            elif isinstance(coefficient, Poly):
                self._coefficient = coefficient.__copy__()

            # Now handle the expressions
            if expressions is None:
                self._expressions = None
            else:
                self._expressions = [list(expression) for expression in expressions]

    @property
    def coefficient(self):
        return self._coefficient

    @property
    def expressions(self):
        return self._expressions

    def simplify(self):  # TODO: experimental, improve it !
        if self._coefficient == 0:
            self._expressions = [[None, Mono(0), 1]]
        for index, (method, inside, power) in enumerate(self._expressions):
            if power == 0:  # Then we can remove this item since multiplying by 1 doesn't change the expression
                self._expressions.pop(index)

    @property
    def variables(self):
        variables = self._coefficient.variables
        for inner_list in self._expressions:
            variables.update(inner_list[1].variables)
        return variables

    def __add_or_sub(self, other, operation: str = '+'):
        if other == 0:
            return self
        if isinstance(other, (int, float, str)):
            if isinstance(other, (int, float)):
                my_evaluation = self.try_evaluate()
                if my_evaluation is None:
                    if operation == '+':
                        return TrigoExprs((self, TrigoExpr(other)))
                    return TrigoExprs((self, -TrigoExpr(other)))
                else:
                    return Mono(my_evaluation + other)
        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            my_evaluation = self.try_evaluate()
            if None not in (other_evaluation, my_evaluation):
                if operation == '+':
                    return Mono(other_evaluation + my_evaluation)
                return Mono(other_evaluation - my_evaluation)

            if isinstance(other, TrigoExpr):
                if self._expressions == other._expressions:
                    if operation == '+':
                        self._coefficient += other._coefficient
                    else:
                        self._coefficient -= other._coefficient
                    if self._coefficient == 0:
                        self._expressions = [[None, Mono(0), 1]]
                    return self
                else:
                    if operation == '+':
                        return TrigoExprs((self, other))
                    return TrigoExprs((self, -other))

            elif isinstance(other, TrigoExprs):
                if operation == '+':
                    return other.__add__(self)
                return other.__sub__(self)
            else:
                if operation == '+':
                    return ExpressionSum((self, other))
                return ExpressionSum((self, -other))
        else:
            raise TypeError(f"Invalid type {type(other)} while adding or subtracting trigonometric expressions."
                            f"Expected types: TrigoExpr, TrigoExprs, int, float, str, Poly, Mono")

    def __iadd__(self, other: Union[IExpression, int, float, str]):  # TODO: update this
        return self.__add_or_sub(other, operation='+')

    def __isub__(self, other):
        return self.__add_or_sub(other, operation='-')

    @staticmethod  # Because this method is only relevant to the TrigoExpr class, it is static
    def __find_similar_expressions(expressions: List[list], given_method, given_expression: IExpression) -> Iterator:
        """
        Returns a generator expression that yields the expressions that can be multiplied or divided with the
        given expression, i.e, expressions with the same coefficient and inside expression.
        """
        return (index for index, expression in enumerate(expressions) if
                expression[0] is given_method and expression[1] == given_expression)

    @staticmethod
    def __find_exact_expressions(expressions: List[list], given_method, given_expression, given_power):
        return (index for index, expression in enumerate(expressions) if
                expression[0] is given_method and expression[1] == given_expression and expression[2] == given_power)

    def divide_by_trigo(self, other: "TrigoExpr"):
        if other == 0:
            return ZeroDivisionError("Cannot divide by a TrigoExpr object that evaluates to 0")
        if len(self._expressions) == 0 and len(other._expressions) == 0:
            return True

    @staticmethod
    def __divide_identities(self, other):
        for index, [method, inside, power] in enumerate(self._expressions):
            if not other._expressions:
                return self
            if method == TrigoMethods.SIN:
                try:
                    matching_index = next(
                        self.__find_similar_expressions(other._expressions, TrigoMethods.COS, inside))
                    my_power, other_power = self._expressions[index][2], other._expressions[matching_index][2]
                    if my_power == other_power:
                        self._expressions[index][0] = TrigoMethods.TAN
                        del other._expressions[matching_index]
                    elif my_power > other_power:
                        self._expressions[index][2] -= other._expressions[matching_index][2]
                        self._expressions.append([TrigoMethods.TAN, other._expressions[matching_index][1].__copy__(),
                                                  copy_expression(other._expressions[matching_index][2])])
                        del other._expressions[matching_index]
                    else:
                        self._expressions[index][0] = TrigoMethods.TAN
                        other._expressions[matching_index][2] -= my_power


                except StopIteration:
                    pass
                    # Didn't find a match then do nothing..

            if power == 1 and method == TrigoMethods.SIN:
                try:  # If found sin
                    matching_index = next(
                        self.__find_exact_expressions(other._expressions, TrigoMethods.SIN, inside / 2, 1))
                    self._coefficient *= 2
                    del self._expressions[index]
                    del other._expressions[matching_index]
                except StopIteration:  # If not, search for cos
                    try:  # if found cos
                        matching_index = next(
                            self.__find_exact_expressions(other._expressions, TrigoMethods.COS, inside / 2, 1))
                        del other._expressions[matching_index]
                        self._coefficient *= 2
                        self._expressions[index] = [TrigoMethods.SIN, inside / 2, 1]
                        continue
                    except:  # if didn't find cos
                        continue

                try:  # if found sin and found cos too
                    matching_index = next(
                        self.__find_exact_expressions(other._expressions, TrigoMethods.COS, inside / 2, 1))
                    del other._expressions[matching_index]
                except:  # if didn't find cos
                    self._expressions.append([TrigoMethods.COS, inside / 2, 1])
        return other

    def __mul_trigo(self, other: "TrigoExpr") -> None:
        self._coefficient *= other._coefficient
        for index, other_expression in enumerate(other._expressions):
            try:
                # Try to find a similar expression that can be multiplied with the other expression
                matching_index = next(
                    self.__find_similar_expressions(self._expressions, other_expression[0], other_expression[1]))
                self._expressions[matching_index][2] += other._expressions[index][2]  # Add up the powers

            except StopIteration:  # If not found, add the expression to the end of the list
                self._expressions.append(other_expression.copy())

    def __imul__(self, other: Union[IExpression, int, float, str]):
        if isinstance(other, (int, float)):
            if other == 0:
                self._coefficient, self._expressions = Poly(0), [[None, Mono(0), 1]]
            else:
                self._coefficient *= other
            self.simplify()
            return self
        elif isinstance(other, str):
            other = TrigoExprs(other)
        if isinstance(other, (Poly, Mono)):
            self._coefficient *= other
            self.simplify()
            return self
        elif isinstance(other, TrigoExpr):
            if other._coefficient == 0:
                self._coefficient, self._expressions = Poly(0), [[None, Mono(0), 1]]
            else:
                self.__mul_trigo(other)
            self.simplify()
            return self

        elif isinstance(other, TrigoExprs):
            if len(other.expressions) == 1:
                self.__mul_trigo(other.expressions[0])
            else:
                return other.__mul__(self)
        elif isinstance(other, IExpression):
            return ExpressionMul((self, other))
        else:
            raise TypeError(f"Encountered Invalid type {type(other)} when multiplying trigonometric expressions.")

    def __mul__(self, other):
        return self.__copy__().__imul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __itruediv__(self, other: "Union[int,float,IExpression]"):
        if isinstance(other, (int, float)):
            self._coefficient /= other
            return self
        elif isinstance(other, TrigoExpr):
            if other._coefficient == 0:
                raise ZeroDivisionError(f"Tried to divide '{self.__str__()} by {other.__str__()}'")
            if self == other:  # Check if they are equal
                self._expressions = []
                self._coefficient = Mono(1)
                return self
            self._coefficient /= other._coefficient
            denominator = []
            if len(self._expressions) == 1:
                other_length = len(other._expressions)
                if other_length == 1:
                    if self._expressions[0][1] == other._expressions[0][1]:
                        if self._expressions[0][0] == TrigoMethods.SIN and other._expressions[0][0] == TrigoMethods.COS:
                            my_power, other_power = self._expressions[0][2], other._expressions[0][2]
                            if my_power == other_power:
                                self._expressions[0][0] = TrigoMethods.TAN
                                return self
                            elif my_power > other_power:
                                self._expressions[0][2] -= other._expressions[0][2]
                                self._expressions.append([TrigoMethods.TAN, other._expressions[0][1].__copy__(),
                                                          copy_expression(other._expressions[0][2])])
                                return self
                            else:
                                self._expressions[0][0] = TrigoMethods.TAN
                                other_copy = other.__copy__()
                                other_copy._expressions[0][0] -= my_power
                                return Fraction(self.__copy__(), other_copy, gen_copies=False)

                        elif self._expressions[0][0] == TrigoMethods.COS and other._expressions[0][
                            0] == TrigoMethods.SIN:
                            my_power, other_power = self._expressions[0][2], other._expressions[0][2]
                            if my_power == other_power:
                                self._expressions[0][0] = TrigoMethods.COT
                                return self
                            elif my_power > other_power:
                                self._expressions[0][2] -= other._expressions[0][2]
                                self._expressions.append([TrigoMethods.COT, other._expressions[0][1].__copy__(),
                                                          copy_expression(other._expressions[0][2])])
                                return self
                            else:
                                self._expressions[0][0] = TrigoMethods.COT
                                other_copy = other.__copy__()
                                other_copy._expressions[0][0] -= my_power
                                return Fraction(self.__copy__(), other_copy, gen_copies=False)

            for index, other_expression in enumerate(other._expressions):
                try:
                    matching_index = next(
                        self.__find_similar_expressions(self._expressions, other_expression[0], other_expression[1]))
                    self._expressions[matching_index][2] -= other_expression[2]  # Add up the powers

                except StopIteration:
                    denominator.append(other_expression.copy())
            self.simplify()
            if not denominator:
                return self
            else:
                other = TrigoExpr(expressions=denominator, coefficient=1)
                self.__divide_identities(self, other)
                self.__divide_identities(other, self)
                print(f"self is {self} and other is {other}")

            if not other._expressions:
                return self
            return Fraction(self, other)
        elif isinstance(other, TrigoExprs):  # TODO: implement it
            pass
        else:
            return Fraction(self, other)

    def __neg__(self) -> "TrigoExpr":
        copy_of_self = self.__copy__()
        copy_of_self._coefficient *= -1
        return copy_of_self

    def flip_sign(self):
        """flips the sign of the expression - from positive to negative, or from negative to positive"""
        self._coefficient *= -1

    def __ipow__(self, power):  # TODO: check if works
        self._coefficient **= power
        for index, [method, inside, degree] in enumerate(self._expressions):
            self._expressions[index] = [method, inside, degree * power]
        self.simplify()
        return self

    def assign(self, **kwargs):
        self._coefficient.assign(**kwargs)
        for index, [method, inside, degree] in enumerate(self._expressions):
            inside.assign(**kwargs)
            if new_inside := inside.try_evaluate():  # if the inside expression can be evaluated, then simplify
                new_inside = method.value[0](new_inside) ** degree
                self._expressions[index][0] = None
                self._expressions[index][1] = Poly(new_inside)
                self._expressions[index][2] = 1

    def try_evaluate(self) -> Optional[float]:
        evaluated_coefficient = self._coefficient.try_evaluate()
        if not self._expressions:
            return evaluated_coefficient
        if evaluated_coefficient is None or any(
                None in (inside, degree) for [method, inside, degree] in self._expressions):
            return False
        my_sum = 0
        for method, inside, degree in self._expressions:
            if isinstance(inside, IExpression):
                inside = inside.try_evaluate()
                if inside is None:
                    return None
            if method is None:
                my_sum += inside
            else:
                my_sum += method.value[0](inside ** degree)
        return my_sum

    def derivative(self):
        length = len(self._expressions)
        if length == 0:
            return 0
        else:
            coefficient_eval = self._coefficient.try_evaluate()
            if coefficient_eval is not None:
                if length == 1:
                    temp = self._expressions[0]
                    if temp[2] == 0:  # for instance, 3 * sin(x)^0 just 3
                        return self._coefficient.__copy__()
                    elif temp[2] == 1:  # TODO: later optimize some of these
                        if coefficient_eval is not None:  # meaning the coefficient is a free number
                            if temp[0] == TrigoMethods.SIN:
                                return self._coefficient * temp[1].derivative() * Cos(temp[1])
                            elif temp[0] == TrigoMethods.COS:
                                return -self._coefficient * temp[1].derivative() * Sin(temp[1])
                            elif temp == TrigoMethods.TAN:
                                return self._coefficient * temp[1].derivative() * Sec(temp[1]) ** 2
                            elif temp[0] == TrigoMethods.COT:
                                return -self._coefficient * temp[1].derivative() * Csc(temp[1]) ** 2
                            elif temp[0] == TrigoMethods.SEC:
                                return self._coefficient * temp[1].derivative() * Sec(temp[1]) * Tan(temp[1])
                            elif temp[0] == TrigoMethods.CSC:
                                return -self._coefficient * temp[1].derivative() * Csc(temp[1]) * Cot(temp[1])
                            elif temp[0] == TrigoMethods.ASIN:
                                return Fraction(self._coefficient, Root(1 - temp[1] ** 2))
                            elif temp[0] == TrigoMethods.ACOS:
                                return Fraction(-self._coefficient, Root(1 - temp[1] ** 2))
                            elif temp[0] == TrigoMethods.ATAN:
                                return Fraction(self._coefficient, temp[1] ** 2 + 1)
                            elif temp[0] == TrigoMethods.ACOT:
                                return Fraction(-self._coefficient, temp[1] ** 2 + 1)
                            elif temp[0] == TrigoMethods.ASEC:
                                pass  # TODO: implement this!!!
                            elif temp[0] == TrigoMethods.ACSC:
                                pass  # TODO: implement this!!!
                            else:
                                raise ValueError(f"Unrecognized trigonometric method has been used: {temp[0]}")
                        else:  # the coefficient is an algebraic expression, so the whole expression can be (3x+2)sin(x) for instance
                            pass

                elif length == 2:  # TODO: Later modify this to a smarter code
                    if self._expressions[0][0] == self._expressions[1][0] and self._expressions[0][1] == \
                            self._expressions[1][
                                1]:
                        power = self._expressions[0][2] + self._expressions[1][2]
                        if self._expressions[0][
                            0] == TrigoMethods.SIN:  # We're dealing with _coefficient* Sin(f(x))^power
                            return TrigoExpr(coefficient=self._coefficient * power,
                                             expressions=[(TrigoMethods.SIN, self._expressions[0][1], power - 1),
                                                          (TrigoMethods.COS, self._expressions[0][1], 1)])
                    else:
                        pass  # TODO: how to derive sin(x)cos(x)tan(x) for example ?

    def integral(self):  # TODO: poorly written !
        warnings.warn("This feature is currently extremely limited. Wait for the next versions (Sorry!) ")
        length = len(self._expressions)
        if length == 0:
            return 0
        if length == 1:
            if self._expressions[0][0] == TrigoMethods.SIN:
                self._expressions[0][0] = TrigoMethods.COS
                self._coefficient = -self._coefficient
            elif self._expressions[0][0] == TrigoMethods.COS:
                self._expressions[0][0] = TrigoMethods.SIN
            else:
                pass

    def plot(self, start: float = -8, stop: float = 8,
             step: float = 0.3, ymin: float = -3, ymax: float = 3, title=None, show_axis=True, show=True,
             fig=None, ax=None, formatText=True, values=None):
        variables = self.variables

        num_of_variables = len(variables)
        if num_of_variables == 1:
            plot_function(self.to_lambda(), start, stop, step, ymin, ymax, title, show_axis, show, fig, ax, formatText,
                          values)
        elif num_of_variables == 2:
            plot_function_3d(given_function=self.to_lambda(), start=start, stop=stop, )
        else:
            raise ValueError(f"Cannot plot a trigonometric expression with {num_of_variables} variables")

    def newton(self, initial_value: float = 0, epsilon: float = 0.00001, nmax: int = 10_000):
        return newton_raphson(self.to_lambda(), self.derivative().to_lambda(), initial_value, epsilon, nmax)

    def to_dict(self):
        new_expressions = [[(method_chosen.value[0].__name__ if method_chosen is not None else None), inside.to_dict() \
            if hasattr(inside, 'to_dict') else inside, power.to_dict() if hasattr(power, 'to_dict') else power] for
                           [method_chosen, inside, power] in self._expressions]
        return {'type': 'TrigoExpr', 'data': {'coefficient': (
            self._coefficient.to_dict() if hasattr(self._coefficient, 'to_dict') else self._coefficient),
            'expressions': new_expressions}}
        # TODO: shallow copying .. check for bugs

    @staticmethod
    def from_dict(given_dict: dict):
        expressions_objects = [
            [_TrigoMethodFromString(expression[0]), create_from_dict(expression[1]), create_from_dict(expression[2])]
            for expression in given_dict['data']['expressions']]
        return TrigoExpr(coefficient=create_from_dict(given_dict['data']['coefficient']),
                         expressions=expressions_objects)

    def __simple_derivative(self, expression_info: "List[Optional[Any],Optional[Union[Mono,Poly,Var,TrigoExpr,"
                                                   "PolyLog]],float]"):
        copied_expression = self._expressions[0]
        if copied_expression[0] == TrigoMethods.SIN:
            copied_expression[0] = TrigoMethods.COS
            coefficient = self._coefficient
        elif copied_expression == TrigoMethods.COS:
            copied_expression[0] = TrigoMethods.SIN
            coefficient = -self._coefficient
        elif copied_expression == TrigoMethods.TAN:
            pass
        else:
            pass

    def __single_partial(self, variable: str):
        # Extract all the sub-expressions that contain the given variable
        relevant_expressions = [], irrelevant_expressions = []
        for mini_expression in self._expressions:
            if mini_expression[1] is not None and mini_expression[1].contains_variable(variable):
                relevant_expressions.append(mini_expression)
            else:
                irrelevant_expressions.append(mini_expression)

        if self._coefficient.contains_variable(variable):
            relevant_expressions.append(self._coefficient)
        else:
            irrelevant_expressions.append(self._coefficient)

    def partial_derivative(self, variables: Iterable[str]):  # Get an iterable of the variables_dict
        copy_of_self = self.__copy__()
        for variable in variables:
            copy_of_self.__single_partial(variable)

    def __str__(self):
        """ Returns a string representation of the expression"""
        if self._coefficient == 0:
            return "0"  # If it's zero, then it's zero. Aristotle ( 300 BC )
        if not self._expressions:  # If it's only a free number
            return f"{self._coefficient}"
        if self._coefficient == 1:
            accumulator = ""
        elif self._coefficient == -1:
            accumulator = '-'
        else:
            accumulator = f"{self._coefficient}"
        for method_chosen, inside, power in self._expressions:
            if method_chosen:
                accumulator += f"{method_chosen.value[0].__name__}({inside})"
            else:
                accumulator += f"{inside}"
            if power != 1:
                accumulator += f"^{round_decimal(power)}"
            accumulator += '*'
        return accumulator[:-1]

    def python_syntax(self):
        if self._coefficient == 0:
            return '0'
        accumulator = f"{self._coefficient.python_syntax()}*"
        if not self._expressions:
            return accumulator
        if accumulator == '1*':
            accumulator = ""
        elif accumulator == '-1*':
            accumulator = "-"
        for method_chosen, inside, power in self._expressions:
            accumulator += f"{method_chosen.value[0].__name__}({inside.python_syntax()})"
            if power != 1:
                accumulator += f"**{round_decimal(power)}"
            accumulator += '*'
        return accumulator[:-1]

    def __copy__(self):  # Copying all the way without any (risky) memory sharing between copies of the same objects
        expressions = []
        if self._expressions:
            for [method, inside, power] in self._expressions:
                inside = inside.__copy__() if inside is not None and hasattr(inside, '__copy__') else inside
                expressions.append([method, inside, power])

        return TrigoExpr(self._coefficient, expressions=expressions)

    @staticmethod
    # TODO: handle the decimal point bug in python with an epsilon or round_decimal()
    def equal_subexpressions(coef1, first_sub: Tuple[Optional[TrigoMethods], IExpression, float],
                             coef2, second_sub: Tuple[Optional[TrigoMethods], IExpression, float]):
        """
        Compare two trigonometric basic expressions. for example: 3sin(x) == -3sin(-x) or 2cos(2x) == 4sin(3x) etc..
        The equation is done in regards to trigonometric identities as much as possible.

        :param coef1: The coefficient of the first expression
        :param first_sub: A tuple of the trigonometric method applied, the expression that the method applies to, and the power.
        :param coef2: The coefficient of the second expression
        :param second_sub: A tuple of the trigonometric method applied, the expression that the method applies to, and the power.
        :return: Returns True if the two
        """
        if coef1 == 0 == coef2:
            return True
        method1, expression1, power1 = first_sub
        method2, expression2, power2 = second_sub
        expression_difference = expression1 - expression2
        evaluated_difference = expression_difference.try_evaluate()
        if evaluated_difference is not None:
            evaluated_difference %= 2 * pi
        if method1 is method2 and power1 == power2:
            if method1 in (TrigoMethods.SIN, TrigoMethods.COS, TrigoMethods.CSC, TrigoMethods.SEC):
                if coef1 == -coef2 and evaluated_difference == pi:
                    return True  # sin(x + pi) = - sin(x), cos(x+pi) = -cos(x) .. etc.
            if evaluated_difference is not None and evaluated_difference % 360 == 0 and coef1 == coef2:
                return True
            expression_sum = expression1 + expression2
            if method1 is TrigoMethods.SIN:  # sin()
                if expression_sum == pi and coef1 == coef2:  # sin(x) = sin(180-x)
                    return True
                elif coef1 + coef2 == 0 == expression_sum:  # -5sin(x) = 5sin(-x)
                    return True
                else:
                    return False

            elif method1 is TrigoMethods.COS or method1 is TrigoMethods.SEC:
                if expression_sum == 0:  # cos(x) = cos(-x), sec(-x) = sec(x)
                    return True
                if expression_sum == pi:  # cos(pi - x) = -cos(x) sec(pi - x) = -sec(x)
                    if coef1 == -coef2:
                        return True
            elif method1 is TrigoMethods.TAN or TrigoMethods.COT:
                if evaluated_difference % pi == 0:
                    return True
                if expression_sum == 0:
                    if coef1 == -coef2:
                        return True
                if expression_sum == pi:
                    if coef1 == -coef2:
                        return True
        if (((first_method := method1) is TrigoMethods.SIN and (second_method := method2) is TrigoMethods.COS) or
            ((second_method := method1) is TrigoMethods.COS and (first_method := method2) is TrigoMethods.SIN)) or \
                (((first_method := method1) is TrigoMethods.CSC and (second_method := method2) is TrigoMethods.SEC) or
                 ((second_method := method1) is TrigoMethods.SEC and (first_method := method2) is TrigoMethods.CSC)):

            if evaluated_difference == pi / 2 and coef1 == coef2:
                return True
        if evaluated_difference == -pi / 2 and coef1 == -coef2:
            return True

        else:
            return False

    def __eq__(self, other: Union[IExpression, int, float]):
        if other is None:
            return False
        if isinstance(other, (int, float)):
            if not self._expressions:
                return self._coefficient == other
            else:
                my_evaluation = self.try_evaluate()
                return my_evaluation is not None and my_evaluation == other

        elif isinstance(other, IExpression):
            # First check if the expressions can be evaluated into numbers
            my_evaluation = self.try_evaluate()
            other_evaluation = other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                if my_evaluation == other_evaluation:
                    return True
            elif (my_evaluation, other_evaluation) != (None, None):  # if one of them can be evaluated and the other not
                return False

            if isinstance(other, TrigoExpr):
                first_basic: bool = len(other._expressions) == 1
                second_basic: bool = len(self._expressions) == 1
                if first_basic and second_basic:  # If both are subclasses of TrigoExpr, or they have 1 expressions
                    return self.equal_subexpressions(self.coefficient, self._expressions[0], other.coefficient,
                                                     other._expressions[0])
                else:
                    if self._coefficient != other._coefficient:  # If the coefficients are different
                        try:  # Handle the identity sin(2x) = 2sin(x)cos(x)
                            my_length, other_length = len(self._expressions), len(other._expressions)
                            if my_length + other_length != 3:
                                return False  # return False for now
                            coefficient_ratio = self._coefficient / other._coefficient
                            ratio_eval = coefficient_ratio.try_evaluate()
                            if ratio_eval is None:
                                return False

                            # VERIFY THAT THE RATIO BETWEEN THE COEFFICIENT IS 2 OR 0.5 ( 2SIN(X)COS(X) = SIN(2X) )
                            if abs(ratio_eval - 0.5) < 0.000001:
                                first, second = self, other  # first: sin(2x), second: 2sin(x)cos(x)
                                first_length, second_length = my_length, other_length
                            elif abs(ratio_eval - 2) < 0.0000001:
                                first, second = other, self
                                first_length, second_length = other_length, my_length
                            else:
                                return False  # return false for now

                            # VERIFY THE LENGTH OF THE EXPRESSIONS
                            if first_length != 1 or second_length != 2:
                                return False
                            if second._expressions[0][1] != second._expressions[1][1]:  # sin(x)cos(x) ( x == x )
                                return False  # return false for now
                            if first._expressions[0][0] != TrigoMethods.SIN:
                                return False
                            second_methods = second._expressions[0][0], second._expressions[1][0]
                            if second_methods not in (
                                    (TrigoMethods.SIN, TrigoMethods.COS), (TrigoMethods.SIN, TrigoMethods.COS)):
                                return False
                            return True

                        except ZeroDivisionError:
                            return False

                    taken_indices = []
                    for other_index, other_sub_list in enumerate(other._expressions):
                        found = False
                        for my_index, sub_list in enumerate(self._expressions):
                            if sub_list == other_sub_list and my_index not in taken_indices:
                                # if it's equal and we haven't found it already, namely, it's not in "taken_indices"
                                found = True
                                taken_indices.append(my_index)  # append the current index to the taken indices
                            if found:
                                break  # break to the outer loop
                        if not found:
                            return False
                    return True
            else:
                return False  # TODO: check the cases of comparison between TrigoExpr and other ExpressionSum ..

    def __ne__(self, other: Union[IExpression, int, float]):
        return not self.__eq__(other)


def conversion_wrapper(given_func: Callable):
    def inner(self):
        if not len(self._expressions) == 1:
            raise ValueError("expression must contain one item only for cosine conversion: For example, sin(3x)")
        return given_func(self)

    return inner


class Sin(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super().__init__(coefficient=f"sin({expression})", dtype=dtype)
        else:
            super(Sin, self).__init__(1, expressions=((TrigoMethods.SIN, expression, 1),))

    @conversion_wrapper
    def to_cos(self):
        if self._expressions[0][2] == 1:  # If the power is 1
            return Cos(90 - self._expressions[0][1]) * self._coefficient
        elif self._expressions[0][2] == 2:  # If the power is 2, for instance sin(x)^2
            return 1 - Cos(self._expression[0][1]) ** 2

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
        current_x = current_x - (2 * f * f_prime) / (2 * f_prime ** 2 - f * f_double_prime)


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
    warnings.warn("The result might be inaccurate. Try entering different parameters or using different methods")
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
    warnings.warn("The solution might be inaccurate due to insufficient convergence.")
    return xk


def get_bounds(degree: int, coefficients):
    upper = 1 + 1 / abs(coefficients[-1]) * max(abs(coefficients[x]) for x in range(degree))
    lower = abs(coefficients[0]) / (abs(coefficients[0]) + max(abs(coefficients[x]) for x in range(1, degree + 1)))
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
        coefficients = [coefficient / coefficients[0] for coefficient in coefficients]
        f_0 = monic_poly_from_coefficients(coefficients).to_lambda()
    else:
        coefficients = [coefficient for coefficient in coefficients]
    current_guesses = __durandKerner_approximations(coefficients)
    for i in range(nmax):
        if all(abs(f_0(current_guess)) < epsilon for current_guess in current_guesses):
            return {complex(round_decimal(c.real), round_decimal(c.imag)) for c in current_guesses}
        for index in range(len(current_guesses)):
            numerator = f_0(current_guesses[index])
            other_guesses = (guess for j, guess in enumerate(current_guesses) if j != index)
            denominator = reduce(lambda a, b: a * b, (current_guesses[index] - guess for guess in other_guesses))
            current_guesses[index] -= numerator / denominator  # Updating each guess
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
        initial_value -= (f_x / f_tag) * (1 + (f_x * f_tag_tag) / (2 * f_tag ** 2))
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
                sigma = sum(1 / (zk - zj) for j, zj in enumerate(random_guesses) if k != j and zk != zj)
                denominator = 1 - m * sigma
                offsets.append(m / denominator)
            random_guesses = [approximation - offset for approximation, offset in zip(random_guesses, offsets)]
            if all(negligible_complex(f_0(guess), epsilon) for guess in random_guesses):
                break
        solutions = [complex(round_decimal(result.real), round_decimal(result.imag)) for result in random_guesses]
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


def solve_cubic(a: float, b: float, c: float, d: float):
    """ Given the real coefficients of a cubic equation, this method will return the solutions"""
    if a == 0:
        return solve_quadratic(b, c, d)
    delta0 = b * b - 3 * a * c
    delta1 = 2 * pow(b, 3) - 9 * a * b * c + 27 * a * a * d
    deepest_root = cmath.sqrt(pow(delta1, 2) - 4 * pow(delta0, 3))
    C = (0.5 * (delta1 + deepest_root)) ** (1. / 3)
    if C == 0:  # In case
        C = (0.5 * (delta1 - deepest_root)) ** (1. / 3)
    if C == 0:  # If c is still 0 ( can't be because of zero division error after that )
        return [0]
    roots = []
    root_of_unity = complex(-0.5, sqrt(3) / 2)
    for k in range(3):
        root = -((b + C + delta0 / C) / (3 * a))
        roots.append(root)
        C *= root_of_unity

    return list({complex(round_decimal(root.real), round_decimal(root.imag)) for root in
                 roots})


def solve_cubic_real(a: float, b: float, c: float, d: float):  # TODO: improve in next versions
    roots = solve_cubic(a, b, c, d)
    if not roots:
        return []
    return [root.real for root in roots if abs(root.imag) < 0.00001]


def bairstow_method():  # TODO: implement this already ...
    pass


def solve_quartic(a: float, b: float, c: float, d: float, e: float):
    if a == 0:
        return solve_cubic(b, c, d, e)
    if a != 1:  # Divide everything by a to get to the form x^4 + bx^3 + cx^2 + dx + e
        b /= a
        c /= a
        d /= a
        e /= a
        a = 1
    f = c - (3 * b ** 2 / 8)
    g = d + (b ** 3 / 8) - (b * c / 2)
    h = e - (3 * b ** 4 / 256) + (b ** 2 * c / 16) - (b * d / 4)
    three_roots = solve_cubic(1, f / 2, (f ** 2 - 4 * h) / 16, -g ** 2 / 64)
    if len(three_roots) == 1 and three_roots[0] == 0:
        return [0]
    else:
        y1, y2, y3 = three_roots
    non_zero_roots = [sol for sol in (y1, y2, y3) if sol != 0]
    if len(non_zero_roots) == 3:
        non_zero_roots = non_zero_roots[:-1]
    elif len(non_zero_roots) < 2:
        return [0]
    p, q = cmath.sqrt(non_zero_roots[0]), cmath.sqrt(non_zero_roots[1])
    r = -g / (8 * p * q)
    s = b / (4 * a)
    sol1, sol2, sol3, sol4 = p + q + r - s, p - q - r - s, -p + q - r - s, -p - q + r - s
    return list({sol1, sol2, sol3, sol4})


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


def extract_variables_from_expression(expression: str):
    return {character for character in expression if character.isalpha()}


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
    expression = clean_from_spaces(expression)
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


def simplify_expression(expression: str, variables: Iterable[str], format_abs=False, format_factorial=False) -> dict:
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


def contains_from_list(lst: list, s: str) -> bool:
    """
    checks whether a string appears in a list of strings
    :param lst: the list of strings, for example : ["hello","world"]
    :param s: the string, for example: "hello"
    :return: True if contains, else False
    """
    return bool([x for x in lst if x in s])


def clean_from_spaces(equation: str) -> str:
    """cleans a string from spaces.
    """
    return "".join([character for character in equation if character != ' '])


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


def __data_from_single(single_expression: str, variable_name: str):
    """
    Extracts data from a single-variable monomial, such as 3x^2, or y^2, 82 , etc

    :param single_expression:
    :param variable_name:
    :return:  A tuple with the _coefficient as the first element, and a dictionary of the variable name and its power
    as the second element.
    """
    single_expression = clean_from_spaces(single_expression)
    if not variable_name:
        return extract_coefficient(single_expression), None
    variable_place = single_expression.find(variable_name)
    coefficient = extract_coefficient(single_expression[:variable_place])

    power_index = single_expression.rfind('^')
    power = 1 if power_index == -1 else float(single_expression[power_index + 1:])
    return coefficient, {variable_name: power}


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
    first_dict = simplify_expression(expression=first_side, variables=variables)
    second_dict = simplify_expression(expression=second_side, variables=variables)
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
    first_dict = simplify_expression(first_side, variables)
    second_dict = simplify_expression(second_side, variables)
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
        first_dict = simplify_expression(side1, variables)
        second_dict = simplify_expression(side2, variables)
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


class Equation(ABC):
    def __init__(self, equation: str, variables: Iterable = None, calc_now: bool = False):
        """The base function of creating a new Equation"""
        self._equation = clean_from_spaces(equation)
        if variables is None:
            self._variables = get_equation_variables(equation)
            self._variables_dict = self._extract_variables()
            try:
                index = self._variables.index("number")
                del self._variables[index]
            except ValueError:
                pass
        else:
            self._variables = list(variables)
            self._variables_dict = {variable: 0 for variable in variables}
            self._variables_dict["number"] = 0

        if calc_now:
            self._solution = self.solve()
        else:
            self._solution = None

    # PROPERTIES
    @property
    def equation(self):
        return self._equation

    @property
    def variables(self):
        return self._variables

    @property
    def num_of_variables(self):
        return len(self._variables)

    @property
    def first_side(self):
        return self._equation[:self._equation.rfind("=")]

    @property
    def second_side(self):
        return self._equation[self._equation.rfind("=") + 1:]

    @property
    def solution(self):
        if self._solution is None:
            self._solution = self.solve()
        return self._solution

    @property
    def variables_dict(self):
        return self._variables_dict

    @abstractmethod
    def _extract_variables(self):
        return extract_dict_from_equation(self._equation)

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def __copy__(self):
        return LinearEquation(self._equation)

    def __reversed__(self):
        """
        reverses the sides of the equation. for example: '3x+5=14' -> '14=3x+5'
        :return:
        """
        equal_index = self.equation.find('=')
        first_side, second_side = self.equation[:equal_index], self.equation[equal_index + 1:]
        return LinearEquation(f'{second_side}={first_side}')

    @abstractmethod
    def __repr__(self):
        pass

    def __str__(self):
        return self._equation


class LinearEquation(Equation):  # TODO: support features for more than 1 variable such as assignment and parameters.

    def __init__(self, equation: str, variables=None, calc_now=False):
        super().__init__(equation, variables, calc_now)
        try:
            index = self._variables.index("number")
            del self._variables[index]
        except ValueError:
            pass

    def solve(self):
        if self._solution is None:
            self._solution = solve_linear(self.equation, self.variables_dict)
        return self._solution

    def simplify(self, round_coefficients=True):
        first_dict = simplify_expression(expression=self.first_side, variables=self._variables_dict)
        second_dict = simplify_expression(expression=self.second_side, variables=self._variables_dict)
        result_dict = {key: value for key, value in subtract_dicts(dict1=first_dict, dict2=second_dict).items() if key}
        self._variables_dict = result_dict.copy()
        num = result_dict['number']
        del result_dict['number']
        self._equation = f"{format_linear_dict(result_dict, round_coefficients=round_coefficients)} = {round_decimal(-num)}"

    def __format_expressions(self, expressions):
        accumulator = ""
        for index, expression in enumerate(expressions):
            if expression.variables_dict in ({}, None):
                if expression.coefficient > 0 and index > 0:
                    accumulator += f"\033[93m+{expression}\33[0m "
                else:
                    accumulator += f"\033[96m{expression}\33[0m "
            elif expression.coefficient != 0:

                if expression.coefficient > 0 and index > 0:
                    accumulator += f"\033[93m+{expression}\33[0m "
                else:
                    accumulator += f"\033[96m{expression}\33[0m "
        return accumulator

    def show_steps(self):  # Only for univariate linear equations
        variables = self.variables_dict
        if len(variables) > 2:
            raise NotImplementedError(
                f"This feature is currently only available with 1-variable equation, got {len(variables)}")
        if len(variables) < 2:
            first_side, second_side = self.first_side, self.second_side
            accumulator = "\033[1m1. First step: recognize that this equation only contains free numbers," \
                          "and hence either it has no solutions, or it has infinite solutions \33[0m\n"
            accumulator += f"\033[93m{first_side.replace('+', ' +').replace('-', ' -')}\33[0m"
            accumulator += " = "
            accumulator += f"\033[93m{second_side.replace('+', ' +').replace('-', ' -')}\33[0m\n"
            accumulator += "\033[1m2. Second Step: sum all the numbers in both sides\33[0m\n"
            first_expression = simplify_expression(expression=first_side, variables=variables)
            second_expression = simplify_expression(expression=first_side, variables=variables)
            accumulator += f"\033[93m{first_expression['number']}\33[0m"
            accumulator += ' = '
            accumulator += f"\033[93m{second_expression['number']}\33[0m\n"
            if first_expression["number"] == second_expression["number"]:
                accumulator += "\033[1mFinal Step:  The expression above is always true, and hence there are infinite solutions " \
                               "to the equation.\33[0m\n"
                self._solution = "Infinite"
            else:
                accumulator += "\033[1mFinal Step: The expression above is always false, and hence there are infinite solutions" \
                               " to the equation.\33[0m\n"
                self._solution = None
            return accumulator

        first_variable = list(self.variables_dict.keys())[0]
        first_side, second_side = self.first_side, self.second_side
        first_expressions = poly_from_str(first_side, get_list=True)
        second_expressions = poly_from_str(second_side, get_list=True)
        accumulator = f"\033[1m1. First Step : Identify the free numbers and the expressions with {first_variable} in each side\33[0m\n"
        accumulator += self.__format_expressions(first_expressions) + " = " + self.__format_expressions(
            second_expressions) + "\n"
        accumulator += "\033[1m2. Second step: Sum the matching groups in each side ( if it's possible )\33[0m\n"
        free_sum1, variables_sum1 = 0, 0  # Later perhaps adjust it to support multiple variables_dict
        for mono_expression in first_expressions:
            if mono_expression.is_number():
                free_sum1 += mono_expression.coefficient
            else:
                variables_sum1 += mono_expression.coefficient
        accumulator += f"\033[96m{variables_sum1}{first_variable}\33[0m "
        if free_sum1 > 0:
            accumulator += f"+\033[93m{free_sum1}\33[0m"
        elif free_sum1 != 0:
            accumulator += f"\033[93m{free_sum1}\33[0m"
        accumulator += " = "
        free_sum2, variables_sum2 = 0, 0  # Later perhaps adjust it to support multiple variables_dict
        for mono_expression in second_expressions:
            if mono_expression.is_number():
                free_sum2 += mono_expression.coefficient
            else:
                variables_sum2 += mono_expression.coefficient
        accumulator += f"\033[96m{variables_sum2}{first_variable}\33[0m "
        if free_sum1 > 0:
            accumulator += f"+\033[93m{free_sum2}\33[0m"
        elif free_sum1 != 0:
            accumulator += f"\033[93m{free_sum2}\33[0m"
        accumulator += "\n"
        accumulator += "\033[1m3. Third Step: Move all the variables to the right, and the free numbers to the left \33[0m\n"
        variable_difference = variables_sum1 - variables_sum2
        if variable_difference == 0:
            accumulator += '0'
        else:
            accumulator += f"\033[96m{variable_difference}{first_variable}\33[0m"

        accumulator += " = "
        free_sum_difference = free_sum2 - free_sum1
        accumulator += f"\033[93m{free_sum_difference}\33[0m\n"
        if variable_difference == 0:  # If the variables_dict have vanished in the simplification process
            if free_sum_difference == 0:
                accumulator += "\033[1m3 Therefore, there are infinite solutions !\33[0m\n"
                self._solution = "Infinite"
                return accumulator
            else:
                accumulator += "\033[1m3 Therefore, there is no solution to the equation !\33[0m\n"
                self._solution = None
                return accumulator

        accumulator += "\033[1m4. Final step: divide both sides by the coefficient of the right side \33[0m\n"
        accumulator += f"\033[96m{first_variable}\33[0m = \033[93m{free_sum_difference / variable_difference}\33[0m"
        return accumulator

    def plot_solution(self, start: float = -10, stop: float = 10, step: float = 0.01, ymin: float = -10,
                      ymax: float = 10, show_axis=True, show=True, title: str = None, with_legend=True):
        """
        Plot the solution of the linear equation, as the intersection of two linear functions ( in each side )
        """
        if is_number(self.first_side):
            first_function = Function(f"f(x) = {self.first_side}")
        else:
            first_function = Function(self.first_side)
        if is_number(self.second_side):
            second_function = Function(f"f(x) = {self.second_side}")
        else:
            second_function = Function(self.second_side)
        if title is None:
            title = f"{self.first_side}={self.second_side}"
        plot_functions([first_function, second_function], start=start, stop=stop, step=step, ymin=ymin, ymax=ymax,
                       show_axis=show_axis, show=False, title=title, with_legend=with_legend)
        x = self.solution
        if x is not None and not isinstance(x, str):
            y = first_function(x)
            plt.scatter([x], [y], color="red")  # Write the intersection with a label
            if show:
                plt.show()
            return x, y
        return x

    def _extract_variables(self):
        return extract_dict_from_equation(self._equation)

    @staticmethod
    def random_expression(values=(1, 20), items_range=(4, 7), variable=None):
        """
        Generates a string that represents a random linear expression, according to the parameters
        :param values: a tuple which contains two items: the min value, and max value possible.
        :param items_range: the numbers of item in the expression: ( min_number,max_number)
        :param variable: the variable's name. if not mentioned - it'll be chosen randomly from a list of letters
        :return:
        """
        accumulator = ""
        if not variable or not isinstance(variable, str):
            variable = random.choice(allowed_characters)
        num_of_items = random.randint(items_range[0], items_range[1])
        for i in range(num_of_items):
            if random.randint(0, 1):
                accumulator = "".join((accumulator, '-'))
            elif accumulator:  # add a '+' if but not to the first char
                accumulator = "".join((accumulator, '+'))
            coefficient = random.randint(values[0], values[1])
            if coefficient != 0:
                if random.randint(0, 1):  # add a variable
                    accumulator += f'{format_coefficient(coefficient)}{variable}'
                else:  # add a free number
                    accumulator += f"{coefficient}"
        return accumulator if accumulator != "" else "0"

    @staticmethod
    def random_equation(values=(1, 20), items_per_side=(4, 7), digits_after=2, get_solution=False, variable=None,
                        get_variable=False):
        """
        generates a random equation
        :param values: the range of the values
        :param items_per_side: the range of the number of items per side
        :param digits_after: determines the maximum number of digits after the dot a __solution can contain. For example,
        if digits_after=2, and the __solution of the equation is 3.564, __equations will be randomly generated
        until a valid __solution like 5.31 will appear.
        :param get_solution:
        :return: returns a random equation, that follows by all the condition given in the parameters.
        """
        if not variable:
            variable = random.choice(['x', 'y', 'z', 't', 'y', 'm', 'n', 'k', 'a', 'b'])
        equation = f'{LinearEquation.random_expression(values=values, items_range=items_per_side, variable=variable)} '
        equation += f'= {LinearEquation.random_expression(values=values, items_range=items_per_side, variable=variable)}'
        solution = LinearEquation(equation).solve()
        solution_string = str(solution)
        for i in range(1000):  # Limit the number of tries to 1000, to prevent cases that it searches forever
            if len(solution_string[solution_string.find('.') + 1:]) <= digits_after:
                if get_solution:
                    if get_variable:
                        return equation, solution, variable
                    return equation, solution
                if get_variable:
                    return equation, variable
                return equation
            equation = f'{LinearEquation.random_expression(values=values, items_range=items_per_side, variable=variable)} '
            equation += f'= {LinearEquation.random_expression(values=values, items_range=items_per_side, variable=variable)}'
            solution = LinearEquation(equation).solve()
            solution_string = str(solution)
        if get_solution:
            if get_variable:
                return equation, solution, variable
            return equation, solution
        if get_variable:
            return equation, variable
        return equation

    @staticmethod
    def random_worksheet(path, title="Equation Worksheet", num_of_equations=10, values=(1, 20),
                         items_per_side=(4, 8), after_point=2, get_solutions=False) -> bool:
        """
        Generates a PDF page with random __equations
        :return:
        """

        equations = [LinearEquation.random_equation(values, items_per_side, after_point, get_solutions) for _ in
                     range(num_of_equations)]
        return create_pdf(path=path, title=title, lines=equations)

    @staticmethod
    def random_worksheets(path: str, num_of_pages: int = 2, equations_per_page=20, values=(1, 20),
                          items_per_side=(4, 8), after_point=1, get_solutions=False, titles=None):
        if get_solutions:
            lines = []
            for i in range(num_of_pages):
                equations, solutions = [], []
                for j in range(equations_per_page):
                    equation, solution, variable = LinearEquation.random_equation(values=values,
                                                                                  items_per_side=items_per_side,
                                                                                  digits_after=after_point,
                                                                                  get_solution=True,
                                                                                  get_variable=True)
                    equations.append(f"{j + 1}. {equation}")
                    solutions.append(f"{j + 1}. {variable} = {solution}")
                lines.extend((equations, solutions))

            if titles is None:
                titles = ['Worksheet - Linear Equations', 'Solutions'] * num_of_pages
            create_pages(path=path, num_of_pages=num_of_pages * 2, titles=titles, lines=lines)

        else:

            lines = []
            for i in range(num_of_pages):
                equations = []
                for j in range(equations_per_page):
                    equation = LinearEquation.random_equation(values=values,
                                                              items_per_side=items_per_side,
                                                              digits_after=after_point,
                                                              get_solution=False,
                                                              get_variable=False)
                    equations.append(f"{j + 1}. {equation}")
                lines.append(equations)

            if titles is None:
                titles = ['Worksheet - Linear Equations'] * num_of_pages
            create_pages(path=path, num_of_pages=num_of_pages, titles=titles, lines=lines)

    @staticmethod
    def adjusted_worksheet(title="Equation Worksheet", equations=(),
                           ) -> bool:
        """
        Creates a user-defined PDF worksheet file.
        :param title: the title of the page
        :param equations: the __equations to print out
        :return: returns True if the creation is successful, else False.
        """
        return create_pdf("test", title=title, lines=equations)

    @staticmethod
    def manual_worksheet() -> bool:
        """
        Allows the user to create a PDF worksheet file manually.
        :return: True, if the creation is successful, else False
        """
        try:
            name, title, equations = input("Worksheet's Name:  "), input("Worksheet's Title:  "), []
            print("Enter your equations. To stop, type 'stop' ")
            i = 1
            equation = input(f"{i}.  ")
            i += 1
            while equation.lower() != 'stop':
                equations.append(equation)
                equation = input(f"{i}.  ")
        except Exception as e:  # HANDLE THIS EXCEPTION PROPERLY
            warnings.warn(f"Couldn't create the pdf file due to a {e.__class__} error")
            return False
        return LinearEquation.adjusted_worksheet(title=title, equations=equations)

    def __str__(self):
        return f"{self.equation}"

    def __repr__(self):
        return f"Equation({self.equation})"

    def __copy__(self):
        return LinearEquation(self._equation)


class QuadraticEquation(Equation):

    def __init__(self, equation: str, variables: Optional[Iterable[str]] = None, strict_syntax=False):
        self.__strict_syntax = strict_syntax
        super().__init__(equation, variables)

    def _extract_variables(self):
        return ParseExpression.parse_quadratic(self.first_side, self._variables, strict_syntax=self.__strict_syntax)

    def simplified_str(self) -> str:
        if self.num_of_variables != 1:
            raise ValueError("You can only simplify quadratic equations with 1 variable in the current version")
        my_coefficients = self.coefficients()
        return ParseExpression.coefficients_to_str(my_coefficients, variable=self._variables[0])

    def solve(self, mode='complex'):
        """Solve the quadratic equation"""
        num_of_variables = len(self._variables)
        if num_of_variables == 0:
            pass
        elif num_of_variables == 1:
            x = self._variables[0]
            a, b, c = self._variables_dict[x][0], self._variables_dict[x][1], self._variables_dict['free']
            mode = mode.lower()
            if mode == 'complex':
                return solve_quadratic(a, b, c)
            elif mode == 'real':
                return solve_quadratic_real(a, b, c)
            elif mode == 'parametric':
                return solve_quadratic_params(a, b, c)
        warnings.warn(f"Cannot solve quadratic equations with more than 1 variable, but found {num_of_variables}")
        return None

    def coefficients(self):
        num_of_variables = len(self._variables)
        if num_of_variables == 0:
            return [self._variables_dict['free']]
        elif num_of_variables == 1:
            return self._variables_dict[self._variables[0]] + [self._variables_dict['free']]
        else:
            return self._variables_dict.copy()

    def __str__(self):
        return self._equation

    @staticmethod
    def random(values=(-15, 15), digits_after: int = 0, variable: str = 'x', strict_syntax=True, get_solutions=False):
        if strict_syntax:
            a = random.randint(-5, 5)
            while a == 0:
                a = random.randint(-5, 5)

            m = round(random.uniform(values[0] / a, values[1] / a), digits_after)
            while m == 0:
                m = round(random.uniform(values[0] / a, values[1] / a), digits_after)

            n = round(random.uniform(values[0] / a, values[1] / a), digits_after)
            while n == 0:
                n = round(random.uniform(values[0] / a, values[1] / a), digits_after)

            b, c = round_decimal(round((m + n) * a, digits_after)), round_decimal(round(m * n * a, digits_after))
            a_str = format_coefficient(a)
            b_str = (f"+{b}" if b > 0 else f"{b}") if b != 0 else ""
            if b_str != "":
                if b_str == '1':
                    b_str = f'+{variable}'
                elif b_str == '-1':
                    b_str = f'-{variable}'
                else:
                    b_str += variable
            c_str = (f"+{round_decimal(c)}" if c > 0 else f"{c}") if c != 0 else ""
            equation = f"{a_str}{variable}^2{b_str}{c_str} = 0"
            if get_solutions:
                return equation, (-m, -n)
            return equation
        else:
            raise NotImplementedError("Only strict_syntax=True is available at the moment.")

    @staticmethod
    def random_worksheet(path=None, title="Quadratic Equations Worksheet", num_of_equations=20,
                         solutions_range=(-15, 15), digits_after: int = 0, get_solutions=True):
        lines = []
        if get_solutions:
            equations, solutions = [], []
            for i in range(num_of_equations):
                equ, sol = QuadraticEquation.random(values=solutions_range, digits_after=digits_after,
                                                    get_solutions=True)
                equations.append(f"{i + 1}. {equ}")
                solutions.append(f"{i + 1}. {', '.join(sol)}")
            lines.extend((equations, solutions))
        else:
            equations = []
            for i in range(num_of_equations):
                equ = QuadraticEquation.random(values=solutions_range, digits_after=digits_after,
                                               get_solutions=False)
                equations.append(f"{i + 1}. {equ}")
            lines.append(equations)

        create_pdf(path=path, title=title, lines=lines)

    @staticmethod
    def random_worksheets(path=None, num_of_pages=2, equations_per_page=20, titles=None,
                          solutions_range=(-15, 15), digits_after: int = 0, get_solutions=False):
        if titles is None:
            if get_solutions:
                titles = ["Quadratic Equations Worksheet", "Solutions"] * num_of_pages
            else:
                titles = ["Quadratic Equations Worksheet"] * num_of_pages
        lines = []
        if get_solutions:
            for i in range(num_of_pages):
                equations, solutions = [], []
                for j in range(equations_per_page):
                    equ, sol = QuadraticEquation.random(values=solutions_range, digits_after=digits_after,
                                                        get_solutions=True)
                    equations.append(f"{i + 1}. {equ}")
                    solutions.append(f"{i + 1}. {', '.join(sol)}")
                lines.extend((equations, solutions))
            create_pages(path=path, num_of_pages=num_of_pages * 2, titles=titles, lines=lines)
        else:
            for i in range(num_of_pages):
                equations = []
                for j in range(equations_per_page):
                    equ = QuadraticEquation.random(values=solutions_range, digits_after=digits_after,
                                                   get_solutions=False)
                    equations.append(f"{i + 1}. {equ}")
                lines.append(equations)
            create_pages(path=path, num_of_pages=num_of_pages, titles=titles, lines=lines)

    def __repr__(self):
        return f"QuadraticEquation({self._equation}, variables={self._variables})"

    def __copy__(self):
        return QuadraticEquation(equation=self._equation, variables=self._variables)


class CubicEquation(Equation):

    def __init__(self, equation: str, variables: Iterable[Optional[str]] = None, strict_syntax: bool = False):
        self.__strict_syntax = strict_syntax
        super().__init__(equation, variables)

    def _extract_variables(self):
        return ParseExpression.parse_cubic(self.first_side, self._variables, strict_syntax=self.__strict_syntax)

    def solve(self):
        a, b, c = self._variables_dict['x'][0], self._variables_dict['x'][1], self._variables_dict['x'][2]
        d = self._variables_dict['free']
        return solve_cubic(a, b, c, d)

    def coefficients(self):
        return [self._variables_dict['x'][0], self._variables_dict[1], self._variables_dict[2]]

    @staticmethod
    def random(solutions_range: Tuple[float, float] = (-15, 15), digits_after: int = 0, variable='x',
               get_solutions=False):
        result = random_polynomial(degree=3, solutions_range=solutions_range, digits_after=digits_after,
                                   variable=variable, get_solutions=get_solutions)
        if isinstance(result, str):
            return result + " = 0"
        else:
            return result[0] + "= 0", result[1]

    @staticmethod
    def random_worksheet(path=None, title=" Cubic Equations Worksheet", num_of_equations=20,
                         solutions_range=(-15, 15), digits_after: int = 0, get_solutions=False):
        PolyEquation.random_worksheet(path=path, title=title, num_of_equations=num_of_equations, degrees_range=(3, 3),
                                      solutions_range=solutions_range, digits_after=digits_after,
                                      get_solutions=get_solutions)

    @staticmethod
    def random_worksheets(path=None, num_of_pages=2, equations_per_page=20, titles=None,
                          solutions_range=(-15, 15), digits_after: int = 0, get_solutions=False):
        if titles is None:
            if get_solutions:
                titles = ['Cubic Equations Worksheet', 'Solutions'] * num_of_pages
            else:
                titles = ['Cubic Equations Worksheet'] * num_of_pages
        PolyEquation.random_worksheets(
            path=path, num_of_pages=num_of_pages, titles=titles, equations_per_page=equations_per_page,
            degrees_range=(3, 3),
            solutions_range=solutions_range, digits_after=digits_after, get_solutions=get_solutions
        )

    def __repr__(self):
        return f"CubicEquation({self._equation}, variables={self._variables})"

    def __copy__(self):
        return CubicEquation(equation=self._equation, variables=self._variables)


class QuarticEquation(Equation):
    def __init__(self, equation: str, variables: Iterable[Optional[str]] = None, strict_syntax=False):
        self.__strict_syntax = self.__strict_syntax
        super().__init__(equation, variables)

    def _extract_variables(self):
        return ParseExpression.parse_quartic(self.first_side, self._variables, strict_syntax=self.__strict_syntax)

    def solve(self):
        a, b, c = self._variables_dict['x'][0], self._variables_dict['x'][1], self._variables_dict['x'][2]
        d, e = self._variables_dict['x'][3], self._variables_dict['free']
        return solve_quartic(a, b, c, d, e)

    def coefficients(self):
        a, b, c = self._variables_dict['a'], self._variables_dict['b'], self._variables_dict['c']
        d, e = self._variables_dict['d'], self._variables_dict['e']
        return a, b, c, d, e

    @staticmethod
    def random(solutions_range: Tuple[float, float] = (-15, 15), digits_after: int = 0, variable='x',
               get_solutions=False):
        result = random_polynomial(degree=3, solutions_range=solutions_range, digits_after=digits_after,
                                   variable=variable, get_solutions=get_solutions)
        if isinstance(result, str):
            return result + " = 0"
        else:
            return result[0] + "= 0", result[1]

    @staticmethod
    def random_worksheet(path=None, title=" Cubic Equations Worksheet", num_of_equations=20,
                         solutions_range=(-15, 15), digits_after: int = 0, get_solutions=False):
        PolyEquation.random_worksheet(path=path, title=title, num_of_equations=num_of_equations, degrees_range=(3, 3),
                                      solutions_range=solutions_range, digits_after=digits_after,
                                      get_solutions=get_solutions)

    @staticmethod
    def random_worksheets(path=None, num_of_pages=2, equations_per_page=20, titles=None,
                          solutions_range=(-15, 15), digits_after: int = 0, get_solutions=False):
        if titles is None:
            if get_solutions:
                titles = ['Quartic Equations Worksheet', 'Solutions'] * num_of_pages
            else:
                titles = ['Quartic Equations Worksheet'] * num_of_pages
        PolyEquation.random_worksheets(
            path=path, num_of_pages=num_of_pages, titles=titles, equations_per_page=equations_per_page,
            degrees_range=(4, 4),
            solutions_range=solutions_range, digits_after=digits_after, get_solutions=get_solutions
        )

    def __repr__(self):
        return f"QuarticEquation({self._equation}, variables={self._variables})"

    def __copy__(self):
        return QuarticEquation(equation=self._equation, variables=self._variables)


class PolyEquation(Equation):

    def __init__(self, first_side, second_side=None, variables=None):
        self.__solution = None
        if first_side is None:
            raise TypeError("First argument in PolyEquation.__init__() cannot be None. Try using a string"
                            ", and read the documentation !")
        if second_side is None and isinstance(first_side, str):  # Handling a string as the first parameter,
            # that represents the equation
            left_side, right_side = first_side.split("=")
            self.__first_expression, self.__second_expression = Poly(left_side), Poly(right_side)
            equation = first_side
        else:  # In case both sides are entered
            try:
                if isinstance(first_side, (Mono, Poly)):  # Handling the first side of the equation
                    self.__first_expression = first_side.__copy__()  # TODO: avoid memory sharing ..
                else:
                    self.__first_expression = None
                # Handling the second side of the equation
                if isinstance(second_side, (Mono, Poly)):
                    self.__second_expression = second_side.__copy__()
                else:
                    self.__second_expression = None
                equation = "=".join((str(first_side), str(second_side)))
            except TypeError:
                raise TypeError(f"Unexpected type{type(first_side)} in PolyEquation.__init__()."
                                f"Couldn't convert the parameter to type str.")
        super().__init__(equation, variables)

    def solve(self):  # TODO: try to optimize this method ?
        return (self.__first_expression - self.__second_expression).roots()

    @property
    def solution(self):
        if self.__solution is None:
            self.__solution = self.solve()
        return self.__solution

    @property
    def first_poly(self):
        return self.__first_expression

    @property
    def second_poly(self):
        return self.__second_expression

    def _extract_variables(self):
        return extract_dict_from_equation(self._equation)

    def plot_solutions(self, start: float = -10, stop: float = 10, step: float = 0.01, ymin: float = -10, ymax=10,
                       title: str = None,
                       show_axis=True, show=True):  # TODO: check and modify
        first_func = Function(self.first_side)
        second_func = Function(self.second_side)
        plot_functions([first_func, second_func], start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title,

                       show_axis=show_axis, show=show)

    @staticmethod
    def __random_monomial(values=(1, 20), power: int = None, variable=None):
        if variable is None:
            variable = 'x'
        coefficient = random.randint(values[0], values[1])
        if coefficient == 0:
            return "0"
        elif coefficient == 1:
            coefficient = ""
        elif coefficient == -1:
            coefficient = '-'
        else:
            coefficient = f"{coefficient}"
        if power == 1:
            return f"{coefficient}{variable}"
        elif power == 0:
            return f"{coefficient}"
        return f"{coefficient}{variable}^{power}"

    @staticmethod
    def random_expression(values=(1, 10), of_order: int = None, variable=None, all_powers=False):
        if of_order is None:
            of_order = random.randint(1, 10)
        if of_order == 1:
            return LinearEquation.random_expression(values, variable=variable)
        accumulator = ''
        accumulator += '-' if random.randint(0, 1) else '+'
        accumulator = PolyEquation.__random_monomial(values, of_order, variable)
        for power in range(of_order - 1, 0, -1):
            if random.randint(0, 1) or all_powers:
                accumulator += '-' if random.randint(0, 1) else '+'
                accumulator += PolyEquation.__random_monomial(values, power, variable)
        if random.randint(0, 1) or all_powers:
            accumulator += '-' if random.randint(0, 1) else '+'
            accumulator += f"{random.randint(values[0], values[1])}"
        return accumulator

    @staticmethod
    def random_quadratic(values=(1, 20), variable=None, all_powers=False):
        return f"{PolyEquation.random_expression(values=values, of_order=2, variable=variable, all_powers=all_powers)} = 0"

    @staticmethod
    def random_equation(values=(1, 20), of_order: int = None, variable=None, all_powers=False):
        return f"{PolyEquation.random_expression(values, of_order, variable, all_powers)}={PolyEquation.random_expression(values, of_order, variable, all_powers)}"

    @staticmethod
    def random_worksheet(path=None, title="Equation Worksheet", num_of_equations=20, degrees_range=(2, 5),
                         solutions_range=(-15, 15), digits_after: int = 0, get_solutions=False):
        if get_solutions:
            expressions = [
                random_polynomial(random.randint(degrees_range[0], degrees_range[1]), solutions_range=solutions_range,
                                  digits_after=digits_after, get_solutions=get_solutions) for _ in
                range(num_of_equations)]
            equations = [f"{index + 1}. {expression[0]} = 0" for index, expression in enumerate(expressions)]
            solutions = [f"{index + 1}. " + ",".join([str(solution) for solution in expression[1]]) for
                         index, expression in enumerate(expressions)]
            create_pages(path, 2, ["Polynomial Equations Worksheet", "Solutions"], [equations, solutions])
        else:
            return create_pdf(path=path, title=title, lines=[
                f"{random_polynomial(random.randint(degrees_range[0], degrees_range[1]), solutions_range=solutions_range, digits_after=digits_after)} = 0"
                for _ in range(num_of_equations)])

    @staticmethod
    def random_worksheets(path=None, num_of_pages=2, equations_per_page=20, titles=None, degrees_range=(2, 5),
                          solutions_range=(-15, 15), digits_after: int = 0, get_solutions=False):
        if get_solutions:
            pages_list = []
            for i in range(num_of_pages):
                expressions = [
                    random_polynomial(random.randint(degrees_range[0], degrees_range[1]),
                                      solutions_range=solutions_range,
                                      digits_after=digits_after, get_solutions=True) for _ in
                    range(equations_per_page)]
                equations = [f"{index + 1}. {expression[0]} = 0" for index, expression in enumerate(expressions)]
                solutions = [f"{index + 1}. " + ",".join([str(solution) for solution in expression[1]]) for
                             index, expression in enumerate(expressions)]
                pages_list.append(equations)
                pages_list.append(solutions)
            if titles is None:
                titles = ["Polynomial Equations Worksheet", "Solutions"] * num_of_pages
            create_pages(path, num_of_pages * 2, titles, pages_list)

        else:
            pages_list = []
            for i in range(num_of_pages):
                expressions = [
                    random_polynomial(random.randint(degrees_range[0], degrees_range[1]),
                                      solutions_range=solutions_range,
                                      digits_after=digits_after, get_solutions=False) for _ in
                    range(equations_per_page)]
                equations = [f"{index + 1}. {expression[0]} = 0" for index, expression in enumerate(expressions)]
                pages_list.append(equations)
            if titles is None:
                titles = ["Polynomial Equations Worksheet"] * num_of_pages
            create_pages(path, num_of_pages, titles, pages_list)

    def to_PolyExpr(self):
        return Poly(self._equation)

    def __str__(self):
        return self._equation

    def __repr__(self):
        return f"PolyEquation({self._equation})"

    def __copy__(self):
        return PolyEquation(self._equation)


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
            first_dict = simplify_expression(side1, equation.variables_dict)
            second_dict = simplify_expression(side2, equation.variables_dict)
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


def handle_abs(expression: str):
    """
    An attempt to handle absolute values and evaluate them as necessary.
    :param expression: the expression to be processed, of type str.
    :return:
    """
    copy = expression.replace("|", "~~")
    results = {res: res[2:len(res) - 2] for res in re.findall(f'~~.*?~~', copy)}
    for old, new in results.items():
        if is_evaluatable(new):
            before_index = copy.find(old) - 1
            if before_index > 0 and copy[before_index].isalpha() or copy[before_index].isdigit():
                copy = copy.replace(old, f'*{abs(eval(new))}')

            copy = copy.replace(old, str(abs(eval(new))))
        else:

            before_index = copy.find(old) - 1
            if before_index > 0 and copy[before_index].isalpha() or copy[before_index].isdigit():
                copy = copy.replace(old, f'*abs({new})')
            copy = copy.replace(old, f'abs({new})')
    return copy


def handle_factorial(expression):
    if '!' not in expression:
        return expression

    copy1 = expression.replace(" ", "")
    results = [res for res in re.findall(f'([a-zA-Z0-9]+!|[a-zA-Z0-9]*\([^!]+\)!)', copy1)]
    for result in results:
        if result.startswith('(') and result.endswith(')'):
            result = result[1:-1]
        new = f"factorial({result[:-1]})"
        if is_evaluatable(result[:-1]):
            before_index = copy1.find(result) - 1
            value = factorial(eval(result[:-1]))
            if before_index > 0 and copy1[before_index].isalpha() or copy1[before_index].isdigit():
                copy1 = copy1.replace(result, f'*{value}')

            copy1 = copy1.replace(result, str(value))
        else:

            before_index = copy1.find(result) - 1
            if before_index > 0 and copy1[before_index].isalpha() or copy1[before_index].isdigit():
                copy1 = copy1.replace(result, f'*{new}')
            copy1 = copy1.replace(result, f'{new}')
    return copy1


def is_evaluatable(s):
    """
    Rather ugly and insecure method, but it is necessary for parts of the code.I haven't found a better
    alternative yet that doesn't limit some features.
    :param s: the expression
    :return: True if it can be evaluated, False otherwise.
    """
    try:
        eval(s)
        return True
    except:
        return False


def is_number(suspicious_string: str):
    """
    checks whether a string can be converted into float.
    :param suspicious_string: the string to be checked.
    :return: True if it can be converted, otherwise False.
    """
    try:
        val = float(suspicious_string)
        return True
    except:
        return False


def split_expression(expression: str):
    """splits the expression by delimiters, but doesn't touch what's inside parenthesis """
    delimiters = []
    for index, char in enumerate(expression):
        if char in Function.arithmetic_operations and index > 0:
            parenthesis_index, curly_index = expression[:index].rfind('('), expression[:index].rfind('{')
            closing_paranthesis_index = expression[parenthesis_index:].find(')') + parenthesis_index
            closing_curly = expression[curly_index:].find('}') + curly_index
            square_index = expression[:index].rfind('[')
            close_square = expression[curly_index:].find(']') + square_index
            if not parenthesis_index < index < closing_paranthesis_index and not curly_index < index < closing_curly and not square_index < index < close_square:
                delimiters.append(index)
    expressions = []
    if len(delimiters) > 0:
        expressions.append(expression[:delimiters[0]])
        for i in range(1, len(delimiters)):
            expressions.append(expression[delimiters[i - 1]:delimiters[i]])
        expressions.append(expression[delimiters[len(delimiters) - 1]:])
    else:
        expressions.append(expression)
    return [expression for expression in expressions if expression != ""]


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
            self.__func = clean_from_spaces(func).replace("^", "**")
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
                self.__func_signature = clean_from_spaces(self.__func[:first_equal_index])
                self.__func_expression = clean_from_spaces(self.__func[first_equal_index + 1:])
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

    def chain(self, other_func: "Optional[Function,str]"):
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


class Exponent(IExpression):
    """
    This class enables you to represent expressions such as x^x, e^x, (3x)^sin(x), etc.
    """
    __slots__ = ['_coefficient', '_base', '_power']

    def __init__(self, base: Union[IExpression, float], power: Union[IExpression, float, int],
                 coefficient: Optional[Union[int, float, IExpression]] = None, gen_copies=True):
        if isinstance(base, IExpression):
            self._base = base.__copy__() if gen_copies else base
        elif isinstance(base, (int, float)):
            self._base = Mono(base)
        else:
            raise TypeError(f"Exponent.__init__(): Invalid type {type(base)} for parameter 'base'.")

        if isinstance(power, IExpression):
            self._power = power.__copy__() if gen_copies else power
        elif isinstance(power, (int, float)):
            self._power = Mono(power)
        else:
            raise TypeError(f"Exponent.__init__(): Invalid type {type(power)} for parameter 'power'.")

        if coefficient is None:
            self._coefficient = Mono(1)
        elif isinstance(coefficient, IExpression):
            if gen_copies:
                self._coefficient = coefficient.__copy__()
            else:
                self._coefficient = coefficient
        elif isinstance(coefficient, (int, float)):
            self._coefficient = Mono(coefficient)
        else:
            raise TypeError(f"Invalid type for coefficient of Exponent object: '{coefficient}'.")

    def __add_or_sub(self, other, operation='+'):
        if other == 0:
            return self
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            if my_evaluation is None:
                if operation == '+':
                    return ExpressionSum((self, Mono(other)))
                return ExpressionSum((self, Mono(-other)))
            else:
                if operation == '+':
                    return Mono(my_evaluation + other)
                return Mono(my_evaluation - other)

        elif isinstance(other, IExpression):
            other_evaluation = self.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                if operation == '+':
                    return Mono(my_evaluation + other_evaluation)
                return Mono(my_evaluation - other_evaluation)
            elif other_evaluation is not None:
                if operation == '+':
                    return ExpressionSum((self, Mono(other_evaluation)))
                return ExpressionSum((self, Mono(-other)))
            else:  # both expressions cannot be evaluated into numbers
                if not isinstance(other, Exponent):
                    if operation == '+':
                        return ExpressionSum((self, other))
                    return ExpressionSum((self, -other))
                else:  # If we're dealing with another exponent expression.
                    if self._power == other._power and self._base == other._base:
                        # if the exponents have the same base and powers.
                        if operation == '+':
                            self._coefficient += other._coefficient
                        else:
                            self._coefficient -= other._coefficient
                        return self
                    elif False:  # TODO: check for relations in the base and power pairs?
                        pass
                    else:
                        if operation == '+':
                            return ExpressionSum((self, other))
                        return ExpressionSum((self, -other))

    def __iadd__(self, other: "Union[IExpression, int, float]"):  # TODO: further implement
        return self.__add_or_sub(other, operation='+')

    def __isub__(self, other):
        return self.__add_or_sub(other, operation='-')

    def __imul__(self, other: Union[int, float, IExpression]):  # TODO: further implement
        if other == 0:
            return Mono(0)

        if other == self.base:
            self._power += 1
            return self
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            if my_evaluation is not None:
                return Mono(my_evaluation * other)
            self.multiply_by_number(other)
            return self
        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                return Mono(my_evaluation * other_evaluation)
            elif other_evaluation is not None:
                self.multiply_by_number(other_evaluation)
                return self
            else:
                if isinstance(other, Exponent):
                    if self._power == other._power:
                        self._base *= other._base
                        return self
                    elif False:  # Relation between the powers ( for example x and 2x)
                        pass
                    else:
                        return ExpressionMul((self, other))

    def __mul__(self, other: Union[int, float, IExpression]):
        return self.__copy__().__imul__(other)

    def multiply_by_number(self, number: Union[int, float]):
        self._coefficient *= number

    def divide_by_number(self, number: Union[int, float]):
        if number == 0:
            raise ZeroDivisionError("Cannot divide an expression by 0")
        self._coefficient /= number

    def __itruediv__(self, other: Union[int, float, IExpression]):
        return Fraction(self, other)

    def __ipow__(self, other: Union[IExpression, int, float]):
        if other == 0:
            return self._coefficient.__copy__()  # because then the expression would be: coefficient * expression ^ 0
        self._power *= other
        return self

    def __pow__(self, power: Union[int, float, IExpression], modulo=None):
        return self.__copy__().__ipow__(power)

    def __neg__(self):
        copy_of_self = self.__copy__()
        copy_of_self._coefficient *= -1
        return copy_of_self

    def to_dict(self):
        return {
            "type": "Factorial",
            "coefficient": self._coefficient.to_dict(),
            "base": self._base.to_dict(),
            "power": self._power.to_dict()
        }

    @staticmethod
    def from_dict(given_dict: dict):
        base_obj = create_from_dict(given_dict['base'])
        coefficient_obj = create_from_dict(given_dict['coefficient'])
        power_obj = create_from_dict(given_dict['power'])
        return Exponent(base=base_obj, power=power_obj, coefficient=coefficient_obj)

    def derivative(self):  # TODO: improve this method
        my_variables = self.variables
        variables_length = len(my_variables)
        if variables_length == 0:
            # Free number, then the derivative is 0
            return Mono(0)
        elif variables_length == 1:
            coefficient_eval = self._coefficient.try_evaluate()
            base_eval = self._base.try_evaluate()
            power_eval = self._power.try_evaluate()
            if None not in (coefficient_eval, base_eval, power_eval) or coefficient_eval == 0 or base_eval == 0:
                return Mono(0)

            if power_eval is not None and power_eval == 0:
                return self._coefficient.derivative()  # for instance: x**2 ^0 -> 1

            if coefficient_eval is not None:
                if power_eval is not None:  # cases such as 3x^2 or  5sin(2x)^4
                    expression = (coefficient_eval * self._base ** power_eval).derivative()
                    if hasattr(expression, "derivative"):
                        return expression.derivative()
                    warnings.warn("This kind of derivative isn't supported yet...")
                    return None

                elif base_eval is not None:  # examples such as 2^x
                    if base_eval < 0:
                        warnings.warn(f"The derivative of this expression is undefined")
                        return None
                    return self * self._coefficient.derivative() * ln(base_eval)
        else:
            raise ValueError("For derivatives with more than 1 variable, use partial derivatives")

    @property
    def variables(self):
        my_variables = self._coefficient.variables
        my_variables.update(self._base.variables)
        my_variables.update(self._power.variables)
        return my_variables

    def partial_derivative(self):
        raise NotImplementedError("This feature is not supported yet. Stay tuned for the next versions.")

    def integral(self):
        raise NotImplementedError("This feature is not supported yet. Stay tuned for the next versions.")

    @property
    def base(self):
        return self._base

    @property
    def power(self):
        return self._power

    def assign(self, **kwargs):
        self._coefficient.assign(**kwargs)
        self._base.assign(**kwargs)
        self._power.assign(**kwargs)

    def when(self, **kwargs):
        copy_of_self = self.__copy__()
        copy_of_self.assign(**kwargs)
        return copy_of_self

    def simplify(self) -> None:  # TODO: improve this somehow....
        self._coefficient.simplify()
        self._base.simplify()
        self._power.simplify()

    def try_evaluate(self) -> Optional[Union[int, float]]:
        if self._coefficient == 0:
            return 0
        coefficient_evaluation = self._coefficient.try_evaluate()
        if coefficient_evaluation is None:
            return None
        power_evaluation = self._power.try_evaluate()
        if power_evaluation is None:
            return None
        if power_evaluation == 0:  # 3*x^0 for example will be evaluated to 3
            return coefficient_evaluation
        base_evaluation = self._base.try_evaluate()
        if base_evaluation is None:
            return None
        return coefficient_evaluation * (base_evaluation ** power_evaluation)

    def __eq__(self, other: Union[IExpression, int, float]):
        if other is None:
            return False
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            return my_evaluation == other
        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                return my_evaluation == other_evaluation
            elif my_evaluation is other_evaluation is None:
                # Compare between the objects
                if isinstance(other, Exponent):
                    equal_coefficients = self._coefficient == other._coefficient
                    equal_bases = self._base == other._base
                    equal_powers = self._power == other._power
                    if equal_coefficients and equal_bases and equal_powers:
                        return True
                    # TODO: check for other cases where the expressions will be equal, such as 2^(2x) and 4^x
                else:
                    # TODO: check for cases when other types of objects are equal, such as x^2 (Mono) and x^2 (Exponent)
                    return False
            else:  # One of the expressions is a number, while the other is an algebraic expression
                return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __copy__(self):
        return Exponent(base=self._base, power=self._power, coefficient=self._coefficient)

    def __str__(self):
        if self._coefficient == 0:
            return "0"
        if self._power == 0:
            return self._coefficient.__str__()
        if self._coefficient == 1:
            coefficient_str = ""
        elif self._coefficient == -1:
            coefficient_str = "-"
        else:
            coefficient_str = f"{self._coefficient.__str__()}*"
        base_string, power_string = apply_parenthesis(self._base.__str__()), apply_parenthesis(self._power.__str__())
        return f"{coefficient_str}{base_string}^{power_string}"


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


class Vector:
    def __init__(self, direction_vector=None, start_coordinate=None, end_coordinate=None):
        """
        Creates a new Vector object.


        :param direction_vector: For example, the direction vector of a vector that starts from ( 1,1,1 ) and ends with (4,4,4) is (3,3,3)

        :param start_coordinate: The coordinate that represents the origin of the vector on an axis system.
        :param end_coordinate: The coordinate that represents the end of the vector on an axis system.
        """
        if start_coordinate is not None and end_coordinate is not None:

            if len(start_coordinate) != len(end_coordinate):
                raise ValueError("Cannot handle with vectors with different dimensions in this version.")
            try:
                self._start_coordinate = list(start_coordinate)
            except TypeError:  # If the parameters cannot be converted into lists
                raise TypeError(f"Couldn't convert from type {type(start_coordinate)} to list."
                                f"expected types were tuple,list, set, and dict.")
            try:
                self._end_coordinate = list(end_coordinate)
            except TypeError:
                raise TypeError(f"Couldn't convert from type {type(start_coordinate)} to list."
                                f"expected types were tuple,list, set, and dict.")

            self._direction_vector = list(
                [end_coordinate[i] - start_coordinate[i] for i in range(len(start_coordinate))])
        elif direction_vector is not None and start_coordinate is not None:
            self._start_coordinate = list(start_coordinate)
            self._direction_vector = list(direction_vector)
            self._end_coordinate = [self._start_coordinate[i] + self._direction_vector[i] for i in
                                    range(len(self._start_coordinate))]
        elif direction_vector is not None and end_coordinate is not None:
            self._end_coordinate = list(end_coordinate)
            self._direction_vector = list(direction_vector)
            self._start_coordinate = [self._end_coordinate[i] - self._direction_vector[i] for i in
                                      range(len(self._end_coordinate))]
        elif direction_vector is not None:
            self._direction_vector = list(direction_vector)
            self._end_coordinate = self._direction_vector.copy()
            self._start_coordinate = [0 for _ in range(len(self._end_coordinate))]

    @property
    def start_coordinate(self):
        return self._start_coordinate

    @property
    def end_coordinate(self):
        return self._end_coordinate

    @property
    def direction(self):
        return self._direction_vector

    def plot(self, show=True, arrow_length_ratio: float = 0.05, fig=None, ax=None):
        start_length, end_length = len(self._start_coordinate), len(self._end_coordinate)
        if start_length == end_length == 2:
            plot_vector_2d(
                self._start_coordinate[0], self._start_coordinate[1], self._direction_vector[0],
                self._direction_vector[1], show=show, fig=fig, ax=ax)
        elif start_length == end_length == 3:
            u, v, w = self._direction_vector[0], self._direction_vector[1], self._direction_vector[2]
            start_x, start_y, start_z = self._start_coordinate[0], self._start_coordinate[1], self._start_coordinate[
                2]
            plot_vector_3d(
                (start_x, start_y, start_z), (u, v, w), arrow_length_ratio=arrow_length_ratio, show=show, fig=fig,
                ax=ax)
        else:
            raise ValueError(
                f"Cannot plot a vector with {start_length} dimensions. (Only 2D and 3D plotting is supported")

    def length(self):
        return round_decimal(sqrt(reduce(lambda a, b: a ** 2 + b ** 2, self._direction_vector)))

    def multiply(self, other: "Union[int, float, IExpression, Iterable, Vector, VectorCollection, ]"):
        if isinstance(other, (Vector, Iterable)) and not isinstance(other, (VectorCollection, IExpression)):
            return self.scalar_product(other)
        elif isinstance(other, (int, float, IExpression)):
            return self.multiply_all(other)
        else:
            raise TypeError(f"Vector.multiply(): expected types Vector/tuple/list/int/float but got {type(other)}")

    def multiply_all(self, number: Union[int, float, IExpression]):
        """ Multiplies the vector by the given expression, and returns the current vector ( Which was not copied ) """
        for index in range(len(self._direction_vector)):
            self._direction_vector[index] *= number
        self.__update_end()
        return self

    def scalar_product(self, other: Iterable):
        """

        :param other: other vector
        :return: returns the scalar multiplication of two vectors
        :raise: raises an Exception when the type of other isn't tuple or Vector
        """
        if isinstance(other, Iterable):
            other = Vector(other)
        if isinstance(other, Vector):
            scalar_result = 0
            for a, b in zip(self._direction_vector, other._direction_vector):
                scalar_result += a * b
            return scalar_result
        else:
            raise TypeError(f"Vector.scalar_product(): expected type Vector or tuple, but got {type(other)}")

    # TODO: implement it to work on all vectors, and not to break when zeroes are entered.
    def equal_direction_ratio(self, other):
        """

        :param other: another vector
        :return: True if the two vectors have the same ratio of directions, else False
        """
        try:
            if len(self._direction_vector) != len(other._direction_vector):
                return False
            if len(self._direction_vector) == 0:
                return False
            elif len(self._direction_vector) == 1:
                return self._direction_vector[0] == other._direction_vector[0]
            else:
                ratios = []
                for a, b in zip(self._direction_vector, other._direction_vector):
                    if a == 0 and b != 0 or b == 0 and a != 0:
                        return False  # if that's the case, the vectors can't be equal
                    if not a == b == 0:
                        ratios.append(a / b)
                if ratios:
                    return all(x == ratios[0] for x in ratios)
                return True  # if entered here the list of ratios is empty, meaning all is 0
        except ZeroDivisionError:
            warnings.warn("Cannot check whether the vectors' directions are equal because of a ZeroDivisionError")

    @classmethod
    def random_vector(cls, numbers_range: Tuple[int, int], num_of_dimensions: int = None):
        """
        Generate a random vector object.

        :param numbers_range: the range of possible values
        :param num_of_dimensions: the number of dimensions of the vector. If not set, a number will be chosen
        :return: Returns a Vector object, (or Vector2D or Vector3D objects).
        """
        if cls is Vector2D:
            num_of_dimensions = 2
        elif cls is Vector3D:
            num_of_dimensions = 3

        if num_of_dimensions is None:
            num_of_dimensions = random.randint(2, 9)

        direction = [random.randint(numbers_range[0], numbers_range[1]) for _ in range(num_of_dimensions)]
        start = [random.randint(numbers_range[0], numbers_range[1]) for _ in range(num_of_dimensions)]

        return cls(direction_vector=direction, start_coordinate=start)

    def general_point(self, var_name: str = 'x'):
        """
        Generate an algebraic expression that represents any dot on the vector

        :param var_name: the name of the variable (str)
        :return: returns a list of algebraic expressions
        """
        variable = Var(var_name)
        lst = [start + variable * direction for start, direction in
               zip(self._start_coordinate, self._direction_vector)]
        return lst

    def intersection(self, other: "Union[Vector, VectorCollection, Surface]", get_points=False):
        if self._direction_vector == other._direction_vector:
            print("The vectors have the same directions, unhandled case for now")
            return

        if isinstance(other, Vector):
            my_general, other_general = self.general_point('t'), other.general_point('s')
            solutions_dict = LinearSystem(
                (f"{expr1}={expr2}" for expr1, expr2 in zip(my_general, other_general))).get_solutions()
            if not solutions_dict:
                print("Something went wrong, no solutions were found for t and s !")
            t, s = solutions_dict['t'], solutions_dict['s']
            for expression in my_general:
                expression.assign(t=t)
            if get_points:
                return Point((expression.expressions[0].coefficient for expression in my_general))
            return [expression.expressions[0].coefficient for expression in my_general]
        elif isinstance(other, VectorCollection):
            return any(self.intersection(other_vector) for other_vector in other.vectors)
        elif isinstance(other, Surface):
            return other.intersection(self)
        else:
            raise TypeError(f"Invalid type {type(other)} for searching intersections with a vector. Expected types:"
                            f" Vector, VectorCollection, Surface.")

    def equal_lengths(self, other):
        """

        :param other: another vector
        :return: True if self and other have the same lengths, else otherwise
        """
        return self.length() == other.length()

    @staticmethod
    def fill(dimension: int, value) -> "Vector":
        return Vector(direction_vector=[value for _ in range(dimension)])

    @staticmethod
    def fill_zeros(dimension: int) -> "Vector":
        return Vector.fill(dimension, 0)

    @staticmethod
    def fill_ones(dimension: int) -> "Vector":
        return Vector.fill(dimension, 1)

    def __copy__(self):
        return Vector(start_coordinate=self._start_coordinate, end_coordinate=self._end_coordinate)

    def __eq__(self, other: "Union[Vector, VectorCollection]"):
        """ Returns whether the two vectors have the same starting position, length, and ending position."""
        if other is None:
            return False
        if isinstance(other, Vector):
            return self._direction_vector == other._direction_vector  # Equate by the direction vector.

        elif isinstance(other, VectorCollection):
            if other.num_of_vectors != 1:
                return False
            return self == other.vectors[0]

        else:
            raise TypeError(f"Invalid type {type(other)} for equating vectors.")

    def __ne__(self, other):
        return not self.__eq__(other)

    def __update_end(self):
        self._end_coordinate = [start + direction for start, direction in
                                zip(self._start_coordinate, self._direction_vector)]

    def __imul__(self, other: "Union[IExpression, int, float, Vector, VectorCollection, Surface]"):
        return self.multiply(other)

    def __mul__(self, other):
        return self.__copy__().__imul__(other)

    def __rmul__(self, other):
        return self.multiply(other)

    def power_by_vector(self, other: "Union[Iterable, Vector]"):
        if not isinstance(other, (Iterable, Vector)):
            raise TypeError(f"Invalid type '{type(other)} to raise a vector by another vector ( vector1 ** vector2 )'")
        if isinstance(other, Iterator):
            other = list(other)  # Convert the iterator to list ..
        other_items = other._direction_vector if isinstance(other, Vector) else other
        return Matrix(
            matrix=[[my_item ** other_item for other_item in other_items]
                    for my_item in self._direction_vector]
        )

    def power_by_expression(self, expression: Union[int, float, IExpression]):
        for index in range(len(self._direction_vector)):
            self._direction_vector[index] **= expression
        self.__update_end()
        return self

    def power_by(self, other: "Union[int, float, IExpression, Iterable, Vector, VectorCollection]"):
        return self.__ipow__(other)

    def __ipow__(self, other: "Union[int, float, IExpression, Iterable, Vector]"):
        if isinstance(other, (int, float, IExpression)):
            return self.power_by_expression(other)
        elif isinstance(other, (Vector, Iterable)) and not isinstance(other, (IExpression, VectorCollection)):
            return self.power_by_vector(other)
        else:
            raise TypeError(f"Invalid type '{type(other)}' for raising a Vector by a power.")

    def __pow__(self, other: float):
        return self.__copy__().__ipow__(other)

    def __iadd__(self, other: "Union[Vector, VectorCollection, Surface, IExpression, int, float]"):
        if isinstance(other, Vector):
            for index, other_coordinate in zip(range(len(self._direction_vector)), other._direction_vector):
                self._direction_vector[index] += other_coordinate
            self.__update_end()
            return self
        elif isinstance(other, (IExpression, int, float)):
            for index in range(len(self._direction_vector)):
                self._direction_vector[index] += other
            self.__update_end()
            return self
        elif isinstance(other, VectorCollection):
            other_copy = other.__copy__()
            other_copy.append(self)
            return other_copy
        else:
            raise TypeError(f"Invalid type {type(other)} for adding vectors")

    def __isub__(self, other: "Union[Vector, VectorCollection]"):
        if isinstance(other, Vector):
            for index, other_coordinate in zip(range(len(self._direction_vector)), other._direction_vector):
                self._direction_vector[index] -= other_coordinate
            self._end_coordinate = [start + direction for start, direction in
                                    zip(self._start_coordinate, self._direction_vector)]
            return self
        elif isinstance(other, (IExpression, int, float)):
            for index in range(len(self._direction_vector)):
                self._direction_vector[index] -= other
            self._end_coordinate = [start + direction for start, direction in
                                    zip(self._start_coordinate, self._direction_vector)]
            return self
        elif isinstance(other, VectorCollection):
            other_copy = other.__copy__()
            other_copy.append(-self)
            return other_copy

        else:
            raise TypeError(f"Invalid type {type(other)} for adding vectors")

    def __sub__(self, other: "Union[Vector, VectorCollection]"):
        return self.__copy__().__isub__(other)

    def __rsub__(self, other: "Union[Vector, VectorCollection]"):
        return self.__neg__().__iadd__(other)

    def __add__(self, other: "Union[Vector, VectorCollection]"):
        return self.__copy__().__iadd__(other)

    def __radd__(self, other: "Union[Vector, VectorCollection]"):
        return self.__copy__().__iadd__(other)

    def __neg__(self):
        return Vector(direction_vector=[-x for x in self._direction_vector],
                      start_coordinate=self._end_coordinate,
                      end_coordinate=self._start_coordinate)

    def __str__(self):
        """
        :return: string representation of the vector
        """
        return f"""start: {self._start_coordinate} end: {self._end_coordinate} direction: {self._direction_vector} """

    def __repr__(self):
        """

        :return: returns a string representation of the object's constructor
        """
        return f'Vector(start_coordinate={self._start_coordinate},end_coordinate={self._end_coordinate})'

    def __abs__(self):
        """
        :return: returns a vector with absolute values, preserves the starting coordinate but changes the ending point
        """
        return Vector(direction_vector=[abs(x) for x in self._direction_vector],
                      start_coordinate=self._start_coordinate)

    def __len__(self):
        return self.length()


class Vector2D(Vector, IPlottable):
    def __init__(self, x, y, start_coordinate=None, end_coordinate=None):
        if start_coordinate is not None:
            if len(start_coordinate) != 2:
                raise ValueError(f"Vector2D object can only receive 2D coordinates: got wrong 'start_coordinate' param")
        if end_coordinate is not None:
            if len(end_coordinate) != 2:
                raise ValueError(f"Vector2D object can only receive 2D coordinates: got wrong 'end_coordinate' param")

        super().__init__(direction_vector=(x, y), start_coordinate=start_coordinate,
                         end_coordinate=end_coordinate)

    @property
    def x_step(self):
        return self._direction_vector[0]

    @property
    def y_step(self):
        return self._direction_vector[1]

    def plot(self, show=True, arrow_length_ratio: float = 0.05):
        plot_vector_2d(
            self._start_coordinate[0], self._start_coordinate[1], self._direction_vector[0],
            self._direction_vector[1], show=show)


class Vector3D(Vector, IPlottable):
    def __init__(self, x, y, z, start_coordinate=None, end_coordinate=None):
        if start_coordinate is not None:
            if len(start_coordinate) != 3:
                raise ValueError(f"Vector3D object can only receive 3D coordinates: got wrong 'start_coordinate' param")
        if end_coordinate is not None:
            if len(end_coordinate) != 3:
                raise ValueError(f"Vector3D object can only receive 3D coordinates: got wrong 'end_coordinate' param")

        super(Vector3D, self).__init__(direction_vector=(x, y, z), start_coordinate=start_coordinate,
                                       end_coordinate=end_coordinate)

    @property
    def x_step(self):
        return self._direction_vector[0]

    @property
    def y_step(self):
        return self._direction_vector[1]

    @property
    def z_step(self):
        return self._direction_vector[2]

    def plot(self, show=True, arrow_length_ratio: float = 0.05, fig=None, ax=None):
        u, v, w = self._direction_vector[0], self._direction_vector[1], self._direction_vector[2]
        start_x, start_y, start_z = self._start_coordinate[0], self._start_coordinate[1], self._start_coordinate[
            2]
        plot_vector_3d(
            (start_x, start_y, start_z), (u, v, w), arrow_length_ratio=arrow_length_ratio, show=show, fig=fig, ax=ax)


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


class VectorCollection:
    def __init__(self, *vectors):
        self.__vectors = []
        for vector in vectors:
            if isinstance(vector, Vector):
                self.__vectors.append(vector)
            elif isinstance(vector, Iterable):
                self.__vectors.append(Vector(vector))
            else:
                raise TypeError(f"Encountered invalid type {type(vector)} while building a vector collection.")

    @property
    def vectors(self):
        return self.__vectors

    @property
    def num_of_vectors(self):
        return len(self.__vectors)

    @vectors.setter
    def vectors(self, vectors):
        if isinstance(vectors, (list, set, tuple, Vector)):
            vectors = VectorCollection(vectors)

        if isinstance(vectors, VectorCollection):
            self.__vectors = vectors
        else:
            raise TypeError(
                f"Unexpected type {type(vectors)} in the setter property of vectors in class VectorCollection"
                f".\nExpected types VectorCollection, Vector, tuple, list, set")

    def append(self,
               vector: "Union[Vector, Iterable[Union[Vector, IExpression, VectorCollection, int, float, Iterable]], "
                       "VectorCollection]"):
        """ Append vectors to the collection of vectors """
        if isinstance(vector, Vector):  # if the parameter is a vector
            self.__vectors.append(vector)

        elif isinstance(vector, VectorCollection):
            self.__vectors.extend(vector)

        elif isinstance(vector, Iterable) and not isinstance(vector, IExpression):
            # if the parameter is an Iterable object
            for item in vector:
                if isinstance(item, Vector):
                    self.__vectors.append(item)
                elif isinstance(item, Iterable):
                    self.__vectors.append(Vector(item))
                elif isinstance(item, VectorCollection):
                    self.__vectors.extend(item)
                else:
                    raise TypeError(f"Invalid type {type(vector)} for appending into a VectorCollection")

        else:
            raise TypeError(f"Invalid type {type(vector)} for appending into a VectorCollection")

    def plot(self):  #
        num_of_vectors = len(self.__vectors)
        if num_of_vectors > 0:
            if len(self.__vectors[0].start_coordinate) == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                for vector in self.__vectors:
                    start = (vector.start_coordinate[0], vector.start_coordinate[1], vector.start_coordinate[2])
                    end = (vector.end_coordinate[0], vector.end_coordinate[1], vector.end_coordinate[2])
                    plot_vector_3d(start, end, fig=fig, ax=ax, show=False)
                min_x, max_x, min_y, max_y, min_z, max_z = _get_limits_vectors_3d(self.__vectors)
                ax.set_xlim([min_x, max_x])
                ax.set_ylim([min_y, max_y])
                ax.set_zlim([min_z, max_z])
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                for vector in self.__vectors:
                    vector.plot(show=False, fig=fig, ax=ax)

                min_x, max_x, min_y, max_y = _get_limits_vectors_2d(self.__vectors)
                ax.set_xlim([min_x, max_x])
                ax.set_ylim([min_y, max_y])
        plt.show()

    # TODO: should modify it or delete or leave it like this?
    def filter(self, predicate: Callable[[Any], bool] = lambda x: bool(x)):
        return filter(predicate, self.__vectors)

    def map(self, func: Callable):
        return map(func, self.__vectors)

    def longest(self, get_index=False, remove=False):
        """
        returns the longest vector in the collection

        :param get_index: if True, returns a tuple: (index,longest_vector)
        :param remove: if True, removes the longest vector from the collection
        :return: depends whether get_index evaluates to True or False
        """
        longest_vector = max(self.__vectors, key=operator.attrgetter('_Vector__direction_vector'))
        if not (get_index or remove):
            return longest_vector
        index = self.__vectors.index(longest_vector)
        if remove:
            self.__vectors.pop(index)
        if get_index:
            return index, longest_vector
        return longest_vector

    def shortest(self, get_index=False, remove=False):
        """
        returns the shortest vector in the collection

        :param get_index: if True, returns a tuple: (index,shortest_vector)
        :param remove: if True, removes the shortest vector from the collection
        :return: depends whether get_index evaluates to True or False
        """
        shortest_vector = min(self.__vectors, key=operator.attrgetter('_Vector__direction_vector'))
        if not (get_index or remove):
            return shortest_vector
        index = self.__vectors.index(shortest_vector)
        if remove:
            self.__vectors.pop(index)
        if get_index:
            return index, shortest_vector
        return shortest_vector

    def find(self, vec: Vector):
        for index, vector in enumerate(self.__vectors):
            if vector.__eq__(vector) or vector is vec:
                return index
        return -1

    def nlongest(self, n: int):
        """returns the n longest vector for an integer n"""
        return [self.longest(remove=True) for _ in range(n)]

    def nshortest(self, n: int):
        """returns the n shortest vector for an integer n"""
        return [self.shortest(remove=True) for _ in range(n)]

    def sort_by_length(self, reverse=False):
        self.__vectors.sort(key=lambda vector: vector.length(), reverse=reverse)

    def pop(self, index: int = -1):
        return self.__vectors.pop(index)

    def __iadd__(self, other: "Union[Vector, VectorCollection, Iterable]"):
        self.append(other)
        return self

    def __add__(self, other: "Union[Vector, VectorCollection, Iterable]"):
        return self.__copy__().__iadd__(other)

    def __radd__(self, other):
        return self.__copy__().__iadd__(other)

    def __imul__(self, other):
        if isinstance(other, Vector):
            pass
        elif isinstance(other, VectorCollection):
            pass
        else:
            pass

    def __mul__(self, other):
        return self.__copy__().__imul__(other)

    def __itruediv__(self, other: Union[int, float, IExpression]):
        if not isinstance(other, (int, float, IExpression)):
            raise TypeError(f"Invalid type for dividing a VectorCollection object: {type(other)}. Expected a number"
                            f"or an algebraic expression.")
        if other == 0:
            raise ValueError("Cannot divide a VectorCollection object by 0")

        for i in range(len(self.__vectors)):
            self.__vectors[i] /= other

    def __truediv__(self, other):
        return self.__copy__().__itruediv__(other)

    def __bool__(self):
        return bool(self.__vectors)

    def to_matrix(self):
        return Matrix([[copy_expression(expression) for expression in vector.direction] for vector in self.__vectors])

    # TODO: check this method ..
    def __eq__(self, other: "Union[Vector, VectorCollection, Iterable]"):  # TODO: use other type hint than Iterable
        if other is None:
            return False
        if not isinstance(other, (Vector, VectorCollection)):
            if isinstance(other, Iterable):
                if isinstance(other[0], Iterable):
                    try:
                        other = VectorCollection(other)
                    except (ValueError, TypeError):
                        try:
                            other = Vector(other)
                        except (ValueError, TypeError):
                            raise ValueError("Invalid value for equating VectorCollection objects.")
                else:
                    try:
                        other = Vector(other)
                    except (ValueError, TypeError):
                        raise ValueError("Invalid value for equating VectorCollection objects.")
            else:
                raise TypeError(f"Invalid type '{type(other)}' for equating VectorCollection objects.")

        if isinstance(other, Vector) and len(self.__vectors) == 1:
            return self.__vectors[0] == other
            # comparison between a vector to a single item list should return true
        elif isinstance(other, VectorCollection):
            if len(self.__vectors) != len(other.__vectors):
                return False
            for vec in self.__vectors:
                if other.__vectors.count(vec) != self.__vectors.count(vec):
                    return False
            return True
        else:
            raise TypeError(f"Invalid type {type(other)} for equating VectorCollection objects.")

    def __ne__(self, other: "Union[Vector, VectorCollection, Iterable]"):
        return not self.__eq__(other)

    def __getitem__(self, item):
        return self.__vectors.__getitem__(item)

    def __setitem__(self, key, value):
        return self.__vectors.__setitem__(key, value)

    def __delitem__(self, key):
        return self.__delitem__(key)

    def __copy__(self):
        return VectorCollection(self.__vectors)

    def __contains__(self, other: Vector):
        if isinstance(other, Vector):
            return bool([vector for vector in self.__vectors if vector.__eq__(other)])

    def __iter__(self):
        self.__current_index = 0
        return self

    def __next__(self):
        if self.__current_index < len(self.__vectors):
            x = self.__vectors[self.__current_index]
            self.__current_index += 1
            return x
        else:
            raise StopIteration

    def __len__(self):
        """ number of vectors that the collection contains"""
        return len(self.__vectors)

    def total_number_of_items(self):
        """ Total number of items in all of the vectors. """
        return sum(len(vector) for vector in self.__vectors)


def surface_from_str(input_string: str, get_coefficients=False):
    first_side, second_side = input_string.split('=')
    first_coefficients = re.findall(number_pattern, first_side)

    for index in range(0, 4 - len(first_coefficients), 1):  # format it to be 4 coefficients
        first_coefficients.append('0')
    second_coefficients = re.findall(number_pattern, second_side)
    for index in range(0, 4 - len(second_coefficients), 1):  # format it to be 4 coefficients
        second_coefficients.append('0')

    for first_index, second_value in zip(range(len(first_coefficients)), second_coefficients):
        first_coefficients[first_index] = float(first_coefficients[first_index]) - float(second_value)
    if get_coefficients:
        return first_coefficients
    return Surface(first_coefficients)


class Surface:
    """
    represents a surface of the equation ax+by+cz+d = 0, where (a,b,c) is the perpendicular of the surface, and d
    is a free number.
    """

    def __init__(self, coefs):
        if isinstance(coefs, str):
            self.__a, self.__b, self.__c, self.__d = surface_from_str(coefs, get_coefficients=True)
        elif isinstance(coefs,
                        Iterable):  # TODO: change to a more specific type hint later, but still one that accepts generators
            coefficients = [coef for coef in coefs]
            if len(coefficients) == 4:
                self.__a, self.__b, self.__c, self.__d = coefficients[0], coefficients[1], coefficients[2], \
                                                         coefficients[3]
            elif len(coefficients) == 3:
                self.__a, self.__b, self.__c, self.__d = coefficients[0], coefficients[1], coefficients[2], 0
            else:
                raise ValueError(
                    f"Invalid number of coefficients in coefficients of surface. Got {len(coefficients)}, expected 4 or 3")

    @property
    def a(self):
        return self.__a

    @property
    def b(self):
        return self.__b

    @property
    def c(self):
        return self.__c

    @property
    def d(self):
        return self.__d

    def intersection(self, vector: Vector, get_point=False):  # TODO: check if intersects or that the continuation does
        """
        Finds the intersection between a surface and a vector

        :param get_point: If set to True, a point that represents the intersection will be returned instead of a list that represents the coordinates of the intersection. Default value is false.

        :param vector: An object of type Vector.
        :return: Returns a list of the coordinates of the intersection. If get_point = True, returns corresponding
        point object.
        """
        general_point = vector.general_point('t')
        expression = self.__a * general_point[0] + self.__b * general_point[1] + self.__c + general_point[2] + self.__d
        t_solution = LinearEquation(f"{expression} = 0", variables=('t',), calc_now=True).solution
        for polynomial in general_point:
            polynomial.assign(t=t_solution)
        if get_point:
            return Point((polynomial.expressions[0].coefficient for polynomial in general_point))
        return [polynomial.expressions[0].coefficient for polynomial in general_point]  # TODO: check if this works

    def __str__(self) -> str:
        """Getting the string representation of the algebraic formula of the surface. ax + by + cz + d = 0"""
        accumulator = f"{self.__a}"
        return accumulator + "".join(
            ((f"+{val}{var}" if val > 0 else f"-{val}{var}") for val, var in
             zip((self.__b, self.__c, self.__d), ('x', 'y', 'z', '')) if val != 0))

    def __repr__(self):
        return f'Surface("{self.__str__()}")'

    def to_lambda(self):
        if self.__c == 0:
            warnings.warn("c = 0 might lead to unexpected behaviors in this version.")
            return lambda x, y: 0
        return lambda x, y: (-self.__a * x - self.__b * y - self.__d) / self.__c

    def plot(self, start: float = -3, stop: float = 3,
             step: float = 0.3,
             xlabel: str = "X Values",
             ylabel: str = "Y Values", zlabel: str = "Z Values", show=True, fig=None, ax=None,
             write_labels=True, meshgrid=None):
        plot_function_3d(self.to_lambda(),
                         start=start, stop=stop, step=step, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, show=show,
                         fig=fig,
                         ax=ax, write_labels=write_labels, meshgrid=meshgrid
                         )

    def __eq__(self, other):
        """Equating between surfaces. Surfaces are equal if they have the same a,b,c,d coefficients """
        if other is None:
            return False
        if isinstance(other, Surface):
            return (self.__a, self.__b, self.__c, self.__c) == (other.__a, other.__b, other.__c, other.__d)
        if isinstance(other, list):
            return [self.__a, self.__b, self.__c, self.__d] == other
        if isinstance(other, tuple):
            return (self.__a, self.__b, self.__c, self.__d) == other
        if isinstance(other, set):
            return {self.__a, self.__b, self.__c, self.__d} == other
        raise TypeError(f"Invalid type '{type(other)}' for checking equality with object of instance of class Surface."
                        f"Expected types 'Surface', 'list', 'tuple', 'set'. ")

    def __ne__(self, other):
        return not self.__eq__(other)


class Graph:
    def __init__(self, objs, fig, ax):
        self._items = [obj for obj in objs]
        self._fig, self._ax = fig, ax

    @property
    def items(self):
        return self._items

    def is_empty(self):
        return not self._items

    def add(self, obj):
        self._items.append(obj)

    def plot(self):
        raise NotImplementedError

    def scatter(self):
        raise NotImplementedError


class Graph2D(Graph):

    def __init__(self, objs: Iterable[IPlottable] = tuple()):
        fig, ax = create_grid()
        super(Graph2D, self).__init__(objs, fig, ax)

    def plot(self, start: float = -10, stop: float = 10,
             step: float = 0.01, ymin: float = -10, ymax: float = 10, text=None, show_axis=True, show=True,
             formatText=False, values=None):
        if values is None:
            values = list(decimal_range(start=start, stop=stop, step=step))
        if show_axis:
            draw_axis(self._ax)
        if text is None:
            if len(self._items) >= 3:
                graph_title = ", ".join([obj.__str__() for obj in self._items[:3]]) + "..."
            else:
                graph_title = ", ".join(obj.__str__() for obj in self._items)
        else:
            graph_title = text
        for obj in self._items:
            if isinstance(obj, (IExpression, Function)):
                obj.plot(start=start, stop=stop, step=step, ymin=ymin, ymax=ymax,
                         text=graph_title, show_axis=show_axis, fig=self._fig, ax=self._ax, show=False,
                         formatText=formatText, values=values)
            elif isinstance(obj, Circle):
                obj.plot(fig=self._fig, ax=self._ax)
        if show:
            plt.show()


class Graph3D(Graph):
    def __init__(self, objs=tuple()):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        super(Graph3D, self).__init__(objs, fig, ax)

    def plot(self, functions: Iterable[Union[Callable, str, IExpression]], start: float = -5, stop: float = 5,
             step: float = 0.1,
             xlabel: str = "X Values",
             ylabel: str = "Y Values", zlabel: str = "Z Values"):
        return plot_functions_3d(
            functions=functions, start=start, stop=stop, step=step, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel
        )

    def scatter(self, functions: Iterable[Union[Callable, str, IExpression]], start: float = -5, stop: float = 5,
                step: float = 0.1,
                xlabel: str = "X Values",
                ylabel: str = "Y Values", zlabel: str = "Z Values"):
        return scatter_functions_3d(
            functions=functions, start=start, stop=stop, step=step, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel
        )


class Occurrence:
    def __init__(self, chance: float = 1, identifier: str = ""):
        self._chance = chance
        self._identifier = identifier

    @property
    def chance(self):
        return self._chance

    @chance.setter
    def chance(self, chance: float):
        if 0 <= chance <= 1:
            self._chance = chance
        else:
            warnings.warn(f"Occurrence._chance(setter): failed to set since "
                          f" the probability must be in the range 0 to 1, got {chance} ")

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, identifier: str):
        self._identifier = identifier

    def intersection(self, *occurrences):
        """
        :param occurrences: a collection of occurrences
        :return: returns their intersection of probabilities of type float
        """
        result = self.chance
        for occurrence in occurrences:
            if isinstance(occurrence, float) or isinstance(occurrence, int):
                occurrence = Occurrence(occurrence)
            result *= occurrence.chance
        return result

    def union(self, *occurrences):
        result = self.chance
        for occurrence in occurrences:
            result += occurrence.chance + result - self.intersection(occurrence)
        return result

    def __str__(self):
        return f"probability: {self.chance} , _identifier: {self.identifier}"

    def __repr__(self):
        return f'Occurrence(_chance={self.chance},_identifier={self.identifier})'


class ProbabilityTree:
    """
    A class that is designed in order to represent probability trees and handle with them with greater ease.
    You can also import and export ProbabiltyTree objects in JSON and XML formats.
    """

    @staticmethod
    def tree_from_json(file_path: str) -> Node:
        with open(file_path, 'r') as json_file:
            tree_json = json_file.read()
        tree_structure: dict = json.loads(tree_json)
        tree_values = list(tree_structure.values())
        root_dict = tree_values[0]
        root = Node(name=Occurrence(root_dict["_chance"], identifier=root_dict["_identifier"]), parent=None)
        existing_nodes = [root]
        for node_dict in tree_values[1:]:
            parent_name = node_dict["parent"]
            if parent_name is None:
                parent_node = None
            else:
                parent_node = None
                for existing_node in existing_nodes:
                    if existing_node.name._identifier == parent_name:
                        parent_node = existing_node
                        break
                if parent_node is None:
                    warnings.warn(f"ProbabilityTree.tree_from_json(): couldn't find parent '{parent_name}."
                                  f"{node_dict['_identifier']} will be lost unless it is resolved.'")
            new_node = Node(name=Occurrence(node_dict["_chance"], identifier=node_dict["_identifier"]),
                            parent=parent_node)
            existing_nodes.append(new_node)
        return root

    @staticmethod
    def tree_from_xml(xml_file: str):
        """ import a ProbabilityTree object from an XML file.
         For example:
        <myCustomTree>
            <node>
               <parent>None</parent>
                <identifier>root</identifier>
                <chance>1</chance>
            </node>
            <node>
               <parent>root</parent>
                <identifier>son1</identifier>
                <chance>0.6</chance>
            </node>
            <node>
               <parent>root</parent>
                <identifier>son2</identifier>
                <chance>0.4</chance>
            </node>
        </myCustomTree>
         """
        tree = parse(xml_file)
        root = tree.getroot()  # first tag, contains all
        nodes = []
        for node in root.findall('./node'):
            name = node.find('./identifier').text
            parent_name = node.find('./parent').text
            if parent_name.lower().strip() in ("", "none") or parent_name is None:
                parent_node = None
            else:
                parent_node = None
                for existing_node in nodes:
                    if existing_node.name._identifier == parent_name:
                        parent_node = existing_node
                        break
                if parent_node is None:
                    warnings.warn(f"ProbabilityTree.tree_from_xml(): couldn't find parent '{parent_name}."
                                  f"{name} will be lost unless it is resolved.'")

            probability = node.find('./chance').text
            nodes.append(Node(name=Occurrence(chance=float(probability.strip()), identifier=name), parent=parent_node))
        return nodes[0]

    def to_dict(self):
        root = self.__root
        nodes = [children for children in
                 ZigZagGroupIter(root)]
        new_dict = {}
        for level_nodes in nodes:
            for node in level_nodes:
                new_dict[node.name.identifier] = {"parent": node.parent, "_identifier": node.name.identifier,
                                                  "_chance": node.name.chance}
        return new_dict

    @staticmethod
    def __to_dict_json(tree):
        root = tree.root()
        nodes = [children for children in
                 ZigZagGroupIter(root)]
        new_dict = {}
        for level_nodes in nodes:
            for node in level_nodes:
                new_dict[node.name._identifier] = {"parent": node.parent.name._identifier if node.parent else None,
                                                   "_identifier": node.name._identifier,
                                                   "_chance": node.name._chance}
        return new_dict

    @staticmethod
    def __tree_to_json(path: str, tree):
        dictionary = ProbabilityTree.__to_dict_json(tree)
        with open(path, 'w') as fp:
            json.dump(dictionary, fp, indent=4)

    def export_json(self, path: str):
        return ProbabilityTree.__tree_to_json(path=path, tree=self)

    def to_xml_str(self, root_name: str = "MyTree"):
        xml_accumulator = f"<{root_name}>\n"
        for children in ZigZagGroupIter(self.__root):
            for node in children:
                xml_accumulator += f"\t<node>\n"
                if node.parent is not None:
                    xml_accumulator += f"\t\t<parent>{node.parent.name.identifier}</parent>\n"
                else:
                    xml_accumulator += f"\t\t<parent>None</parent>\n"
                xml_accumulator += f"\t\t<identifier>{node.name.identifier}</identifier>\n"
                xml_accumulator += f"\t\t<chance>{node.name.chance}</chance>\n"
                xml_accumulator += f"\t</node>\n"
        xml_accumulator += f"</{root_name}>"
        return xml_accumulator

    def export_xml(self, file_path: str = "", root_name: str = "MyTree"):
        with open(f"{file_path}", "w") as f:
            f.write(self.to_xml_str(root_name=root_name))

    def __init__(self, root=None, json_path=None, xml_path=None):
        """
        Creating a new probability tree
        :param root: a string that describes the root occurrence of the tree (Optional)
        :param json_path: in order to import the tree from json file (Optional)
        :param xml_path: in order to import the tree from xml file (Optional)
        """
        if json_path:
            self.__root = ProbabilityTree.tree_from_json(json_path)
        elif xml_path:
            self.__root = ProbabilityTree.tree_from_xml(xml_path)
        else:
            if root is None:
                self.__root = Node(Occurrence(1, "root"))
            elif isinstance(root, Occurrence):
                self.__root = Node(root)
            else:
                raise TypeError

    @property
    def root(self):
        return self.__root

    def add(self, probability: float, identifier: str, parent=None):
        """
        creates a new node, adds it to the tree, and returns it
        :param probability: The probability of occurrence
        :param identifier: Unique string that represents the new node
        :param parent: the parent of node that will be created
        :return: returns the node that was created
        """
        occurrence = Occurrence(probability, identifier)
        if parent is None:
            parent = self.__root
        node = Node(name=occurrence, parent=parent, edge=2)
        level_sum = ProbabilityTree.get_level_sum(node)
        if level_sum > 1:
            depth = ProbabilityTree.get_depth(node)
            warnings.warn(f"ProbabilityTree.add_occurrence(): Probability sum in depth {depth} (sum is {level_sum})"
                          f") is bigger than 1, expected 1 or less.")
        return node

    @staticmethod
    def get_depth(node: Node):
        return node.depth

    @staticmethod
    def get_level_sum(node: Node):
        if not node.siblings:
            return node.name.chance
        return node.name.chance + reduce(lambda a, b: a.name.chance + b.name.chance,
                                         node.siblings).name.chance

    def num_of_nodes(self):
        return sum(len(children) for children in ZigZagGroupIter(self.__root))

    def get_probability(self, path=None) -> float:
        """

        :param path: gets a specific path of the names of the nodes, such as ["root","son","grandson"]
        or the node itself
        :type path: list or node
        :return: returns the probability up to that path or node
        :rtype: float
        """
        probability = 1

        if isinstance(path, (list, tuple, str)):
            if isinstance(path, str):
                path = path.split('/')
            if path is None:
                path = self.__root.name.identifier
            nodes = [[node.name for node in children] for children in
                     ZigZagGroupIter(self.__root, filter_=lambda n: n.name.identifier in path)]
            probability: float = 1
            for node in nodes:
                for occurrence in node:
                    probability *= occurrence.chance
        elif isinstance(path, Node):
            for node in path.iter_path_reverse():
                probability *= node.name.chance
        return round(probability, 5)

    @staticmethod
    def biggest_probability_node(node: Node) -> Node:
        # TODO: check whether the new implementation works
        return max(ZigZagGroupIter(node), key=operator.attrgetter(node.name.chance))

    def get_node_path(self, node: Union[str, Node]):
        accumulator = ""
        if isinstance(node, str):
            node = self.get_node_by_id(node)
        if len(node.ancestors) <= 1:
            return node.name.identifier
        for ancestor in node.ancestors[1:]:
            accumulator += f"/{ancestor.name.identifier}"
        accumulator += f"/{node.name.identifier}"
        return accumulator[1:]

    def get_node_by_id(self, identifier: str):
        for children in ZigZagGroupIter(self.__root):
            for node in children:
                if node.name.identifier == identifier:
                    return node
        return None

    def remove(self, *nodes):
        raise NotImplementedError

    def __str__(self):
        accumulator = ""
        for pre, fill, node in RenderTree(self.__root):
            accumulator += ("%s%s:%s\n" % (pre, node.name.identifier, node.name.chance))
        return accumulator

    def __contains__(self, node):
        """

        :param node: a node or the _identifier of the occurrence object inside the name attribute of the node
        :return: returns True if found a match, else False
        """
        if isinstance(node, Node):
            for current_node in PreOrderIter(self.__root):
                if current_node is Node or current_node == node:
                    return True
                # TODO: implement equality with __eq__ so two duplicates will be considered equal as well
                return False
        elif isinstance(node, str):
            for current_node in PreOrderIter(self.__root):
                if current_node.name.identifier == node:
                    return True
            return False
        else:
            raise TypeError(f"ProbabilityTree.__contains__(): expected type 'str' or 'Node', got {type(node)}")

    def __eq__(self, other):
        pass

    def __ne__(self, other):
        return not self.__eq__(other)


def column(matrix, index: int):
    """
    Fetches a column in a matrix

    :param matrix: the matrix from which we fetch the column
    :param index: the index of the column. From 0 to the number of num_of_columns minus 1.
    :return: Returns a list of numbers, that represents the column in the given index
    :raise: Raises index error if the index isn't valid.
    """
    return [row[index] for row in matrix]


class Matrix:

    def __init__(self, matrix: Union[list, str, tuple] = None, dimensions=None, copy_elements=False):
        if (matrix, dimensions) == (None, None):
            raise ValueError("Cannot create an empty Matrix")

        if matrix is not None:
            if isinstance(matrix, str):
                dimensions = matrix
            else:
                if not isinstance(matrix, list):
                    matrix = list(matrix)
                if isinstance(matrix[0], Iterable):  # Find a better type hint
                    self._num_of_rows, self._num_of_columns = len(matrix), len(matrix[0])

                self._matrix = [[copy_expression(item) for item in row] for row in matrix]
        if dimensions is not None:
            if isinstance(dimensions, str):
                if dimensions.count('x') == 1:
                    self._num_of_rows, self._num_of_columns = [int(i) for i in dimensions.strip().split('x')]
                elif dimensions.count(',') == 1:
                    self._num_of_rows, self._num_of_columns = [int(i) for i in dimensions.strip().split(',')]
            elif isinstance(dimensions, (tuple, list, set)):
                self._num_of_rows, self._num_of_columns = int(dimensions[0]), int(dimensions[1])
            else:
                raise TypeError(f"Invalid type {type(dimensions)} for the dimensions of the matrix ")
            self.matrix: List[list] = [[0 for col in range(self.num_of_columns)] for row in range(self.num_of_rows)]

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, mat):
        self._matrix = mat

    @property
    def num_of_rows(self):
        return self._num_of_rows

    @property
    def num_of_columns(self):
        return self._num_of_columns

    @property
    def shape(self):
        return self._num_of_rows, self._num_of_columns

    def add_and_mul(self, line1: int, line2: int, scalar):
        """
        adds a line to another line which is multiplied by a value.

        :param line1: The line that will receive the multiplication result
        :param line2: The line that its multiplication with the scalar value will be added to the other line.
        :param scalar:
        :return: None
        """
        if line1 < 0 or line1 >= self._num_of_rows:
            raise IndexError(f"Invalid line index {line1}. Expected indices between 0 and {self._num_of_rows}")
        if line2 < 0 or line2 >= self._num_of_rows:
            raise IndexError(f"Invalid line index {line2}. Expected indices between 0 and {self._num_of_rows}")
        for i in range(self.num_of_columns):
            self.matrix[line1][i] += self.matrix[line2][i] * scalar

    def replace_rows(self, line1: int, line2: int):
        """
        Replace the values between two num_of_rows in the matrix.

        :param line1: The index of the first row.
        :param line2: The index of the second row.
        :return:
        """
        if line1 < 0 or line1 >= self._num_of_rows:
            raise IndexError(f"Invalid line index {line1}. Expected indices between 0 and {self._num_of_rows - 1}")
        if line2 < 0 or line2 >= self._num_of_rows:
            raise IndexError(f"Invalid line index {line2}. Expected indices between 0 and {self._num_of_rows - 1}")
        for i in range(self.num_of_columns):
            self.matrix[line1][i], self.matrix[line2][i] = self.matrix[line2][i], self.matrix[line1][i]

    def __get_starting_item(self, i: int):
        """ Get the index of the first element in a row that is not 0"""
        for j in range(i, self._num_of_columns):
            if self.matrix[j][i] != 0:
                return j
        return -1

    def multiply_row(self, expression, row: int):
        if not 0 < row <= self._num_of_rows:
            raise IndexError(f"Invalid line index {row}. Expected indices between 0 and {self._num_of_rows}")

        for i in range(self.num_of_columns):
            self.matrix[row][i] *= expression

    def divide_row(self, scalar, row: int):
        """
        Dividing a row in the matrix by a number. The row numbers starts from 1, instead of 0, as it is common
        in indices.
        :param scalar: type float
        :param row: the number of the row, starting from 1, type int.
        :return: Doesn't return anything ( None )
        """
        if row >= self.num_of_rows:
            raise IndexError(f"Row Indices must be bigger than 0 and smaller than length({self.num_of_rows})")
        if scalar == 0:
            raise ZeroDivisionError("Matrix.divide_row(): Can't divide by zero !")
        for i in range(self.num_of_columns):
            self.matrix[row][i] /= scalar

    def divide_all(self, expression):
        if expression == 0:
            raise ValueError(f"Cannot divide a matrix by 0.")
        for row in self._matrix:
            for index in range(len(row)):
                row[index] /= expression

    def multiply_all(self, expression):
        """
        multiplying each number in the matrix by a number of type float
        :param expression: type float, can't be 0.
        :return: Doesn't return anything (None)
        """
        for row in self._matrix:
            for index in range(len(row)):
                row[index] *= expression

    def kronecker(self, other: "Matrix"):
        new_matrix = Matrix(
            dimensions=(self.num_of_rows * other._num_of_rows, self._num_of_columns * other._num_of_columns))
        row_offset, col_offset = 0, 0
        for row in self._matrix:
            for item in row:
                for row_index in range(other._num_of_rows):
                    for col_index in range(other._num_of_columns):
                        new_matrix._matrix[row_offset + row_index][col_offset + col_index] = item * \
                                                                                             other._matrix[row_index][
                                                                                                 col_index]
                col_offset += other._num_of_columns
            col_offset = 0
            row_offset += other._num_of_rows
        return new_matrix

    def add_to_all(self, expression):
        for row in self._matrix:
            for index in range(len(row)):
                row[index] += expression

    def subtract_from_all(self, expression):
        for row in self._matrix:
            for index in range(len(row)):
                row[index] += expression

    def apply_to_all(self, f: Callable):
        for row_index, row in enumerate(self._matrix):
            if isinstance(row, list):
                for index, item in enumerate(row):
                    row[index] = f(item)
                else:
                    self._matrix[row_index] = f(row)

    def gauss(self) -> None: # TODO: fix tasikian bug.
        """
        Ranking a matrix is the most important part in this implementation of gaussian elimination .
        The gaussian elimination is a method for solving a set of linear equations. It is supported in this program
        via the LinearSystem class, but it uses the Matrix class for the solving process.
        """
        number_of_zeroes = 0
        for i in range(self._num_of_rows):
            if i < self.num_of_columns and self.matrix[i][i] == 0:
                index = self.__get_starting_item(i)
                if index != -1:
                    self.replace_rows(i, index)
                else:
                    del self._matrix[i]
                    self._matrix.append([0] * self._num_of_columns)
                    number_of_zeroes += 1
            if self.matrix[i][i] != 0:
                self.divide_row(self.matrix[i][i], i)
            for j in range(self.num_of_rows):
                if i != j:
                    self.add_and_mul(j, i, -self.matrix[j][i])

    def __test_gauss(self) -> None:
        """
        Ranking a matrix is the most important part in this implementation of gaussian elimination .
        The gaussian elimination is a method for solving a set of linear __equations. It is supported in this program
        via the LinearSystem class, but it uses the Matrix class for the solving process.
        """
        number_of_zeroes = 0
        i = 0
        for k in range(self._num_of_columns):
            if i < self.num_of_rows and self.matrix[i][k] == 0:
                index = -1
                for t in range(i, self.num_of_rows):
                    if self._matrix[i][t] != 0:
                        index = t
                if index != -1:
                    self.replace_rows(i, index)
                    i += 1
            if self.matrix[i][k] != 0:
                self.divide_row(self.matrix[i][k], i)
            for j in range(self.num_of_rows):
                if i != j:
                    self.add_and_mul(j, i, -self.matrix[j][i])

    def get_rank(self, copy=True) -> int:
        my_matrix = self.__copy__() if copy else self
        my_matrix.gauss()
        min_span: int = len(my_matrix)
        # Now, check if there are rows with only zeroes.
        num_of_zeroes_lines = 0
        for row in my_matrix:
            if all(item == 0 for item in row):
                num_of_zeroes_lines += 1
        return num_of_zeroes_lines - num_of_zeroes_lines

    def __zero_line(self, row: Iterable) -> int:
        return 1 if all(element == 0 for element in row) else 0

    def determinant(self, rank=False) -> float:
        """
        Finds the determinant of the function, as a byproduct of ranking a copy of it.

        :param rank: If set to True, the original matrix will be ranked in the process. Default is False.
        """
        if self._num_of_rows != self._num_of_columns:
            raise ValueError("Cannot find a determinant of a non-square matrix")
        if self._num_of_rows == 2:  # Decrease time complexity to O(1) in simple cases
            return self._matrix[0][0] * self._matrix[1][1] - self._matrix[1][0] * self._matrix[0][1]
        d: float = 1
        if not rank:
            other = self.__copy__()
        else:
            other = self
        for i in range(other._num_of_rows):
            if i < other._num_of_rows and other.matrix[i][i] == 0:
                # other.replace_rows(i, other.__get_starting_item(i))
                d = -d
            if other.matrix[i][i] != 0:
                d *= other._matrix[i][i]
                other.divide_row(other.matrix[i][i], i)
            for j in range(other.num_of_rows):
                if i != j:
                    other.add_and_mul(j, i, -other.matrix[j][i])
        if any(other.__zero_line(row) for row in other._matrix):
            d = 0
        return d

    def yield_items(self):
        for row in self._matrix:
            for item in row:
                yield item

    def transpose(self):
        """ Computing the transpose of a matrix. M X N -> N X M """
        new_matrix = []
        for col in self.columns():
            new_matrix.append([item.__copy__() if hasattr(item, "__copy__") else item for item in col])
        return Matrix(new_matrix)

    def sum(self):
        """
        The sum of all of the items in the matrix.
        :return: the sum of the items ( float )
        :rtype: should be float
        """
        return sum((sum(lst) for lst in self.matrix))

    def max(self):
        """
        gets the biggest item in the matrix
        :return: biggest item in the matrix ( float)
        :rtype: should be float
        """
        if self.num_of_rows > 1:
            return max((max(row) for row in self.matrix))
        return max(self._matrix)

    def min(self):
        """
        returns the smallest value in the matrix
        :return: the smallest value in the matrix
        :rtype: should be float
        """
        if self.num_of_rows > 1:
            return min(min(row) for row in self.matrix)
        return min(self._matrix)

    def average(self):
        return self.sum() / (self._num_of_rows * self._num_of_columns)

    def average_in_line(self, row_index: int):
        return sum(self._matrix[row_index]) / self._num_of_columns

    def average_in_column(self, column_index: int):
        return sum((row[column_index] for row in self._matrix))

    def __iadd__(self, other: "Union[IExpression, int, float, Matrix, np.array]"):
        if isinstance(other, (int, float, IExpression)):
            self.add_to_all(other)
            return self
        elif isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError(f"Cannot Add matrices with different shapes: {self.shape} and {other.shape}")
            self.add(other)
            return self
        else:
            raise TypeError(f"Invalid type '{type(other)}' for adding matrices")

    def __add__(self, other: "Union[IExpression, int, float, Matrix, np.array]"):
        return self.__copy__().__iadd__(other)

    def __isub__(self, other: "Union[IExpression, int, float, Matrix, np.array]"):
        if isinstance(other, (int, float, IExpression)):
            self.subtract_from_all(other)
            return self
        elif isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError(f"Cannot Add matrices with different shapes: {self.shape} and {other.shape}")
            self.subtract(other)
            return self
        else:
            raise TypeError(f"Invalid type '{type(other)}' for subtracting matrcices")

    def __sub__(self, other: "Union[IExpression, int, float, Matrix,np.array"):
        return self.__copy__().__isub__(other)

    def __imatmul__(self, other: "Union[list, Matrix]"):
        return self.matmul(other)

    def __matmul__(self, other):
        return self.__copy__().matmul(other)

    def __imul__(self, other: "Union[IExpression, int, float, Matrix, np.array]"):
        if isinstance(other, (IExpression, int, float)):
            self.multiply_all(other)
            return self
        elif isinstance(other, (Matrix, list)):
            self.multiply_element_wise(other)
            return self
        else:
            raise TypeError(f"Invalid type '{type(other)} for multiplying matrices'")

    def __mul__(self, other: "Union[IExpression, int, float, Matrix, np.array]"):
        return self.__copy__().__imul__(other)

    def __itruediv__(self, other: "Union[IExpression, int, float, Matrix,np.array]"):
        if other == 0:
            raise ValueError("Cannot divide a matrix by 0")
        if isinstance(other, (int, float, IExpression)):
            self.divide_all(other)
            return self
        elif isinstance(other, (Matrix, Iterable)):
            pass
        else:
            raise TypeError(f"Invalid type '{type(other)} for dividing matrices'")

    def __truediv__(self, other):
        return self.__copy__().__itruediv__(other)

    def __eq__(self, other) -> bool:
        """
        checks if two matrices are equal by overloading the '==' operator.

        :param other: other matrix
        :type other: Matrix / list / tuple,set
        :return: Returns True if the matrices are equal, otherwise it returns False.
        """
        if isinstance(other, (list, tuple, set)):
            other = Matrix(other)
        if isinstance(other, Matrix):
            if len(self.matrix) != len(other.matrix):
                return False
        else:
            raise TypeError(
                f"Unexpected type {type(other)} in Matrix.__eq__(). Expected types list,tuple,set or Matrix ")
        for i in range(len(self.matrix)):
            if len(self._matrix[i]) != len(other._matrix[i]):
                return False
            for j in range(len(self._matrix[i])):
                if self._matrix[i][j] != other._matrix[i][j]:
                    return False
        return True

    def __ne__(self, other) -> bool:
        """Returns True if the matrices aren't equal, and False if they're equal. Overloads the built in != operator."""
        return not self.__eq__(other)

    def __str__(self) -> str:
        """
        A visual representation of the matrix

        :return: a visual representation of the matrix, of str type.

        :rtype: str
        """
        max_length = max([2 + sum([len(str(number)) + 1 for number in row]) for row in self.matrix])
        accumulator = ""
        for row in self.matrix:
            line_aggregator = "| "
            for element in row:
                if isinstance(element, int):
                    element = float(element)
                line_aggregator += f'{element} '
            line_aggregator += ' ' * (max_length - len(line_aggregator)) + "|\n"
            accumulator += line_aggregator
        return accumulator

    def __repr__(self):
        return f"Matrix(matrix={self.matrix})"

    @staticmethod
    def random_matrix(shape: Tuple[int, int] = None, values: Tuple[Union[int, float], Union[int, float]] = (-15, 15),
                      dtype='int'):
        if shape is None:
            shape = (random.randint(1, 5), random.randint(1, 5))
        new_matrix = Matrix(dimensions=shape)
        if dtype == 'int':
            random_method = random.randint
        elif dtype == 'float':
            random_method = random.uniform
        else:
            raise ValueError(f"invalid dtype '{dtype}', currently allowed types are 'int' and 'float'")
        for row in new_matrix:
            for index in range(len(row)):
                if dtype == 'int':
                    row[index] = random_method(values[0], values[1])
                elif dtype == 'float':
                    row[index] = random_method(values[0], values[1])
        return new_matrix

    # TODO: check if it works and the tuple bug doesn't occur
    def add(self, *matrices) -> "Matrix":
        """
        returns the result of the addition of the current matrix and other matrices.
        Flexible with errors: if users enter a list or tuples of matrices, it accepts them too rather than
        returning a type error.

        :param: matrices: the matrices to be added. each matrix should be of type Matrix.
        :return: the result of the addition.
        :rtype: Matrix
        :raise: Raises a type error in case a matrix is not of type Matrix,list,or tuple.
        :raise: Raises an index error if the matrices aren't compatible for addition, i.e, they have different
        dimensions.
        """
        try:
            for matrix in matrices:
                if isinstance(matrix, list) or isinstance(matrix, tuple):
                    matrix = Matrix(matrix)
                if isinstance(matrix, Matrix):
                    if self.num_of_rows != matrix.num_of_rows or self.num_of_columns != matrix.num_of_columns:
                        raise IndexError
                    for row1, row2 in zip(self.matrix, matrix.matrix):
                        for i in range(min((len(row1), len(row2)))):
                            row1[i] += row2[i]
                else:
                    raise TypeError(f"Cannot add invalid type {type(matrix)}, expected types Matrix, list, or tuple.")
            return self.__copy__()

        except IndexError:
            warnings.warn(f"Matrix.add(): Tried to add two matrices with different number of num_of_rows ( fix it !)")
        except TypeError:
            warnings.warn(f"Matrix.add(): Expected types Matrix,list,tuple")

    def filtered_matrix(self, predicate: Callable[[Any], bool] = None, copy=True,
                        get_list=False) -> "Union[List, Matrix]":
        """ returns a new matrix object that its values were filtered by the
        , without changing the original matrix"""
        if copy:
            new_matrix = [[copy_expression(expression) for expression in row if predicate(expression)] for row in
                          self._matrix]
        else:
            new_matrix = [[expression for expression in row if predicate(expression)] for row in
                          self._matrix]
        if get_list:
            return new_matrix
        return Matrix(matrix=new_matrix)

    def mapped_matrix(self, func: Callable) -> "Matrix":
        copy = self.matrix.copy()
        for index, row in enumerate(copy):
            copy[index] = [func(item) for item in row]
        return Matrix(copy)

    # TODO: modify this ?
    def foreach_item(self, func: Callable) -> "Matrix":
        """
        Apply a certain function to all of the elements of the matrix.

        :param func: the given callable function
        :return: Returns the current object
        """
        for current_row in range(self._num_of_rows):
            for current_column in range(self._num_of_columns):
                self._matrix[current_row][current_column] = func(self._matrix[current_row][current_column])
        return self

    # TODO: check if it works and the tuple bug doesn't occur
    def subtract(self, *matrices) -> "Matrix":
        """Similar to the add() method, it returns the result of the subtractions of the current matrix
         with the given matrices. Namely, let 'a' be the current matrix, and 'b', 'c', 'd' the given matrices,
         a-b-c-d will be returned.

         :rtype: Matrix
         """
        try:
            for matrix in matrices:
                if isinstance(matrix, list) or isinstance(matrix, tuple):
                    matrix = Matrix(matrix)
                if isinstance(matrix, Matrix):
                    if self.num_of_rows != matrix.num_of_rows or self.num_of_columns != matrix.num_of_columns:
                        raise IndexError
                    for row1, row2 in zip(self.matrix, matrix.matrix):
                        for i in range(min((len(row1), len(row2)))):
                            row1[i] -= row2[i]
                else:
                    raise TypeError
            return self.__copy__()

        except IndexError:
            warnings.warn(
                f"Matrix.subtract(): Tried to add two matrices with different number of num_of_rows ( fix it !)")
        except TypeError:
            warnings.warn(f"Matrix.subtract(): Expected types Matrix,list,tuple, but got {type(matrix)}")

    def columns(self):
        for column_index in range(self.num_of_columns):
            yield [self._matrix[index][column_index] for index in range(self.num_of_rows)]

    def multiply_element_wise(self, other: "Union[Matrix, List[list], list]"):
        if self.shape != other.shape:
            warnings.warn("If you want to execute matrix multiplication, use the '@' binary operator, or the "
                          "__imatmul__(), __matmul__() methods")
            raise ValueError("Can't perform element-wise multiplication of matrices with different shapes. ")
        for i in range(self.num_of_rows):
            for j in range(self.num_of_columns):
                self._matrix[i][j] *= other._matrix[i][j]
        return self

    def matmul(self, other: "Union[Matrix, List[list], list]"):
        """Matrix multiplication. Can also be done via the '@' operator. """
        if isinstance(other, Iterable) and not isinstance(other, Matrix):
            other = Matrix(other)
        print(self)
        print(other)
        if self.shape[1] != other.shape[0] and self.shape[0] != other.shape[1]:
            raise ValueError(f"The matrices aren't suitable for multiplications: "
                             f"Shapes {self.shape} and {other.shape} ")
        result = []
        columns = list(other.columns())
        for row in self._matrix:
            new_row = []
            for col in columns:
                new_row.append(sum(row_element * col_element for
                                   row_element, col_element in zip(row, col)))
            result.append(new_row)

        return Matrix(result)

    def filter_by_indices(self, predicate: Callable[[int, int], bool]):
        """get a filtered matrix based on the indices duos, starting from (0,0)"""
        return [
            [copy_expression(item) for column_index, item in row if predicate(row_index, column_index)] for
            row_index, row in self._matrix]

    def __getitem__(self, item: Union[Callable[[Any], bool], int, Iterable[int]]):
        if isinstance(item, int):
            return self._matrix[item]
        elif isinstance(item, Callable):  # A predicate
            return self.filtered_matrix(predicate=item, copy=False, get_list=True)
        elif isinstance(item, Iterable):
            return [self._matrix[index] for index in item]
        else:
            raise TypeError(f"Invalid type '{type(bool)}' when accessing items of a matrix with the [] operator")

    def __setitem__(self, key, value):
        return self._matrix.__setitem__(key, value)

    def __delitem__(self, key):
        return self._matrix.__delitem__(key)

    def column(self, index: int):
        return column(self._matrix, index)

    def reversed_columns(self) -> "Matrix":
        return Matrix(matrix=list(reversed([column(self._matrix, i) for i in range(self.num_of_rows)])))

    def reversed_rows(self) -> "Matrix":
        """
        Returns a copy of the matrix object that its lines are in a reversed order.

        :return: Returns a Matrix object that its matrix's lines are reversed compared to the original object.
        """
        return Matrix(matrix=list(reversed(self.matrix)))

    def iterate_by_columns(self) -> Iterator[Optional[Any]]:
        """Yields the elements in the order of the columns"""
        for j in range(self._num_of_columns):
            for i in range(self._num_of_rows):
                yield self._matrix[i][j]

    def range(self) -> Iterator[Tuple[int, int]]:
        """
        yields the indices of the matrix
        For example, for a matrix of dimensions 2x2, the method will yield (0,0), then (0,1), then (1,0), then (1,1)

        :return: yields a generator of the indices in the matrix.
        """
        for i in range(self._num_of_rows):
            for j in range(self._num_of_columns):
                yield i, j

    def __reversed__(self):
        # TODO: UP -> DOWN OR LEFT -> RIGHT ????
        pass

    def inverseWithNumpy(self, verbose=False):
        """ Returns the inverse of the matrix"""
        try:
            return Matrix(matrix=[list(row) for row in inv(self._matrix)])
        except LinAlgError:
            # Not invertible
            if verbose:
                warnings.warn("The matrix has no inverse")
            return None

    @staticmethod
    def unit_matrix(n: int) -> "Matrix":
        zeroes_matrix = Matrix(dimensions=(n, n))
        for i in range(n):
            zeroes_matrix._matrix[i][i] = 1
        return zeroes_matrix

    @staticmethod
    def is_unit_matrix(given_matrix: "Matrix") -> bool:
        """Checking whether the matrix is a unit matrix"""
        if given_matrix._num_of_rows != given_matrix._num_of_columns:
            return False

        for row_index, row in enumerate(given_matrix):
            for col_index, item in enumerate(row):
                if row_index == col_index:
                    if item != 1:
                        return False
                elif item != 0:
                    return False
        return True

    def inverse(self):
        """Finding the inverse of the matrix"""
        if self._num_of_rows != self._num_of_columns:
            return self.inverseWithNumpy()
        n: int = self._num_of_rows
        unit_matrix = Matrix.unit_matrix(n)
        number_of_zeroes = 0
        my_matrix = self.__copy__()
        for i in range(my_matrix._num_of_rows):
            if i < my_matrix.num_of_columns and my_matrix.matrix[i][i] == 0:
                index = my_matrix.__get_starting_item(i)
                if index != -1:
                    my_matrix.replace_rows(i, index)
                    unit_matrix.replace_rows(i, index)
                else:
                    return None
            if my_matrix.matrix[i][i] != 0:
                unit_matrix.divide_row(my_matrix.matrix[i][i], i)
                my_matrix.divide_row(my_matrix.matrix[i][i], i)
            for j in range(my_matrix.num_of_rows):
                if i != j:
                    unit_matrix.add_and_mul(j, i, -my_matrix.matrix[j][i])
                    my_matrix.add_and_mul(j, i, -my_matrix.matrix[j][i])
        if not Matrix.is_unit_matrix(my_matrix):
            return None
        return unit_matrix

    def __len__(self) -> Tuple[int, int]:
        """ Returns the lengths in this format: (num_of_rows,num_of_columns)"""
        # TODO: check if should return in the opposite order
        return self._num_of_rows, self._num_of_columns

    def __copy__(self) -> "Matrix":  # TODO: handle 1 dimensional matrices
        if not isinstance(self._matrix[0], list):
            return Matrix([copy_expression(item) for item in self._matrix])
        new_matrix = []
        for row in self._matrix:
            new_row = []
            for item in row:
                if hasattr(item, '__copy__'):
                    new_row.append(item.__copy__())
                elif hasattr(item, 'copy'):
                    new_row.append(item.copy())
                else:
                    new_row.append(item)
            new_matrix.append(new_row)
        return Matrix(new_matrix)


def main():
    """ main  method """
    x = Var('x')
    my_expression = ExpressionMul([Sin(2*x), Cos(x), x])
    print(my_expression.derivative())
    return
    x = Var('x')
    my_expression = ExpressionMul([3*x, x, x])
    print(my_expression.derivative())
    return
    tasikian_matrix = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    tasikian_matrix.__test_gauss()
    print(tasikian_matrix)
    return
    tasikian_matrix = Matrix([[1, 0, 0], [0, 0, 1], [0, 0, 0]])
    tasikian_matrix.__test_gauss()
    print(tasikian_matrix)
   

    return
    example_matrix = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    print(example_matrix.inverse())

    return
    example_matrix = Matrix([[1, 0], [0, 1]])
    print(example_matrix.inverse())
    return
    example_matrix = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(example_matrix.inverse())
    return
    print(Matrix.unit_matrix(3))
    return
    my_matrix = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(my_matrix)
    print(my_matrix.determinant())
    return
    my_surface = Surface([7, 3, 1, 9])
    my_surface.plot()
    return
    x = Var('x')
    print(Vector((1, 2, 4)).equal_direction_ratio(Vector((2, 4, 8))))
    print(Vector((x, 2 * x, 4 * x)).equal_direction_ratio(Vector((2 * x, 4 * x, 8 * x))))
    print(Vector((x, 2 * x, 4 * x)).equal_direction_ratio(Vector((2 * x ** 2, 4 * x ** 2, 8 * x ** 2))))

    return
    my_graph = Graph2D()
    my_graph.add(Function("f(x) = x^3-3"))
    my_graph.add(Poly("2x^4 - 3x + 5"))
    my_graph.add(Circle(5, (0, 0)))
    my_graph.plot()
    return
    print(durand_kerner2([1, 0, 0, 0, -16]))
    return
    func = lambda x: x ** 4 - 16
    coefficients = [1, 0, 0, 0, -16]
    print(durand_kerner(func, coefficients))
    return
    print(reinman(lambda x: sin(x), 0, pi, 11))
    print(trapz(lambda x: sin(x), 0, pi, 11))
    print(simpson(lambda x: sin(x), 0, pi, 11))
    return
    print(TrigoExprs("3sin(x) - 2cos(x)"))
    return
    fn2 = Function("f(x) = x^2")
    fn2.scatter2d(basic=False)
    return
    plot_complex(complex(5, 4), complex(3, -2))
    return
    plot_functions_3d(["f(x,y) = sin(x)*cos(y)", "f(x,y) = sin(x)*ln(y)"])
    return
    plot_multiple(
        ["f(x) = 4x", "f(x) = x^2", "f(x) = x^3", "f(x)= 8", "f(x)=ln(x)", "f(x)=e^x", "f(x)=|x|", "f(x)=sin(x)",
         "f(x)=cos(x)"])
    return
    print(QuadraticEquation.random(digits_after=1, variable='y'))
    return
    # scatter in 1D
    points = PointCollection([[1], [3], [5], [6]])
    points.scatter()

    # scatter in 2D
    points = Point2DCollection([[1, 2], [6, -4], [-3, 1], [4, 2], [7, -5], [4, -3], [-2, 1], [-3, 4], [5, 2], [1, -5]])
    points.scatter()

    # scatter in 3D
    points = Point3DCollection(
        [[random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)] for _ in range(100)])
    points.scatter()

    # Scatter in 4D
    points = Point4DCollection(
        [[random.randint(1, 100), random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)] for _ in
         range(100)])
    points.scatter()
    return
    points = [Point2D(3, 5), Point2D(-7, 4), Point2D(1, 2), Point2D(-8, 5)]
    print(list(combinations(points, 2)))
    return
    my_points = PointCollection([Point((2, 6)), [2, 4], Point2D(9, 3)])
    print(my_points)
    return
    x = Var('x')
    print(Exponent(2, x) + 2 * Exponent(2, x))
    print(Exponent(2, x) + 0)
    print(Exponent(3, x) - Exponent(3, x))
    print(Exponent(3, x, coefficient=4) - Exponent(3, x))
    print(Exponent(3, x) * 2)
    print(Exponent(3, x) * Exponent(3, x))
    print(Exponent(3, x) * 3)
    return
    print(solve_quadratic_from_str("x^2 + 6x + 8 = 0"))  # TODO: fix ASAP
    return
    x, y = Var('x'), Var('y')
    print((Cos(x) + Sin(y)).when(x=4, y=5))
    print(Sin(x) ** 2 + 2 * Sin(x) * Cos(x) + Cos(x) ** 2)
    print((Sin(x) + Cos(x)) * (Cos(x) + Sin(x)))
    return
    x = Var('x')
    print(Root(x) - Root(x) == 0)
    print(3 * Root(x) - 2 * Root(x))
    # print(Root(x) + Root(x) == 2*Root(x))
    return
    plot_function_3d("f(x,y) = sin(x)*cos(y)")
    return
    my_equation = LinearEquation("4 + 3x + 2y + 5x + 6 = 2 + x - y", variables=('x', 'y'))
    my_equation.simplify()
    print(my_equation)
    return
    my_sequence = GeometricSeq([2], 2)
    my_sequence.plot(start=1, stop=10)
    return
    pdf_worksheet = PDFWorksheet("Functions")
    pdf_worksheet.add_exercise(PDFCubicFunction(lang='es'))
    pdf_worksheet.add_exercise(PDFQuadraticFunction())
    pdf_worksheet.add_exercise(PDFLinearFromPointAndSlope())
    pdf_worksheet.end_page()
    pdf_worksheet.next_page("Equations")
    for _ in range(10):
        pdf_worksheet.add_exercise(PDFLinearEquation())
        pdf_worksheet.add_exercise(PDFQuadraticEquation())
    pdf_worksheet.end_page()
    pdf_worksheet.create("worksheetExample.pdf")
    return
    x = Var('x')
    print(Sqrt(x) ** 2)
    print(Sqrt(5) ** 2)
    return
    x = Var('x')
    print(Sin(x).newton(initial_value=2))
    return
    # TODO: simplify logarithm division!
    x = Var('x')
    my_log = 3 * Log(x) ** 2
    other_log = 2 * Log(x)
    print(my_log / other_log)
    return
    first_mono, second_mono = Mono("3x^2"), Mono("2x^2")
    print(first_mono + second_mono)
    print(first_mono - second_mono)
    print(first_mono * second_mono)
    print(first_mono / second_mono)
    return
    x = Var('x')
    my_expressions = Sin(x) + Ln(x) + 4
    my_expressions.assign(x=5)
    print(my_expressions.try_evaluate())
    return
    return
    x = Var('x')
    print(Sqrt(2 * x) + Sqrt(2 * x))
    print(Sqrt(4) + Sqrt(6))
    return
    my_surface = Surface([7, 3, 1, 9])
    my_surface.plot()

    my_surface = Surface("7x + 3y + z + 9 = 0")
    my_surface.plot()

    return
    x = Var('x')
    print((Sin(x) + 1) / (Sin(x) + 1))
    return
    x = Var('x')
    first = 3 * Sin(x) ** 2 + 3 * Cos(x) * Sin(x)
    second = Sin(x)
    print(first / second)
    print(first / 5)
    print(first / Cos(x))
    return
    plot_function("f(x) = x^2")
    return
    scatter_function("f(x) = sin(x)")
    return
    scatter_dots_3d([4, 2, 8, -5, 3, 8, 9, -6, 1], [1, 9, -2, 3, 4, 5, 1, 6, 7], [1, -1, 3, 4, 8, 1, 4, 2, 6])
    return
    scatter_dots([1, 2, 3, 4], [2, 4, 6, 8], title="Matplotlib is awesome")
    return
    my_equation = QuadraticEquation("x^2 + 2x + 4x + 8 = 0")
    print(my_equation)
    print(my_equation.simplified_str())
    solution = my_equation.solve('real')
    print(solution)
    print(my_equation.coefficients())
    return
    my_equation = LinearEquation("3x - 7 + 2x + 6 = 4x - 4 + 6x")
    print(my_equation.show_steps())
    my_equation.plot_solution()
    return
    my_vectors = VectorCollection((1, 4, 3), (7, 5, 3), (2, 2, -4), (-8, 5, 4))
    my_vectors.plot()

    # plot_function("f(x) = sin(x)")
    # plot_multiple(["f(x) = 4x", "f(x) = x^2", "f(x) = x^3", "f(x)= 8", "f(x)=ln(x)", "f(x)=e^x", "f(x)=|x|"])
    plot_complex(3 + 5j, 2 - 1j, 1 + 6j)
    return
    x = Var('x')
    (x ** 2 + 6 * x + 8).export_report("hey.pdf")
    return
    x = Var('x')
    my_ranges = RangeCollection((Range(x, (-1, 1), (LESS_THAN, LESS_THAN)),))
    print(my_ranges)
    return
    x = Var('x')
    y = Var('y')
    (x ** 2 + y ** 2).export_report("3DReport.pdf")
    return
    x = Var('x')
    print((x ** 3 - 6 * x ** 2 + 8 * x).get_report())
    return
    x = Var('x')
    (x ** 3 - 6 * x ** 2 + 8 * x).export_report("report012.pdf")
    return
    vectors = VectorCollection([1, 4, 5], [-5, 2, 7], [3, -2, 6], [6, 4, -1])
    vectors.plot()
    return
    vectors = VectorCollection([1, 4], [-5, 2], [3, -2], [6, 4])
    vectors.plot()
    return

    return
    x = Var('x')
    my_poly = x + 5
    my_fraction = Fraction(3, x + 5)
    print(my_poly + my_fraction)
    return
    import time
    start = time.time()
    my_worksheet = PDFWorksheet(title="Systems of linear equations")
    for _ in range(50):
        my_worksheet.add_exercise(PDFLinearSystem())
        my_worksheet.add_exercise(PDFLinearSystem())
        my_worksheet.add_exercise(PDFLinearSystem())
        my_worksheet.add_exercise(PDFLinearSystem())
        my_worksheet.add_exercise(PDFLinearSystem())
        my_worksheet.end_page()
        my_worksheet.next_page()
    my_worksheet.del_last_page()
    my_worksheet.create("systems.pdf")
    end = time.time()
    print(f"{end - start} seconds elapsed")
    return
    print(random_linear_system(['x', 'y', 'z']))
    return
    # TESTING THE POLYNOMIAL DIVISION
    x = Var('x')
    print((x ** 2 + 6 * x + 8) / (x + 4))
    print((x ** 2 + 6 * x + 8).__truediv__(x + 4, get_remainder=True))
    print((x ** 2 + 6 * x) / 2)
    print((x ** 2 + 6 * x + 8) / (x + 5) * (x + 5))
    return
    my_quadratic = QuadraticEquation("x^2 + 6x + 8 = 0")
    print(my_quadratic.solution)
    my_cubic = CubicEquation("x^3 - 2x^2 + x + 7 = 0")
    print(my_cubic.solution)
    my_quartic = QuarticEquation("x^4 - 16 = 2x^4-32")
    print(my_quartic.solution)
    return
    a, b, c = Var('a'), Var('b'), Var('c')
    results = solve_quadratic_params(a, b, c)
    print(result[0])
    print(result[1])
    return
    arithmetic_prog = ArithmeticProg((2, 4, 6))
    print(arithmetic_prog)
    arithmetic_prog.plot(1, 5)
    geometric_seq = GeometricSeq((3,), ratio=4)
    print(geometric_seq)
    geometric_seq.plot(1, 5)
    recursive_seq = RecursiveSeq("a_n = a_{n-1} + a_{n-2}", first_values=(1, 1, 2))
    recursive_seq.plot(1, 15)
    return
    # Testing the to_dict() and from_dict() methods.
    x = Var('x')
    classes = [Mono, Poly, Exponent, Factorial, Log, Sin]
    expressions = [x, x ** 2 + 6 * x + 8, e ** x, Factorial(x), Log(x), Sin(x)]
    for my_class, expression in zip(classes, expressions):
        assert expression == my_class.from_dict(expression.to_dict()), f"failed in class {my_class}"
    return
    x = Var('x')
    my_exponent = e ** x
    my_exponent.plot()
    my_factorial = Factorial(Sin(0.25 * x))
    my_factorial.plot()
    my_root = Root(Sin(x) + Cos(x))
    my_root.plot()
    plot_vector_2d(1, 1, 3, 3)
    plot_vector_3d((1, 4, 3), (2, 1, 9))
    plot_functions(["f(x) = 4x", "g(x) = x^2"])
    plot_multiple(["f(x) = 4x", "f(x) = x^2", "f(x) = x^3", "f(x)= 8", "f(x)=ln(x)", "f(x)=e^x", "f(x)=|x|"])
    plot_functions_3d(["f(x,y) = sin(x)*cos(y)", "f(x,y) = sin(x)"])
    return
    plot_function_3d("f(x,y) = sin(x)*cos(y)")
    return
    x = Var('x')
    my_poly = x ** 2 + 6 * x + 8
    # print(my_poly.extremums_axes())
    up_range, down_range = my_poly.up_and_down()
    # print(up_range[0])
    # print(down_range[0])
    other_poly = x ** 3 - 3 * x
    print(other_poly.extremums_axes())
    other_up, other_down = other_poly.up_and_down()
    print([up_range.__str__() for up_range in other_up])
    print([down_range.__str__() for down_range in other_down])

    return
    # An interface for representing mathematical ranges
    my_range = Range("3 <= x < 5")
    print(my_range)
    print(my_range.evaluate_when(x=4.5))
    return
    my_worksheet = PDFWorksheet(title="Calculus exercises")
    my_worksheet.add_exercise(PDFPolyFunction())
    my_worksheet.add_exercise(PDFPolyFunction(lang="de"))
    my_worksheet.add_exercise(PDFPolyFunction(lang="es"))
    my_worksheet.end_page()
    my_worksheet.next_page()
    my_worksheet.add_exercise(PDFLinearFunction(lang="de"))
    my_worksheet.add_exercise(PDFLinearFromPoints())
    my_worksheet.end_page()
    my_worksheet.create("calculusPDF1.pdf")
    return
    import time
    start = time.time()
    my_worksheet = PDFWorksheet(title="Mixed Worksheet")
    for i in range(50):
        for j in range(8):
            my_worksheet.add_exercise(PDFQuadraticEquation())
            my_worksheet.add_exercise(PDFCubicEquation())
            my_worksheet.add_exercise(PDFQuarticEquation())
        my_worksheet.end_page()
        my_worksheet.next_page()

    my_worksheet.del_last_page()
    my_worksheet.create("MixedExercises.pdf")
    end = time.time()
    print(f"Created 100 pages and overall 1200 exercises in {end - start} seconds (the slow way)")
    return
    x, y = Var('x'), Var('y')
    (x ** 2 + y ** 2).export_report("3DReport.pdf")
    return
    my_tree = ProbabilityTree()
    pass_exam = my_tree.add(0.4, "Passing")
    fail_exam = my_tree.add(0.6, "Failing")
    excel_exam = my_tree.add(0.3, "Excelling", parent=pass_exam)
    print(my_tree)
    print(my_tree.get_probability("Passing/Excelling"))
    return
    my_worksheet = PDFWorksheet()

    return
    LinearEquation.random_worksheets("linearWorksheet.pdf", 15, get_solutions=True)
    CubicEquation.random_worksheets("cubicWorksheet.pdf", 15, get_solutions=True)

    return
    x = Var('x')
    (x ** 3 - 6 * x ** 2 + 8 * x).export_report("report3.pdf")
    return
    import time
    start = time.time()
    PolyEquation.random_worksheets("worksheet5", num_of_pages=15, get_solutions=True)
    end = time.time()
    print(end - start)
    return
    PolyEquation.random_worksheet(path="aaaaa", get_solutions=True)
    return
    my_poly = Poly("-3x^6+30x^5-93x^4+90x^3")
    print(my_poly.data())
    my_poly.print_report()
    return
    x = Var('x')
    solutions = [0, 5, 3, 2]
    expression = "-3x^6+30x^5-93x^4+90x^3"
    func = lambda x: -3 * x ** 6 + 30 * x ** 5 - 93 * x ** 4 + 90 * x ** 3
    der = lambda x: -18 * x ** 5 + 150 * x ** 4 - 372 * x ** 3 + 270 * x ** 2
    coefficients = [-18, 150, -372, 270, 0, 0]
    print(durand_kerner(der, coefficients))
    # print(aberth_method(func, der, coefficients))
    return
    print(Poly(expression).extermums())
    return
    print(solve_poly_by_factoring([2, 4, -1, -6, -3]))
    print(solve_poly_by_factoring([1, 0, 0, 0, -1, -1]))
    return
    plot_function("f(x) = x^3 +2x^2 - 6x - 5")

    return
    plot_function_3d("f(x,y) = sin(x)*cos(y)")
    return
    x = Var('x')
    print((2 * Sin(x) ** 5) / (Cos(x) ** 2))
    print((2 * Sin(3 * x) * Sin(x) ** 4) / (Cos(x)))
    return
    x = Var('x')
    print(- x ** 2 - 120)
    print(Mono("3x^2") - Factorial(5))
    print(Mono("3x^2") - Factorial(5) == -x ** 2 - 120)
    return
    x, y = Var('x'), Var('y')
    print(Sin(x) * Cos(y) - TrigoExpr("sin(x)*cos(y)"))
    # print(Sin(x) + Cos(y) == TrigoExprs("sin(x) + cos(y)"))
    return
    my_func = Function("f(x)=x^2 + 6x + 8")
    print(my_func.derivative())
    return
    x = Var('x')
    y = Var('y')
    my_mono = x ** 2
    my_mono /= x
    print(my_mono)
    other_mono = 3 * x ** 2 * y ** 4
    other_mono /= 1.5 * x * y
    print(other_mono)
    other_mono1 = 2 * x
    other_mono1 /= 5
    print(other_mono1)
    return
    print((Log(10000) / Log(100)) == Log(100))
    return
    x = Var('x')
    print(Log(10) + Log(100))
    return
    x = Var('x')
    my_abs = Abs(x)
    my_abs.plot(step=0.01)
    y = Var('y')
    my_abs = Abs(x + y)
    my_abs.plot(step=0.3)
    return
    x = Var('x')
    y = Var('y')

    better_trigo = Sin(x) + Cos(y) * Ln(x)
    better_trigo.plot()
    return
    # my_trigo = Sin(x) + Cos(x)
    # my_trigo.plot()
    other_trigo = Sin(x) * Cos(y)
    other_trigo.plot()
    another_trigo = Sin(x) + Cos(y)
    another_trigo.plot()

    return
    x = Var('x')
    my_factorial = Factorial(Sin(x))
    my_factorial.plot(start=-5, stop=5, step=0.01)
    return
    x = Var('x')
    print(Sin(4 * x) / Cos(2 * x))
    return
    f = Function("f(x) = sin(x)!")
    f.plot(start=-5, stop=5, step=0.01)
    return
    x = Var('x')
    my_factorial = Factorial(x ** 2 + 6 * x + 8, coefficient=Sin(x), power=2)
    print(my_factorial)
    return
    print(handle_factorial("3! + x! + 2x! - 5 + (3x+5)! + sin(x)!"))
    return
    import math
    x = Var('x')
    my_lambda = Sin(x).to_lambda()
    print(my_lambda(math.pi / 2))
    return
    print(TrigoExpr('sin(3x)', dtype='poly'))
    print(TrigoExpr('sin(log(2x+5))', dtype='log'))
    return
    print(Log("log(2x^2+5)", dtype='poly'))
    return
    print(log_from_str("log(3x+5)"))
    return
    x = Var('x')
    print(Sin(x).simpson(0, pi, 20))
    return
    x = Var('x')
    print(Sin(x).reinman(0, pi, 20))
    return
    x = Var('x')
    print(Sin(x).trapz(0, pi, 20))
    return
    x = Var('x')
    # print((8 * Sin(x) ** 2 * Cos(x) ** 2) / (2 * Sin(x) * Cos(x)))
    # print(Sin(x) / Cos(x))
    print(3 * Sin(2 * x) / (Cos(x) * Sin(x)))
    print(4 * Sin(2 * x) * Cos(2 * x) / (Cos(x)))
    print(Cos(x) / (Sin(2 * x)))
    # print((3 * x * Sin(x)) / Log(x))
    return
    x = Var('x')
    print(Sin(x) * Cos(x))
    print(5 * Sin(2 * x))
    print(3 * x ** 2 * Tan(Log(x)))
    return
    x = Var('x')
    print(Sin(2 * x) == Sin(2 * x))
    print(Sin(x) == Sin(x + 2 * pi))
    print(Sin(x) * Cos(x) == 0.5 * Sin(2 * x))
    print(Sin(x) == Sin(x + 5))
    return
    x = Var('x')
    print(Sin(x) + Sin(x))
    print(Sin(x) + Cos(x))
    print(Sin(pi / 2) + 4)
    return
    my_trigo = TrigoExpr("2sin(x)*cos(x)")
    return
    # Conversions Testing
    x = Var('x')
    my_sin = Sin(2 * x)
    print(my_sin.to_cos())
    other_sin = Sin(x) ** 2
    print(other_sin.to_cos())
    return
    x = Var('x')
    my_trigo = 2 * Cos(x) * Sin(x) * Tan(x)
    print(my_trigo)
    my_trigo.plot()
    return
    poly_example = FastPoly("x^2 - 4")
    print(poly_example.extremums())
    return
    poly, roots = random_polynomial(get_solutions=True, degree=random.randint(2, 5))
    random_poly = FastPoly(poly)
    poly_integral = random_poly.integral()
    print(poly_integral)
    return
    scatter_functions(("f(x) = 2x", "f(x) = x^2"))
    return
    scatter_function_3d("f(x,y) = x + y")
    scatter_function_3d(lambda x, y: sin(x) * cos(y))
    scatter_function_3d(Function("f(x,y) = xy"), title="Hi there!")
    return
    scatter_function("f(x) = sin(x)")
    scatter_function(lambda x: x ** 2)
    scatter_function(Function("f(x) = 2x"))
    return
    scatter_dots_3d([4, 2, 8, -5, 3, 8, 9, -6, 1], [1, 9, -2, 3, 4, 5, 1, 6, 7], [1, -1, 3, 4, 8, 1, 4, 2, 6])
    return
    scatter_dots([1, 2, 3, 4], [2, 4, 6, 8])
    plot_complex(complex(5, 4), complex(3, -2))
    return
    plot_function_3d("f(x,y) = sin(x)*cos(y)")
    scatter_function_3d("f(x,y) = sin(x)*cos(y)")
    return
    print(FastPoly("7").integral(c=5, variable='n'))
    print(FastPoly("3x+7").integral(c=5))
    print(integral("3x+7", c=5, get_string=True))
    print(derivative("x^2 + 6x + 5", get_string=True))
    return
    print(ParseExpression.coefficients_to_str([-2, -6, -7]))
    return
    print(integral("3x+7"))
    print(derivative("x^2 + 6x + 5"))
    return
    print(ParseExpression.to_coefficients("x^2 - 6x - 8"))
    print(ParseExpression.to_coefficients("2x - 6"))
    print(ParseExpression.to_coefficients("-4x^3 + 6"))
    return
    print(integral([3, 7]))
    return
    function_chain = FunctionChain("f(x) = x^2", "g(x) = x+5", "h(x) = sin(x)")
    print(function_chain.execute_all(1))
    return
    functions = FunctionCollection(Function("f(x) =x^2"), "g(x) = sin(x)", Function("h(x) = 2x"))
    print(functions.random_value(1, 10))
    functions = FunctionCollection("f(x,y) = x + y", "g(x,y) = x - y", "h(x,y) = sin(x) * cos(y)")
    print(functions.random_value(1, 10))
    print(functions[::-1])
    return
    my_graph = Graph2D([Function("f(x) = x**2"), Function("f(x) = 2x")])
    my_graph.add(Function("f(x) = x^3-3"))
    my_graph.add(Poly("2x^4 - 3x + 5"))
    my_graph.add(Circle(5, (0, 0)))
    my_graph.plot()
    return
    fast_poly = FastPoly("x^2 + 6x + 8")
    print(fast_poly.derivative())
    return
    print(derivative([1, 2, 1]))
    print(solve_polynomial("x^2 + 3x^3 + 12x + 5 = 2x -4 + x^2"))
    return
    poly1 = FastPoly("2x^3 + 6x^2 + 5")
    poly2 = FastPoly("x^4 + x^3 - 5x^2 + 6")
    print(poly1 - poly2)
    return
    print(add_or_sub_coefficients([1, 2, 1], [2, 1], mode='sub'))
    return
    print(ParseEquation.parse_quadratic("x^2 + 2x + 1", strict_syntax=True))
    print(ParseEquation.parse_quadratic("2x^2 + 3x + 1 = x^2 + 2x"))
    return
    fast_poly = FastPoly("x^3 - 2x + 1")
    fast_poly.plot()
    return
    poly1 = FastPoly("5")
    poly2 = FastPoly("x^2 + 6x + 8")
    print(poly1.try_evaluate())
    print(poly2.try_evaluate())
    return
    my_poly = FastPoly("x^2 + y^2")
    my_poly.assign(x=5)
    print(my_poly)
    return

    return
    poly1 = FastPoly("2x^3 + 5x -7")
    poly2 = FastPoly("x^2 + 4y^2 - x^3 + 5x + 6")
    print(poly1 + poly2)
    return
    print(random_polynomial(degree=5))
    return
    x, n, m, k, d = Var('x'), Var('n'), Var('m'), Var('k'), Var('d')
    print((x + n) * (x + m) * (x + k) * (x + d))
    return
    print(random_polynomial2(6))
    print(CubicEquation.random(digits_after=0, get_solutions=True))
    return
    print(solve_polynomial([1, 0, 0, 0, 0, 0, 2, 0, 0, -18]))
    return
    my_poly = Poly("x^9 + 2x^3 -18")
    print(my_poly.coefficients())
    print([expression.__str__() for expression in coefficients_to_expressions(my_poly.coefficients())])
    print(my_poly.derivative())
    return
    # print(solve_polynomial("x^9 + 2x^3 -18"))
    return
    print(ParseExpression.parse_poly_equation("x^3 + y + 7x - 2 = 2x^3 - 5x + 3y^4 + 2y + 6", variables=('x', 'y')))
    return
    my_equation = LinearEquation("3x + 3y = 6 - 2x + 5y -z + 4x + z", variables=('x', 'y', 'z'))
    print(my_equation.simplify())
    return
    solutions = solve_poly_system(["x^2 + y^2 = 25", "2x + 3y = 18"], {'x': 2, 'y': 1})
    print(solutions)
    return
    poly1, poly2 = FastPoly("x^2 + x + 5 + 2y + z"), FastPoly("x^3 + 2x + 7 + 2y^2 - 6 + z^2")
    print((poly1 + poly2))
    poly3, poly4 = FastPoly("x^2 + 2x + 5"), FastPoly("x^5 + 3x^3 - 2x + 1")
    poly3.plot()
    poly4.plot()
    graph_obj = Graph2D()
    graph_obj.add(poly3)
    graph_obj.add(poly4)
    graph_obj.plot()
    return
    quad_equation = QuadraticEquation("x^2 + 2x + 1 = 0")
    print(quad_equation.coefficients())
    print(quad_equation.solve())
    return
    my_poly = FastPoly("x^2 + 5")
    print(my_poly)
    print(-my_poly)
    print(my_poly.python_syntax())
    print(my_poly.when(x=2))
    my_poly.assign(x=3)
    print(my_poly)
    return
    my_poly = FastPoly("x^4 + x + y + y^2 - 7")
    print(my_poly.degree)
    return
    my_poly = FastPoly([1, 0, 0, 0, -16])
    print(my_poly.__str__())  # Unparse the polynomial..
    print(my_poly.roots())
    print(my_poly.degree)
    return
    my_poly = FastPoly("x^4 - 16")
    print(my_poly.variables_dict)
    print(my_poly.to_dict())
    print(my_poly.to_json())
    print(my_poly.roots())
    return
    func = Function("w24dfgd")
    func(6, 3, 2, 3, 6, 3)
    return
    x = Var('x')
    expression = x ** 5 + 6 * x - 2 * x

    return
    my_graph = Graph2D([Function("f(x) = x**2"), Function("f(x) = 2x")])
    my_graph.add(Function("f(x) = x^3-3"))
    my_graph.add(Poly("2x^4 - 3x + 5"))
    my_graph.add(Circle(5, (0, 0)))
    my_graph.plot()
    return
    print(Matrix.random_matrix((3, 3), (-15, 15)))
    return
    print(QuadraticEquation.random(strict_syntax=True))
    return
    print(ParseExpression._parse_monomial("y^2", ('x', 'y'), True))
    print(ParseExpression.parse_quadratic('x^2 + 2x - 8', variables=('x',), strict_syntax=True))
    print(ParseExpression.parse_cubic("-x^3 + 5x^2 + 7x -4", variables=('x',), strict_syntax=True))
    print(ParseExpression.parse_quartic("5x^4 + 2x^2 - 7x -6", variables=('x',), strict_syntax=True))
    print(ParseExpression.parse_polynomial("3x^5 - 2x + 5", ('x',), True, numpy_array=True))

    return
    x = Var('x')
    print((Sin(2 * x) + 3 * Cos(5 * x + 4)).to_json())
    return
    x = Var('x')
    print((3 * x ** 2 + 6 * x + 7).to_json())
    return
    first_matrix = Matrix([[1, 2], [3, 4]])
    second_matrix = Matrix([[0, 5], [6, 7]])
    print(first_matrix.kronecker(second_matrix))
    return

    return
    my_matrix = Matrix([[1, 2], [3, 4], [5, 6]])
    print(my_matrix.transpose())
    return
    my_matrix = Matrix([[4, 3, 3], [6, 5, 8], [7, 2, 5]])
    print(my_matrix.determinant())
    return
    my_tree = ProbabilityTree()
    pass_exam = my_tree.add(0.4, "Passing")
    fail_exam = my_tree.add(0.6, "Failing")
    excel_exam = my_tree.add(0.3, "Excelling", parent=pass_exam)
    return

    return
    print(gradient_ascent(lambda x: -2 * x, 8))
    return
    accumulator = 0
    for _ in range(10000):
        start = time.time()
        solve_linear("3x+5+5x+9776-2040x+38x+5555 = -673x + 1231 + 12x - 21 + 34x + 321x - 351x + 132x",
                     variables=('x',))
        end = time.time()
        accumulator += end - start
    print(accumulator)
    return
    x = Var('x')
    my_poly = x ** 2 + 7
    my_poly.scatter()
    return
    linear_equation = LinearEquation("3x + 5 = 14")
    linear_equation.plot_solution()  # TODO: add a label to the intersection, and handle cases where there isn't 1 sol
    return
    print(solve_quartic(1, 0, 0, 0, -16))
    return
    linear_system = LinearSystem(("3x - 4y + 3 = -z + 9", "-3x + 5 -2z = 2y - 9 + x", "2x + 4y - z = 4"))
    print(linear_system.get_solutions())
    # linear_system.solutions - will also work
    return
    solutions = solve_linear_system(("3x - 4y + 3 = -z + 9", "-3x + 5 -2z = 2y - 9 + x", "2x + 4y - z = 4"))
    print(solutions)
    return
    return

    return
    print(my_tree.get_probability("Passing/Excelling"))
    print(my_tree.get_depth(excel_exam))
    print(my_tree.num_of_nodes())
    print(my_tree.get_node_path("Excelling"))
    # print(my_tree.export_xml("newTree.xml"))
    # print(my_tree.to_json("newTree.json"))
    return
    print(PolyEquation.random_quadratic(all_powers=True))
    return
    # print(approximate_jacobian([lambda x, y:2*x**2+y**2, lambda x,y:3*x-5*y], values=[1, 1]))
    print(broyden([lambda x, y: 2 * x ** 2 + y ** 2, lambda x, y: 3 * x - 5 * y], initial_values=[1, 1]))
    return
    print(trapz(lambda x: sin(x), 0, pi, 1000))
    print(simpson(lambda x: sin(x), 0, pi, 1000))
    return
    print(solve_linear_inequality("3x+5>=10"))
    return
    # Simple intersection between two circles
    first_circle = Circle(radius=5, center=(2, 3))
    second_circle = Circle(radius=7, center=(5, 6))
    print(first_circle.intersection(second_circle))
    return
    print(solve_poly_system(["4x^2-y^3 + 28 = 0", "3x^2 + 4y^2 -145 = 0"], {'x': 1, 'y': 1}))
    return
    a = Var('a')
    b = Var('b')
    my_circle = Circle(a, (a + 2, b - 4))
    print(my_circle.equation)
    my_circle.y_intersection()
    return
    my_circle = Circle(5, (8, 2))  # Example of circle
    print(my_circle.equation)
    print(my_circle.y_intersection())
    my_circle.plot()
    return
    # Graph class

    return
    x = Var('x')
    y = Var('y')
    print((x + y) * (x - y))
    return
    print(5 * x + 15 - (5 * x + 15))
    print(((x ** 2 + 8 * x + 15) / (x + 3)))
    return
    my_poly = x + 2 * x ** 2 + 5 - 4 * x ** 3
    my_poly.sort()
    print(my_poly)
    return
    my_function = Function("f(x,y) = x*y")
    my_function = my_function.range_3d(x_start=0, x_end=3, x_step=1, y_start=0, y_end=3, y_step=1)
    print(my_function)
    return
    my_function = Function("f(x) = sin(x)")
    my_function.plot(start=-2 * pi, stop=2 * pi)
    return
    with Function("f(x) = x^2") as fn:
        pass  # Do some stuff here

    my_equation = PolyEquation("x^2 = 1")
    print(my_equation.solution)

    return
    x = Var('x')
    original_poly = x ** 2 + 6 * x + 8

    with copy(original_poly) as copied_poly:
        copied_poly -= 6 * x + 7
        print(f"copy: {copied_poly}")
    print(f"original: {original_poly}")

    return
    v = Vector((5, 3, 1))
    print(v)
    v.plot()
    return
    point_collection = PointCollection(points=[1, 4, 7, 9, 11])  # 1D Plotting
    point_collection.scatter()
    return
    x = Var('x')
    y = Var('y')
    my_poly = x ** 2 + 6 * x + 8
    # print(my_poly / 6)  # First test = Success !
    # print((my_poly) / (x + 2))  # Second test - success !
    # print((my_poly) / (x))  # Third test - Partial Success, since Fractions weren't used ..
    other_expression = x * y + y ** 2
    print(other_expression)
    other_expression2 = x + y
    print(f"result={other_expression / other_expression2}")
    return
    x = Var('x')
    my_root = Root(2 * x)
    print(my_root)
    print(my_root.derivative())
    return
    x = Var('x')
    y = Var('y')
    original_poly = 3 * x ** 2 * y + 5 * x - 2 * y + 7 * y ** 3
    print(original_poly.partial_derivative("x"))
    print(original_poly.partial_derivative("y"))
    print(original_poly.partial_derivative("yx"))
    print(original_poly.partial_derivative("xy"))

    return
    monomial = Mono("3x^2*y^3")
    print(monomial)
    print(monomial.partial_derivative("xy"))
    return

    return
    my_matrix = Matrix(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(my_matrix)
    print(f"determinant is {my_matrix.determinant()}")

    return
    x = Var('x')
    root_object = Root(3 * x)
    print(root_object.try_evaluate())
    print(root_object.when(x=3))
    print(root_object.python_syntax())
    root_object.plot()
    log_object = Log(Log(x) + 3 * x)
    log_object.plot()
    return
    x = Var('x')
    print(Sin(x) * Log(x))
    general_expression = 3 * x + 6 + Log(x)
    print(f"general expression is {general_expression}")
    return
    x = Var('x')
    my_matrix = Matrix(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(Log(x + 8) * 3)
    print(my_matrix * (Log(x) - Tan(2 * x)))
    return
    x = Var('x')
    y = Var('y')
    my_root = Root(x + y)
    print(my_root ** 4)
    return
    print("starting ... ")
    x = Var('x')
    first = Mono("3x^2")
    first *= 5
    first *= 3 * x
    first *= 2 * x + 6
    print(first)
    log_of_x = Log(x)
    print(log_of_x)
    first *= log_of_x
    print(first)
    return
    trigo_expr = TrigoExpr("sin(2)")
    my_root = Root(4 * x)
    another_root = Root(2 * x)
    print(my_root * another_root)
    return
    x = Var('x')
    poly_example = 3 * x ** 4 + 3 * x
    print(poly_example.gcd())
    return
    x = Var('x')
    my_log = Log(x)
    new_log = 2 * my_log ** 3
    print(new_log - my_log)
    return
    other_log = Log(2 * x + 6)
    print(my_log + other_log)

    return
    print(my_log)
    my_log.assign(x=50)
    num = my_log.try_evaluate()
    print(num)
    return
    x = Var('x')
    first_expression = -1 * Cos(x)
    print(isinstance(first_expression, TrigoExpr))
    second_expression = Cos(pi - x)
    print(first_expression.__eq__(second_expression))  # Basic comparison !

    return
    x = Var('x')  # Declaring a variable named x
    poly = 3 * x - 2  # Creating a polynomial
    poly.assign(x=4)  # Assigning a value to it, so it will hold a free number
    number = poly.try_evaluate()  # Extract the free number from the polynomial
    print(f"poly represents {poly} and its type is {type(poly)}")
    print(f"number is {number} and its type is {type(number)}")
    return
    x = Var('x')
    expr = TrigoExprs("sin(x) + 3sin(y)")
    print(expr)
    a = TrigoExpr(coefficient=5, expressions=[[TrigoMethods.SIN, x, 2], [TrigoMethods.COS, 3 * x, 4]])
    print(a)
    b = TrigoExpr(coefficient=2, expressions=[[TrigoMethods.SIN, x, 4]])
    print(b)
    print(a * b)
    print(a / b)
    return

    return
    equation = LinearEquation("3x + 5 - 6x = - x + 7 - 9x + 12")
    print(equation.show_steps())
    return
    fibonacci = RecursiveSeq("a_n = a_{n-1} + a_{n-2}", (1, 1, 2))
    print(f"Fibonacci at place 9 is {fibonacci.at_n(9)}")
    factorial = RecursiveSeq("a_n = a_{n-1} * n", (1, 2))
    print(f"Factorial of 7 is {factorial.at_n(7)}")
    return

    return

    plot_complex(complex(5, 4), complex(3, -2))
    return
    func = Function("f(x,y) = sin(x)*sin(y)")
    func.plot(start=-5, stop=5, step=0.05)
    return
    x = Var('x')
    my_root = PolyRoot(3 * x + 5)
    my_root.try_evaluate()
    my_root.assign(x=4)
    print(my_root.try_evaluate())
    return
    fequation = LinearEquation("12 + 5 - 6 = -4 + 8 - 7 + 1 - 16")
    print(fequation.show_steps())

    return
    original_poly = lagrange_polynomial([x for x in range(1, 20)], [ln(x) for x in range(1, 20)])
    plot_function(lambda x: original_poly.when(x=x).expressions[0].coefficient, start=-20, stop=20, step=0.01)
    print(original_poly.when(x=e ** 2))

    return
    function = Function("x => 2^x")
    return

    my_equation = LinearEquation("3x - 5 = 16", variables_dict=('x',))
    solution = my_equation.solve()
    print(solution)
    poly_equation = PolyEquation(3 * x ** 3 - 5 * x + 9, -4 * x ** 4 + 9)
    print(poly_equation.solution)

    return
    # Function("f(x) = x%2==0").plot(start=0,end=10,step=1)

    custom_recursion = RecursiveSeq("a_n = a_{n-1}^2 + 0.5*ln(a_{n-2})", [e, 1, 1.5])
    print(f"The custom recursive method is {custom_recursion.at_n(4)} at place 4")

    return
    print(lambda_from_recursive("a_n = a_{n-1} + a_{n-2}"))
    print(to_lambda("a_np1 + 5", ["a_np1"])(2))
    return
    f_0 = lambda n: 2 * n ** 3 - 5 * n ** 2 - 23 * n - 10
    f_1 = lambda n: 6 * n ** 2 - 10 * n - 23
    f_2 = lambda n: 12 * n - 10
    initial_value = 0
    print(chebychevs_method(f_0, f_1, f_2, initial_value))
    return
    func = lambda x: 5 * x ** 4 - 1
    der = lambda x: 20 * x ** 3
    coefficients = [5, 0, 0, 0, -1]
    print(aberth_method(func, der, coefficients))
    return
    will_it_work = Function("x+y")
    print(will_it_work)
    return
    print(solve_poly_by_factoring([1, -6, 11, -6]))
    return
    print(solve_cubic(1, -6, 11, -6))
    return
    mat = Matrix([[1, 1], [1, 1]])
    print(mat.determinant())
    mat.gauss()
    print(mat)
    return
    print(durand_kerner2([1, 0, 0, -27]))
    return
    print(steffensen_method(lambda x: 2 * x ** 3 - 5 * x - 7, 8))
    return
    parabola = lambda x: x ** 2 - 5 * x
    print(bisection_method(parabola, 2, 9))
    return
    return

    return
    one = lambda x: x ** 2 - 6 * x + 8
    der = lambda x: 2 * x - 6
    print(ostrowski_method(one, der, 3))
    return
    func = Function("f(x) = x^4 + 8")
    print(func.roots())
    return
    a = TrigoExpr("3*sin(-2x^2+5)*cos(-x+2)*tan(4x-3)")
    print(a ** 3)

    return
    func = Function("f(x) = x^5 - 6x + 8")
    coefficients = func.coefficients()
    import time
    start = time.time()
    for i in range(10000):
        func.roots()
    end = time.time()
    start2 = time.time()
    for i in range(10000):
        roots(coefficients)
    end2 = time.time()

    print(f"first time is {end - start}")
    print(f"second time is {end2 - start2}")
    return

    return
    print(solve_quartic(1, 0, 0, 0, -16))  # TEST HAS FAILED
    return
    f_0 = lambda n: n ** 5 - n + 1
    f_1 = lambda n: 5 * n ** 4 - 1
    print(aberth_method(f_0, f_1, [1, 0, 0, 0, -1, 1]))

    return
    x = Var('x')
    print(((x ** 2 - 6 * x + 8) / (x - 2))[0])
    vec1, vec2 = Vector((6, 8)), Vector(direction_vector=(7, 8), start_coordinate=(-1, -3))
    print(vec1.intersection(vec2))

    return
    points = PointCollection()
    points.add_point(Point((6, 4, 2)))
    points.add_point(Point((2, 5, 9)))
    points.add_point(Point((12, 3, -4)))
    print(points.__repr__())
    points.scatter()
    PointCollection([(43, 99), (21, 65), (25, 79), (42, 75), (57, 87), (59, 81)]).scatter_with_regression()
    PointCollection([(2, 8), (4, 7), (6, 16)]).scatter_with_regression()

    return
    ages = (43, 21, 25, 42, 57, 59)
    glucose_levels = (99, 65, 79, 75, 87, 81)
    print(linear_regression(ages, glucose_levels, get_values=True))
    return
    point = Point((4, 2))
    point.plot()
    return
    sin_x = TrigoExpr("sin(x)")
    sin_y = TrigoExpr("sin(y)")
    print(sin_x * sin_y)
    return
    vector = Vector((0.5, 0.5, 0.5))
    vector.plot()
    return
    print(TrigoExprs("4sin(2x)-3sin(3x)*cos(5x)"))
    return
    print(TrigoExpr("4sin(4x^2+4x -2x^3)^2*2cos(3x^2-5x)"))
    return
    x = Var('x')

    return
    func = Function("f(x)=-(20x^4)+5x^3+17x^2-29x+87 ")
    print(func.derivative())
    return
    func = Function("f(x) = 2x^2+7x+3")
    print(func.try_poly_derivative())
    return

    return
    f_0 = lambda n: 2 * n ** 3 - 5 * n ** 2 - 23 * n - 10
    f_1 = lambda n: 6 * n ** 2 - 10 * n - 23
    f_2 = lambda n: 12 * n - 10
    func = Function("f(x)=-20(x^4)+5x^3+17x^2-29x+87 ")
    print(func(3))
    # print(laguerre_method(f_0, f_1, f_2, -7523, 3))
    x = Var('x')
    g_0 = -20 * x ** 4 + 5 * x ** 3 + 17 * x ** 2 - 29 * x + 87
    g_1 = g_0.derivative()
    lambda_expression = str(g_0)
    print(lambda_expression)
    print(aberth_method2(g_0.to_lambda(), g_1.__to_lambda(), 3))
    return
    origin_function = lambda n: 2 * n ** 3 - 5 * n ** 2 - 23 * n - 10
    first_derivative = lambda n: 6 * n ** 2 - 10 * n - 23
    initial_value = -10
    print(newton_raphson(origin_function, first_derivative, initial_value))
    return
    func = Function("f(x) = x^2 -6x + 8")
    print(func.root_with_newton(1))
    return

    print(solve_cubic(1, -6, 11, -6))
    return
    print(bisection_method(lambda x: x ** 2 - 5 * x, 3, 10))
    print(steffensen_method(lambda x: 2 * x ** 3 - 5 * x - 7, 5))

    return

    f = lambda n: 2 * n ** 3 - 5 * n ** 2 - 23 * n - 10
    print(inverse_interpolation(f, -5, -4, -3))
    return
    f = lambda n: 2 * n ** 3 - 5 * n ** 2 - 23 * n - 10
    print(secant_method(f, 7, 4, 0.00001))
    return
    f_0 = lambda n: 2 * n ** 3 - 5 * n ** 2 - 23 * n - 10
    f_1 = lambda n: 6 * n ** 2 - 10 * n - 23
    print(newton_raphson(f_0, f_1, 15234))
    return
    poly_func = Function("f(x) = 3x^2 + 6x + 7")
    print(poly_func.classification)
    print(poly_func.derivative())
    print(poly_func.derivative().derivative())
    return
    expression = Poly("a^2+2*a*b+b^2")
    print(expression(a=2))
    return

    return
    expression = Mono("3x^2*y^4")
    print(expression)
    return
    parabola = Function("g(x) = x^2-2x+8")
    print(parabola)
    return
    parabola.scatter2d(step=0.1)
    return
    LinearEquation.random_worksheet()
    return
    x = Var('x')
    print((3 * x) / (3 * x))
    print((3 * x + 6 * x ** 2).divide_by_gcd())
    return
    eq = LinearEquation('-3x+5=11')
    print(eq.solve())
    return
    mat = Matrix(matrix=((1, 2, 3), (4, 5, 6), (7, 8, 9)))
    for i, j in mat.range():
        print(f"The item in row {i} and column {j} equals to {mat[i][j]}")

    return

    for row in mat.matrix:
        for item in row:
            print(item)
    return
    eq = Equation("3x = 3x+6", calc_now=True)
    print(eq.solution)
    return
    x = Var('x')
    func = Function(lambda x: sin(x))
    func.plot()
    y = Var('y')
    print((4.8 * x ** 2 * y ** 2 + 3.6 * x ** 3 * y ** 2 + 9.6 * x ** 2 * y).gcd())
    (3 * x ** 2 + 5 * x).to_lambda()
    func = Function("f(x) = 6ln(ex)-7")
    func.plot()
    return
    print(((x ** 2 + 8 * x + 16) / (x + 4))[0])
    return
    a, b = Var('a'), Var('b')
    print((a + b) ** 2)
    return
    x = Var('x')
    print(((x ** 2 - 2 * x - 8) / (x + 2))[0])
    return
    y = Var('y')
    print(3 * y == 2 * x + 6)
    return
    example_function = Function("f(x)=xe^sin(x)-3ln(|4x|)")
    example_function.plot(start=-10, stop=10)
    return
    four = Mono(4)
    print(four)
    return

    x = Var('x')
    print((3 * x ** 2 + 2 * x + 5).derivative())
    print((4 * x ** 3 + 3 * x ** 2).integral())
    return

    return
    import time
    start = time.time()
    for i in range(1000000):
        example_function(i)
    end = time.time()
    print(f'{end - start} seconds elapsed')
    print(example_function.__func_expression)
    return
    linear = Function("f(x)=3ln(x)-xsin(x)+4")
    linear.plot(start=-20, stop=20)
    Function.plot_all("f(x)=ln(x)", lambda x: e ** x, 'g(x)=x**2', 'h(x)=tan(x)', 'f(x)=sin(x)')
    return

    print("hello")
    parabola = Function("f(x)=cos(x)")
    print(linear.search_intersections(parabola))
    return
    invalid_sine = Function("f(x)=xln(-2x**2+5)")
    invalid_sine.scatter2d(start=-20, stop=20, ymin=-20, ymax=20)
    return
    x = Var('x')
    expressions = 2 * x ** 4 - 32
    print(expressions)
    print(expressions.roots())
    return
    y = Var('y')
    print(5 - y)
    print((x + y) / 2)
    # print((x ** 2) * (x - 2))
    expressions = x + y + 5
    print(expressions ** 4)
    return
    expression1 = Mono(coefficient=-6, variables_dict={'x': 3})  # -6x^3
    expression2 = Mono(coefficient=4, variables_dict={'x': 5})  # 4x^5
    expression3 = Mono(coefficient=3, variables_dict={'x': 4})  # 3x^4
    expression4 = Mono(coefficient=-1, variables_dict={'x': 2})  # -x^2
    expressions = Poly([expression1, expression2])
    expressions2 = Poly([expression3, expression4])
    print(expressions * expressions2)

    return
    func = Function("f(x)=ln(x)")
    print(func(6))
    print(func.search_roots_in_range(val_range=(-100, 100)))
    func.scatter2d()
    """__equations = LinearSystem(["3x+y-2z=6+z-4x", "5x+2z-5y=-6+4y+3z", "-z-5=-9x+7y-4z"])
    __equations.print_solutions()
    function = Function("f(x) = x**2-3x + 3sin(4x) - 7")
    function.scatter()
    vectors = VectorCollection(Vector(start_coordinate=(6, 5, 9), end_coordinate=(9, 8, 7)), (6, 7, 2))
    vectors.plot_all()
    twoDVectors = VectorCollection(Vector((3, 4)), Vector((-5, -3)), Vector((2, -1)))
    twoDVectors.plot_all()
    print(polynomial_solve([1, 2, -17, -18, 72]))
    return"""
    return

    return
    # print(polynomial_solve([2, -2, -14, 2, 12]))
    tree = ProbabilityTree(root=Occurrence(1, "taking_the_test"))  # STEP 1
    pass_test = tree.add(Occurrence(0.4, "pass_test"))  # STEP 2
    fail_test = tree.add(Occurrence(0.6, "fail_test"))  # STEP 2
    ace_test = tree.add(Occurrence(0.1, "ace_test"), parent=pass_test)  # STEP 3
    # Here, we did the following steps:
    # 1. Definedc the tree, with the root - the probability to take the test is 1, i.e, everyone takes the test
    # 2. We declared the probability to pass the test is 0.4, and to fail 0.6, by creating son nodes to the root
    # 3. We said that out of the 40% who passed the test, only 10% ( 0.1 ) aced it, by creating another node
    # That means that only 4% aced the test ( 0.4 * 0.1 = 0.04 ). Lets verify it !
    print(tree.get_probability(path=["taking_the_test", "pass_test", "ace_test"]))


if __name__ == '__main__':
    main()
