from __future__ import annotations
import math
from math import (
    sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
    asinh, acosh, atanh, sqrt, e, pi, floor, ceil, log,
    log10, log2, exp, erf, erfc, gamma, lgamma, tau, comb, degrees, radians
)
import random
import string
import warnings
import inspect
import re
from functools import reduce
from collections import Counter
from typing import Union, Tuple, List, Optional, Any, Callable, Iterator, Set, Iterable
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.units import cm
from reportlab.platypus import Image
from reportlab.lib import utils

from kiwicalc.core.constants import (
    TRIGONOMETRY_CONSTANTS, MATHEMATICAL_CONSTANTS,
    ptn, number_pattern, allowed_characters, pi, e, tau
)
from kiwicalc.core.interfaces import IExpression

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

def decimal_range(start: float, stop: float, step: float=1):
    while start <= stop:
        yield start
        start += step

def extract_coefficient(coefficient: str) -> float:
    """[method for inside use]"""
    return -1 if coefficient == '-' else 1 if coefficient in ('+', '') else float(coefficient)

def format_coefficient(coefficient: 'Union[int, float, IExpression]') -> str:
    if coefficient == 1:
        return ''
    if coefficient == -1:
        return '-'
    return coefficient.__str__()

def format_free_number(free_number: Union[int, float]):
    if free_number == 0:
        return ''
    if free_number < 0:
        return f'{round_decimal(free_number)}'
    return f'+{round_decimal(free_number)}'

def linear_regression(axes, y_values, get_values: bool=False):
    """
    Receives a collection of x values, and a collection of their corresponding y values, and builds a fitting
    linear line from then. If the parameter "get_values" is set to True, a tuple will be returned : (slope,free_number)
    otherwise, a lambda equation of the form - lambda x : a*x + b, namely, f(x) = ax+b , will be returned.
    """
    if len(axes) != len(y_values):
        raise ValueError(f'Each x must have a corresponding y value ( Got {len(axes)} x values and {len(y_values)} y values ).')
    n = len(axes)
    sum_x, sum_y = (sum(axes), sum(y_values))
    sum_x_2, sum_xy = (sum((x ** 2 for x in axes)), sum((x * y for x, y in zip(axes, y_values))))
    denominator = n * sum_x_2 - sum_x ** 2
    b = (sum_y * sum_x_2 - sum_x * sum_xy) / denominator
    a = (n * sum_xy - sum_x * sum_y) / denominator
    if get_values:
        return (a, b)
    return lambda x: a * x + b

def lagrange_polynomial(axes, y_values):
    """
    Get a collection of corresponding x and y values, and return a polynomial that passes through these dots

    :param axes: A collection of x values
    :param y_values: A collection of corresponding y values
    :return: A polynomial that passes through of all of the dots
    """
    from kiwicalc.expressions.var import Var
    from kiwicalc.expressions.poly import Poly
    x = Var('x')
    result = Poly(0)
    for i, xi in enumerate(axes):
        numerator, denominator = (Poly(1), 1)
        for j, xj in enumerate(axes):
            if xi != xj:
                numerator *= x - xj
                denominator *= xi - xj
        result += numerator / denominator * y_values[i]
    result.simplify()
    return result

def taylor_polynomial(func: 'Union[Function, Poly, Mono]', n: int, a: float, var: str='x'):
    """This feature is under testing and development at the moment."""
    from kiwicalc.expressions.var import Var
    from kiwicalc.expressions.poly import Poly
    mono_expressions = [func(a)]
    current_var = Var(var)
    ith_derivative = func
    for i in range(n):
        ith_derivative = ith_derivative.derivative()
        expression = ith_derivative(a) / factorial(i + 1) * (current_var - a) ** (i + 1)
        mono_expressions.append(expression)
    return Poly(mono_expressions)

def apply_on(func: Callable, collection: Iterable) -> Iterable:
    """Apply a certain given function on a collection of items"""
    if isinstance(collection, (list, set)):
        for index, value in enumerate(collection):
            collection[index] = func(value)
        return collection
    return [func(item) for item in collection]

def float_gcd(a: float, b: float, rtol: float=1e-05, atol: float=1e-08):
    """ finding the greatest common divisor for 2 float numbers"""
    if a == 0 or b == 0:
        return 0
    t = min((abs(a), abs(b)))
    while abs(b) > rtol * t + atol:
        a, b = (b, a % b)
    return round_decimal(a)

def gcd(decimal_numbers: Iterable):
    """
    Finding the greatest common divisor. For example, for the tuple (2.5,3.5,1.5) the result would be 0.5.

    :param decimal_numbers: a list or tuple with decimal numbers
    :return: the greatest common divisor of those numbers
    """
    return reduce(lambda a, b: float_gcd(a, b), decimal_numbers)

def copy_expression(expression):
    if hasattr(expression, "__copy__"):
        return expression.__copy__()
    if isinstance(expression, (list, set)) or hasattr(expression, "copy"):
        return expression.copy()
    return expression
def apply_parenthesis(given_string: str, delimiters=('+', '-', '*', '**')):
    """put parenthesis on expressions such as x+5, 3*x , etc - if needed."""
    if any((character in delimiters for character in given_string)):
        return f'({given_string})'
    return given_string

def handle_parenthesis(expression: str):
    """
    [INSIDE USE][Needs to be fixed and reviewed] This method processes the appearances of parenthesis in expressions.

    :param expression:
    :return:
    """
    new_expression = ''
    for character_index in range(len(expression)):
        if expression[character_index] == '(' and character_index > 0 and (expression[character_index - 1] not in ('+', '-', '*', '/', '%')):
            new_expression += '*'
        new_expression += expression[character_index]
        if expression[character_index] == ')' and character_index < len(expression) - 1 and (expression[character_index + 1] not in ('+', '-', '*', '/', '%', '!')):
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
    from kiwicalc.parsing.parse_expression import split_expression
    if format_abs:
        expression = handle_abs(expression)
    if format_factorial:
        expression = handle_factorial(expression)
    expressions = split_expression(expression.replace('^', '**'))
    modified_variables = list(variables) + list(constants)
    for index, expression in enumerate(expressions):
        new_expression = ''
        occurrences = []
        for variable in modified_variables:
            occurrences += [m.start() for m in re.finditer(variable, expression)]
        for character_index in range(len(expression)):
            " if expression[character_index] == '(' and character_index > 0 and expression[character_index - 1] not in (\n                    '+', '-', '*', '/', '%'):\n                new_expression += '*'\n            "
            new_expression += expression[character_index]
            "if expression[character_index] == ')' and character_index < len(expression) - 1 and expression[\n                character_index + 1] not in ('+', '-', '*', '/', '%', '!'):\n                new_expression += '*'"
            if character_index + 1 in occurrences and (expression[character_index].isdigit() or expression[character_index].isalpha()) and (expression[character_index] + expression[character_index + 1] not in modified_variables):
                new_expression += '*'
        expressions[index] = new_expression
    return ''.join(expressions)

def to_lambda(expression: str, variables, constants=(), format_abs=False, format_factorial=False):
    """
    Generate an executable lambda expression from a string
    """
    from kiwicalc.parsing.parse_expression import split_expression
    modified_expression = formatted_expression(expression, variables, constants, format_abs=format_abs,
                                               format_factorial=format_factorial)
    eval_globals = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
        'asinh': math.asinh, 'acosh': math.acosh, 'atanh': math.atanh,
        'cot': cot, 'sec': sec, 'csc': csc, 'acsc': acsc, 'asec': asec,
        'acot': lambda x: 1 / math.atan(x) if x != 0 else math.pi / 2,
        'ln': ln, 'log': math.log, 'log10': math.log10, 'log2': math.log2, 'exp': math.exp,
        'sqrt': math.sqrt, 'abs': abs, 'factorial': factorial,
        'pi': math.pi, 'e': math.e, 'tau': math.tau if hasattr(math, 'tau') else 2 * math.pi,
        'math': math, 'np': np
    }
    return eval(f'lambda {",".join(variables)}:{modified_expression}', eval_globals)
def derivative(coefficients, get_string=False) -> Union[int, float, list]:
    """ receives the coefficients of a polynomial or a string
     and returns the derivative ( either list, float, or integer) """
    from kiwicalc.parsing.parse_expression import ParseExpression
    if isinstance(coefficients, str):
        coefficients = ParseExpression.to_coefficients(coefficients)
    num_of_coefficients = len(coefficients)
    if num_of_coefficients == 0:
        raise ValueError('At least one coefficient is required')
    elif num_of_coefficients == 1:
        return 0
    elif num_of_coefficients == 2:
        return coefficients[0]
    result = [coefficients[index] * (num_of_coefficients - index - 1) for index in range(num_of_coefficients - 1)]
    if get_string:
        return ParseExpression.coefficients_to_str(result)
    return result

def integral(coefficients, c=0, modify_original=False, get_string=False):
    """ receives the coefficients of a polynomial or a string
     and returns the integral ( either list, float, or integer) """
    from kiwicalc.parsing.parse_expression import ParseExpression
    if isinstance(coefficients, str):
        coefficients = ParseExpression.to_coefficients(coefficients)
    num_of_coefficients = len(coefficients)
    if num_of_coefficients == 0:
        raise ValueError('At least one coefficient is required')
    elif num_of_coefficients == 1:
        return [coefficients[0], c]
    else:
        coefficients = coefficients if modify_original and (not isinstance(coefficients, (tuple, set))) else list(coefficients)
        coefficients.insert(0, coefficients[0] / num_of_coefficients)
        for i in range(1, num_of_coefficients):
            coefficients[i] = coefficients[i + 1] / (num_of_coefficients - i)
        coefficients[-1] = c
        if get_string:
            return ParseExpression.coefficients_to_str(coefficients)
        return coefficients

def round_decimal(number: float):
    """
    Rounds a decimal number, or at least tries to.... Since python is weird with it
    :param number: ugly number we wish to round
    :return: less ugly number
    """
    if number - floor(number) < 1e-06:
        return floor(number)
    elif abs(number - ceil(number)) < 1e-06:
        return ceil(number)
    return round(number, 5)

def create_grid():
    """ Create a grid in matplotlib"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('equal')
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    return (fig, ax)

def draw_axis(ax):
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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
    return ''.join([character for character in equation if character != ' '])

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

def only_numbers_letters(given_string: str):
    """
    checks whether a string contains only letters and numbers.
    :param given_string:
    :return:
    """
    if given_string == '' or given_string is None:
        return False
    char_array = list(given_string)
    if char_array[0] == '-':
        del char_array[0]
    return bool([char for char in char_array if char.isalpha() or char.isdigit()])

def _format_minus(expression1, expression2):
    """
    Internal method. Not for outside use !!!
    For formatting strings in the format (x-a)^2
    """
    expression1_str, expression2_str = (expression1.__str__(), expression2.__str__())
    if '+' in expression1_str or '-' in expression1_str:
        expression1_str = f'({expression1})'
    else:
        expression1_str = f'{expression1}'
    if '-' in expression2_str or '+' in expression2_str:
        expression2_str = f'({expression2})'
    else:
        expression2_str = f'{expression2}'
    if expression2 == 0:
        if expression1 == 0:
            return '0'
        return f'{expression1_str}^2'
    elif expression1 == 0:
        return f'{expression2_str}^2'
    return f'({expression1_str}-{expression2_str})^2'

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

def sorted_expressions(expressions: 'Iterable[Union[Poly,Mono]]'):
    assert all((expression.variables_dict is not None for expression in expressions)), 'This method cannot accept free numbers'
    return sorted(expressions, key=lambda item: max(item.variables_dict.values()), reverse=True)

def process_object(expression: Union[IExpression, int, float], class_name: str, method_name: str, param_name: str):
    from kiwicalc.expressions.mono import Mono
    if isinstance(expression, (int, float)):
        return Mono(expression)
    elif isinstance(expression, IExpression):
        return expression.__copy__()
    raise TypeError(f"Invalid type '{type(expression)}' of paramater '{param_name}' in method {method_name} in class {class_name}")

def handle_abs(expression: str):
    """
    An attempt to handle absolute values and evaluate them as necessary.
    :param expression: the expression to be processed, of type str.
    :return:
    """
    copy = expression.replace('|', '~~')
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
    copy1 = expression.replace(' ', '')
    results = [res for res in re.findall(f'([a-zA-Z0-9]+!|[a-zA-Z0-9]*\\([^!]+\\)!)', copy1)]
    for result in results:
        if result.startswith('(') and result.endswith(')'):
            result = result[1:-1]
        new = f'factorial({result[:-1]})'
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

def handle_trigo_calculation(expression: str):
    """ getting the result of a single trigonometric operation, e.g : sin(90) -> 1"""
    selected_operation = [op for op in TRIGONOMETRY_CONSTANTS.keys() if op in expression]
    selected_operation = selected_operation[0]
    start_index = expression.find(selected_operation) + len(selected_operation) + 1
    coef = expression[:expression.find(selected_operation)]
    if coef == '' or coef is None or coef == '+':
        coef = 1
    elif coef == '-':
        coef = 1
    else:
        coef = float(coef)
    parameter = expression[start_index] if expression[start_index].isdigit() or expression[start_index] == '-' else ''
    for i in range(start_index + 1, expression.rfind(')')):
        parameter += expression[i]
    if is_evaluatable(parameter):
        parameter = float(eval(parameter))
    parameter = -float(parameter) if expression[0] == '-' else float(parameter)
    return round_decimal(coef * TRIGONOMETRY_CONSTANTS[selected_operation](parameter))

def handle_trigo_expression(expression: str):
    """ handles a whole trigonometric expression, for example: 2sin(90)+3sin(60)"""
    from kiwicalc.parsing.parse_expression import split_expression
    expressions = split_expression(expression)
    result = 0
    for expr in expressions:
        result += handle_trigo_calculation(expr)
    return result

def lambda_from_recursive(recursive_function: str):
    elements = set(ptn.findall(recursive_function))
    elements = sorted(elements, key=lambda element: 0 if '{' not in element else float(element[element.find('n') + 1:element.find('}')]))
    indices = [element[element.find('{') + 1:element.find('}')] if '{' in element else 'n' for element in elements]
    new_elements = [element.replace('{', '').replace('}', '').replace('+', 'p').replace('-', 'd').replace('n', 'k') for element in elements]
    recursive_function = recursive_function[recursive_function.find('=') + 1:]
    for element, new_element in zip(elements, new_elements):
        recursive_function = recursive_function.replace(element, new_element)
    del new_elements[-1]
    new_elements.append('n')
    lambda_expression = to_lambda(recursive_function, new_elements, list(TRIGONOMETRY_CONSTANTS.keys()) + list(MATHEMATICAL_CONSTANTS.keys()))
    del indices[-1]
    return (lambda_expression, indices)

def format_matplot_polynomial(expression: str):
    """
    formats a polynomial expression into matplot's format
    :param expression:
    :return:
    """
    from kiwicalc.parsing.parse_expression import split_expression
    expressions = split_expression(expression)
    for index, expr in enumerate(expressions):
        expr = expr.replace('**', '^')
        accumulator = ''
        skip = 0
        for i in range(len(expr)):
            character = expr[i]
            if skip == 0:
                if character == '^':
                    accumulator = ''.join((accumulator, '^{'))
                    j = i + 1
                    while j < len(expr) and expr[j] not in ('^', '*', '+', '-'):
                        accumulator += expr[j]
                        j += 1
                    accumulator = ''.join((accumulator, '}'))
                    skip = j - i - 1
                else:
                    accumulator += character
            else:
                skip -= 1
        expressions[index] = f'{accumulator}'
    return f'{''.join(expressions)}'

def format_matplot_function(expression: str):
    raise NotImplementedError

def format_matplot(expression: str):
    return format_matplot_polynomial(expression)

def format_linear_dict(algebraic_dict: dict, round_coefficients: bool=True) -> str:
    """ Receives a dictionary that represents a linear expression and creates a new string from it.
        For instance, the dictionary {'x':2, 'y':-4, 'number':5} represents the expression 2x - 4y + 5
     """
    if not algebraic_dict:
        return ''
    accumulator = ''
    for key, value in algebraic_dict.items():
        if value != 0:
            value = round_decimal(value) if round_coefficients else value
            coef = f'+{value}' if value > 0 else f'{value}'
            if key == 'number':
                accumulator += coef
            elif value == 1:
                accumulator += f'+{key}'
            elif value == -1:
                accumulator += f'-{key}'
            else:
                accumulator += f'{coef}{key}'
    return accumulator[1:] if accumulator[0] == '+' else accumulator

def format_poly_dict(algebraic_dict: dict):
    """
    Internal method: Receives a dictionary that represents a polynomial, in the Equation's class format, and turns it into string.
    *** This method might become redundant / deleted / updated.
    :param algebraic_dict: the dictionary of the expression, for example : {'x**2':3,'x':2,'number':-1} -> 3x^2+2x-1
    :return:
    """
    accumulator = ''
    for expression, coefficient in algebraic_dict.items():
        if expression == 'number':
            if coefficient >= 0:
                accumulator += f'+{round_decimal(coefficient)}'
            else:
                accumulator += f'{round_decimal(coefficient)}'
        elif coefficient != 0:
            power = float(expression[expression.find('**') + 2:])
            if coefficient == int(coefficient):
                coefficient = int(coefficient)
            if power == int(power):
                power = int(power)
            if power == 1:
                accumulator += f'{coefficient}{expression[:expression.find('**')]}'
            elif power == 0:
                algebraic_dict['number'] += 1
            else:
                accumulator += f' {coefficient}{expression} +'
    return accumulator.replace('++', '+').replace('+-', '-').replace('--', '-')

def get_image(path, width=1 * cm):
    """Utility method for building images for PDF files. Only for internal use."""
    img = utils.ImageReader(path)
    iw, ih = img.getSize()
    aspect = ih / float(iw)
    return Image(path, width=width, height=width * aspect)

def max_power(expressions):
    return max(expressions, key=lambda expression: max(expression.variables_dict.values()))
