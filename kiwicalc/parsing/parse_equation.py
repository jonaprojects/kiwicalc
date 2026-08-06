from __future__ import annotations
import re
import string
import warnings
from typing import Union, Tuple, List, Optional, Any, Callable, Iterator, Set, Dict, Iterable

from kiwicalc.core.constants import allowed_characters
from kiwicalc.core.utils import (
    clean_from_spaces, extract_coefficient, format_coefficient,
    format_free_number, is_number, contains_from_list, round_decimal,
    handle_abs
)
from kiwicalc.parsing.parse_expression import (
    split_expression, ParseExpression, extract_variables_from_expression,
    poly_from_str
)

def extract_dict_from_equation(equation: str, delimiter='='):
    """
    This method should accept an equation, and extract the variable from it. It is still quite basic..

    :param equation: the equation, of type string
    :param delimiter: separator
    :return: returns a dictionary of the __variables and the number. for example, for the equation 3x-y+8 = 6+y+x the
    dictionary returned would be {'x':0,'y':0,'number':0}
    """
    variables = dict()
    first_side, second_side = equation.split(delimiter)
    accumulator = ''
    for expression in split_expression(first_side) + split_expression(second_side):
        start_index = -1
        for index, character in enumerate(expression):
            if character.isalpha():
                start_index = index
                break
        if start_index != -1:
            accumulator = ''
            for character in expression[start_index:]:
                accumulator += character
        variables[accumulator.strip()] = 0
    variables['number'] = 0
    return {key: value for key, value in variables.items() if key != ''}

def add_or_sub_coefficients(first_coefficients, second_coefficients, mode='add', copy_first=True):
    first_coefficients = list(first_coefficients) if copy_first else first_coefficients
    second_coefficients = list(second_coefficients)
    my_variables_length = len(first_coefficients)
    other_variables_length = len(second_coefficients)
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
    while first_coefficients[0] == 0:
        del first_coefficients[0]
    return first_coefficients

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

def linear_expression_to_dict(expression: str, variables: Iterable) -> dict:
    """alternative way to """
    expression = clean_from_spaces(expression)
    my_dict = dict()
    if expression[-1] != ' ':
        expression += ' '
    matches = list(re.finditer(f'([-+]?\\d+[.,]?\\d*)?\\*?([a-zA-Z]+)', expression))
    for variable in variables:
        my_dict[variable] = sum((extract_coefficient(match.group(1)) for match in matches if match.group(2) == variable))
    matches = re.finditer(f'([-+]?\\d+[.,]?\\d*)[-+\\s]', expression)
    numbers_sum = sum((extract_coefficient(match.group(1)) for match in matches))
    my_dict['number'] = numbers_sum
    return my_dict

def equation_to_one_side(equation: str) -> str:
    """ Move all of the items of the equation to one side"""
    equal_sign = equation.find('=')
    if equal_sign == -1:
        raise ValueError("Invalid equation - an equation must have two sides, separated by '=' ")
    first_side, second_side = (equation[:equal_sign], equation[equal_sign + 1:])
    second_side = ''.join(('+' if character == '-' else '-' if character == '+' else character for character in second_side))
    second_side = f'-{second_side}' if second_side[0] not in ('+', '-') else second_side
    if second_side[0] in ('+', '-'):
        return first_side + second_side
    return first_side + second_side

def get_equation_variables(equation: str) -> List[Optional[str]]:
    return list({character for character in equation if character.isalpha()})

def simplify_expression(expression: str, variables: Iterable[str], format_abs=False, format_factorial=False) -> dict:
    if format_abs:
        expression = handle_abs(expression)
    if format_factorial:
        expression = handle_abs(expression)
    expr = expression.replace('-', '+-').replace(' ', '')
    expressions = [num for num in expr.split('+') if num != '' and num is not None]
    if isinstance(variables, dict):
        new_dict = variables.copy()
    else:
        new_dict = {variable_name: 0 for variable_name in variables}
    if 'number' not in new_dict:
        new_dict['number'] = 0
    for item in expressions:
        if item[-1].isalpha() or contains_from_list(allowed_characters, item):
            if item[-1] in new_dict.keys():
                if len(item) == 1:
                    item = f'1{item}'
                elif len(item) == 2 and item[0] == '-':
                    item = f'-1{item[-1]}'
                new_dict[item[-1]] += float(item[:-1])
            elif not is_number(item):
                raise ValueError(f'Unrecognized expression {item}')
        else:
            new_dict['number'] += float(item)
    return new_dict

def coefficients_to_expressions(coefficients, variable: str='x'):
    """
    Getting a list of coefficients and the name of the variable, and returns a list of polynomial expressions,
    namely, a list of Mono objects.
    :param coefficients: the coefficients, for example : [ 1,0,2,3] ( the output expression would be x^3+2x+3 for x )
    :param variable: the name of the variable, the default is "x"
    :return: returns a list of polynomials with the corresponding coefficients and powers.
    """
    from kiwicalc.expressions.mono import Mono
    return [Mono(coefficient=coef, variables_dict={variable: len(coefficients) - 1 - index}) for index, coef in enumerate(coefficients) if coef != 0]

class ParseEquation:

    @staticmethod
    def parse_polynomial(equation: str):
        variables = get_equation_variables(equation)
        if len(variables) != 1:
            raise ValueError('can only parse quadratic equations with 1 variable')
        variable = variables[0]
        first_side, second_side = equation.split('=')
        first_dict = ParseExpression.parse_polynomial(first_side, variables=variables)
        second_dict = ParseExpression.parse_polynomial(second_side, variables=variables)
        add_or_sub_coefficients(first_dict[variable], second_dict[variable], copy_first=False, mode='sub')
        return first_dict[variable] + [first_dict['free'] - second_dict['free']]

    @staticmethod
    def parse_quadratic(equation: str, strict_syntax=False):
        if strict_syntax:
            return ParseExpression.parse_quadratic(equation, strict_syntax=True)
        return ParseEquation.parse_polynomial(equation)
