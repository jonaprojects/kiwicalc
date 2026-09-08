from __future__ import annotations
import re
import string
import warnings
from typing import Union, Tuple, List, Optional, Any, Callable, Iterator, Set, Dict, Iterable

from kiwicalc.core.constants import allowed_characters
from kiwicalc.core.utils import (
    clean_from_spaces, extract_coefficient, format_coefficient,
    format_free_number, is_number, contains_from_list, round_decimal,
    handle_abs, handle_factorial
)
from kiwicalc.parsing.parse_expression import (
    split_expression, ParseExpression, extract_variables_from_expression,
    poly_from_str
)
from kiwicalc.parsing.errors import EquationParseError


def _split_equation(equation: str, delimiter: str='=') -> Tuple[str, str]:
    """Validate and split a two-sided equation without silently truncating it."""
    if not isinstance(equation, str):
        raise TypeError('equation must be a string')
    if not isinstance(delimiter, str) or not delimiter:
        raise EquationParseError('delimiter must be a non-empty string')
    if equation.count(delimiter) != 1:
        raise EquationParseError(
            f"An equation must have two sides separated by exactly one '{delimiter}'"
        )
    first_side, second_side = (side.strip() for side in equation.split(delimiter))
    if not first_side or not second_side:
        raise EquationParseError('Both sides of an equation must be non-empty')
    return first_side, second_side

def extract_dict_from_equation(equation: str, delimiter='='):
    """
    This method should accept an equation, and extract the variable from it. It is still quite basic..

    :param equation: the equation, of type string
    :param delimiter: separator
    :return: returns a dictionary of the __variables and the number. for example, for the equation 3x-y+8 = 6+y+x the
    dictionary returned would be {'x':0,'y':0,'number':0}
    """
    first_side, second_side = _split_equation(equation, delimiter)
    variables = sorted(
        extract_variables_from_expression(first_side)
        | extract_variables_from_expression(second_side)
    )
    return {**{variable: 0 for variable in variables}, 'number': 0}

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
    while first_coefficients and first_coefficients[0] == 0:
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
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        raise TypeError('subtract_dicts expects two dictionaries')
    ordered_keys = list(dict1) + [key for key in dict2 if key not in dict1]
    for key in ordered_keys:
        if key not in dict1:
            warnings.warn(f"variable {key} wasn't found in the first data structure")
        if key not in dict2:
            warnings.warn(f"variable {key} wasn't found in the second data structure")
    return {key: dict1.get(key, 0) - dict2.get(key, 0) for key in ordered_keys}

def linear_expression_to_dict(expression: str, variables: Iterable) -> dict:
    """alternative way to """
    parsed = ParseExpression.parse_linear(expression, variables)
    return {
        **{variable: parsed[variable] for variable in variables},
        'number': parsed['free'],
    }

def equation_to_one_side(equation: str) -> str:
    """ Move all of the items of the equation to one side"""
    first_side, second_side = _split_equation(equation)
    first_side = clean_from_spaces(first_side)
    second_side = clean_from_spaces(second_side)
    negated_terms = []
    for term in split_expression(second_side):
        if term.startswith('+'):
            negated_terms.append('-' + term[1:])
        elif term.startswith('-'):
            negated_terms.append('+' + term[1:])
        else:
            negated_terms.append('-' + term)
    return first_side + ''.join(negated_terms)

def get_equation_variables(equation: str) -> List[Optional[str]]:
    _split_equation(equation)
    return sorted(extract_variables_from_expression(equation))

def simplify_expression(expression: str, variables: Iterable[str], format_abs=False, format_factorial=False) -> dict:
    if format_abs:
        expression = handle_abs(expression)
    if format_factorial:
        expression = handle_factorial(expression)
    if isinstance(variables, dict):
        new_dict = variables.copy()
    else:
        new_dict = {variable_name: 0 for variable_name in variables}
    if 'number' not in new_dict:
        new_dict['number'] = 0
    variable_names = [key for key in new_dict if key != 'number']
    try:
        parsed = ParseExpression.parse_linear(expression, variable_names)
    except ValueError as error:
        if 'Expected a linear expression' in str(error):
            raise
        raise ValueError(f'Unrecognized expression {expression}') from error
    for variable in variable_names:
        new_dict[variable] += parsed[variable]
    new_dict['number'] += parsed['free']
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
        first_side, second_side = _split_equation(equation)
        variables = get_equation_variables(equation)
        if len(variables) != 1:
            raise ValueError('Can only parse polynomial equations with 1 variable')
        variable = variables[0]
        first_dict = ParseExpression.parse_polynomial(first_side, variables=variables)
        second_dict = ParseExpression.parse_polynomial(second_side, variables=variables)
        add_or_sub_coefficients(first_dict[variable], second_dict[variable], copy_first=False, mode='sub')
        return first_dict[variable] + [first_dict['free'] - second_dict['free']]

    @staticmethod
    def parse_quadratic(equation: str, strict_syntax=False):
        if strict_syntax:
            coefficients = ParseEquation.parse_polynomial(equation)
            if len(coefficients) != 3:
                raise ValueError('Strict quadratic syntax requires a degree-2 equation')
            return coefficients
        return ParseEquation.parse_polynomial(equation)
