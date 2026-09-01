from __future__ import annotations
import math
from math import log, e
import re
import string
import warnings
import numpy as np
from itertools import combinations
from typing import Union, Tuple, List, Optional, Any, Callable, Iterator, Set, Dict, Iterable

from kiwicalc.core.constants import allowed_characters, number_pattern
from kiwicalc.core.operators import _TrigoMethodFromString, TrigoMethods
from kiwicalc.core.utils import (
    clean_from_spaces, extract_coefficient, format_coefficient,
    format_free_number, is_number, contains_from_list, round_decimal,
    handle_abs
)

def split_expression(expression: str):
    """splits the expression by delimiters, but doesn't touch what's inside parenthesis """
    delimiters = []
    for index, char in enumerate(expression):
        if char in ('+', '-') and index > 0:
            parenthesis_index, curly_index = (expression[:index].rfind('('), expression[:index].rfind('{'))
            closing_paranthesis_index = expression[parenthesis_index:].find(')') + parenthesis_index
            closing_curly = expression[curly_index:].find('}') + curly_index
            square_index = expression[:index].rfind('[')
            close_square = expression[curly_index:].find(']') + square_index
            if not parenthesis_index < index < closing_paranthesis_index and (not curly_index < index < closing_curly) and (not square_index < index < close_square):
                delimiters.append(index)
    expressions = []
    if len(delimiters) > 0:
        expressions.append(expression[:delimiters[0]])
        for i in range(1, len(delimiters)):
            expressions.append(expression[delimiters[i - 1]:delimiters[i]])
        expressions.append(expression[delimiters[len(delimiters) - 1]:])
    else:
        expressions.append(expression)
    return [expression for expression in expressions if expression != '']

def fetch_variable(variables: dict):
    """ Brings the first variable in a dictionary of variables_dict and their values """
    try:
        return f'{next(iter(variables))}'
    except (IndexError, StopIteration):
        return None

def fetch_power(variables: dict):
    return variables[next(iter(variables))]

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
        return (extract_coefficient(single_expression), None)
    variable_place = single_expression.find(variable_name)
    coefficient = extract_coefficient(single_expression[:variable_place])
    power_index = single_expression.rfind('^')
    power = 1 if power_index == -1 else float(single_expression[power_index + 1:])
    return (coefficient, {variable_name: power})

def extract_variables_from_expression(expression: str):
    return {character for character in expression if character.isalpha()}

def mono_from_str(mono_expression: str, get_tuple=False):
    """
    Analyzes a string, such as "3x^2*y^2" and creates a monomial expression ( of type Mono )
    :param mono_expression: the string that represents the monomial
    :param get_tuple: if set to True, instead of a Mono object, the _coefficient(float) and __variables(dict)
    will be returned.
    :return: The monomial, or if get_tuple=True, then its _coefficient and __variables.
    :rtype: Mono or tuple
    """
    from kiwicalc.expressions.mono import Mono
    try:
        mono_expression = clean_from_spaces(mono_expression)
        number = float(mono_expression)
        if get_tuple:
            return (number, None)
        return Mono(number)
    except (ValueError, TypeError):
        mono_expression: str = mono_expression.strip().replace('**', '^')
        for variable in (character for character in mono_expression if character in allowed_characters):
            occurrences = [m.start() for m in re.finditer(variable, mono_expression)]
        new_expression: str = ''
        for character_index in range(len(mono_expression)):
            new_expression = ''.join((new_expression, mono_expression[character_index]))
            if character_index + 1 in occurrences and (mono_expression[character_index].isdigit() or mono_expression[character_index].isalpha()):
                new_expression += '*'
        basic_expressions: list = new_expression.split('*')
        final_coefficient, variables_and_powers = (1, dict())
        for basic_expression in basic_expressions:
            variable: str = ''.join([character for character in basic_expression if character in allowed_characters])
            current_coefficient, dictionary_item = __data_from_single(basic_expression, variable)
            final_coefficient *= current_coefficient
            if dictionary_item is not None:
                variables_and_powers = {**variables_and_powers, **dictionary_item}
        if get_tuple:
            return (final_coefficient, variables_and_powers)
        return Mono(coefficient=final_coefficient, variables_dict=variables_and_powers)

def poly_from_str(poly_expression: str, get_list=False) -> 'Union[Poly,List]':
    """
    Analyzes a string, such as "3x^2 + 2xy - 7" and generates a polynomial expression
    :param poly_expression:
    :param get_list: if set to True, a list of the monomials ( Mono objects ) will be returned instead
    of a Poly object
    :return: a polynomial corresponding to the string, or a list of monomials.
    :rtype: Poly or list
    """
    from kiwicalc.expressions.mono import Mono
    from kiwicalc.expressions.poly import Poly
    poly_expression = clean_from_spaces(poly_expression)
    expressions = (mono_expression for mono_expression in poly_expression.replace('-', '+-').split('+') if mono_expression != '')
    expressions = [mono_from_str(expression) for expression in expressions]
    if get_list:
        return expressions
    return Poly(expressions)

def monic_poly_from_coefficients(coefficients, var_name='x') -> 'Poly':
    from kiwicalc.expressions.mono import Mono
    from kiwicalc.expressions.poly import Poly
    length = len(coefficients)
    return Poly([Mono(coefficient=coef, variables_dict={var_name: length - 1 - index}) for index, coef in enumerate(coefficients)])

def poly_frac_from_str(expression: str, get_tuple=False):
    """
    Generates a PolyFraction object from a given string

    :param expression: The given string that represents a polynomial fraction
    :param get_tuple : If set to True, the a tuple of length 2 with the numerator at index 0 and the denoominator at index 1 will be returned.
    :return: Returns a new PolyFraction object, unless get_tuple is True, and then returns the corresponding tuple.
    """
    from kiwicalc.expressions.poly import Poly
    from kiwicalc.expressions.fractions import PolyFraction
    first_expression, second_expression = expression.split('/')
    if get_tuple:
        return (Poly(first_expression), Poly(second_expression))
    return PolyFraction(Poly(first_expression), Poly(second_expression))

def coefficient_to_float(coefficient: str) -> Optional[float]:
    return float(coefficient)

def __helper_trigo(expression: str) -> Optional[Tuple[int, Optional[float]]]:
    try:
        first_letter_index = expression.find(next((character for character in expression if character.isalpha() and character not in ('e', 'i'))))
        return (first_letter_index, coefficient_to_float(str(extract_coefficient(expression[:first_letter_index]))))
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
    from kiwicalc.expressions.trigonometry import TrigoExpr
    from kiwicalc.expressions.factory import create
    trigo_expression = trigo_expression.strip().replace('**', '^').replace(' ', '')
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
        return (coefficient, method_chosen, inside, power)
    return TrigoExpr(coefficient, [(method_chosen, inside, power)])

def TrigoExpr_from_str(trigo_expression: str, get_tuple=False, dtype='poly') -> 'Union[Tuple[IExpression,List[list]],TrigoExpr]':
    """

    :param trigo_expression:
    :param get_tuple:
    :return:
    """
    from kiwicalc.expressions.poly import Poly
    from kiwicalc.expressions.trigonometry import TrigoExpr
    trigo_expression = trigo_expression.strip().replace('**', '^')
    coefficient = Poly(1)
    expressions = [expression for expression in trigo_expression.split('*') if expression.strip() != '']
    new_expressions = []
    for expression in expressions:
        if is_number(expression):
            coefficient *= float(expression)
        else:
            new_expressions.append(expression)
    analyzed_generator = (analyze_single_trigo(expression, get_tuple=True, dtype=dtype) for expression in new_expressions)
    analyzed_expressions = []
    for coef, method_chosen, inside, power in analyzed_generator:
        analyzed_expressions.append([method_chosen, inside, power])
        coefficient *= coef
    if not analyzed_expressions:
        analyzed_expressions = None
    if get_tuple:
        return (coefficient, analyzed_expressions)
    return TrigoExpr(coefficient, expressions=analyzed_expressions)

def TrigoExprs_from_str(trigo_expression: str, get_list=False):
    """

    :param trigo_expression:
    :param get_tuple:
    :return:
    """
    from kiwicalc.expressions.trigonometry import TrigoExprs
    trigo_expressions: list = split_expression(trigo_expression)
    new_expressions: list = [TrigoExpr_from_str(expression) for expression in trigo_expressions]
    if get_list:
        return new_expressions
    return TrigoExprs(new_expressions)

def log_from_str(expression: str, get_tuple=False, dtype: str='poly'):
    from kiwicalc.expressions.log import Log, PolyLog
    from kiwicalc.expressions.factory import create
    expression = expression.strip().lower()
    if 'log' in expression or 'ln' in expression:
        coefficient = expression[:expression.find('l')]
        if coefficient == '':
            coefficient = 1
        elif coefficient == '-':
            coefficient = -1
        else:
            try:
                coefficient = float(coefficient)
            except ValueError:
                raise ValueError(f"Invalid _coefficient '{coefficient}' in expression {expression}, while creatinga PolyLog object from a given string.")
        start_parenthesis = expression.find('(')
        if start_parenthesis == -1:
            raise ValueError(f"Invalid string '{expression}' without opening parenthesis for the expression.")
        ending_parenthesis = expression.find(')')
        if ending_parenthesis == -1:
            raise ValueError(f"Invalid string: '{ending_parenthesis} without ending parenthesis for the expression'")
        if 'log' in expression:
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
        power_index = expression.find('^')
        if power_index == -1:
            power_index = expression.find('**')
        if power_index == -1:
            power = 1
        else:
            close_parenthesis_index = expression.rfind(')')
            if power_index > close_parenthesis_index:
                power = float(expression[power_index + 1:])
            else:
                power = 1
        if get_tuple:
            return (coefficient, inside, base, power)
        return Log(expression=[[inside, base, power]], coefficient=coefficient)
    else:
        raise ValueError('The string need to contain log() or ln()')

def surface_from_str(input_string: str, get_coefficients=False):
    from kiwicalc.linalg.spaces import Surface
    first_side, second_side = input_string.split('=', 1)
    variables = ('x', 'y', 'z')
    first = ParseExpression.parse_linear(first_side, variables)
    second = ParseExpression.parse_linear(second_side, variables)
    coefficients = [first[variable] - second[variable] for variable in variables]
    coefficients.append(first['free'] - second['free'])
    if get_coefficients:
        return coefficients
    return Surface(coefficients)

class ParseExpression:

    @staticmethod
    def parse_linear(expression, variables):
        parsed = ParseExpression.parse_polynomial(expression, variables=variables)
        result = {'free': parsed['free']}
        for variable in variables:
            coefficients = parsed[variable]
            if len(coefficients) > 1:
                raise ValueError(f"Expected a linear expression, but found a power greater than 1 for '{variable}'")
            result[variable] = coefficients[0] if coefficients else 0
        return {variable: result[variable] for variable in variables} | {'free': result['free']}

    @staticmethod
    def unparse_linear(variables_dict: dict, free_number: float=None):
        accumulator = []
        for variable, coefficients in variables_dict.items():
            if variable == 'free':
                continue
            if isinstance(coefficients, (int, float)):
                coefficients = (coefficients,)
            for coefficient in coefficients:
                if coefficient != 0:
                    coefficient_str = format_coefficient(coefficient)
                    sign = '+' if coefficient > 0 else ''
                    accumulator.append(f'{sign}{coefficient_str}{variable}')
        if free_number is None:
            free_number = variables_dict.get('free', 0)
        accumulator.append(format_free_number(free_number))
        result = ''.join(accumulator)
        if not result:
            return '0'
        if result[0] == '+':
            return result[1:]
        return result

    @staticmethod
    def parse_quadratic(expression: str, variables=None, strict_syntax=True):
        expression = expression.replace(' ', '').replace('**', '^')
        if variables is None:
            variables = get_equation_variables(expression)
        if strict_syntax:
            if len(variables) != 1:
                raise ValueError(f'Strict quadratic syntax must contain exactly 1 variable, found {len(variables)}')
            variable = variables[0]
            parsed = ParseExpression.parse_polynomial(expression, variables, strict_syntax=False)
            if len(parsed[variable]) != 2:
                raise ValueError(f"Didn't find a quadratic term containing '{variable}^2'")
            return parsed
        return ParseExpression.parse_polynomial(expression, variables, strict_syntax=False)

    @staticmethod
    def parse_cubic(expression: str, variables, strict_syntax=True):
        expression = expression.replace(' ', '').replace('**', '^')
        if strict_syntax:
            if len(variables) != 1:
                raise ValueError(f'Strict cubic syntax must contain exactly 1 variable, found {len(variables)}')
            variable = variables[0]
            parsed = ParseExpression.parse_polynomial(expression, variables, strict_syntax=False)
            if len(parsed[variable]) != 3:
                raise ValueError(f"Didn't find a cubic term containing '{variable}^3'")
            return parsed
        return ParseExpression.parse_polynomial(expression, variables, strict_syntax=False)

    @staticmethod
    def parse_quartic(expression: str, variables, strict_syntax=True):
        expression = expression.replace(' ', '').replace('**', '^')
        if strict_syntax:
            if len(variables) != 1:
                raise ValueError(f'Strict quartic syntax must contain exactly 1 variable, found {len(variables)}')
            variable = variables[0]
            parsed = ParseExpression.parse_polynomial(expression, variables, strict_syntax=False)
            if len(parsed[variable]) != 4:
                raise ValueError(f"Didn't find a quartic term containing '{variable}^4'")
            return parsed
        return ParseExpression.parse_polynomial(expression, variables, strict_syntax=False)

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
        if numpy_array and len(variables) == 1:
            result = np.append(variables_dict[variables[0]], variables_dict['free'])
            if not get_variables:
                return result
            return (result, variables)
        if not get_variables:
            return variables_dict
        return (variables_dict, variables)

    @staticmethod
    def unparse_polynomial(parsed_dict: dict, syntax=''):
        """Taking a parsed polynomial and returning a string from it"""
        accumulator = []
        if syntax not in ('', 'pythonic'):
            warnings.warn(f"Unrecognized syntax: {syntax}. Either use the default or 'pythonic' ")
        for variable, coefficients in parsed_dict.items():
            if variable == 'free':
                continue
            sub_accumulator, num_of_coefficients = ([], len(coefficients))
            for index, coefficient in enumerate(coefficients):
                if coefficient != 0:
                    coefficient_str = format_coefficient(round_decimal(coefficient))
                    if coefficient_str not in ('', '-') and syntax == 'pythonic':
                        coefficient_str += '*'
                    power = len(coefficients) - index
                    sign = '' if coefficient < 0 or (not accumulator and (not sub_accumulator)) else '+'
                    if power == 1:
                        sub_accumulator.append(f'{sign}{coefficient_str}{variable}')
                    elif syntax == 'pythonic':
                        sub_accumulator.append(f'{sign}{coefficient_str}{variable}**{power}')
                    else:
                        sub_accumulator.append(f'{sign}{coefficient_str}{variable}^{power}')
            accumulator.extend(sub_accumulator)
        free_number = parsed_dict['free']
        if free_number != 0 or not accumulator:
            sign = '' if free_number < 0 or not accumulator else '+'
            accumulator.append(f'{sign}{round_decimal(free_number)}')
        return ''.join(accumulator)

    @staticmethod
    def _parse_monomial(expression: str, variables):
        """ Extracting the coefficient an power from a monomial, this method is used while parsing polynomials"""
        variable_index = -1
        for suspect_variable in variables:
            suspect_variable_index = expression.find(suspect_variable)
            if suspect_variable_index != -1:
                variable_index = suspect_variable_index
                break
        if variable_index == -1:
            try:
                return (float(expression), 'free', 0)
            except ValueError:
                raise ValueError("Couldn't parse the expression! Found no variables, but the free number isn't valid.")
        else:
            variable = expression[variable_index]
            try:
                coefficient = extract_coefficient(expression[:variable_index])
            except ValueError:
                raise ValueError(f"Encountered an invalid coefficient '{expression[:variable_index]}' whileparsing the monomial '{expression}'")
            power_index = expression.find('^')
            if power_index == -1:
                return (coefficient, variable, 1)
            try:
                power = float(expression[power_index + 1:])
                return (coefficient, variable, power)
            except ValueError:
                raise ValueError(f"encountered an invalid power '{expression[power_index + 1:]} while parsing themonomial '{expression}'")

    @staticmethod
    def to_coefficients(expression: str, variable=None, strict_syntax=True, get_variable=False):
        expression = clean_from_spaces(expression)
        if variable is None:
            variables = sorted({character for character in expression if character.isalpha()})
            num_of_variables = len(variables)
            if num_of_variables == 0:
                return [float(expression)]
            elif num_of_variables != 1:
                raise ValueError(f'Can only parse polynomials with 1 variable, but got {num_of_variables}')
            variable = variables[0]
        parsed = ParseExpression.parse_polynomial(expression, variables=(variable,), strict_syntax=strict_syntax)
        coefficients_list = list(parsed[variable]) + [parsed['free']]
        if not get_variable:
            return coefficients_list
        return (coefficients_list, variable)

    @staticmethod
    def coefficients_to_str(coefficients, variable='x', syntax=''):
        """Taking a parsed polynomial and returning a string from it"""
        accumulator = []
        if syntax not in ('', 'pythonic'):
            warnings.warn(f"Unrecognized syntax: {syntax}. Either use the default or 'pythonic' ")
        num_of_coefficients = len(coefficients)
        if num_of_coefficients == 0:
            raise ValueError('At least 1 coefficient is required')
        elif num_of_coefficients == 1:
            return f'{coefficients[0]}'
        for index in range(num_of_coefficients - 1):
            coefficient = coefficients[index]
            if coefficient != 0:
                coefficient_str = format_coefficient(round_decimal(coefficient))
                if coefficient_str not in ('', '-') and syntax == 'pythonic':
                    coefficient_str += '*'
                power = len(coefficients) - index - 1
                sign = '' if coefficient < 0 or not accumulator else '+'
                if power == 1:
                    accumulator.append(f'{sign}{coefficient_str}{variable}')
                elif syntax == 'pythonic':
                    accumulator.append(f'{sign}{coefficient_str}{variable}**{power}')
                else:
                    accumulator.append(f'{sign}{coefficient_str}{variable}^{power}')
        free_number = coefficients[-1]
        if free_number != 0 or not accumulator:
            sign = '' if free_number < 0 or not accumulator else '+'
            accumulator.append(f'{sign}{round_decimal(free_number)}')
        return ''.join(accumulator)
