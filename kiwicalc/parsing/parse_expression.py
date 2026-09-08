from __future__ import annotations
import math
from math import log, e
import re
import string
import warnings
import numpy as np
from dataclasses import dataclass
from itertools import combinations
from typing import Union, Tuple, List, Optional, Any, Callable, Iterator, Set, Dict, Iterable

from kiwicalc.core.constants import allowed_characters, number_pattern
from kiwicalc.core.operators import _TrigoMethodFromString, TrigoMethods
from kiwicalc.core.utils import (
    clean_from_spaces, extract_coefficient, format_coefficient,
    format_free_number, is_number, contains_from_list, round_decimal,
    handle_abs
)
from kiwicalc.parsing.errors import (
    EquationParseError, UnsupportedExpressionError, AmbiguousVariableError,
)


@dataclass(frozen=True)
class _PolynomialToken:
    kind: str
    value: str
    position: int


def _tokenize_polynomial(expression: str, variables=None):
    """Tokenize the supported polynomial grammar without evaluating text."""
    if not isinstance(expression, str):
        raise TypeError('expression must be a string')
    explicit_variables = variables is not None
    variables = (
        list(variables) if explicit_variables
        else sorted(extract_variables_from_expression(expression))
    )
    if len(set(variables)) != len(variables) or any(not isinstance(value, str) or not value for value in variables):
        raise AmbiguousVariableError('variables must contain distinct, non-empty strings')
    ordered_variables = sorted(variables, key=lambda value: (-len(value), value))
    tokens = []
    index = 0
    while index < len(expression):
        character = expression[index]
        if character.isspace():
            index += 1
            continue
        number_match = re.match(r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', expression[index:])
        if number_match:
            value = number_match.group(0)
            tokens.append(_PolynomialToken('NUMBER', value, index))
            index += len(value)
            continue
        if expression.startswith('**', index):
            tokens.append(_PolynomialToken('POWER', '**', index))
            index += 2
            continue
        if character == '^':
            tokens.append(_PolynomialToken('POWER', character, index))
            index += 1
            continue
        if character in '+-*':
            tokens.append(_PolynomialToken('OP', character, index))
            index += 1
            continue
        if character == '(':
            tokens.append(_PolynomialToken('LPAREN', character, index))
            index += 1
            continue
        if character == ')':
            tokens.append(_PolynomialToken('RPAREN', character, index))
            index += 1
            continue
        if character.isalpha() or character == '_':
            if tokens and tokens[-1].kind == 'POWER':
                raise UnsupportedExpressionError(
                    f"Encountered an invalid power beginning at position {index}"
                )
            if explicit_variables:
                variable = next(
                    (candidate for candidate in ordered_variables if expression.startswith(candidate, index)),
                    None,
                )
                if variable is None:
                    if not ordered_variables:
                        raise EquationParseError(
                            "Couldn't parse the expression: expected a valid free number"
                        )
                    if any(candidate in expression[index:] for candidate in ordered_variables):
                        raise EquationParseError(
                            f"Encountered an invalid coefficient at position {index}"
                        )
                    raise AmbiguousVariableError(
                        f"Unknown variable beginning at position {index}: {expression[index:]!r}"
                    )
            else:
                # Legacy KiwiCalc syntax treats adjacent letters as multiplied
                # single-character variables (``xy`` means ``x*y``).
                variable = character
            tokens.append(_PolynomialToken('VARIABLE', variable, index))
            index += len(variable)
            continue
        raise EquationParseError(f"Unexpected character {character!r} at position {index}")
    tokens.append(_PolynomialToken('EOF', '', len(expression)))
    return tokens, variables


class _PolynomialParser:
    _MAX_EXPONENT = 1000
    _MAX_TERMS = 10000

    def __init__(self, tokens, variables):
        self.tokens = tokens
        self.variables = list(variables)
        self.variable_indexes = {variable: index for index, variable in enumerate(self.variables)}
        self.position = 0

    @property
    def current(self):
        return self.tokens[self.position]

    def advance(self):
        token = self.current
        self.position += 1
        return token

    def parse(self):
        result = self.parse_sum()
        if self.current.kind != 'EOF':
            raise EquationParseError(
                f"Unexpected token {self.current.value!r} at position {self.current.position}"
            )
        return self._clean(result)

    def parse_sum(self):
        result = self.parse_product()
        while self.current.kind == 'OP' and self.current.value in ('+', '-'):
            operation = self.advance().value
            other = self.parse_product()
            result = self._add(result, other, 1 if operation == '+' else -1)
        return result

    def parse_product(self):
        result = self.parse_unary()
        previous_kind = self.tokens[self.position - 1].kind
        while True:
            if self.current.kind == 'OP' and self.current.value == '*':
                self.advance()
                other = self.parse_unary()
            elif self.current.kind in ('VARIABLE', 'LPAREN', 'NUMBER'):
                if previous_kind == 'NUMBER' and self.current.kind == 'NUMBER':
                    raise EquationParseError(
                        f"Missing operator before number at position {self.current.position}"
                    )
                other = self.parse_unary()
            else:
                break
            result = self._multiply(result, other)
            previous_kind = self.tokens[self.position - 1].kind
        return result

    def parse_unary(self):
        if self.current.kind == 'OP' and self.current.value in ('+', '-'):
            operation = self.advance().value
            result = self.parse_unary()
            return result if operation == '+' else {powers: -value for powers, value in result.items()}
        return self.parse_power()

    def parse_power(self):
        result = self.parse_primary()
        if self.current.kind == 'POWER':
            self.advance()
            if self.current.kind != 'NUMBER':
                raise UnsupportedExpressionError('Polynomial exponents must be non-negative integers')
            exponent_token = self.advance()
            exponent_value = float(exponent_token.value)
            if not exponent_value.is_integer() or exponent_value < 0:
                raise UnsupportedExpressionError(
                    f"Polynomial powers must be non-negative integers, found {exponent_token.value!r}"
                )
            exponent = int(exponent_value)
            if exponent > self._MAX_EXPONENT:
                raise UnsupportedExpressionError(
                    f'Polynomial exponent {exponent} exceeds the supported limit {self._MAX_EXPONENT}'
                )
            result = self._power(result, exponent)
        return result

    def parse_primary(self):
        token = self.current
        zero_powers = (0,) * len(self.variables)
        if token.kind == 'NUMBER':
            self.advance()
            return {zero_powers: float(token.value)}
        if token.kind == 'VARIABLE':
            self.advance()
            powers = [0] * len(self.variables)
            powers[self.variable_indexes[token.value]] = 1
            return {tuple(powers): 1.0}
        if token.kind == 'LPAREN':
            self.advance()
            result = self.parse_sum()
            if self.current.kind != 'RPAREN':
                raise EquationParseError(f"Unclosed '(' at position {token.position}")
            self.advance()
            return result
        raise EquationParseError(f"Expected a number, variable, or '(' at position {token.position}")

    def _add(self, first, second, factor=1):
        result = dict(first)
        for powers, coefficient in second.items():
            result[powers] = result.get(powers, 0.0) + factor * coefficient
        return self._clean(result)

    def _multiply(self, first, second):
        if len(first) * len(second) > self._MAX_TERMS:
            raise UnsupportedExpressionError('Polynomial expansion exceeds the supported term limit')
        result = {}
        for first_powers, first_coefficient in first.items():
            for second_powers, second_coefficient in second.items():
                powers = tuple(a + b for a, b in zip(first_powers, second_powers))
                result[powers] = result.get(powers, 0.0) + first_coefficient * second_coefficient
        return self._clean(result)

    def _power(self, polynomial, exponent):
        result = {(0,) * len(self.variables): 1.0}
        factor = polynomial
        while exponent:
            if exponent & 1:
                result = self._multiply(result, factor)
            exponent //= 2
            if exponent:
                factor = self._multiply(factor, factor)
        return result

    @staticmethod
    def _clean(polynomial):
        return {powers: value for powers, value in polynomial.items() if value != 0}


def _parse_polynomial_terms(expression, variables=None):
    tokens, resolved_variables = _tokenize_polynomial(expression, variables)
    parser = _PolynomialParser(tokens, resolved_variables)
    return parser.parse(), resolved_variables


def _terms_to_legacy_dict(terms, variables):
    result = {variable: [] for variable in variables}
    result['free'] = 0.0
    for powers, coefficient in terms.items():
        active = [index for index, power in enumerate(powers) if power]
        if not active:
            result['free'] += coefficient
            continue
        if len(active) != 1:
            monomial = '*'.join(
                f'{variables[index]}^{powers[index]}' for index in active
            )
            raise UnsupportedExpressionError(
                f"The legacy coefficient dictionary cannot represent mixed monomial {monomial!r}"
            )
        variable_index = active[0]
        variable = variables[variable_index]
        power = powers[variable_index]
        coefficients = result[variable]
        if len(coefficients) < power:
            coefficients[:0] = [0.0] * (power - len(coefficients))
        coefficients[len(coefficients) - power] += coefficient
    return result

def split_expression(expression: str):
    """splits the expression by delimiters, but doesn't touch what's inside parenthesis """
    if not isinstance(expression, str):
        raise TypeError('expression must be a string')
    delimiters = []
    depths = {'(': 0, '{': 0, '[': 0}
    closing = {')': '(', '}': '{', ']': '['}
    for index, char in enumerate(expression):
        if char in depths:
            depths[char] += 1
            continue
        if char in closing:
            opener = closing[char]
            depths[opener] -= 1
            if depths[opener] < 0:
                raise ValueError(f"Unmatched closing delimiter '{char}'")
            continue
        if char in ('+', '-') and index > 0 and not any(depths.values()):
            previous = expression[index - 1]
            if previous not in ('e', 'E', '^', '*'):
                delimiters.append(index)
    if any(depths.values()):
        raise ValueError('Unclosed grouping delimiter in expression')
    boundaries = [0] + delimiters + [len(expression)]
    return [expression[start:stop] for start, stop in zip(boundaries, boundaries[1:]) if expression[start:stop]]


def _matching_closing_index(expression: str, opening_index: int) -> int:
    """Find the parenthesis that closes the group at ``opening_index``."""
    depth = 0
    for index in range(opening_index, len(expression)):
        if expression[index] == '(':
            depth += 1
        elif expression[index] == ')':
            depth -= 1
            if depth == 0:
                return index
    return -1

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
    if not isinstance(expression, str):
        expression = str(expression)
    # Scientific-notation markers are part of the number, not variables.
    without_scientific_numbers = re.sub(
        r'(?<![A-Za-z_])(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+', '', expression
    )
    return {character for character in without_scientific_numbers if character.isalpha()}

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

def _poly_from_str(poly_expression: str, get_list=False, variables=None) -> 'Union[Poly,List]':
    """Build a ``Poly`` using the recursive parser and optional explicit variables."""
    from kiwicalc.expressions.mono import Mono
    from kiwicalc.expressions.poly import Poly
    terms, resolved_variables = _parse_polynomial_terms(poly_expression, variables)
    expressions = [
        Mono(
            coefficient=coefficient,
            variables_dict={
                variable: powers[index]
                for index, variable in enumerate(resolved_variables)
                if powers[index]
            },
        )
        for powers, coefficient in terms.items()
    ]
    if not expressions:
        expressions = [Mono(0)]
    if get_list:
        return expressions
    return Poly(expressions)


def poly_from_str(poly_expression: str, get_list=False) -> 'Union[Poly,List]':
    """
    Analyzes a string, such as "3x^2 + 2xy - 7" and generates a polynomial expression
    :param poly_expression:
    :param get_list: if set to True, a list of the monomials ( Mono objects ) will be returned instead
    of a Poly object
    :return: a polynomial corresponding to the string, or a list of monomials.
    :rtype: Poly or list
    """
    return _poly_from_str(poly_expression, get_list=get_list)

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
    new_expressions: list = [
        mono_from_str(expression)
        if is_number(clean_from_spaces(expression).lstrip('+'))
        else TrigoExpr_from_str(expression)
        for expression in trigo_expressions
    ]
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
        ending_parenthesis = _matching_closing_index(expression, start_parenthesis)
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
        if not isinstance(expression, str):
            raise TypeError('expression must be a string')
        if not clean_from_spaces(expression):
            raise ValueError('A polynomial expression cannot be empty')
        terms, variables = _parse_polynomial_terms(expression, variables)
        variables_dict = _terms_to_legacy_dict(terms, variables)
        if numpy_array and len(variables) == 1:
            result = np.asarray(
                list(variables_dict[variables[0]]) + [variables_dict['free']], dtype='float64'
            )
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
        expression = clean_from_spaces(expression).replace('**', '^')
        try:
            return (float(expression), 'free', 0)
        except ValueError:
            pass
        variable_pattern = '|'.join(re.escape(variable) for variable in sorted(variables, key=len, reverse=True))
        if not variable_pattern:
            raise ValueError("Couldn't parse the expression: expected a valid free number")
        number = r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
        match = re.fullmatch(
            rf'(?P<coefficient>[+-]?(?:{number})?)(?P<multiply>\*)?'
            rf'(?P<variable>{variable_pattern})(?:\^(?P<power>[+-]?(?:\d+(?:\.\d*)?|\.\d+)))?',
            expression,
        )
        if match is None or (match.group('multiply') and not match.group('coefficient').lstrip('+-')):
            if '^' in expression:
                raise ValueError(f"Encountered an invalid power while parsing the monomial '{expression}'")
            if any(variable in expression for variable in variables):
                raise ValueError(f"Encountered an invalid coefficient while parsing the monomial '{expression}'")
            raise ValueError(f"Couldn't parse the polynomial monomial '{expression}'")
        coefficient_text = match.group('coefficient')
        coefficient = -1.0 if coefficient_text == '-' else 1.0 if coefficient_text in ('', '+') else float(coefficient_text)
        power_text = match.group('power')
        if power_text is None:
            power = 1
        else:
            power_value = float(power_text)
            if not power_value.is_integer() or power_value < 0:
                raise ValueError(f"Polynomial powers must be non-negative integers, found '{power_text}'")
            power = int(power_value)
        return coefficient, match.group('variable'), power

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
