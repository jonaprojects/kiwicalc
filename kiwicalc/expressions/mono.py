from __future__ import annotations
import math
import cmath
from math import (
    sqrt, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
    asinh, acosh, atanh, floor, ceil, gamma, comb, factorial, log
)
import random
import json
import warnings
from functools import reduce
from collections import Counter
from typing import Union, Tuple, List, Optional, Any, Callable, Iterator, Set, Dict, Iterable
import numpy as np
import matplotlib.pyplot as plt

from kiwicalc.core.constants import (
    TRIGONOMETRY_CONSTANTS, MATHEMATICAL_CONSTANTS,
    ptn, number_pattern, allowed_characters, pi, e, tau
)
from kiwicalc.core.interfaces import IExpression, IPlottable, IScatterable
from kiwicalc.core.utils import (
    copy_expression, clean_from_spaces, extract_coefficient, format_coefficient,
    format_free_number, is_number, is_lambda, round_decimal, to_lambda,
    float_gcd, gcd, sorted_expressions, equal_ignore_order, process_object,
    contains_from_list, decimal_range, apply_on, is_evaluatable, only_numbers_letters,
    apply_parenthesis, handle_parenthesis, handle_abs, handle_factorial,
    _format_minus, ln, max_power, formatted_expression
)
from kiwicalc.parsing.parse_expression import (
    split_expression, extract_variables_from_expression, __data_from_single,
    mono_from_str, poly_from_str, log_from_str, TrigoExpr_from_str,
    TrigoExprs_from_str, poly_frac_from_str, fetch_power, fetch_variable,
    ParseExpression
)
from kiwicalc.expressions.factory import create, create_from_dict


class Mono(IExpression, IPlottable, IScatterable):
    __slots__ = ['_coefficient', '__variables']

    def __init__(self, coefficient: Union[str, int, float]=None, variables_dict: dict=None):
        if isinstance(coefficient, str):
            self._coefficient, self.__variables = mono_from_str(coefficient, get_tuple=True)
        elif isinstance(coefficient, (int, float)):
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
        return 0 if self.__variables is None else len(self.__variables)

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
                if other_evaluation == 0:
                    return self
                return Poly(expressions=(self, Mono(other_evaluation)))
            if isinstance(other, Mono):
                if other._coefficient == 0:
                    return self
                if not self.__variables and (not other.__variables):
                    self._coefficient += other._coefficient
                    return self
                if self.__variables == other.__variables:
                    self._coefficient += other._coefficient
                    if self._coefficient == 0:
                        self.__variables = None
                        return self
                    if self.__variables is not None and other.__variables is not None:
                        self.__variables = {**self.__variables, **other.__variables}
                    else:
                        first_variables = self.__variables if self.__variables is not None else {}
                        second_variables = other.__variables if other.__variables is not None else {}
                        self.__variables = {**first_variables, **second_variables}
                    return self
                else:
                    return Poly([self, other])
            elif isinstance(other, Poly):
                return other.__add__(self)
            return ExpressionSum((self, other))
        else:
            raise TypeError(f'Mono.__add__: invalid type {type(other)}. Expected Mono,Poly, str,int,float.')

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
                if other_evaluation == 0:
                    return self
                return Poly((self, Mono(-other_evaluation)))
            if isinstance(other, Mono):
                if other._coefficient == 0:
                    return self
                if not self.__variables and (not other.__variables):
                    self._coefficient -= other._coefficient
                    return self
                if self.__variables == other.__variables:
                    self._coefficient -= other._coefficient
                    if self._coefficient == 0:
                        self.__variables = None
                        return self
                    if self.__variables is not None and other.__variables is not None:
                        self.__variables = {**self.__variables, **other.__variables}
                    else:
                        first_variables = self.__variables if self.__variables is not None else {}
                        second_variables = other.__variables if other.__variables is not None else {}
                        self.__variables = {**first_variables, **second_variables}
                    self.simplify()
                    return self
                else:
                    return Poly([self, -other])
            elif isinstance(other, Poly):
                return other.__neg__().__iadd__(self)
            else:
                return ExpressionSum((self, -other))
        else:
            raise TypeError(f'Mono.__add__: invalid type {type(other)}. Expected Mono,Poly, str,int,float.')

    def __sub__(self, other: Union[int, float, str, IExpression]) -> 'Union[Mono,Poly]':
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
        if isinstance(other, Mono):
            my_counter = Counter(self.__variables)
            my_counter.update(other.__variables)
            self.__variables = dict(my_counter)
            self.simplify()
            if self.__variables == {}:
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
            if evaluated_other is not None:
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
                raise ValueError('Cannot divide a Mono object by 0.')
            self._coefficient /= other
            return self
        elif isinstance(other, str):
            other = Mono(other)
        if isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if other_evaluation is not None:
                if other_evaluation == 0:
                    raise ValueError('Cannot divide a Mono object by 0')
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
                    raise ZeroDivisionError('Cannot divide by 0 !')
                elif other.__variables in ({}, None):
                    self._coefficient /= other._coefficient
                    return self
                if self.num_of_variables < other.num_of_variables:
                    return PolyFraction(self, other)
                my_variables, other_variables = (self.variables, other.variables)
                if my_variables != other_variables:
                    return PolyFraction(self, other)
                if any((my_value < other_value for my_value, other_value in zip(self.__variables.values(), other.__variables.values()))):
                    return PolyFraction(self, other)
                my_keys = [] if self.__variables in (None, {}) else list(self.variables)
                keys = my_keys + list(other.variables)
                new_variables = dict.fromkeys(keys, 0)
                if self.__variables is not None:
                    for key, value in self.__variables.items():
                        new_variables[key] += value
                for key, value in other.__variables.items():
                    new_variables[key] -= value
                new_variables = {key: value for key, value in new_variables.items() if value != 0}
                if new_variables == {}:
                    new_variables = None
                self._coefficient /= other._coefficient
                self.__variables = new_variables
                return self
            else:
                return Fraction(self, other)
        else:
            raise TypeError(f'Invalid type {type(other)} for dividing a Mono object.')

    def __ipow__(self, power: Union[int, float, IExpression]):
        if isinstance(power, IExpression):
            power_eval = power.try_evaluate()
            if power_eval is None:
                return Exponent(self, power)
            power = power_eval
        if power == 0:
            self._coefficient = 1
            self.__variables = None
            return self
        self._coefficient **= power
        if self.__variables not in (None, {}):
            self.__variables = {variable_name: variable_power * power for variable_name, variable_power in self.__variables.items()}
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

    def __lt__(self, other):
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
            biggest_power1, biggest_power2 = (max(self.__variables.values()), max(other.__variables.values()))
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
                    first_variable, second_variable = (fetch_variable(self.__variables), fetch_variable(other.__variables))
                    return first_variable < second_variable
                else:
                    max_string1 = max(list(self.__variables.keys()))
                    max_string2 = max(list(self.__variables.keys()))
                    return self if max_string1 < max_string2 else other
        raise TypeError(f'Invalid type {type(other)} for __lt__(). Expected type Mono')

    def __str__(self):
        if self.__variables is not None:
            coefficient = round_decimal(self._coefficient) if self._coefficient not in (-1, 1) else '-' if self.coefficient == -1 else ''
            variables = '*'.join([f'{variable}' if power == 1 else f'{variable}^{round_decimal(power)}' for variable, power in self.__variables.items()])
            return f'{coefficient}{variables}'
        result = str(round_decimal(self._coefficient))
        return result

    def contains_variable(self, variable: str) -> bool:
        """Checking whether a given variable appears in the expression """
        if self.__variables in (None, {}):
            return False
        return variable in self.__variables

    def is_number(self) -> bool:
        """Checks whether the Mono represents a free number-
        If it is a free number, True will be returned, otherwise - False
        """
        return self.__variables in ({}, None)

    def latex(self):
        variables = '*'.join([f'{variable}^{{{power}}}' if self._coefficient >= 0 else f'({variable}^{{{power}}})' for variable, power in self.__variables.items()])
        return f'{self._coefficient}*{variables} '

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
        return Mono(coefficient=self._coefficient, variables_dict=self.__variables.copy() if self.__variables is not None else None)

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

    def __call__(self, **kwargs):
        pass

    def simplify(self):
        if self._coefficient == 0:
            self.__variables = None
        elif self.__variables:
            self.__variables = {key: value for key, value in self.__variables.items() if value != 0}
        else:
            self.__variables = None

    def python_syntax(self) -> str:
        if self.__variables in ({}, None):
            return f'{self._coefficient}'
        formatted_variables = '*'.join((f'{variable}' if power == 1 else f'{variable}**{power}' for variable, power in self.__variables.items()))
        coefficient = format_coefficient(self._coefficient)
        if coefficient not in ('', '+', '-'):
            coefficient += '*'
        return f'{coefficient}{formatted_variables}'

    def try_evaluate(self) -> Optional[float]:
        """ trying to evaluate the Mono object into a float number """
        if self.__variables in ({}, None):
            return self._coefficient
        return None

    def derivative(self):
        if self.__variables is not None and len(self.__variables) > 1:
            raise ValueError(f'Try using partial_derivative(), for expression with more than one variable ')
        if self.__variables is None:
            return 0
        power = fetch_power(self.__variables)
        if power == 1:
            return self._coefficient
        elif power == 0:
            return 0
        elif power != 0:
            return Mono(self._coefficient * power, variables_dict={fetch_variable(self.__variables): power - 1})

    def partial_derivative(self, variables: Iterable):
        if self.__variables is None:
            return Mono(0)
        derived_expression = self.__copy__()
        for variable in variables:
            if variable not in self.__variables:
                return Mono(0)
            derived_expression._coefficient *= derived_expression.__variables[variable]
            derived_expression.__variables[variable] -= 1
            if derived_expression.__variables[variable] == 0:
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
            raise ValueError(f'Can only compute the derivative with one variable or less ( got {len(self.__variables)}')
        if self.__variables is None:
            return Mono(self._coefficient, variables_dict={variable_name: 1})
        variable, power = (fetch_variable(self.__variables), fetch_power(self.__variables))
        if power == 0:
            return Mono(round_decimal(self._coefficient), variables_dict={variable: 1})
        elif power == -1:
            return self._coefficient * Ln(Abs(Mono(1, variables_dict={variable: 1})))
        elif power != -1:
            return Mono(round_decimal(self._coefficient / (power + 1)), variables_dict={variable: power + 1})

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
        from kiwicalc.functions.function import Function

        return Function(self.python_syntax())


from kiwicalc.expressions.var import Var
from kiwicalc.expressions.mono import Mono
from kiwicalc.expressions.poly import FastPoly, Poly, synthetic_division
from kiwicalc.expressions.sum import ExpressionSum
from kiwicalc.expressions.mul import ExpressionMul
from kiwicalc.expressions.fractions import Fraction, PolyFraction
from kiwicalc.expressions.roots import Root, Sqrt
from kiwicalc.expressions.log import Log, PolyLog, Ln
from kiwicalc.expressions.trigonometry import (
    TrigoExpr, Sin, Asin, Cos, Acos, Tan, Atan, Cot,
    Sec, Acot, ASec, Csc, ACsc, TrigoExprs
)
from kiwicalc.expressions.special import Factorial, Abs, Exponent

