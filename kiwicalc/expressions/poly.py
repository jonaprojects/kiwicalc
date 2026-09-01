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

import os
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.units import cm
from itertools import permutations, combinations
from kiwicalc.core.ranges import Range, RangeOR, LESS_THAN
from kiwicalc.parsing.parse_equation import add_or_sub_coefficients
from kiwicalc.expressions.mono import Mono

class FastPoly(IExpression, IPlottable):
    __slots__ = ['__variables', '__variables_dict']

    def __init__(self, polynomial: Union[str, dict, list, tuple, float, int], variables: Iterable=None):
        self.__variables = None if variables is None else list(variables)
        if isinstance(polynomial, (int, float)):
            self.__variables = []
            self.__variables_dict = {'free': polynomial}
        elif isinstance(polynomial, str):
            self.__variables_dict = ParseExpression.parse_polynomial(polynomial, self.__variables, strict_syntax=True)
            self.__variables_dict = ParseExpression.parse_polynomial(polynomial, self.__variables, strict_syntax=True)
        elif isinstance(polynomial, dict):
            if 'free' not in polynomial.keys():
                raise KeyError(f"Key 'free' must appear in FastPoly.__init__() when entering dict. Its value is thefree number of the expression")
            self.__variables_dict = {
                key: value.copy() if isinstance(value, list) else value
                for key, value in polynomial.items()
            }
        elif isinstance(polynomial, (list, tuple)):
            if not polynomial:
                raise ValueError(f'FastPoly.__init__(): At least one coefficient is required.')
            if self.__variables is None:
                self.__variables = ['x']
            elif len(self.__variables) > 1:
                raise ValueError(f'FastPoly.__init__(): When entering a list of coefficients, only 1 variableis accepted, but found {len(self.__variables)}')
            x_coefficients, free_number = (polynomial[:-1], polynomial[-1])
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

    def derivative(self) -> 'FastPoly':
        from kiwicalc.core.utils import derivative

        num_of_variables = self.num_of_variables
        if num_of_variables == 0:
            return FastPoly(0)
        elif num_of_variables == 1:
            variable = self.__variables[0]
            derivative_coefficients = derivative(self.__variables_dict[variable] + [self.__variables_dict['free']])
            if isinstance(derivative_coefficients, (int, float)):
                return FastPoly(derivative_coefficients)
            return FastPoly(derivative_coefficients, variables=[variable])
        else:
            raise ValueError('Please use the partial_derivative() method for polynomials with several variables')

    def partial_derivative(self, variables: Iterable[str]):
        pass

    def extremums(self):
        from kiwicalc.geometry.points import Point2D
        from kiwicalc.geometry.point_collections import PointCollection

        num_of_variables = len(self.__variables)
        if num_of_variables == 0:
            return None
        elif num_of_variables == 1:
            my_lambda = self.to_lambda()
            my_derivative = self.derivative()
            if my_derivative.is_free_number:
                return None
            derivative_roots = my_derivative.roots(nmax=1000)
            myRoots = [Point2D(root.real, my_lambda(root.real)) for root in derivative_roots if root.imag <= 1e-05]
            return PointCollection(myRoots)
        else:
            pass

    def integral(self, c: float=0, variable='x'):
        from kiwicalc.core.utils import integral

        num_of_variables = len(self.__variables)
        if num_of_variables == 0:
            return FastPoly({variable: [self.__variables_dict['free']], 'free': c})
        elif num_of_variables != 1:
            raise ValueError('Cannot integrate a PolyFast object with more than 1 variable')
        coefficients = self.__variables_dict[self.__variables[0]] + [self.__variables_dict['free']]
        return FastPoly(integral(coefficients, c=c), variables=[self.__variables[0]])

    def newton(self, initial: float=0, epsilon: float=1e-05, nmax=10000):
        from kiwicalc.numeric.roots import newton_raphson

        return newton_raphson(self.to_lambda(), self.derivative().to_lambda(), initial, epsilon, nmax)

    def halley(self, initial: float=0, epsilon: float=1e-05, nmax=10000):
        from kiwicalc.numeric.roots import halleys_method

        first_derivative = self.derivative()
        second_derivative = first_derivative.derivative()
        second_callable = (
            (lambda _: second_derivative.try_evaluate())
            if second_derivative.num_of_variables == 0
            else second_derivative.to_lambda()
        )
        return halleys_method(self.to_lambda(), first_derivative.to_lambda(), second_callable, initial, epsilon, nmax)

    def __add_or_sub(self, other: 'FastPoly', mode: str):
        for variable in other.__variables:
            if variable in self.__variables:
                add_or_sub_coefficients(self.__variables_dict[variable], other.__variables_dict[variable], mode=mode, copy_first=False)
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
        if isinstance(other, (int, float)):
            if other == 0:
                return self
            self.__variables_dict['free'] += other
            return self
        if not isinstance(other, IExpression):
            raise TypeError(f"Invalid type {type(other)} when adding FastPoly objects. Expected types 'int', 'float', or 'IExpression'")
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
            raise TypeError(f"Invalid type {type(other)} when subtracting FastPoly objects. Expected types 'int', 'float', or 'IExpression'")
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
            del self.__variables_dict[variable]
            self.__variables.remove(variable)

    def simplify(self):
        warnings.warn('FastPoly objects are already simplified. Method is deprecated.')

    def try_evaluate(self) -> Optional[float]:
        if self.num_of_variables == 0:
            return self.__variables_dict['free']
        return None

    def roots(self, epsilon=1e-05, nmax: int=10000):
        from kiwicalc.equations.single import solve_polynomial
        num_of_variables = len(self.__variables)
        if num_of_variables == 0:
            return 'Infinite' if self.__variables_dict['free'] == 0 else None
        elif num_of_variables == 1:
            return solve_polynomial(self.__variables_dict[self.__variables[0]] + [self.__variables_dict['free']], epsilon, nmax)
        else:
            raise ValueError(f'Can only solve polynomials with 1 variable, but found {num_of_variables}')

    def __eq__(self, other: 'Union[IExpression, FastPoly]'):
        """Equate between expressions. not fully compatible with the IExpression classes ..."""
        if other is None:
            return False
        if not isinstance(other, IExpression):
            raise TypeError(f'Invalid type {type(other)} for equating FastPoly objects')
        other_evaluation = other.try_evaluate()
        if other_evaluation is not None:
            my_evaluation = self.try_evaluate()
            if my_evaluation is not None:
                return my_evaluation == other_evaluation
        if not isinstance(other, FastPoly):
            return False
        return self.__variables_dict == other.__variables_dict

    def __ne__(self, other: 'FastPoly'):
        return not self.__eq__(other)

    def __neg__(self):
        new_dict = {variable: [-coefficient for coefficient in coefficients] for variable, coefficients in self.__variables_dict.items() if variable != 'free'}
        new_dict['free'] = -self.__variables_dict['free']
        return FastPoly(new_dict)

    def __copy__(self):
        return FastPoly(self.__variables_dict)

    def to_lambda(self):
        return to_lambda(self.__str__(), self.__variables)

    def plot(self, start: float=-10, stop: float=10, step: float=0.01, ymin: float=-10, ymax: float=10, text=None, show_axis=True, show=True, fig=None, ax=None, formatText=True, values=None):
        from kiwicalc.plotting.plots import plot_function, plot_function_3d
        lambda_expression = self.to_lambda()
        num_of_variables = self.num_of_variables
        if text is None:
            text = self.__str__()
        if num_of_variables == 0:
            raise ValueError('Cannot plot a polynomial with 0 variables_dict')
        elif num_of_variables == 1:
            plot_function(lambda_expression, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=text, show_axis=show_axis, show=show, fig=fig, ax=ax, formatText=formatText, values=values)
        elif num_of_variables == 2:
            plot_function_3d(lambda_expression, start=start, stop=stop, step=step)
        else:
            raise ValueError('Cannot plot a function with more than two variables_dict (As for this version)')

    def to_dict(self):
        return {'type': 'FastPoly', 'data': self.__variables_dict.copy()}

    @staticmethod
    def from_dict(given_dict: dict):
        if given_dict.get('type', '').strip().lower() != 'fastpoly':
            raise ValueError("FastPoly.from_dict() expected a FastPoly serialization payload")
        return FastPoly(given_dict['data'])

    @staticmethod
    def from_json(json_content: str):
        loaded_json = json.loads(json_content)
        if loaded_json['type'].strip().lower() != 'fastpoly':
            raise ValueError(f"Unexpected type '{loaded_json['type']}' when creating a new FastPoly object from JSON (Expected TypePoly).")
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
        elif isinstance(expressions, Poly):
            self._expressions = [mono_expression.__copy__() for mono_expression in expressions._expressions]
            self.simplify()
        elif isinstance(expressions, Iterable):
            self._expressions = []
            for expression in expressions:
                if isinstance(expression, Mono):
                    self._expressions.append(expression.__copy__())
                elif isinstance(expression, str):
                    self._expressions += poly_from_str(expression, get_list=True)
                elif isinstance(expression, Poly):
                    self._expressions.extend(expression.expressions.copy())
                elif isinstance(expression, (int, float)):
                    self._expressions.append(Mono(expression))
                else:
                    warnings.warn(f"Couldn't process expression '{expression} with invalid type {type(expression)}'")
            self.simplify()
        else:
            raise TypeError(f'Invalid type {type(expressions)} in Poly.__init__(). Allowed types: list,tuple,Mono,Poly,str,int,float ')

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
            if all((expression.coefficient == 0 for expression in self.expressions)):
                self.expressions = [Mono(0)]
            self.simplify()
            return self
        elif isinstance(other, str):
            expressions = poly_from_str(other, get_list=True)
            for mono_expression in expressions:
                self.__add_monomial(mono_expression)
            if all((expression.coefficient == 0 for expression in self.expressions)):
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
                if all((expression.coefficient == 0 for expression in self.expressions)):
                    self.expressions = [Mono(0)]
                self.simplify()
                return self
            elif isinstance(other, Poly):
                for mono_expression in other.expressions:
                    self.__add_monomial(mono_expression)
                if all((expression.coefficient == 0 for expression in self.expressions)):
                    self.expressions = [Mono(0)]
                self.simplify()
                return self
            else:
                return ExpressionSum((self, other))
        else:
            raise TypeError(f"__add__ : invalid type '{type(other)}'. Allowed types: str, Mono, Poly, int, or float")

    def __add_monomial(self, other: Mono) -> None:
        self.__filter_zeroes()
        for index, expression in enumerate(self.expressions):
            if expression.variables_dict == other.variables_dict or (not expression.variables and (not other.variables)):
                self._expressions[index] += other
                return
        self._expressions.append(other)

    def __sub_monomial(self, other: Mono) -> None:
        self.__filter_zeroes()
        for index, expression in enumerate(self._expressions):
            if expression.variables_dict == other.variables_dict or (not expression.variables and (not other.variables)):
                self._expressions[index] -= other
                return
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
            raise TypeError(f'Poly.__rsub__: Expected types int,float,str,Mono,Poly, but got {type(other)}')

    def __isub__(self, other: Union[int, float, IExpression, str]):
        if isinstance(other, (int, float)):
            self.__sub_monomial(Mono(other))
            if all((expression.coefficient == 0 for expression in self.expressions)):
                self.expressions = [Mono(0)]
            self.simplify()
            return self
        elif isinstance(other, str):
            expressions = poly_from_str(other, get_list=True)
            for mono_expression in expressions:
                self.__sub_monomial(mono_expression)
            if all((expression.coefficient == 0 for expression in self.expressions)):
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
                if all((expression.coefficient == 0 for expression in self._expressions)):
                    self.expressions = [Mono(0)]
                self.simplify()
                return self
            elif isinstance(other, Poly):
                for mono_expression in other._expressions:
                    self.__sub_monomial(mono_expression)
                if all((expression.coefficient == 0 for expression in self._expressions)):
                    self.expressions = [Mono(0)]
                self.simplify()
                return self
            else:
                return ExpressionSum((self, -other))
        else:
            raise TypeError(f"Invalid type '{type(other)} while subtracting polynomials.")

    def __neg__(self):
        return Poly([-expression for expression in self.expressions])

    def __imul__(self, other: Union[int, float, IExpression]):
        from kiwicalc.geometry.vectors import Vector
        from kiwicalc.linalg.matrix import Matrix

        if isinstance(other, (int, float)) and other == 0:
            return Mono(coefficient=0)
        if isinstance(other, (int, float)):
            for index, expression in enumerate(self._expressions):
                self._expressions[index].coefficient *= other
            if all((expression.coefficient == 0 for expression in self._expressions)):
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
                        for index, new_expression in enumerate(new_expressions):
                            if new_expression.variables_dict == result.variables_dict:
                                addition_result = new_expression + result
                                if addition_result.coefficient == 0:
                                    del new_expressions[index]
                                else:
                                    new_expressions[index] = addition_result
                                found = True
                                break
                        if not found:
                            new_expressions.append(result.__copy__())
                self._expressions = new_expressions
                self.simplify()
                return self
            else:
                return other * self
        elif isinstance(other, Matrix):
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

    def divide_by_poly(self, other: 'Union[Mono, Poly]', get_remainder=False, nmax=1000):
        if isinstance(other, Poly) and len(other.expressions) == 1:
            other = other.expressions[0]
        if isinstance(other, Mono):
            if other.coefficient == 0:
                raise ZeroDivisionError('cannot divide by an expression whose coefficient is zero')
            other_copy = other.__copy__()
            other_copy.coefficient = 1 / other_copy.coefficient
            if other_copy.variables_dict is not None:
                other_copy.variables_dict = {variable: -value for variable, value in other_copy.variables_dict.items()}
            if get_remainder:
                return (self.__imul__(other_copy), 0)
            return self.__imul__(other_copy)
        elif isinstance(other, Poly):
            new_expression, remainder = (Mono(0), 0)
            temp_expressions = Poly(self._expressions.copy())
            for i in range(nmax):
                if len(temp_expressions._expressions) == 0:
                    new_expression.simplify()
                    if get_remainder:
                        return (new_expression, 0)
                    return new_expression
                if len(temp_expressions._expressions) == 1 and temp_expressions.expressions[0].variables_dict is None:
                    if get_remainder:
                        return (new_expression, other._expressions[0])
                    return new_expression + other._expressions[0] / other
                dividend_lead = temp_expressions._expressions[0]
                divisor_lead = other._expressions[0]
                dividend_powers = dividend_lead.variables_dict or {}
                divisor_powers = divisor_lead.variables_dict or {}
                if any(dividend_powers.get(variable, 0) < power for variable, power in divisor_powers.items()):
                    temp_expressions.simplify()
                    temp_expressions.sort()
                    if get_remainder:
                        return (new_expression, temp_expressions)
                    return new_expression + Fraction(temp_expressions, other)
                first_item = temp_expressions._expressions[0] / other._expressions[0]
                new_expression += first_item.__copy__()
                subtraction_expressions = first_item * other
                temp_expressions -= subtraction_expressions
                if len(temp_expressions.expressions) == 1:
                    if temp_expressions.expressions[0].coefficient == 0:
                        if isinstance(new_expression, Poly):
                            new_expression.simplify()
                        if get_remainder:
                            return (new_expression, remainder)
                        if remainder == 0:
                            return new_expression
                        new_expression += Fraction(remainder, other)
                        return new_expression
                    elif temp_expressions.expressions[0].variables_dict is None:
                        if isinstance(new_expression, Poly):
                            new_expression.sort()
                        remainder = temp_expressions.expressions[0].coefficient
                        if get_remainder:
                            return (new_expression, remainder)
                        if remainder == 0:
                            return new_expression
                        new_expression += Fraction(remainder, other)
                        return new_expression
                    else:
                        warnings.warn('Got an algebraic remainder when dividing Poly objects')
                        if isinstance(new_expression, Poly):
                            new_expression.sort()
                        if get_remainder:
                            return (new_expression, remainder)
                        if remainder == 0:
                            return new_expression
                        new_expression += Fraction(remainder, other)
                        return new_expression
            warnings.warn('Division timed out ...')
            return PolyFraction(self, other)

    def __itruediv__(self, other: Union[int, float, IExpression], get_remainder=False):
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError('cannot divide by 0')
            if my_evaluation is None:
                if get_remainder:
                    return (self.divide_by_number(other), 0)
                return self.divide_by_number(other)
            else:
                if get_remainder:
                    return (Mono(coefficient=my_evaluation / other), 0)
                return Mono(coefficient=my_evaluation / other)
        if isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if other_evaluation == 0:
                raise ZeroDivisionError(f'Cannot divide a polynomial by the expression {other} which evaluates to 0')
            if None not in (my_evaluation, other_evaluation):
                if get_remainder:
                    return (Mono(coefficient=my_evaluation / other_evaluation), 0)
                return Mono(coefficient=my_evaluation / other_evaluation)
            elif my_evaluation is None and other_evaluation is not None:
                if get_remainder:
                    return (self.divide_by_number(other_evaluation), 0)
                return self.divide_by_number(other_evaluation)
            elif isinstance(other, (Poly, Mono)):
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
        first, second = (self._expressions[0], self._expressions[1])
        if_number1, if_number2 = (first.variables_dict is None, second.variables_dict is None)
        for k in range(power + 1):
            comb_result = comb(power, k)
            first_power, second_power = (power - k, k)
            if if_number1:
                first_expression = Mono(first.coefficient ** first_power * comb_result)
            else:
                first_expression = Mono(first.coefficient ** first_power * comb_result, {key: value * first_power for key, value in first.variables_dict.items()})
            if if_number2:
                second_expression = Mono(second.coefficient ** second_power)
            else:
                second_expression = Mono(second.coefficient ** second_power, {key: value * second_power for key, value in second.variables_dict.items()})
            expressions.append(first_expression * second_expression)
        return Poly(expressions)

    def __pow__(self, power: Union[int, float, IExpression, str], modulo=None):
        if isinstance(power, float):
            power = int(power)
        if not isinstance(power, int):
            if isinstance(power, str):
                power = Poly(power)
            if isinstance(power, Mono):
                if power.variables_dict is not None:
                    raise ValueError('Cannot perform power with an algebraic exponent on polynomials')
                else:
                    power = power.coefficient
            elif isinstance(power, Poly):
                if len(power._expressions) == 1 and power._expressions[0].variables_dict is None:
                    power = power._expressions[0].coefficient
                else:
                    raise ValueError('Cannot perform power with an algebraic exponent')
        if isinstance(power, float) and power.is_integer():
            power = int(power)
        if power == 0:
            return Poly(1)
        elif power == 1:
            return Poly(self._expressions)
        my_evaluation = self.try_evaluate()
        if my_evaluation is not None:
            return Mono(coefficient=my_evaluation ** power)
        if len(self.expressions) == 2:
            return self.__calc_binomial(power)
        else:
            new_expression = self
            for i in range(power - 1):
                new_expression *= self
            return new_expression

    def __rpow__(self, other, power, modulo=None):
        if len(self._expressions) == 1 and self._expressions[0].variables_dict is None:
            if not isinstance(other, (Mono, Poly)):
                other = Poly(other)
            return other.__pow__(self)
        else:
            return Exponent(self, other)

    def __ipow__(self, other):
        self._expressions = self.__pow__(other)._expressions
        return self

    def is_number(self):
        return all((expression.is_number() for expression in self._expressions))

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
            my_evaluation = self.try_evaluate()
            other_evaluation = other if isinstance(other, (int, float)) else other.try_evaluate()
            if my_evaluation is not None and other_evaluation is not None:
                return my_evaluation == other_evaluation
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
            if my_num_of_variables == 0:
                if len(self._expressions) != len(other._expressions):
                    return False
                return self._expressions[0] == other._expressions[0]
            elif my_num_of_variables == 1:
                return self._expressions == other._expressions
            else:
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
                raise ValueError('Try using partial_derivative() for expression with more than variable')
        derived_expression = Poly([expression.derivative() for expression in self._expressions])
        derived_expression.simplify()
        derived_expression.sort()
        return derived_expression

    def is_empty(self) -> bool:
        return not self._expressions

    def partial_derivative(self, variables: Iterable):
        derived_expression = Poly((monomial.partial_derivative(variables) for monomial in self._expressions))
        derived_expression.simplify()
        derived_expression.sort()
        if derived_expression.is_empty():
            return Mono(0)
        return derived_expression

    def integral(self, add_c=False):
        for expression in self.expressions:
            if expression.variables_dict is not None and len(expression.variables_dict) > 1:
                raise ValueError(f'IExpression {expression.__str__()}: Can only compute the integral with one variable or less ( got {len(expression.variables_dict)}')
        result = Poly([expression.integral() for expression in self.expressions])
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
        if number_of_variables == 0:
            num_of_expressions = len(self._expressions)
            if num_of_expressions == 0:
                return None
            elif num_of_expressions == 1:
                return [self._expressions[0].coefficient]
            elif num_of_expressions > 1:
                self.simplify()
                return [self._expressions[0].coefficient]
        elif number_of_variables > 1:
            raise ValueError(f'Can only fetch the coefficients of a polynomial with 1 variable, found {number_of_variables}')
        sorted_exprs = sorted_expressions([expression for expression in self._expressions if expression.variables_dict is not None])
        biggest_power = max_power(sorted_exprs)
        coefficients = [0] * (int(fetch_power(biggest_power.variables_dict)) + 1)
        for index, sorted_expression in enumerate(sorted_exprs):
            coefficients[len(coefficients) - int(fetch_power(sorted_expression.variables_dict)) - 1] = sorted_expression.coefficient
        free_numbers = [expression for expression in self._expressions if expression.variables_dict is None]
        free_number = sum((expression.coefficient for expression in free_numbers))
        coefficients[-1] = free_number
        return coefficients

    def assign(self, **kwargs):
        for expression in self._expressions:
            expression.assign(**kwargs)
        self.simplify()

    def discriminant(self):
        my_coefficients = self.coefficients()
        length = len(my_coefficients)
        if length == 1:
            return 0
        elif length == 2:
            return 1
        elif length == 3:
            return my_coefficients[1] ** 2 - 4 * my_coefficients[0] * my_coefficients[2]
        elif length == 4:
            if my_coefficients[0] == 1 and my_coefficients[1] == 0:
                return -4 * my_coefficients[2] ** 3 - 27 * my_coefficients[3] ** 2
        elif length == 5:
            a, b, c, d, e = (my_coefficients[0], my_coefficients[1], my_coefficients[2], my_coefficients[3], my_coefficients[4])
            result = 256 * a ** 3 * e ** 3 - 192 * a ** 2 * b * d * e ** 2 - 128 * a ** 2 * c ** 2 * e ** 2 + 144 * a ** 2 * c * d ** 2 * e
            result += -27 * a ** 2 * d ** 4 + 144 * a * b ** 2 * c * e ** 2 - 6 * a * b ** 2 * d ** 2 * e - 80 * a * b * c ** 2 * d * e
            result += 18 * a * b * c * d ** 3 + 16 * a * c ** 4 * e - 4 * a * c ** 3 * d ** 2 - 27 * b ** 4 * e ** 2 + 18 * b ** 3 * c * d * e
            result += -4 * b ** 3 * d ** 3 - 4 * b ** 2 * c ** 3 * e + b ** 2 * c ** 2 * d ** 2
            return result
        else:
            raise ValueError('Discriminants are not supported yet for polynomials with degree 5 or more')

    def roots(self, epsilon=1e-06, nmax=10000):
        from kiwicalc.equations.single import solve_polynomial
        my_coefficients = self.coefficients()
        return solve_polynomial(my_coefficients, epsilon, nmax)

    def real_roots(self):
        pass

    def extremums(self):
        from kiwicalc.geometry.points import Point2D
        from kiwicalc.geometry.point_collections import PointCollection

        num_of_variables = len(self.variables)
        if num_of_variables == 0:
            return None
        elif num_of_variables == 1:
            my_lambda = self.to_lambda()
            my_derivative = self.derivative()
            if my_derivative.is_number():
                return None
            derivative_roots = my_derivative.roots(nmax=1000)
            myRoots = [Point2D(root.real, my_lambda(root.real)) for root in derivative_roots if root.imag <= 1e-05]
            return PointCollection(myRoots)

    def extremums_axes(self, get_derivative=False):
        num_of_variables = len(self.variables)
        if num_of_variables == 0:
            return None
        elif num_of_variables == 1:
            my_derivative = self.derivative()
            if my_derivative.is_number():
                return None
            my_roots = [root.real for root in my_derivative.roots(nmax=1000) if root.imag <= 1e-05]
            my_roots.sort()
            if get_derivative:
                return (my_roots, my_derivative)
            return my_roots

    def up_and_down(self):
        axes_and_derivative = self.extremums_axes(get_derivative=True)
        if axes_and_derivative is None:
            extremums_axes, my_derivative = ([], self.derivative())
        else:
            extremums_axes, my_derivative = axes_and_derivative
        return self.__up_and_down(extremums_axes, my_derivative)

    def __up_and_down(self, extremums_axes, my_derivative=None):
        x = Var('x')
        coefficients = self.coefficients()
        num_of_coefficients: int = len(coefficients)
        if num_of_coefficients == 1:
            return (None, None)
        elif num_of_coefficients == 2:
            if coefficients[0] > 0:
                return (Range(expression=x, limits=(-np.inf, np.inf), operators=(LESS_THAN, LESS_THAN)), None)
            elif coefficients[0] < 0:
                return (None, Range(expression=x, limits=(-np.inf, np.inf), operators=(LESS_THAN, LESS_THAN)))
        elif num_of_coefficients == 3:
            first = Range(expression=x, limits=(-np.inf, extremums_axes[0]), operators=(LESS_THAN, LESS_THAN))
            second = Range(expression=x, limits=(extremums_axes[0], np.inf), operators=(LESS_THAN, LESS_THAN))
            if coefficients[0] > 0:
                return (second, first)
            return (first, second)
        else:
            num_of_extremums = len(extremums_axes)
            if num_of_extremums == 0:
                print("didn't find any extremums...")
            if my_derivative is None:
                my_derivative = self.derivative()
            derivative_lambda = my_derivative.to_lambda()
            up_ranges, down_ranges = ([], [])
            derivatives_values = [derivative_lambda(random.uniform(extremums_axes[i], extremums_axes[i + 1])) for i in range(num_of_extremums - 1)]
            before_value = derivative_lambda(extremums_axes[0] - 1)
            after_value = derivative_lambda(extremums_axes[-1] + 1)
            derivatives_values.append(after_value)
            if before_value > 0:
                up_ranges.append(Range(expression=x, limits=(-np.inf, extremums_axes[0]), operators=(LESS_THAN, LESS_THAN)))
            elif before_value < 0:
                down_ranges.append(Range(expression=x, limits=(-np.inf, extremums_axes[0]), operators=(LESS_THAN, LESS_THAN)))
            else:
                pass
            for i in range(num_of_extremums - 1):
                random_value = derivative_lambda(random.uniform(extremums_axes[i], extremums_axes[i + 1]))
                if random_value > 0:
                    up_ranges.append(Range(expression=x, limits=(extremums_axes[i], extremums_axes[i + 1]), operators=(LESS_THAN, LESS_THAN)))
                elif random_value < 0:
                    down_ranges.append(Range(expression=x, limits=(extremums_axes[i], extremums_axes[i + 1]), operators=(LESS_THAN, LESS_THAN)))
                else:
                    pass
            if after_value > 0:
                up_ranges.append(Range(expression=x, limits=(extremums_axes[-1], np.inf), operators=(LESS_THAN, LESS_THAN)))
            elif after_value < 0:
                down_ranges.append(Range(expression=x, limits=(extremums_axes[-1], np.inf), operators=(LESS_THAN, LESS_THAN)))
            else:
                pass
            return (RangeOR(up_ranges), RangeOR(down_ranges))

    def data(self, no_roots=False):
        """
        Get a dictionary that provides information about the polynomial: string, degree, coefficients, roots, extremums, up and down.
        """
        from kiwicalc.geometry.points import Point2D

        variables = self.variables
        num_of_variables = len(variables)
        my_eval = self.try_evaluate()
        if num_of_variables == 0:
            return {'string': self.__str__(), 'variables': variables, 'plotDimensions': num_of_variables + 1, 'coefficients': [my_eval], 'roots': np.inf if my_eval == 0 else [], 'y_intersection': my_eval, 'extremums': [], 'up': None, 'down': None}
        elif num_of_variables == 1:
            extremums_axes = self.extremums_axes()
            my_lambda = self.to_lambda()
            my_extremums = [Point2D(x, my_lambda(x)) for x in extremums_axes]
            my_derivative = self.derivative()
            up, down = self.__up_and_down(extremums_axes, my_derivative=my_derivative)
            return {'string': self.__str__(), 'variables': variables, 'plotDimensions': num_of_variables + 1, 'coefficients': self.coefficients(), 'roots': [] if no_roots else self.roots(), 'y_intersection': my_lambda(0), 'derivative': my_derivative, 'extremums': my_extremums, 'up': up.__str__(), 'down': down.__str__()}
        else:
            return {'string': self.__str__(), 'variables': variables, 'plotDimensions': num_of_variables + 1}

    def get_report(self, colored=True) -> str:
        if colored:
            accumulator = ''
            for key, value in self.data().items():
                accumulator += f'\x1b[93m{key}\x1b[0m: {value.__str__()}\n'
            return accumulator
        return '\n'.join((value.__str__() for key, value in self.data().items()))

    def _format_report(self, data):
        accumulator = [f"Function: {data['string']}"]
        variables = ', '.join((variable for variable in data['variables']))
        accumulator.append(f'variables: {variables}')
        if len(data['variables']) == 1:
            accumulator.append(f"coefficients: {data['coefficients']}")
            roots = list(data['roots'])
            for index, root in enumerate(roots):
                if isinstance(root, complex):
                    if root.imag < 0.0001:
                        roots[index] = round(root.real, 3)
            roots_string = ', '.join((str(root) for root in roots))
            accumulator.append(f'roots: {roots_string}')
            accumulator.append(f"Intersection with the y axis: {round(data['y_intersection'], 3)}")
            accumulator.append(f"Derivative: {data['derivative']}")
            accumulator.append('Extremums Points:' + ','.join((extremum.__str__() for extremum in data['extremums'])))
            accumulator.append(f"Up: {data['up']}")
            accumulator.append(f"Down: {data['down']}")
        return accumulator

    def print_report(self):
        print(self.get_report())

    def export_report(self, path: str, delete_image=True):
        c = Canvas(path)
        c.setFont('Helvetica-Bold', 22)
        c.drawString(50, 800, 'Function Report')
        textobject = c.beginText(2 * cm, 26 * cm)
        c.setFont('Helvetica', 16)
        data = self.data()
        variables = ','.join(data['variables'])
        for line in self._format_report(data):
            textobject.textLine(line)
            textobject.textLine('')
        c.drawText(textobject)
        if len(variables) == 1:
            plot_function(f"f({variables}) = {data['string']}", show=False)
        else:
            plot_function_3d(f"f({variables}) = {data['string']}", show=False)
        plt.savefig('tempPlot1146151.png')
        if len(data['variables']) == 1 or len(data['variables']) == 2:
            if len(data['variables']) == 1:
                c.drawInlineImage('tempPlot1146151.png', 50, -215, width=500, preserveAspectRatio=True)
            elif len(data['variables']) == 2:
                c.drawInlineImage('tempPlot1146151.png', 50, 200, width=500, preserveAspectRatio=True)
            if delete_image:
                os.remove('tempPlot1146151.png')
        c.showPage()
        c.save()

    def durand_kerner(self):
        from kiwicalc.numeric.roots import durand_kerner

        return durand_kerner(self.to_lambda(), self.coefficients())

    def ostrowski(self, initial_value: float, epsilon=1e-05, nmax=10000):
        from kiwicalc.numeric.roots import ostrowski_method

        return ostrowski_method(self.to_lambda(), self.derivative().to_lambda(), initial_value, epsilon, nmax)

    def laguerres(self, x0: float, epsilon=1e-05, nmax=100000):
        from kiwicalc.numeric.roots import laguerre_method

        my_derivative = self.derivative()
        second_derivative_expression = my_derivative.derivative()
        second_derivative = (
            (lambda _, constant=second_derivative_expression.try_evaluate(): constant)
            if second_derivative_expression.num_of_variables == 0
            else second_derivative_expression.to_lambda()
        )
        degree = len(self.coefficients()) - 1
        return laguerre_method(self.to_lambda(), my_derivative.to_lambda(), second_derivative, x0, degree, epsilon, nmax)

    def halleys(self, initial_value=0, epsilon=1e-05, nmax=10000):
        """
        Halley's method is a root finding method developed by Edmond Halley for functions with continuous second
        derivatives and a single variable.
        :param initial_value:
        :param epsilon:
        :return:
        """
        from kiwicalc.numeric.roots import halleys_method

        f_0 = self
        f_1 = f_0.derivative()
        f_2 = f_1.derivative()
        f_0 = self.to_lambda()
        f_1 = f_1.to_lambda()
        f_2 = (lambda _, constant=f_2.try_evaluate(): constant) if f_2.num_of_variables == 0 else f_2.to_lambda()
        return halleys_method(f_0, f_1, f_2, initial_value, epsilon, nmax)

    def newton(self, initial_value=0, epsilon=1e-05, nmax=10000):
        from kiwicalc.numeric.roots import newton_raphson

        return newton_raphson(self.to_lambda(), self.derivative().to_lambda(), initial_value, epsilon, nmax)

    def __str__(self):
        if len(self._expressions) == 1:
            return self._expressions[0].__str__()
        accumulator = ''
        for index, expression in enumerate(self._expressions):
            accumulator += '+' if expression.coefficient >= 0 and index > 0 else ''
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
    def from_json(json_content: str):
        parsed_dictionary = json.loads(json_content)
        if parsed_dictionary['type'].strip().lower() != 'poly':
            return ValueError(f"Invalid type: {parsed_dictionary['type']}. Expected 'Poly'. ")
        return Poly((Mono.from_dict(mono_dict) for mono_dict in parsed_dictionary['data']))

    @staticmethod
    def import_json(path: str):
        with open(path) as json_file:
            return Poly.from_json(json_file.read())

    def python_syntax(self):
        accumulator = ''
        for index, expression in enumerate(self._expressions):
            accumulator += '+' if expression.coefficient >= 0 and index > 0 else ''
            accumulator += expression.python_syntax()
        return accumulator

    def __fetch_variables_set(self) -> set:
        return {json.dumps(mono_expression.variables_dict) for mono_expression in self._expressions}

    def simplify(self):
        """ simplifying a polynomial"""
        if len(self._expressions) == 0:
            self._expressions = [Mono(0)]
            return
        different_variables: set = self.__fetch_variables_set()
        if '{}' in different_variables:
            different_variables.remove('{}')
            different_variables.add('null')
        new_expressions = []
        for variable_dictionary in different_variables:
            if variable_dictionary == 'null':
                same_variables = [expression for expression in self._expressions if json.dumps(expression.variables_dict) in ('null', '{}')]
            else:
                same_variables = [expression for expression in self._expressions if json.dumps(expression.variables_dict) == variable_dictionary]
            if len(same_variables) > 1:
                assert all((isinstance(same_variable.coefficient, (int, float)) for same_variable in same_variables)), 'Bug detected..'
                coefficients_sum: float = sum((same_variable.coefficient for same_variable in same_variables))
                if coefficients_sum != 0:
                    new_expressions.append(Mono(coefficient=coefficients_sum, variables_dict=same_variables[0].variables_dict))
            elif len(same_variables) == 1:
                if same_variables[0].coefficient != 0:
                    new_expressions.append(same_variables[0])
        self._expressions = new_expressions
        self.sort()

    def sorted_expressions_list(self) -> list:
        """

        :return:
        """
        sorted_exprs = sorted_expressions([expression for expression in self._expressions if expression.variables_dict not in (None, {})])
        free_number = sum((expression.coefficient for expression in self._expressions if expression.variables_dict in (None, {})))
        if free_number != 0:
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
            if mono_expression.contains_variable(variable):
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
            if len(self.expressions) < len(item._expressions):
                return False
            return all((item in self.expressions for item in item._expressions))
        else:
            raise TypeError(f'Poly.__contains__(): unexpected type {type(item)}, expected types: int,float,str,Mono,Poly')

    def __copy__(self):
        return Poly([expression.__copy__() for expression in self._expressions])

    def to_lambda(self):
        """ Returns a lambda expression from the Polynomial"""
        return to_lambda(self.__str__(), self.variables)

    def plot(self, start: float=-10, stop: float=10, step: float=0.01, ymin: float=-10, ymax: float=10, text=None, show_axis=True, show=True, fig=None, ax=None, formatText=True, values=None):
        from kiwicalc.plotting.plots import plot_function, plot_function_3d
        lambda_expression = self.to_lambda()
        num_of_variables = self.num_of_variables
        if text is None:
            text = self.__str__()
        if num_of_variables == 0:
            raise ValueError('Cannot plot a polynomial with 0 variables_dict')
        elif num_of_variables == 1:
            plot_function(lambda_expression, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=text, show_axis=show_axis, show=show, fig=fig, ax=ax, formatText=formatText, values=values)
        elif num_of_variables == 2:
            plot_function_3d(lambda_expression, start=start, stop=stop, step=step)
        else:
            raise ValueError('Cannot plot a function with more than two variables_dict (As for this version)')

    def to_Function(self):
        from kiwicalc.functions.function import Function

        return Function(self.__str__())

    def gcd(self):
        """Greatest common divisor of the expressions: for example, for the expression 3x^2 and 6x,
        the result would be 3x"""
        gcd_coefficient = abs(gcd([expression.coefficient for expression in self._expressions]))
        if any((not expression.variables_dict for expression in self._expressions)):
            return Mono(gcd_coefficient)
        gcd_algebraic = Mono(gcd_coefficient)
        my_variables = self.variables
        for variable in my_variables:
            if all((variable in expression.variables_dict for expression in self._expressions)):
                powers = [expression.variables_dict[variable] for expression in self._expressions]
                if gcd_algebraic.variables_dict is not None:
                    gcd_algebraic.variables_dict = {**gcd_algebraic.variables_dict, **{variable: min(powers)}}
                else:
                    gcd_algebraic.variables_dict = {variable: min(powers)}
        return gcd_algebraic

    def divide_by_gcd(self):
        return self.__itruediv__(self.gcd())

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
    return (new_list, result)


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

