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

from kiwicalc.expressions.mono import Mono
from kiwicalc.expressions.poly import Poly

class Factorial(IExpression, IPlottable, IScatterable):
    __slots__ = ['_coefficient', '_expression', '_power']

    def __init__(self, expression: Optional[Union[IExpression, int, float, str]], coefficient: Union[IExpression, int, float]=Mono(1), power: Union[IExpression, int, float]=Mono(1), dtype=''):
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
        return {'type': 'Factorial', 'coefficient': self._coefficient.to_dict(), 'expression': self._expression.to_dict(), 'power': self._power.to_dict()}

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

    def __itruediv__(self, other: Union[IExpression, int, float]) -> 'Optional[Union[Fraction,Factorial]]':
        if other == 0:
            raise ZeroDivisionError('Cannot divide a factorial expression by 0')
        if other == self._expression:
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
                if other_evaluation == 0:
                    raise ZeroDivisionError('Cannot divide a factorial expression by 0')
                if other_evaluation == self._expression:
                    self._expression -= 1
                    self.simplify()
                    return self
                else:
                    self._coefficient /= other
                    self.simplify()
                    return self
            elif isinstance(other, Factorial):
                if self._expression == other._expression:
                    self._coefficient /= other._coefficient
                    self._power -= other._power
                    self.simplify()
            else:
                return Fraction(self, other)
        else:
            raise TypeError(f"Invalid type for dividing factorials: '{type(other)}'")

    def __rtruediv__(self, other: Union[int, float, IExpression]):
        my_evaluation = self.try_evaluate()
        if my_evaluation == 0:
            raise ZeroDivisionError('Cannot divide by 0: Tried to divide by a Factorial expression that evaluatesto zero')
        if isinstance(other, (int, float)):
            if my_evaluation is not None:
                return Mono(other / my_evaluation)
            return Fraction(other, self)
        elif isinstance(other, IExpression):
            return other.__truediv__(self)
        else:
            raise TypeError('Invalid type for dividing an expression by a Factorial object.')

    def __ipow__(self, other: Union[int, float, IExpression]):
        self._power *= other
        return self

    def __pow__(self, power):
        return self.__copy__().__ipow__(power)

    def __neg__(self):
        if self._expression is None:
            return Factorial(coefficient=self._coefficient.__neg__(), expression=None, power=Mono(1))
        return Factorial(coefficient=self._coefficient.__neg__(), expression=self._expression.__neg__(), power=self._power.__neg__())

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
        if None not in (coefficient_evaluation, expression_evaluation, power_evaluation):
            if expression_evaluation < 0:
                return None
            if expression_evaluation == 0:
                my_factorial = 1
            elif expression_evaluation == int(expression_evaluation):
                my_factorial = factorial(int(expression_evaluation))
            else:
                my_factorial = gamma(expression_evaluation) * expression_evaluation
            return coefficient_evaluation * my_factorial ** power_evaluation
        elif power_evaluation == 0 and coefficient_evaluation is not None:
            return coefficient_evaluation
        return None

    def simplify(self):
        """Try to simplify the factorial expression"""
        self._coefficient.simplify()
        if self._coefficient == 0:
            self._expression = None
            self._power = Mono(1)

    def python_syntax(self):
        if self._expression is None:
            return f'{self._coefficient.python_syntax()}'
        return f'{self._coefficient} * factorial({self._expression.python_syntax()}) ** {self._power.python_syntax()}'

    def __str__(self):
        if self._expression is None:
            return f'{self._coefficient}'
        coefficient_str = format_coefficient(self._coefficient)
        if coefficient_str not in ('', '-'):
            coefficient_str += '*'
        power_str = f'**{self._power.__str__()}' if self._power != 1 else ''
        inside_str = self._expression.__str__()
        if '-' in inside_str or '+' in inside_str or '*' in inside_str or ('/' in inside_str):
            inside_str = f'({inside_str})'
        expression_str = f'({inside_str}!)' if coefficient_str != '' else f'{inside_str}!'
        if power_str == '':
            return f'{coefficient_str}{expression_str}'
        return f'{coefficient_str}({expression_str}){power_str}'

    def __copy__(self):
        return Factorial(coefficient=self._coefficient, expression=self._expression, power=self._power)

    def __eq__(self, other: Union[IExpression, int, float]):
        if other is None:
            return False
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            if my_evaluation is not None:
                return my_evaluation == other
            return False
        elif isinstance(other, IExpression):
            if my_evaluation is not None:
                other_evaluation = other.try_evaluate()
                return other_evaluation is not None and my_evaluation == other_evaluation
            if isinstance(other, Factorial):
                return self._coefficient == other._coefficient and self._expression == other._expression and (self._power == other._power)
            return False
        else:
            raise TypeError(f"Invalid type '{type(other)}' for equating with a Factorial expression.")

    def __ne__(self, other: Union[IExpression, int, float]):
        return not self.__eq__(other)

class Abs(IExpression, IPlottable, IScatterable):
    """A class for representing expressions with absolute values. For instance, Abs(x) is the same as |x|."""
    __slots__ = ['_coefficient', '_expression', '_power']

    def __init__(self, expression: Union[IExpression, int, float], power: Union[int, float, IExpression]=1, coefficient: Union[int, float, IExpression]=1, gen_copies=True):
        if isinstance(expression, (int, float)):
            self._expression = Mono(expression)
        elif isinstance(expression, IExpression):
            self._expression = expression.__copy__() if gen_copies else expression
        else:
            raise TypeError(f'Invalid type {type(expression)} for inner expression when creating an Abs object.')
        if isinstance(power, (int, float)):
            self._power = Mono(power)
        elif isinstance(power, IExpression):
            self._power = power.__copy__() if gen_copies else power
        else:
            raise TypeError(f"Invalid type {type(power)} for 'power' argument when creating a new Abs object.")
        if isinstance(coefficient, (int, float)):
            self._coefficient = Mono(coefficient)
        elif isinstance(coefficient, IExpression):
            self._coefficient = coefficient.__copy__() if gen_copies else coefficient
        else:
            raise TypeError(f"Invalid type {type(coefficient)} for 'coefficient' argument when creating a new Abs object.")

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
        return {'type': 'Abs', 'coefficient': self._coefficient.to_dict(), 'expression': self._expression.to_dict(), 'power': self._power.to_dict()}

    @staticmethod
    def from_dict(given_dict: dict):
        expression_obj = create_from_dict(given_dict['expression'])
        coefficient_obj = create_from_dict(given_dict['coefficient'])
        power_obj = create_from_dict(given_dict['power'])
        return Abs(expression=expression_obj, power=power_obj, coefficient=coefficient_obj)

    def __add_or_sub(self, other, operation: str='+'):
        if isinstance(other, (int, float)):
            my_evaluation = self.try_evaluate()
            if my_evaluation is not None:
                if operation == '+':
                    return Mono(my_evaluation + other)
                else:
                    return Mono(my_evaluation - other)
            elif operation == '+':
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
            raise TypeError(f" Invalid type: {type(other)} when multiplying an Abs object. Expected types 'int', 'float', 'IExpression'.")
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
        if self._expression == other._expression or self._expression == -other._expression:
            self._power += other._power
            self._coefficient *= other._coefficient
            return self
        return ExpressionMul((self, other))

    def __itruediv__(self, other: Union[int, float, IExpression]):
        if not isinstance(other, (int, float, IExpression)):
            raise TypeError(f" Invalid type: {type(other)} when dividing an Abs object. Expected types 'int', 'float', 'IExpression'.")
        if other == 0:
            raise ValueError(f'Cannot divide an Abs object by 0.')
        if isinstance(other, (int, float)):
            self._coefficient /= other
            return self
        my_evaluation, other_evaluation = (self.try_evaluate(), other.try_evaluate())
        if other_evaluation == 0:
            raise ValueError(f'Cannot divide an Abs object by 0.')
        if None not in (my_evaluation, other_evaluation):
            return Mono(my_evaluation / other_evaluation)
        if other_evaluation is not None:
            self._coefficient /= other
            return self
        if not isinstance(other, Abs):
            self._coefficient /= other
            return self
        if self._expression == other._expression or self._expression == -other._expression:
            power_difference = self._power - other._power
            difference_evaluation = power_difference.try_evaluate()
            if difference_evaluation is None:
                self._coefficient /= other._coefficient
                return Exponent(coefficient=self._coefficient, base=self._expression, power=power_difference)
            elif difference_evaluation > 0:
                self._power = Mono(difference_evaluation)
                self._coefficient /= other._coefficient
                return Abs(coefficient=self._coefficient, power=self._power, expression=self._expression, gen_copies=False)
            elif difference_evaluation == 0:
                return self._coefficient
            else:
                return Fraction(self._coefficient / other._coefficient, Abs(self._expression, -difference_evaluation))
        return Fraction(self, other)

    def __ipow__(self, power: Union[int, float, IExpression]):
        if not isinstance(power, (int, float, IExpression)):
            raise TypeError(f"Invalid type: {type(power)} when raising by a power an Abs object. Expected types 'int', 'float', 'IExpression'.")
        if isinstance(power, (int, float)):
            self._coefficient **= power
            self._power *= power
            return self
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
                expression_evaluation = self._expression.try_evaluate()
                if expression_evaluation is not None:
                    return self._coefficient * abs(expression_evaluation) ** self._power == other
            return False
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def derivative(self, get_derivatives=False):
        warnings.warn('Derivatives are still experimental, and might not work for other algebraic expressionsrather than polynomials.')
        num_of_variables = len(self.variables)
        if num_of_variables == 0:
            return lambda x: self.try_evaluate()
        assert num_of_variables == 1, 'Use partial derivatives of expressions with several variables.'
        positive_expression = self._coefficient * self._expression ** self._power
        try:
            positive_derivative = positive_expression.derivative()
        except:
            return None
        negative_derivative = -positive_derivative
        if get_derivatives:
            return (positive_derivative, negative_derivative)
        positive_derivative, negative_derivative = (positive_derivative.to_lambda(), positive_derivative.to_lambda())
        return lambda x: positive_derivative(x) if x > 0 else negative_derivative(x) if x < 0 else 0

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
            return '0'
        if self._power == 0:
            return self._coefficient.__str__()
        elif self._power == 1:
            power_string = ''
        else:
            power_string = f'**{self._power.python_syntax()}'
        if self._coefficient == 1:
            coefficient_string = f''
        elif self._coefficient == -1:
            coefficient_string = f'-'
        else:
            coefficient_string = f'{self._coefficient.__str__()}*'
        return f'{coefficient_string}|{self._expression}|{power_string}'

    def __copy__(self):
        return Abs(expression=self._expression, power=self._power, coefficient=self._coefficient, gen_copies=True)

class Exponent(IExpression):
    """
    This class enables you to represent expressions such as x^x, e^x, (3x)^sin(x), etc.
    """
    __slots__ = ['_coefficient', '_base', '_power']

    def __init__(self, base: Union[IExpression, float], power: Union[IExpression, float, int], coefficient: Optional[Union[int, float, IExpression]]=None, gen_copies=True):
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
            elif not isinstance(other, Exponent):
                if operation == '+':
                    return ExpressionSum((self, other))
                return ExpressionSum((self, -other))
            elif self._power == other._power and self._base == other._base:
                if operation == '+':
                    self._coefficient += other._coefficient
                else:
                    self._coefficient -= other._coefficient
                return self
            elif False:
                pass
            else:
                if operation == '+':
                    return ExpressionSum((self, other))
                return ExpressionSum((self, -other))

    def __iadd__(self, other: 'Union[IExpression, int, float]'):
        return self.__add_or_sub(other, operation='+')

    def __isub__(self, other):
        return self.__add_or_sub(other, operation='-')

    def __imul__(self, other: Union[int, float, IExpression]):
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
            elif isinstance(other, Exponent):
                if self._power == other._power:
                    self._base *= other._base
                    return self
                elif False:
                    pass
                else:
                    return ExpressionMul((self, other))

    def __mul__(self, other: Union[int, float, IExpression]):
        return self.__copy__().__imul__(other)

    def multiply_by_number(self, number: Union[int, float]):
        self._coefficient *= number

    def divide_by_number(self, number: Union[int, float]):
        if number == 0:
            raise ZeroDivisionError('Cannot divide an expression by 0')
        self._coefficient /= number

    def __itruediv__(self, other: Union[int, float, IExpression]):
        return Fraction(self, other)

    def __ipow__(self, other: Union[IExpression, int, float]):
        if other == 0:
            return self._coefficient.__copy__()
        self._power *= other
        return self

    def __pow__(self, power: Union[int, float, IExpression], modulo=None):
        return self.__copy__().__ipow__(power)

    def __neg__(self):
        copy_of_self = self.__copy__()
        copy_of_self._coefficient *= -1
        return copy_of_self

    def to_dict(self):
        return {'type': 'Exponent', 'coefficient': self._coefficient.to_dict(), 'base': self._base.to_dict(), 'power': self._power.to_dict()}

    @staticmethod
    def from_dict(given_dict: dict):
        base_obj = create_from_dict(given_dict['base'])
        coefficient_obj = create_from_dict(given_dict['coefficient'])
        power_obj = create_from_dict(given_dict['power'])
        return Exponent(base=base_obj, power=power_obj, coefficient=coefficient_obj)

    def derivative(self):
        my_variables = self.variables
        variables_length = len(my_variables)
        if variables_length == 0:
            return Mono(0)
        elif variables_length == 1:
            coefficient_eval = self._coefficient.try_evaluate()
            base_eval = self._base.try_evaluate()
            power_eval = self._power.try_evaluate()
            if None not in (coefficient_eval, base_eval, power_eval) or coefficient_eval == 0 or base_eval == 0:
                return Mono(0)
            if power_eval is not None and power_eval == 0:
                return self._coefficient.derivative()
            if coefficient_eval is not None:
                if power_eval is not None:
                    expression = (coefficient_eval * self._base ** power_eval).derivative()
                    if hasattr(expression, 'derivative'):
                        return expression.derivative()
                    warnings.warn("This kind of derivative isn't supported yet...")
                    return None
                elif base_eval is not None:
                    if base_eval < 0:
                        warnings.warn(f'The derivative of this expression is undefined')
                        return None
                    return self * self._coefficient.derivative() * ln(base_eval)
        else:
            raise ValueError('For derivatives with more than 1 variable, use partial derivatives')

    @property
    def variables(self):
        my_variables = self._coefficient.variables
        my_variables.update(self._base.variables)
        my_variables.update(self._power.variables)
        return my_variables

    def partial_derivative(self):
        raise NotImplementedError('This feature is not supported yet. Stay tuned for the next versions.')

    def integral(self):
        raise NotImplementedError('This feature is not supported yet. Stay tuned for the next versions.')

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

    def simplify(self) -> None:
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
        if power_evaluation == 0:
            return coefficient_evaluation
        base_evaluation = self._base.try_evaluate()
        if base_evaluation is None:
            return None
        return coefficient_evaluation * base_evaluation ** power_evaluation

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
                if isinstance(other, Exponent):
                    equal_coefficients = self._coefficient == other._coefficient
                    equal_bases = self._base == other._base
                    equal_powers = self._power == other._power
                    if equal_coefficients and equal_bases and equal_powers:
                        return True
                    if equal_bases:
                        if self._coefficient == self._base and other._coefficient == 1 and self._power + 1 == other._power:
                            return True
                        if other._coefficient == other._base and self._coefficient == 1 and other._power + 1 == self._power:
                            return True
                    return False
            else:
                return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __copy__(self):
        return Exponent(base=self._base, power=self._power, coefficient=self._coefficient)

    def __str__(self):
        if self._coefficient == 0:
            return '0'
        if self._power == 0:
            return self._coefficient.__str__()
        if self._coefficient == 1:
            coefficient_str = ''
        elif self._coefficient == -1:
            coefficient_str = '-'
        else:
            coefficient_str = f'{self._coefficient.__str__()}*'
        base_string, power_string = (apply_parenthesis(self._base.__str__()), apply_parenthesis(self._power.__str__()))
        return f'{coefficient_str}{base_string}^{power_string}'


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

