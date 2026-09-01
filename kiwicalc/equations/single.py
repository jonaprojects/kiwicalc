from __future__ import annotations
import math
import random
import cmath
import operator
import json
import warnings
from itertools import permutations
from functools import reduce
from abc import ABC, abstractmethod
from math import sqrt
from typing import Union, Tuple, List, Optional, Any, Callable, Dict, Set, Iterable
import numpy as np
import matplotlib.pyplot as plt

from kiwicalc.core.constants import allowed_characters, e, pi
from kiwicalc.core.interfaces import IExpression
from kiwicalc.core.ranges import Range
from kiwicalc.core.utils import (
    clean_from_spaces, extract_coefficient, format_coefficient,
    format_free_number, is_number, contains_from_list, round_decimal,
    format_linear_dict
)
from kiwicalc.parsing.parse_equation import (
    ParseEquation, extract_dict_from_equation, linear_expression_to_dict,
    subtract_dicts, get_equation_variables, simplify_expression,
    coefficients_to_expressions
)
from kiwicalc.parsing.parse_expression import split_expression, extract_variables_from_expression, poly_from_str, ParseExpression
from kiwicalc.expressions.poly import Poly
from kiwicalc.expressions.mono import Mono
from kiwicalc.expressions.roots import Sqrt
from kiwicalc.numeric.roots import (
    aberth_method, bairstow_method, bisection_method,
    newton_raphson, halleys_method, secant_method,
    extract_possible_solutions, __find_solutions
)

def solve_quadratic_from_str(expression, real=False, strict_syntax=False):
    if isinstance(expression, str):
        variables = get_equation_variables(expression)
        if len(variables) == 0:
            return tuple()
        elif len(variables) == 1:
            coefficients = ParseEquation.parse_quadratic(expression, strict_syntax=strict_syntax)
            solve_method = solve_quadratic_real if real else solve_quadratic
            return solve_method(*coefficients)
        else:
            raise ValueError("Can't solve a quadratic equation with more than 1 variable")

def solve_quadratic(a: Union[str, float], b: float=None, c: float=None) -> tuple:
    """ Solves a quadratic equation using computations of complex numbers ( utilizing the cmath library)"""
    if isinstance(a, str):
        return solve_quadratic_from_str(a)
    discriminant = b ** 2 - 4 * a * c
    return ((-b + cmath.sqrt(discriminant)) / (2 * a), (-b - cmath.sqrt(discriminant)) / (2 * a))

def solve_quadratic_real(a: Union[str, float], b: float, c: float) -> Optional[Union[Tuple[float, float], float]]:
    """returns onlu the real solutions of the quadratic equation"""
    if isinstance(a, str):
        return solve_quadratic_from_str(a, real=True)
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return None
    if discriminant == 0:
        return (-b + sqrt(discriminant)) / (2 * a)
    return ((-b + sqrt(discriminant)) / (2 * a), (-b - sqrt(discriminant)) / (2 * a))

def solve_quadratic_params(a: 'Union[IExpression, int, float,str]', b: 'Union[IExpression, int, float]', c: 'Union[IExpression,int,float]'):
    if isinstance(a, str):
        print('need to be implemented')
    if all((isinstance(coefficient, (int, float)) for coefficient in (a, b, c))):
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
    if all((isinstance(coefficient, (int, float)) for coefficient in (a, b, c))):
        return ((-b + sqrt(b ** 2 - 4 * a * c)) / (2 * a), (-b - sqrt(b ** 2 - 4 * a * c)) / (2 * a))
    else:
        discriminant_root = Sqrt(b ** 2 - 4 * a * c)
        return ((-b + discriminant_root) / (2 * a), (-b + -discriminant_root) / (2 * a))

def solve_cubic(a: float, b: float, c: float, d: float):
    """ Given the real coefficients of a cubic equation, this method will return the solutions"""
    if a == 0:
        return solve_quadratic(b, c, d)
    delta0 = b * b - 3 * a * c
    delta1 = 2 * pow(b, 3) - 9 * a * b * c + 27 * a * a * d
    deepest_root = cmath.sqrt(pow(delta1, 2) - 4 * pow(delta0, 3))
    C = (0.5 * (delta1 + deepest_root)) ** (1.0 / 3)
    if C == 0:
        C = (0.5 * (delta1 - deepest_root)) ** (1.0 / 3)
    if C == 0:
        return [0]
    roots = []
    root_of_unity = complex(-0.5, sqrt(3) / 2)
    for k in range(3):
        root = -((b + C + delta0 / C) / (3 * a))
        roots.append(root)
        C *= root_of_unity
    return list({complex(round_decimal(root.real), round_decimal(root.imag)) for root in roots})

def solve_cubic_real(a: float, b: float, c: float, d: float):
    roots = solve_cubic(a, b, c, d)
    if not roots:
        return []
    return [root.real for root in roots if abs(root.imag) < 1e-05]

def solve_quartic(a: float, b: float, c: float, d: float, e: float):
    if a == 0:
        return solve_cubic(b, c, d, e)
    if a != 1:
        b /= a
        c /= a
        d /= a
        e /= a
        a = 1
    f = c - 3 * b ** 2 / 8
    g = d + b ** 3 / 8 - b * c / 2
    h = e - 3 * b ** 4 / 256 + b ** 2 * c / 16 - b * d / 4
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
    p, q = (cmath.sqrt(non_zero_roots[0]), cmath.sqrt(non_zero_roots[1]))
    r = -g / (8 * p * q)
    s = b / (4 * a)
    sol1, sol2, sol3, sol4 = (p + q + r - s, p - q - r - s, -p + q - r - s, -p - q + r - s)
    return list({sol1, sol2, sol3, sol4})

def solve_polynomial(coefficients, epsilon: float=1e-06, nmax: int=10000):
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
    polynomial_obj = Poly(coefficients_to_expressions(coefficients))
    poly_derivative = polynomial_obj.derivative().to_lambda()
    return aberth_method(polynomial_obj.to_lambda(), poly_derivative, coefficients, epsilon, nmax)

def solve_poly_by_factoring(coefficients):
    """
    This method attempts to find the roots of a polynomial by synthetic division.
    It won't always return all the solutions, but it is faster than many numerical root finding algorithms, and might
    even be preferable in some cases over these algorithms.
    """
    if coefficients is None:
        return {}
    if len(coefficients) == 3:
        return solve_quadratic_real(coefficients[0], coefficients[1], coefficients[2])
    most_significant = coefficients[0]
    free_number = coefficients[-1]
    possible_solutions = extract_possible_solutions(most_significant, free_number)
    print(possible_solutions)
    solutions = __find_solutions(coefficients, possible_solutions)
    return solutions

def solve_linear(equation: str, variables=None, get_dict=False, get_json=False):
    if variables is None:
        variables = extract_dict_from_equation(equation)
    first_side, second_side = equation.split('=')
    first_dict = simplify_expression(expression=first_side, variables=variables)
    second_dict = simplify_expression(expression=second_side, variables=variables)
    result_dict = {key: value for key, value in subtract_dicts(dict1=first_dict, dict2=second_dict).items() if key}
    if len(result_dict) < 2:
        return None
    elif len(result_dict) == 2:
        if list(result_dict.values())[0] == 0:
            if list(result_dict.values())[1] == 0:
                return np.inf
            return None
        solution = -result_dict['number'] / list(result_dict.values())[0]
        if get_dict:
            return {list(variables.keys())[0]: solution}
        elif get_json:
            return json.dumps({'variable': list(variables.keys())[0], 'result': solution})
        return solution
    elif len(result_dict) > 2:
        raise ValueError('Invalid equation caused an unexpected error')

def solve_linear_inequality(equation: str, variables=None):
    sign = next((candidate for candidate in ('<=', '>=', '<', '>') if candidate in equation), None)
    if sign is None:
        raise ValueError('Invalid equation')
    expressions = equation.split(sign)
    if len(expressions) != 2:
        raise ValueError(f'Invalid equation')
    if variables is None:
        variables = extract_dict_from_equation(equation, delimiter=sign)
    first_side, second_side = expressions
    first_dict = simplify_expression(first_side, variables)
    second_dict = simplify_expression(second_side, variables)
    result_dict = subtract_dicts(first_dict, second_dict)
    first_key = list(result_dict.keys())[0]
    first_value = list(result_dict.values())[0]
    number_value = result_dict['number']
    return f'{first_key}{sign}{round_decimal(-number_value / first_value)}'

def random_linear(coefs_range=(-15, 15), digits_after: int=0, variable='x', get_solution: bool=False, get_coefficients: bool=False):
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
            return (f'{a_str}{variable}{b_str}', round_decimal(-b / a), (a, b))
        return (f'{a_str}{variable}{b_str}', round_decimal(-b / a))
    elif get_coefficients:
        return (f'{a_str}{variable}{b_str}', (a, b))
    return f'{a_str}{variable}{b_str}'

def random_polynomial(degree: int=None, solutions_range=(-5, 5), digits_after=0, variable='x', python_syntax=False, get_solutions=False):
    if degree is None:
        degree = random.randint(2, 9)
    a = round_decimal(round(random.uniform(solutions_range[0], solutions_range[1]), digits_after))
    while a == 0:
        a = round_decimal(round(random.uniform(solutions_range[0], solutions_range[1]), digits_after))
    accumulator = [f'{format_coefficient(a)}x**{degree}'] if python_syntax else [f'{format_coefficient(a)}x^{degree}']
    solutions = {round_decimal(round(random.uniform(solutions_range[0], solutions_range[1]), digits_after)) for _ in range(degree)}
    permutations_length = 1
    for i in range(degree):
        current_permutations = set((tuple(sorted(per)) for per in permutations(solutions, permutations_length)))
        current_sum = 0
        for permutation in current_permutations:
            current_sum += reduce(operator.mul, permutation)
        if current_sum != 0:
            current_power = degree - permutations_length
            coefficient = format_coefficient(round_decimal(current_sum * a)) if current_power != 0 else f'{round_decimal(current_sum * a)}'
            if coefficient != '' and coefficient[0] not in ('+', '-'):
                coefficient = f'+{coefficient}'
            if current_power == 0:
                accumulator.append(f'{coefficient}')
            elif current_power == 1:
                accumulator.append(f'{coefficient}{variable}')
            else:
                accumulator.append(f'{coefficient}{variable}^{current_power}')
        permutations_length += 1
    equation = ''.join(accumulator)
    if get_solutions:
        return (equation, [-solution for solution in solutions])
    return equation

def random_polynomial2(degree: int, values=(-15, 15), digits_after=0, variable='x', python_syntax=False):
    a = round_decimal(round(random.uniform(values[0], values[1]), digits_after))
    while a == 0:
        a = round_decimal(round(random.uniform(values[0], values[1]), digits_after))
    accumulator = []
    while a == 0:
        a = round_decimal(round(random.uniform(values[0], values[1]), digits_after))
    accumulator.append(f'{format_coefficient(a)}{variable}^{degree}')
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
            accumulator.append(f'{coef_str}{variable}{power_str}')
    m = round_decimal(round(random.uniform(values[0], values[1]), digits_after))
    accumulator.append(f'+{round_decimal(m)}' if m > 0 else f'{m}') if m != 0 else ''
    return ''.join(accumulator)

class Equation(ABC):

    def __init__(self, equation: str, variables: Iterable=None, calc_now: bool=False):
        """The base function of creating a new Equation"""
        self._equation = clean_from_spaces(equation)
        if variables is None:
            self._variables = get_equation_variables(equation)
            self._variables_dict = self._extract_variables()
            try:
                index = self._variables.index('number')
                del self._variables[index]
            except ValueError:
                pass
        else:
            self._variables = list(variables)
            self._variables_dict = {variable: 0 for variable in variables}
            self._variables_dict['number'] = 0
        self._solution = None
        if calc_now:
            self._solution = self.solve()

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
        return self._equation[:self._equation.rfind('=')]

    @property
    def second_side(self):
        return self._equation[self._equation.rfind('=') + 1:]

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
        first_side, second_side = (self.equation[:equal_index], self.equation[equal_index + 1:])
        return LinearEquation(f'{second_side}={first_side}')

    @abstractmethod
    def __repr__(self):
        pass

    def __str__(self):
        return self._equation

class LinearEquation(Equation):

    def __init__(self, equation: str, variables=None, calc_now=False):
        super().__init__(equation, variables, calc_now)
        try:
            index = self._variables.index('number')
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
        self._equation = f'{format_linear_dict(result_dict, round_coefficients=round_coefficients)} = {round_decimal(-num)}'

    def __format_expressions(self, expressions):
        accumulator = ''
        for index, expression in enumerate(expressions):
            if expression.variables_dict in ({}, None):
                if expression.coefficient > 0 and index > 0:
                    accumulator += f'\x1b[93m+{expression}\x1b[0m '
                else:
                    accumulator += f'\x1b[96m{expression}\x1b[0m '
            elif expression.coefficient != 0:
                if expression.coefficient > 0 and index > 0:
                    accumulator += f'\x1b[93m+{expression}\x1b[0m '
                else:
                    accumulator += f'\x1b[96m{expression}\x1b[0m '
        return accumulator

    def show_steps(self):
        variables = self.variables_dict
        if len(variables) > 2:
            raise NotImplementedError(f'This feature is currently only available with 1-variable equation, got {len(variables)}')
        if len(variables) < 2:
            first_side, second_side = (self.first_side, self.second_side)
            accumulator = '\x1b[1m1. First step: recognize that this equation only contains free numbers,and hence either it has no solutions, or it has infinite solutions \x1b[0m\n'
            accumulator += f"\x1b[93m{first_side.replace('+', ' +').replace('-', ' -')}\x1b[0m"
            accumulator += ' = '
            accumulator += f"\x1b[93m{second_side.replace('+', ' +').replace('-', ' -')}\x1b[0m\n"
            accumulator += '\x1b[1m2. Second Step: sum all the numbers in both sides\x1b[0m\n'
            first_expression = simplify_expression(expression=first_side, variables=variables)
            second_expression = simplify_expression(expression=first_side, variables=variables)
            accumulator += f"\x1b[93m{first_expression['number']}\x1b[0m"
            accumulator += ' = '
            accumulator += f"\x1b[93m{second_expression['number']}\x1b[0m\n"
            if first_expression['number'] == second_expression['number']:
                accumulator += '\x1b[1mFinal Step:  The expression above is always true, and hence there are infinite solutions to the equation.\x1b[0m\n'
                self._solution = 'Infinite'
            else:
                accumulator += '\x1b[1mFinal Step: The expression above is always false, and hence there are infinite solutions to the equation.\x1b[0m\n'
                self._solution = None
            return accumulator
        first_variable = list(self.variables_dict.keys())[0]
        first_side, second_side = (self.first_side, self.second_side)
        first_expressions = poly_from_str(first_side, get_list=True)
        second_expressions = poly_from_str(second_side, get_list=True)
        accumulator = f'\x1b[1m1. First Step : Identify the free numbers and the expressions with {first_variable} in each side\x1b[0m\n'
        accumulator += self.__format_expressions(first_expressions) + ' = ' + self.__format_expressions(second_expressions) + '\n'
        accumulator += "\x1b[1m2. Second step: Sum the matching groups in each side ( if it's possible )\x1b[0m\n"
        free_sum1, variables_sum1 = (0, 0)
        for mono_expression in first_expressions:
            if mono_expression.is_number():
                free_sum1 += mono_expression.coefficient
            else:
                variables_sum1 += mono_expression.coefficient
        accumulator += f'\x1b[96m{variables_sum1}{first_variable}\x1b[0m '
        if free_sum1 > 0:
            accumulator += f'+\x1b[93m{free_sum1}\x1b[0m'
        elif free_sum1 != 0:
            accumulator += f'\x1b[93m{free_sum1}\x1b[0m'
        accumulator += ' = '
        free_sum2, variables_sum2 = (0, 0)
        for mono_expression in second_expressions:
            if mono_expression.is_number():
                free_sum2 += mono_expression.coefficient
            else:
                variables_sum2 += mono_expression.coefficient
        accumulator += f'\x1b[96m{variables_sum2}{first_variable}\x1b[0m '
        if free_sum1 > 0:
            accumulator += f'+\x1b[93m{free_sum2}\x1b[0m'
        elif free_sum1 != 0:
            accumulator += f'\x1b[93m{free_sum2}\x1b[0m'
        accumulator += '\n'
        accumulator += '\x1b[1m3. Third Step: Move all the variables to the right, and the free numbers to the left \x1b[0m\n'
        variable_difference = variables_sum1 - variables_sum2
        if variable_difference == 0:
            accumulator += '0'
        else:
            accumulator += f'\x1b[96m{variable_difference}{first_variable}\x1b[0m'
        accumulator += ' = '
        free_sum_difference = free_sum2 - free_sum1
        accumulator += f'\x1b[93m{free_sum_difference}\x1b[0m\n'
        if variable_difference == 0:
            if free_sum_difference == 0:
                accumulator += '\x1b[1m3 Therefore, there are infinite solutions !\x1b[0m\n'
                self._solution = 'Infinite'
                return accumulator
            else:
                accumulator += '\x1b[1m3 Therefore, there is no solution to the equation !\x1b[0m\n'
                self._solution = None
                return accumulator
        accumulator += '\x1b[1m4. Final step: divide both sides by the coefficient of the right side \x1b[0m\n'
        accumulator += f'\x1b[96m{first_variable}\x1b[0m = \x1b[93m{free_sum_difference / variable_difference}\x1b[0m'
        return accumulator

    def plot_solution(self, start: float=-10, stop: float=10, step: float=0.01, ymin: float=-10, ymax: float=10, show_axis=True, show=True, title: str=None, with_legend=True):
        """
        Plot the solution of the linear equation, as the intersection of two linear functions ( in each side )
        """
        # Keep these imports local to avoid a module-level cycle between equations,
        # functions, and plotting.
        from kiwicalc.functions.function import Function
        from kiwicalc.plotting.plots import plot_functions

        if is_number(self.first_side):
            first_function = Function(f'f(x) = {self.first_side}')
        else:
            first_function = Function(self.first_side)
        if is_number(self.second_side):
            second_function = Function(f'f(x) = {self.second_side}')
        else:
            second_function = Function(self.second_side)
        if title is None:
            title = f'{self.first_side}={self.second_side}'
        plot_functions([first_function, second_function], start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, show_axis=show_axis, show=False, title=title, with_legend=with_legend)
        x = self.solution
        if x is not None and (not isinstance(x, str)):
            y = first_function(x)
            plt.scatter([x], [y], color='red')
            if show:
                plt.show()
            return (x, y)
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
        accumulator = ''
        if not variable or not isinstance(variable, str):
            variable = random.choice(allowed_characters)
        num_of_items = random.randint(items_range[0], items_range[1])
        for i in range(num_of_items):
            if random.randint(0, 1):
                accumulator = ''.join((accumulator, '-'))
            elif accumulator:
                accumulator = ''.join((accumulator, '+'))
            coefficient = random.randint(values[0], values[1])
            if coefficient != 0:
                if random.randint(0, 1):
                    accumulator += f'{format_coefficient(coefficient)}{variable}'
                else:
                    accumulator += f'{coefficient}'
        return accumulator if accumulator != '' else '0'

    @staticmethod
    def random_equation(values=(1, 20), items_per_side=(4, 7), digits_after=2, get_solution=False, variable=None, get_variable=False):
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
        for i in range(1000):
            if len(solution_string[solution_string.find('.') + 1:]) <= digits_after:
                if get_solution:
                    if get_variable:
                        return (equation, solution, variable)
                    return (equation, solution)
                if get_variable:
                    return (equation, variable)
                return equation
            equation = f'{LinearEquation.random_expression(values=values, items_range=items_per_side, variable=variable)} '
            equation += f'= {LinearEquation.random_expression(values=values, items_range=items_per_side, variable=variable)}'
            solution = LinearEquation(equation).solve()
            solution_string = str(solution)
        if get_solution:
            if get_variable:
                return (equation, solution, variable)
            return (equation, solution)
        if get_variable:
            return (equation, variable)
        return equation

    @staticmethod
    def random_worksheet(path, title='Equation Worksheet', num_of_equations=10, values=(1, 20), items_per_side=(4, 8), after_point=2, get_solutions=False) -> bool:
        """
        Generates a PDF page with random __equations
        :return:
        """
        equations = [LinearEquation.random_equation(values, items_per_side, after_point, get_solutions) for _ in range(num_of_equations)]
        return create_pdf(path=path, title=title, lines=equations)

    @staticmethod
    def random_worksheets(path: str, num_of_pages: int=2, equations_per_page=20, values=(1, 20), items_per_side=(4, 8), after_point=1, get_solutions=False, titles=None):
        if get_solutions:
            lines = []
            for i in range(num_of_pages):
                equations, solutions = ([], [])
                for j in range(equations_per_page):
                    equation, solution, variable = LinearEquation.random_equation(values=values, items_per_side=items_per_side, digits_after=after_point, get_solution=True, get_variable=True)
                    equations.append(f'{j + 1}. {equation}')
                    solutions.append(f'{j + 1}. {variable} = {solution}')
                lines.extend((equations, solutions))
            if titles is None:
                titles = ['Worksheet - Linear Equations', 'Solutions'] * num_of_pages
            create_pages(path=path, num_of_pages=num_of_pages * 2, titles=titles, lines=lines)
        else:
            lines = []
            for i in range(num_of_pages):
                equations = []
                for j in range(equations_per_page):
                    equation = LinearEquation.random_equation(values=values, items_per_side=items_per_side, digits_after=after_point, get_solution=False, get_variable=False)
                    equations.append(f'{j + 1}. {equation}')
                lines.append(equations)
            if titles is None:
                titles = ['Worksheet - Linear Equations'] * num_of_pages
            create_pages(path=path, num_of_pages=num_of_pages, titles=titles, lines=lines)

    @staticmethod
    def adjusted_worksheet(title='Equation Worksheet', equations=()) -> bool:
        """
        Creates a user-defined PDF worksheet file.
        :param title: the title of the page
        :param equations: the __equations to print out
        :return: returns True if the creation is successful, else False.
        """
        return create_pdf('test', title=title, lines=equations)

    @staticmethod
    def manual_worksheet() -> bool:
        """
        Allows the user to create a PDF worksheet file manually.
        :return: True, if the creation is successful, else False
        """
        try:
            name, title, equations = (input("Worksheet's Name:  "), input("Worksheet's Title:  "), [])
            print("Enter your equations. To stop, type 'stop' ")
            i = 1
            equation = input(f'{i}.  ')
            i += 1
            while equation.lower() != 'stop':
                equations.append(equation)
                equation = input(f'{i}.  ')
        except Exception as e:
            warnings.warn(f"Couldn't create the pdf file due to a {e.__class__} error")
            return False
        return LinearEquation.adjusted_worksheet(title=title, equations=equations)

    def __str__(self):
        return f'{self.equation}'

    def __repr__(self):
        return f'Equation({self.equation})'

    def __copy__(self):
        return LinearEquation(self._equation)

class QuadraticEquation(Equation):

    def __init__(self, equation: str, variables: Optional[Iterable[str]]=None, strict_syntax=False):
        self.__strict_syntax = strict_syntax
        super().__init__(equation, variables)

    def _extract_variables(self):
        return ParseExpression.parse_quadratic(self.first_side, self._variables, strict_syntax=self.__strict_syntax)

    def simplified_str(self) -> str:
        if self.num_of_variables != 1:
            raise ValueError('You can only simplify quadratic equations with 1 variable in the current version')
        my_coefficients = self.coefficients()
        return ParseExpression.coefficients_to_str(my_coefficients, variable=self._variables[0])

    def solve(self, mode='complex'):
        """Solve the quadratic equation"""
        num_of_variables = len(self._variables)
        if num_of_variables == 0:
            pass
        elif num_of_variables == 1:
            x = self._variables[0]
            a, b, c = (self._variables_dict[x][0], self._variables_dict[x][1], self._variables_dict['free'])
            mode = mode.lower()
            if mode == 'complex':
                return solve_quadratic(a, b, c)
            elif mode == 'real':
                return solve_quadratic_real(a, b, c)
            elif mode == 'parametric':
                return solve_quadratic_params(a, b, c)
        warnings.warn(f'Cannot solve quadratic equations with more than 1 variable, but found {num_of_variables}')
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
    def random(values=(-15, 15), digits_after: int=0, variable: str='x', strict_syntax=True, get_solutions=False):
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
            b, c = (round_decimal(round((m + n) * a, digits_after)), round_decimal(round(m * n * a, digits_after)))
            a_str = format_coefficient(a)
            b_str = (f'+{b}' if b > 0 else f'{b}') if b != 0 else ''
            if b_str != '':
                if b_str == '1':
                    b_str = f'+{variable}'
                elif b_str == '-1':
                    b_str = f'-{variable}'
                else:
                    b_str += variable
            c_str = (f'+{round_decimal(c)}' if c > 0 else f'{c}') if c != 0 else ''
            equation = f'{a_str}{variable}^2{b_str}{c_str} = 0'
            if get_solutions:
                return (equation, (-m, -n))
            return equation
        else:
            raise NotImplementedError('Only strict_syntax=True is available at the moment.')

    @staticmethod
    def random_worksheet(path=None, title='Quadratic Equations Worksheet', num_of_equations=20, solutions_range=(-15, 15), digits_after: int=0, get_solutions=True):
        lines = []
        if get_solutions:
            equations, solutions = ([], [])
            for i in range(num_of_equations):
                equ, sol = QuadraticEquation.random(values=solutions_range, digits_after=digits_after, get_solutions=True)
                equations.append(f'{i + 1}. {equ}')
                joined_solutions = ', '.join(sol)
                solutions.append(f'{i + 1}. {joined_solutions}')
            lines.extend((equations, solutions))
        else:
            equations = []
            for i in range(num_of_equations):
                equ = QuadraticEquation.random(values=solutions_range, digits_after=digits_after, get_solutions=False)
                equations.append(f'{i + 1}. {equ}')
            lines.append(equations)
        create_pdf(path=path, title=title, lines=lines)

    @staticmethod
    def random_worksheets(path=None, num_of_pages=2, equations_per_page=20, titles=None, solutions_range=(-15, 15), digits_after: int=0, get_solutions=False):
        if titles is None:
            if get_solutions:
                titles = ['Quadratic Equations Worksheet', 'Solutions'] * num_of_pages
            else:
                titles = ['Quadratic Equations Worksheet'] * num_of_pages
        lines = []
        if get_solutions:
            for i in range(num_of_pages):
                equations, solutions = ([], [])
                for j in range(equations_per_page):
                    equ, sol = QuadraticEquation.random(values=solutions_range, digits_after=digits_after, get_solutions=True)
                    equations.append(f'{i + 1}. {equ}')
                    joined_solutions = ', '.join(sol)
                    solutions.append(f'{i + 1}. {joined_solutions}')
                lines.extend((equations, solutions))
            create_pages(path=path, num_of_pages=num_of_pages * 2, titles=titles, lines=lines)
        else:
            for i in range(num_of_pages):
                equations = []
                for j in range(equations_per_page):
                    equ = QuadraticEquation.random(values=solutions_range, digits_after=digits_after, get_solutions=False)
                    equations.append(f'{i + 1}. {equ}')
                lines.append(equations)
            create_pages(path=path, num_of_pages=num_of_pages, titles=titles, lines=lines)

    def __repr__(self):
        return f'QuadraticEquation({self._equation}, variables={self._variables})'

    def __copy__(self):
        return QuadraticEquation(equation=self._equation, strict_syntax=self.__strict_syntax)

class CubicEquation(Equation):

    def __init__(self, equation: str, variables: Iterable[Optional[str]]=None, strict_syntax: bool=False):
        self.__strict_syntax = strict_syntax
        super().__init__(equation, variables)

    def _extract_variables(self):
        return ParseExpression.parse_cubic(self.first_side, self._variables, strict_syntax=self.__strict_syntax)

    def solve(self):
        a, b, c = (self._variables_dict['x'][0], self._variables_dict['x'][1], self._variables_dict['x'][2])
        d = self._variables_dict['free']
        return solve_cubic(a, b, c, d)

    def coefficients(self):
        return self._variables_dict['x'] + [self._variables_dict['free']]

    @staticmethod
    def random(solutions_range: Tuple[float, float]=(-15, 15), digits_after: int=0, variable='x', get_solutions=False):
        result = random_polynomial(degree=3, solutions_range=solutions_range, digits_after=digits_after, variable=variable, get_solutions=get_solutions)
        if isinstance(result, str):
            return result + ' = 0'
        else:
            return (result[0] + '= 0', result[1])

    @staticmethod
    def random_worksheet(path=None, title=' Cubic Equations Worksheet', num_of_equations=20, solutions_range=(-15, 15), digits_after: int=0, get_solutions=False):
        PolyEquation.random_worksheet(path=path, title=title, num_of_equations=num_of_equations, degrees_range=(3, 3), solutions_range=solutions_range, digits_after=digits_after, get_solutions=get_solutions)

    @staticmethod
    def random_worksheets(path=None, num_of_pages=2, equations_per_page=20, titles=None, solutions_range=(-15, 15), digits_after: int=0, get_solutions=False):
        if titles is None:
            if get_solutions:
                titles = ['Cubic Equations Worksheet', 'Solutions'] * num_of_pages
            else:
                titles = ['Cubic Equations Worksheet'] * num_of_pages
        PolyEquation.random_worksheets(path=path, num_of_pages=num_of_pages, titles=titles, equations_per_page=equations_per_page, degrees_range=(3, 3), solutions_range=solutions_range, digits_after=digits_after, get_solutions=get_solutions)

    def __repr__(self):
        return f'CubicEquation({self._equation}, variables={self._variables})'

    def __copy__(self):
        return CubicEquation(equation=self._equation, strict_syntax=self.__strict_syntax)

class QuarticEquation(Equation):

    def __init__(self, equation: str, variables: Iterable[Optional[str]]=None, strict_syntax=False):
        self.__strict_syntax = strict_syntax
        super().__init__(equation, variables)

    def _extract_variables(self):
        return ParseExpression.parse_quartic(self.first_side, self._variables, strict_syntax=self.__strict_syntax)

    def solve(self):
        a, b, c = (self._variables_dict['x'][0], self._variables_dict['x'][1], self._variables_dict['x'][2])
        d, e = (self._variables_dict['x'][3], self._variables_dict['free'])
        return solve_quartic(a, b, c, d, e)

    def coefficients(self):
        return self._variables_dict['x'] + [self._variables_dict['free']]

    @staticmethod
    def random(solutions_range: Tuple[float, float]=(-15, 15), digits_after: int=0, variable='x', get_solutions=False):
        result = random_polynomial(degree=4, solutions_range=solutions_range, digits_after=digits_after, variable=variable, get_solutions=get_solutions)
        if isinstance(result, str):
            return result + ' = 0'
        else:
            return (result[0] + '= 0', result[1])

    @staticmethod
    def random_worksheet(path=None, title=' Cubic Equations Worksheet', num_of_equations=20, solutions_range=(-15, 15), digits_after: int=0, get_solutions=False):
        PolyEquation.random_worksheet(path=path, title=title, num_of_equations=num_of_equations, degrees_range=(3, 3), solutions_range=solutions_range, digits_after=digits_after, get_solutions=get_solutions)

    @staticmethod
    def random_worksheets(path=None, num_of_pages=2, equations_per_page=20, titles=None, solutions_range=(-15, 15), digits_after: int=0, get_solutions=False):
        if titles is None:
            if get_solutions:
                titles = ['Quartic Equations Worksheet', 'Solutions'] * num_of_pages
            else:
                titles = ['Quartic Equations Worksheet'] * num_of_pages
        PolyEquation.random_worksheets(path=path, num_of_pages=num_of_pages, titles=titles, equations_per_page=equations_per_page, degrees_range=(4, 4), solutions_range=solutions_range, digits_after=digits_after, get_solutions=get_solutions)

    def __repr__(self):
        return f'QuarticEquation({self._equation}, variables={self._variables})'

    def __copy__(self):
        return QuarticEquation(equation=self._equation, strict_syntax=self.__strict_syntax)

class PolyEquation(Equation):

    def __init__(self, first_side, second_side=None, variables=None):
        self.__solution = None
        if first_side is None:
            raise TypeError('First argument in PolyEquation.__init__() cannot be None. Try using a string, and read the documentation !')
        if second_side is None and isinstance(first_side, str):
            left_side, right_side = first_side.split('=')
            self.__first_expression, self.__second_expression = (Poly(left_side), Poly(right_side))
            equation = first_side
        else:
            try:
                if isinstance(first_side, (Mono, Poly)):
                    self.__first_expression = first_side.__copy__()
                else:
                    self.__first_expression = Poly(first_side)
                if isinstance(second_side, (Mono, Poly)):
                    self.__second_expression = second_side.__copy__()
                else:
                    self.__second_expression = Poly(second_side)
                equation = '='.join((str(first_side), str(second_side)))
            except TypeError:
                raise TypeError(f"Unexpected type{type(first_side)} in PolyEquation.__init__().Couldn't convert the parameter to type str.")
        super().__init__(equation, variables)

    def solve(self):
        return solve_polynomial(self.to_PolyExpr().coefficients())

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

    def plot_solutions(self, start: float=-10, stop: float=10, step: float=0.01, ymin: float=-10, ymax=10, title: str=None, show_axis=True, show=True):
        first_func = Function(self.first_side)
        second_func = Function(self.second_side)
        plot_functions([first_func, second_func], start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title, show_axis=show_axis, show=show)

    @staticmethod
    def __random_monomial(values=(1, 20), power: int=None, variable=None):
        if variable is None:
            variable = 'x'
        coefficient = random.randint(values[0], values[1])
        if coefficient == 0:
            return '0'
        elif coefficient == 1:
            coefficient = ''
        elif coefficient == -1:
            coefficient = '-'
        else:
            coefficient = f'{coefficient}'
        if power == 1:
            return f'{coefficient}{variable}'
        elif power == 0:
            return f'{coefficient}'
        return f'{coefficient}{variable}^{power}'

    @staticmethod
    def random_expression(values=(1, 10), of_order: int=None, variable=None, all_powers=False):
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
            accumulator += f'{random.randint(values[0], values[1])}'
        return accumulator

    @staticmethod
    def random_quadratic(values=(1, 20), variable=None, all_powers=False):
        return f'{PolyEquation.random_expression(values=values, of_order=2, variable=variable, all_powers=all_powers)} = 0'

    @staticmethod
    def random_equation(values=(1, 20), of_order: int=None, variable=None, all_powers=False):
        return f'{PolyEquation.random_expression(values, of_order, variable, all_powers)}={PolyEquation.random_expression(values, of_order, variable, all_powers)}'

    @staticmethod
    def random_worksheet(path=None, title='Equation Worksheet', num_of_equations=20, degrees_range=(2, 5), solutions_range=(-15, 15), digits_after: int=0, get_solutions=False):
        from kiwicalc.pdf.worksheet import create_pdf, create_pages

        if get_solutions:
            expressions = [random_polynomial(random.randint(degrees_range[0], degrees_range[1]), solutions_range=solutions_range, digits_after=digits_after, get_solutions=get_solutions) for _ in range(num_of_equations)]
            equations = [f'{index + 1}. {expression[0]} = 0' for index, expression in enumerate(expressions)]
            solutions = [f'{index + 1}. ' + ','.join([str(solution) for solution in expression[1]]) for index, expression in enumerate(expressions)]
            create_pages(path, 2, ['Polynomial Equations Worksheet', 'Solutions'], [equations, solutions])
        else:
            return create_pdf(path=path, title=title, lines=[f'{random_polynomial(random.randint(degrees_range[0], degrees_range[1]), solutions_range=solutions_range, digits_after=digits_after)} = 0' for _ in range(num_of_equations)])

    @staticmethod
    def random_worksheets(path=None, num_of_pages=2, equations_per_page=20, titles=None, degrees_range=(2, 5), solutions_range=(-15, 15), digits_after: int=0, get_solutions=False):
        from kiwicalc.pdf.worksheet import create_pages

        if get_solutions:
            pages_list = []
            for i in range(num_of_pages):
                expressions = [random_polynomial(random.randint(degrees_range[0], degrees_range[1]), solutions_range=solutions_range, digits_after=digits_after, get_solutions=True) for _ in range(equations_per_page)]
                equations = [f'{index + 1}. {expression[0]} = 0' for index, expression in enumerate(expressions)]
                solutions = [f'{index + 1}. ' + ','.join([str(solution) for solution in expression[1]]) for index, expression in enumerate(expressions)]
                pages_list.append(equations)
                pages_list.append(solutions)
            if titles is None:
                titles = ['Polynomial Equations Worksheet', 'Solutions'] * num_of_pages
            create_pages(path, num_of_pages * 2, titles, pages_list)
        else:
            pages_list = []
            for i in range(num_of_pages):
                expressions = [random_polynomial(random.randint(degrees_range[0], degrees_range[1]), solutions_range=solutions_range, digits_after=digits_after, get_solutions=False) for _ in range(equations_per_page)]
                equations = [f'{index + 1}. {expression[0]} = 0' for index, expression in enumerate(expressions)]
                pages_list.append(equations)
            if titles is None:
                titles = ['Polynomial Equations Worksheet'] * num_of_pages
            create_pages(path, num_of_pages, titles, pages_list)

    def to_PolyExpr(self):
        return self.__first_expression - self.__second_expression

    def __str__(self):
        return self._equation

    def __repr__(self):
        return f'PolyEquation({self._equation})'

    def __copy__(self):
        return PolyEquation(self._equation)
