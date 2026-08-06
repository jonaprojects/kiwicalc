from __future__ import annotations
import random
from typing import Union, Tuple, List, Optional, Any, Callable, Dict, Set, Iterable
import numpy as np
from numpy.linalg import LinAlgError, solve

from kiwicalc.core.utils import clean_from_spaces, format_linear_dict, format_poly_dict, round_decimal
from kiwicalc.parsing.parse_expression import poly_from_str, extract_variables_from_expression
from kiwicalc.parsing.parse_equation import (
    ParseEquation, extract_dict_from_equation, linear_expression_to_dict,
    subtract_dicts, get_equation_variables, simplify_expression, equation_to_one_side
)
from kiwicalc.equations.single import LinearEquation, PolyEquation
from kiwicalc.linalg.matrix import Matrix, generate_jacobian

class LinearSystem:
    """
    This class represents a system of linear __equations.
    It solves them via a simple implementation of the Gaussian Elimination technique.
    """

    def __init__(self, equations: Iterable, variables: Iterable=None):
        """
        Creating a new equation system

        :param equations: An iterable collection of equations. Each equation in the collection can be of type
        string or Equation
        :param variables:(Optional) an iterable collection of strings that be converted to a list.
        Each item represents a variable in the equations. For example, ('x','y','z').
        """
        self.__equations, self.__variables = ([], list(variables) if variables is not None else [])
        self.__variables_dict = dict()
        for equation in equations:
            if isinstance(equation, str):
                self.__equations.append(LinearEquation(equation))
            elif isinstance(equation, LinearEquation):
                self.__equations.append(equation)
            else:
                raise TypeError

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
                if variable not in variables_dict and variable != 'number':
                    variables_dict[variable] = 0
        variables_dict['number'] = 0
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
            side1, side2 = (equation.equation[:equal_index], equation.equation[equal_index + 1:])
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

def solve_linear_system(equations, variables=None):
    """ Solve a system of linear equations via Guass-Elimination Method with matrices"""
    if not variables:
        variables = set()
        for equation in equations:
            variables.update(extract_variables_from_expression(equation))
    values_matrix = []
    for equation in equations:
        equal_index = equation.find('=')
        side1, side2 = (equation[:equal_index], equation[equal_index + 1:])
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

def solve_poly_system(equations: 'Union[Iterable[Union[str,Poly,Mono]],Iterable[Union[str, Poly, Mono]]]', initial_vals: dict=None, epsilon: float=1e-05, nmax: int=10000, show_steps=False):
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
    variables, initial_values = (list(initial_vals.keys()), Matrix(matrix=initial_vals.values()))
    polynomials = [poly_from_str(equation_to_one_side(equation)) if isinstance(equation, str) else equation for equation in equations]
    jacobian_matrix = Matrix(matrix=generate_jacobian(polynomials, variables))
    current_values_matrix = Matrix(matrix=[[current_value] for current_value in list(initial_vals.values())])
    for i in range(nmax):
        assignment_dictionary = dict(zip(variables, [row[0] for row in current_values_matrix.matrix]))
        assigned_jacobian = jacobian_matrix.mapped_matrix(lambda polynomial: polynomial.when(**assignment_dictionary).try_evaluate())
        jacobian_inverse = assigned_jacobian.inverse()
        assigned_polynomials = Matrix(matrix=[[polynomial.when(**assignment_dictionary).try_evaluate()] for polynomial in polynomials])
        if all((abs(row[0]) < epsilon for row in assigned_polynomials.matrix)):
            return {variables[index]: row[0] for index, row in enumerate(current_values_matrix)}
        interval_matrix = jacobian_inverse @ assigned_polynomials
        current_values_matrix -= interval_matrix

def random_linear_system(variables, solutions_range: Tuple[int, int]=(-10, 10), coefficients_range: Tuple[int, int]=(-10, 10), digits_after=0, get_solutions=False):
    from kiwicalc.parsing.parse_expression import ParseExpression
    equations = []
    num_of_equations = len(variables)
    solutions = [round(random.uniform(solutions_range[0], solutions_range[1]), digits_after) for _ in range(num_of_equations)]
    for _ in range(num_of_equations):
        coefficients_dict = dict()
        equation_sum = 0
        for index, variable in enumerate(variables):
            random_coefficient = random.randint(coefficients_range[0], coefficients_range[1])
            coefficients_dict[variable] = [random_coefficient]
            equation_sum += random_coefficient * solutions[index]
        free_number = random.randint(coefficients_range[0], coefficients_range[1])
        equation_sum += free_number
        other_side_dict = {variable: [] for variable in variables}
        num_of_operations = random.randint(2, 5)
        for _ in range(num_of_operations):
            operation_index = random.randint(0, 1)
            if operation_index == 0:
                random_variable = random.choice(variables)
                random_coefficient = random.randint(coefficients_range[0], coefficients_range[1])
                coefficients_dict[random_variable].append(random_coefficient)
                other_side_dict[random_variable].append(random_coefficient)
            else:
                random_num = random.randint(1, 3)
                for variable, coefficients in coefficients_dict.items():
                    for index in range(len(coefficients)):
                        coefficients[index] *= random_num
                free_number *= random_num
                for variable, coefficients in other_side_dict.items():
                    for index in range(len(coefficients)):
                        coefficients[index] *= random_num
                equation_sum *= random_num
        equations.append(f'{ParseExpression.unparse_linear(coefficients_dict, free_number)}={ParseExpression.unparse_linear(other_side_dict, equation_sum)}')
    if get_solutions:
        return (equations, solutions)
    return equations

def random_poly_system(variables):
    pass
