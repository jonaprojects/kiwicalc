from __future__ import annotations
import random
import warnings
from typing import Union, Tuple, List, Optional, Any, Callable, Dict, Set, Iterable
import numpy as np
from numpy.linalg import LinAlgError, solve

from kiwicalc.core.utils import clean_from_spaces, format_linear_dict, format_poly_dict, round_decimal
from kiwicalc.parsing.parse_expression import poly_from_str, extract_variables_from_expression
from kiwicalc.parsing.parse_equation import (
    ParseEquation, extract_dict_from_equation, linear_expression_to_dict,
    subtract_dicts, get_equation_variables, simplify_expression, _split_equation,
)
from kiwicalc.equations.single import LinearEquation, PolyEquation
from kiwicalc.expressions.poly import Poly
from kiwicalc.expressions.mono import Mono
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
        if self.__variables:
            variable_names = list(self.__variables)
        else:
            inferred = self.__variables_dict if self.__variables_dict else self.__extract_variables()
            variable_names = [name for name in inferred if name != 'number']
        variables = {name: 0 for name in variable_names}
        variables['number'] = 0
        values_matrix = []
        for equation in self.__equations:
            equal_index = equation.equation.find('=')
            side1, side2 = (equation.equation[:equal_index], equation.equation[equal_index + 1:])
            first_dict = simplify_expression(side1, variables)
            second_dict = simplify_expression(side2, variables)
            result_dict = subtract_dicts(second_dict, first_dict)
            values_matrix.append([result_dict[name] for name in variable_names] + [result_dict['number']])
        return values_matrix

    def to_matrix_and_vector(self):
        augmented = self.to_matrix()
        return ([row[:-1] for row in augmented], [-row[-1] for row in augmented])

    def get_solutions(self):
        """
        fetches the solutions
        :return: returns a dictionary that contains the name of each variable, and it's (real) __solution.
        for example: {'x':6,'y':4}
        This comes handy later since you can access simply the solutions.
        """
        return solve_linear_system(
            [equation.equation for equation in self.__equations],
            variables=self.__variables or None,
        )

    def simplify(self):
        for equation in self.__equations:
            equation.simplify()

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
    equations = list(equations)
    if not variables:
        inferred_variables = set()
        for equation in equations:
            inferred_variables.update(extract_variables_from_expression(equation))
        variables = sorted(inferred_variables)
    else:
        variables = list(variables)
    if not variables:
        raise ValueError('A linear system must contain at least one variable')
    if len(set(variables)) != len(variables) or any(not isinstance(variable, str) or not variable for variable in variables):
        raise ValueError('variables must contain distinct, non-empty strings')
    coefficients = []
    constants = []
    for equation in equations:
        if not isinstance(equation, str):
            raise TypeError('Linear-system equations must be strings')
        side1, side2 = _split_equation(equation)
        first_dict = simplify_expression(side1, variables)
        second_dict = simplify_expression(side2, variables)
        result_dict = subtract_dicts(second_dict, first_dict)
        coefficients.append([result_dict[variable] for variable in variables])
        constants.append(-result_dict['number'])
    coefficient_matrix = np.asarray(coefficients, dtype=float)
    constant_vector = np.asarray(constants, dtype=float)
    if coefficient_matrix.ndim != 2 or coefficient_matrix.shape[0] == 0:
        raise ValueError('A linear system must contain at least one equation')
    augmented_matrix = np.column_stack((coefficient_matrix, constant_vector))

    def rank_with_scaled_tolerance(matrix):
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        largest = singular_values[0] if singular_values.size else 0.0
        tolerance = np.finfo(float).eps * max(matrix.shape) * largest
        return int(np.count_nonzero(singular_values > tolerance)), tolerance

    coefficient_rank, rank_tolerance = rank_with_scaled_tolerance(coefficient_matrix)
    augmented_rank, _ = rank_with_scaled_tolerance(augmented_matrix)
    if augmented_rank != coefficient_rank:
        raise ValueError('The linear system is inconsistent and has no solution')
    if coefficient_rank < len(variables):
        raise ValueError('The linear system does not have a unique solution')
    largest_singular = np.linalg.svd(coefficient_matrix, compute_uv=False)[0]
    rcond = rank_tolerance / largest_singular if largest_singular else None
    solution, _, _, _ = np.linalg.lstsq(coefficient_matrix, constant_vector, rcond=rcond)
    residual = coefficient_matrix @ solution - constant_vector
    residual_scale = max(
        1.0,
        np.linalg.norm(coefficient_matrix, ord=np.inf) * np.linalg.norm(solution, ord=np.inf),
        np.linalg.norm(constant_vector, ord=np.inf),
    )
    if np.linalg.norm(residual, ord=np.inf) > 128 * np.finfo(float).eps * residual_scale:
        raise ValueError('The linear system is inconsistent and has no solution')
    return {variable: round_decimal(value) for variable, value in zip(variables, solution)}

def solve_poly_system(equations: 'Union[Iterable[Union[str,Poly,Mono]],Iterable[Union[str, Poly, Mono]]]', initial_vals: dict=None, epsilon: float=1e-05, nmax: int=10000, show_steps=False):
    """
    Find one real solution near the supplied initial approximation using Newton's method.
    :param equations: A collection of equations; each equation must be an equation (of type 'str')  or a polynomial.
    :param initial_vals: A dictionary with some initial approximations to the solutions. For example: {'x':1,'y':2}
    :param epsilon: negligible y value to be considered as 0: for example: 0.001, 0.000001. The smaller epsilon, the more accurate the result and more iterations are required.
    :param nmax: the maximum number of iterations
    :param show_steps: True / False - Whether to show the steps of the solution while solving.
    :return: returns the results of the equation system, a dictionary with variables_dict as keys and their values.
    """
    equations = list(equations)
    if not equations:
        raise ValueError('At least one polynomial equation is required')
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be a positive finite number')
    if isinstance(nmax, bool) or not isinstance(nmax, int) or nmax < 1:
        raise ValueError('nmax must be a positive integer')
    if not isinstance(show_steps, (bool, np.bool_)):
        raise TypeError('show_steps must be a boolean')
    if initial_vals is None:
        variables = set()
        for equation in equations:
            variables.update(extract_variables_from_expression(str(equation)))
        initial_vals = {variable: 0.0 for variable in sorted(variables)}
    else:
        initial_vals = dict(initial_vals)
    if not initial_vals:
        raise ValueError('At least one variable or initial value is required')
    if not all(np.isfinite(value) for value in initial_vals.values()):
        raise ValueError('Initial values must be finite numbers')
    variables = list(initial_vals)
    polynomials = []
    for equation in equations:
        if isinstance(equation, str):
            first_side, second_side = _split_equation(equation)
            polynomials.append(poly_from_str(first_side) - poly_from_str(second_side))
        else:
            polynomials.append(equation)
    if len(polynomials) != len(variables):
        raise ValueError('Newton solving requires one equation per variable')
    if not all(isinstance(polynomial, (Poly, Mono)) for polynomial in polynomials):
        raise TypeError('Equations must be strings, Poly objects, or Mono objects')
    polynomial_variables = set().union(*(polynomial.variables for polynomial in polynomials))
    if polynomial_variables != set(variables):
        raise ValueError('initial_vals keys must match the variables in the polynomial system')
    jacobian = generate_jacobian(polynomials, variables)
    current_values = np.asarray(list(initial_vals.values()), dtype=float)

    def evaluate(point):
        assignment = dict(zip(variables, point))
        values = np.asarray(
            [polynomial.when(**assignment).try_evaluate() for polynomial in polynomials], dtype=float
        )
        if not np.all(np.isfinite(values)):
            raise ValueError('Polynomial system evaluation produced non-finite values')
        return assignment, values

    for i in range(nmax):
        assignment, residuals = evaluate(current_values)
        if show_steps:
            print(f'{i}: {assignment}; residual={residuals.tolist()}')
        residual_norm = np.linalg.norm(residuals, ord=np.inf)
        if residual_norm < epsilon:
            return {variable: current_values[index] for index, variable in enumerate(variables)}
        assigned_jacobian = np.asarray([
            [entry.when(**assignment).try_evaluate() for entry in row]
            for row in jacobian
        ], dtype=float)
        try:
            interval = np.linalg.solve(assigned_jacobian, residuals)
        except np.linalg.LinAlgError:
            interval, _, rank, _ = np.linalg.lstsq(assigned_jacobian, residuals, rcond=None)
            if rank < len(variables) and np.linalg.norm(interval, ord=np.inf) <= np.finfo(float).eps * max(1.0, np.linalg.norm(current_values, ord=np.inf)):
                raise ValueError('Cannot continue: the Jacobian is singular at the current approximation')
        if not np.all(np.isfinite(interval)):
            raise ValueError('Polynomial system iteration produced a non-finite step')

        accepted = False
        step_factor = 1.0
        for _ in range(24):
            candidate = current_values - step_factor * interval
            if not np.all(np.isfinite(candidate)):
                step_factor *= 0.5
                continue
            _, candidate_residuals = evaluate(candidate)
            if np.linalg.norm(candidate_residuals, ord=np.inf) < residual_norm:
                current_values = candidate
                accepted = True
                break
            step_factor *= 0.5
        if not accepted:
            step_size = np.linalg.norm(interval, ord=np.inf)
            point_scale = max(1.0, np.linalg.norm(current_values, ord=np.inf))
            if step_size <= epsilon * point_scale:
                raise ValueError('Polynomial system iteration stagnated before convergence')
            current_values -= step_factor * interval
        if not np.all(np.isfinite(current_values)):
            raise ValueError('Polynomial system iteration produced non-finite values')
    warnings.warn('The polynomial system solution might have not converged properly')
    return {variable: current_values[index] for index, variable in enumerate(variables)}

def random_linear_system(variables, solutions_range: Tuple[int, int]=(-10, 10), coefficients_range: Tuple[int, int]=(-10, 10), digits_after=0, get_solutions=False):
    from kiwicalc.parsing.parse_expression import ParseExpression
    variables = list(variables)
    if not variables or len(set(variables)) != len(variables) or any(
        not isinstance(variable, str) or not variable or not variable.isidentifier()
        for variable in variables
    ):
        raise ValueError('variables must contain distinct, valid identifiers')
    from kiwicalc.equations.single import _validate_generator_options
    _validate_generator_options(solutions_range, digits_after, variables[0], 'solutions_range')
    _validate_generator_options(coefficients_range, digits_after, variables[0], 'coefficients_range')
    num_of_equations = len(variables)
    solutions = [round(random.uniform(solutions_range[0], solutions_range[1]), digits_after) for _ in range(num_of_equations)]
    if coefficients_range[0] == coefficients_range[1] == 0:
        raise ValueError('coefficients_range cannot produce an invertible system')
    coefficient_matrix = None
    for _ in range(1000):
        candidate = np.asarray([
            [random.randint(coefficients_range[0], coefficients_range[1]) for _ in variables]
            for _ in variables
        ], dtype=float)
        if np.linalg.matrix_rank(candidate) == num_of_equations:
            coefficient_matrix = candidate
            break
    if coefficient_matrix is None:
        raise ValueError('coefficients_range could not produce an invertible system')
    equations = []
    for row in coefficient_matrix:
        coefficients_dict = {variable: [round_decimal(value)] for variable, value in zip(variables, row)}
        right_side = round_decimal(float(np.dot(row, solutions)))
        equations.append(f'{ParseExpression.unparse_linear(coefficients_dict, 0)}={right_side}')
    if get_solutions:
        return (equations, solutions)
    return equations

def random_poly_system(variables):
    variables = list(variables)
    if not variables or len(set(variables)) != len(variables) or any(
        not isinstance(variable, str) or not variable or not variable.isidentifier()
        for variable in variables
    ):
        raise ValueError('variables must contain distinct, valid identifiers')
    roots = [random.randint(-10, 10) for _ in variables]
    equations = [
        f'{ParseExpression.coefficients_to_str([1, -2 * root, root ** 2], variable=variable)}=0'
        for variable, root in zip(variables, roots)
    ]
    return equations
