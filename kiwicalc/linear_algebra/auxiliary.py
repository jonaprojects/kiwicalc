"""
Auxiliary functions for linear algebra operations
"""

from typing import List, Union, Iterable
from ..algebra.poly import Poly
from ..algebra.mono import Mono
from ..algebra.algebra_string_analysis import poly_from_str
from ..equations.auxiliary import equation_to_one_side
from .matrices.matrix import Matrix


def generate_jacobian(functions, variables):
    """
    Generate the Jacobian matrix for a system of functions with respect to variables.
    
    :param functions: Collection of functions
    :param variables: Collection of variables
    :return: Jacobian matrix as a list of lists
    """
    if len(functions) != len(variables):
        raise ValueError("The Jacobian matrix must be nxn, so you need to enter an equal number of functions "
                         "and variables")

    return [[func.partial_derivative(variable) for variable in variables] for func in functions]


def approximate_jacobian(functions, values, h=0.001):
    """
    Approximate the Jacobian matrix using finite differences.
    
    :param functions: Collection of functions
    :param values: Values at which to evaluate the Jacobian
    :param h: Step size for finite differences
    :return: Approximate Jacobian matrix
    """
    result_jacobian = []
    for f in functions:
        temp = f(*values)
        new_list = []
        for index, variable in enumerate(values):
            new_list.append(
                (f(*(values[:index] + [values[index] + h] + values[index + 1:])) - temp) / h)
        result_jacobian.append(new_list)
    return result_jacobian


def generate_polynomial_matrix(
        equations: "Union[Iterable[Union[str,Poly,Mono]],Iterable[Union[str, Poly, Mono]]]") -> "Matrix":
    """
    Create a matrix of polynomials from a collection of equations.
    
    :param equations: Collection of equations (strings, Poly, or Mono objects)
    :return: Matrix containing the polynomial expressions
    """
    if isinstance(equations[0], str):
        return Matrix(matrix=[poly_from_str(equation_to_one_side(equation)) for equation in equations])
    return Matrix(matrix=equations)
