"""
Auxiliary functions for algebraic operations
"""

from typing import Iterable, Union, Dict, Any, TYPE_CHECKING
from .IExpression import IExpression
if TYPE_CHECKING:
    from .poly import Poly
    from .mono import Mono


def add_or_sub_coefficients(first_coefficients, second_coefficients, mode='add', copy_first=True):
    """
    Add or subtract coefficient lists for polynomial operations.
    
    :param first_coefficients: First coefficient list
    :param second_coefficients: Second coefficient list
    :param mode: Operation mode ('add' or 'sub')
    :param copy_first: Whether to copy the first list
    :return: Resulting coefficient list
    """
    first_coefficients = list(
        first_coefficients) if copy_first else first_coefficients
    second_coefficients = list(second_coefficients)
    my_variables_length = len(first_coefficients)
    other_variables_length = len(second_coefficients)
    # Make sure the two lists are in the same length
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
    while first_coefficients[0] == 0:  # delete spare zeros
        del first_coefficients[0]

    return first_coefficients


def sorted_expressions(expressions: "Iterable[Union[Poly,Mono]]"):
    """
    Sort expressions by their highest variable power.
    
    :param expressions: Collection of Poly or Mono expressions
    :return: Sorted expressions
    """
    # TODO: add feature for handling free numbers as well ?
    assert all(
        expression.variables_dict is not None for expression in expressions), "This method cannot accept free numbers"
    return sorted(expressions, key=lambda item: max(item.variables_dict.values()),
                  reverse=True)  # sort by the power


def fetch_power(variables: dict):
    """
    Get the power of the first variable in a variables dictionary.
    
    :param variables: Dictionary of variables and their powers
    :return: Power value
    """
    return variables[next(iter(variables))]


def fetch_variable(variables: dict):
    """
    Get the first variable name from a variables dictionary.
    
    :param variables: Dictionary of variables and their powers
    :return: Variable name or None
    """
    try:
        return f'{next(iter(variables))}'
    except (IndexError, StopIteration):
        return None


def process_object(expression: Union[IExpression, int, float], class_name: str, method_name: str, param_name: str):
    """
    Process an object to ensure it's the correct type for algebraic operations.
    
    :param expression: Expression to process
    :param class_name: Name of the calling class
    :param method_name: Name of the calling method
    :param param_name: Name of the parameter
    :return: Processed expression
    """
    if isinstance(expression, (int, float)):
        from .mono import Mono
        return Mono(expression)
    elif isinstance(expression, IExpression):
        return expression.__copy__()
    raise TypeError(f"Invalid type '{type(expression)}' of paramater '{param_name}' in method {method_name} in class"
                    f" {class_name}")


def max_power(expressions):
    """
    Find the expression with the highest variable power.
    
    :param expressions: Collection of expressions
    :return: Expression with maximum power
    """
    return max(expressions, key=lambda expression: max(expression.variables_dict.values()))
