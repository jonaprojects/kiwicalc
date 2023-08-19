from math import floor, ceil

"""
Auxiliary methods for different parts of the library
"""


def decimal_range(start: float, stop: float, step: float = 1):
    while start <= stop:
        yield start
        start += step


def round_decimal(number: float):
    """
    Rounds a decimal number, or at least tries to.... Since python is weird with it
    :param number: ugly number we wish to round
    :return: less ugly number
    """
    if number - floor(number) < 0.000001:
        return floor(number)
    elif abs(number - ceil(number)) < 0.000001:
        return ceil(number)
    return round(number, 5)


def clean_spaces(equation: str) -> str:
    """cleans a string from spaces.
    """
    return "".join([character for character in equation if character != ' '])


def contains_from_list(lst: list, s: str) -> bool:
    """
    checks whether a string appears in a list of strings
    :param lst: the list of strings, for example : ["hello","world"]
    :param s: the string, for example: "hello"
    :return: True if contains, else False
    """
    return bool([x for x in lst if x in s])


def extract_variables_from_expression(expression: str):
    return {character for character in expression if character.isalpha()}


def copy_expression(expression):
    if isinstance(expression, IExpression) or hasattr(expression, "__copy__"):
        return expression.__copy__()

    if isinstance(expression, (list, set)) or hasattr(expression, "copy"):
        return expression.copy()

    return expression
