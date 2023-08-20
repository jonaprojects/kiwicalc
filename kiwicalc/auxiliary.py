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


@contextmanager
def copy(expression):  # TODO: how to do the exception handling correctly? is this right ??
    try:
        # check for __copy__() method
        copy_method = getattr(expression, "__copy__", None)
        if callable(copy_method):
            copy_of_expression = expression.__copy__()
            yield copy_of_expression
        else:
            # check for copy() method
            copy_method = getattr(expression, "copy", None)
            if callable(copy_method):
                copy_of_expression = expression.copy()
                yield copy_of_expression
    finally:
        del copy_of_expression


def only_numbers_letters(given_string: str):
    """
    checks whether a string contains only letters and numbers.
    :param given_string:
    :return:
    """
    if given_string == "" or given_string is None:
        return False
    char_array = list(given_string)
    if char_array[0] == '-':
        del char_array[0]
    return bool([char for char in char_array if char.isalpha() or char.isdigit()])


def values_in_range(func: Callable, start: float, end: float, step: float, round_results: bool = False):
    """
    fetches all the valid values of a function in the specified range
    :param func: A callable function, that accepts one parameter and returns a single result
    :param start: the beginning of the range
    :param end: the end of the range
    :param step: the interval between each item in the range
    :return: returns the values in the range, and their valid results
    """
    if round_results:
        values = [round_decimal(i) for i in decimal_range(start, end, step)]
        results = [round_decimal(func(i)) for i in values]
    else:
        values = [_ for _ in decimal_range(start, end, step)]
        results = [func(i) for i in values]
    for index, result in enumerate(results):
        if result is None:
            del results[index]
            del values[index]
        elif isinstance(result, bool):
            results[index] = float(result)
    return values, results


def create(expression: str, dtype: str = 'poly'):
    if dtype == 'poly':
        return Poly(expression)
    elif dtype == 'log':
        return Log(expression)
    elif dtype == 'ln':
        return Ln(expression)
    elif dtype == 'trigo':
        return TrigoExprs(expression)
    elif dtype == 'root':
        # TODO: implement string constructor in root via dtype as well
        return Root(expression)
    elif dtype == 'factorial':
        # TODO: implement string constructor in root via dtype as well
        return Factorial(expression)
    else:
        raise ValueError(f"Invalid parameter 'dtype': {dtype}")


def create_from_dict(given_dict: dict):
    if isinstance(given_dict, int):
        return Mono(given_dict)
    expression_type = given_dict['type'].lower()
    if expression_type == 'mono':
        return Mono.from_dict(given_dict)
    elif expression_type == 'poly':
        return Poly.from_dict(given_dict)
    elif expression_type == 'trigoexpr':
        return TrigoExpr.from_dict(given_dict)
    elif expression_type == 'trigoexprs':
        return TrigoExprs.from_dict(given_dict)
    elif expression_type == 'log':
        return Log.from_dict(given_dict)
    elif expression_type == 'factorial':
        return Factorial.from_dict(given_dict)
    elif expression_type == 'root':
        return Root.from_dict(given_dict)
    elif expression_type == 'abs':
        return Abs.from_dict(given_dict)
    elif expression_type == 'iexpressions':
        return ExpressionSum.from_dict(given_dict)
    elif expression_type == 'fraction':
        return Fraction.from_dict(given_dict)
    elif expression_type == 'fastpoly':
        return FastPoly.from_dict(given_dict)
    elif expression_type == 'exponent':
        return Exponent.from_dict(given_dict)
    else:
        raise ValueError(f"Invalid type of expression: {given_dict}")


def format_matplot_polynomial(expression: str):
    """
    formats a polynomial expression into matplot's format
    :param expression:
    :return:
    """
    # SPAGHETTI CODE ALERT !!!
    expressions = split_expression(expression)
    for index, expr in enumerate(expressions):
        expr = expr.replace('**', '^')
        accumulator = ""
        skip = 0
        for i in range(len(expr)):
            character = expr[i]
            if skip == 0:
                if character == '^':
                    accumulator = "".join((accumulator, '^{'))
                    j = i + 1
                    while j < len(expr) and expr[j] not in ('^', '*', '+', '-'):
                        accumulator += expr[j]
                        j += 1
                    accumulator = "".join((accumulator, '}'))
                    skip = j - i - 1
                else:
                    accumulator += character
            else:
                skip -= 1
        expressions[index] = rf'{accumulator}'
    return f'{"".join(expressions)}'


def get_factors(n):
    if n == 0:
        return {}
    factors = set(reduce(list.__add__,
                         ([i, n // i] for i in range(1, int(abs(n) ** 0.5) + 1) if n % i == 0)))
    return factors.union({-number for number in factors})


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
    if new_list is None:
        return []
    return new_list, result


def equal_ignore_order(a, b):
    """ Use only when elements are neither hashable nor sortable! """
    if None in (a, b):
        return False
    if len(a) != len(b):
        return False
    unmatched = list(b)
    for element in a:
        try:
            unmatched.remove(element)
        except ValueError:
            return False
    return not unmatched
