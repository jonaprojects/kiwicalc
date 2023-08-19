import re

# might be necessary for the eval()
from math import sin, asin, sinh, cos, acos, cosh, tan, atan, tanh, asinh, acosh, atanh
from math import pi, e, tau, log, exp, log2, sqrt, log10, gamma, lgamma, erf, erfc

# TODO: later replace with the code inside to_lambda


def handle_parenthesis(expression: str):
    """
    [INSIDE USE][Needs to be fixed and reviewed] This method processes the appearances of parenthesis in expressions.

    :param expression:
    :return:
    """
    new_expression = ""
    for character_index in range(len(expression)):
        # Handling cases such as y(3x+6) -> y*(3x+6)
        if expression[character_index] == '(' and character_index > 0 and expression[character_index - 1] not in (
                '+', '-', '*', '/', '%'):
            new_expression += '*'
        new_expression += expression[character_index]
        # Handling cases such as (3+x)y -> (3+x)*y
        if expression[character_index] == ')' and character_index < len(expression) - 1 and expression[
                character_index + 1] not in ('+', '-', '*', '/', '%', '!'):
            new_expression += '*'
    return new_expression


def handle_abs(expression: str):
    """
    An attempt to handle absolute values and evaluate them as necessary.
    :param expression: the expression to be processed, of type str.
    :return:
    """
    copy = expression.replace("|", "~~")
    results = {res: res[2:len(res) - 2]
               for res in re.findall(f'~~.*?~~', copy)}
    for old, new in results.items():
        if is_evaluatable(new):
            before_index = copy.find(old) - 1
            if before_index > 0 and copy[before_index].isalpha() or copy[before_index].isdigit():
                copy = copy.replace(old, f'*{abs(eval(new))}')

            copy = copy.replace(old, str(abs(eval(new))))
        else:

            before_index = copy.find(old) - 1
            if before_index > 0 and copy[before_index].isalpha() or copy[before_index].isdigit():
                copy = copy.replace(old, f'*abs({new})')
            copy = copy.replace(old, f'abs({new})')
    return copy


def handle_factorial(expression):
    if '!' not in expression:
        return expression

    copy1 = expression.replace(" ", "")
    results = [res for res in re.findall(
        f'([a-zA-Z0-9]+!|[a-zA-Z0-9]*\([^!]+\)!)', copy1)]
    for result in results:
        if result.startswith('(') and result.endswith(')'):
            result = result[1:-1]
        new = f"factorial({result[:-1]})"
        if is_evaluatable(result[:-1]):
            before_index = copy1.find(result) - 1
            value = factorial(eval(result[:-1]))
            if before_index > 0 and copy1[before_index].isalpha() or copy1[before_index].isdigit():
                copy1 = copy1.replace(result, f'*{value}')

            copy1 = copy1.replace(result, str(value))
        else:

            before_index = copy1.find(result) - 1
            if before_index > 0 and copy1[before_index].isalpha() or copy1[before_index].isdigit():
                copy1 = copy1.replace(result, f'*{new}')
            copy1 = copy1.replace(result, f'{new}')
    return copy1


def is_evaluatable(s):
    """
    Rather ugly and insecure method, but it is necessary for parts of the code.I haven't found a better
    alternative yet that doesn't limit some features.
    :param s: the expression
    :return: True if it can be evaluated, False otherwise.
    """
    try:
        eval(s)
        return True
    except:
        return False


def is_number(suspicious_string: str):
    """
    checks whether a string can be converted into float.
    :param suspicious_string: the string to be checked.
    :return: True if it can be converted, otherwise False.
    """
    try:
        val = float(suspicious_string)
        return True
    except:
        return False


def split_expression(expression: str):
    """splits the expression by delimiters, but doesn't touch what's inside parenthesis """
    delimiters = []
    for index, char in enumerate(expression):
        if char in Function.arithmetic_operations and index > 0:
            parenthesis_index, curly_index = expression[:index].rfind(
                '('), expression[:index].rfind('{')
            closing_paranthesis_index = expression[parenthesis_index:].find(
                ')') + parenthesis_index
            closing_curly = expression[curly_index:].find('}') + curly_index
            square_index = expression[:index].rfind('[')
            close_square = expression[curly_index:].find(']') + square_index
            if not parenthesis_index < index < closing_paranthesis_index and not curly_index < index < closing_curly and not square_index < index < close_square:
                delimiters.append(index)
    expressions = []
    if len(delimiters) > 0:
        expressions.append(expression[:delimiters[0]])
        for i in range(1, len(delimiters)):
            expressions.append(expression[delimiters[i - 1]:delimiters[i]])
        expressions.append(expression[delimiters[len(delimiters) - 1]:])
    else:
        expressions.append(expression)
    return [expression for expression in expressions if expression != ""]


def formatted_expression(expression: str, variables, constants=(), format_abs=False, format_factorial=False):
    """
    Formats an expression
    For example: The string "3x^2 + 5x + 6" -> "3*x**2+5*x+6"

    :param expression: The expression entered
    :param variables: The variables_dict appearing in the expression
    :param constants: Constants that appear in the expression

    :return: A new string, with proper pythonic algebraic syntax
    """
    if format_abs:
        expression = handle_abs(expression)

    if format_factorial:
        expression = handle_factorial(expression)

    expressions = split_expression(expression.replace(
        "^", "**"))  # Handling absolute value notations
    # and splitting the expression into sub-expressions.
    modified_variables = list(variables) + list(constants)
    for index, expression in enumerate(expressions):
        new_expression = ""
        occurrences = []
        for variable in modified_variables:
            occurrences += [m.start()
                            for m in re.finditer(variable, expression)]
        for character_index in range(len(expression)):
            # Handling cases such as y(3x+6) -> y*(3x+6)
            """ if expression[character_index] == '(' and character_index > 0 and expression[character_index - 1] not in (
                    '+', '-', '*', '/', '%'):
                new_expression += '*'
            """
            new_expression += expression[character_index]
            # Handling cases such as (3+x)y -> (3+x)*y
            """if expression[character_index] == ')' and character_index < len(expression) - 1 and expression[
                character_index + 1] not in ('+', '-', '*', '/', '%', '!'):
                new_expression += '*'"""
            if character_index + 1 in occurrences and (expression[character_index].isdigit() or expression[
                    character_index].isalpha()) and expression[character_index] + expression[
                    character_index + 1] not in modified_variables:
                new_expression += '*'
        expressions[index] = new_expression
    return "".join(expressions)


def to_lambda(expression: str, variables, constants=(), format_abs=False, format_factorial=False):
    """
    Generate an executable lambda expression from a string

    :param expression: The string that represents the expression ( focuses on processing algebraic expressions)
    :param variables: The variables_dict of the expression.
    :param constants: Constants or methods to be used in the expression.
    :param format_abs:  Whether to check for absolute values and process them.
    :param format_factorial: Whether to check for factorials and process them.
    :return: Returns a lambda expression corresponding to the expression given.
    """
    modified_expression = formatted_expression(expression, variables, constants, format_abs=format_abs,
                                               format_factorial=format_factorial)
    # print(f'lambda {",".join(variables_dict)}:{modified_expression}')
    return eval(f'lambda {",".join(variables)}:{modified_expression}')


def apply_parenthesis(given_string: str, delimiters=('+', '-', '*', '**')):
    """put parenthesis on expressions such as x+5, 3*x , etc - if needed."""
    if any(character in delimiters for character in given_string):
        return f"({given_string})"
    return given_string


def extract_coefficient(coefficient: str) -> float:
    """[method for inside use]"""
    return -1 if coefficient == '-' else 1 if coefficient in ('+', '') else float(coefficient)
