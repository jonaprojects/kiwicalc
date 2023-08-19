# built in imports
import re

# kiwicalc imports
from models.mono import Mono
from auxiliary import clean_spaces
from ..string_analysis import extract_coefficient


def __data_from_single(single_expression: str, variable_name: str):
    """
    Extracts data from a single-variable monomial, such as 3x^2, or y^2, 82 , etc

    :param single_expression:
    :param variable_name:
    :return:  A tuple with the _coefficient as the first element, and a dictionary of the variable name and its power
    as the second element.
    """
    single_expression = clean_spaces(single_expression)
    if not variable_name:
        return extract_coefficient(single_expression), None
    variable_place = single_expression.find(variable_name)
    coefficient = extract_coefficient(single_expression[:variable_place])

    power_index = single_expression.rfind('^')
    power = 1 if power_index == - \
        1 else float(single_expression[power_index + 1:])
    return coefficient, {variable_name: power}


def mono_from_str(mono_expression: str, get_tuple=False):
    """
    Analyzes a string, such as "3x^2*y^2" and creates a monomial expression ( of type Mono )
    :param mono_expression: the string that represents the monomial
    :param get_tuple: if set to True, instead of a Mono object, the _coefficient(float) and __variables(dict)
    will be returned.
    :return: The monomial, or if get_tuple=True, then its _coefficient and __variables.
    :rtype: Mono or tuple
    """
    try:
        # Try just converting it to float, if it's a normal number
        mono_expression = clean_spaces(mono_expression)
        number = float(mono_expression)
        if get_tuple:
            return number, None
        return Mono(number)
    except (ValueError, TypeError):
        mono_expression: str = mono_expression.strip().replace("**", "^")
        for variable in (character for character in mono_expression if character in allowed_characters):
            occurrences = [m.start()
                           for m in re.finditer(variable, mono_expression)]
        new_expression: str = ''
        for character_index in range(len(mono_expression)):
            new_expression = "".join(
                (new_expression, mono_expression[character_index]))
            if character_index + 1 in occurrences and (mono_expression[character_index].isdigit() or mono_expression[
                    character_index].isalpha()):
                new_expression += '*'
        basic_expressions: list = new_expression.split('*')
        final_coefficient, variables_and_powers = 1, dict()
        for basic_expression in basic_expressions:
            variable: str = "".join(
                [character for character in basic_expression if character in allowed_characters])
            current_coefficient, dictionary_item = __data_from_single(
                basic_expression, variable)
            final_coefficient *= current_coefficient
            if dictionary_item is not None:
                variables_and_powers = {
                    **variables_and_powers, **dictionary_item}
        if get_tuple:
            return final_coefficient, variables_and_powers

        return Mono(coefficient=final_coefficient, variables_dict=variables_and_powers)


def poly_from_str(poly_expression: str, get_list=False) -> "Union[Poly,List]":
    """
    Analyzes a string, such as "3x^2 + 2xy - 7" and generates a polynomial expression
    :param poly_expression:
    :param get_list: if set to True, a list of the monomials ( Mono objects ) will be returned instead
    of a Poly object
    :return: a polynomial corresponding to the string, or a list of monomials.
    :rtype: Poly or list
    """
    poly_expression = clean_spaces(poly_expression)
    expressions = (mono_expression for mono_expression in poly_expression.replace('-', '+-').split('+') if
                   mono_expression != "")
    expressions = [mono_from_str(expression) for expression in expressions]
    if get_list:
        return expressions
    return Poly(expressions)
