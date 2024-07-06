def __helper_trigo(expression: str) -> Optional[Tuple[int, Optional[float]]]:
    try:
        first_letter_index = expression.find(next(
            (character for character in expression if character.isalpha() and character not in ('e', 'i'))))
        return first_letter_index, float(str(extract_coefficient(expression[:first_letter_index])))
    except (StopIteration, ValueError):
        print(expression)
        return None


def analyze_single_trigo(trigo_expression: str, get_tuple=False, dtype='poly'):
    """
    Generates a TrigoExpr object from a string with a simplified trigonometric expression, such as sin(5x+7), or sin(45)

    :param trigo_expression: the string
    :param get_tuple: if set to True, a tuple of the _coefficient,chosen trigonometric method, and the inside expression will be returned
    :return: a TrigoExpr object corresponding to the string,or a tuple if get_tuple is set to True
    """
    trigo_expression = trigo_expression.strip().replace(
        "**", '^').replace(" ", "")  # To prevent any stupid mistakes
    left_parenthesis_index: int = trigo_expression.find('(')
    right_parenthesis_index: int = trigo_expression.rfind(')')
    first_letter_index, coefficient = __helper_trigo(trigo_expression)
    method_chosen = trigo_expression[first_letter_index:left_parenthesis_index].upper(
    )
    method_chosen = TrigoMethods[method_chosen]
    inside_string = trigo_expression[left_parenthesis_index +
                                     1:right_parenthesis_index]
    inside = create(inside_string, dtype=dtype)
    power_index = trigo_expression.rfind('^')
    if power_index == -1 or power_index < right_parenthesis_index:
        power = 1
    else:
        power = float(trigo_expression[power_index + 1:])
    if get_tuple:
        return coefficient, method_chosen, inside, power
    return TrigoExpr(coefficient, [(method_chosen, inside, power)])


def TrigoExpr_from_str(trigo_expression: str, get_tuple=False,
                       dtype='poly') -> "Union[Tuple[IExpression,List[list]],TrigoExpr]":
    """

    :param trigo_expression:
    :param get_tuple:
    :return:
    """
    trigo_expression = trigo_expression.strip().replace(
        "**", "^")  # Avoid stupid mistakes
    coefficient = Poly(1)
    expressions = [expression for expression in trigo_expression.split(
        '*') if expression.strip() != ""]
    new_expressions = []
    for expression in expressions:
        if is_number(expression):  # Later add support for i and e in the _coefficient detection
            coefficient *= float(expression)
        else:
            new_expressions.append(expression)
    analyzed_generator = (analyze_single_trigo(expression, get_tuple=True, dtype=dtype) for expression in
                          new_expressions)
    analyzed_expressions = []
    for (coef, method_chosen, inside, power) in analyzed_generator:
        analyzed_expressions.append([method_chosen, inside, power])
        coefficient *= coef
    if not analyzed_expressions:
        analyzed_expressions = None
    if get_tuple:
        return coefficient, analyzed_expressions
    return TrigoExpr(coefficient, expressions=analyzed_expressions)


def TrigoExprs_from_str(trigo_expression: str, get_list=False):
    """

    :param trigo_expression:
    :param get_tuple:
    :return:
    """
    trigo_expressions: list = split_expression(
        trigo_expression)  # TODO: What about the minus in the beginning?
    new_expressions: list = [TrigoExpr_from_str(
        expression) for expression in trigo_expressions]
    if get_list:
        return new_expressions
    return TrigoExprs(new_expressions)
