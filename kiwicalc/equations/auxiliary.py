# TODO: improve this shitty method.
def extract_dict_from_equation(equation: str, delimiter="="):
    """
    This method should accept an equation, and extract the variable from it. It is still quite basic..

    :param equation: the equation, of type string
    :param delimiter: separator
    :return: returns a dictionary of the __variables and the number. for example, for the equation 3x-y+8 = 6+y+x the
    dictionary returned would be {'x':0,'y':0,'number':0}
    """
    variables = dict()
    first_side, second_side = equation.split(delimiter)
    accumulator = ""
    for expression in split_expression(first_side) + split_expression(second_side):
        start_index = -1
        for index, character in enumerate(expression):
            if character.isalpha():
                start_index = index
                break
        if start_index != -1:
            accumulator = ""
            for character in expression[start_index:]:
                accumulator += character
        variables[accumulator.strip()] = 0
    variables["number"] = 0
    return {key: value for key, value in variables.items() if key != ''}


def linear_expression_to_dict(expression: str, variables: Iterable) -> dict:
    """alternative way to """
    expression = clean_spaces(expression)
    my_dict = dict()
    if expression[-1] != ' ':
        expression += ' '
    matches = list(re.finditer(
        fr"([-+]?\d+[.,]?\d*)?\*?([a-zA-Z]+)", expression))
    for variable in variables:
        my_dict[variable] = sum(extract_coefficient(match.group(1))
                                for match in matches if match.group(2) == variable)
    matches = re.finditer(fr"([-+]?\d+[.,]?\d*)[-+\s]", expression)
    numbers_sum = sum(extract_coefficient(match.group(1)) for match in matches)
    my_dict["number"] = numbers_sum
    return my_dict


def subtract_dicts(dict1: dict, dict2: dict) -> dict:
    """
    each side in the equation is processed into a dictionary. in order to reach a result, it is imperative
    to subtract the two sides, and equate what's left to 0.
    This method is responsible for taking both dictionaries, and subtracting them.
    :param dict1: the first dictionary
    :param dict2: the second dictionary
    :return:
    """
    new_dict = {}
    # making sure the dictionaries have the same keys
    for key in dict2.keys():
        if key not in dict1.keys():
            dict1[key] = 0
            warnings.warn(
                f"variable {key} wasn't found in the first data structure")
    for key in dict1.keys():
        if key not in dict2.keys():
            dict2[key] = 0
            warnings.warn(
                f" variable {key} wasn't found in the second data structure")

    for key in dict1.keys():
        new_dict[key] = dict1[key] - dict2[key]

    return new_dict


def coefficients_to_expressions(coefficients, variable: str = "x"):
    """
    Getting a list of coefficients and the name of the variable, and returns a list of polynomial expressions,
    namely, a list of Mono objects.
    :param coefficients: the coefficients, for example : [ 1,0,2,3] ( the output expression would be x^3+2x+3 for x )
    :param variable: the name of the variable, the default is "x"
    :return: returns a list of polynomials with the corresponding coefficients and powers.
    """
    return [Mono(coefficient=coef, variables_dict={variable: len(coefficients) - 1 - index}) for index, coef in
            enumerate(coefficients) if coef != 0]


def extract_possible_solutions(most_significant_coef: float, free_number: float):
    p_factors = get_factors(free_number)
    q_factors = get_factors(most_significant_coef)
    possible_solutions = []
    for p_factor in p_factors:
        for q_factor in q_factors:
            if q_factor != 0:  # Can't divide by 0..
                possible_solutions.append(p_factor / q_factor)
    if possible_solutions:
        return set(possible_solutions)
    return {}


def __find_solutions(coefficients, possible_solutions):
    solutions = set()
    copy = coefficients.copy()
    if copy[-1] == 0:
        solutions.add(0)
        for i in range(len(copy) - 1, 0, -1):
            if copy[i] != 0:
                break
            del copy[i]
        solutions.add(solve_polynomial(copy))

    for solution in possible_solutions:
        division_result, remainder = synthetic_division(coefficients, solution)
        if remainder != 0:
            continue
        solutions.add(solution)
        if len(division_result) >= 4:
            for sol in solve_polynomial(division_result):
                solutions.add(sol)
        elif len(division_result) == 3:
            # Solving a quadratic equation!
            try:
                # find more solutions. right now, this feature only works on real numbers, and when the highest power
                # is 3
                values = solve_quadratic_real(
                    division_result[0], division_result[1], division_result[2])
                solutions.add(values[0])
                solutions.add(values[1])
            except Exception as e:
                warnings.warn(
                    f"Due to an {e.__class__} error in line {exc_info()[-1].tb_lineno}, some solutions might be "
                    f"missing ! ")
        else:
            print("Whoops! it seems something went wrong")
    return solutions


def format_linear_dict(algebraic_dict: dict, round_coefficients: bool = True) -> str:
    """ Receives a dictionary that represents a linear expression and creates a new string from it.
        For instance, the dictionary {'x':2, 'y':-4, 'number':5} represents the expression 2x - 4y + 5
     """
    if not algebraic_dict:
        return ""
    accumulator = ""
    for key, value in algebraic_dict.items():
        if value != 0:
            value = round_decimal(value) if round_coefficients else value
            coef = f"+{value}" if value > 0 else f"{value}"
            if key == 'number':
                accumulator += coef
            elif value == 1:
                accumulator += f"+{key}"
            elif value == -1:
                accumulator += f"-{key}"
            else:
                accumulator += f"{coef}{key}"
    return accumulator[1:] if accumulator[0] == '+' else accumulator


def format_poly_dict(algebraic_dict: dict):
    """
    Internal method: Receives a dictionary that represents a polynomial, in the Equation's class format, and turns it into string.
    *** This method might become redundant / deleted / updated.
    :param algebraic_dict: the dictionary of the expression, for example : {'x**2':3,'x':2,'number':-1} -> 3x^2+2x-1
    :return:
    """
    accumulator = ""
    for expression, coefficient in algebraic_dict.items():
        if expression == 'number':
            if coefficient >= 0:
                accumulator += f'+{round_decimal(coefficient)}'
            else:
                accumulator += f'{round_decimal(coefficient)}'
        else:
            if coefficient != 0:
                power = float(expression[expression.find('**') + 2:])
                if coefficient == int(coefficient):
                    coefficient = int(coefficient)
                if power == int(power):
                    power = int(power)
                if power == 1:
                    accumulator += f"{coefficient}{expression[:expression.find('**')]}"
                elif power == 0:
                    algebraic_dict['number'] += 1
                else:
                    accumulator += f' {coefficient}{expression} +'
    return accumulator.replace("++", "+").replace("+-", "-").replace("--", "-")


def equation_to_one_side(equation: str) -> str:
    """ Move all of the items of the equation to one side"""

    equal_sign = equation.find("=")
    if equal_sign == -1:
        raise ValueError(
            "Invalid equation - an equation must have two sides, separated by '=' ")
    first_side, second_side = equation[:equal_sign], equation[equal_sign + 1:]
    second_side = "".join(
        ('+' if character == '-' else ('-' if character == '+' else character)) for character in second_side)
    second_side = f'-{second_side}' if second_side[0] not in (
        '+', '-') else second_side
    if second_side[0] in ('+', '-'):
        return first_side + second_side
    return first_side + second_side


def float_gcd(a: float, b: float, rtol: float = 1e-05, atol: float = 1e-08):
    """ finding the greatest common divisor for 2 float numbers"""
    if a == 0 or b == 0:
        return 0
    t = min((abs(a), abs(b)))
    while abs(b) > rtol * t + atol:
        a, b = b, a % b
    return round_decimal(a)


def gcd(decimal_numbers: Iterable):
    """
    Finding the greatest common divisor. For example, for the tuple (2.5,3.5,1.5) the result would be 0.5.

    :param decimal_numbers: a list or tuple with decimal numbers
    :return: the greatest common divisor of those numbers
    """
    return reduce(lambda a, b: float_gcd(a, b), decimal_numbers)


def apply_on(func: Callable, collection: Iterable) -> Iterable:
    """Apply a certain given function on a collection of items"""

    if isinstance(collection, (list, set)):  # modify the given collection
        for index, value in enumerate(collection):
            collection[index] = func(value)
        return collection
    return [func(item) for item in collection]


def is_lambda(v) -> bool:
    """ Returns True whether an expression is a lambda expression, otherwise False"""
    def sample_lambda(): return 0
    try:
        return isinstance(v, type(sample_lambda)) and v.__name__ == sample_lambda.__name__
    except:
        return False


def format_coefficient(coefficient: "Union[int, float, IExpression]") -> str:
    if coefficient == 1:
        return ""
    if coefficient == -1:
        return "-"
    return coefficient.__str__()


def format_free_number(free_number: Union[int, float]):
    if free_number == 0:
        return ""
    if free_number < 0:
        return f"{round_decimal(free_number)}"

    return f"+{round_decimal(free_number)}"
