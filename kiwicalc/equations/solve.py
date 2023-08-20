"""
Methods for solving different kinds of equations, system of equations and in equalities
"""


def solve_polynomial(coefficients, epsilon: float = 0.000001, nmax: int = 10_000):
    """
    This method find the roots of a polynomial from a collection of the coefficients of the expression.
    The algorithm chooses the most efficient algorithm in correspondence to the degree of the polynomial.
    for a collection of coefficients of length n, its degree would be n-1.
    For example, [1, -2, 1] represents x^2 - 2x + 1. The length of the list is 3 and the degree of the expression is 2.
    For degrees of 4 or lower, the execution time will be significantly lower, since it is computed via generalized
    formulas or algebra.
    For degrees of 5 or more, there aren't any generalized formulas ( as proven in Abel's impossibility theorem ),
    So instead of a formula, an iterative method is used to approximate the roots ( complex and real ) of
    the polynomial.

    :param coefficients: The coefficients of the polynomial.
    :param epsilon:
    :param nmax: Max number of iterations In case the polynomial is of a degree of 5 or more.Default is 100,000
    :return:
    """
    if isinstance(coefficients, str):
        return solve_polynomial(ParseEquation.parse_polynomial(coefficients))

    if len(coefficients) == 1:
        return None
    if len(coefficients) == 2:
        return [-coefficients[1] / coefficients[0]]
    if len(coefficients) == 3:
        return solve_quadratic(coefficients[0], coefficients[1], coefficients[2])
    if len(coefficients) == 4:
        return solve_cubic(coefficients[0], coefficients[1], coefficients[2], coefficients[3])
    if len(coefficients) == 5:
        return solve_quartic(coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4])
    # Iterative approach via aberth's method, since there are no generalized formulas for polynomial equations
    # of degree 5 or higher, as proven by the Abel–Ruffini theorem.
    polynomial_obj = Poly(coefficients_to_expressions(coefficients))
    poly_derivative = polynomial_obj.derivative().to_lambda()
    return aberth_method(polynomial_obj.to_lambda(), poly_derivative, coefficients, epsilon, nmax)


def solve_poly_by_factoring(coefficients):
    """
    This method attempts to find the roots of a polynomial by synthetic division.
    It won't always return all the solutions, but it is faster than many numerical root finding algorithms, and might
    even be preferable in some cases over these algorithms.
    """
    if len(coefficients) == 3:
        return solve_quadratic_real(coefficients[0], coefficients[1], coefficients[2])
    if coefficients is None:
        return {}
    most_significant = coefficients[0]
    free_number = coefficients[-1]
    possible_solutions = extract_possible_solutions(
        most_significant, free_number)
    print(possible_solutions)
    solutions = __find_solutions(coefficients, possible_solutions)
    return solutions


def solve_linear(equation: str, variables=None, get_dict=False, get_json=False):
    if variables is None:
        variables = extract_dict_from_equation(equation)
    first_side, second_side = equation.split("=")
    first_dict = simplify_linear_expression(
        expression=first_side, variables=variables)
    second_dict = simplify_linear_expression(
        expression=second_side, variables=variables)
    result_dict = {key: value for key, value in subtract_dicts(
        dict1=first_dict, dict2=second_dict).items() if key}
    if len(result_dict) < 2:
        return None
    elif len(result_dict) == 2:
        if list(result_dict.values())[0] == 0:
            if list(result_dict.values())[1] == 0:
                return np.inf
            return None  # There are no solutions, ( like 0 = 5 )
        solution = -result_dict["number"] / list(result_dict.values())[0]
        if get_dict:
            return {list(variables.keys())[0]: solution}
        elif get_json:
            return json.dumps({"variable": list(variables.keys())[0], "result": solution})
        return solution
    elif len(result_dict) > 2:
        raise ValueError("Invalid equation caused an unexpected error")


def solve_linear_inequality(equation: str, variables=None):
    equal_sign_index = equation.find('=')
    if equal_sign_index == -1:
        bigger_sign_index = equation.find('>')
        if bigger_sign_index == -1:
            smaller_sign_index = equation.find('<')
            if smaller_sign_index == -1:
                raise ValueError("Invalid equation")
            else:
                sign = '<'
        else:
            sign = '>'
    else:
        bigger_sign_index = equation.find('>')
        if bigger_sign_index == -1:
            smaller_sign_index = equation.find('<')
            if smaller_sign_index == -1:
                sign = '<='
            else:
                sign = '='

        else:
            sign = '>='

    expressions = equation.split(sign)
    if len(expressions) != 2:
        raise ValueError(f"Invalid equation")
    if variables is None:
        variables = extract_dict_from_equation(equation, delimiter=sign)
    first_side, second_side = expressions
    first_dict = simplify_linear_expression(first_side, variables)
    second_dict = simplify_linear_expression(second_side, variables)
    result_dict = subtract_dicts(first_dict, second_dict)
    first_key = list(result_dict.keys())[0]
    first_value = list(result_dict.values())[0]
    number_value = result_dict["number"]
    return f"{first_key}{sign}{round_decimal(-number_value / first_value)}"


def solve_linear_system(equations, variables=None):
    """ Solve a system of linear equations via Guass-Elimination Method with matrices"""
    if not variables:
        variables = set()
        for equation in equations:
            variables.update(extract_variables_from_expression(equation))
    values_matrix = []
    for equation in equations:
        equal_index = equation.find('=')
        side1, side2 = equation[:equal_index], equation[equal_index + 1:]
        first_dict = simplify_linear_expression(side1, variables)
        second_dict = simplify_linear_expression(side2, variables)
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


def solve_poly_system(
        equations: "Union[Iterable[Union[str,Poly,Mono]],Iterable[Union[str, Poly, Mono]]]",
        initial_vals: dict = None, epsilon: float = 0.00001, nmax: int = 10000, show_steps=False):
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
        variables = {extract_variables_from_expression(
            equation) for equation in equations}
        initial_vals = Matrix(matrix=[0 for _ in range(len(variables))])
    variables, initial_values = list(
        initial_vals.keys()), Matrix(matrix=initial_vals.values())
    polynomials = [(poly_from_str(equation_to_one_side(equation)) if isinstance(equation, str) else equation) for
                   equation in equations]
    jacobian_matrix = Matrix(
        matrix=generate_jacobian(polynomials, variables))  # Generating a corresponding jacobian matrix
    current_values_matrix = Matrix(
        matrix=[[current_value] for current_value in list(initial_vals.values())])
    for i in range(nmax):
        assignment_dictionary = dict(
            zip(variables, [row[0] for row in current_values_matrix.matrix]))
        assigned_jacobian = jacobian_matrix.mapped_matrix(
            lambda polynomial: polynomial.when(**assignment_dictionary).try_evaluate())
        jacobian_inverse = assigned_jacobian.inverse()
        assigned_polynomials = Matrix(matrix=[[polynomial.when(**assignment_dictionary).try_evaluate()] for
                                              polynomial in polynomials])
        if all(abs(row[0]) < epsilon for row in assigned_polynomials.matrix):
            return {variables[index]: row[0] for index, row in enumerate(current_values_matrix)}
        interval_matrix = jacobian_inverse @ assigned_polynomials
        current_values_matrix -= interval_matrix

# TODO: fix this !!!!


def solve_quadratic_from_str(expression, real=False, strict_syntax=False):
    if isinstance(expression, str):
        variables = get_equation_variables(expression)
        if len(variables) == 0:
            return tuple()
        elif len(variables) == 1:
            variable = variables[0]
            parsed_dict = ParseEquation.parse_quadratic(
                expression, variables, strict_syntax=strict_syntax)
            solve_method = solve_quadratic_real if real else solve_quadratic
            return solve_method(parsed_dict[variable][0], parsed_dict[variable][0], parsed_dict['free'])
        else:
            raise ValueError(
                "Can't solve a quadratic equation with more than 1 variable")


def solve_quadratic(a: Union[str, float], b: float = None, c: float = None) -> tuple:
    """ Solves a quadratic equation using computations of complex numbers ( utilizing the cmath library)"""
    if isinstance(a, str):
        return solve_quadratic_from_str(a)
    discriminant = b ** 2 - 4 * a * c
    return (-b + cmath.sqrt(discriminant)) / (2 * a), (-b - cmath.sqrt(discriminant)) / (2 * a)


def solve_quadratic_real(a: Union[str, float], b: float, c: float) -> Optional[Union[Tuple[float, float], float]]:
    """returns onlu the real solutions of the quadratic equation"""
    if isinstance(a, str):
        return solve_quadratic_from_str(a, real=True)
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return None
    if discriminant == 0:
        return (-b + sqrt(discriminant)) / (2 * a)
    return (-b + sqrt(discriminant)) / (2 * a), (-b + sqrt(discriminant) / (2 * a))


def solve_quadratic_params(a: "Union[IExpression, int, float,str]", b: "Union[IExpression, int, float]", c: "Union[IExpression,int,float]"):
    if isinstance(a, str):  # TODO: implement string analysis here
        print("need to be implemented")
    if all(isinstance(coefficient, (int, float)) for coefficient in (a, b, c)):
        return solve_quadratic(a, b, c)
    if isinstance(a, IExpression):
        a_eval = a.try_evaluate()
        if a_eval is not None:
            a = a_eval
    if isinstance(b, IExpression):
        b_eval = b.try_evaluate()
        if b_eval is not None:
            b = b_eval
    if isinstance(c, IExpression):
        c_eval = c.try_evaluate()
        if c_eval is not None:
            c = c_eval
    if all(isinstance(coefficient, (int, float)) for coefficient in (a, b, c)):
        return (-b + sqrt(b ** 2 - 4 * a * c)) / (2 * a), (-b - sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    else:
        return (-b + Sqrt(b ** 2 - 4 * a * c)) / (2 * a), (-b - Sqrt(b ** 2 - 4 * a * c)) / (2 * a)


def solve_cubic(a: float, b: float, c: float, d: float):
    """ Given the real coefficients of a cubic equation, this method will return the solutions"""
    if a == 0:
        return solve_quadratic(b, c, d)
    delta0 = b * b - 3 * a * c
    delta1 = 2 * pow(b, 3) - 9 * a * b * c + 27 * a * a * d
    deepest_root = cmath.sqrt(pow(delta1, 2) - 4 * pow(delta0, 3))
    C = (0.5 * (delta1 + deepest_root)) ** (1. / 3)
    if C == 0:  # In case
        C = (0.5 * (delta1 - deepest_root)) ** (1. / 3)
    if C == 0:  # If c is still 0 ( can't be because of zero division error after that )
        return [0]
    roots = []
    root_of_unity = complex(-0.5, sqrt(3) / 2)
    for k in range(3):
        root = -((b + C + delta0 / C) / (3 * a))
        roots.append(root)
        C *= root_of_unity

    return list({complex(round_decimal(root.real), round_decimal(root.imag)) for root in
                 roots})


# TODO: improve in next versions
def solve_cubic_real(a: float, b: float, c: float, d: float):
    roots = solve_cubic(a, b, c, d)
    if not roots:
        return []
    return [root.real for root in roots if abs(root.imag) < 0.00001]


def solve_quartic(a: float, b: float, c: float, d: float, e: float):
    if a == 0:
        return solve_cubic(b, c, d, e)
    if a != 1:  # Divide everything by a to get to the form x^4 + bx^3 + cx^2 + dx + e
        b /= a
        c /= a
        d /= a
        e /= a
        a = 1
    f = c - (3 * b ** 2 / 8)
    g = d + (b ** 3 / 8) - (b * c / 2)
    h = e - (3 * b ** 4 / 256) + (b ** 2 * c / 16) - (b * d / 4)
    three_roots = solve_cubic(1, f / 2, (f ** 2 - 4 * h) / 16, -g ** 2 / 64)
    if len(three_roots) == 1 and three_roots[0] == 0:
        return [0]
    else:
        y1, y2, y3 = three_roots
    non_zero_roots = [sol for sol in (y1, y2, y3) if sol != 0]
    if len(non_zero_roots) == 3:
        non_zero_roots = non_zero_roots[:-1]
    elif len(non_zero_roots) < 2:
        return [0]
    p, q = cmath.sqrt(non_zero_roots[0]), cmath.sqrt(non_zero_roots[1])
    r = -g / (8 * p * q)
    s = b / (4 * a)
    sol1, sol2, sol3, sol4 = p + q + r - s, p - \
        q - r - s, -p + q - r - s, -p - q + r - s
    return list({sol1, sol2, sol3, sol4})
