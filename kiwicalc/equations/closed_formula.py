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
