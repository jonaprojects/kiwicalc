def random_linear_system(variables, solutions_range: Tuple[int, int] = (-10, 10),
                         coefficients_range: Tuple[int, int] = (-10, 10),
                         digits_after=0, get_solutions=False):
    equations = []
    num_of_equations = len(variables)
    solutions = [round(random.uniform(solutions_range[0], solutions_range[1]), digits_after) for _ in
                 range(num_of_equations)]
    for _ in range(num_of_equations):
        coefficients_dict = dict()
        equation_sum = 0
        # Create the basics: ax + by + cz ... = equation_sum
        for index, variable in enumerate(variables):
            random_coefficient = random.randint(
                coefficients_range[0], coefficients_range[1])
            coefficients_dict[variable] = [random_coefficient]
            equation_sum += random_coefficient * solutions[index]
        # Complicate the equation
        free_number = random.randint(
            coefficients_range[0], coefficients_range[1])
        equation_sum += free_number

        other_side_dict = {variable: [] for variable in variables}
        # TODO: customize these parameters
        num_of_operations = random.randint(2, 5)
        for _ in range(num_of_operations):
            operation_index = random.randint(0, 1)
            if operation_index == 0:  # Addition - add an expression such as '3x', '-5y', etc, to both sides
                random_variable = random.choice(variables)
                random_coefficient = random.randint(
                    coefficients_range[0], coefficients_range[1])
                coefficients_dict[random_variable].append(random_coefficient)
                other_side_dict[random_variable].append(random_coefficient)
            else:  # Multiplication - multiply both sides by a number
                # Also 1, to perhaps prevent very large numbers.
                random_num = random.randint(1, 3)
                for variable, coefficients in coefficients_dict.items():
                    for index in range(len(coefficients)):
                        coefficients[index] *= random_num
                free_number *= random_num
                for variable, coefficients in other_side_dict.items():
                    for index in range(len(coefficients)):
                        coefficients[index] *= random_num
                equation_sum *= random_num
        equations.append(
            f"{ParseExpression.unparse_linear(coefficients_dict, free_number)}={ParseExpression.unparse_linear(other_side_dict, equation_sum)}")
    if get_solutions:
        return equations, solutions
    return equations


def random_poly_system(variables):
    pass


def random_linear(coefs_range=(-15, 15), digits_after: int = 0, variable='x', get_solution: bool = False,
                  get_coefficients: bool = False):
    """
    Generates a random linear expression in the form ax+b
    :param coefs_range: the range from which the coefficients will be chosen randomly
    :param digits_after: the maximum number of digits after the decimal point for the coefficients
    :param variable: the variable that will appear in the string
    :param get_solution: whether to return also the solution
    :param get_coefficients: whether to return also the coefficients, (a, b)
    :return:
    """
    a = round_decimal(round(random.uniform(
        coefs_range[0], coefs_range[1]), digits_after))
    while a == 0:
        a = round_decimal(round(random.uniform(
            coefs_range[0], coefs_range[1]), digits_after))

    b = round_decimal(round(random.uniform(
        coefs_range[0], coefs_range[1]), digits_after))
    a_str = format_coefficient(round_decimal(a))
    b_str = format_free_number(b)

    if get_solution:
        if get_coefficients:
            return f"{a_str}{variable}{b_str}", round_decimal(-b / a), (a, b)
        return f"{a_str}{variable}{b_str}", round_decimal(-b / a)
    elif get_coefficients:
        return f"{a_str}{variable}{b_str}", (a, b)

    return f"{a_str}{variable}{b_str}"


def random_polynomial(degree: int = None, solutions_range=(-5, 5), digits_after=0, variable='x', python_syntax=False,
                      get_solutions=False):
    if degree is None:
        degree = random.randint(2, 9)
    a = round_decimal(round(random.uniform(
        solutions_range[0], solutions_range[1]), digits_after))
    while a == 0:
        a = round_decimal(round(random.uniform(
            solutions_range[0], solutions_range[1]), digits_after))
    accumulator = [f'{format_coefficient(a)}x**{degree}'] if python_syntax else [
        f'{format_coefficient(a)}x^{degree}']
    solutions = {round_decimal(round(random.uniform(solutions_range[0], solutions_range[1]), digits_after)) for _ in
                 range(degree)}
    permutations_length = 1
    for i in range(degree):
        current_permutations = set(
            tuple(sorted(per)) for per in permutations(solutions, permutations_length))
        current_sum = 0
        for permutation in current_permutations:
            current_sum += reduce(operator.mul, permutation)
        if current_sum != 0:
            current_power = degree - permutations_length
            coefficient = format_coefficient(
                round_decimal(current_sum * a)) if current_power != 0 else f"{round_decimal(current_sum * a)}"
            if coefficient != "" and coefficient[0] not in ('+', '-'):
                coefficient = f"+{coefficient}"
            if current_power == 0:
                accumulator.append(f"{coefficient}")
            elif current_power == 1:
                accumulator.append(f"{coefficient}{variable}")
            else:
                accumulator.append(f"{coefficient}{variable}^{current_power}")
        permutations_length += 1
    equation = "".join(accumulator)
    if get_solutions:
        return equation, [-solution for solution in solutions]
    return equation


def random_polynomial2(degree: int, values=(-15, 15), digits_after=0, variable='x', python_syntax=False):
    a = round_decimal(round(random.uniform(
        values[0], values[1]), digits_after))
    while a == 0:
        a = round_decimal(round(random.uniform(
            values[0], values[1]), digits_after))
    accumulator = []
    while a == 0:
        a = round_decimal(round(random.uniform(
            values[0], values[1]), digits_after))
    accumulator.append(f"{format_coefficient(a)}{variable}^{degree}")
    for index in range(1, degree - 1):
        m = round_decimal(round(random.uniform(
            values[0], values[1]), digits_after))
        coef_str = format_coefficient(m)
        if coef_str:
            if coef_str[0] not in ('+', '-'):
                coef_str = f'+{coef_str}'
        power = degree - 1
        power_str = f'^{power}' if power != 1 else f''
        if python_syntax:
            pass
        else:
            accumulator.append(f"{coef_str}{variable}{power_str}")
    m = round_decimal(round(random.uniform(
        values[0], values[1]), digits_after))
    accumulator.append(f"+{round_decimal(m)}" if m >
                       0 else f"{m}") if m != 0 else ""
    return "".join(accumulator)
