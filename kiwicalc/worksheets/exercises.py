class PDFCalculusExercise(PDFExercise):
    def __init__(self, exercise, dtype, solution=None, lang="en"):
        super(PDFCalculusExercise, self).__init__(
            exercise, "calculus", dtype, solution, lang=lang)


class PDFAnalyzeFunction(PDFCalculusExercise):
    def __init__(self, exercise, dtype: str, solution=None, lang="en"):
        super(PDFAnalyzeFunction, self).__init__(
            exercise, dtype=dtype, solution=solution, lang=lang)


class PDFLinearFunction(PDFAnalyzeFunction):
    def __init__(self, with_solution: bool = True, lang: str = 'en'):
        if lang != 'en':  # Translate the exercise for other languages if needed!
            translator = Translator()
        my_linear, solution, coefficients = random_linear(
            get_solution=True, get_coefficients=True)
        random_function = f"f(x) = {my_linear}"
        exercise = f""" The linear function {random_function} is given.
            a) Where does the function intersect with the x axis?
            b) Where does the function intersect with the y axis?
            c) Is the function increasing or decreasing?
            d) What is the derivative of the function?
            e) Sketch the function.
        """

        if lang != 'en':  # Translate the exercise for other languages if needed!
            translation = translator.translate(exercise, dest=lang)
            exercise = translation.text
            print(exercise)
        if with_solution:
            if coefficients[0] > 0:
                answer_for_c = f"Increasing, because the slope of the function is positive"
            else:
                answer_for_c = f"Decreasing, because the slope of the function is negative"
            solution = f"""    a)    ({solution}, 0)
            b)    (0, {coefficients[1]})
            c) {answer_for_c}
            d) f'(x) = {coefficients[0]}
            e) Sketch not supported yet!
             """
            if lang != 'en':
                translation = translator.translate(solution, dest=lang)
                solution = translation.text
        else:
            solution = None

        super(PDFAnalyzeFunction, self).__init__(
            exercise, dtype='linear', solution=solution, lang=lang)


class PDFLinearIntersection(PDFExercise):
    def __init__(self, with_solution=True, lang='en'):
        pass


class PDFLinearSystem(PDFExercise):
    def __init__(self, with_solution=True, lang='en', num_of_equations=None, digits_after: int = 0):
        if num_of_equations is None:
            num_of_equations = random.randint(2, 3)

        variables = ['x', 'y', 'z', 'm', 'n', 't', 'a', 'b']
        num_of_variables = num_of_equations
        if num_of_variables <= len(variables):
            variables = variables[:num_of_variables]
        elif num_of_variables <= 26:
            variables = string.ascii_lowercase[:num_of_variables]
        else:
            raise ValueError(
                "The system does not support systems of equations with more than 26 equations.")
        result = linear_system_exercise(
            variables, get_solution=with_solution, digits_after=digits_after)
        if with_solution:
            exercise, solution = result
        else:
            exercise, solution = result, None
        super(PDFLinearSystem, self).__init__(exercise, exercise_type="system of equations", dtype='linear',
                                              solution=solution, lang=lang)


class PDFLinearFromPoints(PDFAnalyzeFunction):
    def __init__(self, with_solution: bool = True, lang: str = "en"):
        result = linear_from_points_exercise(
            get_solution=with_solution, lang=lang)
        if with_solution:
            exercise, solution = result
        else:
            exercise, solution = result, None

        super(PDFLinearFromPoints, self).__init__(
            exercise, dtype='linear', solution=solution, lang=lang)


class PDFLinearFromPointAndSlope(PDFAnalyzeFunction):
    def __init__(self, with_solution: bool = True, lang: str = "en"):
        result = linearFromPointAndSlope_exercise(
            get_solution=with_solution, lang=lang)
        if with_solution:
            exercise, solution = result
        else:
            exercise, solution = result, None

        super(PDFLinearFromPointAndSlope, self).__init__(
            exercise, dtype='linear', solution=solution, lang=lang)


class PDFPolyFunction(PDFAnalyzeFunction):
    def __init__(self, with_solution: bool = True, degree: int = None, lang: str = 'en'):
        if lang != 'en':  # Translate the exercise for other languages if needed!
            translator = Translator()
        if degree is None:
            degree = random.randint(2, 5)
        random_poly, solutions = random_polynomial(
            degree=degree, get_solutions=True)
        random_function = f"f(x) = {random_poly}"
        exercise = f""" The function {random_function} is given.
            a) What is the domain of the function?
            b) What is the derivative of the function?
            c) What are the extremums of the function?
            d) When is the function increasing, and when is it decreasing?
            e) Find the horizontal asymptotes of the function (if there are any).
            f) sketch the function.
        """
        if lang != 'en':  # Translate the exercise for other languages if needed!
            translation = translator.translate(exercise, dest=lang)
            exercise = translation.text
            print(exercise)
        if with_solution:
            my_poly = Poly(random_poly)
            data = my_poly.data(no_roots=True)
            data['roots'] = solutions
            extremums_string = ", ".join(extremum.__str__()
                                         for extremum in data['extremums'])
            if not extremums_string:
                extremums_string = 'None'
            solution = f"""
            a. Domain: all 
            b. Derivative: {data['derivative']}
            c. Extremums: {extremums_string}
            d. Increase & Decrease: Increase: {data['up']}, Decrease: {data['down']}
            e. Horizontal Asymptotes: Not Supported yet
            f. Sketch: Not supported yet in this format.
             """
            if lang != 'en':
                translation = translator.translate(solution, dest=lang)
                solution = translation.text
        else:
            solution = None
        super(PDFPolyFunction, self).__init__(
            exercise, dtype='poly', solution=solution, lang=lang)


class PDFQuadraticFunction(PDFPolyFunction):
    def __init__(self, with_solution: bool = True, lang: str = "en"):
        super(PDFQuadraticFunction, self).__init__(
            with_solution=with_solution, degree=2, lang=lang)


class PDFCubicFunction(PDFPolyFunction):
    def __init__(self, with_solution: bool = True, lang: str = "en"):
        super(PDFCubicFunction, self).__init__(
            with_solution=with_solution, degree=3, lang=lang)


class PDFQuarticFunction(PDFPolyFunction):
    def __init__(self, with_solution: bool = True, lang: str = "en"):
        super(PDFQuarticFunction, self).__init__(
            with_solution=with_solution, degree=4, lang=lang)


class PDFEquationExercise(PDFExercise):
    def __init__(self, exercise: str, dtype: str, solution=None, number: int = None):
        super(PDFEquationExercise, self).__init__(
            exercise, "equation", dtype, solution, number)


class PDFLinearEquation(PDFEquationExercise):
    def __init__(self, with_solution=True, number: int = None):
        if with_solution:
            equation, solution = LinearEquation.random_equation(
                digits_after=1, get_solution=True)
        else:
            equation, solution = LinearEquation.random_equation(
                digits_after=1, get_solution=False), None

        super(PDFLinearEquation, self).__init__(
            equation, dtype='linear', solution=solution, number=number)


class PDFQuadraticEquation(PDFEquationExercise):
    def __init__(self, with_solution=True, number: int = None):
        if with_solution:
            equation, solutions = random_polynomial(
                degree=2, get_solutions=True)
        else:
            equation, solutions = random_polynomial(degree=2), None
        equation += " = 0"
        super(PDFQuadraticEquation, self).__init__(
            equation, dtype='quadratic', solution=solutions, number=number)


class PDFCubicEquation(PDFEquationExercise):
    def __init__(self, with_solution=True, number: int = None):
        if with_solution:
            equation, solutions = random_polynomial(
                degree=3, get_solutions=True)
        else:
            equation, solutions = random_polynomial(degree=3), None
        equation += " = 0"
        super(PDFCubicEquation, self).__init__(
            equation, dtype='cubic', solution=solutions, number=number)


class PDFQuarticEquation(PDFEquationExercise):
    def __init__(self, with_solution=True, number: int = None):
        if with_solution:
            equation, solutions = random_polynomial(
                degree=4, get_solutions=True)
        else:
            equation, solutions = random_polynomial(degree=4), None
        equation += " = 0"
        super(PDFQuarticEquation, self).__init__(
            equation, dtype='quartic', solution=solutions, number=number)


class PDFPolyEquation(PDFEquationExercise):
    def __init__(self, with_solution=True, number: int = None):
        if with_solution:
            equation, solutions = random_polynomial(
                degree=random.randint(2, 5), get_solutions=True)
        else:
            equation, solutions = random_polynomial(
                degree=random.randint(2, 5)), None
        equation += " = 0"
        super(PDFPolyEquation, self).__init__(
            equation, dtype='poly', solution=solutions, number=number)


def linear_from_points_exercise(get_solution=True, variable='x', lang="en"):
    if lang != 'en':
        translator = Translator()
    first_point = (random.randint(-15, 15), random.randint(-15, 15))
    second_point = (random.randint(-15, 15), random.randint(-15, 15))
    if first_point[1] == second_point[1]:
        first_point = first_point[0], first_point[1] + random.randint(1, 3)
    a = round_decimal(
        (second_point[1] - first_point[1]) / (second_point[0] - first_point[0]))
    b = round_decimal(first_point[1] - a * first_point[0])
    exercise = f""" a)    Find the linear function that passes through the points {first_point} and {second_point}.
           b)    Is the function increasing or decreasing?
           c)    Bonus: Sketch the function.
           """
    if lang != 'en':
        translation = translator.translate(exercise, dest=lang)
        exercise = translation.text
    a_str = format_coefficient(round_decimal(a))
    b_str = format_free_number(b)
    if a > 0:
        answer_for_b = f"Increasing, because the slope of the function is positive"
    else:
        answer_for_b = f"Decreasing, because the slope of the function is negative"

    if get_solution:
        solution = f"""    a)     y = {a_str}{variable}{b_str}
        b.    {answer_for_b}
        c. Sketching isn't supported yet
        """
        if lang != 'en':
            translation = translator.translate(solution, dest=lang)
            solution = translation.text

        return exercise, solution

    return exercise


def linearFromPointAndSlope_exercise(get_solution=True, variable='x', lang="en"):
    if lang != 'en':
        translator = Translator()
    my_point = (random.randint(-15, 15), random.randint(-15, 15))
    my_slope = random.randint(-15, 15)
    while my_slope == 0:
        my_slope = random.randint(-15, 15)

    exercise = f"""The linear function f(x) passes through the point {my_point} and has a slope of {my_slope}.
           a)    Find f(x).
           b)    Find where the function intersects with the x axis.
           c)    Bonus: Sketch the function.
           """
    if lang != 'en':
        translation = translator.translate(exercise, dest=lang)
        exercise = translation.text
    a_str = format_coefficient(my_slope)
    b_str = format_free_number(my_point[1] - my_slope * my_point[0])
    if get_solution:
        solution = f"""    a)     y = {a_str}{variable}{b_str}
        b.  {round_decimal(-my_point[1] / my_slope), 0}
        c. Sketching isn't supported yet
        """
        if lang != 'en':
            translation = translator.translate(solution, dest=lang)
            solution = translation.text

        return exercise, solution

    return exercise


def linear_intersection_exercise(get_solution=True, variable='x', lang='en'):
    pass


def linear_system_exercise(variables, get_solution=True, digits_after: int = 0, lang='en'):
    if lang != 'en':
        translator = Translator()
    if get_solution:
        equations, solutions = random_linear_system(
            variables, get_solutions=get_solution, digits_after=digits_after)
    else:
        equations = random_linear_system(
            variables, get_solutions=get_solution, digits_after=digits_after)

    exercise = """Solve the system of equations:\n""" + \
        "\n".join(f"     {equation}" for equation in equations)
    if lang != 'en':
        exercise = translator.translate(exercise, dest=lang).text

    if get_solution:
        solution = ", ".join(
            f"{variable}={round_decimal(value)}" for variable, value in zip(variables, solutions))
        return exercise, solution
    return exercise


