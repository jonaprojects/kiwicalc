from __future__ import annotations
import os
import random
import string
import warnings
from fractions import Fraction as _Rational
from typing import Union, Tuple, List, Optional, Any, Callable, Dict, Iterator, Iterable
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import cm

from kiwicalc.core.utils import clean_from_spaces, format_coefficient, format_free_number, round_decimal
from kiwicalc.expressions.poly import Poly
from kiwicalc.equations.single import (
    random_linear, random_polynomial, random_polynomial2,
    LinearEquation, QuadraticEquation, CubicEquation, QuarticEquation, PolyEquation
)
from kiwicalc.equations.system import random_linear_system, LinearSystem
from kiwicalc.geometry.points import Point, Point2D
from kiwicalc.parsing.parse_expression import __data_from_single
from kiwicalc.pdf.layout import PDFMath, PDFPlot
from kiwicalc.pdf.arrays import PDFArray, PDFMatrix, PDFVector
from kiwicalc.pdf.formatting import PDFText, format_math, format_polynomial
from kiwicalc.pdf.formatting import _equation_text, _replace_math
from kiwicalc.pdf.style import PDFStyle
from kiwicalc.pdf.theme import get_pdf_theme
from kiwicalc.pdf.blocks import PDFParagraph, PDFHeading, PDFAnswerSpace


def _function_plot(function):
    def draw(ax):
        import numpy as np
        x = np.linspace(-5, 5, 250)
        ax.plot(x, [function(value) for value in x])
        ax.axhline(0, color='gray', linewidth=.7)
        ax.axvline(0, color='gray', linewidth=.7)
        ax.set(xlabel='x', ylabel='f(x)', title='Sketch on -5 <= x <= 5')
        ax.grid(True, alpha=.3)
    return PDFPlot(draw)

def linear_from_points_exercise(get_solution=True, variable='x', lang='en', *, _with_coefficients=False):
    first_point = (random.randint(-15, 15), random.randint(-15, 15))
    second_point = (random.randint(-15, 15), random.randint(-15, 15))
    if first_point[0] == second_point[0]:
        # These exercises ask for a function y=f(x), not a vertical line.
        # Repair the collision without an unbounded random retry loop.
        x, y = second_point
        second_point = (x + 1 if x < 15 else x - 1, y)
    if first_point[1] == second_point[1]:
        first_point = (first_point[0], first_point[1] + random.randint(1, 3))
    a = _Rational(second_point[1] - first_point[1], second_point[0] - first_point[0])
    b = first_point[1] - a * first_point[0]
    exercise = f' a)    Find the linear function that passes through the points {first_point} and {second_point}.\n           b)    Is the function increasing or decreasing?\n           c)    Bonus: Sketch the function.\n           '
    a_str = format_coefficient(a) if a.denominator == 1 else f'({a})'
    for point in (first_point, second_point):
        exercise = _replace_math(exercise, str(point), f'({format_math(point[0])}, {format_math(point[1])})')
    b_str = '' if b == 0 else f'+{b}' if b > 0 else str(b)
    if a > 0:
        answer_for_b = f'Increasing, because the slope of the function is positive'
    else:
        answer_for_b = f'Decreasing, because the slope of the function is negative'
    if get_solution:
        solution = f"    a)     y = {a_str}{variable}{b_str}\n        b.    {answer_for_b}\n        c. Plot the two given points and draw the straight line through them.\n        "
        solution = _replace_math(solution, f'y = {a_str}{variable}{b_str}', 'y=' + format_polynomial([a, b], variable))
        if _with_coefficients:
            return exercise, solution, (a, b)
        return (exercise, solution)
    return exercise

def linearFromPointAndSlope_exercise(get_solution=True, variable='x', lang='en', *, _with_coefficients=False):
    my_point = (random.randint(-15, 15), random.randint(-15, 15))
    my_slope = random.randint(-15, 15)
    while my_slope == 0:
        my_slope = random.randint(-15, 15)
    exercise = f'The linear function f(x) passes through the point {my_point} and has a slope of {my_slope}.\n           a)    Find f(x).\n           b)    Find where the function intersects with the x axis.\n           c)    Bonus: Sketch the function.\n           '
    a_str = format_coefficient(my_slope)
    exercise = _replace_math(exercise, str(my_point), f'({format_math(my_point[0])}, {format_math(my_point[1])})')
    intercept = my_point[1] - my_slope * my_point[0]
    b_str = format_free_number(intercept)
    if get_solution:
        solution = f"    a)     y = {a_str}{variable}{b_str}\n        b.  {(round_decimal(-intercept / my_slope), 0)}\n        c. Plot the given point and use the slope to draw the line.\n        "
        solution = _replace_math(solution, f'y = {a_str}{variable}{b_str}', 'y=' + format_polynomial([my_slope, intercept], variable))
        if _with_coefficients:
            return exercise, solution, (my_slope, intercept)
        return (exercise, solution)
    return exercise

def linear_intersection_exercise(get_solution=True, variable='x', lang='en'):
    from kiwicalc.pdf.generators import intersection
    return intersection(get_solution, variable, lang)

def linear_system_exercise(variables, get_solution=True, digits_after: int=0, lang='en'):
    if get_solution:
        equations, solutions = random_linear_system(variables, get_solutions=get_solution, digits_after=digits_after)
    else:
        equations = random_linear_system(variables, get_solutions=get_solution, digits_after=digits_after)
    exercise = 'Solve the system of equations:\n' + '\n'.join((f'     {equation}' for equation in equations))
    parts = ['Solve the system of equations:']
    for equation in equations:
        formatted = _equation_text(str(equation))
        parts.append('\n')
        parts.extend(formatted.parts if isinstance(formatted, PDFText) else [formatted])
    exercise = PDFText(*parts, plain=exercise)
    if get_solution:
        solution = ', '.join((f'{variable}={round_decimal(value)}' for variable, value in zip(variables, solutions)))
        answer_parts = []
        for variable, value in zip(variables, solutions):
            if answer_parts:
                answer_parts.append(', ')
            answer_parts.append(PDFMath(f'{variable}={format_math(round_decimal(value))}'))
        return (exercise, PDFText(*answer_parts, plain=solution))
    return exercise

def generate_pdf_path() -> str:
    path = f'worksheet1.pdf'
    index = 1
    while os.path.isfile(path):
        index += 1
        path = f'worksheet{index}.pdf'
    return path

def worksheet(path: str=None, dtype='linear', num_of_pages: int=1, equations_per_page: int=20, get_solutions=True, digits_after=0, titles=None, *, seed=None, difficulty='medium', **layout_options):
    for name, value in [('num_of_pages', num_of_pages), ('equations_per_page', equations_per_page)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    from kiwicalc.pdf.algebra_exercises import ALGEBRA_EXERCISE_TYPES
    from kiwicalc.pdf.calculus_exercises import CALCULUS_EXERCISE_TYPES
    from kiwicalc.pdf.linear_algebra_exercises import LINEAR_ALGEBRA_EXERCISE_TYPES
    from kiwicalc.pdf.geometry_exercises import GEOMETRY_EXERCISE_TYPES
    from kiwicalc.pdf.sequence_exercises import SEQUENCE_SERIES_EXERCISE_TYPES
    if seed is not None and dtype not in ('trigo', 'log', 'intersection',
                                          *ALGEBRA_EXERCISE_TYPES, *CALCULUS_EXERCISE_TYPES,
                                          *LINEAR_ALGEBRA_EXERCISE_TYPES,
                                          *GEOMETRY_EXERCISE_TYPES,
                                          *SEQUENCE_SERIES_EXERCISE_TYPES):
        raise ValueError('seed is supported for bounded algebra, calculus, numerical-method, '
                         'linear-algebra, geometry, sequence/series, trigo, log, and '
                         'intersection worksheets')
    if path is None:
        path = generate_pdf_path()
    if dtype == 'linear':
        LinearEquation.random_worksheets(path=path, num_of_pages=num_of_pages, equations_per_page=equations_per_page, after_point=digits_after, get_solutions=get_solutions, titles=titles, **layout_options)
    elif dtype == 'quadratic':
        QuadraticEquation.random_worksheets(path=path, num_of_pages=num_of_pages, equations_per_page=equations_per_page, digits_after=digits_after, get_solutions=get_solutions, titles=titles, **layout_options)
    elif dtype == 'cubic':
        CubicEquation.random_worksheets(path=path, num_of_pages=num_of_pages, equations_per_page=equations_per_page, digits_after=digits_after, get_solutions=get_solutions, titles=titles, **layout_options)
    elif dtype == 'quartic':
        QuarticEquation.random_worksheets(path=path, num_of_pages=num_of_pages, equations_per_page=equations_per_page, digits_after=digits_after, get_solutions=get_solutions, titles=titles, **layout_options)
    elif dtype == 'polynomial':
        PolyEquation.random_worksheets(path=path, titles=titles, equations_per_page=equations_per_page, num_of_pages=num_of_pages, digits_after=digits_after, get_solutions=get_solutions, **layout_options)
    elif dtype in ALGEBRA_EXERCISE_TYPES:
        from kiwicalc.pdf.algebra_exercises import algebra_exercise
        rng = random.Random(seed)
        labels = {
            'simplify': 'Simplifying Expressions', 'expand': 'Expanding Expressions',
            'factor': 'Factoring Polynomials', 'complete_square': 'Completing the Square',
            'substitution': 'Substitution', 'linear_inequality': 'Linear Inequalities',
            'absolute_value': 'Absolute Value Equations', 'exponent_laws': 'Exponent Laws',
            'rational': 'Rational Equations', 'radical': 'Radical Equations',
            'rearrange': 'Rearranging Formulas',
        }
        page_titles = [labels[dtype]]*num_of_pages if titles is None else list(titles)
        if len(page_titles) != num_of_pages:
            raise ValueError('titles must contain one title per exercise page')
        output_titles, output_lines = [], []
        for page_title in page_titles:
            questions, answers = [], []
            for index in range(equations_per_page):
                exercise = algebra_exercise(dtype, difficulty=difficulty,
                                            with_solution=get_solutions, _rng=rng)
                questions.append(exercise.exercise.numbered(index+1))
                if exercise.solution is not None:
                    answers.append(exercise.solution.numbered(index+1, role='solution'))
            output_titles.append(page_title)
            output_lines.append(questions)
            if answers:
                output_titles.append(f'{page_title} - Solutions')
                output_lines.append(answers)
        create_pages(path, len(output_titles), output_titles, output_lines, **layout_options)
    elif dtype in SEQUENCE_SERIES_EXERCISE_TYPES:
        from kiwicalc.pdf.sequence_exercises import sequence_exercise
        rng = random.Random(seed)
        labels = {
            'identify_sequence': 'Identifying Sequences',
            'arithmetic_next_terms': 'Arithmetic Sequences: Next Terms',
            'arithmetic_nth_term': 'Arithmetic Sequences: Nth Terms',
            'arithmetic_difference': 'Arithmetic Common Differences',
            'arithmetic_sum': 'Arithmetic Series',
            'arithmetic_missing_term': 'Arithmetic Missing Terms',
            'geometric_next_terms': 'Geometric Sequences: Next Terms',
            'geometric_nth_term': 'Geometric Sequences: Nth Terms',
            'geometric_ratio': 'Geometric Common Ratios',
            'geometric_sum': 'Finite Geometric Series',
            'infinite_geometric_sum': 'Infinite Geometric Series',
            'recursive_sequence': 'Recursive Sequences', 'fibonacci': 'Fibonacci-type Sequences',
            'sigma_evaluation': 'Sigma Notation', 'sequence_limit': 'Limits of Sequences',
            'convergence_classification': 'Sequence Convergence',
            'p_series': 'P-Series', 'geometric_series_test': 'Geometric Series Tests',
            'alternating_series': 'Alternating Series', 'telescoping_series': 'Telescoping Series',
            'elementary_limit': 'Elementary Function Limits',
            'euler_limit': "Euler's Number Limits",
            'removable_limit': 'Removable-Discontinuity Limits',
            'standard_trig_limit': 'Standard Trigonometric Limits',
        }
        page_titles = [labels[dtype]]*num_of_pages if titles is None else list(titles)
        if len(page_titles) != num_of_pages:
            raise ValueError('titles must contain one title per exercise page')
        output_titles, output_lines = [], []
        for page_title in page_titles:
            questions, answers = [], []
            for index in range(equations_per_page):
                exercise = sequence_exercise(dtype, difficulty=difficulty,
                                             with_solution=get_solutions, _rng=rng)
                questions.append(exercise.exercise.numbered(index+1))
                if exercise.solution is not None:
                    answers.append(exercise.solution.numbered(index+1, role='solution'))
            output_titles.append(page_title)
            output_lines.append(questions)
            if answers:
                output_titles.append(f'{page_title} - Solutions')
                output_lines.append(answers)
        create_pages(path, len(output_titles), output_titles, output_lines, **layout_options)
    elif dtype in GEOMETRY_EXERCISE_TYPES:
        from kiwicalc.pdf.geometry_exercises import geometry_exercise
        rng = random.Random(seed)
        labels = {
            'distance': 'Distance Between Points', 'midpoint': 'Midpoints',
            'slope': 'Slopes', 'line_equation': 'Equations of Lines',
            'point_line_distance': 'Distance from a Point to a Line',
            'parallel_perpendicular': 'Parallel and Perpendicular Lines',
            'triangle_area': 'Triangle Area', 'triangle_centroid': 'Triangle Centroids',
            'pythagorean': 'The Pythagorean Theorem', 'circle_equation': 'Circle Equations',
            'arc_sector': 'Arc Length and Sector Area', 'polygon_angles': 'Polygon Angles',
            'solid_measurement': 'Solid Geometry',
            'coordinate_transformation': 'Coordinate Transformations',
            'vector_from_points': 'Vectors Between Points',
            'vector_relationship': 'Vector Relationships', 'vector_angle': 'Angles Between Vectors',
            'cross_product': 'Cross Products', 'vector_line': 'Vector Equations of Lines',
            'plane_equation': 'Equations of Planes',
        }
        page_titles = [labels[dtype]]*num_of_pages if titles is None else list(titles)
        if len(page_titles) != num_of_pages:
            raise ValueError('titles must contain one title per exercise page')
        output_titles, output_lines = [], []
        for page_title in page_titles:
            questions, answers = [], []
            for index in range(equations_per_page):
                exercise = geometry_exercise(dtype, difficulty=difficulty,
                                             with_solution=get_solutions, _rng=rng)
                questions.append(exercise.exercise.numbered(index+1))
                if exercise.solution is not None:
                    answers.append(exercise.solution.numbered(index+1, role='solution'))
            output_titles.append(page_title)
            output_lines.append(questions)
            if answers:
                output_titles.append(f'{page_title} - Solutions')
                output_lines.append(answers)
        create_pages(path, len(output_titles), output_titles, output_lines, **layout_options)
    elif dtype in LINEAR_ALGEBRA_EXERCISE_TYPES:
        from kiwicalc.pdf.linear_algebra_exercises import linear_algebra_exercise
        rng = random.Random(seed)
        labels = {
            'vector_arithmetic': 'Vector Arithmetic', 'dot_product': 'Dot Products',
            'vector_magnitude': 'Vector Magnitudes', 'unit_vector': 'Unit Vectors',
            'matrix_arithmetic': 'Matrix Arithmetic',
            'scalar_matrix': 'Scalar Matrix Multiplication',
            'matrix_multiplication': 'Matrix Multiplication',
            'determinant': 'Determinants', 'inverse_matrix': 'Inverse Matrices',
            'solve_linear_system': 'Linear Systems', 'row_reduction': 'Row Reduction',
            'rank': 'Matrix Rank', 'linear_independence': 'Linear Independence',
            'basis_coordinates': 'Coordinates in a Basis', 'eigenvalues': 'Eigenvalues',
            'eigenvector': 'Eigenvectors', 'projection': 'Vector Projections',
            'linear_transformation': 'Linear Transformations',
        }
        page_titles = [labels[dtype]]*num_of_pages if titles is None else list(titles)
        if len(page_titles) != num_of_pages:
            raise ValueError('titles must contain one title per exercise page')
        output_titles, output_lines = [], []
        for page_title in page_titles:
            questions, answers = [], []
            for index in range(equations_per_page):
                exercise = linear_algebra_exercise(dtype, difficulty=difficulty,
                                                   with_solution=get_solutions, _rng=rng)
                questions.append(exercise.exercise.numbered(index+1))
                if exercise.solution is not None:
                    answers.append(exercise.solution.numbered(index+1, role='solution'))
            output_titles.append(page_title)
            output_lines.append(questions)
            if answers:
                output_titles.append(f'{page_title} - Solutions')
                output_lines.append(answers)
        create_pages(path, len(output_titles), output_titles, output_lines, **layout_options)
    elif dtype in CALCULUS_EXERCISE_TYPES:
        from kiwicalc.pdf.calculus_exercises import calculus_exercise
        rng = random.Random(seed)
        labels = {
            'difference_quotient': 'Derivatives from First Principles',
            'derivative': 'Differentiation', 'tangent_line': 'Tangent Lines',
            'critical_points': 'Critical Points', 'monotonicity': 'Monotonicity',
            'concavity': 'Concavity and Inflection', 'optimization': 'Optimization',
            'definite_integral': 'Definite Integrals', 'area_between': 'Area Between Curves',
            'numerical_derivative': 'Numerical Differentiation',
            'trapezoidal_rule': 'The Trapezoidal Rule', 'simpson_rule': "Simpson's Rule",
            'newton_iteration': "Newton's Method", 'euler_method': "Euler's Method",
            'runge_kutta': 'Runge-Kutta Method',
        }
        page_titles = [labels[dtype]]*num_of_pages if titles is None else list(titles)
        if len(page_titles) != num_of_pages:
            raise ValueError('titles must contain one title per exercise page')
        output_titles, output_lines = [], []
        for page_title in page_titles:
            questions, answers = [], []
            for index in range(equations_per_page):
                exercise = calculus_exercise(dtype, difficulty=difficulty,
                                             with_solution=get_solutions, _rng=rng)
                questions.append(exercise.exercise.numbered(index+1))
                if exercise.solution is not None:
                    answers.append(exercise.solution.numbered(index+1, role='solution'))
            output_titles.append(page_title)
            output_lines.append(questions)
            if answers:
                output_titles.append(f'{page_title} - Solutions')
                output_lines.append(answers)
        create_pages(path, len(output_titles), output_titles, output_lines, **layout_options)
    elif dtype in ('trigo', 'log', 'intersection'):
        from kiwicalc.pdf import generators
        factory = {'trigo': generators.trigonometric, 'log': generators.logarithmic,
                   'intersection': generators.intersection}[dtype]
        rng = random.Random(seed)
        page_titles = ['Worksheet']*num_of_pages if titles is None else list(titles)
        if len(page_titles) != num_of_pages:
            raise ValueError('titles must contain one title per exercise page')
        output_titles, output_lines = [], []
        for page_title in page_titles:
            questions, answers = [], []
            for index in range(equations_per_page):
                value = factory(get_solution=get_solutions, rng=rng)
                prompt, answer = value if get_solutions else (value, None)
                questions.append(prompt.numbered(index+1) if isinstance(prompt, PDFText) else f'{index+1}. {prompt}')
                if answer is not None:
                    answers.append(answer.numbered(index+1, role='solution') if isinstance(answer, PDFText) else f'{index+1}. {answer}')
            output_titles.append(page_title)
            output_lines.append(questions)
            if answers:
                output_titles.append(f'{page_title} - Solutions')
                output_lines.append(answers)
        create_pages(path, len(output_titles), output_titles, output_lines, **layout_options)
    else:
        choices = ('linear', 'quadratic', 'cubic', 'quartic', 'polynomial',
                   'trigo', 'log', 'intersection', *ALGEBRA_EXERCISE_TYPES,
                   *CALCULUS_EXERCISE_TYPES, *LINEAR_ALGEBRA_EXERCISE_TYPES,
                   *GEOMETRY_EXERCISE_TYPES, *SEQUENCE_SERIES_EXERCISE_TYPES)
        raise ValueError(f"worksheet(): unknown dtype {dtype}: expected one of {', '.join(choices)}")

def create_pdf(path: str, title='Worksheet', lines=(), **layout_options) -> bool:
    """Create a numbered worksheet; preserve the legacy boolean failure contract."""
    try:
        numbered = [PDFParagraph(line, number=index+1, role='question') for index, line in enumerate(lines)]
        create_pages(path, 1, [title], [numbered], **layout_options)
        return True
    except Exception as ex:
        warnings.warn(f"Couldn't create the pdf file: {type(ex).__name__}: {ex}")
        return False


def create_pages(path: str, num_of_pages: int, titles, lines, **layout_options):
    """Render logical pages with automatic wrapping and overflow pagination.

    Layout options: page_size='A4' (or 'Letter'), margin=50 and font_size=12,
    measured in PDF points, plus line_height=1.5 (a font-size multiplier).
    Explicit page boundaries are preserved.
    """
    if isinstance(num_of_pages, bool) or not isinstance(num_of_pages, int) or num_of_pages < 0:
        raise ValueError('num_of_pages must be a nonnegative integer')
    titles, lines = list(titles), list(lines)
    if len(titles) != num_of_pages or len(lines) != num_of_pages:
        raise ValueError('titles and lines must each contain one entry per page')
    from kiwicalc.pdf.layout import render_pages
    render_pages(os.fspath(path), titles, lines, **layout_options)

class PDFExercise:
    """
    This class represents an exercise in a PDF page.
    """
    __slots__ = ['__exercise', '__exercise_type', '__dtype', '__solution', '__number', '__lang', 'solution_plot']

    def __init__(self, exercise: str, exercise_type: str, dtype: str, solution=None, number=None, lang='en'):
        self.__exercise = exercise
        self.__exercise_type = exercise_type
        self.__dtype = dtype
        # One-shot answer iterators must survive repeated worksheet rendering.
        self.__solution = tuple(solution) if isinstance(solution, Iterator) else solution
        self.__number = number
        self.__lang = lang
        self.solution_plot = None

    @property
    def exercise(self):
        return self.__exercise

    @property
    def number(self):
        return self.__number

    @number.setter
    def number(self, number: int):
        self.__number = number

    @property
    def dtype(self):
        return self.__dtype

    @property
    def solution(self):
        return self.__solution

    @property
    def has_solution(self):
        return self.__solution is not None

    @property
    def lang(self):
        return self.__lang

    def __str__(self):
        return self.__exercise

class PDFCalculusExercise(PDFExercise):

    def __init__(self, exercise, dtype, solution=None, lang='en'):
        super(PDFCalculusExercise, self).__init__(exercise, 'calculus', dtype, solution, lang=lang)

class PDFAnalyzeFunction(PDFCalculusExercise):

    def __init__(self, exercise, dtype: str, solution=None, lang='en'):
        super(PDFAnalyzeFunction, self).__init__(exercise, dtype=dtype, solution=solution, lang=lang)

class PDFLinearFunction(PDFAnalyzeFunction):

    def __init__(self, with_solution: bool=True, lang: str='en'):
        my_linear, solution, coefficients = random_linear(get_solution=True, get_coefficients=True)
        random_function = f'f(x) = {my_linear}'
        exercise = f' The linear function {random_function} is given.\n            a) Where does the function intersect with the x axis?\n            b) Where does the function intersect with the y axis?\n            c) Is the function increasing or decreasing?\n            d) What is the derivative of the function?\n            e) Sketch the function.\n        '
        if with_solution:
            if coefficients[0] > 0:
                answer_for_c = f'Increasing, because the slope of the function is positive'
            else:
                answer_for_c = f'Decreasing, because the slope of the function is negative'
            solution = f"    a)    ({solution}, 0)\n            b)    (0, {coefficients[1]})\n            c) {answer_for_c}\n            d) f'(x) = {coefficients[0]}\n            e) See the sketch below.\n             "
        else:
            solution = None
        exercise = _replace_math(exercise, random_function, 'f(x)=' + format_polynomial(coefficients))
        if with_solution:
            solution = _replace_math(solution, f"f'(x) = {coefficients[0]}", r'f\prime(x)=' + format_math(coefficients[0]))
        super(PDFAnalyzeFunction, self).__init__(exercise, dtype='linear', solution=solution, lang=lang)
        if with_solution:
            self.solution_plot = _function_plot(lambda x: coefficients[0]*x+coefficients[1])

class PDFLinearIntersection(PDFExercise):

    def __init__(self, with_solution=True, lang='en', *, seed=None):
        from kiwicalc.pdf.generators import intersection
        result = intersection(with_solution, lang=lang, rng=random.Random(seed), details=True)
        prompt, answer, data = result if with_solution else (result, None, None)
        super().__init__(prompt, 'equation', 'intersection', solution=answer, lang=lang)
        if data is not None:
            a, b, c, d, x, y = data
            def draw(ax):
                import numpy as np
                values = np.linspace(x-3, x+3, 100)
                ax.plot(values, a*values+b, label='first line')
                ax.plot(values, c*values+d, label='second line')
                ax.scatter([x], [y], color='crimson')
                ax.set(xlabel='x', ylabel='y')
                ax.grid(True, alpha=.3)
                ax.legend()
            self.solution_plot = PDFPlot(draw)


class PDFTrigonometricEquation(PDFExercise):
    """Special-angle sine/cosine equation in degrees, on [0,360)."""
    def __init__(self, with_solution=True, number=None, lang='en', *, seed=None):
        from kiwicalc.pdf.generators import trigonometric
        result = trigonometric(with_solution, lang=lang, rng=random.Random(seed))
        prompt, answer = result if with_solution else (result, None)
        super().__init__(prompt, 'equation', 'trigo', solution=answer, number=number, lang=lang)


class PDFLogarithmicEquation(PDFExercise):
    """Real logarithmic equation with a positive-argument domain check."""
    def __init__(self, with_solution=True, number=None, lang='en', *, seed=None):
        from kiwicalc.pdf.generators import logarithmic
        result = logarithmic(with_solution, lang=lang, rng=random.Random(seed))
        prompt, answer = result if with_solution else (result, None)
        super().__init__(prompt, 'equation', 'log', solution=answer, number=number, lang=lang)

class PDFLinearSystem(PDFExercise):

    def __init__(self, with_solution=True, lang='en', num_of_equations=None, digits_after: int=0):
        if num_of_equations is None:
            num_of_equations = random.randint(2, 3)
        if isinstance(num_of_equations, bool) or not isinstance(num_of_equations, int) or num_of_equations < 1:
            raise ValueError('num_of_equations must be a positive integer')
        variables = ['x', 'y', 'z', 'm', 'n', 't', 'a', 'b']
        num_of_variables = num_of_equations
        if num_of_variables <= len(variables):
            variables = variables[:num_of_variables]
        elif num_of_variables <= 26:
            variables = string.ascii_lowercase[:num_of_variables]
        else:
            raise ValueError('The system does not support systems of equations with more than 26 equations.')
        result = linear_system_exercise(variables, get_solution=with_solution, digits_after=digits_after)
        if with_solution:
            exercise, solution = result
        else:
            exercise, solution = (result, None)
        super(PDFLinearSystem, self).__init__(exercise, exercise_type='system of equations', dtype='linear', solution=solution, lang=lang)

class PDFLinearFromPoints(PDFAnalyzeFunction):

    def __init__(self, with_solution: bool=True, lang: str='en'):
        result = linear_from_points_exercise(get_solution=with_solution, lang=lang, _with_coefficients=True)
        if with_solution:
            exercise, solution = result[:2]
        else:
            exercise, solution = (result, None)
        super(PDFLinearFromPoints, self).__init__(exercise, dtype='linear', solution=solution, lang=lang)
        if with_solution and len(result) == 3:
            a, b = result[2]
            self.solution_plot = _function_plot(lambda x: float(a)*x+float(b))

class PDFLinearFromPointAndSlope(PDFAnalyzeFunction):

    def __init__(self, with_solution: bool=True, lang: str='en'):
        result = linearFromPointAndSlope_exercise(get_solution=with_solution, lang=lang, _with_coefficients=True)
        if with_solution:
            exercise, solution = result[:2]
        else:
            exercise, solution = (result, None)
        super(PDFLinearFromPointAndSlope, self).__init__(exercise, dtype='linear', solution=solution, lang=lang)
        if with_solution and len(result) == 3:
            a, b = result[2]
            self.solution_plot = _function_plot(lambda x: a*x+b)

class PDFPolyFunction(PDFAnalyzeFunction):

    def __init__(self, with_solution: bool=True, degree: int=None, lang: str='en'):
        if degree is None:
            degree = random.randint(2, 5)
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
            raise ValueError('degree must be a positive integer')
        random_poly, solutions = random_polynomial(degree=degree, get_solutions=True)
        random_function = f'f(x) = {random_poly}'
        exercise = f' The function {random_function} is given.\n            a) What is the domain of the function?\n            b) What is the derivative of the function?\n            c) What are the extremums of the function?\n            d) When is the function increasing, and when is it decreasing?\n            e) Find the horizontal asymptotes of the function (if there are any).\n            f) sketch the function.\n        '
        if with_solution:
            my_poly = Poly(random_poly)
            data = my_poly.data(no_roots=True)
            data['roots'] = solutions
            extremums_string = ', '.join((extremum.__str__() for extremum in data['extremums']))
            if not extremums_string:
                extremums_string = 'None'
            solution = f"\n            a. Domain: all real numbers\n            b. Derivative: {data['derivative']}\n            c. Extremums: {extremums_string}\n            d. Increase & Decrease: Increase: {data['up']}, Decrease: {data['down']}\n            e. Horizontal Asymptotes: None (nonconstant polynomial).\n            f. See the sketch below.\n             "
        else:
            solution = None
        exercise = _replace_math(exercise, random_function, 'f(x)=' + format_math(Poly(random_poly)))
        if with_solution:
            solution = _replace_math(solution, str(data['derivative']), format_math(data['derivative']))
        super(PDFPolyFunction, self).__init__(exercise, dtype='poly', solution=solution, lang=lang)
        if with_solution:
            self.solution_plot = _function_plot(my_poly.to_lambda())

class PDFQuadraticFunction(PDFPolyFunction):

    def __init__(self, with_solution: bool=True, lang: str='en'):
        super(PDFQuadraticFunction, self).__init__(with_solution=with_solution, degree=2, lang=lang)

class PDFCubicFunction(PDFPolyFunction):

    def __init__(self, with_solution: bool=True, lang: str='en'):
        super(PDFCubicFunction, self).__init__(with_solution=with_solution, degree=3, lang=lang)

class PDFQuarticFunction(PDFPolyFunction):

    def __init__(self, with_solution: bool=True, lang: str='en'):
        super(PDFQuarticFunction, self).__init__(with_solution=with_solution, degree=4, lang=lang)

class PDFEquationExercise(PDFExercise):

    def __init__(self, exercise: str, dtype: str, solution=None, number: int=None):
        super(PDFEquationExercise, self).__init__(_equation_text(exercise), 'equation', dtype, solution, number)

class PDFLinearEquation(PDFEquationExercise):

    def __init__(self, with_solution=True, number: int=None):
        if with_solution:
            equation, solution = LinearEquation.random_equation(digits_after=1, get_solution=True)
        else:
            equation, solution = (LinearEquation.random_equation(digits_after=1, get_solution=False), None)
        super(PDFLinearEquation, self).__init__(equation, dtype='linear', solution=solution, number=number)

class PDFQuadraticEquation(PDFEquationExercise):

    def __init__(self, with_solution=True, number: int=None):
        if with_solution:
            equation, solutions = random_polynomial(degree=2, get_solutions=True)
        else:
            equation, solutions = (random_polynomial(degree=2), None)
        equation += ' = 0'
        super(PDFQuadraticEquation, self).__init__(equation, dtype='quadratic', solution=solutions, number=number)

class PDFCubicEquation(PDFEquationExercise):

    def __init__(self, with_solution=True, number: int=None):
        if with_solution:
            equation, solutions = random_polynomial(degree=3, get_solutions=True)
        else:
            equation, solutions = (random_polynomial(degree=3), None)
        equation += ' = 0'
        super(PDFCubicEquation, self).__init__(equation, dtype='cubic', solution=solutions, number=number)

class PDFQuarticEquation(PDFEquationExercise):

    def __init__(self, with_solution=True, number: int=None):
        if with_solution:
            equation, solutions = random_polynomial(degree=4, get_solutions=True)
        else:
            equation, solutions = (random_polynomial(degree=4), None)
        equation += ' = 0'
        super(PDFQuarticEquation, self).__init__(equation, dtype='quartic', solution=solutions, number=number)

class PDFPolyEquation(PDFEquationExercise):

    def __init__(self, with_solution=True, number: int=None):
        if with_solution:
            equation, solutions = random_polynomial(degree=random.randint(2, 5), get_solutions=True)
        else:
            equation, solutions = (random_polynomial(degree=random.randint(2, 5)), None)
        equation += ' = 0'
        super(PDFPolyEquation, self).__init__(equation, dtype='poly', solution=solutions, number=number)

class PDFPage:

    def __init__(self, title='Worksheet', exercises=None):
        self.__title = title
        if exercises is None:
            self.__exercises = []
        else:
            self.__exercises = list(exercises)

    @property
    def exercises(self):
        return self.__exercises

    @property
    def title(self):
        return self.__title

    def add(self, exercise):
        self.__exercises.append(exercise)

    def __iter__(self):
        return iter(self.__exercises)

class PDFWorksheet:
    """Manual worksheet composition with page objects as the source of truth.

    end_page() enables an answer page for the current exercise page. Repeating
    it refreshes that page instead of adding duplicates. create() refreshes all
    enabled answer pages and renders current exercise content, without caching
    stale lines. Adding after end_page() still adds to the exercise page.
    """
    __slots__ = ['__pages', '__ordered', '__current_page', '__title', '__answer_pages', 'style']

    def __init__(self, title='Worksheet', ordered=True, *, style=None, theme=None):
        if style is not None and not isinstance(style, PDFStyle):
            raise TypeError('style must be a PDFStyle')
        if style is not None and theme is not None:
            raise ValueError('Pass either style or theme, not both')
        self.style = get_pdf_theme(theme).to_style() if theme is not None else style
        self.__pages = [PDFPage(title=title)]
        self.__ordered = ordered
        self.__current_page = self.__pages[0]
        self.__title = title
        self.__answer_pages = {}

    @property
    def num_of_pages(self):
        return len(self.__pages)

    @property
    def pages(self):
        return self.__pages

    @property
    def current_page(self):
        return self.__current_page

    def _renumber(self):
        number = 0
        answer_pages = set(self.__answer_pages.values())
        for page in self.__pages:
            if page in answer_pages:
                continue
            for exercise in page.exercises:
                if isinstance(exercise, (PDFMath, PDFArray, PDFPlot, PDFHeading, PDFAnswerSpace)):
                    continue
                if not isinstance(exercise, PDFExercise):
                    raise TypeError('Worksheet pages must contain PDFExercise objects')
                number += 1
                if self.__ordered:
                    exercise.number = number

    def del_last_page(self):
        if not self.__pages:
            return
        removed = self.__pages.pop()
        for source, answer in list(self.__answer_pages.items()):
            if removed is source or removed is answer:
                del self.__answer_pages[source]
                if removed is source and answer in self.__pages:
                    self.__pages.remove(answer)
        answers = set(self.__answer_pages.values())
        self.__current_page = next((page for page in reversed(self.__pages) if page not in answers), None)
        self._renumber()

    def add_exercise(self, exercise):
        if not isinstance(exercise, PDFExercise):
            raise TypeError('exercise must be a PDFExercise')
        if self.__current_page is None:
            self.next_page()
        self.__current_page.add(exercise)
        self._renumber()
        if self.__current_page in self.__answer_pages:
            self._refresh_answers(self.__current_page)
        return self

    def add_math(self, expression, *, font_size=None):
        if self.__current_page is None:
            self.next_page()
        self.__current_page.add(PDFMath(expression, font_size))
        return self

    def add_matrix(self, values, *, brackets='square', font_size=None):
        """Add a centered matrix from KiwiCalc, NumPy, or nested-sequence data."""
        if self.__current_page is None:
            self.next_page()
        self.__current_page.add(PDFMatrix(values, brackets=brackets, font_size=font_size))
        return self

    def add_vector(self, values, *, orientation='column', brackets='round', font_size=None):
        """Add a centered row or column vector and return this worksheet."""
        if self.__current_page is None:
            self.next_page()
        self.__current_page.add(PDFVector(values, orientation=orientation,
                                          brackets=brackets, font_size=font_size))
        return self

    def add_plot(self, source, *, height=180, caption=None):
        if self.__current_page is None:
            self.next_page()
        self.__current_page.add(PDFPlot(source, height, caption=caption))
        return self

    def add_heading(self, text, *, level=1):
        """Add an unnumbered section heading, kept with its following content."""
        if self.__current_page is None:
            self.next_page()
        self.__current_page.add(PDFHeading(text, level))
        return self

    def add_answer_space(self, height=72, *, pattern='lines', spacing=18):
        """Add an unnumbered blank, ruled, or gridded writing area."""
        if self.__current_page is None:
            self.next_page()
        self.__current_page.add(PDFAnswerSpace(height, pattern, spacing))
        return self

    def _text_lines(self, text, number=None, role='question'):
        if isinstance(text, PDFExercise):
            text = text.exercise
        return [PDFParagraph(text, number=number if self.__ordered else None, role=role)]

    def _refresh_answers(self, source):
        lines = []
        for exercise in source.exercises:
            if isinstance(exercise, (PDFMath, PDFArray, PDFPlot, PDFHeading, PDFAnswerSpace)):
                continue
            solution = exercise.solution
            if solution is None:
                continue
            if not isinstance(solution, (int, float, str)) and isinstance(solution, Iterable):
                values = list(solution)
                plain = ','.join(str(value) for value in values)
                if values and all(isinstance(value, (int, float, _Rational)) and not isinstance(value, bool) for value in values):
                    solution = PDFText(PDFMath(', '.join(format_math(value) for value in values)), plain=plain)
                else:
                    solution = plain
            elif isinstance(solution, (int, float, _Rational)) and not isinstance(solution, bool):
                solution = PDFText(PDFMath(solution), plain=str(solution))
            lines.extend(self._text_lines(solution, exercise.number, role='solution'))
            if exercise.solution_plot is not None:
                if not isinstance(exercise.solution_plot, PDFPlot):
                    raise TypeError('solution_plot must be a PDFPlot')
                lines.append(exercise.solution_plot)
        answer = self.__answer_pages.get(source)
        if not lines:
            if answer is not None:
                self.__pages.remove(answer)
                del self.__answer_pages[source]
            return
        if answer is None:
            answer = PDFPage(title='Solutions')
            self.__answer_pages[source] = answer
            self.__pages.insert(self.__pages.index(source) + 1, answer)
        answer.exercises[:] = lines

    def end_page(self):
        if self.__current_page is not None:
            self._renumber()
            self._refresh_answers(self.__current_page)
        return self

    def next_page(self, title=None):
        if title is None:
            title = self.__title
        self.__current_page = PDFPage(title)
        self.__pages.append(self.__current_page)
        return self

    def create(self, path: str=None, **layout_options):
        """Export the worksheet; e.g. create(path, line_height=1.25).

        line_height defaults to 1.5 for text and headings. Tall inline math may
        require extra height. Paragraph and block spacing remain independent.
        """
        if path is None:
            path = generate_pdf_path()
        titles, lines = self._render_content()
        if 'style' not in layout_options and 'theme' not in layout_options and self.style is not None:
            layout_options['style'] = self.style
        create_pages(path, len(titles), titles, lines, **layout_options)

    def _render_content(self):
        """Prepare current question and answer content for a shared renderer."""
        if not self.__pages:
            raise ValueError('Cannot create a worksheet with no pages')
        self._renumber()
        for source in list(self.__answer_pages):
            self._refresh_answers(source)
        answers = set(self.__answer_pages.values())
        lines = []
        for page in self.__pages:
            if page in answers:
                lines.append(list(page.exercises))
            else:
                lines.append([line for exercise in page.exercises
                              for line in ([exercise] if isinstance(exercise, (PDFMath, PDFArray, PDFPlot, PDFHeading, PDFAnswerSpace))
                                           else self._text_lines(exercise, exercise.number))])
        return [page.title for page in self.__pages], lines
