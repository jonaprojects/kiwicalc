from __future__ import annotations
import os
import random
import string
import warnings
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

def linear_from_points_exercise(get_solution=True, variable='x', lang='en'):
    first_point = (random.randint(-15, 15), random.randint(-15, 15))
    second_point = (random.randint(-15, 15), random.randint(-15, 15))
    if first_point[1] == second_point[1]:
        first_point = (first_point[0], first_point[1] + random.randint(1, 3))
    a = round_decimal((second_point[1] - first_point[1]) / (second_point[0] - first_point[0]))
    b = round_decimal(first_point[1] - a * first_point[0])
    exercise = f' a)    Find the linear function that passes through the points {first_point} and {second_point}.\n           b)    Is the function increasing or decreasing?\n           c)    Bonus: Sketch the function.\n           '
    a_str = format_coefficient(round_decimal(a))
    b_str = format_free_number(b)
    if a > 0:
        answer_for_b = f'Increasing, because the slope of the function is positive'
    else:
        answer_for_b = f'Decreasing, because the slope of the function is negative'
    if get_solution:
        solution = f"    a)     y = {a_str}{variable}{b_str}\n        b.    {answer_for_b}\n        c. Sketching isn't supported yet\n        "
        return (exercise, solution)
    return exercise

def linearFromPointAndSlope_exercise(get_solution=True, variable='x', lang='en'):
    my_point = (random.randint(-15, 15), random.randint(-15, 15))
    my_slope = random.randint(-15, 15)
    while my_slope == 0:
        my_slope = random.randint(-15, 15)
    exercise = f'The linear function f(x) passes through the point {my_point} and has a slope of {my_slope}.\n           a)    Find f(x).\n           b)    Find where the function intersects with the x axis.\n           c)    Bonus: Sketch the function.\n           '
    a_str = format_coefficient(my_slope)
    b_str = format_free_number(my_point[1] - my_slope * my_point[0])
    if get_solution:
        solution = f"    a)     y = {a_str}{variable}{b_str}\n        b.  {(round_decimal(-my_point[1] / my_slope), 0)}\n        c. Sketching isn't supported yet\n        "
        return (exercise, solution)
    return exercise

def linear_intersection_exercise(get_solution=True, variable='x', lang='en'):
    pass

def linear_system_exercise(variables, get_solution=True, digits_after: int=0, lang='en'):
    if get_solution:
        equations, solutions = random_linear_system(variables, get_solutions=get_solution, digits_after=digits_after)
    else:
        equations = random_linear_system(variables, get_solutions=get_solution, digits_after=digits_after)
    exercise = 'Solve the system of equations:\n' + '\n'.join((f'     {equation}' for equation in equations))
    if get_solution:
        solution = ', '.join((f'{variable}={round_decimal(value)}' for variable, value in zip(variables, solutions)))
        return (exercise, solution)
    return exercise

def generate_pdf_path() -> str:
    path = f'worksheet1.pdf'
    index = 1
    while os.path.isfile(path):
        index += 1
        path = f'worksheet{index}.pdf'
    return path

def worksheet(path: str=None, dtype='linear', num_of_pages: int=1, equations_per_page: int=20, get_solutions=True, digits_after=0, titles=None):
    if path is None:
        path = generate_pdf_path()
    if dtype == 'linear':
        LinearEquation.random_worksheets(path=path, num_of_pages=num_of_pages, equations_per_page=equations_per_page, after_point=digits_after, get_solutions=get_solutions, titles=titles)
    elif dtype == 'quadratic':
        QuadraticEquation.random_worksheets(path=path, num_of_pages=num_of_pages, equations_per_page=equations_per_page, digits_after=digits_after, get_solutions=get_solutions, titles=titles)
    elif dtype == 'cubic':
        CubicEquation.random_worksheets(path=path, num_of_pages=num_of_pages, equations_per_page=equations_per_page, digits_after=digits_after, get_solutions=get_solutions, titles=titles)
    elif dtype == 'quartic':
        QuarticEquation.random_worksheets(path=path, num_of_pages=num_of_pages, equations_per_page=equations_per_page, digits_after=digits_after, get_solutions=get_solutions, titles=titles)
    elif dtype == 'polynomial':
        PolyEquation.random_worksheets(path=path, titles=titles, equations_per_page=equations_per_page, num_of_pages=num_of_pages, digits_after=digits_after, get_solutions=get_solutions)
    elif dtype == 'trigo':
        pass
    elif dtype == 'log':
        pass
    else:
        raise ValueError(f"worksheet(): unknown dtype {dtype}: expected 'linear', 'quadratic', 'cubic', 'quartic', 'polynomial', 'trigo', 'log' ")

def create_pdf(path: str, title='Worksheet', lines=()) -> bool:
    try:
        c = Canvas(os.fspath(path))
        c.setFontSize(22)
        c.drawString(50, 800, title)
        textobject = c.beginText(2 * cm, 26 * cm)
        c.setFontSize(14)
        for index, line in enumerate(lines):
            textobject.textLine(f'{index + 1}. {line.strip()}')
            textobject.textLine('')
        c.drawText(textobject)
        c.showPage()
        c.save()
        return True
    except Exception as ex:
        warnings.warn(f"Couldn't create the pdf file due to a {ex.__class__} error")
        return False

def create_pages(path: str, num_of_pages: int, titles, lines):
    c = Canvas(os.fspath(path))
    for i in range(num_of_pages):
        c.setFontSize(22)
        c.drawString(50, 800, titles[i])
        textobject = c.beginText(2 * cm, 26 * cm)
        c.setFontSize(14)
        for index, line in enumerate(lines[i]):
            textobject.textLine(f'{lines[i][index]}')
            textobject.textLine('')
        c.drawText(textobject)
        c.showPage()
    c.save()

class PDFExercise:
    """
    This class represents an exercise in a PDF page.
    """
    __slots__ = ['__exercise', '__exercise_type', '__dtype', '__solution', '__number', '__lang']

    def __init__(self, exercise: str, exercise_type: str, dtype: str, solution=None, number=None, lang='en'):
        self.__exercise = exercise
        self.__exercise_type = exercise_type
        self.__dtype = dtype
        self.__solution = solution
        self.__number = number
        self.__lang = lang

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
            solution = f"    a)    ({solution}, 0)\n            b)    (0, {coefficients[1]})\n            c) {answer_for_c}\n            d) f'(x) = {coefficients[0]}\n            e) Sketch not supported yet!\n             "
        else:
            solution = None
        super(PDFAnalyzeFunction, self).__init__(exercise, dtype='linear', solution=solution, lang=lang)

class PDFLinearIntersection(PDFExercise):

    def __init__(self, with_solution=True, lang='en'):
        pass

class PDFLinearSystem(PDFExercise):

    def __init__(self, with_solution=True, lang='en', num_of_equations=None, digits_after: int=0):
        if num_of_equations is None:
            num_of_equations = random.randint(2, 3)
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
        result = linear_from_points_exercise(get_solution=with_solution, lang=lang)
        if with_solution:
            exercise, solution = result
        else:
            exercise, solution = (result, None)
        super(PDFLinearFromPoints, self).__init__(exercise, dtype='linear', solution=solution, lang=lang)

class PDFLinearFromPointAndSlope(PDFAnalyzeFunction):

    def __init__(self, with_solution: bool=True, lang: str='en'):
        result = linearFromPointAndSlope_exercise(get_solution=with_solution, lang=lang)
        if with_solution:
            exercise, solution = result
        else:
            exercise, solution = (result, None)
        super(PDFLinearFromPointAndSlope, self).__init__(exercise, dtype='linear', solution=solution, lang=lang)

class PDFPolyFunction(PDFAnalyzeFunction):

    def __init__(self, with_solution: bool=True, degree: int=None, lang: str='en'):
        if degree is None:
            degree = random.randint(2, 5)
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
            solution = f"\n            a. Domain: all \n            b. Derivative: {data['derivative']}\n            c. Extremums: {extremums_string}\n            d. Increase & Decrease: Increase: {data['up']}, Decrease: {data['down']}\n            e. Horizontal Asymptotes: Not Supported yet\n            f. Sketch: Not supported yet in this format.\n             "
        else:
            solution = None
        super(PDFPolyFunction, self).__init__(exercise, dtype='poly', solution=solution, lang=lang)

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
        super(PDFEquationExercise, self).__init__(exercise, 'equation', dtype, solution, number)

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
            self.__exercises = exercises

    @property
    def exercises(self):
        return self.__exercises

    @property
    def title(self):
        return self.__title

    def add(self, exercise):
        self.__exercises.append(exercise)

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):
        if self.__index >= len(self.__exercises):
            raise StopIteration
        temp = self.__index
        self.__index += 1
        return self.__exercises[temp]

class PDFWorksheet:
    __slots__ = ['__pages', '__ordered', '__current_page', '__lines', '__title', '__num_of_exercises']

    def __init__(self, title='Worksheet', ordered=True):
        self.__pages = [PDFPage(title=title)]
        self.__ordered = ordered
        self.__current_page = self.__pages[0]
        self.__lines = [[]]
        self.__title = title
        self.__num_of_exercises = 0

    @property
    def num_of_pages(self):
        return len(self.__pages)

    @property
    def pages(self):
        return self.__pages

    def del_last_page(self):
        if len(self.__pages):
            del self.__pages[-1]

    @property
    def current_page(self):
        return self.__current_page

    def add_exercise(self, exercise):
        self.__num_of_exercises += 1
        self.__current_page.add(exercise)
        if '\n' in exercise.__str__():
            lines = exercise.__str__().split('\n')
            if self.__ordered:
                exercise.number = self.__num_of_exercises
                self.__lines[-1].append(f'{exercise.number}.    {lines[0]}')
            else:
                self.__lines[-1].append(lines[0])
            for i in range(1, len(lines)):
                self.__lines[-1].append(lines[i])
            self.__lines[-1].append('')
        elif self.__ordered:
            exercise.number = self.__num_of_exercises
            self.__lines[-1].append(f'{exercise.number}.    {exercise.__str__()}')
        else:
            self.__lines[-1].append(exercise.__str__())

    def end_page(self):
        if any((exercise.has_solution for exercise in self.__current_page.exercises)):
            solutions_string = []
            for index, exercise in enumerate(self.__current_page.exercises):
                if exercise.solution is None:
                    continue
                if not isinstance(exercise.solution, (int, float, str)) and isinstance(exercise.solution, Iterable):
                    str_solution = ','.join((str(solution) for solution in exercise.solution))
                    if self.__ordered:
                        solutions_string.append(f'{exercise.number}.    {str_solution}')
                    else:
                        solutions_string.append(str_solution)
                else:
                    if not isinstance(exercise.solution, str):
                        str_solution = str(exercise.solution)
                    else:
                        str_solution = exercise.solution
                    if '\n' in str_solution:
                        lines = exercise.solution.split('\n')
                        solutions_string.append(f'{exercise.number}. {lines[0]}' if self.__ordered else f'{lines[0]}')
                        for j in range(1, len(lines)):
                            solutions_string.append(lines[j])
                        solutions_string.append('')
                    elif self.__ordered:
                        solutions_string.append(f'{exercise.number}.    {exercise.solution}')
                    else:
                        solutions_string.append(f'{exercise.solution}')
            self.__pages.append(PDFPage(title='Solutions', exercises=solutions_string))
            self.__lines.append(solutions_string)

    def next_page(self, title=None):
        if title is None:
            title = self.__title
        self.__pages.append(PDFPage(title))
        self.__current_page = self.__pages[-1]
        self.__lines.append([])

    def create(self, path: str=None):
        if path is None:
            path = generate_pdf_path()
        create_pages(path, self.num_of_pages, [page.title for page in self.__pages], self.__lines)
