from kiwicalc import *

"""
This file contains all of the examples in the documentation and the google colab.
The examples are divided here to classes by the different topics.
"""


class LinearEquation_Examples:
    @staticmethod
    def solve_linear_example():
        linear_equation = "3x + 7 = -x -5"
        result = solve_linear(linear_equation)
        print(result)

    @staticmethod
    def solve_linear_example1():
        linear_equation = "3x + 7 = -x -5"
        result = solve_linear(linear_equation, ('x',))
        print(result)

    @staticmethod
    def linearEquation_init_example():
        my_equation = LinearEquation("3x + 5 = 8", variables=('x',))

    @staticmethod
    def linearEquation_solution_example():
        my_equation = LinearEquation("3x + 5 = 8", variables=('x',))
        print(my_equation.solution)
        print(my_equation.solve())

    @staticmethod
    def linearEquation_simplify_example():
        my_equation = LinearEquation("4 + 3x + 2y + 5x + 6 = 2 + x - y", variables=('x','y'))
        my_equation.simplify()
        print(my_equation)

    @staticmethod
    def linearEquation_showSteps_example():
        my_equation = LinearEquation("3x - 7 + 2x + 6 = 4x - 4 + 6x")
        print(my_equation.show_steps())

    @staticmethod
    def linearEquation_plotSolution_example():
        my_equation = LinearEquation("3x - 7 + 2x + 6 = 4x - 4 + 6x")
        print(my_equation.plot_solution())


class LinearSystem_Examples:
    @staticmethod
    def solve_linear_system_example():
        solutions = solve_linear_system(("3x - 4y + 3 = -z + 9", "-3x + 5 -2z = 2y - 9 + x", "2x + 4y - z = 4"))
        print(solutions)

    @staticmethod
    def linearSystem_solution_example():
        linear_system = LinearSystem(("3x - 4y + 3 = -z + 9", "-3x + 5 -2z = 2y - 9 + x", "2x + 4y - z = 4"))
        print(linear_system.get_solutions())


class QuadraticEquation_Examples:
    @staticmethod
    def solve_quadratic_example1():
        solutions = solve_quadratic(1, 6, 8)
        real_solutions = solve_quadratic_real(1, 6, 8)
        print(solutions)
        print(real_solutions)

    @staticmethod
    def solve_quadratic_example2():
        solutions2 = solve_quadratic(1, 1, 2)
        real_solutions2 = solve_quadratic_real(1, 1, 2)
        print(solutions2)
        print(real_solutions2)

    @staticmethod
    def solve_quadratic_params_example():
        pass

    @staticmethod
    def quadraticEquation_solution_example():
        my_equation = QuadraticEquation("x^2 + 6x + 8 = 0")
        solution = my_equation.solve()
        print(solution)

    @staticmethod
    def quadraticEquation_realSolutions_example():
        my_equation = QuadraticEquation("x^2 + 6x + 8 = 0")
        solution = my_equation.solve('real')
        print(solution)

    @staticmethod
    def quadraticEquation_simplified_str_example():
        my_equation = QuadraticEquation("x^2 + 2x + 4x + 8 = 0")
        print(my_equation)
        print(my_equation.simplified_str())

    @staticmethod
    def quadraticEquation_coefficients_example():
        my_equation = QuadraticEquation("x^2 + 6x + 8 = 0")
        print(my_equation.coefficients())

    @staticmethod
    def quadraticEquation_randomEquation_example1():
        print(QuadraticEquation.random())

    @staticmethod
    def quadraticEquation_randomEquation_example2():
        print(QuadraticEquation.random(digits_after=1, variable='y'))

    @staticmethod
    def quadraticEquation_randomWorksheet_example():
        print(QuadraticEquation.random_worksheet("worksheet1.pdf", title="Example"))

    @staticmethod
    def quadraticEquation_randomWorksheets_example():
        print(QuadraticEquation.random_worksheets("worksheet2.pdf", num_of_pages=5))


class CubicEquation_Examples:
    @staticmethod
    def solve_cubic_example():
        solutions = solve_cubic(1, 3, -4, -8)
        real_solutions = solve_cubic_real(1, 3, -4, -8)
        print(solutions)
        print(real_solutions)

    @staticmethod
    def cubicEquation_mixed_examples():
        my_equation = CubicEquation("x^3 + 3x^2 - 4x - 8 = 0")
        print(my_equation.solution)
        print(my_equation.coefficients())
        print(my_equation.random())


class QuarticEquation_Examples:

    @staticmethod
    def solve_quartic_example():
        solution = solve_quartic(1, 0, 0, 0, -16)  # x^2 - 16 = 0
        print(solution)

    @staticmethod
    def quarticEquation_example():
        my_equation = QuarticEquation("x^4 - 16 = 0")
        print(my_equation.solution)
        print(my_equation.coefficients())
        print(my_equation.random())


class PolyEquation_Examples:
    @staticmethod
    def solve_polynomial_example():
        print(solve_polynomial([1, 0, 0, 0, 0, 0, 2, 0, 0, -18]))
        print(solve_polynomial("x^2 + 3x^3 + 12x + 5 = 2x -4 + x^2"))
        print(solve_polynomial("x^2 + 6x + 8"))

    @staticmethod
    def polyEquation_example():
        poly_equation = PolyEquation("x^2 - 8x = 15")
        print(poly_equation.solution)


class PolySystem_Examples:
    @staticmethod
    def solve_poly_system_example():
        solutions = solve_poly_system(["x^2 + y^2 = 25", "2x + 3y = 18"], {'x': 2, 'y': 1})
        print(solutions)


class PlottingMethods_Examples:
    @staticmethod
    def scatter_dots_example():
        scatter_dots([1, 2, 3, 4], [2, 4, 6, 8], title="Matplotlib is awesome")

    @staticmethod
    def scatter_dots_3d_example():
        scatter_dots_3d([4, 2, 8, -5, 3, 8, 9, -6, 1], [1, 9, -2, 3, 4, 5, 1, 6, 7], [1, -1, 3, 4, 8, 1, 4, 2, 6])

    @staticmethod
    def plot_function_example():
        plot_function("f(x) = x^2")
        plot_function(Function("f(x) = sin(x)"))
        plot_function(lambda x: 2 * x)

    @staticmethod
    def plot_function_3d_example():
        plot_function_3d("f(x,y) = sin(x)*cos(y)")

    @staticmethod
    def scatter_function_example():
        scatter_function("f(x) = sin(x)")
        scatter_function(lambda x: x ** 2)
        scatter_function(Function("f(x) = 2x"))

    @staticmethod
    def scatter_function_3d_example():
        scatter_function_3d("f(x,y) = x + y")
        scatter_function_3d(lambda x, y: sin(x) * cos(y))
        scatter_function_3d(Function("f(x,y) = xy"), title="Hi there!")

    @staticmethod
    def plot_functions_example():
        plot_functions(["f(x) = 2x", "f(x) = x^2", "f(x) = sin(x)"])

    @staticmethod
    def plot_functions_3d_example():
        plot_functions_3d(["f(x,y) = sin(x)*cos(y)", "f(x,y) = sin(x)*ln(y)"])

    @staticmethod
    def plot_multiple_example():
        plot_multiple(
            ["f(x) = 4x", "f(x) = x^2", "f(x) = x^3", "f(x)= 8", "f(x)=ln(x)", "f(x)=e^x", "f(x)=|x|", "f(x)=sin(x)",
             "f(x)=cos(x)"])


    @staticmethod
    def plot_complex_example():
        plot_complex(complex(5, 4), complex(3, -2))


class Function_Examples:
    @staticmethod
    def function_init_example():
        sine = Function("f(x)=sin(x)")
        sine = Function("lambda x:sin(x)")
        sine = Function("x => sin(x)")
        three_sum = Function("x, y, z => x + y + z")
        x = Var('x')
        func = Function(-x ** 2 + 6 * x + 7)
        sine = Function(lambda x: sin(x))
        print(sine(pi / 2))

    @staticmethod
    def function_call_example():
        parabola = Function("f(x)=x**2")
        print(parabola(5))
        three_sum = Function("g(a,b,c)=a+b+c")
        print(three_sum(6, 5, 4))
        equality = Function("f(a,b) = a==b")
        print(equality(5, 5))
        print(equality(4, 6))
        trigo_op = Function("f(x) = -sin(x) + 2cos(2x)")
        print(trigo_op(pi / 2))
        three_sum = Function("g(a,b,c)=a+b+c")
        print(three_sum.variables)
        print(three_sum[0])
        print(three_sum[1:3])

    @staticmethod
    def function_derivative_example():
        func = Function("f(x)=x**2+2x-5")
        print(func.derivative())

    @staticmethod
    def function_partialDerivative_example():
        func = Function("f(x,y) = x^2 * y^2")
        print(func.partial_derivative('x'))
        print(func.partial_derivative('y'))
        print(func.partial_derivative('xy'))

    @staticmethod
    def function_plot_example():
        fn = Function("f(x) = x^2")
        fn.plot(start=-10, stop=10, step=0.1)

    @staticmethod
    def function_scatter_examples():
        fn = Function("f(x) = x^2")
        fn.scatter2d()

        fn1 = Function("f(x,y) = sin(x) * cos(y)")
        fn1.scatter3d()

        fn2 = Function("f(x) = x^2")
        fn2.scatter2d(basic=False)

    @staticmethod
    def function_context_manager_example():
        with Function("f(x) = x^2") as fn:
            pass  # Do some stuff here


class FunctionCollection_Examples:
    @staticmethod
    def functionCollection_init_example():
        functions = FunctionCollection(Function("f(x) = x^2"), Function("g(x) = sin(x)"), Function("h(x) = 2x"))
        functions = FunctionCollection("f(x) = x^2", "g(x) = sin(x)", "h(x) = 2x")
        functions = FunctionCollection(Function("f(x) =x^2"), "g(x) = sin(x)", Function("h(x) = 2x"))

    @staticmethod
    def functionCollection_add_function_example(self):
        functions = FunctionCollection(Function("f(x) = x^2"), Function("g(x) = sin(x)"), Function("h(x) = 2x"))
        functions.add_function(Function("f(x) = 2x"))
        functions.extend(["f(x,y) = x+y", "g(x,y) = cos(x)+cos(y)", "h(x,y) = sin(x)*sin(y)"])

    @staticmethod
    def functionCollection_clear_example(self):
        functions = FunctionCollection(Function("f(x) = x^2"), Function("g(x) = sin(x)"), Function("h(x) = 2x"))
        print(functions)
        functions.clear()
        print(functions)

    @staticmethod
    def functionCollection_is_empty_example1():
        functions = FunctionCollection()
        print(functions.is_empty())
        functions.add_function("f(x) = ln(x)")
        print(functions.is_empty())

    @staticmethod
    def functionCollection_is_empty_example2():
        functions = FunctionCollection(Function("f(x) =x^2"), "g(x) = sin(x)", Function("h(x) = 2x"))
        functions.clear()
        print(functions.is_empty())

    @staticmethod
    def functionCollection_random_function_example():
        functions = FunctionCollection(Function("f(x) =x^2"), "g(x) = sin(x)", Function("h(x) = 2x"))
        print(functions.random_function())

    @staticmethod
    def functionCollection_random_value_example1():
        functions = FunctionCollection(Function("f(x) =x^2"), "g(x) = sin(x)", Function("h(x) = 2x"))
        print(functions.random_value(1, 10))

    @staticmethod
    def functionCollection_random_value_example2():
        functions = FunctionCollection("f(x,y) = x + y", "g(x,y) = x - y", "h(x,y) = sin(x) * cos(y)")
        print(functions.random_value(1, 10))

    @staticmethod
    def functionCollection_plot_example():
        functions = FunctionCollection("f(x,y) = x + y", "g(x,y) = x - y", "h(x,y) = sin(x) * cos(y)")
        functions.plot()

    @staticmethod
    def functionCollection_scatter_example():
        functions = FunctionCollection("f(x,y) = x + y", "g(x,y) = x - y", "h(x,y) = sin(x) * cos(y)")
        functions.scatter()

    @staticmethod
    def functionCollection_len_example():
        functions = FunctionCollection("f(x) = 2x", "g(x) = x^2", "h(x) = sin(x)")
        print(len(functions))


class FunctionChain_Examples:
    @staticmethod
    def functionChain_init_example():
        function_chain = FunctionChain("f(x) = x^2", "g(x) = x+5", "h(x) = sin(x)")
        function_chain = Function("f(x) = x^2").chain(Function("f(x)=x+5")).chain(Function("h(x) = sin(x)"))

    @staticmethod
    def functionChain_execute_example():
        function_chain = FunctionChain("f(x) = x^2", "g(x) = x+5", "h(x) = sin(x)")
        print(function_chain.execute_all(1))
        print(function_chain(1))

    @staticmethod
    def functionChain_execute_custom():
        function_chain = FunctionChain("f(x) = x^2", "g(x) = x+5", "h(x) = sin(x)")
        print(function_chain.execute_reverse())  # execute in reverse order
        print(function_chain.execute_indices([3, 1, 0, 2]))

    @staticmethod
    def functionChain_plot_example():
        function_chain = FunctionChain("f(x) = x^2", "g(x) = x+5", "h(x) = sin(x)")
        function_chain.plot()

    @staticmethod
    def functionChain_scatter_example():
        function_chain = FunctionChain("f(x) = x^2", "g(x) = x+5", "h(x) = sin(x)")
        function_chain.plot()

    @staticmethod
    def functionChain_chain_example():
        function_chain = FunctionChain("f(x) = x^2", "g(x) = x+5")
        function_chain.chain("h(x) = sin(x)")

    @staticmethod
    def functionChain_index_example():
        function_chain = FunctionChain("f(x) = x^2", "g(x) = x+5")
        print(function_chain[0])


class PDFWorksheets_Examples:

    @staticmethod
    def worksheet_example():
        worksheet()
        worksheet("linearWorkheet.pdf", num_of_pages=10)
        worksheet("myWorksheet.pdf", dtype='quadratic')

    @staticmethod
    def linearFunction_exercise_example():
        exercise = PDFLinearFunction()

    @staticmethod
    def linearSystem_exercise_example():
        exercise = PDFLinearSystem()

    @staticmethod
    def linearFromPoints_exercise_example():
        exercise = PDFLinearFromPoints()

    @staticmethod
    def linearFromPointAndSlope_exercise_example():
        exercise = PDFLinearFromPointAndSlope()

    @staticmethod
    def polyFunction_exercise_example():
        exercise = PDFPolyFunction()

    @staticmethod
    def quadraticFunction_exercise_example():
        exercise = PDFQuadraticFunction()

    @staticmethod
    def cubicFunction_exercise_example():
        exercise = PDFCubicFunction()

    @staticmethod
    def quarticFunction_exercise_example():
        exercise = PDFQuarticFunction()

    @staticmethod
    def linearEquation_exercise_example():
        exercise = PDFLinearEquation()

    @staticmethod
    def quadraticEquation_exercise_example():
        exercise = PDFQuadraticEquation()

    @staticmethod
    def cubicEquation_exercise_example():
        exercise = PDFCubicEquation()

    @staticmethod
    def quarticEquation_exercise_example():
        exercise = PDFQuarticEquation()

    @staticmethod
    def polyEquation_exercise_example():
        exercise = PDFPolyEquation()

    @staticmethod
    def pdf_full_example1():
        pdf_worksheet = PDFWorksheet("Functions")
        pdf_worksheet.add_exercise(PDFCubicFunction(lang='es'))
        pdf_worksheet.add_exercise(PDFQuadraticFunction())
        pdf_worksheet.add_exercise(PDFLinearFromPointAndSlope())
        pdf_worksheet.end_page()
        pdf_worksheet.next_page("Equations")
        for _ in range(10):
            pdf_worksheet.add_exercise(PDFLinearEquation())
            pdf_worksheet.add_exercise(PDFQuadraticEquation())
        pdf_worksheet.end_page()
        pdf_worksheet.create("worksheetExample.pdf")


class ExpressionSum_Examples:

    @staticmethod
    def expressionSum_init_example():
        x = Var('x')
        expressions = 3 * x + Ln(x) + Sin(x)
        other_expressions = ExpressionSum([3 * x, Ln(x), Sin(x)])

    @staticmethod
    def expressionSum_addition_example():
        x = Var('x')
        expressions = Sin(x) + Ln(x)
        other_expressions = Cos(x) + 2 * x
        print(expressions + other_expressions)

    @staticmethod
    def expressionSum_subtraction_example():
        x = Var('x')
        expressions = Sin(x) - Ln(x)
        expressions -= Cos(x)

    @staticmethod
    def expressionSum_multiplication_example():
        x, y = Var('x'), Var('y')
        one = Sin(x) + Cos(x)
        two = 3 * x
        print(one * two)

    @staticmethod
    def expressionSum_division_example():
        x = Var('x')
        print((x ** 2 + Sin(x)) / x)

    @staticmethod
    def expressionSum_power_example():
        x = Var('x')
        print((x + Sin(x)) ** 2)

    @staticmethod
    def expressionSum_tryEvaluate_example():
        x = Var('x')
        my_expressions = Sin(x) + Ln(x) + 5
        my_expressions.assign(x=4)
        print(my_expressions.try_evaluate())

    @staticmethod
    def expressionSum_to_lambda_example():
        x = Var('x')
        my_expressions = Sin(x) + Ln(x) + 4
        print(my_expressions.to_lambda())

    @staticmethod
    def expressionSum_plot_example1():
        x = Var('x')
        result = Sin(x) * Cos(Ln(x ** 3 - 6)) + Root(2 * x - 5)
        result.plot()

    @staticmethod
    def expressionSum_scatter_example2():
        x = Var('x')
        y = Var('y')
        result = Sin(x) + Cos(y) * Ln(x)
        result.plot()

    @staticmethod
    def expressionSum_scatter_example():
        x = Var('x')
        y = Var('y')
        result = Sin(x) + Cos(y) * Ln(x)
        result.scatter()


class Mono_Examples:
    @staticmethod
    def mono_init_example():
        expression = Mono(5, {'x': 2, 'y': 3})
        print(expression)
        expression = Mono("3x^2*y^4")
        print(expression)
        four = Mono(4)
        print(four)
        x = Var('x')
        mono_expression = 3 * x ** 2

    @staticmethod
    def mono_arithmetics_example():
        first_mono, second_mono = Mono("3x^2"), Mono("2x^2")
        print(first_mono + second_mono)
        print(first_mono - second_mono)
        print(first_mono * second_mono)
        print(first_mono / second_mono)

    @staticmethod
    def mono_assign_example1():
        m = Mono(coefficient=3, variables_dict={'x': 2, 'y': 1})  # 3x^2*y
        m.assign(x=4)
        print(m)
        m.assign(y=3)
        print(m)

    @staticmethod
    def mono_assign_example2():
        m = Mono(coefficient=3, variables_dict={'x': 2, 'y': 1})  # 3x^2*y
        m.assign(x=4, y=3)
        print(m)

    @staticmethod
    def mono_when_example():
        m = Mono(coefficient=3, variables_dict={'x': 2, 'y': 1})  # creating the expression
        assigned_expression = m.when(x=4, y=3)  # saving the assigned expression without modifying the original.
        print(assigned_expression)
        print(m)

    @staticmethod
    def mono_tryEvaluate_example1():
        m = Mono(5)
        print(m.try_evaluate())

    @staticmethod
    def mono_tryEvaluate_example2():
        m = Mono(coefficient=2, variables_dict={'x': 4})  # creating 2*x^4
        assigned_expression = m.when(x=3)  # assigning x=3 without changing the original object
        print(f"Value is {assigned_expression} and the type is {type(assigned_expression)}")
        evaluated_number = assigned_expression.try_evaluate()
        print(f"Value is {evaluated_number} and type is {type(evaluated_number)}")

    @staticmethod
    def mono_plot_example():
        my_mono = x ** 2
        my_mono.plot()

    @staticmethod
    def mono_scatter_example():
        my_mono = x ** 2
        my_mono.scatter()


class Var_Examples:

    @staticmethod
    def var_init_example():
        x = Var('x')
        y = Var('y')
        z = Var('z')

    @staticmethod
    def var_arithmetics_example():
        x = Var('x')
        y = Var('y')
        print((x - 5) ** 2)
        print((x + y) * (x - y))
        print((x + y) ** 2)
        print((x + y + 5) ** 3)

    @staticmethod
    def var_derivative_and_integral_example():
        print((3 * x ** 2).derivative())
        print((6 * x).integral())

    @staticmethod
    def var_equal_example():
        x = Var('x')
        y = Var('y')
        print(3 * y == 2 * x + 6)
        print((2 * x + 2 * y) / 2 == y + (2 * x ** 2) / (2 * x))
        print(2 * x + 5 != 3 * y * x ** 2 - 4)


class Poly_Examples:

    @staticmethod
    def poly_init_example():
        x, y = Var('x'), Var('y')
        polynomial = Poly((3 * x ** 2, 2.6, "4y^2", 7 * x ** 4 * y ** 5))
        polynomial2 = Poly((Mono(2, {'x': 3, 'y': 4}), Mono("4xy^3")))
        expression = Poly("8 + 3x^2 + 6x + 9 + 2x^3 + 2x^2")
        print(expression)

    @staticmethod
    def poly_init_copy_example():
        x = Var('x')
        existing_polynomial = 3 * x ** 2 - 6 * x + 7
        new_polynomial = Poly(existing_polynomial)

    @staticmethod
    def poly_addition_example():
        x = Var('x')
        y = Var('y')
        first = 2 * x + 3
        second = 3 * y - 4 * x + 5
        print(first + second)

    @staticmethod
    def poly_subtraction_example():
        x = Var('x')
        first = 3 * x + 5
        second = 2 * x - 1
        print(first - second)

    @staticmethod
    def poly_multiplication_example():
        x = Var('x')
        y = Var('y')
        print((x + 5) * (x - 2))
        print((x + y) * (x - y))

    @staticmethod
    def poly_division_example():
        first = x ** 2 + 6 * x + 8
        second = x + 4
        division_result = first / second
        print(division_result)

    @staticmethod
    def poly_power_example():
        a, b = Var('a'), Var('b')
        print((a + b) ** 2)

    @staticmethod
    def poly_assign_example():
        x = Var('x')
        poly = 2 * x - 4
        poly.assign(x=5)
        print(poly)

    @staticmethod
    def poly_when_example():
        y = Var('y')  # Declaring a variable
        original = 3 * y ** 2 + 6 * y + 7  # Creating the original polynomial
        assigned = original.when(y=-2)  # Saving the assigned polynomial
        print(f"Original is {original}")
        print(f"Assigned is {assigned}")

    @staticmethod
    def poly_tryEvaluate_example():
        x = Var('x')  # Declaring a variable named x
        poly = 3 * x - 2  # Creating a polynomial
        poly.assign(x=4)  # Assigning a value to it, so it will hold a free number
        number = poly.try_evaluate()  # Extract the free number from the polynomial
        print(f"poly represents {poly} and its type is {type(poly)}")
        print(f"number is {number} and its type is {type(number)}")

    @staticmethod
    def poly_plot_example():
        x = Var('x')
        poly = x ** 2 + 6 * x + 8
        poly.plot()

    @staticmethod
    def poly_scatter_example():
        x = Var('x')
        poly = x ** 2 + 6 * x + 8
        poly.scatter()

    @staticmethod
    def poly_derivative_example():
        x = Var('x')
        poly = 2 * x ** 3 - 6 * x + 7
        print(poly.derivative())

    @staticmethod
    def poly_partial_derivative_example1():
        x, y = Var('x'), Var('y')
        f = x ** 2 + y ** 2
        f_x = f.partial_derivative('x')
        print(f_x)

    @staticmethod
    def poly_partial_derivative_example2():
        x, y = Var('x'), Var('y')
        f = x ** 2 + 2 * x * y + y ** 2
        f_xy = f.partial_derivative('xy')


class FastPoly_Examples:

    @staticmethod
    def fast_poly_init():
        fast_poly = FastPoly("3x^5 - 2y^2 + 3x + 14")
        fast_poly = FastPoly("3x^5 - 2y^2 + 3x + 14", variables=('x', 'y'))
        FastPoly({'n': [2, 0, 0, 0], 'free': -32})
        fast_poly = FastPoly([1, 2, 1], variables=('n',))

    @staticmethod
    def fast_poly_addition_example():
        poly1 = FastPoly("2x^3 + 5x -7")
        poly2 = FastPoly("x^2 + 4y^2 - x^3 + 5x + 6")
        print(poly1 + poly2)

    @staticmethod
    def fast_poly_subtraction_example():
        poly1 = FastPoly("2x^3 + 6x^2 + 5")
        poly2 = FastPoly("x^4 + x^3 - 5x^2 + 6")
        print(poly1 - poly2)

    @staticmethod
    def fast_poly_roots_example():
        fast_poly = FastPoly("x^4+8x^3+11x^2-20x")
        print(fast_poly.roots())

    @staticmethod
    def fast_poly_assign_example():
        my_poly = FastPoly("x^2 + y^2")
        my_poly.assign(x=5)
        print(my_poly)

    @staticmethod
    def fast_poly_when_example():
        my_poly = FastPoly("x^2 + y^2")
        print(my_poly.when(x=5))
        print(my_poly)

    @staticmethod
    def fast_poly_tryEvaluate_example():
        poly1 = FastPoly("5")
        poly2 = FastPoly("x^2 + 6x + 8")
        print(poly1.try_evaluate())
        print(poly2.try_evaluate())

    @staticmethod
    def fast_poly_plot_example():
        fast_poly = FastPoly("x^3 - 2x + 1")
        fast_poly.plot()

    @staticmethod
    def fast_poly_scatter_example():
        fast_poly = FastPoly("x^3 - 2x + 1")
        fast_poly.scatter()


class Log_Examples:

    @staticmethod
    def log_init_example():
        my_log = Log("x^2 + 6x + 8", base=2, dtype='poly')
        other_log = Log("sin(x) + cos(2x)", base=10, dtype='trigo')
        x = Var('x')
        my_log = Log(3 * x + 5, base=2)
        other_log = Log(Sin(x), base=5)
        my_log = Log(100, base=10)
        print(my_log.try_evaluate())

    @staticmethod
    def log_addition_example():
        x = Var('x')
        my_log = Log(2 * x)
        other_log = Log(x)
        print(my_log + other_log)

    @staticmethod
    def log_subtraction_example():
        x = Var('x')
        my_log = Log(x ** 2 + 6 * x + 8)
        other_log = Log(x + 4)
        print(my_log - other_log)

    @staticmethod
    def log_multiplication_example():
        n = Var('n')
        my_log = Log(n + 5, base=2)
        my_log *= 3
        my_log *= 2 * n ** 2
        print(my_log)

    @staticmethod
    def log_division_example():  # TODO: simplify logarithm division
        x = Var('x')
        my_log = 3 * Log(x) ** 2
        other_log = 2 * Log(x)
        print(my_log / other_log)

    @staticmethod
    def log_power_example():
        x = Var('x')
        print(Log(x) ** 2)

    @staticmethod
    def log_equating_example():
        x = Var('x')
        my_log = Log(2 * x + 10, base=2)
        other_log = Log(2 * x + 10, base=2)
        print(my_log == other_log)

    @staticmethod
    def log_tryEvaluate_example():
        x = Var('x')
        my_log = Log(2 * x, base=2)
        print(my_log.try_evaluate())
        my_log.assign(x=4)
        print(my_log.try_evaluate())


class TrigoExpr_Examples:

    @staticmethod
    def trigoExpr_init_string_example():
        print(TrigoExpr('sin(3x)', dtype='poly'))
        print(TrigoExpr('sin(log(2x+5))', dtype='log'))

    @staticmethod
    def TrigoExpr_init_subclasses_string_example():
        my_sin = Sin('2x', dtype='poly')
        my_cos = Cos('log(3x)', dtype='log')

    @staticmethod
    def TrigoExpr_init_subclasses_IExpression_example():
        x = Var('x')
        my_sin = Sin(2 * x)
        my_cos = Sin(Log(3 * x))

    @staticmethod
    def TrigoExpr_addition_example():
        x = Var('x')
        print(Sin(x) + Sin(x))
        print(Sin(x) + Cos(x))
        print(Sin(pi / 2) + 4)

    @staticmethod
    def TrigoExpr_subtraction_example():
        x = Var('x')
        print(Cos(x) - 2 * Sin(x))
        print(2 * Sin(x) - Sin(x))
        print(3 * Sin(pi / 2) - 2)

    @staticmethod
    def TrigoExpr_multiplication_example():
        x = Var('x')
        print(Sin(x) * Cos(x))
        print(5 * Sin(2 * x))
        print(3 * x ** 2 * Tan(Log(x)))

    @staticmethod
    def TrigoExpr_division_example():
        x = Var('x')
        print(Sin(x) / Cos(x))
        print(2 * Sin(x) * Cos(x) / Sin(2 * x))
        print((3 * x * Sin(x)) / Log(x))

    @staticmethod
    def TrigoExpr_power_example():
        pass

    @staticmethod
    def TrigoExpr_equating_example():
        x = Var('x')
        print(Sin(2 * x) == Sin(2 * x))
        print(Sin(x) == Sin(x + 2 * pi))
        print(2 * Sin(x) * Cos(x) == Sin(2 * x))
        print(Sin(x) == Sin(x + 5))

    @staticmethod
    def TrigoExpr_simplify_example():
        my_trigo = Sin(x) ** 2 * Cos(x) ** 0
        print(my_trigo)
        my_trigo.simplify()
        print(my_trigo)

    @staticmethod
    def TrigoExpr_toLambda_example():
        x = Var('x')
        my_lambda = Sin(x).to_lambda()
        print(my_lambda(pi / 2))

    @staticmethod
    def TrigoExpr_tryEvaluate_example():
        x = Var('x')
        my_trigo = Sin(pi)
        my_eval = my_trigo.try_evaluate()
        if my_eval is not None:
            print("the expression could be evaluated")
        else:
            print("the expression couldn't be evaluated")

    @staticmethod
    def TrigoExpr_reinman_example():
        x = Var('x')
        print(Sin(x).reinman(0, pi, 20))

    @staticmethod
    def TrigoExpr_simpson_example():
        x = Var('x')
        print(Sin(x).simpson(0, pi, 20))

    @staticmethod
    def TrigoExpr_newton_example():
        x = Var('x')
        print(Sin(x).newton(initial_value=2))

    @staticmethod
    def TrigoExpr_trapz_example():
        x = Var('x')
        print(Sin(x).trapz(0, pi, 20))


class TrigoExprs_Examples:
    @staticmethod
    def trigoExprs_init_example():
        x = Var('x')
        result = Sin(x) + Cos(x)
        print(type(result))

    @staticmethod
    def trigoExprs_init_example1():
        print(TrigoExprs("3sin(x) - 2cos(x)"))

    @staticmethod
    def trigoExprs_addition_example():
        first = TrigoExprs("3sin(x) + 4cos(x)")
        second = TrigoExprs("2sin(x) - cos(x) + 4")
        print(first - second)

    @staticmethod
    def trigoExprs_subtraction_example():
        x = Var('x')
        first = Sin(x) + Cos(x)
        second = Sin(x) - Cos(x)
        print(first * second)

    @staticmethod
    def trigoExprs_division_example():
        x = Var('x')
        first = 3 * Sin(x) ** 2 + 3 * Cos(x) * Sin(x)
        second = Sin(x)
        print(first / second)

    @staticmethod
    def trigoExprs_tryEvaluate_example():
        my_trigo = Sin(pi / 2) + Cos(pi / 2)
        print(my_trigo.try_evaluate())

    @staticmethod
    def trigoExprs_assign_example():
        x = Var('x')
        my_trigo = Sin(x) + Cos(2 * x)
        print(my_trigo.assign(x=pi / 2))

    @staticmethod
    def trigoExprs_toLambda_example1():
        x, y = Var('x'), Var('y')
        my_trigo = Sin(x) * Cos(y) + Sin(y) * Cos(x)
        my_lambda = my_trigo.to_lambda()
        print(my_lambda(pi / 4, pi / 3))

    @staticmethod
    def trigoExprs_toLambda_example2():
        x = Var('x')
        my_trigo = Sin(x) + Cos(x)
        my_lambda = my_trigo.to_lambda()
        print(my_lambda(pi / 2))

    @staticmethod
    def trigoExprs_plot_example1():
        x = Var('x')
        my_trigo = Sin(x) + Cos(x)
        my_trigo.plot()

    @staticmethod
    def trigoExprs_plot_example2():
        x = Var('x')
        y = Var('y')
        my_trigo = Sin(x) * Cos(y)
        my_trigo.plot()

    @staticmethod
    def TrigoExprs_scatter_example1():
        x = Var('x')
        my_trigo = Sin(x) + Cos(x)
        my_trigo.scatter()

    @staticmethod
    def TrigoExprs_scatter_example2():
        x = Var('x')
        y = Var('y')
        my_trigo = Sin(x) * Cos(y)
        my_trigo.scatter()


class Root_Examples:

    @staticmethod
    def root_init_example1():
        x = Var('x')
        my_root = Root(2 * x + 5, 3)
        print(my_root)

    @staticmethod
    def root_init_example2():
        x = Var('x')
        my_root = Sqrt(x ** 2 + 6 * x + 8)

    @staticmethod
    def root_addition_example():
        print(Sqrt(2 * x) + Sqrt(2 * x))
        print(Sqrt(4) + Sqrt(6))

    @staticmethod
    def root_subtraction_example():
        print(2 * Sqrt(x) - Sqrt(x))
        print(Sqrt(6) - Sqrt(4))

    @staticmethod
    def root_multiplication_example():
        print(2 * Sqrt(x) * Sqrt(x))
        print(Sqrt(6) * Sqrt(4))

    @staticmethod
    def root_division_example():
        print(2 * Sqrt(x) / Sqrt(x))
        print(Sqrt(16) / Sqrt(4))

    @staticmethod
    def root_power_example():
        print(Sqrt(x) ** 2)
        print(Sqrt(5) ** 2)

    @staticmethod
    def root_assign_example():
        my_root = Sqrt(x)
        print(my_root.when(x=5))
        my_root.assign(x=4)
        print(my_root)

    @staticmethod
    def root_tryEvaluate_example():
        print(Sqrt(25).try_evaluate())
        print(Sqrt(x).try_evaluate())

    @staticmethod
    def root_plot_example():
        Sqrt(x).plot()

    @staticmethod
    def root_scatter_example():
        Sqrt(x).scatter()


class Abs_Examples:

    @staticmethod
    def abs_init_example():
        x = Var('x')
        print(Abs(x))
        print(Abs(-x))
        print(Abs(5))
        print(Abs(expression=x, power=2, coefficient=3))

    @staticmethod
    def abs_addition_example():
        x = Var('x')
        print(2 * Abs(3 * x) + Abs(3 * x))
        print(Abs(5) + Abs(-5))

    @staticmethod
    def abs_subtraction_example():
        x = Var('x')
        print(2 * Abs(3 * x) * Abs(3 * x))
        print(Abs(5) * Abs(-5))

    @staticmethod
    def abs_multiplication_example():
        x = Var('x')
        print(2 * Abs(3 * x) * Abs(3 * x))
        print(Abs(5) * Abs(-5))

    @staticmethod
    def abs_division_example():
        print(2 * Abs(3 * x) / Abs(3 * x))
        print(Abs(x) / 5)
        print(Abs(5) / Abs(-5))

    @staticmethod
    def abs_power_example():
        print(Abs(x) ** 2)

    @staticmethod
    def abs_assign_example():
        my_abs = Abs(x)
        print(my_abs.when(x=-5))
        my_abs.assign(x=4)
        print(my_abs)

    @staticmethod
    def abs_tryEvaluate_example():
        my_abs = Abs(-5)
        print(my_abs.try_evaluate())

    @staticmethod
    def abs_toLambda_example():
        my_abs = Abs(x)
        print(my_abs.to_lambda())

    @staticmethod
    def abs_plot_example():
        x = Var('x')
        my_abs = Abs(x)
        my_abs.plot()

    @staticmethod
    def abs_plot_example1():
        x, y = Var('x'), Var('y')
        my_abs = Abs(x + y)
        my_abs.plot(step=0.3)

    @staticmethod
    def abs_scatter_example():
        x = Var('x')
        my_abs = Abs(x)
        my_abs.scatter()

    @staticmethod
    def abs_scatter_example1():
        x, y = Var('x'), Var('y')
        my_abs = Abs(x + y)
        my_abs.scatter(step=0.3)


class Exponent_Examples:

    @staticmethod
    def exponent_init_example():
        x = Var('x')
        exponent1 = 3 * Exponent(x, x)
        exponent2 = 3 * x ** x
        exponent3 = Exponent(base=x, power=x, coefficient=3)

    @staticmethod
    def exponent_arithmetic_examples():
        x = Var('x')
        my_exponent = x ** x
        print(my_exponent + x ** x)
        print(my_exponent - x ** x)
        print(my_exponent * 2)
        print(my_exponent / 2)

    @staticmethod
    def exponent_assign_example():
        x, y = Var('x'), Var('y')
        my_exponent = x ** x
        print(my_exponent.when(x=2))

        other_exponent = x ** y
        print(my_exponent.when(y=2))

    @staticmethod
    def exponent_tryEvaluate_example():
        x = Var('x')
        my_exponent = Exponent(2, 2)
        print(my_exponent.try_evaluate())

    @staticmethod
    def exponent_toLambda_example():
        x = Var('x')
        print((x ** x).to_lambda())

    @staticmethod
    def exponent_plotAndScatter_example():
        x, y = Var('x'), Var('y')
        (x ** x).plot()
        (x ** x).scatter()
        (x ** y).plot()
        (x ** y).scatter()


class Factorial_Examples:
    @staticmethod
    def factorial_init_example():
        x = Var('x')
        my_factorial = Factorial(x ** 2 + 6 * x + 8, coefficient=Sin(x), power=2)
        print(my_factorial)
        my_factorial = Factorial('(3x+5)!', dtype='poly')

    @staticmethod
    def factorial_addition_example():
        x = Var('x')
        print(Factorial(x) + Factorial(x))
        print(Factorial(5) + Factorial(4))

    @staticmethod
    def factorial_subtraction_example():
        x = Var('x')
        print(Factorial(x) - 0.5 * Factorial(x))
        print(Factorial(5) - Factorial(4))

    @staticmethod
    def factorial_multiplication_example():
        x = Var('x')
        print(Factorial(x) * Factorial(2 * x))
        print(2 * x * Factorial(Sin(x)))
        print(5 * Factorial(1))

    @staticmethod
    def factorial_division_example():
        x = Var('x')
        print(2 * Factorial(x) / Factorial(x))
        print(Factorial(3 * x + 4) / Factorial(2 * x - 1))

    @staticmethod
    def factorial_power_example():
        x = Var('x')
        print(Factorial(x) ** 2)

    @staticmethod
    def factorial_assign_example():
        my_factorial = Factorial(5)
        my_eval = my_factorial.try_evaluate()
        print(f"the evaluation is {my_eval} and its type is {type(my_eval)}")

    @staticmethod
    def exponent_tryEvaluate_example():
        x = Var('x')
        my_factorial = Factorial(x + 2)
        my_factorial.assign(x=2)
        print(my_factorial)

    @staticmethod
    def factorial_plot_example():
        x = Var('x')
        my_factorial = Factorial(Sin(x))
        my_factorial.plot(start=-5, stop=5, step=0.01)

    @staticmethod
    def factorial_scatter_example():
        x = Var('x')
        my_factorial = Factorial(Sin(x))
        my_factorial.plot(start=-5, stop=5, step=0.01)


class NumericalMethods_Examples:

    @staticmethod
    def reinman_example():
        reinman(lambda x: sin(x), 0, pi, 20)

    @staticmethod
    def trapz_example():
        pass

    @staticmethod
    def simpson_example():
        print(simpson(lambda x: sin(x), 0, pi, 11))

    @staticmethod
    def newton_example1():
        origin_function = lambda x: 2 * x ** 3 - 5 * x ** 2 - 23 * x - 10
        first_derivative = lambda x: 6 * x ** 2 - 10 * x - 23
        initial_value = 8
        print(newton_raphson(origin_function, first_derivative, initial_value))
        other_solution = newton_raphson(origin_function, first_derivative, -10)
        print(other_solution)

    @staticmethod
    def newton_example2():
        origin_function = Function("f(x) = 2x^3 -5x^2 -23x - 10")
        first_derivative: Function = origin_function.derivative()
        solution = newton_raphson(origin_function, first_derivative, 9)
        print(solution)

    @staticmethod
    def newton_example3():
        origin_function = Function("f(x) = 2x^3 -5x^2 -23x - 10")
        print(origin_function.newton(7))

    @staticmethod
    def newton_example_algebraic():
        x = Var('x')
        print((2*x**3 - 5*x**2 - 23*x - 10).newton(5))

    @staticmethod
    def haileys_example():
        f_0 = lambda n: 2 * n ** 3 - 5 * n ** 2 - 23 * n - 10  # function
        f_1 = lambda n: 6 * n ** 2 - 10 * n - 23  # first derivative
        f_2 = lambda n: 12 * n - 10  # second derivative
        initial_value = 0  # initial approximation ( doesn't have to be 0 obviously )
        print(halleys_method(f_0, f_1, f_2, initial_value))

    @staticmethod
    def chebychevs_example():
        f_0 = lambda n: 2 * n ** 3 - 5 * n ** 2 - 23 * n - 10
        f_1 = lambda n: 6 * n ** 2 - 10 * n - 23
        f_2 = lambda n: 12 * n - 10
        initial_value = 0  # It doesn't have to be zero obviously
        print(chebychevs_method(f_0, f_1, f_2, initial_value))

    @staticmethod
    def steffensen_example():
        print(steffensen_method(lambda x: 2 * x ** 3 - 5 * x - 7, 8))

    @staticmethod
    def bisection_example():
        parabola = lambda x: x ** 2 - 5 * x  # creating the function
        print(bisection_method(parabola, 2, 9))

    @staticmethod
    def durand_kerner_lambda():
        func = lambda x: x ** 4 - 16
        coefficients = [1, 0, 0, 0, -16]
        print(durand_kerner(func, coefficients))

    @staticmethod
    def durand_kerner_function_example():
        pass

    @staticmethod
    def durand_kerner_poly_example():
        pass

    @staticmethod
    def aberthMethod_example():
        func = lambda x: 5 * x ** 4 - 1
        derivative = lambda x: 20 * x ** 3
        coefficients = [5, 0, 0, 0, -1]
        print(aberth_method(func, derivative, coefficients))

    @staticmethod
    def generalizedNewton_example():
        solutions = solve_poly_system(["x^2 + y^2 = 25", "2x + 3y = 18"], {'x': 2, 'y': 1})
        print(solutions)


class MachineLearning_Examples:
    @staticmethod
    def gradientDescent_example():
        pass

    @staticmethod
    def gradientAscent_example():
        pass

    @staticmethod
    def mav_example():
        pass

    @staticmethod
    def msv_example():
        pass

    @staticmethod
    def mrv_example():
        pass


class Point_Examples:

    @staticmethod
    def point_init_examples():
        x, y = Var('x'), Var('y')
        print(Point2D(3, 5))
        print(Point((3, 5)))
        print(Point3D(7, 2, 1))
        print(Point((7, 2, 1)))
        print(Point((6, 4, 7, 3)))
        print(Point2D(x + 5, y - 4))
        print(Point3D(x - 3, 4, Sin(y)))

    @staticmethod
    def point_add_sub_example():
        print(Point2D(4, 1) + Point2D(5, -3))
        print(Point3D(2, 1, 7) - Point3D(-7, 4, 9))
        print(Point((4, -2, -1, 6)) + Point((-3, 4, 8, 1)))

    @staticmethod
    def point_max_min_example():
        print(Point((4, 5, 1, 8)).max_coord())
        print(Point((7, 5)).min_coord())

    @staticmethod
    def point_sum_example():
        x = Var('x')
        print(Point((5, 4, 1, 3)).sum())
        print(Point3D(x, 2 * x, x + 6).sum())

    @staticmethod
    def point_distance_example():
        pass

class PointCollection_Examples:

    @staticmethod
    def pointCollection_init_example():
        my_points = PointCollection([Point((2, 6)), [2, 4], Point2D(9, 3)])
        print(my_points)

    @staticmethod
    def pointCollection_addPoint_example():
        my_points = PointCollection([Point((2, 6)), [2, 4], Point2D(9, 3)])
        my_points.add_point(Point((5, 3)))
        print(my_points)

    @staticmethod
    def pointCollection_removePoint_example():
        my_points = PointCollection([Point((2, 6)), [2, 4], Point2D(9, 3)])
        my_points.remove_point(0)  # remove the first point
        print(my_points)

    @staticmethod
    def pointCollection_distances_example1():
        my_points = PointCollection([[1, -4, 5], [7, -5, -2], [5, 3, 9], [-2, 6, 4]])
        print(my_points.longest_distance())
        print(my_points.shortest_distance())

    @staticmethod
    def pointCollection_distances_example2():
        my_points = PointCollection([[1, -4, 5], [7, -5, -2], [5, 3, 9], [-2, 6, 4]])
        longest_distance, longest_pair = my_points.longest_distance(get_points=True)
        print(F"Longest Distance: {longest_distance}, Longest Pair: {longest_pair}")
        shortest_distance, shortest_pair = my_points.shortest_distance(get_points=True)
        print(F"Shortest Distance: {shortest_distance}, Shortest Pair: {shortest_pair}")

    @staticmethod
    def pointCollection_scatter_examples():
        # scatter in 1D
        points = PointCollection([[1], [3], [5], [6]])
        points.scatter()

        # scatter in 2D
        points = Point2DCollection(
            [[1, 2], [6, -4], [-3, 1], [4, 2], [7, -5], [4, -3], [-2, 1], [-3, 4], [5, 2], [1, -5]])
        points.scatter()

        # scatter in 3D
        points = Point3DCollection([[1, 3, 2], [-1, 4, 2], [4, -2, 0], [4, 1, 1], [-2, 3, -1]])
        points.scatter()

        # Scatter in 4D
        points = Point4DCollection(
            [[random.randint(1, 100), random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)] for _ in
             range(100)])
        points.scatter()

class Graph2D_Examples:

    @staticmethod
    def graph2d_init_example1():
        my_graph = Graph2D((Function("f(x) = 2x"), Poly("x^3 - 3")))

    @staticmethod
    def graph2d_init_example2():
        my_graph = Graph2D()

    @staticmethod
    def graph2d_addition_example():
        my_graph = Graph2D()
        my_graph.add(Function("f(x) = x^3-3"))
        my_graph.add(Poly("2x^4 - 3x + 5"))
        my_graph.add(Circle(5, (0, 0)))

    @staticmethod
    def graph2d_plot_example():
        my_graph = Graph2D()
        my_graph.add(Function("f(x) = x^3-3"))
        my_graph.add(Poly("2x^4 - 3x + 5"))
        my_graph.add(Circle(5, (0, 0)))
        my_graph.plot()

    @staticmethod
    def graph2d_scatter_example():
        pass


class Graph3D_Examples:

    @staticmethod
    def graph3d_init_example1():
        my_graph = Graph3D([Function("f(x,y) = x+y"), Function("f(x,y) = sin(x)*cos(y)")])

    @staticmethod
    def graph3d_init_example2():
        my_graph = Graph3D()

    @staticmethod
    def graph3d_addition_example():
        x = Var('x')
        my_graph = Graph3D()
        my_graph.add(Cos(x) * Sin(x))

    @staticmethod
    def graph3d_plot_example():
        pass


class Vector_Examples:
    @staticmethod
    def vector_init_examples():
        Vector(start_coordinate=(1, 2), end_coordinate=(4, 6))
        Vector(direction_vector=(3, 4), start_coordinate=(1, 2))
        Vector(direction_vector=(3, 4), end_coordinate=(4, 6))
        Vector((3, 4))

    @staticmethod
    def vector_copy_examples():
        vec = Vector(start_coordinate=(7, 8, 5), direction_vector=(2, 5, 4))
        print(vec.__copy__())

    @staticmethod
    def vector_addition_example():
        vec1 = Vector((1, 1, 1))
        vec2 = Vector((2, 2, 2))
        vec3 = Vector((3, 3, 3))
        print(vec1 + vec2 + vec3)

    @staticmethod
    def vector_subtraction_example():
        vec1 = Vector((1, 1, 1))
        vec2 = Vector((2, 2, 2))
        vec3 = Vector((3, 3, 3))
        print(vec3 - vec2 - vec1)

    @staticmethod
    def vector_multiplication_example():
        vec1 = Vector((6, 2, 5))
        vec2 = Vector((3, 4, 2))
        print(vec1 * vec2)

    @staticmethod
    def vector_neg_example():
        vec = Vector((-1, 2, 3))
        print(vec)
        print(-vec)

    @staticmethod
    def vector_length_examples():
        vec = Vector((3, 4))
        print(len(vec))  # first approach using len()
        print(vec.__len__())  # second approach using __len__()
        print(vec.length())  # third approach using length()

    @staticmethod
    def vector_equating_example():
        print(Vector((3, 4, 8)) == Vector((3, 4, 8)))

    @staticmethod
    def vector_same_direction_example():
        print(Vector((1, 2, 4)).equal_direction_ratio(Vector((2, 4, 8))))
        print(Vector((x, 2 * x, 4 * x)).equal_direction_ratio(Vector((2 * x, 4 * x, 8 * x))))
        print(Vector((x, 2 * x, 4 * x)).equal_direction_ratio(Vector((2 * x ** 2, 4 * x ** 2, 8 * x ** 2))))

    @staticmethod
    def vector_plot():
        pass

    @staticmethod
    def vector_scatter():
        pass


class Circle_Examples:

    @staticmethod
    def circle_init_example():
        x = Var('x')
        my_circle = Circle(5, (3, 3))
        other_circle = Circle(x + 5, (Cos(x), Sin(x)))

    @staticmethod
    def circle_area_example():
        r = Var('r')
        print(Circle(5, (3, 1)).area())
        print(Circle(r, (1, -4)).area())

    @staticmethod
    def circle_perimeter_example():
        r = Var('r')
        print(Circle(5, (3, 1)).perimeter())
        print(Circle(r, (1, -4)).perimeter())

    @staticmethod
    def circle_inside_point_example():
        my_circle = Circle(5, (0, 0))
        print(my_circle.point_inside((1, 1)))

    @staticmethod
    def circle_inside_example():
        small_circle = Circle(5, (0, 0))
        big_circle = Circle(10, (0, 0))

    @staticmethod
    def circle_plot_example():
        my_circle = Circle(5, (0, 0))
        my_circle.plot()


class VectorCollection_Examples:

    @staticmethod
    def vectorCollection_example1():
        # You can enter Vector objects, lists, or tuples.
        vectors = VectorCollection(Vector((7, 5, 3)), (8, 1, 2), [9, 4, 7])

        # Or if you already have an existing list of vectors:
        lst_of_vectors = [Vector((7, 5, 3)), (8, 1, 2), [9, 4, 7]]
        other_vectors = VectorCollection(*lst_of_vectors)

    @staticmethod
    def plot_example2d():
        vectors = VectorCollection(Vector(start_coordinate=(7, 5), end_coordinate=(3, 4)), Vector((1, 9)))
        vectors.append(Vector(start_coordinate=(2, 2), end_coordinate=(7, 7)))
        vectors.plot()

    @staticmethod
    def plot_example3d():
        vectors = VectorCollection(Vector((7, 5, 3)), Vector((8, 1, 2)), Vector((9, 4, 7)))
        vectors.plot()

    @staticmethod
    def max_min_length_example():
        # creating the vectors
        vectors = VectorCollection(Vector((7, 5, 3)), Vector((8, 1, 2)), Vector((9, 4, 7)))
        # finding and storing the longest vector
        longest_vector = vectors.longest()
        # finding and storing the shortest vector
        shortest_vector = vectors.shortest()
        # printing it out ( the vectors will be
        # printed according to the __str__() , method in the Vector class.
        print(f"the longest vector:{longest_vector}")
        print(f"the shortest vector:{shortest_vector}")

    @staticmethod
    def max_min_length_example1():
        # creating the vectors and printing the initial collection
        vectors = VectorCollection(Vector((2, 1, 6)), Vector((8, 1, 2)), Vector((9, 4, 7)))

        # fetching both the index of the longest vector in the collection, and the longest vector itself.
        index, longest_vector = vectors.longest(get_index=True)
        print(f"Vector {longest_vector} is in index {index}")

        # fetching both the index of the shortest vector and the shortest vector, but this time, also removing it from
        # the collection.
        index, shortest_vector = vectors.shortest(get_index=True, remove=True)
        print(f"removed the vector {shortest_vector} from index {index}")


class Matrix_Examples:

    @staticmethod
    def matrix_init_examples():
        mat = Matrix(dimensions=(3, 3))
        mat1 = Matrix(dimensions="2x2")
        mat2 = Matrix(dimensions="1,3")
        mat3 = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        print(mat)
        print(mat1)
        print(mat2)
        print(mat3)

    @staticmethod
    def matrix_swap_row_example():
        mat = Matrix(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        print(mat)

        mat.replace_rows(0, 1)  # replacing the first row with the third
        print(mat)

    @staticmethod
    def matrix_multiply_row_example():
        mat = Matrix(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        mat.multiply_row(2, row=1)
        print(mat)

    @staticmethod
    def matrix_divide_row_example():
        my_matrix = Matrix([[1, 2, 3], [4, 5, 6]])
        my_matrix.divide_row(2, row=0)

    @staticmethod
    def matrix_max_min_example():
        mat = Matrix(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        print(f'the maximum is {mat.max()}')
        print(f'the minimum is {mat.min()}')

    @staticmethod
    def matrix_sum_example():
        mat = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        print(mat.sum())

    @staticmethod
    def matrix_equating_example():
        mat = Matrix([[5, 3, 2], [8, 7, 1]])
        mat1 = Matrix([[6, 4, 7], [9, 1, 3]])

        if mat == mat1:
            print("The matrices are equal")
        else:
            print("The matrices are not equal")

    @staticmethod
    def matrix_iteration_example1():
        mat = Matrix(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
        for row in mat.matrix:
            for item in row:
                print(item)

    @staticmethod
    def matrix_iteration_example2():
        mat = Matrix(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
        for row in mat:
            for item in row:
                print(item)

    @staticmethod
    def matrix_iteration_example3():
        mat = Matrix(matrix=((1, 2, 3), (4, 5, 6), (7, 8, 9)))
        for i, j in mat.range():
            print(f"The item in row {i} and column {j} is {mat[i][j]}")

    @staticmethod
    def matrix_addition_example():
        mat1 = Matrix([[1, 3], [7, 6]])
        mat2 = Matrix([[5, 8], [4, 2]])
        print(mat1 + mat2)

    @staticmethod
    def matrix_subtraction_example():
        mat1 = Matrix([[2, 4], [1, 7], [8, 3]])
        mat2 = Matrix([[5, 5], [1, 4], [3, 2]])
        print(mat1 - mat2)

    @staticmethod
    def matrix_element_wise_mul_example():
        pass

    @staticmethod
    def matrix_multiplication_example():

        mat1 = Matrix([[3, 2], [7, 4], [9, 1]])
        mat2 = Matrix([[2, 4, 1], [6, 3, 2]])
        print(mat1 @ mat2)

    @staticmethod
    def matrix_kronecker_multiplication_example():
        pass

    @staticmethod
    def matrix_determinant_example():
        mat = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        print(mat.determinant())

    @staticmethod
    def matrix_transpose_example():
        mat = Matrix([[1, 2], [3, 4], [5, 6]])
        print(mat.transpose())

    @staticmethod
    def matrix_inverse_example():
        mat = Matrix([[1, 2], [3, 4]])
        print(mat.inverse())


class Surface_Examples:

    @staticmethod
    def surface_example_1():
        my_surface = Surface([7, 3, 1, 9])
        my_surface.plot()

    @staticmethod
    def surface_example_2():
        my_surface = Surface("7x + 3y + z + 9 = 0")
        my_surface.plot()

    @staticmethod
    def surface_equality_example():
        first_surface = Surface("7x + 3y + z + 9 = 0")
        second_surface = Surface([7, 1, 3, 9])
        print(first_surface == second_surface)


class ArithmeticProg_Examples:

    @staticmethod
    def arithmetic_init_example():
        my_sequence = ArithmeticProg([3], 2)
        print(my_sequence)

    @staticmethod
    def arithmetic_init_example1():
        my_sequence = ArithmeticProg([3, 5])
        print(my_sequence)

    @staticmethod
    def arithmetic_inIndex_example():
        my_sequence = ArithmeticProg([3], 2)
        print(my_sequence.in_index(2))

    @staticmethod
    def arithmetic_firstNSum_example():
        my_sequence = ArithmeticProg([3], 2)
        print(my_sequence.sum_first_n(3))

    @staticmethod
    def arithmetic_indexOf_example():
        my_sequence = ArithmeticProg([3], 2)
        print(my_sequence.index_of(7))
        print(my_sequence.index_of(10))


class GeometricSeq_Examples:
    @staticmethod
    def geometricSeq_init_example1():
        my_sequence = GeometricSeq([2], 2)
        print(my_sequence)

    @staticmethod
    def geometricSeq_init_example2():
        my_sequence = GeometricSeq([2, 4])
        print(my_sequence)

    @staticmethod
    def geometricSeq_inIndex_example():
        my_sequence = GeometricSeq([2, 4])
        print(my_sequence.in_index(3))

    @staticmethod
    def geometricSeq_indexOf_example():
        my_sequence = GeometricSeq([2], 2)
        print(my_sequence.index_of(8))

    @staticmethod
    def geometricSeq_in_example():
        my_sequence = GeometricSeq([2], 2)
        print(8 in my_sequence)
        print(7 in my_sequence)

    @staticmethod
    def geometricSeq_plot_example():
        my_sequence = GeometricSeq([2], 2)
        my_sequence.plot(start=1, stop=10, step=1)


class SequenceReq_Examples:

    @staticmethod
    def sequenceReq_init_examples():
        fibonacci = RecursiveSeq("a_n = a_{n-1} + a_{n-2}", (1, 1, 2))
        my_factorial = RecursiveSeq("a_n = a_{n-1} * n", (1, 2))


class ProbabilityTree_Examples:

    @staticmethod
    def tree_init_example1():
        tree = ProbabilityTree(root=Occurrence(1, "taking_the_test"))
        pass_test = tree.add(0.4, "pass_test")
        fail_test = tree.add(0.6, "fail_test")
        ace_test = tree.add(0.1, "ace_test", parent=pass_test)
        print(tree.get_probability(path="taking_the_test/pass_test/ace_test"))

    @staticmethod
    def tree_init_example2():
        tree = ProbabilityTree(json_path='cooltree.json')

    @staticmethod
    def tree_init_example3():
        tree = ProbabilityTree(xml_path="cooltree.xml")

    @staticmethod
    def tree_to_json():
        tree = ProbabilityTree()
        first_son = tree.add(0.5, "son1")  # adding a son
        second_son = tree.add(0.5, "son2")  # adding another son
        tree.export_json("mytree.json")  # creating the file in the given path

    @staticmethod
    def tree_getNodeById_example():
        pass

    @staticmethod
    def getNodePath_example():
        pass


if __name__ == '__main__':
    main()
