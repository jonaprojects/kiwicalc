# TODO: support features for more than 1 variable such as assignment and parameters.
class LinearEquation(Equation):

    def __init__(self, equation: str, variables=None, calc_now=False):
        super().__init__(equation, variables, calc_now)
        try:
            index = self._variables.index("number")
            del self._variables[index]
        except ValueError:
            pass

    def solve(self):
        if self._solution is None:
            self._solution = solve_linear(self.equation, self.variables_dict)
        return self._solution

    def simplify(self, round_coefficients=True):
        first_dict = simplify_linear_expression(
            expression=self.first_side, variables=self._variables_dict)
        second_dict = simplify_linear_expression(
            expression=self.second_side, variables=self._variables_dict)
        result_dict = {key: value for key, value in subtract_dicts(
            dict1=first_dict, dict2=second_dict).items() if key}
        self._variables_dict = result_dict.copy()
        num = result_dict['number']
        del result_dict['number']
        self._equation = f"{format_linear_dict(result_dict, round_coefficients=round_coefficients)} = {round_decimal(-num)}"

    def __format_expressions(self, expressions):
        accumulator = ""
        for index, expression in enumerate(expressions):
            if expression.variables_dict in ({}, None):
                if expression.coefficient > 0 and index > 0:
                    accumulator += f"\033[93m+{expression}\33[0m "
                else:
                    accumulator += f"\033[96m{expression}\33[0m "
            elif expression.coefficient != 0:

                if expression.coefficient > 0 and index > 0:
                    accumulator += f"\033[93m+{expression}\33[0m "
                else:
                    accumulator += f"\033[96m{expression}\33[0m "
        return accumulator

    def show_steps(self):  # Only for univariate linear equations
        variables = self.variables_dict
        if len(variables) > 2:
            raise NotImplementedError(
                f"This feature is currently only available with 1-variable equation, got {len(variables)}")
        if len(variables) < 2:
            first_side, second_side = self.first_side, self.second_side
            accumulator = "\033[1m1. First step: recognize that this equation only contains free numbers," \
                          "and hence either it has no solutions, or it has infinite solutions \33[0m\n"
            accumulator += f"\033[93m{first_side.replace('+', ' +').replace('-', ' -')}\33[0m"
            accumulator += " = "
            accumulator += f"\033[93m{second_side.replace('+', ' +').replace('-', ' -')}\33[0m\n"
            accumulator += "\033[1m2. Second Step: sum all the numbers in both sides\33[0m\n"
            first_expression = simplify_linear_expression(
                expression=first_side, variables=variables)
            second_expression = simplify_linear_expression(
                expression=first_side, variables=variables)
            accumulator += f"\033[93m{first_expression['number']}\33[0m"
            accumulator += ' = '
            accumulator += f"\033[93m{second_expression['number']}\33[0m\n"
            if first_expression["number"] == second_expression["number"]:
                accumulator += "\033[1mFinal Step:  The expression above is always true, and hence there are infinite solutions " \
                               "to the equation.\33[0m\n"
                self._solution = "Infinite"
            else:
                accumulator += "\033[1mFinal Step: The expression above is always false, and hence there are infinite solutions" \
                               " to the equation.\33[0m\n"
                self._solution = None
            return accumulator

        first_variable = list(self.variables_dict.keys())[0]
        first_side, second_side = self.first_side, self.second_side
        first_expressions = poly_from_str(first_side, get_list=True)
        second_expressions = poly_from_str(second_side, get_list=True)
        accumulator = f"\033[1m1. First Step : Identify the free numbers and the expressions with {first_variable} in each side\33[0m\n"
        accumulator += self.__format_expressions(first_expressions) + " = " + self.__format_expressions(
            second_expressions) + "\n"
        accumulator += "\033[1m2. Second step: Sum the matching groups in each side ( if it's possible )\33[0m\n"
        # Later perhaps adjust it to support multiple variables_dict
        free_sum1, variables_sum1 = 0, 0
        for mono_expression in first_expressions:
            if mono_expression.is_number():
                free_sum1 += mono_expression.coefficient
            else:
                variables_sum1 += mono_expression.coefficient
        accumulator += f"\033[96m{variables_sum1}{first_variable}\33[0m "
        if free_sum1 > 0:
            accumulator += f"+\033[93m{free_sum1}\33[0m"
        elif free_sum1 != 0:
            accumulator += f"\033[93m{free_sum1}\33[0m"
        accumulator += " = "
        # Later perhaps adjust it to support multiple variables_dict
        free_sum2, variables_sum2 = 0, 0
        for mono_expression in second_expressions:
            if mono_expression.is_number():
                free_sum2 += mono_expression.coefficient
            else:
                variables_sum2 += mono_expression.coefficient
        accumulator += f"\033[96m{variables_sum2}{first_variable}\33[0m "
        if free_sum1 > 0:
            accumulator += f"+\033[93m{free_sum2}\33[0m"
        elif free_sum1 != 0:
            accumulator += f"\033[93m{free_sum2}\33[0m"
        accumulator += "\n"
        accumulator += "\033[1m3. Third Step: Move all the variables to the right, and the free numbers to the left \33[0m\n"
        variable_difference = variables_sum1 - variables_sum2
        if variable_difference == 0:
            accumulator += '0'
        else:
            accumulator += f"\033[96m{variable_difference}{first_variable}\33[0m"

        accumulator += " = "
        free_sum_difference = free_sum2 - free_sum1
        accumulator += f"\033[93m{free_sum_difference}\33[0m\n"
        if variable_difference == 0:  # If the variables_dict have vanished in the simplification process
            if free_sum_difference == 0:
                accumulator += "\033[1m3 Therefore, there are infinite solutions !\33[0m\n"
                self._solution = "Infinite"
                return accumulator
            else:
                accumulator += "\033[1m3 Therefore, there is no solution to the equation !\33[0m\n"
                self._solution = None
                return accumulator

        accumulator += "\033[1m4. Final step: divide both sides by the coefficient of the right side \33[0m\n"
        accumulator += f"\033[96m{first_variable}\33[0m = \033[93m{free_sum_difference / variable_difference}\33[0m"
        return accumulator

    def plot_solution(self, start: float = -10, stop: float = 10, step: float = 0.01, ymin: float = -10,
                      ymax: float = 10, show_axis=True, show=True, title: str = None, with_legend=True):
        """
        Plot the solution of the linear equation, as the intersection of two linear functions ( in each side )
        """
        if is_number(self.first_side):
            first_function = Function(f"f(x) = {self.first_side}")
        else:
            first_function = Function(self.first_side)
        if is_number(self.second_side):
            second_function = Function(f"f(x) = {self.second_side}")
        else:
            second_function = Function(self.second_side)
        if title is None:
            title = f"{self.first_side}={self.second_side}"
        plot_functions([first_function, second_function], start=start, stop=stop, step=step, ymin=ymin, ymax=ymax,
                       show_axis=show_axis, show=False, title=title, with_legend=with_legend)
        x = self.solution
        if x is not None and not isinstance(x, str):
            y = first_function(x)
            # Write the intersection with a label
            plt.scatter([x], [y], color="red")
            if show:
                plt.show()
            return x, y
        return x

    def _extract_variables(self):
        return extract_dict_from_equation(self._equation)

    @staticmethod
    def random_expression(values=(1, 20), items_range=(4, 7), variable=None):
        """
        Generates a string that represents a random linear expression, according to the parameters
        :param values: a tuple which contains two items: the min value, and max value possible.
        :param items_range: the numbers of item in the expression: ( min_number,max_number)
        :param variable: the variable's name. if not mentioned - it'll be chosen randomly from a list of letters
        :return:
        """
        accumulator = ""
        if not variable or not isinstance(variable, str):
            variable = random.choice(allowed_characters)
        num_of_items = random.randint(items_range[0], items_range[1])
        for i in range(num_of_items):
            if random.randint(0, 1):
                accumulator = "".join((accumulator, '-'))
            elif accumulator:  # add a '+' if but not to the first char
                accumulator = "".join((accumulator, '+'))
            coefficient = random.randint(values[0], values[1])
            if coefficient != 0:
                if random.randint(0, 1):  # add a variable
                    accumulator += f'{format_coefficient(coefficient)}{variable}'
                else:  # add a free number
                    accumulator += f"{coefficient}"
        return accumulator if accumulator != "" else "0"

    @staticmethod
    def random_equation(values=(1, 20), items_per_side=(4, 7), digits_after=2, get_solution=False, variable=None,
                        get_variable=False):
        """
        generates a random equation
        :param values: the range of the values
        :param items_per_side: the range of the number of items per side
        :param digits_after: determines the maximum number of digits after the dot a __solution can contain. For example,
        if digits_after=2, and the __solution of the equation is 3.564, __equations will be randomly generated
        until a valid __solution like 5.31 will appear.
        :param get_solution:
        :return: returns a random equation, that follows by all the condition given in the parameters.
        """
        if not variable:
            variable = random.choice(
                ['x', 'y', 'z', 't', 'y', 'm', 'n', 'k', 'a', 'b'])
        equation = f'{LinearEquation.random_expression(values=values, items_range=items_per_side, variable=variable)} '
        equation += f'= {LinearEquation.random_expression(values=values, items_range=items_per_side, variable=variable)}'
        solution = LinearEquation(equation).solve()
        solution_string = str(solution)
        # Limit the number of tries to 1000, to prevent cases that it searches forever
        for i in range(1000):
            if len(solution_string[solution_string.find('.') + 1:]) <= digits_after:
                if get_solution:
                    if get_variable:
                        return equation, solution, variable
                    return equation, solution
                if get_variable:
                    return equation, variable
                return equation
            equation = f'{LinearEquation.random_expression(values=values, items_range=items_per_side, variable=variable)} '
            equation += f'= {LinearEquation.random_expression(values=values, items_range=items_per_side, variable=variable)}'
            solution = LinearEquation(equation).solve()
            solution_string = str(solution)
        if get_solution:
            if get_variable:
                return equation, solution, variable
            return equation, solution
        if get_variable:
            return equation, variable
        return equation

    @staticmethod
    def random_worksheet(path, title="Equation Worksheet", num_of_equations=10, values=(1, 20),
                         items_per_side=(4, 8), after_point=2, get_solutions=False) -> bool:
        """
        Generates a PDF page with random __equations
        :return:
        """

        equations = [LinearEquation.random_equation(values, items_per_side, after_point, get_solutions) for _ in
                     range(num_of_equations)]
        return create_pdf(path=path, title=title, lines=equations)

    @staticmethod
    def random_worksheets(path: str, num_of_pages: int = 2, equations_per_page=20, values=(1, 20),
                          items_per_side=(4, 8), after_point=1, get_solutions=False, titles=None):
        if get_solutions:
            lines = []
            for i in range(num_of_pages):
                equations, solutions = [], []
                for j in range(equations_per_page):
                    equation, solution, variable = LinearEquation.random_equation(values=values,
                                                                                  items_per_side=items_per_side,
                                                                                  digits_after=after_point,
                                                                                  get_solution=True,
                                                                                  get_variable=True)
                    equations.append(f"{j + 1}. {equation}")
                    solutions.append(f"{j + 1}. {variable} = {solution}")
                lines.extend((equations, solutions))

            if titles is None:
                titles = ['Worksheet - Linear Equations',
                          'Solutions'] * num_of_pages
            create_pages(path=path, num_of_pages=num_of_pages *
                         2, titles=titles, lines=lines)

        else:

            lines = []
            for i in range(num_of_pages):
                equations = []
                for j in range(equations_per_page):
                    equation = LinearEquation.random_equation(values=values,
                                                              items_per_side=items_per_side,
                                                              digits_after=after_point,
                                                              get_solution=False,
                                                              get_variable=False)
                    equations.append(f"{j + 1}. {equation}")
                lines.append(equations)

            if titles is None:
                titles = ['Worksheet - Linear Equations'] * num_of_pages
            create_pages(path=path, num_of_pages=num_of_pages,
                         titles=titles, lines=lines)

    @staticmethod
    def adjusted_worksheet(title="Equation Worksheet", equations=(),
                           ) -> bool:
        """
        Creates a user-defined PDF worksheet file.
        :param title: the title of the page
        :param equations: the __equations to print out
        :return: returns True if the creation is successful, else False.
        """
        return create_pdf("test", title=title, lines=equations)

    @staticmethod
    def manual_worksheet() -> bool:
        """
        Allows the user to create a PDF worksheet file manually.
        :return: True, if the creation is successful, else False
        """
        try:
            name, title, equations = input(
                "Worksheet's Name:  "), input("Worksheet's Title:  "), []
            print("Enter your equations. To stop, type 'stop' ")
            i = 1
            equation = input(f"{i}.  ")
            i += 1
            while equation.lower() != 'stop':
                equations.append(equation)
                equation = input(f"{i}.  ")
        except Exception as e:  # HANDLE THIS EXCEPTION PROPERLY
            warnings.warn(
                f"Couldn't create the pdf file due to a {e.__class__} error")
            return False
        return LinearEquation.adjusted_worksheet(title=title, equations=equations)

    def __str__(self):
        return f"{self.equation}"

    def __repr__(self):
        return f"Equation({self.equation})"

    def __copy__(self):
        return LinearEquation(self._equation)
