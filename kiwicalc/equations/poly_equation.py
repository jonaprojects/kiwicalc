class PolyEquation(Equation):

    def __init__(self, first_side, second_side=None, variables=None):
        self.__solution = None
        if first_side is None:
            raise TypeError("First argument in PolyEquation.__init__() cannot be None. Try using a string"
                            ", and read the documentation !")
        # Handling a string as the first parameter,
        if second_side is None and isinstance(first_side, str):
            # that represents the equation
            left_side, right_side = first_side.split("=")
            self.__first_expression, self.__second_expression = Poly(
                left_side), Poly(right_side)
            equation = first_side
        else:  # In case both sides are entered
            try:
                # Handling the first side of the equation
                if isinstance(first_side, (Mono, Poly)):
                    # TODO: avoid memory sharing ..
                    self.__first_expression = first_side.__copy__()
                else:
                    self.__first_expression = None
                # Handling the second side of the equation
                if isinstance(second_side, (Mono, Poly)):
                    self.__second_expression = second_side.__copy__()
                else:
                    self.__second_expression = None
                equation = "=".join((str(first_side), str(second_side)))
            except TypeError:
                raise TypeError(f"Unexpected type{type(first_side)} in PolyEquation.__init__()."
                                f"Couldn't convert the parameter to type str.")
        super().__init__(equation, variables)

    def solve(self):  # TODO: try to optimize this method ?
        return (self.__first_expression - self.__second_expression).roots()

    @property
    def solution(self):
        if self.__solution is None:
            self.__solution = self.solve()
        return self.__solution

    @property
    def first_poly(self):
        return self.__first_expression

    @property
    def second_poly(self):
        return self.__second_expression

    def _extract_variables(self):
        return extract_dict_from_equation(self._equation)

    def plot_solutions(self, start: float = -10, stop: float = 10, step: float = 0.01, ymin: float = -10, ymax=10,
                       title: str = None,
                       show_axis=True, show=True):  # TODO: check and modify
        first_func = Function(self.first_side)
        second_func = Function(self.second_side)
        plot_functions([first_func, second_func], start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title,

                       show_axis=show_axis, show=show)

    @staticmethod
    def __random_monomial(values=(1, 20), power: int = None, variable=None):
        if variable is None:
            variable = 'x'
        coefficient = random.randint(values[0], values[1])
        if coefficient == 0:
            return "0"
        elif coefficient == 1:
            coefficient = ""
        elif coefficient == -1:
            coefficient = '-'
        else:
            coefficient = f"{coefficient}"
        if power == 1:
            return f"{coefficient}{variable}"
        elif power == 0:
            return f"{coefficient}"
        return f"{coefficient}{variable}^{power}"

    @staticmethod
    def random_expression(values=(1, 10), of_order: int = None, variable=None, all_powers=False):
        if of_order is None:
            of_order = random.randint(1, 10)
        if of_order == 1:
            return LinearEquation.random_expression(values, variable=variable)
        accumulator = ''
        accumulator += '-' if random.randint(0, 1) else '+'
        accumulator = PolyEquation.__random_monomial(
            values, of_order, variable)
        for power in range(of_order - 1, 0, -1):
            if random.randint(0, 1) or all_powers:
                accumulator += '-' if random.randint(0, 1) else '+'
                accumulator += PolyEquation.__random_monomial(
                    values, power, variable)
        if random.randint(0, 1) or all_powers:
            accumulator += '-' if random.randint(0, 1) else '+'
            accumulator += f"{random.randint(values[0], values[1])}"
        return accumulator

    @staticmethod
    def random_quadratic(values=(1, 20), variable=None, all_powers=False):
        return f"{PolyEquation.random_expression(values=values, of_order=2, variable=variable, all_powers=all_powers)} = 0"

    @staticmethod
    def random_equation(values=(1, 20), of_order: int = None, variable=None, all_powers=False):
        return f"{PolyEquation.random_expression(values, of_order, variable, all_powers)}={PolyEquation.random_expression(values, of_order, variable, all_powers)}"

    @staticmethod
    def random_worksheet(path=None, title="Equation Worksheet", num_of_equations=20, degrees_range=(2, 5),
                         solutions_range=(-15, 15), digits_after: int = 0, get_solutions=False):
        if get_solutions:
            expressions = [
                random_polynomial(random.randint(degrees_range[0], degrees_range[1]), solutions_range=solutions_range,
                                  digits_after=digits_after, get_solutions=get_solutions) for _ in
                range(num_of_equations)]
            equations = [f"{index + 1}. {expression[0]} = 0" for index,
                         expression in enumerate(expressions)]
            solutions = [f"{index + 1}. " + ",".join([str(solution) for solution in expression[1]]) for
                         index, expression in enumerate(expressions)]
            create_pages(path, 2, ["Polynomial Equations Worksheet", "Solutions"], [
                         equations, solutions])
        else:
            return create_pdf(path=path, title=title, lines=[
                f"{random_polynomial(random.randint(degrees_range[0], degrees_range[1]), solutions_range=solutions_range, digits_after=digits_after)} = 0"
                for _ in range(num_of_equations)])

    @staticmethod
    def random_worksheets(path=None, num_of_pages=2, equations_per_page=20, titles=None, degrees_range=(2, 5),
                          solutions_range=(-15, 15), digits_after: int = 0, get_solutions=False):
        if get_solutions:
            pages_list = []
            for i in range(num_of_pages):
                expressions = [
                    random_polynomial(random.randint(degrees_range[0], degrees_range[1]),
                                      solutions_range=solutions_range,
                                      digits_after=digits_after, get_solutions=True) for _ in
                    range(equations_per_page)]
                equations = [f"{index + 1}. {expression[0]} = 0" for index,
                             expression in enumerate(expressions)]
                solutions = [f"{index + 1}. " + ",".join([str(solution) for solution in expression[1]]) for
                             index, expression in enumerate(expressions)]
                pages_list.append(equations)
                pages_list.append(solutions)
            if titles is None:
                titles = ["Polynomial Equations Worksheet",
                          "Solutions"] * num_of_pages
            create_pages(path, num_of_pages * 2, titles, pages_list)

        else:
            pages_list = []
            for i in range(num_of_pages):
                expressions = [
                    random_polynomial(random.randint(degrees_range[0], degrees_range[1]),
                                      solutions_range=solutions_range,
                                      digits_after=digits_after, get_solutions=False) for _ in
                    range(equations_per_page)]
                equations = [f"{index + 1}. {expression[0]} = 0" for index,
                             expression in enumerate(expressions)]
                pages_list.append(equations)
            if titles is None:
                titles = ["Polynomial Equations Worksheet"] * num_of_pages
            create_pages(path, num_of_pages, titles, pages_list)

    def to_PolyExpr(self):
        return Poly(self._equation)

    def __str__(self):
        return self._equation

    def __repr__(self):
        return f"PolyEquation({self._equation})"

    def __copy__(self):
        return PolyEquation(self._equation)
