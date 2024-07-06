class QuadraticEquation(Equation):

    def __init__(self, equation: str, variables: Optional[Iterable[str]] = None, strict_syntax=False):
        self.__strict_syntax = strict_syntax
        super().__init__(equation, variables)

    def _extract_variables(self):
        return ParseExpression.parse_quadratic(self.first_side, self._variables, strict_syntax=self.__strict_syntax)

    def simplified_str(self) -> str:
        if self.num_of_variables != 1:
            raise ValueError(
                "You can only simplify quadratic equations with 1 variable in the current version")
        my_coefficients = self.coefficients()
        return ParseExpression.coefficients_to_str(my_coefficients, variable=self._variables[0])

    def solve(self, mode='complex'):
        """Solve the quadratic equation"""
        num_of_variables = len(self._variables)
        if num_of_variables == 0:
            pass
        elif num_of_variables == 1:
            x = self._variables[0]
            a, b, c = self._variables_dict[x][0], self._variables_dict[x][1], self._variables_dict['free']
            mode = mode.lower()
            if mode == 'complex':
                return solve_quadratic(a, b, c)
            elif mode == 'real':
                return solve_quadratic_real(a, b, c)
            elif mode == 'parametric':
                return solve_quadratic_params(a, b, c)
        warnings.warn(
            f"Cannot solve quadratic equations with more than 1 variable, but found {num_of_variables}")
        return None

    def coefficients(self):
        num_of_variables = len(self._variables)
        if num_of_variables == 0:
            return [self._variables_dict['free']]
        elif num_of_variables == 1:
            return self._variables_dict[self._variables[0]] + [self._variables_dict['free']]
        else:
            return self._variables_dict.copy()

    def __str__(self):
        return self._equation

    @staticmethod
    def random(values=(-15, 15), digits_after: int = 0, variable: str = 'x', strict_syntax=True, get_solutions=False):
        if strict_syntax:
            a = random.randint(-5, 5)
            while a == 0:
                a = random.randint(-5, 5)

            m = round(random.uniform(
                values[0] / a, values[1] / a), digits_after)
            while m == 0:
                m = round(random.uniform(
                    values[0] / a, values[1] / a), digits_after)

            n = round(random.uniform(
                values[0] / a, values[1] / a), digits_after)
            while n == 0:
                n = round(random.uniform(
                    values[0] / a, values[1] / a), digits_after)

            b, c = round_decimal(round((m + n) * a, digits_after)
                                 ), round_decimal(round(m * n * a, digits_after))
            a_str = format_coefficient(a)
            b_str = (f"+{b}" if b > 0 else f"{b}") if b != 0 else ""
            if b_str != "":
                if b_str == '1':
                    b_str = f'+{variable}'
                elif b_str == '-1':
                    b_str = f'-{variable}'
                else:
                    b_str += variable
            c_str = (f"+{round_decimal(c)}" if c >
                     0 else f"{c}") if c != 0 else ""
            equation = f"{a_str}{variable}^2{b_str}{c_str} = 0"
            if get_solutions:
                return equation, (-m, -n)
            return equation
        else:
            raise NotImplementedError(
                "Only strict_syntax=True is available at the moment.")

    @staticmethod
    def random_worksheet(path=None, title="Quadratic Equations Worksheet", num_of_equations=20,
                         solutions_range=(-15, 15), digits_after: int = 0, get_solutions=True):
        lines = []
        if get_solutions:
            equations, solutions = [], []
            for i in range(num_of_equations):
                equ, sol = QuadraticEquation.random(values=solutions_range, digits_after=digits_after,
                                                    get_solutions=True)
                equations.append(f"{i + 1}. {equ}")
                solutions.append(f"{i + 1}. {', '.join(sol)}")
            lines.extend((equations, solutions))
        else:
            equations = []
            for i in range(num_of_equations):
                equ = QuadraticEquation.random(values=solutions_range, digits_after=digits_after,
                                               get_solutions=False)
                equations.append(f"{i + 1}. {equ}")
            lines.append(equations)

        create_pdf(path=path, title=title, lines=lines)

    @staticmethod
    def random_worksheets(path=None, num_of_pages=2, equations_per_page=20, titles=None,
                          solutions_range=(-15, 15), digits_after: int = 0, get_solutions=False):
        if titles is None:
            if get_solutions:
                titles = ["Quadratic Equations Worksheet",
                          "Solutions"] * num_of_pages
            else:
                titles = ["Quadratic Equations Worksheet"] * num_of_pages
        lines = []
        if get_solutions:
            for i in range(num_of_pages):
                equations, solutions = [], []
                for j in range(equations_per_page):
                    equ, sol = QuadraticEquation.random(values=solutions_range, digits_after=digits_after,
                                                        get_solutions=True)
                    equations.append(f"{i + 1}. {equ}")
                    solutions.append(f"{i + 1}. {', '.join(sol)}")
                lines.extend((equations, solutions))
            create_pages(path=path, num_of_pages=num_of_pages *
                         2, titles=titles, lines=lines)
        else:
            for i in range(num_of_pages):
                equations = []
                for j in range(equations_per_page):
                    equ = QuadraticEquation.random(values=solutions_range, digits_after=digits_after,
                                                   get_solutions=False)
                    equations.append(f"{i + 1}. {equ}")
                lines.append(equations)
            create_pages(path=path, num_of_pages=num_of_pages,
                         titles=titles, lines=lines)

    def __repr__(self):
        return f"QuadraticEquation({self._equation}, variables={self._variables})"

    def __copy__(self):
        return QuadraticEquation(equation=self._equation, variables=self._variables)
