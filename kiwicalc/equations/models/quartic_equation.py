class QuarticEquation(Equation):
    def __init__(self, equation: str, variables: Iterable[Optional[str]] = None, strict_syntax=False):
        self.__strict_syntax = self.__strict_syntax
        super().__init__(equation, variables)

    def _extract_variables(self):
        return ParseExpression.parse_quartic(self.first_side, self._variables, strict_syntax=self.__strict_syntax)

    def solve(self):
        a, b, c = self._variables_dict['x'][0], self._variables_dict['x'][1], self._variables_dict['x'][2]
        d, e = self._variables_dict['x'][3], self._variables_dict['free']
        return solve_quartic(a, b, c, d, e)

    def coefficients(self):
        a, b, c = self._variables_dict['a'], self._variables_dict['b'], self._variables_dict['c']
        d, e = self._variables_dict['d'], self._variables_dict['e']
        return a, b, c, d, e

    @staticmethod
    def random(solutions_range: Tuple[float, float] = (-15, 15), digits_after: int = 0, variable='x',
               get_solutions=False):
        result = random_polynomial(degree=3, solutions_range=solutions_range, digits_after=digits_after,
                                   variable=variable, get_solutions=get_solutions)
        if isinstance(result, str):
            return result + " = 0"
        else:
            return result[0] + "= 0", result[1]

    @staticmethod
    def random_worksheet(path=None, title=" Cubic Equations Worksheet", num_of_equations=20,
                         solutions_range=(-15, 15), digits_after: int = 0, get_solutions=False):
        PolyEquation.random_worksheet(path=path, title=title, num_of_equations=num_of_equations, degrees_range=(3, 3),
                                      solutions_range=solutions_range, digits_after=digits_after,
                                      get_solutions=get_solutions)

    @staticmethod
    def random_worksheets(path=None, num_of_pages=2, equations_per_page=20, titles=None,
                          solutions_range=(-15, 15), digits_after: int = 0, get_solutions=False):
        if titles is None:
            if get_solutions:
                titles = ['Quartic Equations Worksheet',
                          'Solutions'] * num_of_pages
            else:
                titles = ['Quartic Equations Worksheet'] * num_of_pages
        PolyEquation.random_worksheets(
            path=path, num_of_pages=num_of_pages, titles=titles, equations_per_page=equations_per_page,
            degrees_range=(4, 4),
            solutions_range=solutions_range, digits_after=digits_after, get_solutions=get_solutions
        )

    def __repr__(self):
        return f"QuarticEquation({self._equation}, variables={self._variables})"

    def __copy__(self):
        return QuarticEquation(equation=self._equation, variables=self._variables)
