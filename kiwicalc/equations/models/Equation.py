class Equation(ABC):
    def __init__(self, equation: str, variables: Iterable = None, calc_now: bool = False):
        """The base function of creating a new Equation"""
        self._equation = clean_spaces(equation)
        if variables is None:
            self._variables = get_equation_variables(equation)
            self._variables_dict = self._extract_variables()
            try:
                index = self._variables.index("number")
                del self._variables[index]
            except ValueError:
                pass
        else:
            self._variables = list(variables)
            self._variables_dict = {variable: 0 for variable in variables}
            self._variables_dict["number"] = 0

        if calc_now:
            self._solution = self.solve()
        else:
            self._solution = None

    # PROPERTIES
    @property
    def equation(self):
        return self._equation

    @property
    def variables(self):
        return self._variables

    @property
    def num_of_variables(self):
        return len(self._variables)

    @property
    def first_side(self):
        return self._equation[:self._equation.rfind("=")]

    @property
    def second_side(self):
        return self._equation[self._equation.rfind("=") + 1:]

    @property
    def solution(self):
        if self._solution is None:
            self._solution = self.solve()
        return self._solution

    @property
    def variables_dict(self):
        return self._variables_dict

    @abstractmethod
    def _extract_variables(self):
        return extract_dict_from_equation(self._equation)

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def __copy__(self):
        return LinearEquation(self._equation)

    def __reversed__(self):
        """
        reverses the sides of the equation. for example: '3x+5=14' -> '14=3x+5'
        :return:
        """
        equal_index = self.equation.find('=')
        first_side, second_side = self.equation[:
                                                equal_index], self.equation[equal_index + 1:]
        return LinearEquation(f'{second_side}={first_side}')

    @abstractmethod
    def __repr__(self):
        pass

    def __str__(self):
        return self._equation
