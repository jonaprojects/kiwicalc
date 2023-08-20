class LinearSystem:
    """
    This class represents a system of linear __equations.
    It solves them via a simple implementation of the Gaussian Elimination technique.
    """

    def __init__(self, equations: Iterable, variables: Iterable = None):
        """
        Creating a new equation system

        :param equations: An iterable collection of equations. Each equation in the collection can be of type
        string or Equation
        :param variables:(Optional) an iterable collection of strings that be converted to a list.
        Each item represents a variable in the equations. For example, ('x','y','z').
        """
        self.__equations, self.__variables = [], list(
            variables) if variables is not None else []
        self.__variables_dict = dict()
        for equation in equations:
            if isinstance(equation, str):
                self.__equations.append(LinearEquation(equation))
            elif isinstance(equation, LinearEquation):
                self.__equations.append(equation)
            else:
                raise TypeError

    # PROPERTIES
    @property
    def equations(self):
        return self.__equations

    @property
    def variables(self):
        return self.__variables

    def add_equation(self, equation: str):
        self.__equations.append(LinearEquation(equation))

    def __extract_variables(self):
        variables_dict = {}
        for equation in self.__equations:
            if not equation.variables_dict:
                equation.__variables = equation.variables_dict
            for variable in equation.variables_dict:
                if variable not in variables_dict and variable != "number":
                    variables_dict[variable] = 0
        variables_dict["number"] = 0
        self.__variables_dict = variables_dict
        return variables_dict

    def to_matrix(self):
        """
        Converts the equation system to a matrix of _coefficient, so later the Gaussian elimination method wil
        be implemented on it, in order to solve the system.
        :return:
        """
        variables = self.__variables_dict if self.__variables_dict else self.__extract_variables()
        values_matrix = []
        for equation in self.__equations:
            equation.__variables = variables
            equal_index = equation.equation.find('=')
            side1, side2 = equation.equation[:equal_index], equation.equation[equal_index + 1:]
            first_dict = simplify_linear_expression(
                side1, equation.variables_dict)
            second_dict = simplify_linear_expression(
                side2, equation.variables_dict)
            result_dict = subtract_dicts(second_dict, first_dict)
            values_matrix.append(list(result_dict.values()))

        return values_matrix

    def to_matrix_and_vector(self):
        pass

    def get_solutions(self):
        """
        fetches the solutions
        :return: returns a dictionary that contains the name of each variable, and it's (real) __solution.
        for example: {'x':6,'y':4}
        This comes handy later since you can access simply the solutions.
        """
        values_matrix = self.to_matrix()
        matrix_obj = Matrix(matrix=values_matrix)
        matrix_obj.gauss()
        answers = {}
        keys = list(self.__variables_dict.keys())
        i = 0
        for row in matrix_obj.matrix:
            answers[keys[i]] = -round_decimal(row[len(row) - 1])
            i += 1
        return answers

    def simplify(self):
        pass

    def print_solutions(self):
        """
        prints out the solutions of the equation.
        :return: None
        """
        solutions = self.get_solutions()
        for key, value in solutions.items():
            print(f'{key} = {value}')
