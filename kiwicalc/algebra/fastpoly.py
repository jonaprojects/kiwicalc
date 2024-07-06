class FastPoly(IExpression, IPlottable):
    __slots__ = ['__variables', '__variables_dict']

    def __init__(self, polynomial: Union[str, dict, list, tuple, float, int], variables: Iterable = None):

        self.__variables = None if variables is None else list(variables)
        if isinstance(polynomial, (int, float)):
            self.__variables = []
            self.__variables_dict = {'free': polynomial}
        if isinstance(polynomial, str):  # Parse a given string
            self.__variables_dict = ParseExpression.parse_polynomial(
                polynomial, self.__variables, strict_syntax=True)
            self.__variables_dict = ParseExpression.parse_polynomial(
                polynomial, self.__variables, strict_syntax=True)
        elif isinstance(polynomial, dict):  # Enter the parsed dictionary
            if "free" not in polynomial.keys():
                raise KeyError(f"Key 'free' must appear in FastPoly.__init__() when entering dict. Its value is the"
                               f"free number of the expression")
            self.__variables_dict = polynomial.copy()
        # Enter the coefficients of a polynomial with 1 variable
        elif isinstance(polynomial, (list, tuple)):
            if not polynomial:
                raise ValueError(
                    f"FastPoly.__init__(): At least one coefficient is required.")
            if self.__variables is None:
                # Default variable is 'x' for simplicity..
                self.__variables = ['x']
            # More than one variable entered - invalid for this method..
            elif len(self.__variables) > 1:
                raise ValueError("FastPoly.__init__(): When entering a list of coefficients, only 1 variable"
                                 f"is accepted, but found {len(self.__variables)}")
            x_coefficients, free_number = polynomial[:-1], polynomial[-1]
            if not x_coefficients:
                self.__variables = []
                self.__variables_dict = {'free': free_number}
            else:
                self.__variables_dict = {
                    self.__variables[0]: x_coefficients, 'free': free_number}

        else:
            raise TypeError(
                f"Invalid type {type(polynomial)} in FastPoly.__init__(). Expected types 'str' or 'dict'")
        if self.__variables is None:
            self.__variables = [
                key for key in self.__variables_dict.keys() if key != 'free']

    @property
    def variables(self):
        return self.__variables.copy()

    @property
    def num_of_variables(self):
        return len(self.__variables)

    @property
    def variables_dict(self):
        return self.__variables_dict.copy()

    @property
    def degree(self) -> Union[float, dict]:
        num_of_variables = len(self.__variables)
        if num_of_variables == 0:
            return 0
        elif num_of_variables == 1:
            return len(self.__variables_dict[self.__variables[0]])
        return {variable: len(self.__variables_dict[variable]) for variable in self.__variables}

    @property
    def is_free_number(self):
        return self.num_of_variables == 0 or len(self.__variables_dict.keys()) == 1

    def derivative(self) -> "FastPoly":
        num_of_variables = self.num_of_variables
        if num_of_variables == 0:
            return FastPoly(0)
        elif num_of_variables == 1:
            variable = self.__variables[0]
            derivative_coefficients = derivative(
                self.__variables_dict[variable] + [self.__variables_dict['free']])
            free_number = derivative_coefficients[-1]
            del derivative_coefficients[-1]
            # if a number is returned ..
            if isinstance(derivative_coefficients, (int, float)):
                derivative_coefficients = []
                self.__variables_dict['free'] = derivative_coefficients
            return FastPoly({variable: derivative_coefficients, 'free': free_number})
        else:
            raise ValueError("Please use the partial_derivative() method for polynomials with several "
                             "variables")

    def partial_derivative(self, variables: Iterable[str]):
        pass

    def extremums(self):
        num_of_variables = len(self.__variables)
        if num_of_variables == 0:
            return None
        elif num_of_variables == 1:
            my_lambda = self.to_lambda()
            my_derivative = self.derivative()
            if my_derivative.is_free_number:
                return None
            derivative_roots = my_derivative.roots(nmax=1000)
            myRoots = [Point2D(root.real, my_lambda(root.real))
                       for root in derivative_roots if root.imag <= 0.00001]
            return PointCollection(myRoots)
        else:
            pass

    def integral(self, c: float = 0, variable='x'):
        num_of_variables = len(self.__variables)
        if num_of_variables == 0:
            return FastPoly({variable: [self.__variables_dict['free']], 'free': c})
        elif num_of_variables != 1:
            raise ValueError(
                "Cannot integrate a PolyFast object with more than 1 variable")
        variables = self.__variables_dict[self.__variables[0]
                                          ] + [self.__variables_dict['free']]
        result = integral(variables, modify_original=True)
        del result[-1]
        return FastPoly({self.__variables[0]: result, 'free': c})

    def newton(self, initial: float = 0, epsilon: float = 0.00001, nmax=10_000):
        return newton_raphson(self.to_lambda(), self.derivative().to_lambda(), initial, epsilon, nmax)

    def halley(self, initial: float = 0, epsilon: float = 0.00001, nmax=10_000):
        first_derivative = self.derivative()
        second_derivative = first_derivative.derivative()
        return halleys_method(self.to_lambda(), first_derivative.to_lambda(), second_derivative.to_lambda(), initial,
                              epsilon, nmax)

    def __add_or_sub(self, other: "FastPoly", mode: str):
        for variable in other.__variables:
            if variable in self.__variables:
                add_or_sub_coefficients(self.__variables_dict[variable], other.__variables_dict[variable], mode=mode,
                                        copy_first=False)
            else:
                self.__variables.append(variable)
                if mode == 'add':
                    self.__variables_dict[variable] = other.__variables_dict[variable].copy(
                    )
                elif mode == 'sub':
                    self.__variables_dict[variable] = [
                        -coef for coef in other.__variables_dict[variable]]

        if mode == 'add':
            self.__variables_dict['free'] += other.__variables_dict['free']
        elif mode == 'sub':
            self.__variables_dict['free'] -= other.__variables_dict['free']

    def __iadd__(self, other: Union[IExpression, int, float]):
        if other == 0:
            return self
        if isinstance(other, (int, float)):
            self.__variables_dict['free'] += other
            return self
        if not isinstance(other, IExpression):
            raise TypeError(f"Invalid type {type(other)} when adding FastPoly objects. Expected types "
                            f"'int', 'float', or 'IExpression'")
        other_evaluation = other.try_evaluate()
        if other_evaluation is not None:
            self.__variables_dict['free'] += other_evaluation
            return self
        if not isinstance(other, FastPoly):
            return ExpressionSum((self, other))
        self.__add_or_sub(other, mode='add')
        return self

    def __isub__(self, other):
        if isinstance(other, (int, float)):
            self.__variables_dict['free'] -= other
            return self
        if not isinstance(other, IExpression):
            raise TypeError(f"Invalid type {type(other)} when subtracting FastPoly objects. Expected types "
                            f"'int', 'float', or 'IExpression'")
        other_evaluation = other.try_evaluate()
        if other_evaluation is not None:
            self.__variables_dict['free'] -= other_evaluation
            return self
        if not isinstance(other, FastPoly):
            return ExpressionSum((self, other))
        self.__add_or_sub(other, mode='sub')
        return self

    def __imul__(self, other):
        pass

    def __itruediv__(self, other):
        pass

    def __ipow__(self, other):
        pass

    def assign(self, **kwargs):
        for variable, value in kwargs.items():
            if variable not in self.__variables_dict:
                continue
            coefficients_length = len(self.__variables_dict[variable])
            for index, coefficient in enumerate(self.__variables_dict[variable]):
                self.__variables_dict['free'] += coefficient * \
                    value ** (coefficients_length - index)
            # Delete the key value pair as it was evaluated into free numbers
            del self.__variables_dict[variable]

    def simplify(self):
        warnings.warn(
            "FastPoly objects are already simplified. Method is deprecated.")

    def try_evaluate(self) -> Optional[float]:
        if self.num_of_variables == 0:
            return self.__variables_dict['free']
        return None

    def roots(self, epsilon=0.00001, nmax: int = 10000):
        num_of_variables = len(self.__variables)
        if num_of_variables == 0:
            return "Infinite" if self.__variables_dict['free'] == 0 else None
        elif num_of_variables == 1:
            return solve_polynomial(self.__variables_dict[self.__variables[0]] + [self.__variables_dict['free']],
                                    epsilon, nmax)
        else:
            raise ValueError(
                f"Can only solve polynomials with 1 variable, but found {num_of_variables}")

    def __eq__(self, other: "Union[IExpression, FastPoly]"):
        """Equate between expressions. not fully compatible with the IExpression classes ..."""
        if other is None:
            return False
        if not isinstance(other, IExpression):
            raise TypeError(
                f"Invalid type {type(other)} for equating FastPoly objects")
        other_evaluation = other.try_evaluate()
        if other_evaluation is not None:
            my_evaluation = self.try_evaluate()
            if my_evaluation is not None:
                return my_evaluation == other_evaluation
        if not isinstance(other, FastPoly):
            return False
        return self.__variables_dict == other.__variables_dict

    def __ne__(self, other: "FastPoly"):
        return not self.__eq__(other)

    def __neg__(self):
        new_dict = {variable: [-coefficient for coefficient in coefficients] for variable, coefficients in
                    self.__variables_dict.items() if variable != 'free'}
        new_dict['free'] = -self.__variables_dict['free']
        return FastPoly(new_dict)

    def __copy__(self):
        # the dictionary is later copied so no memory will be shared
        return FastPoly(self.__variables_dict)

    def to_lambda(self):
        return to_lambda(self.__str__(), self.__variables)

    def plot(self, start: float = -10, stop: float = 10,
             step: float = 0.01, ymin: float = -10, ymax: float = 10, text=None, show_axis=True, show=True,
             fig=None, ax=None, formatText=True, values=None):
        lambda_expression = self.to_lambda()
        num_of_variables = self.num_of_variables
        if text is None:
            text = self.__str__()
        if num_of_variables == 0:  # TODO: plot this in a number axis
            raise ValueError("Cannot plot a polynomial with 0 variables_dict")
        elif num_of_variables == 1:
            plot_function(lambda_expression, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=text,
                          show_axis=show_axis, show=show, fig=fig, ax=ax, formatText=formatText, values=values)
        elif num_of_variables == 2:
            # TODO: update the parameters
            plot_function_3d(lambda_expression, start=start,
                             stop=stop, step=step)
        else:
            raise ValueError(
                "Cannot plot a function with more than two variables_dict (As for this version)")

    def to_dict(self):
        return {"type": "FastPoly", "data": self.__variables_dict.copy()}

    @staticmethod
    def from_dict(given_dict: dict):
        return FastPoly(given_dict)

    @staticmethod
    def from_json(json_content: str):
        loaded_json = json.loads(json_content)
        if loaded_json['type'].strip().lower() != 'fastpoly':
            raise ValueError(f"Unexpected type '{loaded_json['type']}' when creating a new "
                             f"FastPoly object from JSON (Expected TypePoly).")
        return FastPoly(loaded_json['data'])

    @staticmethod
    def import_json(path):
        with open(path, 'r') as json_file:
            return FastPoly.from_json(json_file.read())

    def python_syntax(self):
        return ParseExpression.unparse_polynomial(parsed_dict=self.__variables_dict, syntax='pythonic')

    def __str__(self):
        return ParseExpression.unparse_polynomial(parsed_dict=self.__variables_dict)
