class Abs(IExpression, IPlottable, IScatterable):
    """A class for representing expressions with absolute values. For instance, Abs(x) is the same as |x|."""
    __slots__ = ['_coefficient', '_expression', '_power']

    def __init__(self, expression: Union[IExpression, int, float], power: Union[int, float, IExpression] = 1,
                 coefficient: Union[int, float, IExpression] = 1, gen_copies=True):

        # Handling the expression
        if isinstance(expression, (int, float)):
            self._expression = Mono(expression)
        elif isinstance(expression, IExpression):
            self._expression = expression.__copy__() if gen_copies else expression
        else:
            raise TypeError(f"Invalid type {type(expression)} for inner expression when creating an Abs object.")

        # Handling the power
        if isinstance(power, (int, float)):
            self._power = Mono(power)
        elif isinstance(power, IExpression):  # Allow algebraic powers here?
            self._power = power.__copy__() if gen_copies else power
        else:
            raise TypeError(f"Invalid type {type(power)} for 'power' argument when creating a new Abs object.")

        # Handling the coefficient
        if isinstance(coefficient, (int, float)):
            self._coefficient = Mono(coefficient)
        elif isinstance(coefficient, IExpression):  # Allow algebraic powers here?
            self._coefficient = coefficient.__copy__() if gen_copies else coefficient
        else:
            raise TypeError(
                f"Invalid type {type(coefficient)} for 'coefficient' argument when creating a new Abs object.")

    @property
    def coefficient(self):
        return self._coefficient

    @property
    def expression(self):
        return self._expression

    @property
    def power(self):
        return self._power

    @property
    def variables(self):
        variables = self._coefficient.variables
        variables.update(self._expression.variables)
        variables.update(self._power.variables)
        return variables

    def simplify(self):
        self._coefficient.simplify()
        self._expression.simplify()
        self._power.simplify()

    def assign(self, **kwargs):
        self._coefficient.assign(**kwargs)
        self._expression.assign(**kwargs)
        self._expression.assign(**kwargs)

    def to_dict(self):
        return {
            "type": "Abs",
            "coefficient": self._coefficient.to_dict(),
            "expression": self._expression.to_dict(),
            "power": self._power.to_dict()
        }

    @staticmethod
    def from_dict(given_dict: dict):
        expression_obj = create_from_dict(given_dict['expression'])
        coefficient_obj = create_from_dict(given_dict['coefficient'])
        power_obj = create_from_dict(given_dict['power'])
        return Abs(expression=expression_obj, power=power_obj, coefficient=coefficient_obj)

    def __add_or_sub(self, other, operation: str = '+'):
        if isinstance(other, (int, float)):
            my_evaluation = self.try_evaluate()
            if my_evaluation is not None:
                if operation == '+':
                    return Mono(my_evaluation + other)
                else:
                    return Mono(my_evaluation - other)
            else:
                if operation == '+':
                    return ExpressionSum([self, Mono(other)])
                else:
                    return ExpressionSum([self, Mono(-other)])

        elif isinstance(other, IExpression):
            my_evaluation = self.try_evaluate()
            other_evaluation = other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                if operation == '+':
                    return Mono(my_evaluation + other_evaluation)
                return Mono(my_evaluation - other_evaluation)
            if (my_evaluation, other_evaluation) == (None, None):
                if isinstance(other, Abs):
                    if self._power == other._power:
                        if self._expression == other._expression or self._expression == -other._expression:
                            # |x| = |x|, or |x| = |-x|. Find whether the two expressions are compatible for addition.
                            if operation == '+':
                                self._coefficient += other._coefficient
                            else:
                                self._coefficient -= other._coefficient
                            return self

            return ExpressionSum((self, other))

    def __iadd__(self, other: Union[int, float, IExpression]):
        return self.__add_or_sub(other, operation='+')

    def __isub__(self, other):
        return self.__add_or_sub(other, operation='-')

    def __imul__(self, other: Union[int, float, IExpression]):
        if not isinstance(other, (int, float, IExpression)):
            raise TypeError(f" Invalid type: {type(other)} when multiplying an Abs object."
                            f" Expected types 'int', 'float', 'IExpression'.")
        if isinstance(other, (int, float)):
            self._coefficient *= other
            return self
        my_evaluation = self.try_evaluate()
        other_evaluation = other.try_evaluate()
        if None not in (my_evaluation, other_evaluation):
            return Mono(my_evaluation * other_evaluation)
        if other_evaluation is not None:
            self._coefficient *= other_evaluation
            return self

        if not isinstance(other, Abs):
            self._coefficient *= other
            return self
        # If other is indeed an Abs object:
        # Find whether the two expressions are connected somehow
        if self._expression == other._expression or self._expression == -other._expression:
            self._power += other._power
            self._coefficient *= other._coefficient
            return self
        return ExpressionMul((self, other))  # TODO: implement it later

    def __itruediv__(self, other: Union[int, float, IExpression]):
        if not isinstance(other, (int, float, IExpression)):
            raise TypeError(f" Invalid type: {type(other)} when dividing an Abs object."
                            f" Expected types 'int', 'float', 'IExpression'.")
        if other == 0:
            raise ValueError(f"Cannot divide an Abs object by 0.")
        if isinstance(other, (int, float)):
            self._coefficient /= other
            return self
        my_evaluation, other_evaluation = self.try_evaluate(), other.try_evaluate()
        if other_evaluation == 0:
            raise ValueError(f"Cannot divide an Abs object by 0.")
        if None not in (my_evaluation, other_evaluation):
            return Mono(my_evaluation / other_evaluation)
        if other_evaluation is not None:
            self._coefficient /= other
            return self
        if not isinstance(other, Abs):
            self._coefficient /= other
            return self
        # TODO: revise this solution...
        if self._expression == other._expression or self._expression == -other._expression:
            power_difference = self._power - other._power  # also handle cases such as |x|^x / |x|^(x-1) = |x|
            difference_evaluation = power_difference.try_evaluate()
            if difference_evaluation is None:
                self._coefficient /= other._coefficient
                return Exponent(coefficient=self._coefficient, base=self._expression, power=power_difference)
            else:
                if difference_evaluation > 0:
                    self._power = Mono(difference_evaluation)
                    self._coefficient /= other._coefficient
                    return Abs(coefficient=self._coefficient, power=self._power, expression=self._expression,
                               gen_copies=False)
                elif difference_evaluation == 0:
                    return self._coefficient
                else:
                    return Fraction(self._coefficient / other._coefficient,
                                    Abs(self._expression, -difference_evaluation))
        return Fraction(self, other)  # TODO: implement it later

    def __ipow__(self, power: Union[int, float, IExpression]):
        if not isinstance(power, (int, float, IExpression)):
            raise TypeError(f"Invalid type: {type(power)} when raising by a power an Abs object."
                            f" Expected types 'int', 'float', 'IExpression'.")
        if isinstance(power, (int, float)):
            self._coefficient **= power
            self._power *= power
            return self
        # The power is an algebraic expression
        power_evaluation = power.try_evaluate()
        if power_evaluation is not None:
            self._coefficient **= power
            self._power *= power
            return self
        return Exponent(self, power)

    def __neg__(self):
        return Abs(expression=self._expression, power=self._power, coefficient=self._coefficient.__neg__())

    def __eq__(self, other: Union[IExpression, int, float]):
        if isinstance(other, (int, float)):
            my_evaluation = self.try_evaluate()
            return my_evaluation == other

        if isinstance(other, IExpression):
            my_evaluation = self.try_evaluate()
            other_evaluation = other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                return my_evaluation == other_evaluation
            if (my_evaluation, other_evaluation) == (None, None):
                if isinstance(other, Abs):
                    if self._expression == other._expression:
                        return (self._coefficient, self._power) == (other._coefficient, other._power)
                expression_evaluation = self._expression.try_evaluate()  # computed the second time - waste..
                if expression_evaluation is not None:
                    return self._coefficient * abs(expression_evaluation) ** self._power == other
            return False
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def derivative(self, get_derivatives=False):
        warnings.warn("Derivatives are still experimental, and might not work for other algebraic expressions"
                      "rather than polynomials.")
        num_of_variables = len(self.variables)
        if num_of_variables == 0:
            return lambda x: self.try_evaluate()
        assert num_of_variables == 1, "Use partial derivatives of expressions with several variables."
        positive_expression = self._coefficient * self._expression ** self._power
        try:
            positive_derivative = positive_expression.derivative()
        except:
            return None
        negative_derivative = -positive_derivative
        if get_derivatives:
            return positive_derivative, negative_derivative
        positive_derivative, negative_derivative = positive_derivative.to_lambda(), positive_derivative.to_lambda()
        return lambda x: positive_derivative(x) if x > 0 else (negative_derivative(x) if x < 0 else 0)

    def integral(self, other):
        pass

    def try_evaluate(self) -> Optional[Union[int, float]]:
        coefficient_evaluation = self._coefficient.try_evaluate()
        if coefficient_evaluation is None:
            return None
        if coefficient_evaluation == 0:
            return 0
        expression_evaluation = self._expression.try_evaluate()
        power_evaluation = self._power.try_evaluate()
        if power_evaluation is None:
            return None
        if power_evaluation == 0:
            return coefficient_evaluation
        if expression_evaluation is None:
            return None
        return coefficient_evaluation * abs(expression_evaluation) ** power_evaluation

    def __str__(self):
        if self._coefficient == 0 or self._expression == 0:
            return "0"
        if self._power == 0:
            return self._coefficient.__str__()
        elif self._power == 1:
            power_string = ""
        else:
            power_string = f"**{self._power.python_syntax()}"
        if self._coefficient == 1:
            coefficient_string = f""
        elif self._coefficient == -1:
            coefficient_string = f"-"
        else:
            coefficient_string = f"{self._coefficient.__str__()}*"
        return f"{coefficient_string}|{self._expression}|{power_string}"

    def __copy__(self):
        return Abs(expression=self._expression, power=self._power, coefficient=self._coefficient, gen_copies=True)

