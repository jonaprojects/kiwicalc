class Factorial(IExpression, IPlottable, IScatterable):
    __slots__ = ['_coefficient', '_expression', '_power']

    def __init__(self, expression: Optional[Union[IExpression, int, float, str]],
                 coefficient: Union[IExpression, int, float] = Mono(1),
                 power: Union[IExpression, int, float] = Mono(1), dtype=''):

        if isinstance(coefficient, (int, float)):
            self._coefficient = Mono(coefficient)
        else:
            self._coefficient = coefficient.__copy__()

        if isinstance(power, (int, float)):
            self._power = Mono(power)
        else:
            self._power = power.__copy__()

        if isinstance(expression, (int, float)):
            self._expression = Mono(expression)
        else:
            self._expression = expression.__copy__()

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
        """ A set of all of the existing variables_dict inside the expression"""
        coefficient_variables: set = self._coefficient.variables
        coefficient_variables.update(self._expression.variables)
        coefficient_variables.update(self._power.variables)
        return coefficient_variables

    def to_dict(self):
        return {
            "type": "Factorial",
            "coefficient": self._coefficient.to_dict(),
            "expression": self._expression.to_dict(),
            "power": self._power.to_dict()
        }

    @staticmethod
    def from_dict(given_dict: dict):
        expression_obj = create_from_dict(given_dict['expression'])
        coefficient_obj = create_from_dict(given_dict['coefficient'])
        power_obj = create_from_dict(given_dict['power'])
        return Factorial(expression=expression_obj, power=power_obj, coefficient=coefficient_obj)

    def __iadd__(self, other: Union[int, float, IExpression]):
        if other == 0:
            return self
        if isinstance(other, (int, float)):
            other = Mono(other)
        if isinstance(other, Factorial):
            if self._expression == other._expression and self._power == other._power:
                self._coefficient += other._coefficient
                return self
        return ExpressionSum((self, other))

    def __isub__(self, other):
        if other == 0:
            return self
        if isinstance(other, (int, float)):
            other = Mono(other)
        if isinstance(other, Factorial):
            if self._expression == other._expression and self._power == other._power:
                self._coefficient -= other._coefficient
                return self
        return ExpressionSum((self, other))

    def __imul__(self, other: Union[IExpression, int, float]):
        if self._expression == other - 1:
            self._expression += 1
            return self
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            if my_evaluation is None:
                self._coefficient *= other
                return self
            return Mono(coefficient=my_evaluation * other)
        elif isinstance(other, IExpression):
            if isinstance(other, Factorial):
                if self._expression == other._expression:
                    self._coefficient *= other._coefficient
                    self._power += other._power
                    return self
                else:
                    return ExpressionSum((self, other))
            other_evaluation = other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                return Mono(coefficient=my_evaluation * other_evaluation)
            else:
                self._coefficient *= other
                return self
        else:
            raise TypeError(
                f"Invalid type '{type(other)}' when multiplying a factorial object with id:{id(other)}")

    def __mul__(self, other: Union[IExpression, int, float]):
        return self.__copy__().__imul__(other)

    def __itruediv__(self, other: Union[IExpression, int, float]) -> "Optional[Union[Fraction,Factorial]]":
        if other == 0:
            raise ZeroDivisionError(
                "Cannot divide a factorial expression by 0")
        if other == self._expression:  # For example: 8! / 8 = 7!
            if other == self._coefficient:
                self._coefficient = Mono(1)
                return self
            if isinstance(other, IExpression):
                division_with_coefficient = self._coefficient / other
                division_eval = division_with_coefficient.try_evaluate()
                if division_eval is not None:
                    self._coefficient = Mono(division_eval)
                    return self

            self._expression -= 1
            self.simplify()
            return self

        if isinstance(other, (int, float)):
            self._coefficient /= other
            self.simplify()
            return self

        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if other_evaluation is not None:
                # The expression can be evaluated into a float or an int
                if other_evaluation == 0:
                    raise ZeroDivisionError(
                        "Cannot divide a factorial expression by 0")
                if other_evaluation == self._expression:
                    self._expression -= 1
                    self.simplify()
                    return self
                else:
                    self._coefficient /= other
                    self.simplify()
                    return self
            elif isinstance(other, Factorial):  # TODO: poorly implemented!
                if self._expression == other._expression:
                    self._coefficient /= other._coefficient
                    self._power -= other._power
                    self.simplify()

            else:  # Just a random IExpression - just return a Fraction ..
                return Fraction(self, other)

        else:
            raise TypeError(
                f"Invalid type for dividing factorials: '{type(other)}'")

    def __rtruediv__(self, other: Union[int, float, IExpression]):
        my_evaluation = self.try_evaluate()
        if my_evaluation == 0:
            raise ZeroDivisionError("Cannot divide by 0: Tried to divide by a Factorial expression that evaluates"
                                    "to zero")
        if isinstance(other, (int, float)):
            if my_evaluation is not None:
                return Mono(other / my_evaluation)
            return Fraction(other, self)
        elif isinstance(other, IExpression):
            return other.__truediv__(self)
        else:
            raise TypeError(
                "Invalid type for dividing an expression by a Factorial object.")

    def __ipow__(self, other: Union[int, float, IExpression]):
        self._power *= other
        return self

    def __pow__(self, power):
        return self.__copy__().__ipow__(power)

    def __neg__(self):
        if self._expression is None:
            return Factorial(
                coefficient=self._coefficient.__neg__(),
                expression=None,
                power=Mono(1)
            )
        return Factorial(
            coefficient=self._coefficient.__neg__(),
            expression=self._expression.__neg__(),
            power=self._power.__neg__()
        )

    def assign(self, **kwargs):
        self._coefficient.assign(**kwargs)
        self._expression.assign(**kwargs)
        self._power.assign(**kwargs)
        self.simplify()

    def try_evaluate(self) -> Optional[Union[int, float]]:
        if self._coefficient == 0:
            return 0
        coefficient_evaluation = self._coefficient.try_evaluate()
        if self._expression is None:
            if coefficient_evaluation is not None:
                return coefficient_evaluation
            return None
        expression_evaluation = self._expression.try_evaluate()
        power_evaluation = self._power.try_evaluate()
        # If all can be evaluated
        if None not in (coefficient_evaluation, expression_evaluation, power_evaluation):
            if expression_evaluation < 0:
                return None  # Cannot evaluate negative factorials
            if expression_evaluation == 0:
                my_factorial = 1
            elif expression_evaluation == int(
                    expression_evaluation):  # If the expression can be evaluated to an integer
                my_factorial = factorial(int(expression_evaluation))
            else:  # Factorials of decimal numbers
                my_factorial = gamma(expression_evaluation) * \
                    expression_evaluation

            return coefficient_evaluation * my_factorial ** power_evaluation
        elif power_evaluation == 0 and coefficient_evaluation is not None:
            # Can disregard if the expression can't be evaluated, because power by 0 is 1
            # coefficient * (...) ** 0 = coefficient * 1 = coefficient
            return coefficient_evaluation
        return None  # Couldn't evaluate

    def simplify(self):
        """Try to simplify the factorial expression"""
        self._coefficient.simplify()
        if self._coefficient == 0:
            self._expression = None
            self._power = Mono(1)

    def python_syntax(self):
        if self._expression is None:
            return f"{self._coefficient.python_syntax()}"
        return f"{self._coefficient} * factorial({self._expression.python_syntax()}) ** {self._power.python_syntax()}"

    def __str__(self):
        if self._expression is None:
            return f"{self._coefficient}"
        coefficient_str = format_coefficient(self._coefficient)
        if coefficient_str not in ('', '-'):
            coefficient_str += '*'
        power_str = f"**{self._power.__str__()}" if self._power != 1 else ""
        inside_str = self._expression.__str__()
        if '-' in inside_str or '+' in inside_str or '*' in inside_str or '/' in inside_str:
            inside_str = f'({inside_str})'
        expression_str = f"({inside_str}!)" if coefficient_str != "" else f"{inside_str}!"
        if power_str == "":
            return f"{coefficient_str}{expression_str}"
        return f"{coefficient_str}({expression_str}){power_str}"

    def __copy__(self):
        return Factorial(
            coefficient=self._coefficient,
            expression=self._expression,
            power=self._power

        )

    def __eq__(self, other: Union[IExpression, int, float]):
        if other is None:
            return False
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            if my_evaluation is not None:
                return my_evaluation == other
            return False
        elif isinstance(other, IExpression):
            if my_evaluation is not None:  # If self can be evaluated
                other_evaluation = other.try_evaluate()
                return other_evaluation is not None and my_evaluation == other_evaluation
            if isinstance(other, Factorial):
                return self._coefficient == other._coefficient and self._expression == other._expression and self._power == other._power
            return False

        else:
            raise TypeError(
                f"Invalid type '{type(other)}' for equating with a Factorial expression.")

    def __ne__(self, other: Union[IExpression, int, float]):
        return not self.__eq__(other)
