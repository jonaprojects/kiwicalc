class Exponent(IExpression):
    """
    This class enables you to represent expressions such as x^x, e^x, (3x)^sin(x), etc.
    """
    __slots__ = ['_coefficient', '_base', '_power']

    def __init__(self, base: Union[IExpression, float], power: Union[IExpression, float, int],
                 coefficient: Optional[Union[int, float, IExpression]] = None, gen_copies=True):
        if isinstance(base, IExpression):
            self._base = base.__copy__() if gen_copies else base
        elif isinstance(base, (int, float)):
            self._base = Mono(base)
        else:
            raise TypeError(
                f"Exponent.__init__(): Invalid type {type(base)} for parameter 'base'.")

        if isinstance(power, IExpression):
            self._power = power.__copy__() if gen_copies else power
        elif isinstance(power, (int, float)):
            self._power = Mono(power)
        else:
            raise TypeError(
                f"Exponent.__init__(): Invalid type {type(power)} for parameter 'power'.")

        if coefficient is None:
            self._coefficient = Mono(1)
        elif isinstance(coefficient, IExpression):
            if gen_copies:
                self._coefficient = coefficient.__copy__()
            else:
                self._coefficient = coefficient
        elif isinstance(coefficient, (int, float)):
            self._coefficient = Mono(coefficient)
        else:
            raise TypeError(
                f"Invalid type for coefficient of Exponent object: '{coefficient}'.")

    def __add_or_sub(self, other, operation='+'):
        if other == 0:
            return self
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            if my_evaluation is None:
                if operation == '+':
                    return ExpressionSum((self, Mono(other)))
                return ExpressionSum((self, Mono(-other)))
            else:
                if operation == '+':
                    return Mono(my_evaluation + other)
                return Mono(my_evaluation - other)

        elif isinstance(other, IExpression):
            other_evaluation = self.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                if operation == '+':
                    return Mono(my_evaluation + other_evaluation)
                return Mono(my_evaluation - other_evaluation)
            elif other_evaluation is not None:
                if operation == '+':
                    return ExpressionSum((self, Mono(other_evaluation)))
                return ExpressionSum((self, Mono(-other)))
            else:  # both expressions cannot be evaluated into numbers
                if not isinstance(other, Exponent):
                    if operation == '+':
                        return ExpressionSum((self, other))
                    return ExpressionSum((self, -other))
                else:  # If we're dealing with another exponent expression.
                    if self._power == other._power and self._base == other._base:
                        # if the exponents have the same base and powers.
                        if operation == '+':
                            self._coefficient += other._coefficient
                        else:
                            self._coefficient -= other._coefficient
                        return self
                    elif False:  # TODO: check for relations in the base and power pairs?
                        pass
                    else:
                        if operation == '+':
                            return ExpressionSum((self, other))
                        return ExpressionSum((self, -other))

    # TODO: further implement
    def __iadd__(self, other: "Union[IExpression, int, float]"):
        return self.__add_or_sub(other, operation='+')

    def __isub__(self, other):
        return self.__add_or_sub(other, operation='-')

    # TODO: further implement
    def __imul__(self, other: Union[int, float, IExpression]):
        if other == 0:
            return Mono(0)

        if other == self.base:
            self._power += 1
            return self
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            if my_evaluation is not None:
                return Mono(my_evaluation * other)
            self.multiply_by_number(other)
            return self
        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                return Mono(my_evaluation * other_evaluation)
            elif other_evaluation is not None:
                self.multiply_by_number(other_evaluation)
                return self
            else:
                if isinstance(other, Exponent):
                    if self._power == other._power:
                        self._base *= other._base
                        return self
                    # Relation between the powers ( for example x and 2x)
                    elif False:
                        pass
                    else:
                        return ExpressionMul((self, other))

    def __mul__(self, other: Union[int, float, IExpression]):
        return self.__copy__().__imul__(other)

    def multiply_by_number(self, number: Union[int, float]):
        self._coefficient *= number

    def divide_by_number(self, number: Union[int, float]):
        if number == 0:
            raise ZeroDivisionError("Cannot divide an expression by 0")
        self._coefficient /= number

    def __itruediv__(self, other: Union[int, float, IExpression]):
        return Fraction(self, other)

    def __ipow__(self, other: Union[IExpression, int, float]):
        if other == 0:
            # because then the expression would be: coefficient * expression ^ 0
            return self._coefficient.__copy__()
        self._power *= other
        return self

    def __pow__(self, power: Union[int, float, IExpression], modulo=None):
        return self.__copy__().__ipow__(power)

    def __neg__(self):
        copy_of_self = self.__copy__()
        copy_of_self._coefficient *= -1
        return copy_of_self

    def to_dict(self):
        return {
            "type": "Factorial",
            "coefficient": self._coefficient.to_dict(),
            "base": self._base.to_dict(),
            "power": self._power.to_dict()
        }

    @staticmethod
    def from_dict(given_dict: dict):
        base_obj = create_from_dict(given_dict['base'])
        coefficient_obj = create_from_dict(given_dict['coefficient'])
        power_obj = create_from_dict(given_dict['power'])
        return Exponent(base=base_obj, power=power_obj, coefficient=coefficient_obj)

    def derivative(self):  # TODO: improve this method
        my_variables = self.variables
        variables_length = len(my_variables)
        if variables_length == 0:
            # Free number, then the derivative is 0
            return Mono(0)
        elif variables_length == 1:
            coefficient_eval = self._coefficient.try_evaluate()
            base_eval = self._base.try_evaluate()
            power_eval = self._power.try_evaluate()
            if None not in (coefficient_eval, base_eval, power_eval) or coefficient_eval == 0 or base_eval == 0:
                return Mono(0)

            if power_eval is not None and power_eval == 0:
                return self._coefficient.derivative()  # for instance: x**2 ^0 -> 1

            if coefficient_eval is not None:
                if power_eval is not None:  # cases such as 3x^2 or  5sin(2x)^4
                    expression = (coefficient_eval * self._base **
                                  power_eval).derivative()
                    if hasattr(expression, "derivative"):
                        return expression.derivative()
                    warnings.warn(
                        "This kind of derivative isn't supported yet...")
                    return None

                elif base_eval is not None:  # examples such as 2^x
                    if base_eval < 0:
                        warnings.warn(
                            f"The derivative of this expression is undefined")
                        return None
                    return self * self._coefficient.derivative() * ln(base_eval)
        else:
            raise ValueError(
                "For derivatives with more than 1 variable, use partial derivatives")

    @property
    def variables(self):
        my_variables = self._coefficient.variables
        my_variables.update(self._base.variables)
        my_variables.update(self._power.variables)
        return my_variables

    def partial_derivative(self):
        raise NotImplementedError(
            "This feature is not supported yet. Stay tuned for the next versions.")

    def integral(self):
        raise NotImplementedError(
            "This feature is not supported yet. Stay tuned for the next versions.")

    @property
    def base(self):
        return self._base

    @property
    def power(self):
        return self._power

    def assign(self, **kwargs):
        self._coefficient.assign(**kwargs)
        self._base.assign(**kwargs)
        self._power.assign(**kwargs)

    def when(self, **kwargs):
        copy_of_self = self.__copy__()
        copy_of_self.assign(**kwargs)
        return copy_of_self

    def simplify(self) -> None:  # TODO: improve this somehow....
        self._coefficient.simplify()
        self._base.simplify()
        self._power.simplify()

    def try_evaluate(self) -> Optional[Union[int, float]]:
        if self._coefficient == 0:
            return 0
        coefficient_evaluation = self._coefficient.try_evaluate()
        if coefficient_evaluation is None:
            return None
        power_evaluation = self._power.try_evaluate()
        if power_evaluation is None:
            return None
        if power_evaluation == 0:  # 3*x^0 for example will be evaluated to 3
            return coefficient_evaluation
        base_evaluation = self._base.try_evaluate()
        if base_evaluation is None:
            return None
        return coefficient_evaluation * (base_evaluation ** power_evaluation)

    def __eq__(self, other: Union[IExpression, int, float]):
        if other is None:
            return False
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            return my_evaluation == other
        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                return my_evaluation == other_evaluation
            elif my_evaluation is other_evaluation is None:
                # Compare between the objects
                if isinstance(other, Exponent):
                    equal_coefficients = self._coefficient == other._coefficient
                    equal_bases = self._base == other._base
                    equal_powers = self._power == other._power
                    if equal_coefficients and equal_bases and equal_powers:
                        return True
                    # TODO: check for other cases where the expressions will be equal, such as 2^(2x) and 4^x
                else:
                    # TODO: check for cases when other types of objects are equal, such as x^2 (Mono) and x^2 (Exponent)
                    return False
            else:  # One of the expressions is a number, while the other is an algebraic expression
                return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __copy__(self):
        return Exponent(base=self._base, power=self._power, coefficient=self._coefficient)

    def __str__(self):
        if self._coefficient == 0:
            return "0"
        if self._power == 0:
            return self._coefficient.__str__()
        if self._coefficient == 1:
            coefficient_str = ""
        elif self._coefficient == -1:
            coefficient_str = "-"
        else:
            coefficient_str = f"{self._coefficient.__str__()}*"
        base_string, power_string = apply_parenthesis(
            self._base.__str__()), apply_parenthesis(self._power.__str__())
        return f"{coefficient_str}{base_string}^{power_string}"
