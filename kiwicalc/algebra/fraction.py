# Does everything that inherit from IExpression will be accepted here ?
class Fraction(IExpression):
    __slots__ = ['_numerator', '_denominator']

    def __init__(self, numerator: "Union[IExpression,float,int]",
                 denominator: "Optional[Union[IExpression,float,int]]" = None, gen_copies=True):
        # Handle the numerator
        if isinstance(numerator, (float, int)):
            self._numerator = Mono(numerator)
        elif isinstance(numerator, IExpression):
            self._numerator = numerator.__copy__() if gen_copies else numerator
        else:
            raise TypeError(f"Unexpected type {type(numerator)} in Fraction.__init__."
                            f"Modify the type of the numerator parameter to a valid one.")

        if denominator is None:
            self._denominator = Mono(1)
            return
        # Handle the denominator
        # Create a Mono object instead of ints and floats
        if isinstance(denominator, (float, int)):
            self._denominator = Mono(denominator)
        elif isinstance(denominator, IExpression):
            self._denominator = denominator.__copy__() if gen_copies else denominator
        else:
            raise TypeError(f"Unexpected type {type(denominator)} in Fraction.__init__. Modify the type of the"
                            f"denominator parameter to a valid one")

    @property
    def numerator(self):
        return self._numerator

    @property
    def denominator(self):
        return self._denominator

    @property
    def variables(self):
        return self._numerator.variables_dict.union(self._denominator.variables_dict)

    def assign(self, **kwargs):
        self._numerator.assign(**kwargs)
        self._denominator.assign(**kwargs)

    def derivative(self):
        return (self._numerator.derivative() * self._denominator - self._numerator * self._denominator.derivative) \
            / self._denominator ** 2

    def integral(self):  # TODO: got no idea how
        pass

    def simplify(self):  # TODO: try to divide the numerator and denominator and check whether it can be done
        pass  # TODO: how to check whether the division is successful..

    def try_evaluate(self) -> Optional[Union[int, float]]:
        """ try to evaluate the expression into a float or int value, if not successful, return None"""
        numerator_evaluation = self._numerator.try_evaluate()
        denominator_evaluation = self._denominator.try_evaluate()
        if denominator_evaluation is None:
            if self._numerator == 0:
                return 0

        if denominator_evaluation == 0:
            raise ZeroDivisionError(f"Denominator of fraction {self.__str__()} was evaluated into 0. Cannot divide "
                                    f"by 0.")
        if None not in (numerator_evaluation, denominator_evaluation):
            return numerator_evaluation / denominator_evaluation
        division_result = (self._numerator / self._denominator)
        # bug fix: preventing a recursive endless loop .....
        if isinstance(division_result, Fraction):
            return None
        division_evaluation = division_result.try_evaluate()
        if division_evaluation is not None:
            return division_evaluation
        return None

    def to_dict(self):
        return {
            "type": "Fraction",
            "numerator": self._numerator.to_dict(),
            "denominator": self._denominator.to_dict() if self._denominator is not None else None
        }

    @staticmethod
    def from_dict(given_dict: dict):
        numerator_obj = create_from_dict(given_dict['numerator'])
        denominator_obj = create_from_dict(given_dict['denominator'])
        return Fraction(numerator=numerator_obj, denominator=denominator_obj)

    def __iadd__(self, other: Union[IExpression, int, float]):
        #  TODO: add simplifications for expressions that can be evaluated like Log(5) for instance
        if isinstance(other, (int, float)):  # If we're adding a number
            my_evaluation = self.try_evaluate()
            if my_evaluation is not None:  # If the fraction can be evaluated into number
                return Mono(coefficient=my_evaluation + other)
            else:
                return ExpressionSum((self, Mono(coefficient=other)))
        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            my_evaluation = self.try_evaluate()
            # Both of the expressions can be evaluated into numbers
            if None not in (other_evaluation, my_evaluation):
                return Mono(coefficient=my_evaluation + other_evaluation)
            if isinstance(other, ExpressionSum):
                copy_of_other = other.__copy__()
                copy_of_other += self
                return copy_of_other
            # TODO: try to improve this section with common denominator?
            elif isinstance(other, Fraction):
                # If the denominators are equal, just add the numerator.
                if self._denominator == other._denominator:
                    self._numerator += other._numerator
                else:
                    return ExpressionSum((self, other))

            # Other types of IExpression that don't need special-case handling.
            else:
                return ExpressionSum((self, other))

        else:
            raise TypeError(
                f"Invalid type '{type(other)}' for addition with fractions")

    def __isub__(self, other: Union[IExpression, int, float]):
        return self.__iadd__(-other)

    def __neg__(self):
        copy_of_self = self.__copy__()
        copy_of_self._numerator *= -1
        return copy_of_self

    def __imul__(self, other: Union[IExpression, int, float]):
        if isinstance(other, Fraction):
            self._numerator *= other._numerator
            self._denominator *= other._denominator
            return self
        if self._denominator == other:
            self._denominator = Mono(1)
            my_evaluation = self.try_evaluate()
            if my_evaluation is not None:
                return Mono(my_evaluation)
            return self._numerator
        self._numerator *= other
        return self

    def __mul__(self, other: Union[IExpression, int, float]):
        return self.__copy__().__imul__(other)

    def __itruediv__(self, other: Union[IExpression, int, float]):
        if isinstance(other, Fraction):
            self._numerator *= other._denominator
            self._denominator *= other._numerator
        else:
            self._denominator *= other
        return self

    def __rmul__(self, other: Union[IExpression, int, float]):
        return self.__copy__().__imul__(other)

    def __ipow__(self, other: Union[IExpression, int, float]):
        self._numerator **= other
        self._denominator **= other
        self.simplify()
        return self

    def __rpow__(self, other):
        return Exponent(self, other)

    def __copy__(self):
        return Fraction(self._numerator, self._denominator)

    def __eq__(self, other: Union[IExpression, int, float]) -> Optional[bool]:
        if other is None:
            return False
        numerator_evaluation = self._numerator.try_evaluate()
        if numerator_evaluation == 0:
            return other == 0
        denominator_evaluation = self._denominator.try_evaluate()
        if denominator_evaluation == 0:  # making sure a ZeroDivisionError won't occur somehow
            raise ValueError(f"Denominator of a fraction cannot be 0.")
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):  # if the parameter is a number
            if my_evaluation is not None:
                return my_evaluation == other
            return None  # Algebraic expression isn't equal to a free number

        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            # Both expressions can be evaluated into numbers
            if None not in (my_evaluation, other_evaluation):
                return my_evaluation == other_evaluation
            if isinstance(other, Fraction):
                if self._numerator == other._numerator and self._denominator == other._denominator:
                    return True
                # Won't reach here if any of them is zero, so no reason to worry about ZeroDivisionError
                # Check for cases such as 0.2x / y and x / 5y , which are the same.
                numerator_ratio = self._numerator / other._numerator
                denominator_ratio = self._denominator / other._denominator
                return numerator_ratio == denominator_ratio
            else:
                pass  # Implement it ..

    def __ne__(self, other: Union[IExpression, int, float]):
        return not self.__eq__(other)

    def python_syntax(self) -> str:
        return f"({self._numerator.python_syntax()})/({self._denominator.python_syntax()})"

    def __str__(self):
        return f"({self._numerator.__str__()})/({self._denominator.__str__()})"
