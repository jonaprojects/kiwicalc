class Root(IExpression, IPlottable, IScatterable):
    __slots__ = ['_coefficient', '_inside', '_root']

    def __init__(self, inside: Union[IExpression, float, int], root_by: Union[IExpression, float, int] = 2, coefficient: Union[int, float, IExpression] = Mono(1)):
        self._coefficient = process_object(coefficient,
                                           class_name="Root", method_name="__init__", param_name="coefficient")
        self._inside = process_object(inside,
                                      class_name="Root", method_name="__init__", param_name="inside")
        self._root = process_object(root_by,
                                    class_name="Root", method_name="__init__", param_name="root_by")

    @property
    def coefficient(self):
        return self._coefficient

    @property
    def inside(self):
        return self._inside

    @property
    def root(self):
        return self._root

    def assign(self, **kwargs):
        self._coefficient.assign(**kwargs)
        self._inside.assign(**kwargs)
        self._root.assign(**kwargs)

    def try_evaluate(self) -> Optional[Union[complex, float, ValueError]]:
        coefficient_evaluation = self._coefficient.try_evaluate()
        if coefficient_evaluation == 0:
            return 0
        inside_evaluation = self._inside.try_evaluate()
        root_evaluation = self._root.try_evaluate()
        if None not in (coefficient_evaluation, inside_evaluation, root_evaluation):
            if root_evaluation == 0:
                return ValueError("Cannot compute root by 0")
            return coefficient_evaluation * inside_evaluation ** (1 / root_evaluation)
        return None

    def simplify(self) -> None:
        self._coefficient.simplify()
        self._root.simplify()
        self._inside.simplify()

    @property
    def variables(self):
        variables = self._coefficient.variables
        variables.update(self._inside.variables)
        variables.update(self._root.variables)
        return variables

    def to_dict(self):
        return {
            "type": "Root",
            "coefficient": self._coefficient.to_dict(),
            "inside": self._inside.to_dict(),
            "root_by": self._root.to_dict()
        }

    @staticmethod
    def from_dict(given_dict: dict):
        coefficient_obj = create_from_dict(given_dict['coefficient'])
        inside_obj = create_from_dict(given_dict['inside'])
        root_obj = create_from_dict(given_dict['root_by'])
        return Root(coefficient=coefficient_obj, inside=inside_obj, root_by=root_obj)

    @staticmethod
    def dependant_roots(first_root: "Root", second_root: "Root") -> Optional[Tuple[IExpression, str]]:
        if first_root._root != second_root._root:
            return None
        # If the second root is the common denominator
        result = first_root._inside.__truediv__(second_root._inside)
        if isinstance(result, Fraction) or result is None:
            return None
        if isinstance(result, tuple):
            result, remainder = result
            if remainder == 0:
                return result, "first"  # the first is bigger
            return None
        result = second_root._inside.__truediv__(first_root._inside)
        if isinstance(result, Fraction) or result is None:
            return None
        if isinstance(result, tuple):
            result, remainder = result
            if remainder == 0:
                return result, "second"  # the first is bigger
            return None
        return result, "second"

    def __iadd__(self, other: Union[IExpression, float, int, str]):
        if other == 0:
            return self
        if isinstance(other, IExpression):  # If the expression is IExpression
            if isinstance(other, Root):  # if it's a another root
                division_result: Optional[IExpression] = Root.dependant_roots(
                    self, other)
                if division_result is not None:
                    root_evaluation = self._root.try_evaluate()
                    if division_result[1] == "first":
                        other_copy = other.__copy__()
                        # later maybe make it common denominator too
                        other_copy._coefficient = Mono(1)
                        division_result = division_result[0]
                        if root_evaluation is not None:
                            return other_copy * (division_result ** (
                                1 / root_evaluation) + other_copy._coefficient * self._coefficient)
                    else:  # The second is bigger
                        division_result = division_result[0]
                        if root_evaluation is not None:
                            self_copy = self.__copy__()
                            self_copy._coefficient = Mono(1)
                            return self * (division_result ** (
                                1 / root_evaluation) + self._coefficient * other._coefficient)
        if isinstance(other, (int, float)):
            other = Mono(other)
        return ExpressionSum((self, other))  # If it's not another root

    def __isub__(self, other):
        if other == 0:
            return self
        # TODO: Should I create here too a separate implementation ?
        return self.__iadd__(-other)

    def multiply_by_root(self, other: "Root"):
        other_evaluation = other.try_evaluate()
        # If the root can be evaluated into a number, such as Root of 2:
        if other_evaluation is not None:
            self._coefficient *= other_evaluation
            return self
        if self._root == other._root:
            self._inside *= other._inside
            return self
        else:
            return ExpressionMul((self, other))

    def __imul__(self, other: Union[IExpression, float, int, str]):
        if isinstance(other, (int, float)):
            self._coefficient *= other
            self.simplify()
            return self
        if isinstance(other, str):  # TODO: implement a kind of string-processing method
            pass

        if isinstance(other, IExpression):
            if isinstance(other, Root):
                return self.multiply_by_root(other)
            else:
                self._coefficient *= other
                return self

        return TypeError(f"Invalid type {type(other)} for multiplying roots.")

    def __mul__(self, other: Union[int, float, IExpression]):
        return self.__copy__().__imul__(other)

    def __rmul__(self, other: Union[int, float, IExpression]):
        return self.__copy__().__mul__(other)

    def __neg__(self):
        copy_of_self = self.__copy__()
        copy_of_self._coefficient *= -1
        return copy_of_self

    def __ipow__(self, other: Union[int, float, IExpression]):
        if other == 1:
            return self
        if other == 0:
            pass  # TODO: Return Mono(1) or change the current object somehow?
        root_division = self._root / other  # A good start. Will be developed later
        if isinstance(root_division, IExpression):
            evaluated_division = root_division.try_evaluate()
            if evaluated_division is None:
                self._root = root_division
                return self
        elif isinstance(root_division, (int, float)):
            evaluated_division = root_division
        else:
            raise TypeError(
                f"Invalid type '{type(other)} when dividing Root objects.'")
        if 0 < evaluated_division < 1:
            return self._inside ** (1 / evaluated_division)
        elif evaluated_division == 1:
            return self._inside
        self._root = evaluated_division
        return self

    def __pow__(self, power):
        return self.__copy__().__ipow__(power)

    def __itruediv__(self, other: Union[int, float, IExpression]):
        if other == 0:
            return ZeroDivisionError("Cannot divide a Root object by 0")
        if isinstance(other, (int, float)):
            self._coefficient /= other
            return self
        else:  # Other is of type IExpression
            other_evaluation = other.try_evaluate()
            if other_evaluation is not None:  # The other argument can be evaluated into an int or a float
                self._coefficient /= other_evaluation
                return self
            if isinstance(other, Root):  # If we're dividing by another root
                if self._root == other._root:  # the other expression has the same root number
                    if self == other:  # The two expressions are equal
                        return Mono(1)
                elif self._inside == other._inside:
                    # Different roots, but same inside expressions and thus it can be evaluated ..
                    my_root_evaluation = self._root.try_evaluate()
                    other_root_evaluation = other._root.try_evaluate()
                    if my_root_evaluation and other_root_evaluation:
                        # Both roots can be evaluated into numbers, and not 0 ( 0 is false )
                        self._coefficient /= other._coefficient
                        power_difference = (
                            1 / my_root_evaluation) - (1 / other_root_evaluation)
                        self._root = 1 / power_difference
                        return self
            else:
                return Fraction(self, other)

            return Fraction(self, other)

    def __copy__(self):
        return Root(
            inside=self._inside,
            root_by=self._root,
            coefficient=self._coefficient
        )

    def __str__(self):
        if self._coefficient == 0:
            return "0"
        if self._coefficient == 1:
            coefficient = ""
        elif self._coefficient == -1:
            coefficient = '-'
        else:
            coefficient = f"{self._coefficient} * "
        root = f"{self._root}^" if self._root != 2 else ""
        return f"{coefficient}{root}√({self._inside})"

    def __eq__(self, other: Union[IExpression, int, float]):
        """ Compare between a Root object and other expressions"""
        if other is None:
            return False
        if isinstance(other, (int, float)):
            my_evaluation = self.try_evaluate()
            print(my_evaluation)
            return my_evaluation == other
        if isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            my_evaluation = self.try_evaluate()
            if None not in (other_evaluation, my_evaluation):  # Both can be evaluated
                return my_evaluation == other_evaluation
            if (my_evaluation, other_evaluation) == (None, None):  # None can be evaluated
                if isinstance(other, Root):  #
                    if self._coefficient == other._coefficient and self._inside == other._inside and self._root == other._root:
                        return True
                    return False  # TODO: handle cases in which it will be true, considering the coefficient and stuff
        return False  # Can be wrong ?

    def __ne__(self, other: Union[IExpression, int, float]):
        return not self.__eq__(other)

    def derivative(self):

        if self._inside is None:
            return 0
        my_evaluation = self.try_evaluate()
        # If the expression can be evaluated into number, the derivative is 0.
        if my_evaluation is not None:
            if my_evaluation < 0:
                warnings.warn("Root evaluated to negative result. Complex Analysis is yet to be supported "
                              "in this version")
            return 0
        if self._coefficient == 0:
            return 0  # If the coefficient is 0 then it's gg, the whole expression is 0
        coefficient_evaluation = self._coefficient.try_evaluate()
        root_evaluation = self._root.try_evaluate()
        inside_evaluation = self._inside.try_evaluate()
        # everything can be evaluated ..
        if None not in (coefficient_evaluation, root_evaluation, inside_evaluation):
            return 0
        inside_variables = self._inside.variables
        if None not in (coefficient_evaluation, root_evaluation) and len(inside_variables) == 1:
            # if the coefficient and the root can be evaluated into free numbers, and only one variable ...
            new_power = (1 / root_evaluation) - 1
            new_root = 1 / new_power
            # Might not always work ..
            inside_derivative = self._inside.derivative()
            if new_power > 1:
                monomial = Mono(coefficient=coefficient_evaluation, variables_dict={
                                inside_variables: new_power})
                monomial *= inside_derivative
                return monomial

            elif new_power == 0:  # then the inside expression is 1, and it's multiplied by the coefficient
                inside_derivative *= coefficient_evaluation
                return inside_derivative
            else:
                if new_root == 1:
                    return coefficient_evaluation * self._inside
                inside_derivative *= coefficient_evaluation
                if new_root < 0:
                    return Fraction(
                        numerator=inside_derivative,
                        denominator=Root(
                            coefficient=1,
                            root_by=abs(new_root),
                            inside=self._inside.__copy__()
                        )
                    )
                else:
                    return Root(
                        coefficient=inside_derivative,
                        root_by=new_root,
                        inside=self._inside.__copy__()
                    )

        else:  # Handling more complex derivatives
            pass

    def integral(self):
        pass

    def python_syntax(self) -> str:  # Create a separate case for the Log class
        """ Returns a string that can be evaluated using the eval() method to actual objects from the class, if
        imported properly
        """
        if isinstance(self._coefficient, Log):
            coefficient_str = self._coefficient.python_syntax()
        else:
            coefficient_str = self._coefficient.__str__()

        if isinstance(self._inside, Log):
            inside_str = self._inside.python_syntax()
        else:
            inside_str = self._inside.__str__()

        if isinstance(self._root, Log):
            root_str = self._root.python_syntax()
        else:
            root_str = self._root.__str__()

        return f"{coefficient_str}*({inside_str}) ** (1/{root_str})"


class Sqrt(Root):  # A class for clarity purposes, so it's clear when someone is using a square root
    def __init__(self, inside: Union[IExpression, float, int], coefficient: Union[int, float, IExpression] = Mono(1)):
        super(Sqrt, self).__init__(inside=inside,
                                   root_by=2, coefficient=coefficient)
