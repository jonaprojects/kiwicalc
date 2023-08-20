class ExpressionSum(IExpression, IPlottable, IScatterable):
    __slots__ = ['_expressions', '_current_index']

    def __init__(self, expressions: Iterable[IExpression] = None, copy=True):
        self._current_index = 0
        if expressions is None:
            self._expressions = []
        else:
            if copy:
                self._expressions = [copy_expression(
                    expression) for expression in expressions]
            else:
                self._expressions = [expression for expression in expressions]
        # Now we need to check for any "ExpressionSum" object to unpack, and for numbers
        expressions_to_add = []
        indices_to_delete = []
        for index, expression in enumerate(self._expressions):
            if isinstance(expression, ExpressionSum):
                # the expressions should be unpacked
                expressions_to_add.extend(expression._expressions)
                indices_to_delete.append(index)
            elif isinstance(expression, (int, float)):
                self._expressions[index] = Mono(expression)
        self._expressions = [expression for index, expression in enumerate(
            self._expressions) if index not in indices_to_delete]
        self._expressions.extend(expressions_to_add)

    @property
    def expressions(self):
        return self._expressions

    def append(self, expression: IExpression):
        self._expressions.append(expression)

    def assign_to_all(self, **kwargs):
        for expression in self._expressions:
            expression.assign(**kwargs)

    def when_all(self, **kwargs):
        return ExpressionSum((expression.when(**kwargs) for expression in self._expressions), copy=False)
        # Prevent unnecessary double copying

    def __add_or_sub(self, other: "Union[IExpression, ExpressionSum]", operation='+'):
        if isinstance(other, (int, float)):
            my_evaluation = self.try_evaluate()
            if my_evaluation is not None:
                if operation == '+':
                    return Mono(my_evaluation + other)
                else:
                    return Mono(my_evaluation - other)
            if operation == '+':
                self._expressions.append(Mono(other))
            else:
                self._expressions.append(Mono(-other))
            self.simplify()
            return self
        elif isinstance(other, IExpression):
            my_evaluation, other_evaluation = self.try_evaluate(), other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                return Mono(my_evaluation + other_evaluation)
            elif my_evaluation is not None:
                if operation == '+':
                    return other.__add__(my_evaluation)
                else:
                    return other.__sub__(my_evaluation)
            elif other_evaluation is not None:
                if operation == '+':
                    self._expressions.append(Mono(other_evaluation))
                else:
                    self._expressions.append(Mono(-other_evaluation))
                self.simplify()
                return self
            else:
                # nothing can be evaluated.
                if operation == '+':
                    self._expressions.append(other.__copy__())
                else:
                    self._expressions.append(other.__neg__())
                self.simplify()
                return self

            if isinstance(other, ExpressionSum):
                if operation == '+':
                    for expression in other._expressions:
                        self._expressions.append(expression)
                else:
                    for expression in other._expressions:
                        self._expressions.append(expression)

                self.simplify()
                return self

        self.simplify()
        return self

    def __iadd__(self, other: "Union[IExpression, ExpressionSum]"):
        return self.__add_or_sub(other, operation='+')

    def __isub__(self, other: "Union[IExpression, int, float, ExpressionSum]"):
        return self.__add_or_sub(other, operation='-')

    def __rsub__(self, other: "Union[IExpression, int, float, ExpressionSum]"):
        return ExpressionSum((other, -self))

    def __neg__(self):
        return ExpressionSum((expression.__neg__() for expression in self._expressions))

    def __imul__(self, other: "Union[IExpression, int, float, ExpressionSum]"):
        if isinstance(other, ExpressionSum):
            final_expressions: List[Optional[IExpression]] = []
            for my_expression in self._expressions:
                for other_expression in other._expressions:
                    final_expressions.append(my_expression * other_expression)
            result = self.to_poly()
            if result is not None:
                return result
            self.simplify()
            return self
        else:
            for index in range(len(self._expressions)):
                self._expressions[index] *= other

            result = self.to_poly()
            if result is not None:
                return result
            self.simplify()
            return self

    def __ipow__(self, power: Union[IExpression, int, float]):
        if isinstance(power, (int, float)):
            length = len(self._expressions)
            if length == 0:  # No expression
                return None
            if power == 0:
                return Mono(1)
            if length == 1:  # Single expression
                self._expressions[0] **= power
                return self
            if length == 2:
                # Binomial
                if 0 < power < 1:  # Root
                    pass
                elif power > 0:
                    pass
                else:  # Negative powers
                    pass
            elif length > 2:  # More than two items
                if 0 < power < 1:  # Root
                    pass
                elif power > 0:
                    copy_of_self = self.__copy__()
                    for index in range(power - 1):
                        self.__imul__(copy_of_self)
                    return self
                else:  # Negative powers
                    return Fraction(1, self.__ipow__(abs(power)))
        # Return an exponent if the expression can't be evaluated into number
        elif isinstance(power, IExpression):
            other_evaluation = power.try_evaluate()
            if other_evaluation is None:
                # TODO: return here an exponent object
                pass
            else:
                return self.__ipow__(other_evaluation)
        else:
            raise TypeError(
                f"Invalid type '{type(power)}' for raising an 'ExpressionSum' object by a power")

    def __pow__(self, power: Union[IExpression, int, float]):
        return self.__copy__().__ipow__(power)

    def __itruediv__(self, other: Union[IExpression, int, float]) -> "Union[ExpressionSum,IExpression]":
        if other == 0:
            raise ValueError("Cannot divide an ExpressionSum object by 0.")
        if isinstance(other, (int, float)):
            for my_expression in self._expressions:
                my_expression /= other
            return self

        other_evaluation = other.try_evaluate()
        if other_evaluation is not None:
            if other == 0:
                raise ValueError("Cannot divide an ExpressionSum object by 0.")
            for my_expression in self._expressions:
                my_expression /= other_evaluation
            return self

        if isinstance(other, (ExpressionSum, Poly, TrigoExprs)):
            # put the expressions in a Fraction object when encountering collections
            return Fraction(self, other)
        if not isinstance(other, (int, float, IExpression)):
            raise TypeError(
                f"Invalid type {type(other)} for dividing with 'ExpressionSum' class.")
        for my_expression in self._expressions:
            my_expression /= other
        # try to simplify to a polynomial (there's heavy support for polynomials)
        result = self.to_poly()
        if result is not None:
            return result
        self.simplify()
        return self

    def assign(self, **kwargs) -> None:
        for expression in self._expressions:
            expression.assign(**kwargs)

    def is_poly(self):
        return all(isinstance(expression, (Mono, Poly)) for expression in self._expressions)

    def to_poly(self) -> "Optional[Poly]":
        """Tries to convert the ExpressionSum object to a Poly object (to a polynomial).
        If not successful, None will be returned.
        """
        if not self.is_poly():
            return None
        my_poly = Poly(0)
        for expression in self._expressions:
            my_poly += expression
        return my_poly

    def simplify(self):
        for expression in self._expressions:
            expression.simplify()
        evaluation_sum: float = 0
        delete_indices = []
        for index, expression in enumerate(self._expressions):
            expression_evaluation = expression.try_evaluate()
            if expression_evaluation is not None:
                evaluation_sum += expression_evaluation
                delete_indices.append(index)
        self._expressions = [expression for index, expression in enumerate(self._expressions) if
                             index not in delete_indices]
        # if evaluation sum is not 0, add it. ( Because there's no point in adding trailing zeroes)
        if evaluation_sum:
            self._expressions.append(Mono(evaluation_sum))

    def try_evaluate(self):
        """ Try to evaluate the expressions into float or an int """
        evaluation_sum = 0
        for expression in self._expressions:
            expression_evaluation: Optional[Union[int,
                                                  float]] = expression.try_evaluate()
            if expression_evaluation is None:
                return None
            evaluation_sum += expression_evaluation
        return evaluation_sum

    @property
    def variables(self):
        variables = set()
        for expression in self._expressions:
            variables.update(variables.union(expression.variables))
        return variables

    def derivative(self):
        warnings.warn(
            "This feature is still experimental, and might not work.")
        if any(not hasattr(expression, 'derivative') for expression in self._expressions):
            raise AttributeError("Not all expressions support derivatives")
        return ExpressionSum([expression.derivative() for expression in self._expressions], copy=False)
        # Prevent unnecessary copies by setting copy to False

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self):
        if self._current_index < len(self._expressions):
            value = self._expressions[self._current_index]
            self._current_index += 1
            return value
        raise StopIteration

    def __getitem__(self, item):
        return self._expressions.__getitem__(item)

    def __len__(self):
        return len(self._expressions)

    def __copy__(self):
        return ExpressionSum(
            (expression.__copy__() for expression in self._expressions))  # Generator for memory saving..

    def __str__(self):
        accumulator = ""
        for expression in self._expressions:
            expression_string: str = expression.__str__()
            if not expression_string.startswith('-'):
                accumulator += "+"
            accumulator += expression_string
        if accumulator[0] == '+':
            return accumulator[1:]
        return accumulator

    def python_syntax(self) -> str:
        accumulator = ""
        for expression in self._expressions:
            expression_string: str = expression.python_syntax()
            if not expression_string.startswith('-'):
                accumulator += "+"
            accumulator += expression_string
        if accumulator[0] == '+':
            return accumulator[1:]
        return accumulator

    def to_dict(self):
        return {
            "type": "ExpressionSum",
            "expressions": [expression.to_dict() for expression in self._expressions]
        }

    def from_dict(self):
        pass

    # TODO: improve this method
    def __eq__(self, other: Union[IExpression, int, float]):
        """Tries to figure out whether the expressions are equal. May not apply to special cases such as trigonometric
        identities"""
        if isinstance(other, (int, float)):
            if len(self._expressions) == 1:
                return self._expressions[0] == other
        elif isinstance(other, IExpression):
            if isinstance(other, ExpressionSum):
                if len(self._expressions) != len(other._expressions):
                    return False
                for my_expression in self._expressions:
                    my_count = self._expressions.count(my_expression)
                    other_count = other._expressions.count(my_expression)
                    if my_count != other_count:
                        return False
                return True
            else:  # Equating between ExpressionSum to a single expression
                if len(self._expressions) == 1:
                    return self._expressions[0] == other
                else:
                    other_evaluation = other.try_evaluate()
                    if other_evaluation is None:
                        return False
                    my_evaluation = self.try_evaluate()
                    if my_evaluation is None:
                        return False
                    # If reached here, both expressions can be evaluated
                    return my_evaluation == other_evaluation

    def __ne__(self, other: Union[IExpression, int, float]):
        return not self.__eq__(other)
