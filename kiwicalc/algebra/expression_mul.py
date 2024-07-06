class ExpressionMul(IExpression, IPlottable, IScatterable):
    __slots__ = ['_coefficient', '_expressions']

    def __init__(self, expressions: Union[Iterable[Union[IExpression, float, int, str]], str], gen_copies=True):
        if isinstance(expressions, str):
            # TODO: classify expressions types and use string analysis methods
            raise NotImplementedError
        else:
            self._expressions = list()
            for expression in expressions:
                if isinstance(expression, (float, int)):
                    self._expressions.append(Mono(expression))
                elif isinstance(expression, IExpression):
                    if gen_copies:
                        self._expressions.append(
                            expression.__copy__())  # TODO: copy method could possibly lead to infinite recursion?
                    else:
                        self._expressions.append(expression)
                elif isinstance(expression, str):
                    # TODO: implement a string analysis method
                    raise NotImplementedError
                else:
                    raise TypeError(f"Encountered an invalid type: '{type(expression)}', when creating a new "
                                    f"Expression object.")

    @property
    def expressions(self):
        return self._expressions

    def assign(self, **kwargs):
        for expression in self._expressions:
            expression.assign(**kwargs)

    def python_syntax(self):
        if not self._expressions:
            return self._coefficient.python_syntax()
        accumulator = f"({self._coefficient})*"
        for iexpression in self._expressions:
            accumulator += f"({iexpression.python_syntax()})*"
        return accumulator[:-1]

    def simplify(self):
        if self._coefficient == 0:
            self._expressions = []

    @property
    def variables(self):
        variables = set()
        for expression in self._expressions:
            variables.update(expression.variables)
        return variables

    def try_evaluate(self):
        evaluated_expressions = [expression.try_evaluate()
                                 for expression in self._expressions]
        if all(evaluated_expressions):
            # TODO: check if this actually works!
            return sum(evaluated_expressions)

    def __split_expressions(self, num_of_expressions: int):
        return ExpressionMul(self._expressions[:num_of_expressions // 2]), ExpressionMul(
            self._expressions[num_of_expressions // 2:])

    def derivative(self):
        print(
            f"calculating the derivative of {self}, num of expressions: {len(self._expressions)}")
        # Assuming all the expressions can be derived
        num_of_expressions = len(self._expressions)
        if num_of_expressions == 0:  # No expressions, then no derivative!
            return None
        if num_of_expressions == 1:
            return self._expressions[0].derivative()
        elif num_of_expressions == 2:
            first, second = self._expressions[0], self._expressions[1]
            return first.derivative() * second + second.derivative() * first
        else:  # more than 2 expressions
            expressionMul1, expressionMul2 = self.__split_expressions(
                num_of_expressions)
            first_derivative, second_derivative = expressionMul1.derivative(
            ), expressionMul2.derivative()
            if isinstance(first_derivative, (int, float)):
                first_derivative = Mono(first_derivative)
            if isinstance(second_derivative, (int, float)):
                second_derivative = Mono(second_derivative)
            return first_derivative * expressionMul2 + second_derivative * expressionMul1

    def __copy__(self):
        return ExpressionMul(self._expressions)

    def __neg__(self):
        copy_of_self = self.__copy__()
        copy_of_self._coefficient *= 1
        return copy_of_self

    def __iadd__(self, other):
        return ExpressionSum((self, other))

    def __isub__(self, other):
        return ExpressionSum((self, other.__neg__()))

    def __imul__(self, other):
        self._expressions.append(other)
        return self

    def __itruediv__(self, other):
        return Fraction(self, other)

    def __rtruediv__(self, other):
        return Fraction(other, self)

    def __ipow__(self, power):
        for index, expression in enumerate(self._expressions):
            self._expressions[index] = expression.__pow__(power)
        return self

    def __rpow__(self, other):  # TODO: Implement exponents for that
        return Exponent(other, self)

    def __str__(self) -> str:
        accumulator = f""
        for index, expression in enumerate(self._expressions):
            content = expression.__str__()
            if index > 0 and not content.startswith('-'):
                content = f"*{content}"
            if not content.endswith(')'):
                # Add parenthesis to clarify the order of actions
                content = f'({content})'
            accumulator += content
        return accumulator

    def __eq__(self, other: Union[IExpression, int, float]) -> bool:
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            return my_evaluation is not None and my_evaluation == other
        elif isinstance(other, IExpression):
            # First check equality with the evaluations, if the expressions can be evaluated
            other_evaluation = other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                return my_evaluation == other_evaluation
            if isinstance(other, ExpressionMul):
                pass  # TODO: Use an outside method for this complicated equality checking, like the TrigoExpr
            else:
                if len(self._expressions) == 1 and self._expressions[0] == other:
                    return True
                # Add more checks ?
                return False
        else:
            raise TypeError(
                f"Invalid type {type(other)} for equality checking in Expression class")

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_dict(self):
        pass

    def from_dict(self):
        pass
