class TrigoExprs(ExpressionSum, IPlottable, IScatterable):
    def _transform_expression(self, expression, dtype='poly'):
        if isinstance(expression, (int, float)):
            return Mono(expression)
        elif isinstance(expression, str):
            return create(expression, dtype=dtype)
        elif isinstance(expression, (IExpression)):
            return expression.__copy__()
        else:
            raise TypeError(
                f"Unexpected type {type(expression)} in TrigoExpr.__init__()")

    def __init__(self, expressions: Union[str, Iterable[IExpression]], dtype='poly'):
        if isinstance(expressions, str):
            expressions = TrigoExprs_from_str(expressions, get_list=True)
        if isinstance(expressions, Iterable):
            expressions_list = []
            # print(f"received {[expression.__str__() for expression in expressions]} in init")
            for index, expression in enumerate(expressions):
                try:
                    matching_index = next(index for index, existing in enumerate(expressions_list) if
                                          equal_ignore_order(existing.expressions, expression.expressions))
                    expressions_list[matching_index]._coefficient += expression.coefficient
                except StopIteration:
                    expressions_list.append(expression)
            # print([f"{expr.__str__()}" for expr in expressions_list])
            super(TrigoExprs, self).__init__(
                [self._transform_expression(expression, dtype=dtype) for expression in expressions_list])
        else:
            raise TypeError(
                f"Unexpected  type {type(expressions)} in TrigoExpr.__init__(). Expected an iterable collection or a "
                f"string.")
        # now

    def __add_TrigoExpr(self, other: TrigoExpr):  # TODO: check this methods
        """Add a TrigoExpr expression"""
        try:
            index = next((index for index, expression in enumerate(self._expressions) if
                          expression.expressions == other.expressions))  # Find the first matching expression
            self._expressions[index]._coefficient += other.coefficient
        except StopIteration:
            # No matching expression
            self._expressions.append(other)

    def __sub_TrigoExpr(self, other: TrigoExpr):
        """ Subtract a TrigoExpr expression"""
        try:
            index = next((index for index, expression in enumerate(self._expressions) if
                          expression.expressions == other.expressions))  # Find the first matching expression
            self._expressions[index]._coefficient -= other.coefficient
        except StopIteration:
            # No matching expression
            self._expressions.append(other)

    def __iadd__(self, other: Union[int, float, IExpression]):
        print(f"adding {other} to {self}")
        if other == 0:
            return self
        if isinstance(other, str):
            other = TrigoExprs(other)
        if isinstance(other, IExpression):
            my_evaluation, other_evaluation = self.try_evaluate(), other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                return Mono(my_evaluation + other_evaluation)

            if isinstance(other, TrigoExpr):
                self.__add_TrigoExpr(other)
                return self
            elif isinstance(other, TrigoExprs):
                for other_expression in other._expressions:
                    self.__add_TrigoExpr(other_expression)
                return self
            else:
                return ExpressionSum((self, other))
        else:
            raise TypeError(
                f"Invalid type for adding trigonometric expressions: {type(other)}")

    def __isub__(self, other: Union[int, float, IExpression]):
        if isinstance(other, str):
            other = TrigoExprs(other)
        if isinstance(other, IExpression):
            my_evaluation, other_evaluation = self.try_evaluate(), other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                return Mono(my_evaluation - other_evaluation)
            if isinstance(other, TrigoExpr):
                self.__sub_TrigoExpr(other)
                return self
            elif isinstance(other, TrigoExprs):
                for other_expression in other._expressions:
                    self.__sub_TrigoExpr(other_expression)
                    return self
            else:
                return ExpressionSum((self, other))
        else:
            raise TypeError(
                f"Invalid type for subtracting trigonometric expressions: {type(other)}")

    @property
    def variables(self):
        variables = set()
        for trigo_expression in self._expressions:
            variables.update(trigo_expression.variables)
        return variables

    def flip_signs(self):
        for expression in self._expressions:
            expression *= -1

    def __neg__(self):
        copy_of_self = self.__copy__()
        copy_of_self.flip_signs()
        return copy_of_self

    def __rsub__(self, other):
        if isinstance(other, IExpression):
            return other.__sub__(self)
        elif isinstance(other, (int, float)):
            pass  # TrigoExprs or TrigoExpr
        else:
            raise TypeError(
                f"Invalid type while subtracting a TrigoExprs object: {type(other)} ")

    def __imul__(self, other: Union[int, float, IExpression]):
        if isinstance(other, (int, float, IExpression)):
            if isinstance(other, IExpression):
                other_evaluation = other.try_evaluate()
                value = other_evaluation if other_evaluation is not None else other
            else:
                value = other
            if isinstance(value, TrigoExprs):
                expressions_list = []
                for other_expression in value.expressions:
                    for my_expression in self._expressions:
                        expressions_list.append(
                            my_expression * other_expression)
                return TrigoExprs(expressions_list)
            else:
                for index in range(len(self._expressions)):
                    self._expressions[index] *= value
                return self
        else:
            raise TypeError(
                f"Invalid type '{type(other)}' when multiplying a TrigoExprs object.")

    def __mul__(self, other):
        return self.__copy__().__imul__(other)

    def __rmul__(self, other):  # same as mul
        return self.__mul__(other)

    # TODO: implement it
    def __itruediv__(self, other: Union[int, float, IExpression]):
        my_evaluation = self.try_evaluate()
        if other == 0:
            raise ZeroDivisionError("Cannot divide a TrigoExprs object by 0")
        if isinstance(other, (int, float)):
            if my_evaluation is None:
                for trigo_expression in self._expressions:
                    trigo_expression /= other
                return self
            else:
                return Mono(coefficient=my_evaluation / other)
        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if other_evaluation is not None:
                if my_evaluation is not None:
                    return Mono(coefficient=my_evaluation / other_evaluation)
                else:
                    for trigo_expression in self._expressions:
                        trigo_expression /= other
                    return self
            # the current instance represents a number but 'other' does not.
            elif my_evaluation is not None:
                # Therefore, a case such as 1/x will be created, which can be represented via a Fraction object.
                return Fraction(self, other)
            elif isinstance(other, TrigoExpr):
                if len(self._expressions) == 1:
                    return self._expressions[0] / other
                for [method, inside, power] in other.expressions:
                    for trigo_expression in self._expressions:
                        found = False
                        for [method1, inside1, power1] in trigo_expression.expressions:
                            if (method, inside) == (method1, inside1) and power1 >= power:
                                found = True
                                break

                        if not found:
                            return Fraction(self, other)
                for [method, inside, power] in other.expressions:
                    for trigo_expression in self._expressions:
                        delete_indices = []
                        for index, [method1, inside1, power1] in enumerate(trigo_expression.expressions):
                            if (method, inside) == (method1, inside1) and power1 >= power:
                                trigo_expression.expressions[index][2] -= power
                                # we can cancel results like sin(x)^2
                                if trigo_expression.expressions[index][2] == 0:
                                    delete_indices.append(index)
                        if delete_indices:
                            trigo_expression._expressions = [item for index, item in
                                                             enumerate(trigo_expression.expressions) if
                                                             index not in delete_indices]

                print("done!")
                return self

            elif isinstance(other, TrigoExprs):
                # First of all, check for equality, then return 1
                if all(trigo_expr in other.expressions for trigo_expr in self._expressions) and all(
                        trigo_expr in self.expressions for trigo_expr in other._expressions):
                    return Mono(1)
                # TODO: Further implement it.
                return Fraction(self, other)

            else:
                return Fraction(self, other)
        else:
            raise TypeError(
                f"Invalid type '{type(other)}' for dividing a TrigoExprs object")

    def __pow__(self, power: Union[int, float, IExpression]):  # Check if this works
        if isinstance(power, IExpression):
            power_evaluation = power.try_evaluate()
            if power_evaluation is not None:
                power = power_evaluation
            else:
                return Exponent(self, power)
        if power == 0:
            return TrigoExpr(1)
        elif power == 1:
            return self.__copy__()
        items = len(self._expressions)
        if items == 1:
            return self._expressions[0].__copy__().__ipow__()
        elif items == 2:
            expressions = []
            # Binomial theorem
            for k in range(power + 1):
                comb_result = comb(power, k)
                first_power, second_power = power - k, k
                first = self._expressions[0] ** first_power
                first *= comb_result
                first *= self._expressions[1] ** second_power
                expressions.append(first)
            return TrigoExprs(expressions)
        elif items > 2:
            for i in range(power - 1):
                self.__imul__(self)  # Outstanding move!
            return self
        else:
            raise ValueError(
                f"Cannot raise an EMPTY TrigoExprs object by the power of {power}")

    def assign(self, **kwargs):  # TODO: implement it
        for trigo_expression in self._expressions:
            trigo_expression.assign(**kwargs)
        self.simplify()

    def try_evaluate(self, **kwargs):  # TODO: implement it
        self.simplify()  # Simplify first
        evaluation_sum = 0
        for trigo_expression in self._expressions:
            trigo_evaluation = trigo_expression.try_evaluate()
            if trigo_evaluation is None:
                return None
            evaluation_sum += trigo_evaluation
        return evaluation_sum

    def to_lambda(self):
        return to_lambda(self.python_syntax(), self.variables)

    def plot(self, start: float = -8, stop: float = 8,
             step: float = 0.3, ymin: float = -3, ymax: float = 3, title=None, show_axis=True, show=True,
             fig=None, ax=None, formatText=True, values=None):
        variables = self.variables
        num_of_variables = len(variables)
        if num_of_variables == 1:
            plot_function(self.to_lambda(), start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title,
                          show_axis=show_axis, show=show, fig=fig, ax=ax, formatText=formatText, values=values)
        elif num_of_variables == 2:
            plot_function_3d(given_function=self.to_lambda(),
                             start=start, stop=stop, step=step)
        else:
            raise ValueError(
                f"Cannot plot a trigonometric expression with {num_of_variables} variables.")

    def to_dict(self):
        return {'type': 'TrigoExprs', 'data': [expression.to_dict() for expression in self._expressions]}

    @staticmethod
    def from_dict(given_dict: dict):
        return TrigoExprs([TrigoExpr.from_dict(sub_dict) for sub_dict in given_dict['expressions']])

    def __eq__(self, other):
        result = super(TrigoExprs, self).__eq__(other)
        if result:
            return result
        # TODO: HANDLE TRIGONOMETRIC IDENTITIES

    def __str__(self):
        return "+".join((expression.__str__() for expression in self._expressions)).replace("+-", "-")

    def __copy__(self):
        return TrigoExprs(self._expressions.copy())
