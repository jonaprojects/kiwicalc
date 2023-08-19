class TrigoExpr(IExpression, IPlottable, IScatterable):
    """ This class represents a single trigonometric expression, such as 3sin(2x)cos(x) for example. """
    __slots__ = ['_coefficient', '_expressions']

    def __init__(self, coefficient, dtype='poly',
                 expressions: "Iterable[Iterable[Union[int,float,Mono,Poly,TrigoMethods]]]" = None):
        self._coefficient = None
        self._expressions: list = []
        # TODO: delete copy constructor and modify coefficient
        if isinstance(coefficient, TrigoExpr):
            self._coefficient = coefficient._coefficient
            self._expressions = coefficient._expressions.copy()
        if isinstance(coefficient, str):
            self._coefficient, self._expressions = TrigoExpr_from_str(
                coefficient, get_tuple=True, dtype=dtype)
        else:
            # First handle the _coefficient parameter
            if isinstance(coefficient, (int, float)):
                self._coefficient = Mono(coefficient)
            elif isinstance(coefficient, Mono):
                self._coefficient = coefficient.__copy__()
            elif isinstance(coefficient, Poly):
                self._coefficient = coefficient.__copy__()

            # Now handle the expressions
            if expressions is None:
                self._expressions = None
            else:
                self._expressions = [list(expression)
                                     for expression in expressions]

    @property
    def coefficient(self):
        return self._coefficient

    @property
    def expressions(self):
        return self._expressions

    def simplify(self):  # TODO: experimental, improve it !
        if self._coefficient == 0:
            self._expressions = [[None, Mono(0), 1]]
        for index, (method, inside, power) in enumerate(self._expressions):
            if power == 0:  # Then we can remove this item since multiplying by 1 doesn't change the expression
                self._expressions.pop(index)

    @property
    def variables(self):
        variables = self._coefficient.variables
        for inner_list in self._expressions:
            variables.update(inner_list[1].variables)
        return variables

    def __add_or_sub(self, other, operation: str = '+'):
        if other == 0:
            return self
        if isinstance(other, (int, float, str)):
            if isinstance(other, (int, float)):
                my_evaluation = self.try_evaluate()
                if my_evaluation is None:
                    if operation == '+':
                        return TrigoExprs((self, TrigoExpr(other)))
                    return TrigoExprs((self, -TrigoExpr(other)))
                else:
                    return Mono(my_evaluation + other)
        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            my_evaluation = self.try_evaluate()
            if None not in (other_evaluation, my_evaluation):
                if operation == '+':
                    return Mono(other_evaluation + my_evaluation)
                return Mono(other_evaluation - my_evaluation)

            if isinstance(other, TrigoExpr):
                if self._expressions == other._expressions:
                    if operation == '+':
                        self._coefficient += other._coefficient
                    else:
                        self._coefficient -= other._coefficient
                    if self._coefficient == 0:
                        self._expressions = [[None, Mono(0), 1]]
                    return self
                else:
                    if operation == '+':
                        return TrigoExprs((self, other))
                    return TrigoExprs((self, -other))

            elif isinstance(other, TrigoExprs):
                if operation == '+':
                    return other.__add__(self)
                return other.__sub__(self)
            else:
                if operation == '+':
                    return ExpressionSum((self, other))
                return ExpressionSum((self, -other))
        else:
            raise TypeError(f"Invalid type {type(other)} while adding or subtracting trigonometric expressions."
                            f"Expected types: TrigoExpr, TrigoExprs, int, float, str, Poly, Mono")

    # TODO: update this
    def __iadd__(self, other: Union[IExpression, int, float, str]):
        return self.__add_or_sub(other, operation='+')

    def __isub__(self, other):
        return self.__add_or_sub(other, operation='-')

    @staticmethod  # Because this method is only relevant to the TrigoExpr class, it is static
    def __find_similar_expressions(expressions: List[list], given_method, given_expression: IExpression) -> Iterator:
        """
        Returns a generator expression that yields the expressions that can be multiplied or divided with the
        given expression, i.e, expressions with the same coefficient and inside expression.
        """
        return (index for index, expression in enumerate(expressions) if
                expression[0] is given_method and expression[1] == given_expression)

    @staticmethod
    def __find_exact_expressions(expressions: List[list], given_method, given_expression, given_power):
        return (index for index, expression in enumerate(expressions) if
                expression[0] is given_method and expression[1] == given_expression and expression[2] == given_power)

    def divide_by_trigo(self, other: "TrigoExpr"):
        if other == 0:
            return ZeroDivisionError("Cannot divide by a TrigoExpr object that evaluates to 0")
        if len(self._expressions) == 0 and len(other._expressions) == 0:
            return True

    @staticmethod
    def __divide_identities(self, other):
        for index, [method, inside, power] in enumerate(self._expressions):
            if not other._expressions:
                return self
            if method == TrigoMethods.SIN:
                try:
                    matching_index = next(
                        self.__find_similar_expressions(other._expressions, TrigoMethods.COS, inside))
                    my_power, other_power = self._expressions[index][2], other._expressions[matching_index][2]
                    if my_power == other_power:
                        self._expressions[index][0] = TrigoMethods.TAN
                        del other._expressions[matching_index]
                    elif my_power > other_power:
                        self._expressions[index][2] -= other._expressions[matching_index][2]
                        self._expressions.append([TrigoMethods.TAN, other._expressions[matching_index][1].__copy__(),
                                                  copy_expression(other._expressions[matching_index][2])])
                        del other._expressions[matching_index]
                    else:
                        self._expressions[index][0] = TrigoMethods.TAN
                        other._expressions[matching_index][2] -= my_power

                except StopIteration:
                    pass
                    # Didn't find a match then do nothing..

            if power == 1 and method == TrigoMethods.SIN:
                try:  # If found sin
                    matching_index = next(
                        self.__find_exact_expressions(other._expressions, TrigoMethods.SIN, inside / 2, 1))
                    self._coefficient *= 2
                    del self._expressions[index]
                    del other._expressions[matching_index]
                except StopIteration:  # If not, search for cos
                    try:  # if found cos
                        matching_index = next(
                            self.__find_exact_expressions(other._expressions, TrigoMethods.COS, inside / 2, 1))
                        del other._expressions[matching_index]
                        self._coefficient *= 2
                        self._expressions[index] = [
                            TrigoMethods.SIN, inside / 2, 1]
                        continue
                    except:  # if didn't find cos
                        continue

                try:  # if found sin and found cos too
                    matching_index = next(
                        self.__find_exact_expressions(other._expressions, TrigoMethods.COS, inside / 2, 1))
                    del other._expressions[matching_index]
                except:  # if didn't find cos
                    self._expressions.append([TrigoMethods.COS, inside / 2, 1])
        return other

    def __mul_trigo(self, other: "TrigoExpr") -> None:
        self._coefficient *= other._coefficient
        for index, other_expression in enumerate(other._expressions):
            try:
                # Try to find a similar expression that can be multiplied with the other expression
                matching_index = next(
                    self.__find_similar_expressions(self._expressions, other_expression[0], other_expression[1]))
                # Add up the powers
                self._expressions[matching_index][2] += other._expressions[index][2]

            except StopIteration:  # If not found, add the expression to the end of the list
                self._expressions.append(other_expression.copy())

    def __imul__(self, other: Union[IExpression, int, float, str]):
        if isinstance(other, (int, float)):
            if other == 0:
                self._coefficient, self._expressions = Poly(0), [
                    [None, Mono(0), 1]]
            else:
                self._coefficient *= other
            self.simplify()
            return self
        elif isinstance(other, str):
            other = TrigoExprs(other)
        if isinstance(other, (Poly, Mono)):
            self._coefficient *= other
            self.simplify()
            return self
        elif isinstance(other, TrigoExpr):
            if other._coefficient == 0:
                self._coefficient, self._expressions = Poly(0), [
                    [None, Mono(0), 1]]
            else:
                self.__mul_trigo(other)
            self.simplify()
            return self

        elif isinstance(other, TrigoExprs):
            if len(other.expressions) == 1:
                self.__mul_trigo(other.expressions[0])
            else:
                return other.__mul__(self)
        elif isinstance(other, IExpression):
            return ExpressionMul((self, other))
        else:
            raise TypeError(
                f"Encountered Invalid type {type(other)} when multiplying trigonometric expressions.")

    def __mul__(self, other):
        return self.__copy__().__imul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __itruediv__(self, other: "Union[int,float,IExpression]"):
        if isinstance(other, (int, float)):
            self._coefficient /= other
            return self
        elif isinstance(other, TrigoExpr):
            if other._coefficient == 0:
                raise ZeroDivisionError(
                    f"Tried to divide '{self.__str__()} by {other.__str__()}'")
            if self == other:  # Check if they are equal
                self._expressions = []
                self._coefficient = Mono(1)
                return self
            self._coefficient /= other._coefficient
            denominator = []
            if len(self._expressions) == 1:
                other_length = len(other._expressions)
                if other_length == 1:
                    if self._expressions[0][1] == other._expressions[0][1]:
                        if self._expressions[0][0] == TrigoMethods.SIN and other._expressions[0][0] == TrigoMethods.COS:
                            my_power, other_power = self._expressions[0][2], other._expressions[0][2]
                            if my_power == other_power:
                                self._expressions[0][0] = TrigoMethods.TAN
                                return self
                            elif my_power > other_power:
                                self._expressions[0][2] -= other._expressions[0][2]
                                self._expressions.append([TrigoMethods.TAN, other._expressions[0][1].__copy__(),
                                                          copy_expression(other._expressions[0][2])])
                                return self
                            else:
                                self._expressions[0][0] = TrigoMethods.TAN
                                other_copy = other.__copy__()
                                other_copy._expressions[0][0] -= my_power
                                return Fraction(self.__copy__(), other_copy, gen_copies=False)

                        elif self._expressions[0][0] == TrigoMethods.COS and other._expressions[0][
                                0] == TrigoMethods.SIN:
                            my_power, other_power = self._expressions[0][2], other._expressions[0][2]
                            if my_power == other_power:
                                self._expressions[0][0] = TrigoMethods.COT
                                return self
                            elif my_power > other_power:
                                self._expressions[0][2] -= other._expressions[0][2]
                                self._expressions.append([TrigoMethods.COT, other._expressions[0][1].__copy__(),
                                                          copy_expression(other._expressions[0][2])])
                                return self
                            else:
                                self._expressions[0][0] = TrigoMethods.COT
                                other_copy = other.__copy__()
                                other_copy._expressions[0][0] -= my_power
                                return Fraction(self.__copy__(), other_copy, gen_copies=False)

            for index, other_expression in enumerate(other._expressions):
                try:
                    matching_index = next(
                        self.__find_similar_expressions(self._expressions, other_expression[0], other_expression[1]))
                    # Add up the powers
                    self._expressions[matching_index][2] -= other_expression[2]

                except StopIteration:
                    denominator.append(other_expression.copy())
            self.simplify()
            if not denominator:
                return self
            else:
                other = TrigoExpr(expressions=denominator, coefficient=1)
                self.__divide_identities(self, other)
                self.__divide_identities(other, self)
                print(f"self is {self} and other is {other}")

            if not other._expressions:
                return self
            return Fraction(self, other)
        elif isinstance(other, TrigoExprs):  # TODO: implement it
            pass
        else:
            return Fraction(self, other)

    def __neg__(self) -> "TrigoExpr":
        copy_of_self = self.__copy__()
        copy_of_self._coefficient *= -1
        return copy_of_self

    def flip_sign(self):
        """flips the sign of the expression - from positive to negative, or from negative to positive"""
        self._coefficient *= -1

    def __ipow__(self, power):  # TODO: check if works
        self._coefficient **= power
        for index, [method, inside, degree] in enumerate(self._expressions):
            self._expressions[index] = [method, inside, degree * power]
        self.simplify()
        return self

    def assign(self, **kwargs):
        self._coefficient.assign(**kwargs)
        for index, [method, inside, degree] in enumerate(self._expressions):
            inside.assign(**kwargs)
            if new_inside := inside.try_evaluate():  # if the inside expression can be evaluated, then simplify
                new_inside = method.value[0](new_inside) ** degree
                self._expressions[index][0] = None
                self._expressions[index][1] = Poly(new_inside)
                self._expressions[index][2] = 1

    def try_evaluate(self) -> Optional[float]:
        evaluated_coefficient = self._coefficient.try_evaluate()
        if not self._expressions:
            return evaluated_coefficient
        if evaluated_coefficient is None or any(
                None in (inside, degree) for [method, inside, degree] in self._expressions):
            return False
        my_sum = 0
        for method, inside, degree in self._expressions:
            if isinstance(inside, IExpression):
                inside = inside.try_evaluate()
                if inside is None:
                    return None
            if method is None:
                my_sum += inside
            else:
                my_sum += method.value[0](inside ** degree)
        return my_sum

    def derivative(self):
        length = len(self._expressions)
        if length == 0:
            return 0
        else:
            coefficient_eval = self._coefficient.try_evaluate()
            if coefficient_eval is not None:
                if length == 1:
                    temp = self._expressions[0]
                    if temp[2] == 0:  # for instance, 3 * sin(x)^0 just 3
                        return self._coefficient.__copy__()
                    elif temp[2] == 1:  # TODO: later optimize some of these
                        if coefficient_eval is not None:  # meaning the coefficient is a free number
                            if temp[0] == TrigoMethods.SIN:
                                return self._coefficient * temp[1].derivative() * Cos(temp[1])
                            elif temp[0] == TrigoMethods.COS:
                                return -self._coefficient * temp[1].derivative() * Sin(temp[1])
                            elif temp == TrigoMethods.TAN:
                                return self._coefficient * temp[1].derivative() * Sec(temp[1]) ** 2
                            elif temp[0] == TrigoMethods.COT:
                                return -self._coefficient * temp[1].derivative() * Csc(temp[1]) ** 2
                            elif temp[0] == TrigoMethods.SEC:
                                return self._coefficient * temp[1].derivative() * Sec(temp[1]) * Tan(temp[1])
                            elif temp[0] == TrigoMethods.CSC:
                                return -self._coefficient * temp[1].derivative() * Csc(temp[1]) * Cot(temp[1])
                            elif temp[0] == TrigoMethods.ASIN:
                                return Fraction(self._coefficient, Root(1 - temp[1] ** 2))
                            elif temp[0] == TrigoMethods.ACOS:
                                return Fraction(-self._coefficient, Root(1 - temp[1] ** 2))
                            elif temp[0] == TrigoMethods.ATAN:
                                return Fraction(self._coefficient, temp[1] ** 2 + 1)
                            elif temp[0] == TrigoMethods.ACOT:
                                return Fraction(-self._coefficient, temp[1] ** 2 + 1)
                            elif temp[0] == TrigoMethods.ASEC:
                                pass  # TODO: implement this!!!
                            elif temp[0] == TrigoMethods.ACSC:
                                pass  # TODO: implement this!!!
                            else:
                                raise ValueError(
                                    f"Unrecognized trigonometric method has been used: {temp[0]}")
                        # the coefficient is an algebraic expression, so the whole expression can be (3x+2)sin(x) for instance
                        else:
                            pass

                elif length == 2:  # TODO: Later modify this to a smarter code
                    if self._expressions[0][0] == self._expressions[1][0] and self._expressions[0][1] == \
                            self._expressions[1][
                                1]:
                        power = self._expressions[0][2] + \
                            self._expressions[1][2]
                        if self._expressions[0][
                                0] == TrigoMethods.SIN:  # We're dealing with _coefficient* Sin(f(x))^power
                            return TrigoExpr(coefficient=self._coefficient * power,
                                             expressions=[(TrigoMethods.SIN, self._expressions[0][1], power - 1),
                                                          (TrigoMethods.COS, self._expressions[0][1], 1)])
                    else:
                        # TODO: how to derive sin(x)cos(x)tan(x) for example ?
                        pass

    def integral(self):  # TODO: poorly written !
        warnings.warn(
            "This feature is currently extremely limited. Wait for the next versions (Sorry!) ")
        length = len(self._expressions)
        if length == 0:
            return 0
        if length == 1:
            if self._expressions[0][0] == TrigoMethods.SIN:
                self._expressions[0][0] = TrigoMethods.COS
                self._coefficient = -self._coefficient
            elif self._expressions[0][0] == TrigoMethods.COS:
                self._expressions[0][0] = TrigoMethods.SIN
            else:
                pass

    def plot(self, start: float = -8, stop: float = 8,
             step: float = 0.3, ymin: float = -3, ymax: float = 3, title=None, show_axis=True, show=True,
             fig=None, ax=None, formatText=True, values=None):
        variables = self.variables

        num_of_variables = len(variables)
        if num_of_variables == 1:
            plot_function(self.to_lambda(), start, stop, step, ymin, ymax, title, show_axis, show, fig, ax, formatText,
                          values)
        elif num_of_variables == 2:
            plot_function_3d(given_function=self.to_lambda(),
                             start=start, stop=stop, )
        else:
            raise ValueError(
                f"Cannot plot a trigonometric expression with {num_of_variables} variables")

    def newton(self, initial_value: float = 0, epsilon: float = 0.00001, nmax: int = 10_000):
        return newton_raphson(self.to_lambda(), self.derivative().to_lambda(), initial_value, epsilon, nmax)

    def to_dict(self):
        new_expressions = [[(method_chosen.value[0].__name__ if method_chosen is not None else None), inside.to_dict()
                            if hasattr(inside, 'to_dict') else inside, power.to_dict() if hasattr(power, 'to_dict') else power] for
                           [method_chosen, inside, power] in self._expressions]
        return {'type': 'TrigoExpr', 'data': {'coefficient': (
            self._coefficient.to_dict() if hasattr(self._coefficient, 'to_dict') else self._coefficient),
            'expressions': new_expressions}}
        # TODO: shallow copying .. check for bugs

    @staticmethod
    def from_dict(given_dict: dict):
        expressions_objects = [
            [_TrigoMethodFromString(expression[0]), create_from_dict(
                expression[1]), create_from_dict(expression[2])]
            for expression in given_dict['data']['expressions']]
        return TrigoExpr(coefficient=create_from_dict(given_dict['data']['coefficient']),
                         expressions=expressions_objects)

    def __simple_derivative(self, expression_info: "List[Optional[Any],Optional[Union[Mono,Poly,Var,TrigoExpr,PolyLog]],float]"):
        copied_expression = self._expressions[0]
        if copied_expression[0] == TrigoMethods.SIN:
            copied_expression[0] = TrigoMethods.COS
            coefficient = self._coefficient
        elif copied_expression == TrigoMethods.COS:
            copied_expression[0] = TrigoMethods.SIN
            coefficient = -self._coefficient
        elif copied_expression == TrigoMethods.TAN:
            pass
        else:
            pass

    def __single_partial(self, variable: str):
        # Extract all the sub-expressions that contain the given variable
        relevant_expressions = [], irrelevant_expressions = []
        for mini_expression in self._expressions:
            if mini_expression[1] is not None and mini_expression[1].contains_variable(variable):
                relevant_expressions.append(mini_expression)
            else:
                irrelevant_expressions.append(mini_expression)

        if self._coefficient.contains_variable(variable):
            relevant_expressions.append(self._coefficient)
        else:
            irrelevant_expressions.append(self._coefficient)

    # Get an iterable of the variables_dict
    def partial_derivative(self, variables: Iterable[str]):
        copy_of_self = self.__copy__()
        for variable in variables:
            copy_of_self.__single_partial(variable)

    def __str__(self):
        """ Returns a string representation of the expression"""
        if self._coefficient == 0:
            return "0"  # If it's zero, then it's zero. Aristotle ( 300 BC )
        if not self._expressions:  # If it's only a free number
            return f"{self._coefficient}"
        if self._coefficient == 1:
            accumulator = ""
        elif self._coefficient == -1:
            accumulator = '-'
        else:
            accumulator = f"{self._coefficient}"
        for method_chosen, inside, power in self._expressions:
            if method_chosen:
                accumulator += f"{method_chosen.value[0].__name__}({inside})"
            else:
                accumulator += f"{inside}"
            if power != 1:
                accumulator += f"^{round_decimal(power)}"
            accumulator += '*'
        return accumulator[:-1]

    def python_syntax(self):
        if self._coefficient == 0:
            return '0'
        accumulator = f"{self._coefficient.python_syntax()}*"
        if not self._expressions:
            return accumulator
        if accumulator == '1*':
            accumulator = ""
        elif accumulator == '-1*':
            accumulator = "-"
        for method_chosen, inside, power in self._expressions:
            accumulator += f"{method_chosen.value[0].__name__}({inside.python_syntax()})"
            if power != 1:
                accumulator += f"**{round_decimal(power)}"
            accumulator += '*'
        return accumulator[:-1]

    def __copy__(self):  # Copying all the way without any (risky) memory sharing between copies of the same objects
        expressions = []
        if self._expressions:
            for [method, inside, power] in self._expressions:
                inside = inside.__copy__() if inside is not None and hasattr(
                    inside, '__copy__') else inside
                expressions.append([method, inside, power])

        return TrigoExpr(self._coefficient, expressions=expressions)

    @staticmethod
    # TODO: handle the decimal point bug in python with an epsilon or round_decimal()
    def equal_subexpressions(coef1, first_sub: Tuple[Optional[TrigoMethods], IExpression, float],
                             coef2, second_sub: Tuple[Optional[TrigoMethods], IExpression, float]):
        """
        Compare two trigonometric basic expressions. for example: 3sin(x) == -3sin(-x) or 2cos(2x) == 4sin(3x) etc..
        The equation is done in regards to trigonometric identities as much as possible.

        :param coef1: The coefficient of the first expression
        :param first_sub: A tuple of the trigonometric method applied, the expression that the method applies to, and the power.
        :param coef2: The coefficient of the second expression
        :param second_sub: A tuple of the trigonometric method applied, the expression that the method applies to, and the power.
        :return: Returns True if the two
        """
        if coef1 == 0 == coef2:
            return True
        method1, expression1, power1 = first_sub
        method2, expression2, power2 = second_sub
        expression_difference = expression1 - expression2
        evaluated_difference = expression_difference.try_evaluate()
        if evaluated_difference is not None:
            evaluated_difference %= 2 * pi
        if method1 is method2 and power1 == power2:
            if method1 in (TrigoMethods.SIN, TrigoMethods.COS, TrigoMethods.CSC, TrigoMethods.SEC):
                if coef1 == -coef2 and evaluated_difference == pi:
                    # sin(x + pi) = - sin(x), cos(x+pi) = -cos(x) .. etc.
                    return True
            if evaluated_difference is not None and evaluated_difference % 360 == 0 and coef1 == coef2:
                return True
            expression_sum = expression1 + expression2
            if method1 is TrigoMethods.SIN:  # sin()
                # sin(x) = sin(180-x)
                if expression_sum == pi and coef1 == coef2:
                    return True
                # -5sin(x) = 5sin(-x)
                elif coef1 + coef2 == 0 == expression_sum:
                    return True
                else:
                    return False

            elif method1 is TrigoMethods.COS or method1 is TrigoMethods.SEC:
                if expression_sum == 0:  # cos(x) = cos(-x), sec(-x) = sec(x)
                    return True
                # cos(pi - x) = -cos(x) sec(pi - x) = -sec(x)
                if expression_sum == pi:
                    if coef1 == -coef2:
                        return True
            elif method1 is TrigoMethods.TAN or TrigoMethods.COT:
                if evaluated_difference % pi == 0:
                    return True
                if expression_sum == 0:
                    if coef1 == -coef2:
                        return True
                if expression_sum == pi:
                    if coef1 == -coef2:
                        return True
        if (((first_method := method1) is TrigoMethods.SIN and (second_method := method2) is TrigoMethods.COS) or
            ((second_method := method1) is TrigoMethods.COS and (first_method := method2) is TrigoMethods.SIN)) or \
                (((first_method := method1) is TrigoMethods.CSC and (second_method := method2) is TrigoMethods.SEC) or
                 ((second_method := method1) is TrigoMethods.SEC and (first_method := method2) is TrigoMethods.CSC)):

            if evaluated_difference == pi / 2 and coef1 == coef2:
                return True
        if evaluated_difference == -pi / 2 and coef1 == -coef2:
            return True

        else:
            return False

    def __eq__(self, other: Union[IExpression, int, float]):
        if other is None:
            return False
        if isinstance(other, (int, float)):
            if not self._expressions:
                return self._coefficient == other
            else:
                my_evaluation = self.try_evaluate()
                return my_evaluation is not None and my_evaluation == other

        elif isinstance(other, IExpression):
            # First check if the expressions can be evaluated into numbers
            my_evaluation = self.try_evaluate()
            other_evaluation = other.try_evaluate()
            if None not in (my_evaluation, other_evaluation):
                if my_evaluation == other_evaluation:
                    return True
            # if one of them can be evaluated and the other not
            elif (my_evaluation, other_evaluation) != (None, None):
                return False

            if isinstance(other, TrigoExpr):
                first_basic: bool = len(other._expressions) == 1
                second_basic: bool = len(self._expressions) == 1
                if first_basic and second_basic:  # If both are subclasses of TrigoExpr, or they have 1 expressions
                    return self.equal_subexpressions(self.coefficient, self._expressions[0], other.coefficient,
                                                     other._expressions[0])
                else:
                    if self._coefficient != other._coefficient:  # If the coefficients are different
                        try:  # Handle the identity sin(2x) = 2sin(x)cos(x)
                            my_length, other_length = len(
                                self._expressions), len(other._expressions)
                            if my_length + other_length != 3:
                                return False  # return False for now
                            coefficient_ratio = self._coefficient / other._coefficient
                            ratio_eval = coefficient_ratio.try_evaluate()
                            if ratio_eval is None:
                                return False

                            # VERIFY THAT THE RATIO BETWEEN THE COEFFICIENT IS 2 OR 0.5 ( 2SIN(X)COS(X) = SIN(2X) )
                            if abs(ratio_eval - 0.5) < 0.000001:
                                # first: sin(2x), second: 2sin(x)cos(x)
                                first, second = self, other
                                first_length, second_length = my_length, other_length
                            elif abs(ratio_eval - 2) < 0.0000001:
                                first, second = other, self
                                first_length, second_length = other_length, my_length
                            else:
                                return False  # return false for now

                            # VERIFY THE LENGTH OF THE EXPRESSIONS
                            if first_length != 1 or second_length != 2:
                                return False
                            # sin(x)cos(x) ( x == x )
                            if second._expressions[0][1] != second._expressions[1][1]:
                                return False  # return false for now
                            if first._expressions[0][0] != TrigoMethods.SIN:
                                return False
                            second_methods = second._expressions[0][0], second._expressions[1][0]
                            if second_methods not in (
                                    (TrigoMethods.SIN, TrigoMethods.COS), (TrigoMethods.SIN, TrigoMethods.COS)):
                                return False
                            return True

                        except ZeroDivisionError:
                            return False

                    taken_indices = []
                    for other_index, other_sub_list in enumerate(other._expressions):
                        found = False
                        for my_index, sub_list in enumerate(self._expressions):
                            if sub_list == other_sub_list and my_index not in taken_indices:
                                # if it's equal and we haven't found it already, namely, it's not in "taken_indices"
                                found = True
                                # append the current index to the taken indices
                                taken_indices.append(my_index)
                            if found:
                                break  # break to the outer loop
                        if not found:
                            return False
                    return True
            else:
                # TODO: check the cases of comparison between TrigoExpr and other ExpressionSum ..
                return False

    def __ne__(self, other: Union[IExpression, int, float]):
        return not self.__eq__(other)
