class Log(IExpression, IPlottable, IScatterable):
    __slots__ = ['_coefficient', '_expressions']

    def __init__(self, expression, base: Union[int, float] = 10, coefficient: Union[IExpression, int, float] = 1,
                 dtype='poly', gen_copies=True):

        if isinstance(expression, (int, float)):
            if expression < 0:
                raise ValueError(
                    "Negative logarithms are not defined in this program.")
            elif expression == 0 and base > 0:
                raise ValueError(
                    f"{expression}^n cannot be 0 for any real n, Thus the expression isn't defined!")
            else:
                self._coefficient = Mono(log(expression, base))
                self._expressions = []
                return

        elif isinstance(expression, str):
            result = log_from_str(expression, get_tuple=True, dtype=dtype)
            coefficient, expression = result[0], [
                [result[1], result[2], result[3]]]

        elif isinstance(expression, IExpression):
            if isinstance(coefficient, (int, float)):
                self._coefficient = Mono(coefficient)
            elif isinstance(coefficient, IExpression):
                self._coefficient = coefficient.__copy__() if gen_copies else coefficient
            else:
                raise TypeError(
                    f"Log.__init__(): invalid type for coefficient: {type(coefficient)}")
            self._expressions = [
                [expression.__copy__() if gen_copies else expression, base, 1]]
            return

        if isinstance(expression, List) and len(expression) and isinstance(expression[0], list):
            self._expressions = []
            for inner_list in expression:
                self._expressions.append(
                    [inner_list[0].__copy__(), inner_list[1], inner_list[2]])

        else:
            self._expressions: List[List[IExpression, Union[float, int], Union[int, float]]] = [
                list(expression) for expression in
                expression]  # inside, base, power
        self._coefficient = coefficient.__copy__() if isinstance(
            coefficient, IExpression) else Mono(coefficient)

    @property
    def coefficient(self):
        return self._coefficient

    def index_of(self, other_list):
        """ """
        if other_list in self._expressions:
            return self._expressions.index(other_list)
        return -1

    def all_bases(self) -> Set[float]:
        return {inner_list[1] for inner_list in self._expressions}

    def biggest_power(self) -> float:
        return max((inner_list[2] for inner_list in self._expressions))

    @property
    def variables(self):
        variables = set()
        for (inside, base, power) in self._expressions:
            variables.update(inside.variables_dict)
        return variables

    def simplify(self):  # TODO: further implementation needed
        self._coefficient.simplify()
        for (inside, base, power) in self._expressions:
            if hasattr(inside, "simplify"):
                inside.simplify()
            if hasattr(base, "simplify"):
                base.simplify()
            if hasattr(power, "simplify"):
                power.simplify()

    def __iadd__(self, other: "Union[int,float,IExpression]"):
        if other == 0:
            return self
        if isinstance(other, (int, float)):
            if not self._expressions:  # Meaning that the current object basically represents a free number
                self._coefficient += other
                return self
            return ExpressionSum(expressions=(self, Mono(other)))
        elif isinstance(other, Log):
            my_eval = self.try_evaluate()
            other_eval = other.try_evaluate()
            if None not in (my_eval, other_eval):
                return Mono(my_eval + other_eval)
            if my_eval is None:
                pass
            elif other_eval is None:
                pass
            if len(self._expressions) == 1 == len(other._expressions) and not isinstance(self._coefficient, Log) \
                    and not isinstance(other._coefficient, Log) and self._expressions[0][1] == other._expressions[0][1]:
                if self._coefficient == other._coefficient:  # The coefficient stays the same
                    self._expressions[0][0] *= other._expressions[0][0]
                    return self
                else:
                    try:
                        self._expressions[0][0] *= other._expressions[0][0] ** other._expressions[0][2]
                    except (TypeError, ValueError):
                        return ExpressionSum((self, other))
            for inner_list in other._expressions:
                self._expressions.append(
                    [inner_list[0].__copy__(), inner_list[1], inner_list[2]])  # Appending the
                # other's items to our items
            return self
        else:
            return ExpressionSum((self, other))

    def __radd__(self, other):
        return self.__copy__().__iadd__(other)

    def __isub__(self, other):
        if isinstance(other, (int, float)):
            if not self._expressions:  # Meaning that the current object basically represents a free number
                self._coefficient -= other
                return self
            return ExpressionSum(expressions=(self, Mono(other)))
        elif isinstance(other, Log):
            if len(self._expressions) == 1 == len(other._expressions) and not isinstance(self._coefficient, Log) \
                    and not isinstance(other._coefficient, Log) and self._expressions[0][1] == other._expressions[0][1]:
                if self._coefficient == other._coefficient:  # The coefficient stays the same
                    self._expressions[0][0] /= other._expressions[0][0]
                    return self
                else:
                    try:
                        self._expressions[0][0] /= other._expressions[0][0] ** other._expressions[0][2]
                    except (TypeError, ValueError):
                        return ExpressionSum((self, other))

        return ExpressionSum((self, other.__neg__()))

    def __sub__(self, other):
        return self.__copy__().__isub__(other)

    def __imul__(self, other: "Union[int, float, IExpression]"):
        if isinstance(other, Log):
            other_evaluation = other.try_evaluate()
            if other_evaluation is not None:
                self._coefficient *= other_evaluation
            else:
                self._coefficient *= other._coefficient
                for other_list in other._expressions:
                    existing_appearance: int = self.index_of(other_list)
                    if existing_appearance != -1:
                        self._expressions[existing_appearance][2] += other_list[2]
                    else:
                        self._expressions.append(
                            [other_list[0].__copy__(), other_list[1], other_list[2]])
            return self
        else:
            self._coefficient *= other

            return self

    def __mul__(self, other: "Union[int, float, IExpression]"):
        return self.__copy__().__imul__(other)

    def __rmul__(self, other):
        return self.__copy__().__imul__(other)

    def __itruediv__(self, other: "Union[int, float, IExpression]"):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError(
                    "Cannot Divide A logarithm by an expression that evaluates to 0")
            self._coefficient /= other
            return self
        if isinstance(other, IExpression):
            my_eval, other_eval = self.try_evaluate(), other.try_evaluate()
            if None not in (my_eval, other_eval):
                if other_eval == 0:
                    raise ZeroDivisionError(
                        "Cannot Divide A logarithm by an expression that evaluates to 0")
                self._expressions = []
                self._coefficient = my_eval / other_eval
                if isinstance(self._coefficient, (int, float)):
                    self._coefficient = Mono(self._coefficient)
                return self
            elif other_eval is not None:
                self._coefficient /= other_eval
                return self
            else:
                # TODO: add here checks to see if the expressions are dependant and thus can be simplified!
                return Fraction(self, other)

    def __ipow__(self, power):
        self._coefficient **= power
        for mini_expression in self._expressions:
            mini_expression[2] *= power
        return self

    def __neg__(self):
        self._coefficient *= -1
        return self

    def assign(self, **kwargs):
        for expression in self._expressions:
            # TODO: can be dangerous, requires further checking
            expression[0].assign(**kwargs)

    def try_evaluate(self):
        """ return an int / float evaluation of the expression. If not possible, return None."""
        evaluated_coefficient = self._coefficient.try_evaluate()
        if evaluated_coefficient is None:
            return None
        if not self._expressions:
            return round_decimal(evaluated_coefficient)
        evaluated_inside = self._expressions[0][0].try_evaluate()
        if not isinstance(self._expressions[0][1], (int, float)):
            evaluated_base = self._expressions[0][1].try_evaluate()
        else:
            evaluated_base = self._expressions[0][1]
        if evaluated_inside is None:
            return None
        power = self._expressions[0][2]
        if isinstance(power, IExpression):
            power = power.try_evaluate()
            if power is None:
                return None
        return round_decimal(evaluated_coefficient * log(evaluated_inside, evaluated_base) ** power)

    def _single_log_str(self, inside: IExpression, base, power_by) -> str:

        if power_by == 0:
            return "1"
        if power_by == 1:
            power_by = ""
        else:
            power_by = f"^{round_decimal(power_by)} "
        if abs(base - e) < 0.00001:
            return f"ln({inside.__str__()}){power_by}"
        return f"log{base}({inside.__str__()}){power_by}"

    def python_syntax(self) -> str:
        """ Return a string that represents the expression and can be evaluated into expression using eval()"""
        if isinstance(self._coefficient, IExpression):
            coefficient_str = self._coefficient.python_syntax()
        else:
            coefficient_str = self._coefficient.__str__()
        if coefficient_str == "1":
            coefficient_str = ""
        elif coefficient_str == "-1":
            coefficient_str = "-"
        else:
            coefficient_str += "*"
        expression_str = ""
        for (expression, base, power) in self._expressions:
            if isinstance(power, IExpression):
                power = power.python_syntax()
            if power == 1:
                power = ""
            else:
                power = f"** {power}"
            if isinstance(expression, IExpression):
                expression_str += f"log({expression.python_syntax()},{base}){power}*"
            else:
                expression_str += f"log({expression.__str__()},{base}){power}*"
        if len(expression_str):  # If the string isn't empty because the object represents a free number
            # remove the last star from the string
            expression_str = expression_str[:-1]

        return f"{coefficient_str}{expression_str}"

    def to_dict(self):
        return {'type': 'Log', 'data': {'coefficient': (self._coefficient.to_dict() if
                                                        hasattr(self._coefficient,
                                                                'to_dict') else self._coefficient),
                                        'expressions': [[(inside.to_dict() if
                                                          hasattr(inside, 'to_dict') else inside),
                                                         (base.to_dict() if hasattr(
                                                             base, 'to_dict') else base),
                                                         (power.to_dict() if
                                                          hasattr('power', 'to_dict') else power)] for
                                                        [inside, base, power] in self._expressions]}}

    @staticmethod
    def from_dict(given_dict: dict):
        coefficient_obj = create_from_dict(given_dict['data']['coefficient'])
        expressions_objs = [
            [create_from_dict(expression[0]), create_from_dict(expression[1]), create_from_dict(expression[2])] for
            expression in given_dict['data']['expressions']]
        return Log(expression=expressions_objs, coefficient=coefficient_obj)

    def __str__(self) -> str:
        if self._coefficient == 0:
            return "0"
        if not self._expressions:
            return f"{self._coefficient}"
        coefficient_str: str = format_coefficient(self._coefficient)
        if coefficient_str not in ("", "-"):
            coefficient_str += '*'

        return coefficient_str + "*".join(
            self._single_log_str(log_list[0], log_list[1], log_list[2])
            for log_list in self._expressions)

    def __copy__(self):
        new_log = Log(self._expressions)
        new_log._coefficient = self._coefficient.__copy__()
        return new_log

    # TODO: implement and use in __eq__
    def _equate_single_logs(self, other):
        pass

    def __eq__(self, other: Union[int, float, IExpression]):
        if other is None:
            return False
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            # TODO: check if this applies to all cases
            return my_evaluation is not None and my_evaluation == other
        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if my_evaluation == other_evaluation:
                return True
            if isinstance(other, Log):
                if self._coefficient != other._coefficient:
                    return False
                # Do equating similarly to what is done in TrigoExpr , between the expressions
            else:
                return False
        else:
            raise TypeError(
                f"Invalid type '{type(other)}' for equating logarithms.")

    def __ne__(self, other: Union[IExpression, int, float]):
        return not self.__eq__(other)


class PolyLog(Log):  # Find out the proper implementation
    def __init__(self, expressions: Union[
            Iterable[Union[Poly, Mono]], Iterable[List[list]], Union[Poly, Mono], str, float, int],
            base: Union[int, float] = 10, coefficient: Union[int, float] = Poly(1)):
        super(PolyLog, self).__init__(expression=expressions,
                                      base=base, coefficient=coefficient)


class Ln(Log):
    def __init__(self, expressions: Union[Iterable[IExpression], Iterable[List[list]], IExpression, str, float, int]):
        super(Ln, self).__init__(expressions, base=e)


def log_from_str(expression: str, get_tuple=False, dtype: str = 'poly'):
    # Make sure the string entered is lower case and without spaces
    expression = expression.strip().lower()
    if "log" in expression or "ln" in expression:

        # STEP 1 : Extract the _coefficient
        coefficient = expression[:expression.find('l')]
        if coefficient == '':
            coefficient = 1
        elif coefficient == '-':
            coefficient = -1
        else:
            try:
                # Add support on coefficients such as e and pi here perhaps later
                coefficient = float(coefficient)
            except ValueError:  # Raise an appropriate message to the user, so he knows his mistake. Is this verbose?
                raise ValueError(f"Invalid _coefficient '{coefficient}' in expression {expression}, while creating"
                                 f"a PolyLog object from a given string.")
        #  STEP 2: Extract the polynomial and the base
        start_parenthesis = expression.find('(')
        if start_parenthesis == -1:
            raise ValueError(
                F"Invalid string '{expression}' without opening parenthesis for the expression.")
        ending_parenthesis = expression.find(')')
        if ending_parenthesis == -1:
            raise ValueError(
                f"Invalid string: '{ending_parenthesis} without ending parenthesis for the expression'")
        if "log" in expression:
            inside = expression[start_parenthesis + 1:ending_parenthesis]
            if ',' in inside:
                inside, base = inside.split(',')
                base = float(base)
            else:
                base = 10
            inside = create(inside, dtype=dtype)
        else:
            base = 'e'
            inside = create(
                expression[start_parenthesis + 1:ending_parenthesis], dtype=dtype)

        # STEP 3: Extract the power
        power_index = expression.find('^')
        if power_index == -1:
            power_index = expression.find('**')
        if power_index == -1:  # In case no sign of power was found, the default is 1
            power = 1
        else:
            close_parenthesis_index = expression.rfind(')')
            if power_index > close_parenthesis_index:
                power = float(expression[power_index + 1:])
            else:
                power = 1

        if get_tuple:
            return coefficient, inside, base, power
        return Log(expression=[[inside, base, power]], coefficient=coefficient)
    else:
        raise ValueError("The string need to contain log() or ln()")
