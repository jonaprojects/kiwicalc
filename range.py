class Range:
    __slots__ = ['__expression', '__minimum',
                 '__maximum', '__min_operator', '__max_operator']

    def __init__(self, expression: "Union[str,IExpression, Function, int, float]",
                 limits: Union[set, list, tuple] = None,
                 operators: Union[set, list, tuple] = None, dtype='poly', copy: bool = True):
        # Handle the expression parameter
        if isinstance(expression, str):
            self.__expression, (self.__minimum, self.__maximum), (
                self.__min_operator, self.__max_operator) = create_range(expression, get_tuple=True)
            return
        elif isinstance(expression, (IExpression, Function)):
            self.__expression = expression.__copy__() if copy else expression
        elif isinstance(expression, (int, float)):
            self.__expression = Mono(expression)
        else:
            raise TypeError(
                f"Range.__init__(): Invalid type of expression: {type(expression)}.Expected types 'IExpression', 'Function', "
                f"or str.")

        # check whether limits is valid
        if not isinstance(limits, (set, list, tuple)):
            raise TypeError(
                f"Range.__init__(): Invalid type of limits: {type(limits)}. Expected types 'list', 'tuple', 'set'.")
        if len(limits) != 2:
            raise ValueError("The length")

        # handle the minimum
        if limits[0] in (np.inf, -np.inf):
            self.__minimum = limits[0]
        elif isinstance(limits[0], (int, float)):
            self.__minimum = Mono(limits[0])
        elif isinstance(limits[0], (IExpression, Function)):
            self.__minimum = limits[0].__copy__() if copy else limits[0]
        elif limits[0] is None:
            self.__minimum = -np.inf

        else:
            raise TypeError(
                "Minimum of the range must be of type 'IExpression', 'Function', None, and inf ")

        # handle the maximum
        if limits[1] in (np.inf, -np.inf):
            self.__maximum = limits[1]
        elif isinstance(limits[1], (int, float)):
            self.__maximum = Mono(limits[1])
        elif isinstance(limits[1], (IExpression, Function)):
            self.__maximum = limits[1].__copy__() if copy else limits[1]
        elif limits[1] is None:
            self.__maximum = -np.inf

        else:
            raise TypeError(
                "Maximum of the range must be of type 'IExpression', 'Function', None, and inf ")

        # handle the operators
        if not isinstance(operators, (list, set, tuple)):
            raise TypeError(
                f"Range.__init__(): Invalid type of operators: {type(limits)}. Expected types 'list', 'tuple', 'set'.")

        if not len(operators) == 2:
            raise ValueError(
                f"Range.__init__(): The length of the operators must be 2.")

        if copy:
            self.__min_operator = operators[0].__copy__() if hasattr(
                operators[0], "__copy__") else operators[0]
            self.__max_operator = operators[1].__copy__() if hasattr(
                operators[1], "__copy__") else operators[1]
        else:
            self.__min_operator, self.__max_operator = operators

    @property
    def expression(self):
        return self.__expression

    @property
    def min_limit(self):
        return self.__minimum

    @property
    def max_limit(self):
        return self.__maximum

    @property
    def min_operator(self):
        return self.__min_operator

    @property
    def max_operator(self):
        return self.__max_operator

    def try_evaluate(self):
        return self.__evaluate()

    def evaluate_when(self, **kwargs):
        if isinstance(self.__minimum, IExpression):
            min_eval = self.__minimum.when(**kwargs).try_evaluate()
        else:
            min_eval = None

        expression_eval = self.__expression.when(**kwargs).try_evaluate()

        if isinstance(self.__maximum, IExpression):
            max_eval = self.__maximum.when(**kwargs).try_evaluate()
        else:
            max_eval = None

        return self.__evaluate(min_eval, expression_eval, max_eval)

    def __evaluate(self, min_eval: float = None, expression_eval: float = None, max_eval: float = None) -> Optional[
            bool]:
        if self.__minimum == np.inf or self.__maximum == -np.inf:
            return False
        expression_eval = self.__expression.try_evaluate(
        ) if expression_eval is None else expression_eval
        if self.__minimum != -np.inf:
            minimum_evaluation = self.__minimum.try_evaluate() if min_eval is None else min_eval
            if self.__maximum != np.inf:
                maximum_evaluation = self.__maximum.try_evaluate() if max_eval is None else max_eval
                if None not in (minimum_evaluation, maximum_evaluation):
                    if maximum_evaluation < minimum_evaluation:
                        return False

                if None not in (maximum_evaluation, expression_eval):
                    if not self.__max_operator.method(expression_eval, maximum_evaluation):
                        return False

            if None not in (minimum_evaluation, expression_eval):
                return self.__min_operator.method(minimum_evaluation, expression_eval)
            return None
        else:
            maximum_evaluation = self.__maximum.try_evaluate() if max_eval is None else max_eval
            if None not in (maximum_evaluation, expression_eval):
                return self.__max_operator.method(expression_eval, maximum_evaluation)
            return None

    def __str__(self):
        if self.__minimum == -np.inf and self.__maximum == np.inf:
            return f"-∞{self.__min_operator}{self.__expression}{self.__max_operator}∞"
        if self.__minimum == -np.inf:
            minimum_str = ""
        else:
            minimum_str = f"{self.__minimum}{self.__min_operator}"

        if self.__maximum == np.inf:
            maximum_str = ""
        else:
            maximum_str = f"{self.__max_operator}{self.__maximum}"
        return f"{minimum_str}{self.__expression}{maximum_str}"

    def __copy__(self):
        return Range(self.__expression, (self.__minimum, self.__maximum), (self.__min_operator, self.__max_operator),
                     copy=True)


def range_operator_from_string(operator_str: str):
    if operator_str == '>':
        return GREATER_THAN
    if operator_str == '<':
        return LESS_THAN
    if operator_str == '>=':
        return GREATER_OR_EQUAL
    if operator_str == '<=':
        return LESS_OR_EQUAL
    raise ValueError(
        f"Invalid operator: {operator_str}. Expected: '>', '<', '>=' ,'<='")


def create_range(expression: str, min_dtype: str = 'poly', expression_dtype: str = 'poly', max_dtype='poly',
                 get_tuple=False):
    exprs = re.split('(<=|>=|>|<)', expression)
    num_of_expressions = len(exprs)
    if num_of_expressions == 5:  # 3 < x < 6
        limits = create(exprs[0], dtype=min_dtype), create(
            exprs[4], dtype=max_dtype)
        middle = create(exprs[2], dtype=expression_dtype)
        min_operator, max_operator = range_operator_from_string(
            exprs[1]), range_operator_from_string(exprs[3])
    elif num_of_expressions == 3:
        middle = create(exprs[0], dtype=min_dtype)
        my_operator = exprs[1]
        if '>' in my_operator:
            my_operator = my_operator.replace('>', '<')
            min_operator, max_operator = range_operator_from_string(
                my_operator), None
            limits = (create(exprs[2], dtype=min_dtype), None)
        elif '<' in my_operator:
            min_operator, max_operator = None, range_operator_from_string(
                my_operator)
            limits = (None, create(exprs[2], dtype=max_dtype))
        else:
            raise ValueError(
                f"Invalid operator: {my_operator}. Expected: '>', '<', '>=' ,'<='")
    else:
        raise ValueError(f"Invalid string for creating a Range expression: {expression}. Expected expressions"
                         f" such as '3<x<5', 'x^2 > 16', etc..")

    if get_tuple:
        return middle, limits, (min_operator, max_operator)

    return Range(expression=middle, limits=limits, operators=(min_operator, max_operator))


class RangeCollection:
    __slots__ = ['_ranges']

    def __init__(self, ranges: "Iterable[Range, RangeCollection]", copy=False):
        if copy:
            self._ranges = [my_range.__copy__() for my_range in ranges]
        else:
            self._ranges = [my_range for my_range in ranges]

    @property
    def ranges(self):
        return self._ranges

    def chain(self, range_obj: Range, copy=False):
        if not isinstance(range_obj, Range):
            return TypeError(f"Invalid type {type(range_obj)} for chaining Ranges. Expected type: 'Range' ")
        self._ranges.append((range_obj.__copy__() if copy else range_obj))
        return self

    def __or__(self, other: Range):
        return RangeOR((self, other))

    def __and__(self, other):
        return RangeAND((self, other))

    def __copy__(self):
        return RangeCollection(ranges=self._ranges, copy=True)

    def __str__(self):
        return ", ".join(
            (f"({my_range.__str__()})" if isinstance(my_range, RangeCollection) else my_range.__str__()) for my_range in
            self._ranges)


class RangeOR(RangeCollection):
    """
    This class represents several ranges or collection of ranges with the OR method.
    For instance:
    (x^2 > 25) or (x^2 < 9)
    Or a more complicated example:
    (5<x<6 and x^2>26) or x<7 or (sin(x)>=0 or sin(x) < 0.5)
    """

    def __init__(self, ranges: "Iterable[Range, RangeCollection]", copy=False):
        super(RangeOR, self).__init__(ranges)

    def try_evaluate(self):
        pass

    def simplify(self) -> Optional[Union[Range, RangeCollection]]:
        pass

    def __str__(self):
        return " or ".join(
            (f"({my_range.__str__()})" if isinstance(my_range, RangeCollection) else my_range.__str__()) for my_range in
            self._ranges)

    def __copy__(self):
        return RangeOR(self._ranges, copy=True)


class RangeAND(RangeCollection):
    """
    This class represents several ranges or collection of ranges with the AND method.
    For instance:
    (x^2 > 25) and (x>0)
    Or a more complicated example:
    (5<x<6 and x^2>26) and x<7 and (sin(x)>=0 or sin(x) < 0.5)
    """

    def __init__(self, ranges: "Iterable[Range, RangeCollection]", copy=False):
        super(RangeAND, self).__init__(ranges)

    def try_evaluate(self):
        pass

    def simplify(self) -> Optional[Union[Range, RangeCollection]]:
        pass

    def __str__(self):
        return " and ".join(
            (f"({my_range.__str__()})" if isinstance(my_range, RangeCollection) else my_range.__str__()) for my_range in
            self._ranges)

    def __copy__(self):
        return RangeOR(self._ranges, copy=True)
