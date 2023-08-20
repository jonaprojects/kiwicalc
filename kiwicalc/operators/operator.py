class Operator:
    __slots__ = ['__sign', '__method']

    def __init__(self, sign: str, method: Callable):
        self.__sign = sign
        self.__method = method

    @property
    def sign(self) -> str:
        return self.__sign

    @property
    def method(self):
        return self.__method

    def __str__(self):
        return self.__sign


class GreaterThan(Operator):
    def __init__(self):
        super(GreaterThan, self).__init__(">", operator.gt)


class LessThan(Operator):
    def __init__(self):
        super(LessThan, self).__init__("<", operator.lt)


class GreaterOrEqual(Operator):
    def __init__(self):
        super(GreaterOrEqual, self).__init__(">=", operator.ge)


class LessOrEqual(Operator):
    def __init__(self):
        super(LessOrEqual, self).__init__("<=", operator.le)


GREATER_THAN, GREATER_OR_EQUAL, LESS_THAN, LESS_OR_EQUAL = GreaterThan(
), GreaterOrEqual(), LessThan(), LessOrEqual()
ptn = re.compile(r"a_(?:n|{n-\d})")  # a_n a_{n-5}
number_pattern = r"\d+[.,]?\d*"  # 3.14159265358979323466
