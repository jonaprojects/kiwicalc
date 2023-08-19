class Sin(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super().__init__(coefficient=f"sin({expression})", dtype=dtype)
        else:
            super(Sin, self).__init__(1, expressions=((TrigoMethods.SIN, expression, 1),))

    @conversion_wrapper
    def to_cos(self):
        if self._expressions[0][2] == 1:  # If the power is 1
            return Cos(90 - self._expressions[0][1]) * self._coefficient
        elif self._expressions[0][2] == 2:  # If the power is 2, for instance sin(x)^2
            return 1 - Cos(self._expression[0][1]) ** 2

    @conversion_wrapper
    def to_tan(self) -> "Tan":
        pass

    @conversion_wrapper
    def to_cot(self) -> "Cot":
        pass

    @conversion_wrapper
    def to_sec(self) -> "Sec":
        pass

    @conversion_wrapper
    def to_csc(self) -> "Csc":
        pass
