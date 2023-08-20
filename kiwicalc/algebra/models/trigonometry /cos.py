class Cos(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Cos, self).__init__(
                coefficient=f"cos({expression})", dtype=dtype)
        else:
            super(Cos, self).__init__(1, expressions=(
                (TrigoMethods.COS, expression, 1),))

    @conversion_wrapper
    def to_sin(self) -> "Sin":
        pass

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
