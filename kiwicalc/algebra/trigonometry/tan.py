class Tan(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Tan, self).__init__(
                coefficient=f"tan({expression})", dtype=dtype)
        else:
            super(Tan, self).__init__(1, expressions=(
                (TrigoMethods.TAN, expression, 1),))

    @conversion_wrapper
    def to_sin(self) -> "Sin":
        pass

    @conversion_wrapper
    def to_cos(self) -> "Cos":
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
