from .trigoexpr import TrigoExpr
from ...auxiliary import conversion_wrapper 
from .trigomethods import TrigoMethods


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



class Cot(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Cot, self).__init__(
                coefficient=f"cot{expression}", dtype=dtype)
        else:
            super(Cot, self).__init__(1, expressions=(
                (TrigoMethods.COT, expression, 1),))

    @conversion_wrapper
    def to_sin(self) -> "Sin":
        pass

    @conversion_wrapper
    def to_cos(self) -> "Cos":
        pass

    @conversion_wrapper
    def to_tan(self) -> "Tan":
        pass

    @conversion_wrapper
    def to_sec(self) -> "Sec":
        pass

    @conversion_wrapper
    def to_csc(self) -> "Csc":
        pass


class Sec(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Sec, self).__init__(coefficient=f"sec({expression})")
        else:
            super(Sec, self).__init__(1, expressions=(
                (TrigoMethods.SEC, expression, 1),))

    @conversion_wrapper
    def to_sin(self) -> "Sin":
        pass

    @conversion_wrapper
    def to_cos(self):
        return Fraction(1, Sin(self.expressions[0][1]))

    @conversion_wrapper
    def to_tan(self) -> "Tan":
        pass

    @conversion_wrapper
    def to_cot(self) -> "Cot":
        pass

    @conversion_wrapper
    def to_csc(self) -> "Csc":
        pass



class Csc(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Csc, self).__init__(
                coefficient=f"csc({expression})", dtype=dtype)
        else:
            super(Csc, self).__init__(1, expressions=(
                (TrigoMethods.CSC, expression, 1),))

    @conversion_wrapper
    def to_sin(self) -> "Sin":
        pass

    @conversion_wrapper
    def to_cos(self) -> "Cos":
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



class Asin(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Asin, self).__init__(
                coefficient=f"asin{expression}", dtype=dtype)
        super(Asin, self).__init__(1, expressions=(
            (TrigoMethods.ASIN, expression, 1),))
        


class Acos(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Acos, self).__init__(
                coefficient=f"acos({expression})", dtype=dtype)
        else:
            super(Acos, self).__init__(1, expressions=(
                (TrigoMethods.ACOS, expression, 1),))
            


class Atan(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Atan, self).__init__(
                coefficient=f"atan{expression}", dtype=dtype)
        else:
            super(Atan, self).__init__(1, expressions=(
                (TrigoMethods.ATAN, expression, 1),))


# TODO: add ACOT TO THE SUPPORTED METHODS
class Acot(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(Acot, self).__init__(
                coefficient=f"asec({expression})", dtype=dtype)
        else:
            super(Acot, self).__init__(1, expressions=(
                (TrigoMethods.ACOT, expression, 1),))
            


class ACsc:
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(ACsc, self).__init__(
                coefficient=f"acsc({expression})", dtype=dtype)
        else:
            super(ACsc, self).__init__(1, expressions=(
                (TrigoMethods.ACSC, expression, 1),))
            



class ASec(TrigoExpr):
    def __init__(self, expression, dtype='poly'):
        if isinstance(expression, str):
            super(ASec, self).__init__(
                coefficient=f"asec({expression})", dtype=dtype)
        else:
            super(ASec, self).__init__(1, expressions=(
                (TrigoMethods.ASEC, expression, 1),))