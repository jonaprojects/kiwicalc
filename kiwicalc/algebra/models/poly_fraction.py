class PolyFraction(Fraction):
    """
    Creating a new algebraic fraction with a polynomial numerator and denominator.
    In later version, further types of expressions will be allowed in fractions.
    """

    def __init__(self, numerator, denominator=None, gen_copies=True):
        if denominator is None:  # if the user chooses to enter only one parameter in the constructor
            if isinstance(numerator, str):  # in a case of a string in the format : (...) / (...)
                numerator1, denominator1 = poly_frac_from_str(
                    numerator, get_tuple=True)
                super().__init__(numerator1, denominator1)
            elif isinstance(numerator, PolyFraction):
                super().__init__(
                    numerator._numerator.__copy__() if gen_copies else numerator._numerator,
                    numerator._denominator.__copy__() if gen_copies else numerator._denominator)
            elif isinstance(numerator, (int, float, Mono, Poly)):
                super().__init__(Poly(numerator), Mono(1))
            else:
                raise TypeError(
                    f"Invalid type for a numerator in PolyFraction : {type(numerator)}.")

        else:  # if the user chooses to specify both parameter in the constructor.
            if isinstance(numerator, Poly):  # Handling the numerator
                numerator = numerator.__copy__()
            elif isinstance(numerator, (int, float, str, Mono)):
                numerator = Poly(numerator)
            else:
                raise TypeError(f"Invalid type for a numerator in PolyFraction : {type(numerator)}. Expected types "
                                f" Poly, Mono, str , float , int")

            if isinstance(denominator, Poly):  # Handling the denominator
                denominator = denominator.__copy__()
            elif isinstance(denominator, (int, float, str, Mono)):
                denominator = Poly(denominator)
            else:
                raise TypeError(f"Invalid type for a denominator in PolyFraction : {type(denominator)}. Expected types "
                                f" Poly, Mono, str , float , int")
            super().__init__(numerator, denominator)

    def roots(self, epsilon: float = 0.000001, nmax: int = 100000):
        return self._numerator.roots(epsilon, nmax)

    def invalid_values(self):
        """ When the denominator evaluates to 0"""
        return self._denominator.roots()  # TODO: hopefully it works..

    def horizontal_asymptote(self):  # TODO: what about multiple asymptotes ?
        power1, power2 = self._numerator.expressions[0].highest_power(), \
            self._denominator.expressions[0].highest_power()
        if power1 > power2 or power1 == power2 == 0:
            return tuple()
        if power1 < power2:
            return 0
        return power1 / power2,

    def __str__(self):
        return f"({self._numerator})/({self._denominator})"

    def __repr__(self):
        return f"PolyFraction({self._numerator.__str__()},{self._denominator.__str__()})"

    def __iadd__(self, other):
        if other == 0:
            return self
        if isinstance(other, PolyFraction):
            if self._denominator == other._denominator:
                self._numerator += other._numerator
                return self
            elif (division_result := self._denominator.__truediv__(other._denominator, get_remainder=True))[
                    1] == 0:  # My denominator is bigger
                self._numerator += other._numerator * division_result[0]
                return self
            # My denominator is smaller
            elif (division_result := other._denominator / self._denominator)[1] == 0:
                self._numerator *= division_result[0]
                self._denominator *= division_result[0]
                self._numerator += other._numerator
                return self
            else:  # There is no linear connection between the two denominators
                raise NotImplemented
        else:
            raise NotImplemented

    def __radd__(self, other):
        new_copy = self.__copy__()
        return new_copy.__iadd__(other)

    def __isub__(self, other):
        if isinstance(other, PolyFraction):
            if self._denominator == other._denominator:
                self._numerator -= other._numerator
                return self
            # My denominator is bigger
            elif (division_result := self._denominator / other._denominator)[1] == 0:
                self._numerator -= other._numerator * division_result[0]
                return self
            # My denominator is smaller
            elif (division_result := other._denominator / self._denominator)[1] == 0:
                self._numerator *= division_result[0]
                self._denominator *= division_result[0]
                self._numerator -= other._numerator
                return self
            else:  # There is no linear connection between the two denominators
                raise NotImplemented
        else:
            raise NotImplemented

    def __sub__(self, other):
        new_copy = self.__copy__()
        return new_copy.__isub__(other)

    def __rsub__(self, other):  # TODO: does this even work ??
        new_copy = self.__copy__()
        new_copy.__isub__(other)
        new_copy.__imul__(-1)

    def __imul__(self, other):
        if isinstance(other, PolyFraction):
            self._numerator *= other._numerator
            self._denominator *= other._denominator
            return self
        elif isinstance(other, (int, float, Mono, Poly)):
            self._numerator *= other
            return self
        else:
            raise TypeError(f"Invalid type {type(other)} for multiplying PolyFraction objects. Allowed types: "
                            f" PolyFraction, Mono, Poly, int, float")

    def __mul__(self, other):
        new_copy = self.__copy__()
        return new_copy.__imul__(other)

    def __rmul__(self, other):
        new_copy = self.__copy__()
        new_copy.__imul__(other)
        return new_copy

    def __rtruediv__(self, other):
        inverse_fraction: PolyFraction = self.reciprocal()
        return inverse_fraction.__imul__(other)

    def reciprocal(self):
        return PolyFraction(self._denominator, self._numerator)

    def __copy__(self):
        """Create a new copy of the polynomial fraction"""
        return PolyFraction(self._numerator, self._denominator)
