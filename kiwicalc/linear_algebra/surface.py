class Surface:
    """
    represents a surface of the equation ax+by+cz+d = 0, where (a,b,c) is the perpendicular of the surface, and d
    is a free number.
    """

    def __init__(self, coefs):
        if isinstance(coefs, str):
            self.__a, self.__b, self.__c, self.__d = surface_from_str(
                coefs, get_coefficients=True)
        elif isinstance(coefs,
                        Iterable):  # TODO: change to a more specific type hint later, but still one that accepts generators
            coefficients = [coef for coef in coefs]
            if len(coefficients) == 4:
                self.__a, self.__b, self.__c, self.__d = coefficients[0], coefficients[1], coefficients[2], \
                    coefficients[3]
            elif len(coefficients) == 3:
                self.__a, self.__b, self.__c, self.__d = coefficients[
                    0], coefficients[1], coefficients[2], 0
            else:
                raise ValueError(
                    f"Invalid number of coefficients in coefficients of surface. Got {len(coefficients)}, expected 4 or 3")

    @property
    def a(self):
        return self.__a

    @property
    def b(self):
        return self.__b

    @property
    def c(self):
        return self.__c

    @property
    def d(self):
        return self.__d

    # TODO: check if intersects or that the continuation does
    def intersection(self, vector: Vector, get_point=False):
        """
        Finds the intersection between a surface and a vector

        :param get_point: If set to True, a point that represents the intersection will be returned instead of a list that represents the coordinates of the intersection. Default value is false.

        :param vector: An object of type Vector.
        :return: Returns a list of the coordinates of the intersection. If get_point = True, returns corresponding
        point object.
        """
        general_point = vector.general_point('t')
        expression = self.__a * \
            general_point[0] + self.__b * general_point[1] + \
            self.__c + general_point[2] + self.__d
        t_solution = LinearEquation(
            f"{expression} = 0", variables=('t',), calc_now=True).solution
        for polynomial in general_point:
            polynomial.assign(t=t_solution)
        if get_point:
            return Point((polynomial.expressions[0].coefficient for polynomial in general_point))
        # TODO: check if this works
        return [polynomial.expressions[0].coefficient for polynomial in general_point]

    def __str__(self) -> str:
        """Getting the string representation of the algebraic formula of the surface. ax + by + cz + d = 0"""
        accumulator = f"{self.__a}"
        return accumulator + "".join(
            ((f"+{val}{var}" if val > 0 else f"-{val}{var}") for val, var in
             zip((self.__b, self.__c, self.__d), ('x', 'y', 'z', '')) if val != 0))

    def __repr__(self):
        return f'Surface("{self.__str__()}")'

    def to_lambda(self):
        if self.__c == 0:
            warnings.warn(
                "c = 0 might lead to unexpected behaviors in this version.")
            return lambda x, y: 0
        return lambda x, y: (-self.__a * x - self.__b * y - self.__d) / self.__c

    def plot(self, start: float = -3, stop: float = 3,
             step: float = 0.3,
             xlabel: str = "X Values",
             ylabel: str = "Y Values", zlabel: str = "Z Values", show=True, fig=None, ax=None,
             write_labels=True, meshgrid=None):
        plot_function_3d(self.to_lambda(),
                         start=start, stop=stop, step=step, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, show=show,
                         fig=fig,
                         ax=ax, write_labels=write_labels, meshgrid=meshgrid
                         )

    def __eq__(self, other):
        """Equating between surfaces. Surfaces are equal if they have the same a,b,c,d coefficients """
        if other is None:
            return False
        if isinstance(other, Surface):
            return (self.__a, self.__b, self.__c, self.__c) == (other.__a, other.__b, other.__c, other.__d)
        if isinstance(other, list):
            return [self.__a, self.__b, self.__c, self.__d] == other
        if isinstance(other, tuple):
            return (self.__a, self.__b, self.__c, self.__d) == other
        if isinstance(other, set):
            return {self.__a, self.__b, self.__c, self.__d} == other
        raise TypeError(f"Invalid type '{type(other)}' for checking equality with object of instance of class Surface."
                        f"Expected types 'Surface', 'list', 'tuple', 'set'. ")

    def __ne__(self, other):
        return not self.__eq__(other)
