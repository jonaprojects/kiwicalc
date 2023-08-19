class Vector:
    def __init__(self, direction_vector=None, start_coordinate=None, end_coordinate=None):
        """
        Creates a new Vector object.


        :param direction_vector: For example, the direction vector of a vector that starts from ( 1,1,1 ) and ends with (4,4,4) is (3,3,3)

        :param start_coordinate: The coordinate that represents the origin of the vector on an axis system.
        :param end_coordinate: The coordinate that represents the end of the vector on an axis system.
        """
        if start_coordinate is not None and end_coordinate is not None:

            if len(start_coordinate) != len(end_coordinate):
                raise ValueError("Cannot handle with vectors with different dimensions in this version.")
            try:
                self._start_coordinate = list(start_coordinate)
            except TypeError:  # If the parameters cannot be converted into lists
                raise TypeError(f"Couldn't convert from type {type(start_coordinate)} to list."
                                f"expected types were tuple,list, set, and dict.")
            try:
                self._end_coordinate = list(end_coordinate)
            except TypeError:
                raise TypeError(f"Couldn't convert from type {type(start_coordinate)} to list."
                                f"expected types were tuple,list, set, and dict.")

            self._direction_vector = list(
                [end_coordinate[i] - start_coordinate[i] for i in range(len(start_coordinate))])
        elif direction_vector is not None and start_coordinate is not None:
            self._start_coordinate = list(start_coordinate)
            self._direction_vector = list(direction_vector)
            self._end_coordinate = [self._start_coordinate[i] + self._direction_vector[i] for i in
                                    range(len(self._start_coordinate))]
        elif direction_vector is not None and end_coordinate is not None:
            self._end_coordinate = list(end_coordinate)
            self._direction_vector = list(direction_vector)
            self._start_coordinate = [self._end_coordinate[i] - self._direction_vector[i] for i in
                                      range(len(self._end_coordinate))]
        elif direction_vector is not None:
            self._direction_vector = list(direction_vector)
            self._end_coordinate = self._direction_vector.copy()
            self._start_coordinate = [0 for _ in range(len(self._end_coordinate))]

    @property
    def start_coordinate(self):
        return self._start_coordinate

    @property
    def end_coordinate(self):
        return self._end_coordinate

    @property
    def direction(self):
        return self._direction_vector

    def plot(self, show=True, arrow_length_ratio: float = 0.05, fig=None, ax=None):
        start_length, end_length = len(self._start_coordinate), len(self._end_coordinate)
        if start_length == end_length == 2:
            plot_vector_2d(
                self._start_coordinate[0], self._start_coordinate[1], self._direction_vector[0],
                self._direction_vector[1], show=show, fig=fig, ax=ax)
        elif start_length == end_length == 3:
            u, v, w = self._direction_vector[0], self._direction_vector[1], self._direction_vector[2]
            start_x, start_y, start_z = self._start_coordinate[0], self._start_coordinate[1], self._start_coordinate[
                2]
            plot_vector_3d(
                (start_x, start_y, start_z), (u, v, w), arrow_length_ratio=arrow_length_ratio, show=show, fig=fig,
                ax=ax)
        else:
            raise ValueError(
                f"Cannot plot a vector with {start_length} dimensions. (Only 2D and 3D plotting is supported")

    def length(self):
        return round_decimal(sqrt(reduce(lambda a, b: a ** 2 + b ** 2, self._direction_vector)))

    def multiply(self, other: "Union[int, float, IExpression, Iterable, Vector, VectorCollection, ]"):
        if isinstance(other, (Vector, Iterable)) and not isinstance(other, (VectorCollection, IExpression)):
            return self.scalar_product(other)
        elif isinstance(other, (int, float, IExpression)):
            return self.multiply_all(other)
        else:
            raise TypeError(f"Vector.multiply(): expected types Vector/tuple/list/int/float but got {type(other)}")

    def multiply_all(self, number: Union[int, float, IExpression]):
        """ Multiplies the vector by the given expression, and returns the current vector ( Which was not copied ) """
        for index in range(len(self._direction_vector)):
            self._direction_vector[index] *= number
        self.__update_end()
        return self

    def scalar_product(self, other: Iterable):
        """

        :param other: other vector
        :return: returns the scalar multiplication of two vectors
        :raise: raises an Exception when the type of other isn't tuple or Vector
        """
        if isinstance(other, Iterable):
            other = Vector(other)
        if isinstance(other, Vector):
            scalar_result = 0
            for a, b in zip(self._direction_vector, other._direction_vector):
                scalar_result += a * b
            return scalar_result
        else:
            raise TypeError(f"Vector.scalar_product(): expected type Vector or tuple, but got {type(other)}")

    # TODO: implement it to work on all vectors, and not to break when zeroes are entered.
    def equal_direction_ratio(self, other):
        """

        :param other: another vector
        :return: True if the two vectors have the same ratio of directions, else False
        """
        try:
            if len(self._direction_vector) != len(other._direction_vector):
                return False
            if len(self._direction_vector) == 0:
                return False
            elif len(self._direction_vector) == 1:
                return self._direction_vector[0] == other._direction_vector[0]
            else:
                ratios = []
                for a, b in zip(self._direction_vector, other._direction_vector):
                    if a == 0 and b != 0 or b == 0 and a != 0:
                        return False  # if that's the case, the vectors can't be equal
                    if not a == b == 0:
                        ratios.append(a / b)
                if ratios:
                    return all(x == ratios[0] for x in ratios)
                return True  # if entered here the list of ratios is empty, meaning all is 0
        except ZeroDivisionError:
            warnings.warn("Cannot check whether the vectors' directions are equal because of a ZeroDivisionError")

    @classmethod
    def random_vector(cls, numbers_range: Tuple[int, int], num_of_dimensions: int = None):
        """
        Generate a random vector object.

        :param numbers_range: the range of possible values
        :param num_of_dimensions: the number of dimensions of the vector. If not set, a number will be chosen
        :return: Returns a Vector object, (or Vector2D or Vector3D objects).
        """
        if cls is Vector2D:
            num_of_dimensions = 2
        elif cls is Vector3D:
            num_of_dimensions = 3

        if num_of_dimensions is None:
            num_of_dimensions = random.randint(2, 9)

        direction = [random.randint(numbers_range[0], numbers_range[1]) for _ in range(num_of_dimensions)]
        start = [random.randint(numbers_range[0], numbers_range[1]) for _ in range(num_of_dimensions)]

        return cls(direction_vector=direction, start_coordinate=start)

    def general_point(self, var_name: str = 'x'):
        """
        Generate an algebraic expression that represents any dot on the vector

        :param var_name: the name of the variable (str)
        :return: returns a list of algebraic expressions
        """
        variable = Var(var_name)
        lst = [start + variable * direction for start, direction in
               zip(self._start_coordinate, self._direction_vector)]
        return lst

    def intersection(self, other: "Union[Vector, VectorCollection, Surface]", get_points=False):
        if self._direction_vector == other._direction_vector:
            print("The vectors have the same directions, unhandled case for now")
            return

        if isinstance(other, Vector):
            my_general, other_general = self.general_point('t'), other.general_point('s')
            solutions_dict = LinearSystem(
                (f"{expr1}={expr2}" for expr1, expr2 in zip(my_general, other_general))).get_solutions()
            if not solutions_dict:
                print("Something went wrong, no solutions were found for t and s !")
            t, s = solutions_dict['t'], solutions_dict['s']
            for expression in my_general:
                expression.assign(t=t)
            if get_points:
                return Point((expression.expressions[0].coefficient for expression in my_general))
            return [expression.expressions[0].coefficient for expression in my_general]
        elif isinstance(other, VectorCollection):
            return any(self.intersection(other_vector) for other_vector in other.vectors)
        elif isinstance(other, Surface):
            return other.intersection(self)
        else:
            raise TypeError(f"Invalid type {type(other)} for searching intersections with a vector. Expected types:"
                            f" Vector, VectorCollection, Surface.")

    def equal_lengths(self, other):
        """

        :param other: another vector
        :return: True if self and other have the same lengths, else otherwise
        """
        return self.length() == other.length()

    @staticmethod
    def fill(dimension: int, value) -> "Vector":
        return Vector(direction_vector=[value for _ in range(dimension)])

    @staticmethod
    def fill_zeros(dimension: int) -> "Vector":
        return Vector.fill(dimension, 0)

    @staticmethod
    def fill_ones(dimension: int) -> "Vector":
        return Vector.fill(dimension, 1)

    def __copy__(self):
        return Vector(start_coordinate=self._start_coordinate, end_coordinate=self._end_coordinate)

    def __eq__(self, other: "Union[Vector, VectorCollection]"):
        """ Returns whether the two vectors have the same starting position, length, and ending position."""
        if other is None:
            return False
        if isinstance(other, Vector):
            return self._direction_vector == other._direction_vector  # Equate by the direction vector.

        elif isinstance(other, VectorCollection):
            if other.num_of_vectors != 1:
                return False
            return self == other.vectors[0]

        else:
            raise TypeError(f"Invalid type {type(other)} for equating vectors.")

    def __ne__(self, other):
        return not self.__eq__(other)

    def __update_end(self):
        self._end_coordinate = [start + direction for start, direction in
                                zip(self._start_coordinate, self._direction_vector)]

    def __imul__(self, other: "Union[IExpression, int, float, Vector, VectorCollection, Surface]"):
        return self.multiply(other)

    def __mul__(self, other):
        return self.__copy__().__imul__(other)

    def __rmul__(self, other):
        return self.multiply(other)

    def power_by_vector(self, other: "Union[Iterable, Vector]"):
        if not isinstance(other, (Iterable, Vector)):
            raise TypeError(f"Invalid type '{type(other)} to raise a vector by another vector ( vector1 ** vector2 )'")
        if isinstance(other, Iterator):
            other = list(other)  # Convert the iterator to list ..
        other_items = other._direction_vector if isinstance(other, Vector) else other
        return Matrix(
            matrix=[[my_item ** other_item for other_item in other_items]
                    for my_item in self._direction_vector]
        )

    def power_by_expression(self, expression: Union[int, float, IExpression]):
        for index in range(len(self._direction_vector)):
            self._direction_vector[index] **= expression
        self.__update_end()
        return self

    def power_by(self, other: "Union[int, float, IExpression, Iterable, Vector, VectorCollection]"):
        return self.__ipow__(other)

    def __ipow__(self, other: "Union[int, float, IExpression, Iterable, Vector]"):
        if isinstance(other, (int, float, IExpression)):
            return self.power_by_expression(other)
        elif isinstance(other, (Vector, Iterable)) and not isinstance(other, (IExpression, VectorCollection)):
            return self.power_by_vector(other)
        else:
            raise TypeError(f"Invalid type '{type(other)}' for raising a Vector by a power.")

    def __pow__(self, other: float):
        return self.__copy__().__ipow__(other)

    def __iadd__(self, other: "Union[Vector, VectorCollection, Surface, IExpression, int, float]"):
        if isinstance(other, Vector):
            for index, other_coordinate in zip(range(len(self._direction_vector)), other._direction_vector):
                self._direction_vector[index] += other_coordinate
            self.__update_end()
            return self
        elif isinstance(other, (IExpression, int, float)):
            for index in range(len(self._direction_vector)):
                self._direction_vector[index] += other
            self.__update_end()
            return self
        elif isinstance(other, VectorCollection):
            other_copy = other.__copy__()
            other_copy.append(self)
            return other_copy
        else:
            raise TypeError(f"Invalid type {type(other)} for adding vectors")

    def __isub__(self, other: "Union[Vector, VectorCollection]"):
        if isinstance(other, Vector):
            for index, other_coordinate in zip(range(len(self._direction_vector)), other._direction_vector):
                self._direction_vector[index] -= other_coordinate
            self._end_coordinate = [start + direction for start, direction in
                                    zip(self._start_coordinate, self._direction_vector)]
            return self
        elif isinstance(other, (IExpression, int, float)):
            for index in range(len(self._direction_vector)):
                self._direction_vector[index] -= other
            self._end_coordinate = [start + direction for start, direction in
                                    zip(self._start_coordinate, self._direction_vector)]
            return self
        elif isinstance(other, VectorCollection):
            other_copy = other.__copy__()
            other_copy.append(-self)
            return other_copy

        else:
            raise TypeError(f"Invalid type {type(other)} for adding vectors")

    def __sub__(self, other: "Union[Vector, VectorCollection]"):
        return self.__copy__().__isub__(other)

    def __rsub__(self, other: "Union[Vector, VectorCollection]"):
        return self.__neg__().__iadd__(other)

    def __add__(self, other: "Union[Vector, VectorCollection]"):
        return self.__copy__().__iadd__(other)

    def __radd__(self, other: "Union[Vector, VectorCollection]"):
        return self.__copy__().__iadd__(other)

    def __neg__(self):
        return Vector(direction_vector=[-x for x in self._direction_vector],
                      start_coordinate=self._end_coordinate,
                      end_coordinate=self._start_coordinate)

    def __str__(self):
        """
        :return: string representation of the vector
        """
        return f"""start: {self._start_coordinate} end: {self._end_coordinate} direction: {self._direction_vector} """

    def __repr__(self):
        """

        :return: returns a string representation of the object's constructor
        """
        return f'Vector(start_coordinate={self._start_coordinate},end_coordinate={self._end_coordinate})'

    def __abs__(self):
        """
        :return: returns a vector with absolute values, preserves the starting coordinate but changes the ending point
        """
        return Vector(direction_vector=[abs(x) for x in self._direction_vector],
                      start_coordinate=self._start_coordinate)

    def __len__(self):
        return self.length()


