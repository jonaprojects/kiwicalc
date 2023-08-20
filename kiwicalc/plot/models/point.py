class Point:
    def __init__(self, coordinates: Union[Iterable, int, float]):
        if isinstance(coordinates, Iterable):
            self._coordinates = [coordinate for coordinate in coordinates]
            for index, coordinate in enumerate(self._coordinates):
                if isinstance(coordinate, IExpression):
                    self._coordinates[index] = coordinate.__copy__()
        elif isinstance(coordinates, (int, float)):
            self._coordinates = [coordinates]
        else:
            raise TypeError(
                f"Invalid type {type(coordinates)} for creating a new Point object")

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates: Iterable):
        self._coordinates = coordinates

    @property
    def dimensions(self):
        return len(self._coordinates)

    def plot(self):
        self.scatter()

    def scatter(self, show=True):  # TODO: create it with a grid and stuff
        if len(self._coordinates) == 1:
            plt.scatter(self._coordinates[0], 0)
        if len(self._coordinates) == 2:
            plt.scatter(self._coordinates[0], self._coordinates[1])
        elif len(self._coordinates) == 3:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter2d(
                self._coordinates[0], self._coordinates[1], self._coordinates[2])
        if show:
            plt.show()

    def __iadd__(self, other: "Union[Iterable, Point]"):
        if isinstance(other, Iterable):
            self._coordinates = [coord1 + coord2 for coord1,
                                 coord2 in zip(self._coordinates, other)]
            return self
        elif isinstance(other, Point):
            self._coordinates = [coord1 + coord2 for coord1,
                                 coord2 in zip(self._coordinates, other._coordinates)]
            return self
        else:
            raise TypeError(f"Encountered unexpected type {type(other)} while attempting to add points. Expected types"
                            f"Iterable or Point")

    def __isub__(self, other: "Union[Iterable, Point]"):
        if isinstance(other, Point):
            self._coordinates = [coord1 - coord2 for coord1,
                                 coord2 in zip(self._coordinates, other._coordinates)]
            return self
        elif isinstance(other, Iterable):
            self._coordinates = [coord1 - coord2 for coord1,
                                 coord2 in zip(self._coordinates, other)]
            return self

        else:
            raise TypeError(f"Encountered unexpected type {type(other)} while attempting to subtract points. Expected"
                            f"types Iterable or Point")

    def __add__(self, other: "Union[Iterable, Point]"):
        return self.__copy__().__iadd__(other)

    def __radd__(self, other: "Union[Iterable, Point]"):
        if isinstance(other, Iterable):
            other = Point(Iterable)

        if isinstance(other, Point):
            return other.__add__(self)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        return self.__copy__().__isub__(other)

    def __rsub__(self, other: "Union[Point,PointCollection]"):
        if isinstance(other, Iterable):
            other = Point(Iterable)

        if isinstance(other, Point):
            return other.__sub__(self)
        else:
            raise NotImplementedError

    def __imul__(self, other: "Union[int, float, Point, PointCollection, IExpression]"):
        if isinstance(other, (int, float, IExpression)):
            if isinstance(other, IExpression):
                other_evaluation = other.try_evaluate()
                if other_evaluation is not None:
                    other = other_evaluation
            for index in range(len(self._coordinates)):
                self._coordinates[index] *= other
                return self
        elif isinstance(other, Point):
            return reduce(lambda tuple1, tuple2: tuple1[0] * tuple2[0] + tuple1[1] * tuple2[1],
                          zip(self._coordinates, other._coordinates))
        elif isinstance(other, PointCollection):
            raise NotImplementedError(
                "This feature isn't implemented yet in this version")

    def __mul__(self, other: "Union[int, float, Point, PointCollection, IExpression]"):
        return self.__copy__().__imul__(other)

    def __rmul__(self, other: "Union[int, float, Point, PointCollection, IExpression]"):
        return self.__copy__().__imul__(other)

    def coord_at(self, index: int):
        return self._coordinates[index]

    def max_coord(self):
        return max(self._coordinates)

    def min_coord(self):
        return min(self._coordinates)

    def sum(self):
        return sum(self._coordinates)

    def distance(self, other_point: "Point"):
        if len(self.coordinates) != len(other_point.coordinates):
            raise ValueError(
                f"Cannot calculate distance between points in different dimensions.")
        return sqrt(sum((coord1 - coord2) ** 2 for (coord1, coord2) in zip(self.coordinates, other_point.coordinates)))

    def __eq__(self, other: "Union[Point, PointCollection]"):
        if other is None:
            return False
        if isinstance(other, PointCollection):
            if len(other.points) != 1:
                return False
            other = other.points[0]

        if isinstance(other, Point):
            return self._coordinates == other._coordinates
        else:
            raise TypeError(
                f"Invalid type {type(other)} for comparing with a Point object")

    def __ne__(self, other: "Union[Point, PointCollection]"):
        return not self.__eq__(other)

    def __neg__(self):
        return Point(coordinates=[-coordinate for coordinate in self._coordinates])

    def __repr__(self):
        return f"Point({self._coordinates})"

    def __str__(self):
        if all(isinstance(coordinate, (int, float)) for coordinate in self._coordinates):
            return f"({','.join(str(round(coordinate, 3)) for coordinate in self._coordinates)})"
        return f"({','.join(coordinate.__str__() for coordinate in self._coordinates)})"

    def __copy__(self):
        # New coordinates will be created in the init, so memory won't be shared
        return Point(self._coordinates)
        # between different objects

    def __len__(self):
        return len(self._coordinates)


class Point1D(Point, IPlottable):
    def __init__(self, x: Union[int, float, IExpression]):
        super(Point1D, self).__init__((x,))

    @property
    def x(self):
        return self._coordinates[0]


class Point2D(Point, IPlottable):
    def __init__(self, x: Union[int, float, IExpression], y: Union[int, float, IExpression]):
        super(Point2D, self).__init__((x, y))

    @property
    def x(self):
        return self._coordinates[0]

    @property
    def y(self):
        return self._coordinates[1]


class Point3D(Point):
    def __init__(self, x: Union[int, float, IExpression], y: Union[int, float, IExpression],
                 z: Union[int, float, IExpression]):
        super(Point3D, self).__init__((x, y, z))

    @property
    def x(self):
        return self._coordinates[0]

    @property
    def y(self):
        return self._coordinates[1]

    @property
    def z(self):
        return self._coordinates[2]


class Point4D(Point):
    def __init__(self, x: Union[int, float, IExpression], y: Union[int, float, IExpression],
                 z: Union[int, float, IExpression], c: Union[int, float, IExpression]):
        super(Point4D, self).__init__((x, y, z, c))

    @property
    def x(self):
        return self._coordinates[0]

    @property
    def y(self):
        return self._coordinates[1]

    @property
    def z(self):
        return self._coordinates[2]

    @property
    def c(self):
        return self._coordinates[3]
