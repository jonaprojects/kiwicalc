class Circle(IPlottable):
    def __init__(self, radius: Union[float, int, IExpression],
                 center: Union[Iterable[Union[int, float, IExpression]], Point] = (
                     0, 0),
                 gen_copies=False):
        # Handling the radius
        if isinstance(radius, (int, float)):
            self._radius = Mono(radius)
        elif isinstance(radius, IExpression):
            if gen_copies:
                self._radius = radius.__copy__()
            else:
                self._radius = radius
        else:
            raise TypeError(
                f"Invalid type {type(radius)} for radius when creating a Circle object")
        # Handling the Center
        if isinstance(center, Iterable) and not isinstance(center, Point):
            center_list = [coordinate for coordinate in center]
            if any(not isinstance(coordinate, (IExpression, int, float)) for coordinate in center_list):
                raise TypeError(
                    f"Invalid types of coordinates when creating a Circle object")
            for index, coordinate in enumerate(center_list):
                if isinstance(coordinate, (int, float)):
                    center_list[index] = Mono(coordinate)
            center = Point(center_list)

        if isinstance(center, Point):
            if center.dimensions != 2:
                raise ValueError(
                    f"Circle object can only contain a 2D Point as a center ( Got {center.dimensions}D")
            self._center = center.__copy__() if gen_copies else center
        else:
            raise TypeError(
                f"Invalid type {type(center)} for the center point when creating a Circle object")

    @property
    def radius(self):
        return self._radius

    @property
    def diameter(self):
        return self._radius * 2

    @property
    def center(self) -> Point:
        return self._center

    @property
    def left_edge(self):
        return Point((-self._radius + self.center_x, self.center_y))

    @property
    def right_edge(self):
        return Point((self._radius + self.center_x, self.center_y))

    @property
    def top_edge(self):
        return Point((self.center_x, self._radius + self.center_y))

    @property
    def bottom_edge(self):
        return Point((self.center_x, -self._radius + self.center_y))

    @property
    def center_x(self):
        return self._center.coordinates[0]

    @property
    def center_y(self):
        return self._center.coordinates[1]

    def area(self):
        result = self._radius ** 2 * pi
        if isinstance(result, IExpression):
            result_eval = result.try_evaluate()
            if result_eval is not None:
                return result_eval
            return result
        return result

    def perimeter(self):
        result = self._radius * 2 * pi
        if isinstance(result, IExpression):
            result_eval = result.try_evaluate()
            if result_eval is not None:
                return result_eval
            return result
        return result

    def point_inside(self, point: Union[Point, Iterable], already_evaluated: Tuple[float, float, float] = None) -> bool:
        """
        Checks whether a 2D point is inside the circle

        :param point: the point
        :param already_evaluated: Evaluations of the radius and center point of the circle as floats.
        :return: Returns True if the point is indeed inside the circle or touches it from the inside, otherwise False.
        """
        if isinstance(point, Point):
            # TODO: later accept only Point2D objects..
            x, y = point.coordinates[0], point.coordinates[1]
        elif isinstance(point, Iterable):
            coordinates = [coord for coord in point]
            # TODO: later accept only Point2D objects..
            if len(coordinates) != 2:
                raise ValueError("Can only accept points with 2 dimensions")
            x, y = coordinates[0], coordinates[1]
        else:
            raise ValueError(f"Invalid type {type(point)} for this method.")
        if already_evaluated is not None:
            radius_eval, center_x_eval, center_y_eval = already_evaluated
        else:
            radius_eval = self._radius.try_evaluate()
            center_x_eval = self.center_x.try_evaluate()
            center_y_eval = self.center_y.try_evaluate()
        if None not in (radius_eval, center_x_eval, center_y_eval):
            # TODO: check for all edges
            if x > center_x_eval + radius_eval:  # After the right edge
                return False
            if x < center_x_eval - radius_eval:  # Before the right edge
                return False
            if y > center_y_eval + radius_eval:
                return False
            if y < center_y_eval - radius_eval:
                return False
            return True

        else:
            raise ValueError(
                "This feature is only supported for Circles without any additional parameters")

    def is_inside(self, other_circle: "Circle") -> bool:
        if not isinstance(other_circle, Circle):
            raise TypeError(
                f"Invalid type '{type(other_circle)}'. Expected type 'circle'. ")
        my_radius_eval = self._radius.try_evaluate()
        my_center_x_eval = self.center_x.try_evaluate()
        my_center_y_eval = self.center_y.try_evaluate()
        other_radius_eval = other_circle._radius.try_evaluate()
        other_center_x_eval = other_circle.center_x.try_evaluate()
        other_center_y_eval = other_circle.center_y.try_evaluate()

        if None not in (my_radius_eval, my_center_x_eval, my_center_y_eval, other_radius_eval, other_center_x_eval,
                        other_center_y_eval):
            # Check for all edges
            if not other_circle.point_inside(self.top_edge, already_evaluated=(
                    other_radius_eval, other_center_x_eval, other_center_y_eval)):
                return False
            if not other_circle.point_inside(self.bottom_edge, already_evaluated=(
                    other_radius_eval, other_center_x_eval, other_center_y_eval)):
                return False
            if not other_circle.point_inside(self.right_edge, already_evaluated=(
                    other_radius_eval, other_center_x_eval, other_center_y_eval)):
                return False
            if not other_circle.point_inside(self.left_edge, already_evaluated=(
                    other_radius_eval, other_center_x_eval, other_center_y_eval)):
                return False
            return True
        else:
            raise ValueError("Can't determine whether a circle is inside another, when one or more of them "
                             "are expressed via parameters")

    def plot(self, fig=None, ax=None):
        radius_eval = self._radius.try_evaluate()
        center_x_eval = self.center_x.try_evaluate()
        center_y_eval = self.center_y.try_evaluate()
        if None in (radius_eval, center_x_eval, center_y_eval):
            raise ValueError(
                "Can only plot circles with real numbers (and not algebraic expressions)")
        circle1 = plt.Circle((center_x_eval, center_y_eval),
                             radius_eval, color='r', fill=False)
        if None in (fig, ax):
            fig, ax = plt.subplots()  # note we must use plt.subplots, not plt.subplot
        ax.add_patch(circle1)
        ax.set_aspect('equal', adjustable='datalim')
        ax.plot()  # Causes an autoscale update.
        plt.show()

    def to_lambda(self):
        warnings.warn("This is an experimental feature!")
        radius_evaluation = self._radius.try_evaluate()
        center_x_evaluation = self.center_x.try_evaluate()
        center_y_evaluation = self.center_y.try_evaluate()
        if None not in (radius_evaluation, center_x_evaluation, center_y_evaluation):
            # If we're just working with numbers
            return lambda x: (sqrt(abs(radius_evaluation ** 2 - (x - center_x_evaluation) ** 2)) + center_y_evaluation,
                              -sqrt(abs(radius_evaluation ** 2 - (x - center_x_evaluation) ** 2)) + center_y_evaluation)

        return lambda x: (Sqrt(Abs(self._radius ** 2 - (x - self.center_x) ** 2)) + self.center_y,
                          -Sqrt(Abs(self._radius ** 2 - (x - self.center_x) ** 2)) + self.center_y)

    @property
    def equation(self) -> str:
        x_part = _format_minus('x', self.center_x)

        if self.center_y == 0:
            y_part = "y^2"
        elif '+' in self.center_y.__str__() or '-' in self.center_y.__str__():
            y_part = f"(y-({self.center_y}))^2"
        else:
            y_part = f"(y-{self.center_y})^2"

        radius_eval = self._radius.try_evaluate()
        radius_part = f"{self._radius}^2" if radius_eval is None or (
            radius_eval is not None and radius_eval > 100) else f"{radius_eval ** 2}"
        return f"{x_part} + {y_part} = {radius_part}"

    def x_intersection(self):
        pass  # it's like the y intersection ...

    def _expression(self):
        x = Var('x')
        y = Var('y')
        return (x - self.center_x) ** 2 + (y - self.center_y) ** 2 - self._radius ** 2

    # This will work when poly_from_str() will be updated...
    def intersection(self, other):
        if isinstance(other, Circle):
            if self.has_parameters() or other.has_parameters():
                raise ValueError("This feature hasn't been implemented yet for Circle equations with additional"
                                 "parameters")
            else:
                initial_x = (self.center_x + other.center_x) / \
                    2  # The initial x is the average
                # between the x coordinates of the centers of the circles.
                initial_y = (self.center_y + other.center_y) / 2
                intersections = solve_poly_system([self._expression(), other._expression()],
                                                  initial_vals={'x': initial_x, 'y': initial_y})
                return intersections

    def has_parameters(self) -> bool:
        coefficient_eval = self._radius.try_evaluate()
        if coefficient_eval is None:
            return True
        center_x_eval = self.center_x.try_evaluate()
        if center_x_eval is None:
            return True
        center_y_eval = self.center_y.try_evaluate()
        if center_y_eval is None:
            return True
        return False

    def y_intersection(self, get_complex=False):
        center_x_eval = self.center_x.try_evaluate()
        center_y_eval = self.center_y.try_evaluate()
        radius_eval = self._radius.try_evaluate()
        if None not in (center_x_eval, radius_eval):
            # If those are numbers, we will be able to simplify the root
            # The equation is : +-sqrt(r**2 - a**2)
            # Then the inside of the root will be negative
            if abs(center_x_eval) > abs(radius_eval):
                if get_complex:
                    warnings.warn("Solving the intersections with complex numbers is still experimental..."
                                  "The issue will be resolved in later versions. Sorry!")
                    val = cmath.sqrt(radius_eval ** 2 - center_x_eval ** 2)
                    if center_y_eval is not None:
                        y1, y2 = val + center_y_eval, -val + center_y_eval
                    else:
                        # TODO: create a way to represent these...
                        y1, y2 = val + self.center_y, -val + self.center_y
                    # TODO: return a complex point object instead
                    return Point((0, y1)), Point((0, y2))
                return None
            else:  # Then we will find real solutions !
                val = sqrt(radius_eval ** 2 - center_x_eval ** 2)
                if val == 0:
                    if center_y_eval is not None:
                        return Point((0, center_y_eval))
                    else:
                        return Point((0, self.center_y))
                else:
                    if center_y_eval is not None:
                        y1, y2 = val + center_y_eval, -val + center_y_eval
                    else:
                        y1, y2 = val + self.center_y, -val + self.center_y
                    return Point((0, y1)), Point((0, y2))
        else:  # TODO: finish this part
            my_root = f"sqrt({_format_minus(self._radius, 0)} - {_format_minus(self.center_x, 0)})"

    def assign(self, **kwargs):
        self._radius.assign(**kwargs)
        self._center.coordinates[0].assign(**kwargs)
        self._center.coordinates[1].assign(**kwargs)

    def when(self, **kwargs):
        copy_of_self = self.__copy__()
        copy_of_self.assign(**kwargs)
        return copy_of_self

    def __copy__(self):
        return Circle(radius=self._radius.__copy__(), center=self.center.__copy__())

    def __call__(self, x: Union[int, float, IExpression], **kwargs):
        pass

    def __repr__(self):
        return f"Circle(radius={self._radius}, center={self._center})"

    def __str__(self):  # Get the equation or the repr ?
        return f"Circle(radius={self._radius}, center={self._center})"
