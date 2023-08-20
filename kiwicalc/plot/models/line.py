class Line2D(IPlottable):
    def __init__(self, point1: Union[Point2D, Iterable], point2: Union[Point2D, Iterable], gen_copies=True):
        if isinstance(point1, Point2D):
            self._point1 = point1.__copy__() if gen_copies else point1
        elif isinstance(point1, Iterable):
            x, y = point1
            self._point1 = Point2D(x, y)
        else:
            raise TypeError(
                f"Invalid type for param 'point1' when creating a Line object.")

        if isinstance(point2, Point2D):
            self._point2 = point2.__copy__() if gen_copies else point2
        elif isinstance(point2, Iterable):
            x, y = point2
            self._point2 = Point2D(x, y)
        else:
            raise TypeError(
                f"Invalid type for param 'point2' when creating a Line object.")

    def middle(self):
        return Point2D((self._point1.x + self._point2) / 2, (self._point1.y + self._point2.y) / 2)

    def length(self):
        inside_root = (self._point1.x - self._point2.x) ** 2 + \
            (self._point1.y - self._point2.y) ** 2
        if isinstance(inside_root, (int, float)):
            return sqrt(inside_root)

    @property
    def slope(self):
        x1, x2 = self._point1.x, self._point2.x
        y1, y2 = self._point1.y, self._point2.y
        numerator, denominator = y2 - y1, x2 - x1
        if denominator is None:
            warnings.warn(
                "There's no slope for a single x value with two y values.")
            return None
        return numerator / denominator

    @property
    def free_number(self):
        m = self.slope
        if m is None:
            warnings.warn(
                "There's no free number for a single x value with two y values.")
            return None
        return self._point1.y - self._point1.x * m

    def equation(self):
        m = self.slope
        if m is None:
            warnings.warn(
                "There's no slope for a single x value with two y values.")
            return None
        b = self._point1.y - self._point1.x * m
        m_str = format_coefficient(m)
        b_str = format_free_number(b)
        return f"{m_str}x{b_str}"

    def to_lambda(self):
        m = self.slope
        if m is None:
            warnings.warn(
                "Cannot generate a lambda expression for a single x value with two y values.")
            return None
        b = self._point1.y - self._point1.x * m
        return lambda x: m * x + b

    def intersection(self):  # TODO: implement it.......
        pass

    def plot(self, start: float = -6, stop: float = 6, step: float = 0.3, ymin: float = -10,
             ymax: float = 10, title: str = None, formatText: bool = False,
             show_axis: bool = True, show: bool = True, fig=None, ax=None, values=None):
        my_lambda = self.to_lambda()
        if my_lambda is None:
            pass  # TODO: implement it.
        plot_function(my_lambda,
                      start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title,
                      show_axis=show_axis, show=show, fig=fig, formatText=formatText, ax=ax,
                      values=values)

    def scatter(self, start: float = -10, stop: float = 10,
                step: float = 0.05, ymin: float = -10, ymax: float = 10, title=None, show_axis=True, show=True,
                fig=None, ax=None, formatText=True, values=None):

        lambda_expression = self.to_lambda()
        if not lambda_expression:
            pass  # TODO: implement it

        if title is None:
            title = self.__str__()

        scatter_function(lambda_expression, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title,
                         show_axis=show_axis, show=show, fig=fig, ax=ax, values=values)
