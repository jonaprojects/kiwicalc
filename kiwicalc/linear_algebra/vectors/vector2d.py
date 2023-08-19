class Vector2D(Vector, IPlottable):
    def __init__(self, x, y, start_coordinate=None, end_coordinate=None):
        if start_coordinate is not None:
            if len(start_coordinate) != 2:
                raise ValueError(
                    f"Vector2D object can only receive 2D coordinates: got wrong 'start_coordinate' param")
        if end_coordinate is not None:
            if len(end_coordinate) != 2:
                raise ValueError(
                    f"Vector2D object can only receive 2D coordinates: got wrong 'end_coordinate' param")

        super().__init__(direction_vector=(x, y), start_coordinate=start_coordinate,
                         end_coordinate=end_coordinate)

    @property
    def x_step(self):
        return self._direction_vector[0]

    @property
    def y_step(self):
        return self._direction_vector[1]

    def plot(self, show=True, arrow_length_ratio: float = 0.05):
        plot_vector_2d(
            self._start_coordinate[0], self._start_coordinate[1], self._direction_vector[0],
            self._direction_vector[1], show=show)
