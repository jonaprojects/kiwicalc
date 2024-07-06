from kiwicalc.plotting.plot import plot_vector_2d


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


def _get_limits_vectors_2d(vectors):
    """Internal method: find the edge values for the scope of the 2d frame"""
    min_x = min(min(
        vector.start_coordinate[0], vector.end_coordinate[0]) for vector in vectors) * 1.05
    max_x = max(max(
        vector.start_coordinate[0], vector.end_coordinate[0]) for vector in vectors) * 1.05
    min_y = min(min(
        vector.start_coordinate[1], vector.end_coordinate[1]) for vector in vectors) * 1.05
    max_y = max(max(
        vector.start_coordinate[1], vector.end_coordinate[1]) for vector in vectors) * 1.05
    return min_x, max_x, min_y, max_y