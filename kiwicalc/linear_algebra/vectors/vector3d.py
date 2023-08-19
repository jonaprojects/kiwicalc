class Vector3D(Vector, IPlottable):
    def __init__(self, x, y, z, start_coordinate=None, end_coordinate=None):
        if start_coordinate is not None:
            if len(start_coordinate) != 3:
                raise ValueError(
                    f"Vector3D object can only receive 3D coordinates: got wrong 'start_coordinate' param")
        if end_coordinate is not None:
            if len(end_coordinate) != 3:
                raise ValueError(
                    f"Vector3D object can only receive 3D coordinates: got wrong 'end_coordinate' param")

        super(Vector3D, self).__init__(direction_vector=(x, y, z), start_coordinate=start_coordinate,
                                       end_coordinate=end_coordinate)

    @property
    def x_step(self):
        return self._direction_vector[0]

    @property
    def y_step(self):
        return self._direction_vector[1]

    @property
    def z_step(self):
        return self._direction_vector[2]

    def plot(self, show=True, arrow_length_ratio: float = 0.05, fig=None, ax=None):
        u, v, w = self._direction_vector[0], self._direction_vector[1], self._direction_vector[2]
        start_x, start_y, start_z = self._start_coordinate[0], self._start_coordinate[1], self._start_coordinate[
            2]
        plot_vector_3d(
            (start_x, start_y, start_z), (u, v, w), arrow_length_ratio=arrow_length_ratio, show=show, fig=fig, ax=ax)
