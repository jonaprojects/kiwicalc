class PointCollection:
    def __init__(self, points: Iterable = ()):
        self._points = []
        for point in points:
            if isinstance(point, (Iterable, int, float)) and not isinstance(point, Point):
                if isinstance(point, Iterable):
                    num_of_coordinates = len(point)
                    if num_of_coordinates == 2:
                        point = Point2D(point[0], point[1])
                    elif num_of_coordinates == 3:
                        point = Point3D(point[0], point[1], point[2])
                    else:
                        point = Point(point)

            if isinstance(point, Point):
                self._points.append(point)
            else:
                raise TypeError(f"encountered invalid type '{type(point)}' of value {point} when creating a "
                                f"PointCollection object. ")

    @property
    def points(self):
        return self._points

    def coords_at(self, index: int):
        """ returns all of the coordinates of the points in the specified index. For example, for an index of 0,
         a list of all of the x coordinates will be returned.
        """
        try:
            return [point.coordinates[index] for point in self._points]
        except IndexError:
            raise IndexError(
                f"The PointCollection object doesn't have points with coordinates of index {index}")

    def add_point(self, point: Point):
        self._points.append(point)

    def remove_point(self, index: int):
        del self._points[index]

    def max_coord_at(self, index: int):
        """
        Fetch the biggest coordinate at the specified index. For example, for the index of 0, the biggest
        x value will be returned.
        """
        try:
            return max(self.coords_at(index))
        except IndexError:
            raise IndexError(
                f"The PointCollection object doesn't have points with coordinates of index {index}")

    def min_coord_at(self, index: int):
        """
        Fetch the smallest coordinate at the specified index. For example, for the index of 0, the smallest
        x value will be returned.
        """
        try:
            return min(self.coords_at(index))
        except IndexError:
            raise IndexError(
                f"The PointCollection object doesn't have points with coordinates of index {index}")

    def avg_coord_at(self, index: int):
        """Returns the average value for the points' coordinates at the specified index. For example,
        for an index of 0, the average x value of the points will be returned, for an index of 1, the average y value
        of all of the dots will be returned, and so on.
        """
        try:
            coords = self.coords_at(index)
            return sum(coords) / len(coords)
        except IndexError:
            raise IndexError(
                f"The PointCollection object doesn't have points with coordinates of index {index}")

    def longest_distance(self, get_points=False):
        """ Gets the longest distance between two dots in the collection"""  # TODO: improve with combinations
        if len(self._points) <= 1:
            return 0
        pairs = combinations(self._points, 2)
        p1, p2 = max(pairs, key=lambda p1, p2: sum(
            (coord1 - coord2) ** 2 for (coord1, coord2) in zip(p1.coordinates, p2.coordinates)))
        max_distance = sqrt(sum((coord1 - coord2) ** 2 for (coord1,
                            coord2) in zip(p1.coordinates, p2.coordinates)))
        if get_points:
            return max_distance, (p1, p2)
        return max_distance

    def shortest_distance(self, get_points=False):
        """ Gets the shortest distance between two dots in the collection"""
        if len(self._points) <= 1:
            return 0
        pairs = combinations(self._points, 2)
        p1, p2 = min(pairs, key=lambda p1, p2: sum(
            (coord1 - coord2) ** 2 for (coord1, coord2) in zip(p1.coordinates, p2.coordinates)))
        min_distance = sqrt(sum((coord1 - coord2) ** 2 for (coord1,
                            coord2) in zip(p1.coordinates, p2.coordinates)))
        if get_points:
            return min_distance, (p1, p2)
        return min_distance

    def scatter(self, show=True):  # Add limits of x and y
        dimensions = len(self._points[0].coordinates)
        x_values = self.coords_at(0)
        if dimensions == 1:
            min_value, max_value = min(x_values), max(x_values)
            plt.hlines(0, min_value, max_value)  # Draw a horizontal line
            plt.xlim(0.8 * min_value, 1.2 * max_value)
            plt.ylim(-1, 1)
            y = np.zeros(len(self._points))  # Make all y values the same
            # Plot a line at each location specified in a
            plt.plot(x_values, y, '|', ms=70)
            plt.axis('on')
        elif dimensions == 2:
            y_values = self.coords_at(1)
            plt.scatter(x_values, y_values)
        elif dimensions == 3:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.coords_at(0), self.coords_at(1), self.coords_at(2))
        # Use a heat map in order to represent 4D dots ( The color is the 4th dimension )
        elif dimensions == 4:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x, y, z, c = self.coords_at(0), self.coords_at(
                1), self.coords_at(2), self.coords_at(3)
            img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
            fig.colorbar(img)
        else:
            raise ValueError(
                f"Can only scatter in 1D, 2D, 3D and 4D dimension, But got {dimensions}D")
        if show:
            plt.show()

    def __str__(self):
        return ", ".join((point.__str__() for point in self._points))

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):
        if self.__index == len(self._points):
            raise StopIteration
        index = self.__index
        self.__index += 1
        return self._points[index]

    def __repr__(self):
        return f"PointCollection(({self.__str__()}))"


class Point1DCollection(PointCollection):
    def __init__(self, points: Iterable = ()):
        points_list = list(points)
        if any(len(point) != 1 for point in points_list):
            raise ValueError("All points must have 1 coordinates.")
        super(Point1DCollection, self).__init__(points_list)


class Point2DCollection(PointCollection):
    def __init__(self, points: Iterable = ()):
        points_list = list(points)
        if any(len(point) != 2 for point in points_list):
            raise ValueError("All points must have 2 coordinates.")
        super(Point2DCollection, self).__init__(points_list)

    def plot_regression(self, show=True):
        if len(self._points[0].coordinates) == 2:
            linear_function = self.linear_regression()  # a lambda expression is returned
            min_x, max_x = self.min_coord_at(0) - 50, self.max_coord_at(0) + 50
            plt.plot((min_x, max_x), (linear_function(
                min_x), linear_function(max_x)))
            if show:
                plt.show()

    @property
    def x_values(self):
        return self.coords_at(0)

    @property
    def y_values(self):
        return self.coords_at(1)

    def add_point(self, point: Point2D):
        if len(point) != 2:
            raise ValueError("Can only accept 2D points to Point2DCollection")
        self._points.append(point)

    def sum(self):
        return Point2D(sum(self.x_values), sum(self.y_values))

    def scatter(self, show=True):
        x_values, y_values = self.coords_at(0), self.coords_at(1)
        plt.scatter(x_values, y_values)
        if show:
            plt.show()

    def scatter_with_regression(self, show=True):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('equal')
        # And a corresponding grid
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        ax.spines['bottom'].set_position('zero')
        ax.spines['left'].set_position('zero')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self.scatter(show=show)
        self.plot_regression(show=show)

    def linear_regression(self, get_tuple=False):
        if len(self._points[0].coordinates) == 2:
            return linear_regression([point.coordinates[0] for point in self._points],
                                     [point.coordinates[1] for point in
                                      self._points], get_tuple)


class Point3DCollection(PointCollection):
    def __init__(self, points: Iterable = ()):
        points_list = list(points)
        if any(len(point) != 3 for point in points_list):
            raise ValueError("All points must have 3 coordinates.")
        super(Point3DCollection, self).__init__(points_list)

    @property
    def x_values(self):
        return self.coords_at(0)

    @property
    def y_values(self):
        return self.coords_at(1)

    @property
    def z_values(self):
        return self.coords_at(2)

    def sum(self):
        return Point3D(sum(self.x_values), sum(self.y_values), sum(self.z_values))

    def scatter(self, show=True):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x_values, self.y_values, self.z_values)
        if show:
            plt.show()

    def add_point(self, point: Point3D):
        if len(point) != 3:
            raise ValueError("Can only accept 3D points to Point3DCollection")
        self._points.append(point)


class Point4DCollection(PointCollection):
    def __init__(self, points: Iterable = ()):
        points_list = list(points)
        if any(len(point) != 4 for point in points_list):
            raise ValueError("All points must have 4 coordinates.")
        super(Point4DCollection, self).__init__(points_list)

    @property
    def x_values(self):
        return self.coords_at(0)

    @property
    def y_values(self):
        return self.coords_at(1)

    @property
    def z_values(self):
        return self.coords_at(2)

    @property
    def c_values(self):
        return self.coords_at(3)

    def sum(self):
        return Point4D(sum(self.x_values), sum(self.y_values), sum(self.z_values), sum(self.c_values))

    def scatter(self, show=True):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z, c = self.coords_at(0), self.coords_at(
            1), self.coords_at(2), self.coords_at(3)
        img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
        fig.colorbar(img)
        if show:
            plt.show()
