class VectorCollection:
    def __init__(self, *vectors):
        self.__vectors = []
        for vector in vectors:
            if isinstance(vector, Vector):
                self.__vectors.append(vector)
            elif isinstance(vector, Iterable):
                self.__vectors.append(Vector(vector))
            else:
                raise TypeError(
                    f"Encountered invalid type {type(vector)} while building a vector collection.")

    @property
    def vectors(self):
        return self.__vectors

    @property
    def num_of_vectors(self):
        return len(self.__vectors)

    @vectors.setter
    def vectors(self, vectors):
        if isinstance(vectors, (list, set, tuple, Vector)):
            vectors = VectorCollection(vectors)

        if isinstance(vectors, VectorCollection):
            self.__vectors = vectors
        else:
            raise TypeError(
                f"Unexpected type {type(vectors)} in the setter property of vectors in class VectorCollection"
                f".\nExpected types VectorCollection, Vector, tuple, list, set")

    def append(self,
               vector: "Union[Vector, Iterable[Union[Vector, IExpression, VectorCollection, int, float, Iterable]], VectorCollection]"):
        """ Append vectors to the collection of vectors """
        if isinstance(vector, Vector):  # if the parameter is a vector
            self.__vectors.append(vector)

        elif isinstance(vector, VectorCollection):
            self.__vectors.extend(vector)

        elif isinstance(vector, Iterable) and not isinstance(vector, IExpression):
            # if the parameter is an Iterable object
            for item in vector:
                if isinstance(item, Vector):
                    self.__vectors.append(item)
                elif isinstance(item, Iterable):
                    self.__vectors.append(Vector(item))
                elif isinstance(item, VectorCollection):
                    self.__vectors.extend(item)
                else:
                    raise TypeError(
                        f"Invalid type {type(vector)} for appending into a VectorCollection")

        else:
            raise TypeError(
                f"Invalid type {type(vector)} for appending into a VectorCollection")

    def plot(self):  #
        num_of_vectors = len(self.__vectors)
        if num_of_vectors > 0:
            if len(self.__vectors[0].start_coordinate) == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                for vector in self.__vectors:
                    start = (
                        vector.start_coordinate[0], vector.start_coordinate[1], vector.start_coordinate[2])
                    end = (
                        vector.end_coordinate[0], vector.end_coordinate[1], vector.end_coordinate[2])
                    plot_vector_3d(start, end, fig=fig, ax=ax, show=False)
                min_x, max_x, min_y, max_y, min_z, max_z = _get_limits_vectors_3d(
                    self.__vectors)
                ax.set_xlim([min_x, max_x])
                ax.set_ylim([min_y, max_y])
                ax.set_zlim([min_z, max_z])
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                for vector in self.__vectors:
                    vector.plot(show=False, fig=fig, ax=ax)

                min_x, max_x, min_y, max_y = _get_limits_vectors_2d(
                    self.__vectors)
                ax.set_xlim([min_x, max_x])
                ax.set_ylim([min_y, max_y])
        plt.show()

    # TODO: should modify it or delete or leave it like this?
    def filter(self, predicate: Callable[[Any], bool] = lambda x: bool(x)):
        return filter(predicate, self.__vectors)

    def map(self, func: Callable):
        return map(func, self.__vectors)

    def longest(self, get_index=False, remove=False):
        """
        returns the longest vector in the collection

        :param get_index: if True, returns a tuple: (index,longest_vector)
        :param remove: if True, removes the longest vector from the collection
        :return: depends whether get_index evaluates to True or False
        """
        longest_vector = max(self.__vectors, key=operator.attrgetter(
            '_Vector__direction_vector'))
        if not (get_index or remove):
            return longest_vector
        index = self.__vectors.index(longest_vector)
        if remove:
            self.__vectors.pop(index)
        if get_index:
            return index, longest_vector
        return longest_vector

    def shortest(self, get_index=False, remove=False):
        """
        returns the shortest vector in the collection

        :param get_index: if True, returns a tuple: (index,shortest_vector)
        :param remove: if True, removes the shortest vector from the collection
        :return: depends whether get_index evaluates to True or False
        """
        shortest_vector = min(self.__vectors, key=operator.attrgetter(
            '_Vector__direction_vector'))
        if not (get_index or remove):
            return shortest_vector
        index = self.__vectors.index(shortest_vector)
        if remove:
            self.__vectors.pop(index)
        if get_index:
            return index, shortest_vector
        return shortest_vector

    def find(self, vec: Vector):
        for index, vector in enumerate(self.__vectors):
            if vector.__eq__(vector) or vector is vec:
                return index
        return -1

    def nlongest(self, n: int):
        """returns the n longest vector for an integer n"""
        return [self.longest(remove=True) for _ in range(n)]

    def nshortest(self, n: int):
        """returns the n shortest vector for an integer n"""
        return [self.shortest(remove=True) for _ in range(n)]

    def sort_by_length(self, reverse=False):
        self.__vectors.sort(
            key=lambda vector: vector.length(), reverse=reverse)

    def pop(self, index: int = -1):
        return self.__vectors.pop(index)

    def __iadd__(self, other: "Union[Vector, VectorCollection, Iterable]"):
        self.append(other)
        return self

    def __add__(self, other: "Union[Vector, VectorCollection, Iterable]"):
        return self.__copy__().__iadd__(other)

    def __radd__(self, other):
        return self.__copy__().__iadd__(other)

    def __imul__(self, other):
        if isinstance(other, Vector):
            pass
        elif isinstance(other, VectorCollection):
            pass
        else:
            pass

    def __mul__(self, other):
        return self.__copy__().__imul__(other)

    def __itruediv__(self, other: Union[int, float, IExpression]):
        if not isinstance(other, (int, float, IExpression)):
            raise TypeError(f"Invalid type for dividing a VectorCollection object: {type(other)}. Expected a number"
                            f"or an algebraic expression.")
        if other == 0:
            raise ValueError("Cannot divide a VectorCollection object by 0")

        for i in range(len(self.__vectors)):
            self.__vectors[i] /= other

    def __truediv__(self, other):
        return self.__copy__().__itruediv__(other)

    def __bool__(self):
        return bool(self.__vectors)

    def to_matrix(self):
        return Matrix([[copy_expression(expression) for expression in vector.direction] for vector in self.__vectors])

    # TODO: check this method ..
    # TODO: use other type hint than Iterable
    def __eq__(self, other: "Union[Vector, VectorCollection, Iterable]"):
        if other is None:
            return False
        if not isinstance(other, (Vector, VectorCollection)):
            if isinstance(other, Iterable):
                if isinstance(other[0], Iterable):
                    try:
                        other = VectorCollection(other)
                    except (ValueError, TypeError):
                        try:
                            other = Vector(other)
                        except (ValueError, TypeError):
                            raise ValueError(
                                "Invalid value for equating VectorCollection objects.")
                else:
                    try:
                        other = Vector(other)
                    except (ValueError, TypeError):
                        raise ValueError(
                            "Invalid value for equating VectorCollection objects.")
            else:
                raise TypeError(
                    f"Invalid type '{type(other)}' for equating VectorCollection objects.")

        if isinstance(other, Vector) and len(self.__vectors) == 1:
            return self.__vectors[0] == other
            # comparison between a vector to a single item list should return true
        elif isinstance(other, VectorCollection):
            if len(self.__vectors) != len(other.__vectors):
                return False
            for vec in self.__vectors:
                if other.__vectors.count(vec) != self.__vectors.count(vec):
                    return False
            return True
        else:
            raise TypeError(
                f"Invalid type {type(other)} for equating VectorCollection objects.")

    def __ne__(self, other: "Union[Vector, VectorCollection, Iterable]"):
        return not self.__eq__(other)

    def __getitem__(self, item):
        return self.__vectors.__getitem__(item)

    def __setitem__(self, key, value):
        return self.__vectors.__setitem__(key, value)

    def __delitem__(self, key):
        return self.__delitem__(key)

    def __copy__(self):
        return VectorCollection(self.__vectors)

    def __contains__(self, other: Vector):
        if isinstance(other, Vector):
            return bool([vector for vector in self.__vectors if vector.__eq__(other)])

    def __iter__(self):
        self.__current_index = 0
        return self

    def __next__(self):
        if self.__current_index < len(self.__vectors):
            x = self.__vectors[self.__current_index]
            self.__current_index += 1
            return x
        else:
            raise StopIteration

    def __len__(self):
        """ number of vectors that the collection contains"""
        return len(self.__vectors)

    def total_number_of_items(self):
        """ Total number of items in all of the vectors. """
        return sum(len(vector) for vector in self.__vectors)


def surface_from_str(input_string: str, get_coefficients=False):
    first_side, second_side = input_string.split('=')
    first_coefficients = re.findall(number_pattern, first_side)

    for index in range(0, 4 - len(first_coefficients), 1):  # format it to be 4 coefficients
        first_coefficients.append('0')
    second_coefficients = re.findall(number_pattern, second_side)
    for index in range(0, 4 - len(second_coefficients), 1):  # format it to be 4 coefficients
        second_coefficients.append('0')

    for first_index, second_value in zip(range(len(first_coefficients)), second_coefficients):
        first_coefficients[first_index] = float(
            first_coefficients[first_index]) - float(second_value)
    if get_coefficients:
        return first_coefficients
    return Surface(first_coefficients)
