class ArithmeticProg(Sequence):
    """A class for representing arithmetic progressions. for instance: 2, 4, 6, 8, 10 ..."""

    def __init__(self, first_numbers: Union[tuple, list, set, str, int, float], difference: float = None):
        if isinstance(first_numbers, str):
            if ',' in first_numbers:
                first_numbers = [float(i)
                                 for i in tuple(first_numbers.split(','))]
            else:
                first_numbers = [float(i)
                                 for i in tuple(first_numbers.split(' '))]
        elif isinstance(first_numbers, (int, float)):
            first_numbers = [first_numbers]

        if isinstance(first_numbers, (tuple, list, set)):
            if not first_numbers:
                raise ValueError("ArithmeticProg.__init__(): Cannot accept an empty collection for parameter "
                                 "'first_numbers'")
            self.__first = first_numbers[0]
            if difference is not None:
                self.__difference = difference
                return
            # We get here only if the difference is None
            if len(first_numbers) == 1:
                raise ValueError("ArithmeticProg.__init__(): Please Enter more initial values,"
                                 " or specify the difference of the sequence.")
            self.__difference = first_numbers[1] - first_numbers[0]

        else:
            raise TypeError(f"ArithmeticProg.__init__():"
                            f"Invalid type {type(first_numbers)} for parameter 'first_numbers'. Expected types"
                            f" 'tuple', 'list', 'set', 'str', 'int', 'float' ")

    @property
    def first(self):
        return self.__difference

    @property
    def difference(self):
        return self.__difference

    def in_index(self, index: int) -> float:
        return self.__first + self.__difference * (index - 1)

    def index_of(self, item: float) -> float:
        result = (item - self.__first) / self.__difference + 1
        if not result.is_integer():
            return -1
        return result

    def sum_first_n(self, n: int) -> float:
        return 0.5 * n * (2 * self.__first + (n - 1) * self.__difference)

    def __str__(self):
        return f"{self.__first}, {self.in_index(2)}, {self.in_index(3)} ... (difference = {self.__difference})"

    def __repr__(self):
        return f"Sequence(first_numbers=({self.__first},),difference={self.__difference})"
