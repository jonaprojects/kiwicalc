class GeometricSeq(Sequence, IPlottable):
    """
    A class that represents a geometric sequence, namely, a sequence in which every item can be
    multiplied by a constant (the ratio of the sequence) to reach the next item.
    """

    def __init__(self, first_numbers: Union[tuple, list, set, str, int, float], ratio: float = None):
        """Create a new GeometricSeq object"""

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
                raise ValueError("GeometricSeq.__init__(): Cannot accept an empty collection for parameter "
                                 "'first_numbers'")
            if any(number == 0 for number in first_numbers):
                raise ValueError(
                    "GeometricSeq.__init__(): Zeroes aren't allowed in geometric sequences")
            self.__first = first_numbers[0]
            if ratio is not None:
                self.__ratio = ratio
                return
            # We get here only if ratio is None
            if len(first_numbers) == 1:
                raise ValueError("GeometricSeq.__init__(): Please Enter more initial values, or specify the ratio of "
                                 "the sequence.")
            self.__ratio = first_numbers[1] / first_numbers[0]

        else:
            raise TypeError(f"GeometricSeq.__init__():"
                            f"Invalid type {type(first_numbers)} for parameter 'first_numbers'. Expected types"
                            f" 'tuple', 'list', 'set', 'str', 'int', 'float' ")

    @property
    def first(self):
        return self.__first

    @property
    def ratio(self):
        return self.__ratio

    def in_index(self, index: int) -> float:
        return self.__first * pow(self.__ratio, (index - 1))

    def index_of(self, item: float) -> float:
        result = log(item / self.__first, self.__ratio) + 1
        if not result.is_integer():
            return -1
        return result

    def sum_first_n(self, n: int) -> float:
        return self.__first * (self.__ratio ** n - 1) / (self.__ratio - 1)

    def __repr__(self):
        return f"Sequence(first_numbers=({self.__first},),ratio={self.__ratio})"

    def __str__(self):
        return f"{self.__first}, {self.in_index(2)}, {self.in_index(3)} ... (ratio = {self.__ratio})"
