class Sequence(ABC):

    @property
    @abstractmethod
    def first(self):
        pass

    @abstractmethod
    def in_index(self, index: int) -> float:
        pass

    @abstractmethod
    def index_of(self, item: float) -> float:
        pass

    @abstractmethod
    def sum_first_n(self, n: int) -> float:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    def range(self, start: int, stop: int):
        return (self.in_index(current_index) for current_index in range(start, stop))

    def product_in_range(self, start: int, end: int):
        return reduce(lambda a, b: a * b, (self.in_index(i) for i in range(start, end)))

    def product_first_n(self, end: int):
        return self.product_in_range(0, end)

    def sum_in_range(self, start: int, end: int):  # TODO: improve this
        return sum(self.in_index(current_index) for current_index in range(start, end))

    def __contains__(self, item: float):
        index = self.index_of(item)
        return index > 0

    def __getitem__(self, item):
        if isinstance(item, slice):  # handling slicing
            if item.start == 0:
                warnings.warn(
                    "Sequence indices start from 1 and not from 0, skipped to 1")
                start = 1
            else:
                start = item.start
            step = 1 if item.step is None else item.step
            return [self.in_index(i) for i in range(start, item.stop + 1, step)]
        elif isinstance(item, int):
            return self.in_index(item)

    def __generate_data(self, start: int, stop: int, step: int):
        return list(range(start, stop, step)), [self.in_index(index) for index in range(start, stop, step)]

    def plot(self, start: int, stop: int, step: int = 1, show=True):
        axes, y_values = self.__generate_data(start, stop, step)
        plt.plot(axes, y_values)
        if show:
            plt.show()
