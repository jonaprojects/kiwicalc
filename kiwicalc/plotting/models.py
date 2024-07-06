from abc import abstractmethod, ABC


class IPlottable(ABC):
    @abstractmethod
    def plot(self):
        pass


class IScatterable(ABC):
    @abstractmethod
    def scatter(self):
        pass


class IPlottable3D(ABC):
    @abstractmethod
    def plot3d(self):
        pass


class IScatterable3D(ABC):
    @abstractmethod
    def scatter3d(self):
        pass
