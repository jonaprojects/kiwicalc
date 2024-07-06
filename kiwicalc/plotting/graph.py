from abc import abstractmethod, ABC
from matplotlib import pyplot as plt
from plot import draw_axis, scatter_functions_3d, plot_functions_3d, create_grid
from typing import Iterable, Union, Callable
from models import *
from auxiliary import decimal_range
from ..algebra.models.IExpression import IExpression


class Graph:
    """A base class for creating graph systems"""

    def __init__(self, objs, fig, ax):
        self._items = [obj for obj in objs]
        self._fig, self._ax = fig, ax

    @property
    def items(self):
        return self._items

    def is_empty(self):
        return not self._items

    def add(self, obj):
        self._items.append(obj)

    def plot(self):
        raise NotImplementedError

    def scatter(self):
        raise NotImplementedError


class Graph2D(Graph):
    """ Create a 2D graph system. 
    Initialize the Graph2D object, add items, and then plot them together in 2D Space
    """

    def __init__(self, objs: Iterable[IPlottable] = tuple()):
        fig, ax = create_grid()
        super(Graph2D, self).__init__(objs, fig, ax)

    def plot(self, start: float = -10, stop: float = 10,
             step: float = 0.01, ymin: float = -10, ymax: float = 10, text=None, show_axis=True, show=True,
             formatText=False, values=None):
        if values is None:
            values = list(decimal_range(start=start, stop=stop, step=step))
        if show_axis:
            draw_axis(self._ax)
        if text is None:
            if len(self._items) >= 3:
                graph_title = ", ".join([obj.__str__()
                                        for obj in self._items[:3]]) + "..."
            else:
                graph_title = ", ".join(obj.__str__() for obj in self._items)
        else:
            graph_title = text
        for obj in self._items:
            if isinstance(obj, (IExpression, Function)):
                obj.plot(start=start, stop=stop, step=step, ymin=ymin, ymax=ymax,
                         text=graph_title, show_axis=show_axis, fig=self._fig, ax=self._ax, show=False,
                         formatText=formatText, values=values)
            elif isinstance(obj, Circle):
                obj.plot(fig=self._fig, ax=self._ax)
        if show:
            plt.show()


class Graph3D(Graph):
    """ Create a 3D graph system. 
    Initialize the Graph3D object, add items, and then plot them together in 3D Space
    """

    def __init__(self, objs=tuple()):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        super(Graph3D, self).__init__(objs, fig, ax)

    def plot(self, functions: Iterable[Union[Callable, str, IExpression]], start: float = -5, stop: float = 5,
             step: float = 0.1,
             xlabel: str = "X Values",
             ylabel: str = "Y Values", zlabel: str = "Z Values"):
        return plot_functions_3d(
            functions=functions, start=start, stop=stop, step=step, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel
        )

    def scatter(self, functions: Iterable[Union[Callable, str, IExpression]], start: float = -5, stop: float = 5,
                step: float = 0.1,
                xlabel: str = "X Values",
                ylabel: str = "Y Values", zlabel: str = "Z Values"):
        return scatter_functions_3d(
            functions=functions, start=start, stop=stop, step=step, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel
        )
