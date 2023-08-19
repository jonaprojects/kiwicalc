from abc import ABC, abstractmethod


class IExpression(ABC):

    @abstractmethod
    def assign(self, **kwargs):
        pass

    def when(self, **kwargs):
        copy_of_self = self.__copy__()
        copy_of_self.assign(**kwargs)
        return copy_of_self

    @abstractmethod
    def try_evaluate(self):
        pass

    @property
    @abstractmethod
    def variables(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __iadd__(self, other):
        pass

    def __add__(self, other: "Union[int, float, IExpression]"):
        return self.__copy__().__iadd__(other)

    def __radd__(self, other: "Union[int, float, IExpression]"):
        return self.__copy__().__iadd__(other)

    @abstractmethod
    def __isub__(self, other):
        pass

    def __sub__(self, other: "Union[int, float, IExpression]"):
        return self.__copy__().__isub__(other)

    @abstractmethod
    def __imul__(self, other):
        pass

    def __mul__(self, other: "Union[int, float, IExpression]"):
        return self.__copy__().__imul__(other)

    def __rmul__(self, other: "Union[int, float, IExpression]"):
        return self.__copy__().__imul__(other)

    @abstractmethod
    def __itruediv__(self, other):
        pass

    def __truediv__(self, other: "Union[int, float, IExpression]"):
        return self.__copy__().__itruediv__(other)

    @abstractmethod
    def __ipow__(self, other):
        pass

    def __pow__(self, other: "Union[int, float, IExpression]"):
        return self.__copy__().__ipow__(other)

    @abstractmethod
    def __neg__(self):
        pass

    @abstractmethod
    def __copy__(self):
        pass

    @abstractmethod
    def simplify(self):
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __ne__(self, other) -> bool:
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @staticmethod
    @abstractmethod
    def from_dict(given_dict: dict):
        pass

    def __abs__(self):
        return Abs(self)

    def __rpow__(self, other: "Union[IExpression, int, float]"):
        my_evaluation = self.try_evaluate()
        if my_evaluation is not None:
            return other ** my_evaluation
        return Exponent(other, self)

    def python_syntax(self, format_abs=True, format_factorial=True):
        return formatted_expression(self.__str__(), variables=self.variables, format_abs=format_abs,
                                    format_factorial=format_factorial)

    def to_lambda(self, variables=None, constants=tuple(), format_abs=True, format_factorial=True):
        if variables is None:
            variables = self.variables
        return to_lambda(self.python_syntax(), variables, constants, format_abs=format_abs,
                         format_factorial=format_factorial)

    def reinman(self, a: float, b: float, N: int):
        return reinman(self.to_lambda(), a, b, N)

    def trapz(self, a: float, b: float, N: int):
        return trapz(self.to_lambda(), a, b, N)

    def simpson(self, a: float, b: float, N: int):
        return simpson(self.to_lambda(), a, b, N)

    def secant(self, n_0: float, n_1: float, epsilon: float = 0.00001, nmax: int = 10_000):
        return secant_method(self.to_lambda(), n_0, n_1, epsilon, nmax)

    def bisection(self, a: float, b: float, epsilon: float = 0.00001, nmax=100000):
        return bisection_method(self.to_lambda(), a, b, epsilon, nmax)

    def plot(self, start: float = -6, stop: float = 6, step: float = 0.3, ymin: float = -10,
             ymax: float = 10, title: str = None, formatText: bool = False,
             show_axis: bool = True, show: bool = True, fig=None, ax=None, values=None, meshgrid=None):
        variables = self.variables
        num_of_variables = len(variables)
        if num_of_variables == 1:
            plot_function(self.to_lambda(),
                          start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title,
                          show_axis=show_axis, show=show, fig=fig, formatText=formatText, ax=ax,
                          values=values)
        elif num_of_variables == 2:
            plot_function_3d(given_function=self.to_lambda(),
                             start=start, stop=stop, meshgrid=meshgrid)
        else:
            raise ValueError(
                f"Cannot plot an expression with {num_of_variables} variables")

    def scatter(self, start: float = -10, stop: float = 10,
                step: float = 0.05, ymin: float = -10, ymax: float = 10, title=None, show_axis=True, show=True,
                fig=None, ax=None, values=None):

        lambda_expression = self.to_lambda()
        num_of_variables = len(self.variables)
        if title is None:
            title = self.__str__()
        if num_of_variables == 0:  # TODO: plot this in a number axis
            raise ValueError("Cannot plot a polynomial with 0 variables_dict")
        elif num_of_variables == 1:
            scatter_function(lambda_expression, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax,
                             show_axis=show_axis, show=show, fig=fig, ax=ax, values=values, title=title)
        elif num_of_variables == 2:
            scatter_function_3d(lambda_expression, start=start, stop=stop, step=step,
                                title=title)  # TODO: update the parameters
        else:
            raise ValueError(
                "Cannot plot a function with more than two variables_dict (As for this version)")

    def to_json(self):
        return json.dumps(self.to_dict())

    def export_json(self, path: str):
        with open(path, 'w') as json_file:
            json_file.write(self.to_json())

    def to_Function(self) -> "Optional[Function]":
        try:
            return Function(self.__str__())
        except:
            return None
