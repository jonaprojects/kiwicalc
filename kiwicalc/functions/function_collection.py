class FunctionCollection(IPlottable, IScatterable):
    def __init__(self, *functions, gen_copies=False):
        self._functions = []
        for func in functions:
            # lambda expressions might not work here
            if isinstance(func, str) or is_lambda(func):
                self._functions.append(Function(func))
            elif isinstance(func, Function):
                if gen_copies:
                    self._functions.append(func.__copy__())
                else:
                    self._functions.append(func)
            elif isinstance(func, (FunctionChain, FunctionCollection)):
                if all(isinstance(f, Function) for f in func):
                    for f in func:
                        if gen_copies:
                            self._functions.append(f.__copy__())
                        else:
                            self._functions.append(f)
                else:
                    pass  # Recursive Algorithm to break down anything, or raise an Error

    @property
    def functions(self):
        return self._functions

    @property
    def num_of_functions(self):
        return self._functions

    @property
    def variables(self):
        variables_set = set()
        for func in self._functions:
            variables_set.update(func.variables)
        return variables_set

    @property
    def num_of_variables(self):
        if not self._functions:
            return 0
        return len(self.variables)

    def clear(self):
        self._functions = []

    def is_empty(self):
        return not self._functions

    def add_function(self, func: Union[Function, str]):
        if isinstance(func, Function):
            self._functions.append(func)
        elif isinstance(func, str):
            self._functions.append(Function(func))
        else:
            raise TypeError(
                f"Invalid type {type(func)}. Allowed types for this method are 'str' and 'Function'")

    def extend(self, functions: Iterable[Union[Function, str]]):
        for function in functions:
            if isinstance(function, str):
                function = Function(function)
            self._functions.append(function)

    def values(self, *args, **kwargs):
        pass

    def random_function(self):
        return random.choice(self._functions)

    def random_value(self, a: Union[int, float], b: Union[int, float], mode='int'):
        my_random_function = self.random_function()
        if a > b:
            a, b = b, a
        num_of_variables = my_random_function.num_of_variables
        if mode == 'float':
            parameters = [random.uniform(a, b)
                          for _ in range(num_of_variables)]
        elif mode == 'int':
            parameters = [random.randint(a, b)
                          for _ in range(num_of_variables)]
        else:
            raise ValueError(f"invalid mode {mode}: expected 'int' or 'float'")
        return my_random_function(*parameters)

    def derivatives(self):
        if any(func.num_of_variables != 1 for func in self._functions):
            raise ValueError(
                "All functions must have exactly 1 parameter (For this version)")
        return [func.derivative() for func in self._functions]

    def filter(self, predicate: Callable[[Function], bool]):
        return filter(predicate, self._functions)

    def __len__(self):
        return len(self._functions)

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):
        if self.__index < len(self._functions):
            value = self._functions[self.__index]
            self.__index += 1
            return value
        else:
            raise StopIteration

    def __getitem__(self, item):
        return FunctionCollection(*(self._functions.__getitem__(item)))

    def __str__(self):
        return "\n".join((f"{index + 1}. {function.__str__()}" for index, function in enumerate(self._functions)))

    def plot(self, start: float = -10, stop: float = 10,
             step: float = 0.01, ymin: float = -10, ymax: float = 10, text=None, show_axis=True, show=True):
        plot_functions(self._functions, start, stop, step,
                       ymin, ymax, text, show_axis, show)

    def scatter(self, start: float = -10, stop: float = 10,
                step: float = 0.01, ymin: float = -10, ymax: float = 10, text=None, show_axis=True, show=True):
        scatter_functions(self._functions, start, stop, step,
                          ymin, ymax, text, show_axis, show)
