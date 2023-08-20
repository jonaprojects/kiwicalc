class FunctionChain(FunctionCollection):
    def __init__(self, *functions):
        super(FunctionChain, self).__init__(*functions)

    def execute_all(self, *args):
        """Execute all of the functions consecutively"""
        if not len(self._functions):
            raise ValueError("Cannot execute an empty FunctionChain object!")

        final_x = self._functions[0](*args)
        for index in range(1, len(self._functions)):
            final_x = self._functions[index](*args)
        return final_x

    def execute_reverse(self, *args):
        if not len(self._functions):
            raise ValueError("Cannot execute an empty FunctionChain object!")

        final_x = self._functions[-1](*args)
        for index in range(len(self._functions) - 2, 0, -1):
            final_x = self._functions[index](*args)
        return final_x

    def execute_indices(self, indices, *args):
        if not len(indices):
            raise ValueError("Cannot execute an empty FunctionChain object!")

        final_x = indices[0](*args)
        for index in indices[1:]:
            final_x = self._functions[index](*args)
        return final_x

    def __call__(self, *args):
        return self.execute_all(*args)

    def chain(self, func: "Union[Function, FunctionChain, str]"):
        self.add_function(func)
        return self

    def plot(self, start: float = -10, stop: float = 10,
             step: float = 0.01, ymin: float = -10, ymax: float = 10, title=None, show_axis=True, show=True,
             fig=None, ax=None, formatText=False, values=None):
        if not self._functions:
            raise ValueError("Cannot plot an empty FunctionChain object")
        num_of_variables = self._functions[0].num_of_variables
        if num_of_variables == 0:
            raise ValueError("Cannot plot functions without any variables")

        elif num_of_variables == 1:
            plot_function(
                self.execute_all, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title,
                show_axis=show_axis, show=show, fig=fig, ax=ax, formatText=formatText, values=values)

        elif num_of_variables == 2:
            return plot_function_3d(self.execute_all, start=start, stop=stop, step=step, ax=ax, fig=fig,
                                    meshgrid=values)
        else:
            raise ValueError(
                f"Can only plot functions with one or two variables: found ")

    def scatter(self, start: float = -10, stop: float = 10,
                step: float = 0.01, ymin: float = -10, ymax: float = 10, title=None, show_axis=True, show=True,
                fig=None, ax=None, values=None):
        scatter_function(func=self.execute_all, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title,
                         fig=None, ax=None)

    def __getitem__(self, item):
        return FunctionChain(*(self._functions.__getitem__(item)))
