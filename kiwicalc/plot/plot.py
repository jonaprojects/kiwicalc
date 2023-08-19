from matplotlib import pyplot as plt
import warnings


def create_grid():
    """ Create a grid in matplotlib"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('equal')
    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    return fig, ax


def draw_axis(ax):
    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create 'x' and 'y' labels placed at the end of the axes


def scatter_dots(x_values, y_values, title: str = "", ymin: float = -10, ymax: float = 10, color=None,
                 show_axis=True, show=True, fig=None, ax=None):  # change start and end to be automatic?
    if (length := len(x_values)) != (y_length := len(y_values)):
        raise ValueError(f"You must enter an equal number of x and y values. Got {length} x values and "
                         f"{y_length} y values.")
    if None in (fig, ax):
        fig, ax = create_grid()
    if show_axis:
        draw_axis(ax)
    plt.title(title, fontsize=14)
    plt.ylim(ymin, ymax)
    plt.scatter(x=x_values, y=y_values, s=90, c=color)
    if show:
        plt.show()


def scatter_dots_3d(x_values, y_values, z_values, title: str = "", xlabel: str = "X Values",
                    ylabel: str = "Y Values", zlabel: str = "Z Values", fig=None,
                    ax=None, show=True, write_labels=True):
    if None in (fig, ax):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    if title:
        plt.title(title)
    ax.scatter(x_values, y_values, z_values)

    if write_labels:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

    if show:
        plt.show()


def scatter_function(func: Union[Callable, str], start: float = -10, stop: float = 10,
                     step: float = 0.5, ymin: float = -10, ymax: float = 10, title="", color=None,
                     show_axis=True, show=True, fig=None, ax=None, values=None):
    if isinstance(func, str):
        func = Function(func)
    if values is not None:
        results = [func(value) for value in values]
    else:
        values, results = values_in_range(func, start, stop, step)
    scatter_dots(values, results, title=title, ymin=ymin, ymax=ymax, color=color, show_axis=show_axis, show=show,
                 fig=fig, ax=ax)


def scatter_function_3d(func: "Union[Callable, str, IExpression]", start: float = -3, stop: float = 3,
                        step: float = 0.3,
                        xlabel: str = "X Values",
                        ylabel: str = "Y Values", zlabel: str = "Z Values", show=True, fig=None, ax=None,
                        write_labels=True, meshgrid=None, title=""):
    if isinstance(func, str):
        func = Function(func)

    if meshgrid is None:
        x = y = np.arange(start, stop, step)
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = meshgrid
    zs = np.array([])
    for x_value, y_value in zip(np.ravel(X), np.ravel(Y)):
        try:
            zs = np.append(zs, func(x_value, y_value))
        except:
            zs = np.append(zs, np.nan)
    Z = zs.reshape(X.shape)
    scatter_dots_3d(
        X, Y, Z, fig=fig, ax=ax, title=title, show=show, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
        write_labels=write_labels)


def scatter_functions_3d(functions: "Iterable[Union[Callable, str, IExpression]]", start: float = -5, stop: float = 5,
                         step: float = 0.1,
                         xlabel: str = "X Values",
                         ylabel: str = "Y Values", zlabel: str = "Z Values"):
    pass


def process_to_points(func: Union[Callable, str], start: float = -10, stop: float = 10,
                      step: float = 0.01, ymin: float = -10, ymax: float = 10, values=None):
    if isinstance(func, str):
        func = Function(func)
    if values is None:
        values = list(decimal_range(start, stop, step)
                      ) if values is None else values
    results = []
    for index, value in enumerate(values):
        try:
            current_result = func(value)
            results.append(current_result)
        except ValueError:
            results.append(None)

    return values, results


def plot_function(func: Union[Callable, str], start: float = -10, stop: float = 10,
                  step: float = 0.01, ymin: float = -10, ymax: float = 10, title=None, show_axis=True, show=True,
                  fig=None, ax=None, formatText=False, values=None):
    # Create the setup of axis and stuff

    if None in (fig, ax):  # If at least one of those parameters is None, then create a new ones
        fig, ax = create_grid()
    if show_axis:
        draw_axis(ax)
    values, results = process_to_points(
        func, start, stop, step, ymin, ymax, values)
    if title is not None:
        if formatText:
            plt.title(fr"${format_matplot(title)}$", fontsize=14)
        else:
            plt.title(fr"{title}", fontsize=14)
    plt.ylim(ymin, ymax)
    plt.plot(values, results)

    if show:
        plt.show()


def plot_function_3d(given_function: "Union[Callable, str, IExpression]", start: float = -3, stop: float = 3,
                     step: float = 0.3,
                     xlabel: str = "X Values",
                     ylabel: str = "Y Values", zlabel: str = "Z Values", show=True, fig=None, ax=None,
                     write_labels=True, meshgrid=None):
    if step < 0.1:
        step = 0.3
        warnings.warn(
            "step parameter modified to 0.3 to avoid lag when plotting in 3D")
    if isinstance(given_function, str):
        given_function = Function(given_function)
    elif isinstance(given_function, IExpression):
        num_of_variables = len(given_function.variables)
        if num_of_variables != 2:
            raise ValueError(
                f"Invalid expression: {given_function}. Found {num_of_variables} variables, expected 2.")
        if hasattr(given_function, "to_lambda"):
            given_function = given_function.to_lambda()
        elif hasattr(given_function, "__call__"):
            pass
        else:
            raise ValueError(
                f"This type of algebraic expression isn't supported for plotting in 3D!")

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')
    if meshgrid is None:
        x = y = np.arange(start, stop, step)
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = meshgrid

    zs = np.array([])
    for x_value, y_value in zip(np.ravel(X), np.ravel(Y)):
        try:
            result = given_function(x_value, y_value)
            if result is None:
                result = np.nan
            zs = np.append(zs, result)
        except ValueError:
            zs = np.append(zs, np.nan)
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z)
    if write_labels:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
    if show:
        plt.show()


def plot_functions_3d(functions: "Iterable[Union[Callable, str, IExpression]]", start: float = -5, stop: float = 5,
                      step: float = 0.1,
                      xlabel: str = "X Values",
                      ylabel: str = "Y Values", zlabel: str = "Z Values"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(start, stop, step)
    for func in functions:
        plot_function_3d(func, show=False, write_labels=False,
                         fig=fig, ax=ax, meshgrid=np.meshgrid(x, y))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()


def plot_functions(functions, start: float = -10, stop: float = 10, step: float = 0.01, ymin: float = -10,
                   ymax: float = 10, title: str = None, formatText: bool = False,
                   show_axis: bool = True, show: bool = True, with_legend=True):
    fig, ax = create_grid()
    if show_axis:
        draw_axis(ax)
    values = np.arange(start, stop, step)
    plt.ylim(ymin, ymax)
    if title is not None:
        if formatText:
            plt.title(fr"${format_matplot(title)}$", fontsize=14)
        else:
            plt.title(title, fontsize=14)
    for given_function in functions:
        if isinstance(given_function, str):
            label = given_function
            given_function = Function(given_function).lambda_expression
        elif isinstance(given_function, Function):
            label = given_function.function_string
            given_function = given_function.lambda_expression
        elif isinstance(given_function, IExpression):
            label = given_function.__str__()
            if hasattr(given_function, "to_lambda"):
                given_function = given_function.to_lambda()
            else:
                raise ValueError(
                    f"Invalid algebraic expression for plotting: {given_function}")
        else:
            label = None
        plt.plot(values, [given_function(value)
                 for value in values], label=label)
    if with_legend:
        plt.legend()
    if show:
        plt.show()


def scatter_functions(functions, start: float = -10, stop: float = 10, step: float = 0.5, ymin: float = -10,
                      ymax: float = 10, title: str = None,
                      show_axis: bool = True, show: bool = True):
    fig, ax = create_grid()
    cycol = cycle('bgrcmykw')
    values = np.arange(start, stop, step)
    for index, current_function in enumerate(functions):
        scatter_function(func=current_function, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=title,
                         color=next(cycol), show_axis=True, show=False, fig=fig, ax=ax, values=values)
    plt.show()


def plot_vector_2d(x_start: float, y_start: float, x_distance: float, y_distance: float, show=True, fig=None, ax=None):
    if None in (fig, ax):
        fig, ax = plt.subplots(figsize=(10, 8))
    ax.arrow(x_start, y_start, x_distance, y_distance,
             head_width=0.1,
             width=0.01)
    if show:
        plt.show()


def plot_vector_3d(starts: Tuple[float, float, float], distances: Tuple[float, float, float], arrow_length_ratio=0.08,
                   show=True, fig=None, ax=None):
    """plot a 3d vector"""
    u, v, w = distances
    start_x, start_y, start_z = starts

    if (fig, ax) == (None, None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([start_x, (start_x + u)])
        ax.set_ylim([start_y, (start_y + v)])
        ax.set_zlim([start_z, (start_z + w)])

    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = fig.add_subplot(111, projection='3d')

    ax.quiver(start_x, start_y, start_z, u, v, w,
              arrow_length_ratio=arrow_length_ratio)
    if show:
        plt.show()


def plot_complex(*numbers: complex, title: str = "", show=True):
    """
    plot complex numbers on the complex plane

    :param numbers: The complex numbers to be plotted
    :param show: If set to false, the plotted
    :return: fig, ax
    """
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_title(title, va='bottom')
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    plt.title(title)
    max_radius = abs(numbers[0])
    for c in numbers:
        radius = abs(c)
        if radius > max_radius:
            max_radius = radius
        ax.scatter(cmath.phase(c), radius)

    ax.set_rticks(np.linspace(0, int(max_radius) * 2, num=5))  # Less radial
    ax.set_rmax(max_radius * 1.25)
    if show:
        plt.show()
    return fig, ax


def generate_subplot_shape(num_of_functions: int):
    square_root = sqrt(num_of_functions)
    # if an integer square root is found, then we're over
    if square_root == int(square_root):
        return int(square_root), int(square_root)
    # Then find the 2 biggest factors
    try:
        result = min([(first, second) for first, second in combinations(range(1, num_of_functions), 2) if
                      first * second == num_of_functions], key=lambda x: abs(x[1] - x[0]))
        if result[0] > result[1]:
            return result[1], result[0]
        return result
    except ValueError:
        return ceil(square_root), ceil(square_root)


def plot_multiple(funcs, shape: Tuple[int, int] = None, start: float = -10, stop: float = 10,
                  step: float = 0.01, ymin: float = -10, ymax: float = 10, title=None, show_axis=True, show=True,
                  values=None):
    num_of_functions = len(funcs)
    if shape is None:
        shape = generate_subplot_shape(num_of_functions)
    fig, ax = plt.subplots(shape[0], shape[1])

    fig.tight_layout()
    func_index = 0
    for i in range(shape[0]):
        if func_index >= num_of_functions:
            break
        for j in range(shape[1]):
            if func_index >= num_of_functions:
                break
            values, results = process_to_points(
                funcs[func_index], start, stop, step, ymin, ymax, values)
            current_ax = ax[i, j] if shape[0] > 1 else ax[j]
            current_ax.plot(values, results, label=funcs[func_index])
            current_ax.set_title(funcs[func_index])
            if show_axis:
                draw_axis(current_ax)
            func_index += 1

    if title is not None:
        plt.title(title)
    if show:
        try:  # try to plot these in full screen.
            wm = plt.get_current_fig_manager()
            wm.window.state('zoomed')
        except:
            warnings.warn("Couldn't plot in full screen!")
        plt.show()
