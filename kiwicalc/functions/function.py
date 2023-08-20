class Function(IPlottable, IScatterable):
    arithmetic_operations = ('+', '-')

    class Classification(Enum):
        linear = 1,
        quadratic = 2,
        polynomial = 3,
        trigonometric = 4,
        logarithmic = 5,
        exponent = 6,
        constant = 7,
        command = 8,
        linear_several_parameters = 8,
        non_linear_several_parameters = 9,
        exponent_several_parameters = 10,
        predicate = 11

    def __init__(self, func=None):
        """ creating a new instance of a function
        You can enter a string, such as "f(x) = 3x^2+6x+6sin(2x)

        """
        self.__analyzed = None
        if isinstance(func, str):
            self.__func = clean_spaces(func).replace("^", "**")
            if "lambda" in self.__func and ':' in self.__func:
                # eval it from python lambda syntax
                lambda_index, colon_index = self.__func.find(
                    "lambda") + 6, self.__func.find(':')
                self.__func_signature = f"f({self.__func[lambda_index:colon_index]})"
                self.__func_expression = self.__func[colon_index + 1:]
                self.__func = f"{self.__func_signature}={self.__func_expression}"
            elif "=>" in self.__func:
                # eval it in C# or java like lambda expression
                self.__func_signature, self.__func_expression = self.__func.split(
                    '=>')
                self.__func = f"f({self.__func_signature})={self.__func_expression}"
            first_equal_index = self.__func.find('=')
            if first_equal_index == -1:  # Get here if there is no declaration of function
                self.__variables = list(
                    extract_variables_from_expression(func))
                self.__func_signature = f'f({",".join(self.__variables)})'
                self.__func_expression = func
                self.__func = self.__func_signature + "=" + self.__func_expression
            else:
                self.__func_signature = clean_spaces(
                    self.__func[:first_equal_index])
                self.__func_expression = clean_spaces(
                    self.__func[first_equal_index + 1:])
                self.__variables = self.__func_signature[
                    self.__func_signature.find('(') + 1:self.__func_signature.rfind(')')].split(',')
                self.__variables = [x for x in self.__variables if x != ""]
            self.__num_of_variables, self.__classification = len(
                self.__variables), None
            self.classify_function()
            try:
                self.__lambda_expression = self.__to_lambda()
            except:
                warnings.warn(
                    "Couldn't generate an executable lambda function from the input, trying manual execution")
                self.__lambda_expression = None
        elif isinstance(func, (Mono, Poly)):
            self.__variables = list(func.variables)
            self.__num_of_variables = len(self.__variables)
            self.__lambda_expression = func.to_lambda()
            self.__func_expression = func.__str__().replace("^", "**")
            self.__func_signature = f'f({",".join(self.__variables)}'
            self.__func = f"{self.__func_signature})={self.__func_expression}"
            self.classify_function()

        elif not isinstance(func, str):
            if is_lambda(func):
                self.__analyzed = None
                self.__lambda_expression = func
                lambda_str = (inspect.getsourcelines(func)[0][0])
                declaration_start = lambda_str.rfind("lambda") + 6
                declaration_end = lambda_str.find(":")
                inside_signature: str = lambda_str[declaration_start:declaration_end].strip(
                )
                self.__func_signature = f'f({inside_signature})'
                self.__variables = inside_signature.split(',')
                self.__num_of_variables = len(self.__variables)  # DEFAULT
                self.__func_expression: str = lambda_str[declaration_end:lambda_str.rfind(
                    ')')]
                self.__func = "".join(
                    (self.__func_signature, self.__func_expression))
                self.classify_function()
                # TODO: CHANGE THE EXTRACTION OF DATA FROM LAMBDAS, SINCE IT WON'T WORK WHEN THERE'S MORE THAN 1 LAMBDA
                # TODO: IN A SINGLE LINE, AND ALSO IT WON'T WORK IN CASE THE LAMBDA WAS SAVED IN A VARIABLE ....
            else:
                raise TypeError(f"Function.__init__(). Unexpected type {type(func)}, expected types str,Mono,"
                                f"Poly, or a lambda expression")

        else:
            raise TypeError(f"Function.__init__(). Unexpected type {type(func)}, expected types str,Mono,"
                            f"Poly, or a lambda expression")

    # PROPERTIES - GET ACCESS ONLY

    @property
    def function_string(self):
        return self.__func

    @property
    def function_signature(self):
        return self.__func_signature

    @property
    def function_expression(self):
        return self.__func_expression

    @property
    def lambda_expression(self):
        return self.__lambda_expression

    @property
    def variables(self):
        return self.__variables

    @property
    def num_of_variables(self):
        return self.__num_of_variables

    def determine_power_role(self):  # TODO: wtf is this
        found = False
        for variable in self.__variables:
            if f'{variable}**' in self.__func:
                current_index = self.__func_expression.find(
                    f'{variable}**') + len(variable) + 2
                if current_index > len(self.__func_expression):
                    raise ValueError(
                        "Invalid syntax: '**' is misplaced in the string. ")
                else:
                    finish_index = current_index + 1
                    while finish_index < len(self.__func_expression) and only_numbers_letters(
                            self.__func_expression[finish_index].strip()):
                        finish_index += 1
                    power = self.__func_expression[current_index:finish_index]
                    found = True
                    break
        if not found:
            return False
        return "polynomial" if is_number(power) and float(power) > 1 else "exponent"

    def classify_function(self):
        if '==' in self.__func_expression:
            self.__classification = self.Classification.predicate
        elif self.__num_of_variables == 1:
            if contains_from_list(list(TRIGONOMETRY_CONSTANTS.keys()), self.__func_expression):
                self.__classification = self.Classification.trigonometric
            elif f'{self.__variables[0]}**2' in self.__func_expression:
                self.__classification = Function.Classification.quadratic
            elif f'{self.__variables[0]}**' in self.__func_expression:
                power_role = self.determine_power_role()
                if power_role == 'polynomial':
                    self.__classification = self.Classification.polynomial
                elif power_role == 'exponent':
                    self.__classification = self.Classification.exponent
                else:
                    self.__classification = self.Classification.linear
            elif is_evaluatable(self.__func_expression):
                self.__classification = Function.Classification.constant
            else:
                self.__classification = Function.Classification.linear
        elif self.__num_of_variables < 1:
            self.__classification = Function.Classification.command
        else:
            # implement several __variables trigonometric choice
            power_role = self.determine_power_role()
            if power_role == 'polynomial':
                self.__classification = self.Classification.non_linear_several_parameters
            elif power_role == 'exponent':
                self.__classification = self.Classification.exponent_several_parameters
            else:
                self.__classification = self.Classification.linear_several_parameters

    @property
    def classification(self):
        return self.__classification

    def compute_value(self, *parameters):
        """ getting the result of the function for the specified parameters"""
        if self.__lambda_expression is not None:  # If an executable lambda has already been created
            try:
                if len(parameters) == 1:
                    if isinstance(parameters[0], Matrix):
                        matrix_copy = parameters[0].__copy__()
                        matrix_copy.apply_to_all(self.__lambda_expression)
                        return matrix_copy

                return self.__lambda_expression(*parameters)
            except ZeroDivisionError:  # It means that value wasn't valid.
                return None
            except ValueError:
                return None
            except OverflowError or MemoryError:
                warnings.warn(f"The operation on the parameters: '{parameters}'"
                              f" have exceeded python's limitations ( too big )")
        else:
            warnings.warn("Couldn't compute the expression entered! Check for typos and invalid syntax."
                          "Valid working examples: f(x) = x^2 + 8 , g(x,y) = sin(x) + 3sin(y)")
            return None

    def apply_on(self, collection):
        return apply_on(self.lambda_expression, collection)

    def range_gen(self, start: float, end: float, step: float = 1) -> Iterator:
        """
        Yields tuples of (x,y) values of the function within the range and interval specified.
        For example, for f(x) = x^2, in the range 2 and 4, and the step of 1, the function will
        yield (2,4), and then (3,9).
        Currently Available only to functions with one variable!

        :param start: the start value
        :param end: the end value
        :param step: the difference between each value
        :return: yields a (value,result) tuple every time
        """
        if self.__num_of_variables > 1:
            warnings.warn(
                "Cannot give the range of functions with more than one variable!")
        for val in decimal_range(start, end, step):
            yield val, self.compute_value(val)

    def toIExpression(self):
        """
        Try to convert the function into an algebraic expression.
        :return: If successful, an algebraic expression will be returned. Otherwise - None
        """
        try:
            my_type = self.__classification
            poly_types = [self.Classification.linear, self.Classification.quadratic,
                          self.Classification.polynomial, self.Classification.linear_several_parameters]
            if my_type in poly_types:
                return Poly(self.__func_expression)
            elif my_type == self.Classification.trigonometric:
                return TrigoExprs(self.__func_expression)
            elif my_type == self.Classification.logarithmic:
                return Log(self.__func_expression)
            elif my_type in (self.Classification.exponent, self.Classification.exponent_several_parameters):
                return Exponent(self.__func_expression)
            else:
                raise ValueError
        except:
            raise ValueError("Cannot convert the function to an algebraic expression! Either the method is "
                             "invalid, or it's not supported yet for this feature. Wait for next versions!")
            return None

    def derivative(self) -> "Optional[Union[Function, int, float]]":
        """
        Try to compute the derivative of the function. If not successful - return None
        :return: a string representation of the derivative of the function
        """
        num_of_variables = self.__num_of_variables
        if num_of_variables == 0:
            return 0
        elif num_of_variables == 1:
            poly_types = [self.Classification.linear,
                          self.Classification.quadratic, self.Classification.polynomial]
            if self.__classification in poly_types:
                poly_string = Poly(
                    self.__func_expression).derivative().__str__()
                return Function(poly_string)
            my_expression = self.toIExpression()
            if my_expression is None or not hasattr(my_expression, "derivative"):
                return None
            return Function(my_expression.derivative())
        else:
            raise ValueError(
                "Use the partial_derivative() method for functions with multiple variables")

    def partial_derivative(self, variables: Iterable) -> "Optional[Union[Function, int, float]]":
        """Experimental Feature: Try to compute the partial derivative of the function."""
        num_of_variables = self.__num_of_variables
        if num_of_variables == 0:
            return 0
        elif num_of_variables == 1:
            return self.derivative()
        else:
            my_expression = self.toIExpression()
            if my_expression is None or not hasattr(my_expression, "partial_derivative"):
                return None
            return Function(my_expression.partial_derivative(variables).__str__())

    def integral(self) -> "Optional[Union[Function, int, float]]":
        """Computing the integral of the function. Currently without adding C"""
        num_of_variables = self.__num_of_variables
        if num_of_variables > 1:
            raise ValueError(
                "Integrals with multiple variables are not supported yet")
        my_expression = self.toIExpression()
        if my_expression is None or not hasattr(my_expression, "integral"):
            return None
        return my_expression.integral()

    def to_dict(self):
        return {
            "type": "function",
            "string": self.__func
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    def export_json(self, path: str):
        with open(path, 'w') as json_file:
            json_file.write(self.to_json())

    @staticmethod
    def from_dict(func_dict: dict):
        return Function(func_dict['string'])

    @staticmethod
    def from_json(json_string: str):
        return Function.from_dict(json.loads(json_string))

    def trapz(self, a: float, b: float, N: int):
        return trapz(self.__lambda_expression, a, b, N)

    def simpson(self, a: float, b: float, N: int):
        return simpson(self.__lambda_expression, a, b, N)

    def reinman(self, a: float, b: float, N: int):
        return reinman(self.__lambda_expression, a, b, N)

    def range(self, start: float, end: float, step: float, round_results: bool = False):
        """
        fetches all the valid values of a function in the specified range
        :param start: the beginning of the range
        :param end: the end of the range
        :param step: the interval between each item in the range
        :return: returns the values in the range, and their valid results
        """
        if round_results:
            values = [round_decimal(i)
                      for i in decimal_range(start, end, step)]
            results = [self.compute_value(i) for i in values]
            for index, result in enumerate(results):
                if result is not None:
                    results[index] = round_decimal(result)
        else:
            values = [i for i in decimal_range(start, end, step)]
            results = [self.compute_value(i) for i in values]
        for index, result in enumerate(results):
            if result is None:
                del results[index]
                del values[index]
            elif isinstance(result, bool):
                results[index] = float(result)
        return values, results

    def range_3d(self, x_start: float, x_end: float, x_step: float, y_start: float, y_end: float, y_step: float,
                 round_results: bool = False):
        x_values, y_values, z_values = [], [], []
        for x in decimal_range(x_start, x_end, x_step):
            for y in decimal_range(y_start, y_end, y_step):
                if round_results:
                    x_values.append(round_decimal(x))
                    y_values.append(round_decimal(y))
                    z = self.compute_value(x, y)
                    z_values.append(round_decimal(z))
                else:
                    x_values.append(x)
                    y_values.append(y)
                    z = self.compute_value(x, y)
                    z_values.append(z)

        return x_values, y_values, z_values

    def random(self, a: int = 1, b: int = 10, custom_values=None, as_point=False, as_tuple=False):
        """ returns a random value from the function"""
        if self.num_of_variables == 1:
            random_number = random.randint(
                a, b) if custom_values is None else custom_values
            if not as_point:
                if as_tuple:
                    return random_number, self.compute_value(random_number)
                return self.compute_value(random_number)
            return Point2D(random_number, self.compute_value(random_number))
        else:
            values = [random.randint(a, b) for _ in
                      range(self.num_of_variables)] if custom_values is None else custom_values
            if not as_point:
                if as_tuple:
                    return values, self.compute_value(*values)
                return self.compute_value(*values)
            values.append(self.compute_value(*values))
            if len(values) == 3:
                return Point3D(values[0], values[1], values[2])
            return Point(values)

    def plot(self, start: float = -10, stop: float = 10, step: float = 0.05, ymin: float = -10, ymax: float = 10, text: str = None, others: "Optional[Iterable[Function]]" = None, fig=None, ax=None, show_axis=True, show=True, formatText=True, values=None):
        """ plots the graph of the function using matplotlib.
        currently operational only for 1 parameter functions """
        if self.__num_of_variables == 1:
            if others is None:
                plot_function(func=self.lambda_expression, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax,
                              title=text, show_axis=show_axis, show=show, fig=fig, ax=ax, formatText=formatText,
                              values=values)
            else:

                funcs = [func for func in others]
                funcs.append(self)
                plot_functions(funcs, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax,
                               show_axis=show_axis, title=text, show=show)  # TODO: fix this!!!!!!!
        elif self.__num_of_variables == 2:
            # functions with two variables_dict in the form f(x,y) = .... can plotted in 3D
            plot_function_3d(given_function=self.lambda_expression,
                             start=start, stop=stop, step=step)

    def scatter(self, start: float = -3, stop: float = 3,
                step: float = 0.3, show_axis=True, show=True):
        num_of_variables = len(self.__variables)
        if num_of_variables == 1:
            self.scatter2d(start=start, step=step,
                           show_axis=show_axis, show=show)
        elif num_of_variables == 2:
            self.scatter3d(start=start, stop=stop, step=step, show=show)
        else:
            raise ValueError(
                f"Cannot scatter a function with {num_of_variables} variables")

    def scatter3d(self, start: float = -3, stop: float = 3,
                  step: float = 0.3,
                  xlabel: str = "X Values",
                  ylabel: str = "Y Values", zlabel: str = "Z Values", show=True, fig=None, ax=None,
                  write_labels=True, meshgrid=None, title=""):
        return scatter_function_3d(
            func=self.__lambda_expression, start=start, stop=stop, step=step, xlabel=xlabel, ylabel=ylabel,
            zlabel=zlabel, show=show, fig=fig, ax=ax, write_labels=write_labels, meshgrid=meshgrid, title=title)

    def scatter2d(self, start: float = -15, stop: float = 15, step: float = 0.3, ymin=-15, ymax=15, show_axis=True,
                  show=True, basic=True):
        if basic:
            scatter_function(self.__lambda_expression, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax,
                             show_axis=show_axis, show=show)
            return
        values, results = self.range(start, stop, step, round_results=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('equal')
        # And a corresponding grid
        ax.grid(which='both')

        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        if show_axis:
            # Set bottom and left spines as x and y axes of coordinate system
            ax.spines['bottom'].set_position('zero')
            ax.spines['left'].set_position('zero')

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Create 'x' and 'y' labels placed at the end of the axes
        plt.title(fr"${format_matplot(self.__func)}$", fontsize=14)

        norm = plt.Normalize(-10, 10)
        cmap = plt.cm.RdYlGn
        colors = [-5 for _ in range(len(results))]
        # plt.plot(values, results, 'o')
        sc = plt.scatter(x=values, y=results, c=colors,
                         s=90, cmap=cmap, norm=norm)
        annotation = ax.annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                                 bbox=dict(boxstyle="round", fc="w"),
                                 arrowprops=dict(arrowstyle="->"))
        annotation.set_visible(False)

        def hover(event):
            vis = annotation.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    pos = sc.get_offsets()[ind["ind"][0]]
                    annotation.xy = pos
                    text = f"{pos[0], pos[1]}"
                    annotation.set_text(text)
                    annotation.get_bbox_patch().set_facecolor(
                        cmap(norm(colors[ind["ind"][0]])))
                    annotation.get_bbox_patch().set_alpha(0.4)
                    annotation.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annotation.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.ylim(ymin, ymax)
        plt.ylim(ymin, ymax)
        if show:
            plt.show()

    @staticmethod
    def plot_all(*functions, show_axis=True, start=-10, end=10, step=0.01, ymin=-20, ymax=20):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('equal')

        # And a corresponding grid
        ax.grid(which='both')

        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        if show_axis:
            # Set bottom and left spines as x and y axes of coordinate system
            ax.spines['bottom'].set_position('zero')
            ax.spines['left'].set_position('zero')

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        for func_object in functions:
            if isinstance(func_object, str) or is_lambda(func_object):
                func_object = Function(func_object)

            if isinstance(func_object, Function):
                values, results = func_object.range_gen(start, end, step)
                plt.plot(values, results)
            else:
                raise TypeError(
                    f"Invalid type for a function: type {type(func_object)} ")
        plt.ylim(ymin, ymax)
        plt.show()

    def search_roots_in_range(self, val_range: tuple, step=0.01, epsilon=0.01, verbose=True):
        """
        Iterates on the function in a certain range, to find the estimated roots in the range.
        This is not a rather efficient method compared to other root-finding algorithms, however, it should work in
        almost all cases and types of functions, depending on the step of each iteration and the epsilon.

        :param val_range: A tuple with the range which the roots will be searched in. For example, (-50,50)
        :param step: The interval between each X value in the iteration.
        :param epsilon: If y of a point is smaller than epsilon, the corresponding x will be considered a root.
        :return:Returns a list with all of the different roots which were found ( these roots were also rounded ).
        """
        if verbose:
            warnings.warn(
                "The use of this method for finding roots is not recommended!")
        if len(val_range) != 2:
            raise IndexError(
                "The range of values must be an iterable containing 2 items: the minimum value, and the maximum."
                "\nFor example, (0,10) "
            )

        values, results = self.range_gen(val_range[0], val_range[1], step)
        matching_values = [(value, result) for value, result in
                           zip(values, results) if values is not None and result is not None and abs(result) < epsilon]
        return [(round_decimal(value), round_decimal(result)) for value, result in matching_values]

    def newton(self, initial_value: float):
        """
        Finds a single root of the function with the Newton-Raphson method.

        :param initial_value: An arbitrary number, preferably close the root.
        :return: The closest root of the function to the given initial value.

        """
        if self.__lambda_expression is not None:
            return newton_raphson(self.__lambda_expression, self.derivative().__lambda_expression, initial_value)
        return newton_raphson(self, self.derivative(), initial_value)

    def finite_integral(self, a, b, N: int, method: str = 'trapz'):
        if not isinstance(method, str):
            raise TypeError(f"Invalid type for param 'method' in method finite_integral() of class Function."
                            f"Expected type 'str'.")
        method = method.lower()
        if method == 'trapz':
            return self.trapz(a, b, N)
        elif method == 'simpson':
            return self.simpson(a, b, N)
        elif method == 'reinman':
            return self.reinman(a, b, N)
        else:
            raise ValueError(
                f"Invalid method '{method}'. The available methods are 'trapz' and 'simpson'. ")

    # TODO: implement this, or check if analyze can be used for this in someway
    def coefficients(self):
        if self.__classification in (self.Classification.polynomial, self.Classification.quadratic,
                                     self.Classification.linear, self.Classification.constant):
            return Poly(self.__func_expression).coefficients()
        else:
            raise ValueError(
                f"Function's classification ({self.__classification}) doesn't support this feature.")

    # TODO: add more root-finding algorithms here
    def roots(self, epsilon=0.000001, nmax=100000):
        return aberth_method(self.__to_lambda(), self.derivative().__to_lambda(), self.coefficients(), epsilon, nmax)

    def max_and_min(self):
        """
        tries to find the max and min points
        """
        if self.__classification not in (
                self.Classification.quadratic, self.Classification.polynomial, self.Classification.linear):
            raise NotImplementedError
        first_derivative = self.derivative()
        second_derivative = first_derivative.derivative()
        derivative_roots = aberth_method(first_derivative.lambda_expression, second_derivative.lambda_expression,
                                         first_derivative.coefficients())
        derivative_roots = (root for root in derivative_roots if
                            abs(root.imag) < 0.000001)  # Accepting only real solutions for now
        max_points, min_points = [], []
        for derivative_root in list(derivative_roots):
            # if 0, it's not min and max
            val = second_derivative(derivative_root.real)
            value = derivative_root.real
            result = round_decimal(self.lambda_expression(value))
            if val.real > 0:
                min_points.append((value, result))
            elif val.real < 0:
                max_points.append((value, result))
        return max_points, min_points

    def incline_and_decline(self):
        return NotImplementedError

    def chain(self, other_func: "Optional[Union[Function,str]]"):
        if isinstance(other_func, Function):
            return FunctionChain(self, other_func)
        else:
            try:
                return FunctionChain(self, Function(other_func))
            except ValueError:
                raise ValueError(
                    f"Invalid value {other_func} when creating a FunctionChain object")
            except TypeError:
                raise TypeError(
                    f"Invalid type {type(other_func)} when creating a FunctionChain object")

    def y_intersection(self):
        """
        Finds the intersection of the function with the y axis

        :return: Returns the y value when x = 0, or None, if function is not defined in x=0, or an error has occurred.
        """
        try:
            return self.compute_value(0)
        except:
            return None

    def __call__(self, *parameters):
        return self.compute_value(*parameters)

    def __str__(self):
        return self.__func

    def __repr__(self):
        return f'Function("{self.__func}")'

    def __getitem__(self, item):
        """
        :param item: a slice object which represents indices range or an int that represent index
        :return: returns the variable name in the index, or the variable names in the indices in case of slicing
        """
        if isinstance(item, slice):  # handling slicing
            start = item.start
            step = 1 if item.step is None else item.step
            return [self.__variables[i] for i in range(start, item.stop, step)]
        elif isinstance(item, int):
            return self.__variables[item]

    def __setitem__(self, key, value):
        return self.__variables.__setitem__(key, value)

    def __delitem__(self, key):
        return self.__variables.__delitem__(key)

    def __eq__(self, other):
        """ when equating between 2 functions, a list of the approximations of their intersections will be returned
        :rtype: list
        """
        if other is None:
            return False
        if isinstance(other, str):
            other = Function(other)
        if isinstance(other, Function):
            if other.__num_of_variables != other.__num_of_variables:
                return False
            for _ in range(3):  # check equality between 3 random values
                values, results = self.random(as_tuple=True)
                other_results = other.random(custom_values=values)
                if results != other_results:
                    return False

            return self.__lambda_expression.__code__.co_code == other.__lambda_expression.__code__.co_code
        elif is_lambda(other):
            other_num_of_variables = other.__code__.co_argcount
            if self.__num_of_variables != other_num_of_variables:
                return False
            for _ in range(3):  # check equality between 3 random values
                values = [random.randint(1, 10)
                          for _ in range(other_num_of_variables)]
                my_results, other_results = self.compute_value(
                    *values), other.compute_value(*values)
                if my_results != other_results:
                    return False

            return self.__lambda_expression.__code__.co_code == other.__lambda_expression.__code__.co_code

        else:
            raise TypeError(
                f"Invalid type {type(other)} for equating, allowed types: Function,str,lambda expression")

    def __ne__(self, other):
        return not self.__eq__(other)

    def __to_lambda(self):
        """Returns a lambda expression from the function. You should use the lambda_expression property."""
        return to_lambda(self.__func_expression, self.__variables,
                         (list(TRIGONOMETRY_CONSTANTS.keys()) +
                          list(MATHEMATICAL_CONSTANTS.keys())),
                         format_abs=True, format_factorial=True)

    def search_intersections(self, other_func, values_range=(-100, 100), step=0.01, precision=0.01):
        """
        Returns the intersections between the current function to another function
        Currently works only in functions with only one parameter ... """
        if isinstance(other_func, str):
            other_func = Function(other_func)
        # handle user-defined functions and lambdas
        elif inspect.isfunction(other_func):
            if is_lambda(other_func):
                other_func = Function(other_func)
            else:
                raise TypeError("Cannot perform intersection! only lambda functions are supported "
                                "so far in this version.")
        if isinstance(other_func, Function):
            intersections = []
            for i in np.arange(values_range[0], values_range[1], step):
                first_result = self.compute_value(i)
                if first_result is None:
                    continue
                first_result = round_decimal(first_result)
                second_result = other_func.compute_value(i)
                if second_result is None:
                    continue
                second_result = round_decimal(second_result)
                if abs(first_result - second_result) <= precision:
                    found = False
                    for x, y in intersections:
                        x_difference = abs(round_decimal(x - i))
                        if x_difference <= precision:
                            found = True
                            break
                    if not found:
                        intersections.append(
                            (round_decimal(i), round_decimal((first_result + second_result) / 2)))  # add the best
                        # approximation for y
            return intersections

        else:
            raise TypeError(f"Invalid type {type(other_func)} for a function. Use types Function ( recommended) or str"
                            f"instead")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

    # Inefficient single the lambda expression will be re-calculated for no reason ...
    def __copy__(self):
        return Function(func=self.__func)
