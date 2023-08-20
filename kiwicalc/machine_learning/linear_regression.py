def linear_regression(axes, y_values, get_values: bool = False):
    """
    Receives a collection of x values, and a collection of their corresponding y values, and builds a fitting
    linear line from then. If the parameter "get_values" is set to True, a tuple will be returned : (slope,free_number)
    otherwise, a lambda equation of the form - lambda x : a*x + b, namely, f(x) = ax+b , will be returned.
    """
    if len(axes) != len(y_values):
        raise ValueError(f"Each x must have a corresponding y value ( Got {len(axes)} x values and {len(y_values)} y "
                         f"values ).")
    n = len(axes)
    sum_x, sum_y = sum(axes), sum(y_values)
    sum_x_2, sum_xy = sum(x ** 2 for x in axes), sum(x *
                                                     y for x, y in zip(axes, y_values))
    denominator = n * sum_x_2 - sum_x ** 2
    b = (sum_y * sum_x_2 - sum_x * sum_xy) / denominator
    a = (n * sum_xy - sum_x * sum_y) / denominator
    if get_values:
        return a, b
    return lambda x: a * x + b
