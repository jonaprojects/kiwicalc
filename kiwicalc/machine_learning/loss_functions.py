
def mav(func1: Callable, func2: Callable, start: float, stop: float, step: float):
    """ Mean absolute value"""
    my_sum = 0
    num_of_points = 0
    for value in decimal_range(start=start, stop=stop, step=step):
        my_sum += abs(func1(value) - func2(value))
        num_of_points += 1
    if num_of_points == 0:
        raise ZeroDivisionError("Cannot process 0 points")
    return my_sum / num_of_points


def msv(func1: Callable, func2: Callable, start: float, stop: float, step: float):
    """mean square value"""
    my_sum = 0
    num_of_points = 0
    for value in decimal_range(start=start, stop=stop, step=step):
        my_sum += (func1(value) - func2(value)) ** 2
        num_of_points += 1
    if num_of_points == 0:
        raise ZeroDivisionError("Cannot process 0 points")
    return my_sum / num_of_points


def mrv(func1: Callable, func2: Callable, start: float, stop: float, step: float):
    """ mean root value """
    my_sum = 0
    num_of_points = 0
    for value in decimal_range(start=start, stop=stop, step=step):
        my_sum += (func1(value) - func2(value)) ** 2
        num_of_points += 1
    if num_of_points == 0:
        raise ZeroDivisionError("Cannot process 0 points")
    return sqrt(my_sum / num_of_points)
