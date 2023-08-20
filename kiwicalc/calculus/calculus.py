def derivative(coefficients, get_string=False) -> Union[int, float, list]:
    """ receives the coefficients of a polynomial or a string
     and returns the derivative ( either list, float, or integer) """
    if isinstance(coefficients, str):
        coefficients = ParseExpression.to_coefficients(coefficients)

    num_of_coefficients = len(coefficients)
    if num_of_coefficients == 0:
        raise ValueError("At least one coefficient is required")
    elif num_of_coefficients == 1:  # Derivative of a free number is 0
        return 0
    elif num_of_coefficients == 2:  # f(x) = 2x, f'(x) = 2
        return coefficients[0]
    result = [coefficients[index] * (num_of_coefficients - index - 1)
              for index in range(num_of_coefficients - 1)]
    if get_string:
        return ParseExpression.coefficients_to_str(result)
    return result


def integral(coefficients, c=0, modify_original=False, get_string=False):
    """ receives the coefficients of a polynomial or a string
     and returns the integral ( either list, float, or integer) """

    if isinstance(coefficients, str):
        coefficients = ParseExpression.to_coefficients(coefficients)
    num_of_coefficients = len(coefficients)
    if num_of_coefficients == 0:
        raise ValueError("At least one coefficient is required")
    elif num_of_coefficients == 1:
        return [coefficients[0], c]
    else:
        coefficients = coefficients if modify_original and not isinstance(coefficients, (tuple, set)) else list(
            coefficients)
        coefficients.insert(0, coefficients[0] / num_of_coefficients)
        # num_of_coefficient is now like len(coefficients)-1
        for i in range(1, num_of_coefficients):
            coefficients[i] = coefficients[i + 1] / (num_of_coefficients - i)
        coefficients[-1] = c
        if get_string:
            return ParseExpression.coefficients_to_str(coefficients)
        return coefficients
