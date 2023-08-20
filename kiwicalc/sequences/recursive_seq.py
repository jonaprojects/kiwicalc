

class RecursiveSeq(Sequence):
    def __init__(self, recursive_function: str, first_values: Iterable):
        """
        Create a new instance of a recursive sequence.

        :param recursive_function:
        :param first_values:
        """
        self.__first_values = {index: value for index,
                               value in enumerate(first_values)}
        self.__recursive_string = recursive_function.strip()
        self.__lambda, self.__indices = lambda_from_recursive(
            self.__recursive_string)

    @property
    def first(self):
        return self.__first_values[0]

    def in_index(self, n: int, accumulate=True):
        return self.at_n(n, accumulate)

    def index_of(self, item):
        raise NotImplementedError

    def sum_first_n(self, n: int):
        raise NotImplementedError

    def at_n(self, n: int, accumulate=True):
        if n == 0:
            raise ValueError(
                "Sequence indices start from 1, not from 0 - a1,a2,a3....")
        return self.__at_n(n - 1, accumulate)

    # if accumulate set to true, these values will be saved in a buffer
    def __at_n(self, n: int, accumulate=True):
        """
        Get the nth element in the series.

        :param n: The place of the desired element. Must be an integer and greater than zero.
        :param accumulate: If set to true, results of computations will be saved to shorten execution time ( on the expense of the allocated memory).
        :return: Returns the nth element of the series.
        """
        if len(self.__indices) > len(self.__first_values) - 1:  # TODO: modify this condition later
            raise ValueError(
                f"Not enough initial values were entered for the series, got {len(self.__first_values)}, expected at least {len(self.__indices)} values")
        if n in self.__first_values:  # if there is already a computed answer for the calculation
            return self.__first_values[n]
        new_indices = [int(eval(old_index.replace('k', str(n)))) for old_index in
                       self.__indices]  # TODO: Later modify this too
        pre_defined_values, undefined_indices = [], []
        for new_index in new_indices:
            if new_index in self.__first_values:
                pre_defined_values.append(self.__first_values[new_index])
            else:
                undefined_indices.append(new_index)

        if undefined_indices:
            pre_defined_values.extend(
                [self.__at_n(index, accumulate) for index in undefined_indices])
        pre_defined_values.append(n + 1)  # Decide what to do about the indices
        result = self.__lambda(*pre_defined_values)  # The item on place N
        if accumulate:
            self.__first_values[n] = result
        return result

    def place_already_found(self, n: int) -> bool:
        """
        Checks if the value in the specified place in the recursive series has already been computed.

        :param n: The place on the series, starting from 1. Must be an integer.
        :return: Returns True if the value has been computed, otherwise, False
        """
        return n in self.__first_values.keys()

    def __str__(self):
        return f"{self.__recursive_string}"


def lambda_from_recursive(
        recursive_function: str):  # For now assuming it's of syntax a_n = ....... later on order it with highest power at left?
    elements = set(ptn.findall(recursive_function))
    elements = sorted(elements, key=lambda element: 0 if "{" not in element else float(
        element[element.find('n') + 1:element.find('}')]))
    indices = [(element[element.find(
        '{') + 1:element.find('}')] if '{' in element else 'n') for element in elements]
    new_elements = [element.replace('{', '').replace('}', '').replace('+', 'p').replace('-', 'd').replace('n', 'k') for
                    element in
                    elements]  # that's enough for now
    recursive_function = recursive_function[recursive_function.find('=') + 1:]
    for element, new_element in zip(elements, new_elements):
        recursive_function = recursive_function.replace(element, new_element)
    del new_elements[-1]
    new_elements.append('n')
    lambda_expression = to_lambda(recursive_function, new_elements,
                                  (list(TRIGONOMETRY_CONSTANTS.keys()) + list(MATHEMATICAL_CONSTANTS.keys())))
    del indices[-1]
    return lambda_expression, indices  # Chef's kiss
