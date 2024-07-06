from mono import Mono


class Var(Mono):
    """
    A class for representing variables in algebraic operations.
    It is currently represented as a subclass of the 'mono' class,
    since it is technically a monomial.

    Variables are supposed to be immutable.
    When activating operators such as '+=' on a Var object, 
    it creates new objects instead of changing the original.
    """

    def __init__(self, variable='x'):
        super().__init__(coefficient=1, variables_dict={variable: 1})

    # Make sure the variables_dict don't change, by creating copies
    def __iadd__(self, other):
        return super().__add__(other)

    def __isub__(self, other):
        return super().__sub__(other)

    def __imul__(self, other):
        return super().__mul__(other)

    def __itruediv__(self, other):
        return super().__truediv__(other)

    def __ipow__(self, other):
        return super().__pow__(other)
