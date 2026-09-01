from __future__ import annotations
import random
import warnings
from typing import Union, Tuple, List, Optional, Any, Callable, Iterator, Iterable
import numpy as np
from numpy.linalg import inv, LinAlgError

from kiwicalc.core.interfaces import IExpression
from kiwicalc.core.utils import copy_expression
from kiwicalc.parsing.parse_equation import equation_to_one_side
from kiwicalc.parsing.parse_expression import poly_from_str

def column(matrix, index: int):
    """
    Fetches a column in a matrix

    :param matrix: the matrix from which we fetch the column
    :param index: the index of the column. From 0 to the number of num_of_columns minus 1.
    :return: Returns a list of numbers, that represents the column in the given index
    :raise: Raises index error if the index isn't valid.
    """
    return [row[index] for row in matrix]

class Matrix:

    def __init__(self, matrix: Union[list, str, tuple]=None, dimensions=None, copy_elements=False):
        if (matrix, dimensions) == (None, None):
            raise ValueError('Cannot create an empty Matrix')
        if matrix is not None:
            if isinstance(matrix, str):
                dimensions = matrix
            else:
                if not isinstance(matrix, list):
                    matrix = list(matrix)
                if len(matrix) > 0 and isinstance(matrix[0], Iterable) and not isinstance(matrix[0], str):
                    expected_columns = len(matrix[0])
                    if any(not isinstance(row, Iterable) or isinstance(row, str) or len(row) != expected_columns for row in matrix):
                        raise ValueError('Matrix rows must all have the same length')
                    self._num_of_rows, self._num_of_columns = len(matrix), len(matrix[0])
                    self._matrix = [[copy_expression(item) for item in row] for row in matrix]
                else:
                    self._num_of_rows, self._num_of_columns = 1, len(matrix)
                    self._matrix = [copy_expression(item) for item in matrix]
        if dimensions is not None:
            if isinstance(dimensions, str):
                if dimensions.count('x') == 1:
                    self._num_of_rows, self._num_of_columns = [int(i) for i in dimensions.strip().split('x')]
                elif dimensions.count(',') == 1:
                    self._num_of_rows, self._num_of_columns = [int(i) for i in dimensions.strip().split(',')]
            elif isinstance(dimensions, (tuple, list, set)):
                self._num_of_rows, self._num_of_columns = (int(dimensions[0]), int(dimensions[1]))
            else:
                raise TypeError(f'Invalid type {type(dimensions)} for the dimensions of the matrix ')
            self.matrix: List[list] = [[0 for col in range(self.num_of_columns)] for row in range(self.num_of_rows)]

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, mat):
        self._matrix = mat

    @property
    def num_of_rows(self):
        return self._num_of_rows

    @property
    def num_of_columns(self):
        return self._num_of_columns

    @property
    def shape(self):
        return (self._num_of_rows, self._num_of_columns)

    def add_and_mul(self, line1: int, line2: int, scalar):
        """
        adds a line to another line which is multiplied by a value.

        :param line1: The line that will receive the multiplication result
        :param line2: The line that its multiplication with the scalar value will be added to the other line.
        :param scalar:
        :return: None
        """
        if line1 < 0 or line1 >= self._num_of_rows:
            raise IndexError(f'Invalid line index {line1}. Expected indices between 0 and {self._num_of_rows}')
        if line2 < 0 or line2 >= self._num_of_rows:
            raise IndexError(f'Invalid line index {line2}. Expected indices between 0 and {self._num_of_rows}')
        for i in range(self.num_of_columns):
            self.matrix[line1][i] += self.matrix[line2][i] * scalar

    def replace_rows(self, line1: int, line2: int):
        """
        Replace the values between two num_of_rows in the matrix.

        :param line1: The index of the first row.
        :param line2: The index of the second row.
        :return:
        """
        if line1 < 0 or line1 >= self._num_of_rows:
            raise IndexError(f'Invalid line index {line1}. Expected indices between 0 and {self._num_of_rows - 1}')
        if line2 < 0 or line2 >= self._num_of_rows:
            raise IndexError(f'Invalid line index {line2}. Expected indices between 0 and {self._num_of_rows - 1}')
        for i in range(self.num_of_columns):
            self.matrix[line1][i], self.matrix[line2][i] = (self.matrix[line2][i], self.matrix[line1][i])

    def __get_starting_item(self, i: int):
        """ Get the index of the first element in a row that is not 0"""
        for j in range(i, self._num_of_rows):
            if self.matrix[j][i] != 0:
                return j
        return -1

    def multiply_row(self, expression, row: int):
        if not 0 < row <= self._num_of_rows:
            raise IndexError(f'Invalid line index {row}. Expected indices between 0 and {self._num_of_rows}')
        for i in range(self.num_of_columns):
            self.matrix[row][i] *= expression

    def divide_row(self, scalar, row: int):
        """
        Dividing a row in the matrix by a number. The row numbers starts from 1, instead of 0, as it is common
        in indices.
        :param scalar: type float
        :param row: the number of the row, starting from 1, type int.
        :return: Doesn't return anything ( None )
        """
        if row >= self.num_of_rows:
            raise IndexError(f'Row Indices must be bigger than 0 and smaller than length({self.num_of_rows})')
        if scalar == 0:
            raise ZeroDivisionError("Matrix.divide_row(): Can't divide by zero !")
        for i in range(self.num_of_columns):
            self.matrix[row][i] /= scalar

    def divide_all(self, expression):
        if expression == 0:
            raise ValueError(f'Cannot divide a matrix by 0.')
        for row in self._matrix:
            for index in range(len(row)):
                row[index] /= expression

    def multiply_all(self, expression):
        """
        multiplying each number in the matrix by a number of type float
        :param expression: type float, can't be 0.
        :return: Doesn't return anything (None)
        """
        for row in self._matrix:
            for index in range(len(row)):
                row[index] *= expression

    def kronecker(self, other: 'Matrix'):
        new_matrix = Matrix(dimensions=(self.num_of_rows * other._num_of_rows, self._num_of_columns * other._num_of_columns))
        row_offset, col_offset = (0, 0)
        for row in self._matrix:
            for item in row:
                for row_index in range(other._num_of_rows):
                    for col_index in range(other._num_of_columns):
                        new_matrix._matrix[row_offset + row_index][col_offset + col_index] = item * other._matrix[row_index][col_index]
                col_offset += other._num_of_columns
            col_offset = 0
            row_offset += other._num_of_rows
        return new_matrix

    def add_to_all(self, expression):
        for row in self._matrix:
            for index in range(len(row)):
                row[index] += expression

    def subtract_from_all(self, expression):
        for row in self._matrix:
            for index in range(len(row)):
                row[index] -= expression

    def apply_to_all(self, f: Callable):
        for row_index, row in enumerate(self._matrix):
            if isinstance(row, list):
                for index, item in enumerate(row):
                    row[index] = f(item)
            else:
                self._matrix[row_index] = f(row)

    def gauss(self) -> None:
        """
        Ranking a matrix is the most important part in this implementation of gaussian elimination .
        The gaussian elimination is a method for solving a set of linear equations. It is supported in this program
        via the LinearSystem class, but it uses the Matrix class for the solving process.
        """
        pivot_row = 0
        for pivot_column in range(self._num_of_columns):
            if pivot_row >= self._num_of_rows:
                break
            candidate = next(
                (row for row in range(pivot_row, self._num_of_rows)
                 if self.matrix[row][pivot_column] != 0),
                None,
            )
            if candidate is None:
                continue
            if candidate != pivot_row:
                self.replace_rows(pivot_row, candidate)
            pivot = self.matrix[pivot_row][pivot_column]
            self.divide_row(pivot, pivot_row)
            for row in range(self._num_of_rows):
                if row != pivot_row and self.matrix[row][pivot_column] != 0:
                    self.add_and_mul(row, pivot_row, -self.matrix[row][pivot_column])
            pivot_row += 1

    def __test_gauss(self) -> None:
        """
        Ranking a matrix is the most important part in this implementation of gaussian elimination .
        The gaussian elimination is a method for solving a set of linear __equations. It is supported in this program
        via the LinearSystem class, but it uses the Matrix class for the solving process.
        """
        number_of_zeroes = 0
        i = 0
        for k in range(self._num_of_columns):
            if i < self.num_of_rows and self.matrix[i][k] == 0:
                index = -1
                for t in range(i, self.num_of_rows):
                    if self._matrix[i][t] != 0:
                        index = t
                if index != -1:
                    self.replace_rows(i, index)
                    i += 1
            if self.matrix[i][k] != 0:
                self.divide_row(self.matrix[i][k], i)
            for j in range(self.num_of_rows):
                if i != j:
                    self.add_and_mul(j, i, -self.matrix[j][i])

    def get_rank(self, copy=True) -> int:
        my_matrix = self.__copy__() if copy else self
        my_matrix.gauss()
        num_of_zeroes_lines = 0
        for row in my_matrix:
            if all((item == 0 for item in row)):
                num_of_zeroes_lines += 1
        return my_matrix.num_of_rows - num_of_zeroes_lines

    def __zero_line(self, row: Iterable) -> int:
        return 1 if all((element == 0 for element in row)) else 0

    def determinant(self, rank=False) -> float:
        """
        Finds the determinant of the function, as a byproduct of ranking a copy of it.

        :param rank: If set to True, the original matrix will be ranked in the process. Default is False.
        """
        if self._num_of_rows != self._num_of_columns:
            raise ValueError('Cannot find a determinant of a non-square matrix')
        if self._num_of_rows == 2:
            return self._matrix[0][0] * self._matrix[1][1] - self._matrix[1][0] * self._matrix[0][1]
        d: float = 1
        other = self if rank else self.__copy__()
        for column_index in range(other._num_of_columns):
            pivot_row = next(
                (row for row in range(column_index, other._num_of_rows)
                 if other.matrix[row][column_index] != 0),
                None,
            )
            if pivot_row is None:
                if rank:
                    other.gauss()
                return 0
            if pivot_row != column_index:
                other.replace_rows(column_index, pivot_row)
                d = -d
            pivot = other.matrix[column_index][column_index]
            d *= pivot
            other.divide_row(pivot, column_index)
            for row in range(column_index + 1, other.num_of_rows):
                if other.matrix[row][column_index] != 0:
                    other.add_and_mul(row, column_index, -other.matrix[row][column_index])
        if rank:
            other.gauss()
        return d

    def yield_items(self):
        for row in self._matrix:
            for item in row:
                yield item

    def transpose(self):
        """ Computing the transpose of a matrix. M X N -> N X M """
        new_matrix = []
        for col in self.columns():
            new_matrix.append([item.__copy__() if hasattr(item, '__copy__') else item for item in col])
        return Matrix(new_matrix)

    def sum(self):
        """
        The sum of all of the items in the matrix.
        :return: the sum of the items ( float )
        :rtype: should be float
        """
        return sum((sum(lst) for lst in self.matrix))

    def max(self):
        """
        gets the biggest item in the matrix
        """
        if self.num_of_rows > 1:
            return max((max(row) for row in self.matrix))
        if len(self._matrix) > 0 and isinstance(self._matrix[0], Iterable) and not isinstance(self._matrix[0], str):
            return max(self._matrix[0])
        return max(self._matrix)

    def min(self):
        """
        returns the smallest value in the matrix
        """
        if self.num_of_rows > 1:
            return min(min(row) for row in self.matrix)
        if len(self._matrix) > 0 and isinstance(self._matrix[0], Iterable) and not isinstance(self._matrix[0], str):
            return min(self._matrix[0])
        return min(self._matrix)

    def average(self):
        return self.sum() / (self._num_of_rows * self._num_of_columns)

    def average_in_line(self, row_index: int):
        return sum(self._matrix[row_index]) / self._num_of_columns

    def average_in_column(self, column_index: int):
        return sum((row[column_index] for row in self._matrix)) / self._num_of_rows

    def __iadd__(self, other: 'Union[IExpression, int, float, Matrix, np.array]'):
        if isinstance(other, (int, float, IExpression)):
            self.add_to_all(other)
            return self
        elif isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError(f'Cannot Add matrices with different shapes: {self.shape} and {other.shape}')
            self.add(other)
            return self
        else:
            raise TypeError(f"Invalid type '{type(other)}' for adding matrices")

    def __add__(self, other: 'Union[IExpression, int, float, Matrix, np.array]'):
        return self.__copy__().__iadd__(other)

    def __isub__(self, other: 'Union[IExpression, int, float, Matrix, np.array]'):
        if isinstance(other, (int, float, IExpression)):
            self.subtract_from_all(other)
            return self
        elif isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError(f'Cannot Add matrices with different shapes: {self.shape} and {other.shape}')
            self.subtract(other)
            return self
        else:
            raise TypeError(f"Invalid type '{type(other)}' for subtracting matrcices")

    def __sub__(self, other: 'Union[IExpression, int, float, Matrix,np.array'):
        return self.__copy__().__isub__(other)

    def __imatmul__(self, other: 'Union[list, Matrix]'):
        return self.matmul(other)

    def __matmul__(self, other):
        return self.__copy__().matmul(other)

    def __imul__(self, other: 'Union[IExpression, int, float, Matrix, np.array]'):
        if isinstance(other, (IExpression, int, float)):
            self.multiply_all(other)
            return self
        elif isinstance(other, (Matrix, list)):
            self.multiply_element_wise(other)
            return self
        else:
            raise TypeError(f"Invalid type '{type(other)} for multiplying matrices'")

    def __mul__(self, other: 'Union[IExpression, int, float, Matrix, np.array]'):
        return self.__copy__().__imul__(other)

    def __itruediv__(self, other: 'Union[IExpression, int, float, Matrix,np.array]'):
        if other == 0:
            raise ValueError('Cannot divide a matrix by 0')
        if isinstance(other, (int, float, IExpression)):
            self.divide_all(other)
            return self
        elif isinstance(other, (Matrix, Iterable)):
            other = other if isinstance(other, Matrix) else Matrix(other)
            if self.shape != other.shape:
                raise ValueError(f"Cannot divide matrices with different shapes: {self.shape} and {other.shape}")
            for row_index, column_index in self.range():
                divisor = other._matrix[row_index][column_index]
                if divisor == 0:
                    raise ZeroDivisionError('Cannot divide by a matrix containing zero')
                self._matrix[row_index][column_index] /= divisor
            return self
        else:
            raise TypeError(f"Invalid type '{type(other)} for dividing matrices'")

    def __truediv__(self, other):
        return self.__copy__().__itruediv__(other)

    def __eq__(self, other) -> bool:
        """
        checks if two matrices are equal by overloading the '==' operator.

        :param other: other matrix
        :type other: Matrix / list / tuple,set
        :return: Returns True if the matrices are equal, otherwise it returns False.
        """
        if isinstance(other, (list, tuple, set)):
            other = Matrix(other)
        if isinstance(other, Matrix):
            if len(self.matrix) != len(other.matrix):
                return False
        else:
            raise TypeError(f'Unexpected type {type(other)} in Matrix.__eq__(). Expected types list,tuple,set or Matrix ')
        for i in range(len(self.matrix)):
            if len(self._matrix[i]) != len(other._matrix[i]):
                return False
            for j in range(len(self._matrix[i])):
                if self._matrix[i][j] != other._matrix[i][j]:
                    return False
        return True

    def __ne__(self, other) -> bool:
        """Returns True if the matrices aren't equal, and False if they're equal. Overloads the built in != operator."""
        return not self.__eq__(other)

    def __str__(self) -> str:
        """
        A visual representation of the matrix

        :return: a visual representation of the matrix, of str type.

        :rtype: str
        """
        max_length = max([2 + sum([len(str(number)) + 1 for number in row]) for row in self.matrix])
        accumulator = ''
        for row in self.matrix:
            line_aggregator = '| '
            for element in row:
                if isinstance(element, int):
                    element = float(element)
                line_aggregator += f'{element} '
            line_aggregator += ' ' * (max_length - len(line_aggregator)) + '|\n'
            accumulator += line_aggregator
        return accumulator

    def __repr__(self):
        return f'Matrix(matrix={self.matrix})'

    @staticmethod
    def random_matrix(shape: Tuple[int, int]=None, values: Tuple[Union[int, float], Union[int, float]]=(-15, 15), dtype='int'):
        if shape is None:
            shape = (random.randint(1, 5), random.randint(1, 5))
        new_matrix = Matrix(dimensions=shape)
        if dtype == 'int':
            random_method = random.randint
        elif dtype == 'float':
            random_method = random.uniform
        else:
            raise ValueError(f"invalid dtype '{dtype}', currently allowed types are 'int' and 'float'")
        for row in new_matrix:
            for index in range(len(row)):
                if dtype == 'int':
                    row[index] = random_method(values[0], values[1])
                elif dtype == 'float':
                    row[index] = random_method(values[0], values[1])
        return new_matrix

    def add(self, *matrices) -> 'Matrix':
        """
        returns the result of the addition of the current matrix and other matrices.
        Flexible with errors: if users enter a list or tuples of matrices, it accepts them too rather than
        returning a type error.

        :param: matrices: the matrices to be added. each matrix should be of type Matrix.
        :return: the result of the addition.
        :rtype: Matrix
        :raise: Raises a type error in case a matrix is not of type Matrix,list,or tuple.
        :raise: Raises an index error if the matrices aren't compatible for addition, i.e, they have different
        dimensions.
        """
        try:
            for matrix in matrices:
                if isinstance(matrix, list) or isinstance(matrix, tuple):
                    matrix = Matrix(matrix)
                if isinstance(matrix, Matrix):
                    if self.num_of_rows != matrix.num_of_rows or self.num_of_columns != matrix.num_of_columns:
                        raise IndexError
                    for row1, row2 in zip(self.matrix, matrix.matrix):
                        for i in range(min((len(row1), len(row2)))):
                            row1[i] += row2[i]
                else:
                    raise TypeError(f'Cannot add invalid type {type(matrix)}, expected types Matrix, list, or tuple.')
            return self.__copy__()
        except IndexError:
            warnings.warn(f'Matrix.add(): Tried to add two matrices with different number of num_of_rows ( fix it !)')
        except TypeError:
            warnings.warn(f'Matrix.add(): Expected types Matrix,list,tuple')

    def filtered_matrix(self, predicate: Callable[[Any], bool]=None, copy=True, get_list=False) -> 'Union[List, Matrix]':
        """ returns a new matrix object that its values were filtered by the
        , without changing the original matrix"""
        if copy:
            new_matrix = [[copy_expression(expression) for expression in row if predicate(expression)] for row in self._matrix]
        else:
            new_matrix = [[expression for expression in row if predicate(expression)] for row in self._matrix]
        if get_list:
            return new_matrix
        if new_matrix and any(len(row) != len(new_matrix[0]) for row in new_matrix):
            return new_matrix
        return Matrix(matrix=new_matrix)

    def mapped_matrix(self, func: Callable) -> 'Matrix':
        copy = self.matrix.copy()
        for index, row in enumerate(copy):
            copy[index] = [func(item) for item in row]
        return Matrix(copy)

    def foreach_item(self, func: Callable) -> 'Matrix':
        """
        Apply a certain function to all of the elements of the matrix.

        :param func: the given callable function
        :return: Returns the current object
        """
        for current_row in range(self._num_of_rows):
            for current_column in range(self._num_of_columns):
                self._matrix[current_row][current_column] = func(self._matrix[current_row][current_column])
        return self

    def subtract(self, *matrices) -> 'Matrix':
        """Similar to the add() method, it returns the result of the subtractions of the current matrix
         with the given matrices. Namely, let 'a' be the current matrix, and 'b', 'c', 'd' the given matrices,
         a-b-c-d will be returned.

         :rtype: Matrix
         """
        try:
            for matrix in matrices:
                if isinstance(matrix, list) or isinstance(matrix, tuple):
                    matrix = Matrix(matrix)
                if isinstance(matrix, Matrix):
                    if self.num_of_rows != matrix.num_of_rows or self.num_of_columns != matrix.num_of_columns:
                        raise IndexError
                    for row1, row2 in zip(self.matrix, matrix.matrix):
                        for i in range(min((len(row1), len(row2)))):
                            row1[i] -= row2[i]
                else:
                    raise TypeError
            return self.__copy__()
        except IndexError:
            warnings.warn(f'Matrix.subtract(): Tried to add two matrices with different number of num_of_rows ( fix it !)')
        except TypeError:
            warnings.warn(f'Matrix.subtract(): Expected types Matrix,list,tuple, but got {type(matrix)}')

    def columns(self):
        for column_index in range(self.num_of_columns):
            yield [self._matrix[index][column_index] for index in range(self.num_of_rows)]

    def multiply_element_wise(self, other: 'Union[Matrix, List[list], list]'):
        if isinstance(other, list):
            other = Matrix(other)
        if self.shape != other.shape:
            warnings.warn("If you want to execute matrix multiplication, use the '@' binary operator, or the __imatmul__(), __matmul__() methods")
            raise ValueError("Can't perform element-wise multiplication of matrices with different shapes. ")
        for i in range(self.num_of_rows):
            for j in range(self.num_of_columns):
                self._matrix[i][j] *= other._matrix[i][j]
        return self

    def matmul(self, other: 'Union[Matrix, List[list], list]'):
        """Matrix multiplication. Can also be done via the '@' operator. """
        if isinstance(other, Iterable) and (not isinstance(other, Matrix)):
            other = Matrix(other)
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"The matrices aren't suitable for multiplications: Shapes {self.shape} and {other.shape} ")
        result = []
        columns = list(other.columns())
        for row in self._matrix:
            new_row = []
            for col in columns:
                new_row.append(sum((row_element * col_element for row_element, col_element in zip(row, col))))
            result.append(new_row)
        return Matrix(result)

    def filter_by_indices(self, predicate: Callable[[int, int], bool]):
        """get a filtered matrix based on the indices duos, starting from (0,0)"""
        return [[copy_expression(item) for column_index, item in enumerate(row) if predicate(row_index, column_index)] for row_index, row in enumerate(self._matrix)]

    def __getitem__(self, item: Union[Callable[[Any], bool], int, Iterable[int]]):
        if isinstance(item, int):
            return self._matrix[item]
        elif isinstance(item, Callable):
            return self.filtered_matrix(predicate=item, copy=False, get_list=True)
        elif isinstance(item, Iterable):
            return [self._matrix[index] for index in item]
        else:
            raise TypeError(f"Invalid type '{type(bool)}' when accessing items of a matrix with the [] operator")

    def __setitem__(self, key, value):
        return self._matrix.__setitem__(key, value)

    def __delitem__(self, key):
        return self._matrix.__delitem__(key)

    def column(self, index: int):
        return column(self._matrix, index)

    def reversed_columns(self) -> 'Matrix':
        return Matrix(matrix=[list(reversed(row)) for row in self._matrix])

    def reversed_rows(self) -> 'Matrix':
        """
        Returns a copy of the matrix object that its lines are in a reversed order.

        :return: Returns a Matrix object that its matrix's lines are reversed compared to the original object.
        """
        return Matrix(matrix=list(reversed(self.matrix)))

    def iterate_by_columns(self) -> Iterator[Optional[Any]]:
        """Yields the elements in the order of the columns"""
        for j in range(self._num_of_columns):
            for i in range(self._num_of_rows):
                yield self._matrix[i][j]

    def range(self) -> Iterator[Tuple[int, int]]:
        """
        yields the indices of the matrix
        For example, for a matrix of dimensions 2x2, the method will yield (0,0), then (0,1), then (1,0), then (1,1)

        :return: yields a generator of the indices in the matrix.
        """
        for i in range(self._num_of_rows):
            for j in range(self._num_of_columns):
                yield (i, j)

    def __reversed__(self):
        return self.reversed_rows()

    def inverseWithNumpy(self, verbose=False):
        """ Returns the inverse of the matrix"""
        try:
            return Matrix(matrix=[list(row) for row in inv(self._matrix)])
        except LinAlgError:
            if verbose:
                warnings.warn('The matrix has no inverse')
            return None

    @staticmethod
    def unit_matrix(n: int) -> 'Matrix':
        zeroes_matrix = Matrix(dimensions=(n, n))
        for i in range(n):
            zeroes_matrix._matrix[i][i] = 1
        return zeroes_matrix

    @staticmethod
    def is_unit_matrix(given_matrix: 'Matrix') -> bool:
        """Checking whether the matrix is a unit matrix"""
        if given_matrix._num_of_rows != given_matrix._num_of_columns:
            return False
        for row_index, row in enumerate(given_matrix):
            for col_index, item in enumerate(row):
                if row_index == col_index:
                    if item != 1:
                        return False
                elif item != 0:
                    return False
        return True

    def inverse(self):
        """Finding the inverse of the matrix"""
        if self._num_of_rows != self._num_of_columns:
            return self.inverseWithNumpy()
        n: int = self._num_of_rows
        unit_matrix = Matrix.unit_matrix(n)
        number_of_zeroes = 0
        my_matrix = self.__copy__()
        for i in range(my_matrix._num_of_rows):
            if i < my_matrix.num_of_columns and my_matrix.matrix[i][i] == 0:
                index = my_matrix.__get_starting_item(i)
                if index != -1:
                    my_matrix.replace_rows(i, index)
                    unit_matrix.replace_rows(i, index)
                else:
                    return None
            if my_matrix.matrix[i][i] != 0:
                unit_matrix.divide_row(my_matrix.matrix[i][i], i)
                my_matrix.divide_row(my_matrix.matrix[i][i], i)
            for j in range(my_matrix.num_of_rows):
                if i != j:
                    unit_matrix.add_and_mul(j, i, -my_matrix.matrix[j][i])
                    my_matrix.add_and_mul(j, i, -my_matrix.matrix[j][i])
        if not Matrix.is_unit_matrix(my_matrix):
            return None
        return unit_matrix

    def __len__(self) -> int:
        """Return the number of rows, matching Python's collection protocol."""
        return self._num_of_rows

    def __copy__(self) -> 'Matrix':
        if not isinstance(self._matrix[0], list):
            return Matrix([copy_expression(item) for item in self._matrix])
        new_matrix = []
        for row in self._matrix:
            new_row = []
            for item in row:
                if hasattr(item, '__copy__'):
                    new_row.append(item.__copy__())
                elif hasattr(item, 'copy'):
                    new_row.append(item.copy())
                else:
                    new_row.append(item)
            new_matrix.append(new_row)
        return Matrix(new_matrix)

def generate_jacobian(functions, variables):
    if len(functions) != len(variables):
        raise ValueError('The Jacobian matrix must be nxn, so you need to enter an equal number of functions and variables_dict')
    return [[func.partial_derivative(variable) for variable in variables] for func in functions]

def approximate_jacobian(functions, values, h=0.001):
    result_jacobian = []
    for f in functions:
        temp = f(*values)
        new_list = []
        for index, variable in enumerate(values):
            new_list.append((f(*values[:index] + [values[index] + h] + values[index + 1:]) - temp) / h)
        result_jacobian.append(new_list)
    return result_jacobian

def generate_polynomial_matrix(equations: 'Union[Iterable[Union[str,Poly,Mono]],Iterable[Union[str, Poly, Mono]]]') -> 'Matrix':
    """Creating a matrix of polynomials from a collection of equations"""
    if isinstance(equations[0], str):
        return Matrix(matrix=[poly_from_str(equation_to_one_side(equation)) for equation in equations])
    return Matrix(matrix=equations)

def broyden(functions, initial_values, h: float=0.0001, epsilon: float=1e-05, nmax: int=10000):
    if all((abs(initial_value) <= epsilon for initial_value in initial_values)):
        return initial_values
    initial_jacobian = Matrix(approximate_jacobian(functions, initial_values, h))
    initial_inverse = initial_jacobian.inverse()
    for _ in range(1):
        if all((abs(initial_value) <= epsilon for initial_value in initial_values)):
            return initial_values
        F_x0 = [[f(*initial_values)] for f in functions]
        F_x0_matrix = Matrix(F_x0)
        initial_values_matrix = Matrix(initial_values)
        current_values = initial_values_matrix - initial_inverse * F_x0
        F_x1 = [f(*current_values) for f in functions]
        y1 = [item1 - item2 for item1, item2 in zip(F_x1, F_x0)]
        s1 = [item1 - item2 for item1, item2 in zip(initial_values, current_values)]
        temp = s1 * initial_inverse * y1
        print(temp)
