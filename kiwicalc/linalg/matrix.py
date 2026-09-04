from __future__ import annotations
import random
import warnings
from dataclasses import dataclass
from numbers import Number
from typing import Union, Tuple, List, Optional, Any, Callable, Iterator, Iterable
import numpy as np
from numpy.linalg import inv, LinAlgError

from kiwicalc.core.interfaces import IExpression
from kiwicalc.core.utils import copy_expression
from kiwicalc.parsing.parse_equation import equation_to_one_side
from kiwicalc.parsing.parse_expression import poly_from_str


@dataclass(frozen=True)
class LinearSolveResult:
    """Optional diagnostics returned by matrix solving methods."""

    solution: 'Matrix'
    residual_norm: float
    rank: int
    condition_number: float
    method: str

    @property
    def is_exact(self) -> bool:
        """Whether the residual is at floating-point noise level."""
        return self.residual_norm <= 1e-12


@dataclass(frozen=True)
class LUDecomposition:
    """Pivoted LU factors satisfying ``P @ A = L @ U``."""

    permutation: 'Matrix'
    lower: 'Matrix'
    upper: 'Matrix'

    def __iter__(self):
        yield self.permutation
        yield self.lower
        yield self.upper

    def reconstruct(self) -> 'Matrix':
        """Reconstruct the original matrix from the factors."""
        return self.permutation.T @ self.lower @ self.upper


@dataclass(frozen=True)
class QRDecomposition:
    """Orthogonal/unitary and upper-triangular QR factors."""

    q: 'Matrix'
    r: 'Matrix'

    def __iter__(self):
        yield self.q
        yield self.r

    def reconstruct(self) -> 'Matrix':
        return self.q @ self.r


@dataclass(frozen=True)
class SVDDecomposition:
    """Singular-value decomposition factors ``U``, ``s``, and ``Vt``."""

    u: 'Matrix'
    singular_values: Tuple[float, ...]
    vt: 'Matrix'

    def __iter__(self):
        yield self.u
        yield self.singular_values
        yield self.vt

    @property
    def sigma(self) -> 'Matrix':
        sigma = np.zeros((self.u.num_of_columns, self.vt.num_of_rows))
        for index, value in enumerate(self.singular_values):
            sigma[index, index] = value
        return Matrix(sigma.tolist())

    def reconstruct(self) -> 'Matrix':
        return self.u @ self.sigma @ self.vt


@dataclass(frozen=True)
class EigenDecomposition:
    """Eigenvalues and column-wise eigenvectors."""

    eigenvalues: Tuple[Any, ...]
    eigenvectors: 'Matrix'

    def __iter__(self):
        yield self.eigenvalues
        yield self.eigenvectors


@dataclass(frozen=True)
class VectorSpaceBasis:
    """A basis represented by column vectors in a shared ambient space."""

    vectors: Tuple['Matrix', ...]
    ambient_dimension: int
    space: str = 'vector'

    def __post_init__(self):
        if self.ambient_dimension <= 0:
            raise ValueError('ambient_dimension must be positive')
        if any(vector.shape != (self.ambient_dimension, 1) for vector in self.vectors):
            raise ValueError('Basis vectors must be column matrices in the declared ambient space')

    @property
    def dimension(self) -> int:
        return len(self.vectors)

    @property
    def is_trivial(self) -> bool:
        return not self.vectors

    @property
    def matrix(self):
        """Return vectors as matrix columns, or ``None`` for the trivial space."""
        if self.is_trivial:
            return None
        return Matrix(self.to_numpy().tolist())

    def to_numpy(self) -> np.ndarray:
        if self.is_trivial:
            return np.empty((self.ambient_dimension, 0))
        return np.column_stack([vector.to_numpy().reshape(-1) for vector in self.vectors])

    def __len__(self):
        return self.dimension

    def __iter__(self):
        return iter(self.vectors)

    def __getitem__(self, index):
        return self.vectors[index]


@dataclass(frozen=True)
class GramSchmidtStep:
    """One explanatory step in Gram–Schmidt orthonormalization."""

    source_index: int
    original: 'Matrix'
    projection_coefficients: Tuple[Any, ...]
    orthogonal: 'Matrix'
    normalized: Optional['Matrix']
    dependent: bool


@dataclass(frozen=True)
class GramSchmidtResult:
    """An orthonormal basis together with its construction steps."""

    basis: VectorSpaceBasis
    steps: Tuple[GramSchmidtStep, ...]


@dataclass(frozen=True)
class ProjectionResult:
    """Projection onto a column space and its residual diagnostics."""

    projected: 'Matrix'
    residual: 'Matrix'
    coefficients: 'Matrix'
    residual_norm: float

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
    """A rectangular matrix supporting numeric and symbolic values.

    Arithmetic operators and methods such as :meth:`rref`, :meth:`hadamard`,
    and :meth:`matmul` return new matrices. Explicit row-operation methods,
    ``add``, ``subtract``, and methods ending in ``_inplace`` mutate the current
    matrix and return it, allowing optional method chaining.
    """

    def __init__(self, matrix: Union[list, str, tuple]=None, dimensions=None, copy_elements=False):
        if matrix is None and dimensions is None:
            raise ValueError('Cannot create an empty Matrix')
        if matrix is not None and dimensions is not None:
            raise ValueError("Pass either 'matrix' or 'dimensions', not both")
        if matrix is not None:
            if isinstance(matrix, str):
                dimensions = matrix
            else:
                self._set_matrix(matrix, copy_elements=copy_elements)
        if dimensions is not None:
            self._num_of_rows, self._num_of_columns = self._parse_dimensions(dimensions)
            self._matrix: List[list] = [
                [0 for _ in range(self.num_of_columns)]
                for _ in range(self.num_of_rows)
            ]

    @staticmethod
    def _parse_dimensions(dimensions):
        if isinstance(dimensions, str):
            cleaned = dimensions.strip().lower().replace(' ', '')
            separator = 'x' if cleaned.count('x') == 1 else ',' if cleaned.count(',') == 1 else None
            if separator is None:
                raise ValueError("Matrix dimensions must look like '2x3' or '2,3'")
            parts = cleaned.split(separator)
        elif isinstance(dimensions, (tuple, list)):
            if len(dimensions) != 2:
                raise ValueError('Matrix dimensions require exactly two values')
            parts = dimensions
        else:
            raise TypeError(f'Invalid type {type(dimensions)} for matrix dimensions')
        try:
            rows, columns = (int(parts[0]), int(parts[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError('Matrix dimensions must be integers') from exc
        if any(isinstance(value, bool) or str(value).strip() != str(int(value)) for value in parts):
            raise ValueError('Matrix dimensions must be integers')
        if rows <= 0 or columns <= 0:
            raise ValueError('Matrix dimensions must be positive')
        return rows, columns

    def _set_matrix(self, matrix, *, copy_elements=False):
        if isinstance(matrix, (str, bytes)) or not isinstance(matrix, Iterable):
            raise TypeError('Matrix data must be an iterable of values or rows')
        values = list(matrix)
        if not values:
            raise ValueError('Cannot create an empty Matrix')
        first_is_row = isinstance(values[0], Iterable) and not isinstance(values[0], (str, bytes))
        if first_is_row:
            rows = [list(row) if isinstance(row, Iterable) and not isinstance(row, (str, bytes)) else None for row in values]
            if any(row is None for row in rows):
                raise ValueError('Matrix rows must all be iterables')
        else:
            if any(isinstance(item, Iterable) and not isinstance(item, (str, bytes)) for item in values):
                raise ValueError('Matrix data cannot mix rows and scalar values')
            rows = [values]
        if not rows[0]:
            raise ValueError('Matrix rows cannot be empty')
        expected_columns = len(rows[0])
        if any(len(row) != expected_columns for row in rows):
            raise ValueError('Matrix rows must all have the same length')
        copier = copy_expression if copy_elements else lambda item: item
        self._matrix = [[copier(item) for item in row] for row in rows]
        self._num_of_rows = len(rows)
        self._num_of_columns = expected_columns

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, mat):
        self._set_matrix(mat)

    @property
    def num_of_rows(self):
        return self._num_of_rows

    @property
    def num_of_columns(self):
        return self._num_of_columns

    @property
    def shape(self):
        return (self._num_of_rows, self._num_of_columns)

    @classmethod
    def row(cls, values, *, copy_elements=False) -> 'Matrix':
        """Create a one-row matrix from a one-dimensional iterable."""
        items = list(values)
        if not items:
            raise ValueError('A row matrix requires at least one value')
        return cls([items], copy_elements=copy_elements)

    @classmethod
    def column_vector(cls, values, *, copy_elements=False) -> 'Matrix':
        """Create a one-column matrix from a one-dimensional iterable."""
        items = list(values)
        if not items:
            raise ValueError('A column matrix requires at least one value')
        return cls([[item] for item in items], copy_elements=copy_elements)

    @classmethod
    def zeros(cls, rows, columns=None) -> 'Matrix':
        """Create a zero matrix from ``(rows, columns)`` or two integers."""
        dimensions = rows if columns is None else (rows, columns)
        return cls(dimensions=dimensions)

    @classmethod
    def identity(cls, size: int) -> 'Matrix':
        """Create a square identity matrix."""
        return cls.unit_matrix(size)

    @classmethod
    def diagonal(cls, values) -> 'Matrix':
        """Create a square matrix with the supplied diagonal values."""
        items = list(values)
        if not items:
            raise ValueError('A diagonal matrix requires at least one value')
        return cls(np.diag(items).tolist())

    @classmethod
    def from_numpy(cls, array, *, copy=True) -> 'Matrix':
        """Create a matrix from a one- or two-dimensional NumPy array."""
        if not isinstance(array, np.ndarray):
            raise TypeError('from_numpy expects a numpy.ndarray')
        if array.ndim not in (1, 2):
            raise ValueError('A matrix requires a one- or two-dimensional array')
        return cls(array.tolist(), copy_elements=copy)

    def to_numpy(self, dtype=None, *, copy=True) -> np.ndarray:
        """Return the matrix as a NumPy array.

        Numeric matrices infer a numeric dtype. Symbolic matrices naturally use
        NumPy's object dtype unless a compatible ``dtype`` is requested.
        """
        array = np.asarray(self._matrix, dtype=dtype)
        return array.copy() if copy else array

    def to_list(self, *, copy_elements=False) -> List[list]:
        """Return a rectangular list representation detached from matrix rows."""
        copier = copy_expression if copy_elements else lambda item: item
        return [[copier(item) for item in row] for row in self._matrix]

    def copy(self) -> 'Matrix':
        """Return an independent matrix copy."""
        return self.__copy__()

    def as_affine(self, translation=None, *, homogeneous=False):
        """Use this matrix as the linear or homogeneous part of an affine transform.

        A 2×2 or 3×3 matrix is treated as a linear map by default. Set
        ``homogeneous=True`` for a homogeneous 3×3 matrix; homogeneous 4×4
        matrices are recognized automatically.
        """
        from kiwicalc.linalg.transforms import AffineTransformation
        if homogeneous or self.shape == (4, 4):
            if translation is not None:
                raise ValueError("translation is already encoded by a homogeneous matrix")
            return AffineTransformation.from_matrix(self)
        return AffineTransformation.from_linear(self, translation)

    @property
    def T(self) -> 'Matrix':
        """Shorthand for :meth:`transpose`, mirroring NumPy's API."""
        return self.transpose()

    def conjugate(self) -> 'Matrix':
        """Return the element-wise complex conjugate without mutating the matrix."""
        return Matrix([
            [item.conjugate() if callable(getattr(item, 'conjugate', None)) else item for item in row]
            for row in self._matrix
        ])

    @property
    def H(self) -> 'Matrix':
        """Return the conjugate transpose (Hermitian adjoint)."""
        return self.conjugate().transpose()

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
        return self

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
        return self

    def __get_starting_item(self, i: int):
        """ Get the index of the first element in a row that is not 0"""
        for j in range(i, self._num_of_rows):
            if self.matrix[j][i] != 0:
                return j
        return -1

    def multiply_row(self, expression, row: int):
        if row < 0 or row >= self._num_of_rows:
            raise IndexError(f'Invalid row index {row}. Expected an index between 0 and {self._num_of_rows - 1}')
        for i in range(self.num_of_columns):
            self.matrix[row][i] *= expression
        return self

    def divide_row(self, scalar, row: int):
        """
        Divide a row by a scalar. Row indices are zero-based.
        :param scalar: type float
        :param row: zero-based row index.
        :return: the current matrix
        """
        if row < 0 or row >= self.num_of_rows:
            raise IndexError(f'Invalid row index {row}. Expected an index between 0 and {self.num_of_rows - 1}')
        if scalar == 0:
            raise ZeroDivisionError("Matrix.divide_row(): Can't divide by zero !")
        for i in range(self.num_of_columns):
            self.matrix[row][i] /= scalar
        return self

    def divide_all(self, expression):
        if expression == 0:
            raise ValueError(f'Cannot divide a matrix by 0.')
        for row in self._matrix:
            for index in range(len(row)):
                row[index] /= expression
        return self

    def multiply_all(self, expression):
        """
        multiplying each number in the matrix by a number of type float
        :param expression: type float, can't be 0.
        :return: Doesn't return anything (None)
        """
        for row in self._matrix:
            for index in range(len(row)):
                row[index] *= expression
        return self

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
        return self

    def subtract_from_all(self, expression):
        for row in self._matrix:
            for index in range(len(row)):
                row[index] -= expression
        return self

    def apply_to_all(self, f: Callable):
        for row_index, row in enumerate(self._matrix):
            if isinstance(row, list):
                for index, item in enumerate(row):
                    row[index] = f(item)
            else:
                self._matrix[row_index] = f(row)
        return self

    @staticmethod
    def _numeric_tolerance(values, tolerance=None):
        if tolerance is not None:
            if tolerance < 0:
                raise ValueError('tolerance must be non-negative')
            return float(tolerance)
        try:
            array = np.asarray(values)
            if not np.issubdtype(array.dtype, np.number):
                array = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            return 0.0
        scale = max(1.0, float(np.linalg.norm(array, ord=np.inf)))
        return np.finfo(float).eps * max(array.shape) * scale

    @staticmethod
    def _is_zero(value, tolerance=0.0):
        if tolerance:
            try:
                return abs(value) <= tolerance
            except TypeError:
                pass
        return value == 0

    def _as_numeric_array(self) -> np.ndarray:
        """Return numeric matrix data or raise a friendly type error."""
        try:
            array = np.asarray(self._matrix)
            if not np.issubdtype(array.dtype, np.number):
                array = np.asarray(self._matrix, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError('This operation requires a numeric matrix') from exc
        if np.any(~np.isfinite(array)):
            raise ValueError('Matrix values must be finite')
        return array

    def _as_rhs_array(self, right_hand_side) -> np.ndarray:
        if isinstance(right_hand_side, Matrix):
            array = right_hand_side._as_numeric_array()
        else:
            source = getattr(right_hand_side, 'direction', right_hand_side)
            if isinstance(source, (str, bytes)):
                raise TypeError('The right-hand side must contain numeric values')
            if not isinstance(source, (Number, np.number, np.ndarray)):
                try:
                    source = list(source)
                except TypeError as exc:
                    raise TypeError('The right-hand side must contain numeric values') from exc
            try:
                array = np.asarray(source)
                if not np.issubdtype(array.dtype, np.number):
                    array = np.asarray(source, dtype=float)
            except (TypeError, ValueError) as exc:
                raise TypeError('The right-hand side must contain numeric values') from exc
        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            array = array.reshape(-1, 1)
        elif array.ndim != 2:
            raise ValueError('The right-hand side must be one- or two-dimensional')
        if array.shape[0] != self._num_of_rows:
            raise ValueError(
                f'Right-hand side has {array.shape[0]} rows; expected {self._num_of_rows}'
            )
        if np.any(~np.isfinite(array)):
            raise ValueError('Right-hand-side values must be finite')
        return array

    def gauss(self, tolerance=None) -> 'Matrix':
        """
        Reduce this matrix in place to reduced row-echelon form.

        This historical mutating API is retained for compatibility. Prefer
        :meth:`rref` when the original matrix should remain unchanged.
        """
        effective_tolerance = self._numeric_tolerance(self._matrix, tolerance)
        numeric = all(isinstance(item, (Number, np.number)) for item in self.yield_items())
        pivot_row = 0
        for pivot_column in range(self._num_of_columns):
            if pivot_row >= self._num_of_rows:
                break
            candidates = [
                row for row in range(pivot_row, self._num_of_rows)
                if not self._is_zero(self.matrix[row][pivot_column], effective_tolerance)
            ]
            candidate = max(candidates, key=lambda row: abs(self.matrix[row][pivot_column])) if numeric and candidates else (candidates[0] if candidates else None)
            if candidate is None:
                continue
            if candidate != pivot_row:
                self.replace_rows(pivot_row, candidate)
            pivot = self.matrix[pivot_row][pivot_column]
            self.divide_row(pivot, pivot_row)
            for row in range(self._num_of_rows):
                if row != pivot_row and not self._is_zero(self.matrix[row][pivot_column], effective_tolerance):
                    self.add_and_mul(row, pivot_row, -self.matrix[row][pivot_column])
            if effective_tolerance:
                for row in range(self._num_of_rows):
                    for column_index in range(self._num_of_columns):
                        if self._is_zero(self.matrix[row][column_index], effective_tolerance):
                            self.matrix[row][column_index] = 0.0
            pivot_row += 1
        return self

    def rref(self, tolerance=None, *, copy=True) -> 'Matrix':
        """Return the reduced row-echelon form.

        By default a new matrix is returned. Set ``copy=False`` to reduce the
        current matrix in place and return it.
        """
        result = self.__copy__() if copy else self
        return result.gauss(tolerance=tolerance)

    def explain_rref(self, tolerance=None):
        """Return each elementary row operation used to compute the RREF."""
        from kiwicalc.linalg.visualization import explain_rref
        return explain_rref(self, tolerance=tolerance)

    def pivot_columns(self, tolerance=None) -> Tuple[int, ...]:
        """Return the zero-based pivot-column indices of the matrix."""
        reduced = self.rref(tolerance=tolerance)
        effective_tolerance = self._numeric_tolerance(reduced.matrix, tolerance)
        pivots = []
        for row in reduced.matrix:
            pivot = next(
                (index for index, value in enumerate(row)
                 if not self._is_zero(value, effective_tolerance)),
                None,
            )
            if pivot is not None:
                pivots.append(pivot)
        return tuple(pivots)

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

    def get_rank(self, copy=True, tolerance=None) -> int:
        my_matrix = self.__copy__() if copy else self
        effective_tolerance = self._numeric_tolerance(my_matrix.matrix, tolerance)
        my_matrix.gauss(tolerance=effective_tolerance)
        num_of_zeroes_lines = 0
        for row in my_matrix:
            if all((self._is_zero(item, effective_tolerance) for item in row)):
                num_of_zeroes_lines += 1
        return my_matrix.num_of_rows - num_of_zeroes_lines

    def rank(self, tolerance=None) -> int:
        """Return the numerical or symbolic rank without mutating the matrix."""
        return self.get_rank(copy=True, tolerance=tolerance)

    def column_space(self, tolerance=None) -> VectorSpaceBasis:
        """Return a basis formed from independent columns of the original matrix."""
        pivots = self.pivot_columns(tolerance=tolerance)
        vectors = tuple(
            Matrix.column_vector(
                [self._matrix[row_index][column_index] for row_index in range(self._num_of_rows)],
                copy_elements=True,
            )
            for column_index in pivots
        )
        return VectorSpaceBasis(vectors, self._num_of_rows, 'column')

    def row_space(self, tolerance=None) -> VectorSpaceBasis:
        """Return a basis of nonzero rows from reduced row-echelon form."""
        reduced = self.rref(tolerance=tolerance)
        effective_tolerance = self._numeric_tolerance(reduced.matrix, tolerance)
        vectors = tuple(
            Matrix.column_vector(row, copy_elements=True)
            for row in reduced.matrix
            if any(not self._is_zero(value, effective_tolerance) for value in row)
        )
        return VectorSpaceBasis(vectors, self._num_of_columns, 'row')

    def null_space(self, tolerance=None, *, method='auto') -> VectorSpaceBasis:
        """Return a basis for vectors ``x`` satisfying ``A @ x = 0``.

        ``method='auto'`` uses stable SVD for numeric matrices and RREF for
        symbolic matrices. Select ``'rref'`` for a simpler algebraic basis.
        """
        normalized_method = str(method).strip().lower()
        if normalized_method not in ('auto', 'svd', 'rref'):
            raise ValueError("null-space method must be 'auto', 'svd', or 'rref'")
        numeric = all(isinstance(item, (Number, np.number)) for item in self.yield_items())
        use_svd = normalized_method == 'svd' or (normalized_method == 'auto' and numeric)
        if use_svd:
            coefficients = self._as_numeric_array()
            _, singular_values, vh = np.linalg.svd(coefficients, full_matrices=True)
            if tolerance is None:
                largest = float(singular_values[0]) if singular_values.size else 0.0
                effective_tolerance = np.finfo(float).eps * max(coefficients.shape) * largest
            else:
                effective_tolerance = self._numeric_tolerance(coefficients, tolerance)
            rank = int(np.count_nonzero(singular_values > effective_tolerance))
            basis_array = vh[rank:].conj().T
            vectors = tuple(
                Matrix.column_vector(np.real_if_close(basis_array[:, index]).tolist())
                for index in range(basis_array.shape[1])
            )
            return VectorSpaceBasis(vectors, self._num_of_columns, 'null')

        reduced = self.rref(tolerance=tolerance)
        pivots = reduced.pivot_columns(tolerance=tolerance)
        free_columns = [index for index in range(self._num_of_columns) if index not in pivots]
        vectors = []
        for free_column in free_columns:
            values = [0 for _ in range(self._num_of_columns)]
            values[free_column] = 1
            for row_index, pivot_column in enumerate(pivots):
                values[pivot_column] = -reduced.matrix[row_index][free_column]
            vectors.append(Matrix.column_vector(values, copy_elements=True))
        return VectorSpaceBasis(tuple(vectors), self._num_of_columns, 'null')

    def basis(self, space='column', tolerance=None) -> VectorSpaceBasis:
        """Return the requested ``'column'``, ``'row'``, or ``'null'`` basis."""
        normalized = str(space).strip().lower().replace('_space', '')
        methods = {
            'column': self.column_space,
            'columns': self.column_space,
            'row': self.row_space,
            'rows': self.row_space,
            'null': self.null_space,
            'kernel': self.null_space,
        }
        try:
            method = methods[normalized]
        except KeyError as exc:
            raise ValueError("space must be 'column', 'row', or 'null'") from exc
        return method(tolerance=tolerance)

    def is_independent(self, axis='columns', tolerance=None) -> bool:
        """Return whether all rows or columns are linearly independent."""
        normalized = str(axis).strip().lower()
        if normalized in ('column', 'columns'):
            expected_rank = self._num_of_columns
        elif normalized in ('row', 'rows'):
            expected_rank = self._num_of_rows
        else:
            raise ValueError("axis must be 'rows' or 'columns'")
        return self.rank(tolerance=tolerance) == expected_rank

    def orthonormalize(self, axis='columns', tolerance=None, *, return_steps=False, drop_dependent=True):
        """Orthonormalize rows or columns using modified Gram–Schmidt.

        The returned :class:`VectorSpaceBasis` always stores its vectors as
        columns. Set ``return_steps=True`` for projection coefficients and each
        intermediate orthogonal vector.
        """
        coefficients = self._as_numeric_array()
        normalized_axis = str(axis).strip().lower()
        if normalized_axis in ('column', 'columns'):
            source_vectors = [coefficients[:, index] for index in range(self._num_of_columns)]
            ambient_dimension = self._num_of_rows
            space_name = 'column'
        elif normalized_axis in ('row', 'rows'):
            source_vectors = [coefficients[index, :] for index in range(self._num_of_rows)]
            ambient_dimension = self._num_of_columns
            space_name = 'row'
        else:
            raise ValueError("axis must be 'rows' or 'columns'")

        dtype = complex if np.iscomplexobj(coefficients) else float
        effective_tolerance = self._numeric_tolerance(coefficients, tolerance)
        orthonormal_vectors = []
        steps = []
        for source_index, source in enumerate(source_vectors):
            original = source.astype(dtype, copy=True)
            orthogonal = original.copy()
            projection_coefficients = []
            for existing in orthonormal_vectors:
                coefficient = np.vdot(existing, orthogonal)
                projection_coefficients.append(np.real_if_close(coefficient).item())
                orthogonal -= coefficient * existing
            norm = float(np.linalg.norm(orthogonal))
            dependent = norm <= effective_tolerance
            normalized_vector = None if dependent else orthogonal / norm
            if dependent and not drop_dependent:
                raise ValueError(f'Vector at {space_name} index {source_index} is linearly dependent')
            if normalized_vector is not None:
                orthonormal_vectors.append(normalized_vector)

            original_matrix = Matrix.column_vector(np.real_if_close(original).tolist())
            orthogonal_matrix = Matrix.column_vector(np.real_if_close(orthogonal).tolist())
            normalized_matrix = (
                None if normalized_vector is None
                else Matrix.column_vector(np.real_if_close(normalized_vector).tolist())
            )
            steps.append(GramSchmidtStep(
                source_index=source_index,
                original=original_matrix,
                projection_coefficients=tuple(projection_coefficients),
                orthogonal=orthogonal_matrix,
                normalized=normalized_matrix,
                dependent=dependent,
            ))

        basis_vectors = tuple(
            Matrix.column_vector(np.real_if_close(vector).tolist())
            for vector in orthonormal_vectors
        )
        basis = VectorSpaceBasis(basis_vectors, ambient_dimension, space_name)
        return GramSchmidtResult(basis, tuple(steps)) if return_steps else basis

    def project_onto(self, vectors, *, return_info=False, rcond=None):
        """Project one or more vectors onto this matrix's column space."""
        coefficients = self._as_numeric_array()
        targets = self._as_rhs_array(vectors)
        try:
            weights, _, _, _ = np.linalg.lstsq(coefficients, targets, rcond=rcond)
        except LinAlgError as exc:
            raise ValueError('Projection could not be computed') from exc
        projected_values = coefficients @ weights
        residual_values = targets - projected_values
        projected = Matrix(np.real_if_close(projected_values).tolist())
        if not return_info:
            return projected
        return ProjectionResult(
            projected=projected,
            residual=Matrix(np.real_if_close(residual_values).tolist()),
            coefficients=Matrix(np.real_if_close(weights).tolist()),
            residual_norm=float(np.linalg.norm(residual_values)),
        )

    def __zero_line(self, row: Iterable) -> int:
        return 1 if all((element == 0 for element in row)) else 0

    def determinant(self, rank=False, tolerance=None) -> float:
        """
        Finds the determinant of the function, as a byproduct of ranking a copy of it.

        :param rank: If set to True, the original matrix will be ranked in the process. Default is False.
        """
        if self._num_of_rows != self._num_of_columns:
            raise ValueError('Cannot find a determinant of a non-square matrix')
        if self._num_of_rows == 1:
            return self._matrix[0][0]
        d: float = 1
        other = self if rank else self.__copy__()
        effective_tolerance = self._numeric_tolerance(other.matrix, tolerance)
        numeric = all(isinstance(item, (Number, np.number)) for item in other.yield_items())
        for column_index in range(other._num_of_columns):
            candidates = [
                row for row in range(column_index, other._num_of_rows)
                if not self._is_zero(other.matrix[row][column_index], effective_tolerance)
            ]
            pivot_row = max(candidates, key=lambda row: abs(other.matrix[row][column_index])) if numeric and candidates else (candidates[0] if candidates else None)
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
                if not self._is_zero(other.matrix[row][column_index], effective_tolerance):
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
        if isinstance(other, (Number, np.number, IExpression)):
            self.add_to_all(other)
            return self
        elif isinstance(other, Matrix) or (isinstance(other, Iterable) and not isinstance(other, (str, bytes))):
            other = other if isinstance(other, Matrix) else Matrix(other)
            if self.shape != other.shape:
                raise ValueError(f'Cannot Add matrices with different shapes: {self.shape} and {other.shape}')
            for row_index, column_index in self.range():
                self._matrix[row_index][column_index] += other._matrix[row_index][column_index]
            return self
        else:
            raise TypeError(f"Invalid type '{type(other)}' for adding matrices")

    def __add__(self, other: 'Union[IExpression, int, float, Matrix, np.array]'):
        return self.__copy__().__iadd__(other)

    def __isub__(self, other: 'Union[IExpression, int, float, Matrix, np.array]'):
        if isinstance(other, (Number, np.number, IExpression)):
            self.subtract_from_all(other)
            return self
        elif isinstance(other, Matrix) or (isinstance(other, Iterable) and not isinstance(other, (str, bytes))):
            other = other if isinstance(other, Matrix) else Matrix(other)
            if self.shape != other.shape:
                raise ValueError(f'Cannot subtract matrices with different shapes: {self.shape} and {other.shape}')
            for row_index, column_index in self.range():
                self._matrix[row_index][column_index] -= other._matrix[row_index][column_index]
            return self
        else:
            raise TypeError(f"Invalid type '{type(other)}' for subtracting matrcices")

    def __sub__(self, other):
        return self.__copy__().__isub__(other)

    def __imatmul__(self, other: 'Union[list, Matrix]'):
        return self.matmul(other)

    def __matmul__(self, other):
        return self.__copy__().matmul(other)

    def __imul__(self, other: 'Union[IExpression, int, float, Matrix, np.array]'):
        if isinstance(other, (IExpression, Number, np.number)):
            self.multiply_all(other)
            return self
        elif isinstance(other, Matrix) or (isinstance(other, Iterable) and not isinstance(other, (str, bytes))):
            self.hadamard_inplace(other)
            return self
        else:
            raise TypeError(f"Invalid type '{type(other)} for multiplying matrices'")

    def __mul__(self, other: 'Union[IExpression, int, float, Matrix, np.array]'):
        return self.__copy__().__imul__(other)

    def __itruediv__(self, other: 'Union[IExpression, int, float, Matrix,np.array]'):
        if isinstance(other, (Number, np.number, IExpression)):
            if other == 0:
                raise ZeroDivisionError('Cannot divide a matrix by 0')
            self.divide_all(other)
            return self
        elif isinstance(other, Matrix) or (isinstance(other, Iterable) and not isinstance(other, (str, bytes))):
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
        if isinstance(other, (list, tuple, np.ndarray)):
            try:
                other = Matrix(other)
            except (TypeError, ValueError):
                return False
        if isinstance(other, Matrix):
            if len(self.matrix) != len(other.matrix):
                return False
        else:
            return NotImplemented
        if all(isinstance(item, (Number, np.number)) for item in self.yield_items()) and all(
            isinstance(item, (Number, np.number)) for item in other.yield_items()
        ):
            return bool(np.allclose(np.asarray(self._matrix), np.asarray(other._matrix)))
        for i in range(len(self.matrix)):
            if len(self._matrix[i]) != len(other._matrix[i]):
                return False
            for j in range(len(self._matrix[i])):
                if self._matrix[i][j] != other._matrix[i][j]:
                    return False
        return True

    def __ne__(self, other) -> bool:
        """Returns True if the matrices aren't equal, and False if they're equal. Overloads the built in != operator."""
        result = self.__eq__(other)
        return NotImplemented if result is NotImplemented else not result

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
        for matrix in matrices:
            if isinstance(matrix, (list, tuple, np.ndarray)):
                matrix = Matrix(matrix)
            if not isinstance(matrix, Matrix):
                raise TypeError(f'Cannot add invalid type {type(matrix)}, expected Matrix or a rectangular iterable')
            if self.shape != matrix.shape:
                raise ValueError(f'Cannot add matrices with different shapes: {self.shape} and {matrix.shape}')
            for row_index, column_index in self.range():
                self._matrix[row_index][column_index] += matrix._matrix[row_index][column_index]
        return self

    def filtered_matrix(self, predicate: Callable[[Any], bool]=None, copy=True, get_list=False) -> 'Union[List, Matrix]':
        """ returns a new matrix object that its values were filtered by the
        , without changing the original matrix"""
        if predicate is None:
            predicate = lambda _: True
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
        for matrix in matrices:
            if isinstance(matrix, (list, tuple, np.ndarray)):
                matrix = Matrix(matrix)
            if not isinstance(matrix, Matrix):
                raise TypeError(f'Cannot subtract invalid type {type(matrix)}, expected Matrix or a rectangular iterable')
            if self.shape != matrix.shape:
                raise ValueError(f'Cannot subtract matrices with different shapes: {self.shape} and {matrix.shape}')
            for row_index, column_index in self.range():
                self._matrix[row_index][column_index] -= matrix._matrix[row_index][column_index]
        return self

    def columns(self):
        for column_index in range(self.num_of_columns):
            yield [self._matrix[index][column_index] for index in range(self.num_of_rows)]

    def multiply_element_wise(self, other: 'Union[Matrix, List[list], list]'):
        """Multiply element-wise in place and return this matrix.

        Prefer :meth:`hadamard` for a non-mutating operation.
        """
        if not isinstance(other, Matrix):
            try:
                other = Matrix(other)
            except (TypeError, ValueError) as exc:
                raise TypeError('Element-wise multiplication expects a Matrix or rectangular iterable') from exc
        if self.shape != other.shape:
            raise ValueError(f"Cannot multiply matrices element-wise with different shapes: {self.shape} and {other.shape}")
        for i in range(self.num_of_rows):
            for j in range(self.num_of_columns):
                self._matrix[i][j] *= other._matrix[i][j]
        return self

    def hadamard_inplace(self, other) -> 'Matrix':
        """Multiply element-wise in place."""
        if not isinstance(other, Matrix):
            try:
                other = Matrix(other)
            except (TypeError, ValueError) as exc:
                raise TypeError('Hadamard multiplication expects a Matrix or rectangular iterable') from exc
        return self.multiply_element_wise(other)

    def hadamard(self, other) -> 'Matrix':
        """Return the element-wise (Hadamard) product without mutating operands."""
        return self.__copy__().hadamard_inplace(other)

    def matmul(self, other: 'Union[Matrix, List[list], list]'):
        """Matrix multiplication. Can also be done via the '@' operator. """
        if isinstance(other, Iterable) and (not isinstance(other, Matrix)):
            other = Matrix(other)
        if not isinstance(other, Matrix):
            raise TypeError(f'Cannot multiply a matrix by {type(other)}')
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
            raise TypeError(f"Invalid type '{type(item)}' when accessing items of a matrix with the [] operator")

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            raise TypeError('Matrix row indices must be integers')
        row = list(value)
        if len(row) != self._num_of_columns:
            raise ValueError(f'Matrix rows must contain {self._num_of_columns} values')
        self._matrix[key] = row

    def __delitem__(self, key):
        if not isinstance(key, int):
            raise TypeError('Matrix row indices must be integers')
        if self._num_of_rows == 1:
            raise ValueError('Cannot delete the only row from a matrix')
        del self._matrix[key]
        self._num_of_rows -= 1

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

    def inverse(self, tolerance=None):
        """Finding the inverse of the matrix"""
        if self._num_of_rows != self._num_of_columns:
            return None
        if all(isinstance(item, (Number, np.number)) for item in self.yield_items()):
            try:
                array = np.asarray(self._matrix)
                inverse = np.linalg.inv(array)
            except (LinAlgError, TypeError, ValueError):
                return None
            effective_tolerance = self._numeric_tolerance(array, tolerance)
            if np.any(~np.isfinite(inverse)):
                return None
            residual = np.linalg.norm(array @ inverse - np.eye(self._num_of_rows), ord=np.inf)
            if residual > max(effective_tolerance * 10, 1e-12):
                return None
            return Matrix(inverse.tolist())
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

    def solve(self, right_hand_side, *, return_info=False, tolerance=None):
        """Solve ``A @ x = b`` for a square numeric matrix.

        A flat right-hand side is treated as a column vector. Multiple columns
        may be supplied to solve several systems at once. The default return is
        a :class:`Matrix`; set ``return_info=True`` for diagnostics.
        """
        if self._num_of_rows != self._num_of_columns:
            raise ValueError('solve() requires a square matrix; use least_squares() for rectangular systems')
        coefficients = self._as_numeric_array()
        right = self._as_rhs_array(right_hand_side)
        effective_tolerance = self._numeric_tolerance(coefficients, tolerance)
        rank = int(np.linalg.matrix_rank(coefficients, tol=effective_tolerance))
        if rank < self._num_of_columns:
            raise ValueError('Cannot solve the system uniquely because the matrix is singular')
        try:
            values = np.linalg.solve(coefficients, right)
        except LinAlgError as exc:
            raise ValueError('Cannot solve the system uniquely because the matrix is singular') from exc
        solution = Matrix(values.tolist())
        if not return_info:
            return solution
        residual = float(np.linalg.norm(coefficients @ values - right))
        return LinearSolveResult(
            solution=solution,
            residual_norm=residual,
            rank=rank,
            condition_number=self.condition_number(),
            method='solve',
        )

    def least_squares(self, right_hand_side, *, return_info=False, rcond=None):
        """Return the minimum-norm least-squares solution of ``A @ x = b``."""
        coefficients = self._as_numeric_array()
        right = self._as_rhs_array(right_hand_side)
        try:
            values, _, rank, _ = np.linalg.lstsq(coefficients, right, rcond=rcond)
        except LinAlgError as exc:
            raise ValueError('The least-squares solution could not be computed') from exc
        solution = Matrix(values.tolist())
        if not return_info:
            return solution
        residual = float(np.linalg.norm(coefficients @ values - right))
        return LinearSolveResult(
            solution=solution,
            residual_norm=residual,
            rank=int(rank),
            condition_number=self.condition_number(),
            method='least_squares',
        )

    def visualize_least_squares(self, right_hand_side, **options):
        """Visualize observations, fitted values, and least-squares residuals."""
        from kiwicalc.linalg.visualization import visualize_least_squares
        return visualize_least_squares(self, right_hand_side, **options)

    def pseudoinverse(self, rcond=None, *, hermitian=False) -> 'Matrix':
        """Return the Moore-Penrose pseudoinverse of a numeric matrix."""
        coefficients = self._as_numeric_array()
        kwargs = {'hermitian': hermitian}
        if rcond is not None:
            if rcond < 0:
                raise ValueError('rcond must be non-negative')
            kwargs['rcond'] = rcond
        try:
            result = np.linalg.pinv(coefficients, **kwargs)
        except LinAlgError as exc:
            raise ValueError('The pseudoinverse could not be computed') from exc
        return Matrix(result.tolist())

    def condition_number(self, p=2) -> float:
        """Return the matrix condition number for the selected norm."""
        coefficients = self._as_numeric_array()
        if np.linalg.matrix_rank(coefficients) < min(coefficients.shape):
            return float('inf')
        try:
            return float(np.linalg.cond(coefficients, p=p))
        except LinAlgError as exc:
            raise ValueError('The condition number could not be computed') from exc

    def lu(self, tolerance=None) -> LUDecomposition:
        """Return a partial-pivoted LU decomposition satisfying ``P @ A = L @ U``."""
        if self._num_of_rows != self._num_of_columns:
            raise ValueError('LU decomposition currently requires a square matrix')
        coefficients = self._as_numeric_array()
        dtype = complex if np.iscomplexobj(coefficients) else float
        upper = coefficients.astype(dtype, copy=True)
        size = self._num_of_rows
        lower = np.zeros((size, size), dtype=dtype)
        permutation = np.eye(size, dtype=dtype)
        effective_tolerance = self._numeric_tolerance(coefficients, tolerance)

        for column_index in range(size):
            pivot_row = column_index + int(np.argmax(np.abs(upper[column_index:, column_index])))
            if abs(upper[pivot_row, column_index]) <= effective_tolerance:
                raise ValueError('LU decomposition requires a non-singular matrix')
            if pivot_row != column_index:
                upper[[column_index, pivot_row]] = upper[[pivot_row, column_index]]
                permutation[[column_index, pivot_row]] = permutation[[pivot_row, column_index]]
                if column_index:
                    lower[[column_index, pivot_row], :column_index] = lower[[pivot_row, column_index], :column_index]
            lower[column_index, column_index] = 1
            for row_index in range(column_index + 1, size):
                factor = upper[row_index, column_index] / upper[column_index, column_index]
                lower[row_index, column_index] = factor
                upper[row_index, column_index:] -= factor * upper[column_index, column_index:]
                upper[row_index, column_index] = 0

        lower = np.real_if_close(lower)
        upper = np.real_if_close(upper)
        permutation = np.real_if_close(permutation)
        return LUDecomposition(
            permutation=Matrix(permutation.tolist()),
            lower=Matrix(lower.tolist()),
            upper=Matrix(upper.tolist()),
        )

    def qr(self, mode='reduced') -> QRDecomposition:
        """Return a reduced or complete QR decomposition."""
        if mode not in ('reduced', 'complete'):
            raise ValueError("QR mode must be 'reduced' or 'complete'")
        coefficients = self._as_numeric_array()
        try:
            q, r = np.linalg.qr(coefficients, mode=mode)
        except LinAlgError as exc:
            raise ValueError('QR decomposition could not be computed') from exc
        return QRDecomposition(
            q=Matrix(np.real_if_close(q).tolist()),
            r=Matrix(np.real_if_close(r).tolist()),
        )

    def cholesky(self, tolerance=None) -> 'Matrix':
        """Return the lower Cholesky factor of a positive-definite matrix."""
        if self._num_of_rows != self._num_of_columns:
            raise ValueError('Cholesky decomposition requires a square matrix')
        coefficients = self._as_numeric_array()
        effective_tolerance = self._numeric_tolerance(coefficients, tolerance)
        if not np.allclose(coefficients, coefficients.conj().T, atol=effective_tolerance, rtol=0):
            raise ValueError('Cholesky decomposition requires a symmetric or Hermitian matrix')
        try:
            lower = np.linalg.cholesky(coefficients)
        except LinAlgError as exc:
            raise ValueError('Cholesky decomposition requires a positive-definite matrix') from exc
        return Matrix(np.real_if_close(lower).tolist())

    def svd(self, *, full_matrices=False) -> SVDDecomposition:
        """Return the singular-value decomposition of a numeric matrix."""
        coefficients = self._as_numeric_array()
        try:
            u, singular_values, vt = np.linalg.svd(coefficients, full_matrices=full_matrices)
        except LinAlgError as exc:
            raise ValueError('SVD did not converge') from exc
        return SVDDecomposition(
            u=Matrix(np.real_if_close(u).tolist()),
            singular_values=tuple(float(value) for value in singular_values),
            vt=Matrix(np.real_if_close(vt).tolist()),
        )

    def eigen(self) -> EigenDecomposition:
        """Return general eigenvalues and column-wise eigenvectors."""
        if self._num_of_rows != self._num_of_columns:
            raise ValueError('Eigen decomposition requires a square matrix')
        coefficients = self._as_numeric_array()
        try:
            values, vectors = np.linalg.eig(coefficients)
        except LinAlgError as exc:
            raise ValueError('Eigen decomposition did not converge') from exc
        values = np.real_if_close(values)
        vectors = np.real_if_close(vectors)
        return EigenDecomposition(tuple(values.tolist()), Matrix(vectors.tolist()))

    def visualize_transformation(self, **options):
        """Visualize how this 2D or 3D matrix transforms space."""
        from kiwicalc.linalg.visualization import visualize_transformation
        return visualize_transformation(self, **options)

    def visualize_eigenvectors(self, **options):
        """Visualize real eigenvector directions and their images."""
        from kiwicalc.linalg.visualization import visualize_eigenvectors
        return visualize_eigenvectors(self, **options)

    def visualize_svd(self, **options):
        """Visualize the four geometric stages of a 2D singular-value decomposition."""
        from kiwicalc.linalg.visualization import visualize_svd
        return visualize_svd(self, **options)

    def eig(self) -> EigenDecomposition:
        """Alias for :meth:`eigen`."""
        return self.eigen()

    def eigh(self, tolerance=None) -> EigenDecomposition:
        """Return the stable eigen decomposition of a symmetric/Hermitian matrix."""
        if self._num_of_rows != self._num_of_columns:
            raise ValueError('Hermitian eigen decomposition requires a square matrix')
        coefficients = self._as_numeric_array()
        effective_tolerance = self._numeric_tolerance(coefficients, tolerance)
        if not np.allclose(coefficients, coefficients.conj().T, atol=effective_tolerance, rtol=0):
            raise ValueError('eigh() requires a symmetric or Hermitian matrix')
        try:
            values, vectors = np.linalg.eigh(coefficients)
        except LinAlgError as exc:
            raise ValueError('Hermitian eigen decomposition did not converge') from exc
        values = np.real_if_close(values)
        vectors = np.real_if_close(vectors)
        return EigenDecomposition(tuple(values.tolist()), Matrix(vectors.tolist()))

    def __len__(self) -> int:
        """Return the number of rows, matching Python's collection protocol."""
        return self._num_of_rows

    def __iter__(self):
        """Iterate over matrix rows."""
        return iter(self._matrix)

    def __copy__(self) -> 'Matrix':
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
    functions = list(functions)
    variables = list(variables)
    if not functions:
        raise ValueError('A Jacobian requires at least one function')
    if not variables:
        raise ValueError('A Jacobian requires at least one variable')
    return [[func.partial_derivative(variable) for variable in variables] for func in functions]

def approximate_jacobian(functions, values, h=0.001):
    functions = list(functions)
    values = np.asarray(list(values), dtype=float)
    if not functions:
        raise ValueError('A Jacobian requires at least one function')
    if values.ndim != 1 or values.size == 0:
        raise ValueError('values must be a non-empty one-dimensional sequence')
    if h <= 0:
        raise ValueError('h must be positive')
    result_jacobian = []
    for function in functions:
        derivatives = []
        for index in range(values.size):
            step = h * max(1.0, abs(values[index]))
            upper = values.copy()
            lower = values.copy()
            upper[index] += step
            lower[index] -= step
            derivatives.append((function(*upper) - function(*lower)) / (2 * step))
        result_jacobian.append(derivatives)
    return result_jacobian

def generate_polynomial_matrix(equations: 'Union[Iterable[Union[str,Poly,Mono]],Iterable[Union[str, Poly, Mono]]]') -> 'Matrix':
    """Creating a matrix of polynomials from a collection of equations"""
    equations = list(equations)
    if not equations:
        raise ValueError('At least one equation is required')
    if isinstance(equations[0], str):
        return Matrix(matrix=[poly_from_str(equation_to_one_side(equation)) for equation in equations])
    return Matrix(matrix=equations)

def broyden(functions, initial_values, h: float=0.0001, epsilon: float=1e-05, nmax: int=10000):
    """Solve a square nonlinear system using the good Broyden update.

    The return value is a list matching ``initial_values``. A clear exception is
    raised when the initial Jacobian is singular or convergence is not reached.
    """
    functions = list(functions)
    x = np.asarray(list(initial_values), dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError('initial_values must be a non-empty one-dimensional sequence')
    if len(functions) != x.size:
        raise ValueError('Broyden requires one function per unknown')
    if h <= 0 or epsilon <= 0:
        raise ValueError('h and epsilon must be positive')
    if not isinstance(nmax, int) or nmax <= 0:
        raise ValueError('nmax must be a positive integer')

    def evaluate(point):
        values = np.asarray([function(*point) for function in functions], dtype=float)
        if values.shape != (x.size,) or np.any(~np.isfinite(values)):
            raise ValueError('Functions must return finite scalar values')
        return values

    residual = evaluate(x)
    if np.linalg.norm(residual, ord=np.inf) <= epsilon:
        return x.tolist()
    jacobian = np.asarray(approximate_jacobian(functions, x, h), dtype=float)

    for _ in range(nmax):
        try:
            step = np.linalg.solve(jacobian, -residual)
        except LinAlgError as exc:
            raise ValueError('Jacobian is singular during Broyden iteration') from exc
        next_x = x + step
        next_residual = evaluate(next_x)
        if np.linalg.norm(next_residual, ord=np.inf) <= epsilon:
            return next_x.tolist()
        denominator = float(step @ step)
        if denominator <= np.finfo(float).eps:
            raise RuntimeError('Broyden stalled before reaching the requested tolerance')
        jacobian += np.outer(next_residual - residual - jacobian @ step, step) / denominator
        x, residual = next_x, next_residual

    raise RuntimeError(f"Broyden did not converge within {nmax} iterations")
