"""Deterministic linear-algebra exercises with exact structured answers."""
from __future__ import annotations

from fractions import Fraction
import random

from .arrays import PDFMatrix, PDFVector
from .formatting import PDFText
from .layout import PDFMath
from .worksheet import PDFExercise


DIFFICULTIES = ('easy', 'medium', 'hard')
LINEAR_ALGEBRA_EXERCISE_TYPES = (
    'vector_arithmetic', 'dot_product', 'vector_magnitude', 'unit_vector',
    'matrix_arithmetic', 'scalar_matrix', 'matrix_multiplication',
    'determinant', 'inverse_matrix', 'solve_linear_system', 'row_reduction',
    'rank', 'linear_independence', 'basis_coordinates', 'eigenvalues',
    'eigenvector', 'projection', 'linear_transformation',
)


def _settings(difficulty, seed, rng):
    if difficulty not in DIFFICULTIES:
        raise ValueError(f'difficulty must be one of {", ".join(DIFFICULTIES)}')
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError('seed must be an integer or None')
    if rng is not None and seed is not None:
        raise ValueError('seed and the internal random generator cannot both be supplied')
    return rng or random.Random(seed)


def _nonzero(rng, low, high):
    return rng.choice([value for value in range(low, high+1) if value])


def _dimension(difficulty):
    return 3 if difficulty == 'hard' else 2


def _vector(rng, dimension, limit, *, nonzero=True):
    values = tuple(rng.randint(-limit, limit) for _ in range(dimension))
    if nonzero and not any(values):
        values = (1, *values[1:])
    return values


def _matrix(rng, rows, columns, limit):
    return tuple(tuple(rng.randint(-limit, limit) for _ in range(columns))
                 for _ in range(rows))


def _add(left, right):
    return tuple(tuple(a+b for a, b in zip(row_left, row_right))
                 for row_left, row_right in zip(left, right))


def _scale(value, matrix):
    return tuple(tuple(value*entry for entry in row) for row in matrix)


def _matmul(left, right):
    return tuple(tuple(sum(left[i][k]*right[k][j] for k in range(len(right)))
                       for j in range(len(right[0]))) for i in range(len(left)))


def _matvec(matrix, vector):
    return tuple(sum(entry*value for entry, value in zip(row, vector)) for row in matrix)


def _dot(left, right):
    return sum(a*b for a, b in zip(left, right))


def _determinant(matrix):
    if len(matrix) == 1:
        return matrix[0][0]
    return sum((-1)**column*matrix[0][column]*_determinant(tuple(
        row[:column]+row[column+1:] for row in matrix[1:]
    )) for column in range(len(matrix)))


def _rref(matrix):
    rows = [[Fraction(value) for value in row] for row in matrix]
    pivot_row = 0
    pivots = []
    for column in range(len(rows[0])):
        candidate = next((index for index in range(pivot_row, len(rows))
                          if rows[index][column]), None)
        if candidate is None:
            continue
        rows[pivot_row], rows[candidate] = rows[candidate], rows[pivot_row]
        pivot = rows[pivot_row][column]
        rows[pivot_row] = [value/pivot for value in rows[pivot_row]]
        for index, row in enumerate(rows):
            if index == pivot_row or not row[column]:
                continue
            factor = row[column]
            rows[index] = [value-factor*lead for value, lead in zip(row, rows[pivot_row])]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return tuple(tuple(row) for row in rows), tuple(pivots)


class PDFLinearAlgebraExercise(PDFExercise):
    """Base class exposing the exact generated inputs and result."""
    __slots__ = ('kind', 'difficulty', 'data')

    def __init__(self, prompt, kind, solution, difficulty, data):
        super().__init__(prompt, 'linear algebra', kind, solution=solution)
        self.kind = kind
        self.difficulty = difficulty
        self.data = data


class PDFVectorArithmetic(PDFLinearAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        dimension, limit = _dimension(difficulty), {'easy': 5, 'medium': 8, 'hard': 10}[difficulty]
        left, right = _vector(rng, dimension, limit), _vector(rng, dimension, limit)
        operation = rng.choice(('+', '-')) if difficulty != 'easy' else '+'
        result = tuple(a+b if operation == '+' else a-b for a, b in zip(left, right))
        prompt = PDFText('Compute ', PDFVector(left), f' {operation} ', PDFVector(right), '.')
        solution = PDFText('Combine corresponding entries: ', PDFVector(result), '.') if with_solution else None
        super().__init__(prompt, 'vector_arithmetic', solution, difficulty,
                         {'left': left, 'right': right, 'operation': operation, 'result': result})


class PDFDotProduct(PDFLinearAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        dimension, limit = _dimension(difficulty), {'easy': 4, 'medium': 7, 'hard': 9}[difficulty]
        left, right = _vector(rng, dimension, limit), _vector(rng, dimension, limit)
        products, result = tuple(a*b for a, b in zip(left, right)), _dot(left, right)
        prompt = PDFText('Find the dot product ', PDFVector(left), PDFMath(r'\cdot'), PDFVector(right), '.')
        working = '+'.join(str(value) if value >= 0 else f'({value})' for value in products)
        solution = PDFText(PDFMath(working+'='+str(result)), '.') if with_solution else None
        super().__init__(prompt, 'dot_product', solution, difficulty,
                         {'left': left, 'right': right, 'products': products, 'result': result})


class PDFVectorMagnitude(PDFLinearAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        if difficulty == 'easy':
            a, b, length = rng.choice(((3, 4, 5), (5, 12, 13), (8, 15, 17)))
            values = (rng.choice((-a, a)), rng.choice((-b, b)))
            squared = length*length
        else:
            values = _vector(rng, _dimension(difficulty), 7)
            squared = _dot(values, values)
        root = int(squared**.5)
        exact = str(root) if root*root == squared else rf'\sqrt{{{squared}}}'
        prompt = PDFText('Find the magnitude of ', PDFVector(values), '.')
        solution = PDFText(PDFMath(rf'\Vert v\Vert=\sqrt{{{squared}}}={exact}'), '.') if with_solution else None
        super().__init__(prompt, 'vector_magnitude', solution, difficulty,
                         {'vector': values, 'squared_norm': squared, 'result': exact})


class PDFUnitVector(PDFLinearAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        a, b, length = rng.choice(((3, 4, 5), (5, 12, 13), (8, 15, 17)))
        scale = 1 if difficulty == 'easy' else rng.randint(1, 3)
        values = (rng.choice((-a, a))*scale, rng.choice((-b, b))*scale)
        norm = length*scale
        result = tuple(Fraction(value, norm) for value in values)
        prompt = PDFText('Find a unit vector in the direction of ', PDFVector(values), '.')
        solution = PDFText('Divide by ', PDFMath(str(norm)), ': ', PDFVector(result), '.') if with_solution else None
        super().__init__(prompt, 'unit_vector', solution, difficulty,
                         {'vector': values, 'norm': norm, 'result': result})


class PDFMatrixArithmetic(PDFLinearAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        size = _dimension(difficulty)
        left, right = _matrix(rng, size, size, 7), _matrix(rng, size, size, 7)
        operation = rng.choice(('+', '-')) if difficulty != 'easy' else '+'
        result = _add(left, right if operation == '+' else _scale(-1, right))
        prompt = PDFText('Compute ', PDFMatrix(left), f' {operation} ', PDFMatrix(right), '.')
        solution = PDFText('Operate entry by entry: ', PDFMatrix(result), '.') if with_solution else None
        super().__init__(prompt, 'matrix_arithmetic', solution, difficulty,
                         {'left': left, 'right': right, 'operation': operation, 'result': result})


class PDFScalarMatrixMultiplication(PDFLinearAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        size = _dimension(difficulty)
        scalar, matrix = _nonzero(rng, -5, 5), _matrix(rng, size, size, 6)
        result = _scale(scalar, matrix)
        prompt = PDFText('Compute ', PDFMath(str(scalar)), PDFMatrix(matrix), '.')
        solution = PDFText('Multiply every entry: ', PDFMatrix(result), '.') if with_solution else None
        super().__init__(prompt, 'scalar_matrix', solution, difficulty,
                         {'scalar': scalar, 'matrix': matrix, 'result': result})


class PDFMatrixMultiplicationExercise(PDFLinearAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        shapes = {'easy': (2, 2, 2), 'medium': (2, 3, 2), 'hard': (3, 3, 3)}[difficulty]
        rows, shared, columns = shapes
        left, right = _matrix(rng, rows, shared, 4), _matrix(rng, shared, columns, 4)
        result = _matmul(left, right)
        prompt = PDFText('Compute the product ', PDFMatrix(left), PDFMatrix(right), '.')
        solution = PDFText('Row-by-column multiplication gives ', PDFMatrix(result), '.') if with_solution else None
        super().__init__(prompt, 'matrix_multiplication', solution, difficulty,
                         {'left': left, 'right': right, 'result': result})


class PDFDeterminantExercise(PDFLinearAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        size = 3 if difficulty == 'hard' else 2
        matrix = _matrix(rng, size, size, 6)
        result = _determinant(matrix)
        prompt = PDFText('Find the determinant of ', PDFMatrix(matrix, brackets='determinant'), '.')
        solution = PDFText(PDFMath(rf'\det(A)={result}'), '.') if with_solution else None
        super().__init__(prompt, 'determinant', solution, difficulty,
                         {'matrix': matrix, 'result': result})


class PDFInverseMatrix(PDFLinearAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        if difficulty == 'easy':
            a, d = _nonzero(rng, -5, 5), _nonzero(rng, -5, 5)
            matrix = ((a, 0), (0, d))
        else:
            while True:
                matrix = _matrix(rng, 2, 2, 6)
                if _determinant(matrix):
                    break
        determinant = _determinant(matrix)
        a, b = matrix[0]
        c, d = matrix[1]
        result = ((Fraction(d, determinant), Fraction(-b, determinant)),
                  (Fraction(-c, determinant), Fraction(a, determinant)))
        prompt = PDFText('Find the inverse of ', PDFMatrix(matrix), '.')
        solution = PDFText(PDFMath(r'A^{-1}='), PDFMatrix(result), '.') if with_solution else None
        super().__init__(prompt, 'inverse_matrix', solution, difficulty,
                         {'matrix': matrix, 'determinant': determinant, 'result': result})


def _system_data(rng, dimension, limit):
    solution = _vector(rng, dimension, limit, nonzero=False)
    matrix = []
    for row in range(dimension):
        values = [0]*dimension
        values[row] = _nonzero(rng, -3, 3)
        for column in range(row+1, dimension):
            values[column] = rng.randint(-3, 3)
        matrix.append(tuple(values))
    matrix = tuple(matrix)
    right = _matvec(matrix, solution)
    augmented = tuple(row+(value,) for row, value in zip(matrix, right))
    return matrix, right, solution, augmented


class PDFLinearSystemExercise(PDFLinearAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        dimension = _dimension(difficulty)
        matrix, right, result, augmented = _system_data(rng, dimension, 5)
        prompt = PDFText('Solve the system represented by the augmented matrix ',
                         PDFMatrix(augmented), '.')
        solution = PDFText('The solution vector is ', PDFVector(result), '.') if with_solution else None
        super().__init__(prompt, 'solve_linear_system', solution, difficulty,
                         {'matrix': matrix, 'right': right, 'augmented': augmented, 'result': result})


class PDFRowReduction(PDFLinearAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        dimension = _dimension(difficulty)
        matrix, right, result, augmented = _system_data(rng, dimension, 4)
        reduced, pivots = _rref(augmented)
        prompt = PDFText('Reduce to reduced row-echelon form: ', PDFMatrix(augmented), '.')
        solution = PDFText('The RREF is ', PDFMatrix(reduced), '.') if with_solution else None
        super().__init__(prompt, 'row_reduction', solution, difficulty,
                         {'matrix': augmented, 'rref': reduced, 'pivots': pivots,
                          'system_matrix': matrix, 'right': right, 'result': result})


class PDFMatrixRank(PDFLinearAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        columns = 4 if difficulty == 'hard' else 3
        first = (1, 0, *tuple(rng.randint(-4, 4) for _ in range(columns-2)))
        second = (rng.randint(-3, 3), 1, *tuple(rng.randint(-4, 4) for _ in range(columns-2)))
        scalar = _nonzero(rng, -3, 3)
        third = tuple(a+scalar*b for a, b in zip(first, second))
        matrix = (first, third) if difficulty == 'easy' else (first, second, third)
        reduced, pivots = _rref(matrix)
        rank = len(pivots)
        prompt = PDFText('Find the rank of ', PDFMatrix(matrix), '.')
        solution = PDFText('Its RREF is ', PDFMatrix(reduced), ', so ',
                           PDFMath(rf'\operatorname{{rank}}(A)={rank}'), '.') if with_solution else None
        super().__init__(prompt, 'rank', solution, difficulty,
                         {'matrix': matrix, 'rref': reduced, 'pivots': pivots, 'result': rank})


class PDFLinearIndependence(PDFLinearAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        size = _dimension(difficulty)
        independent = rng.choice((True, False))
        matrix = [list(row) for row in _matrix(rng, size, size, 4)]
        for index in range(size):
            matrix[index][index] += 6
        if not independent:
            for row in range(size):
                matrix[row][-1] = sum(matrix[row][column] for column in range(size-1))
        matrix = tuple(tuple(row) for row in matrix)
        determinant = _determinant(matrix)
        # A rare accidental singular independent construction is reported honestly.
        independent = determinant != 0
        prompt = PDFText('Are the columns of ', PDFMatrix(matrix), ' linearly independent?')
        word = 'independent' if independent else 'dependent'
        solution = PDFText(PDFMath(rf'\det(A)={determinant}'), f', so the columns are {word}.') if with_solution else None
        super().__init__(prompt, 'linear_independence', solution, difficulty,
                         {'matrix': matrix, 'determinant': determinant, 'result': independent})


class PDFBasisCoordinates(PDFLinearAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        size = _dimension(difficulty)
        basis = tuple(tuple((rng.randint(-3, 3) if column > row else
                             _nonzero(rng, -4, 4) if column == row else 0)
                            for column in range(size)) for row in range(size))
        coordinates = _vector(rng, size, 4)
        target = _matvec(basis, coordinates)
        prompt = PDFText('Find the coordinate vector of ', PDFVector(target),
                         ' in the basis formed by the columns of ', PDFMatrix(basis), '.')
        solution = PDFText('The coordinate vector is ', PDFVector(coordinates), '.') if with_solution else None
        super().__init__(prompt, 'basis_coordinates', solution, difficulty,
                         {'basis': basis, 'target': target, 'result': coordinates})


class PDFEigenvaluesExercise(PDFLinearAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        size = _dimension(difficulty)
        eigenvalues = rng.sample([value for value in range(-6, 7) if value], size)
        matrix = tuple(tuple(eigenvalues[row] if row == column else
                             rng.randint(-4, 4) if column > row else 0
                             for column in range(size)) for row in range(size))
        prompt = PDFText('Find the eigenvalues of ', PDFMatrix(matrix), '.')
        answer = r',\quad'.join(rf'\lambda_{{{index}}}={value}'
                               for index, value in enumerate(eigenvalues, start=1))
        solution = PDFText('Because the matrix is triangular, ', PDFMath(answer), '.') if with_solution else None
        super().__init__(prompt, 'eigenvalues', solution, difficulty,
                         {'matrix': matrix, 'result': tuple(eigenvalues)})


class PDFEigenvectorExercise(PDFLinearAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        diagonal, off_diagonal = rng.randint(-5, 5), _nonzero(rng, -5, 5)
        matrix = ((diagonal, off_diagonal), (off_diagonal, diagonal))
        result = rng.choice(((1, 1), (1, -1)))
        eigenvalue = diagonal+off_diagonal if result == (1, 1) else diagonal-off_diagonal
        prompt = PDFText('Find an eigenvector of ', PDFMatrix(matrix), ' for ',
                         PDFMath(rf'\lambda={eigenvalue}'), '.')
        solution = PDFText('One valid eigenvector is ', PDFVector(result), '.') if with_solution else None
        super().__init__(prompt, 'eigenvector', solution, difficulty,
                         {'matrix': matrix, 'eigenvalue': eigenvalue, 'result': result})


class PDFVectorProjection(PDFLinearAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        dimension = _dimension(difficulty)
        vector, onto = _vector(rng, dimension, 6), _vector(rng, dimension, 4)
        numerator, denominator = _dot(vector, onto), _dot(onto, onto)
        factor = Fraction(numerator, denominator)
        result = tuple(factor*value for value in onto)
        prompt = PDFText('Project ', PDFVector(vector), ' onto ', PDFVector(onto), '.')
        solution = PDFText(PDFMath(rf'\operatorname{{proj}}_u(v)=\frac{{v\cdot u}}{{u\cdot u}}u'
                                  rf'=\frac{{{numerator}}}{{{denominator}}}u'), '=', PDFVector(result), '.') if with_solution else None
        super().__init__(prompt, 'projection', solution, difficulty,
                         {'vector': vector, 'onto': onto, 'factor': factor, 'result': result})


class PDFLinearTransformationExercise(PDFLinearAlgebraExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        dimension = _dimension(difficulty)
        matrix, vector = _matrix(rng, dimension, dimension, 4), _vector(rng, dimension, 5)
        result = _matvec(matrix, vector)
        prompt = PDFText('For ', PDFMath('T(x)=Ax'), ' with ', PDFMath('A='), PDFMatrix(matrix),
                         ', find ', PDFMath('T(v)'), ' when ', PDFMath('v='), PDFVector(vector), '.')
        solution = PDFText(PDFMath('T(v)=Av='), PDFVector(result), '.') if with_solution else None
        super().__init__(prompt, 'linear_transformation', solution, difficulty,
                         {'matrix': matrix, 'vector': vector, 'result': result})


_FACTORIES = {
    'vector_arithmetic': PDFVectorArithmetic,
    'dot_product': PDFDotProduct,
    'vector_magnitude': PDFVectorMagnitude,
    'unit_vector': PDFUnitVector,
    'matrix_arithmetic': PDFMatrixArithmetic,
    'scalar_matrix': PDFScalarMatrixMultiplication,
    'matrix_multiplication': PDFMatrixMultiplicationExercise,
    'determinant': PDFDeterminantExercise,
    'inverse_matrix': PDFInverseMatrix,
    'solve_linear_system': PDFLinearSystemExercise,
    'row_reduction': PDFRowReduction,
    'rank': PDFMatrixRank,
    'linear_independence': PDFLinearIndependence,
    'basis_coordinates': PDFBasisCoordinates,
    'eigenvalues': PDFEigenvaluesExercise,
    'eigenvector': PDFEigenvectorExercise,
    'projection': PDFVectorProjection,
    'linear_transformation': PDFLinearTransformationExercise,
}

_ALIASES = {
    'vector_addition': 'vector_arithmetic', 'vector_subtraction': 'vector_arithmetic',
    'dot': 'dot_product', 'magnitude': 'vector_magnitude', 'norm': 'vector_magnitude',
    'normalize': 'unit_vector', 'matrix_addition': 'matrix_arithmetic',
    'scalar_multiplication': 'scalar_matrix', 'matmul': 'matrix_multiplication',
    'det': 'determinant', 'inverse': 'inverse_matrix', 'linear_system': 'solve_linear_system',
    'rref': 'row_reduction', 'matrix_rank': 'rank', 'independence': 'linear_independence',
    'coordinates': 'basis_coordinates', 'eigenvalue': 'eigenvalues',
    'eigenvectors': 'eigenvector', 'vector_projection': 'projection',
    'transformation': 'linear_transformation',
}


def linear_algebra_exercise(kind, *, difficulty='medium', seed=None,
                            with_solution=True, _rng=None):
    """Create a linear-algebra exercise by a canonical or friendly name."""
    if not isinstance(kind, str):
        raise TypeError('kind must be text')
    key = kind.strip().lower().replace('-', '_').replace(' ', '_')
    key = _ALIASES.get(key, key)
    try:
        factory = _FACTORIES[key]
    except KeyError as exc:
        choices = ', '.join(LINEAR_ALGEBRA_EXERCISE_TYPES)
        raise ValueError(f'Unknown linear-algebra exercise {kind!r}; choose from {choices}') from exc
    return factory(with_solution=with_solution, difficulty=difficulty, seed=seed, _rng=_rng)


__all__ = [
    'PDFLinearAlgebraExercise', 'PDFVectorArithmetic', 'PDFDotProduct',
    'PDFVectorMagnitude', 'PDFUnitVector', 'PDFMatrixArithmetic',
    'PDFScalarMatrixMultiplication', 'PDFMatrixMultiplicationExercise',
    'PDFDeterminantExercise', 'PDFInverseMatrix', 'PDFLinearSystemExercise',
    'PDFRowReduction', 'PDFMatrixRank', 'PDFLinearIndependence',
    'PDFBasisCoordinates', 'PDFEigenvaluesExercise', 'PDFEigenvectorExercise',
    'PDFVectorProjection', 'PDFLinearTransformationExercise',
    'LINEAR_ALGEBRA_EXERCISE_TYPES', 'linear_algebra_exercise',
]
