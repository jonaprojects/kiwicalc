from fractions import Fraction

import pytest

import kiwicalc as kw
from kiwicalc.pdf import layout


KINDS = kw.LINEAR_ALGEBRA_EXERCISE_TYPES


def matmul(left, right):
    return tuple(tuple(sum(left[i][k]*right[k][j] for k in range(len(right)))
                       for j in range(len(right[0]))) for i in range(len(left)))


def matvec(matrix, vector):
    return tuple(sum(a*b for a, b in zip(row, vector)) for row in matrix)


def dot(left, right):
    return sum(a*b for a, b in zip(left, right))


def determinant(matrix):
    if len(matrix) == 1:
        return matrix[0][0]
    return sum((-1)**column*matrix[0][column]*determinant(tuple(
        row[:column]+row[column+1:] for row in matrix[1:]
    )) for column in range(len(matrix)))


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('difficulty', ('easy', 'medium', 'hard'))
def test_every_linear_algebra_generator_is_deterministic(kind, difficulty):
    first = kw.linear_algebra_exercise(kind, difficulty=difficulty, seed=123)
    second = kw.linear_algebra_exercise(kind, difficulty=difficulty, seed=123)
    assert isinstance(first, kw.PDFLinearAlgebraExercise)
    assert first.kind == kind and first.difficulty == difficulty
    assert str(first.exercise) == str(second.exercise)
    assert str(first.solution) == str(second.solution)
    assert first.data == second.data
    assert first.solution is not None
    assert kw.linear_algebra_exercise(kind, difficulty=difficulty, seed=123,
                                      with_solution=False).solution is None


def test_vector_answer_invariants_across_many_seeds():
    for seed in range(30):
        arithmetic = kw.PDFVectorArithmetic(seed=seed, difficulty='hard').data
        sign = 1 if arithmetic['operation'] == '+' else -1
        assert arithmetic['result'] == tuple(a+sign*b for a, b in
                                              zip(arithmetic['left'], arithmetic['right']))

        product = kw.PDFDotProduct(seed=seed, difficulty='hard').data
        assert product['products'] == tuple(a*b for a, b in zip(product['left'], product['right']))
        assert product['result'] == sum(product['products'])

        magnitude = kw.PDFVectorMagnitude(seed=seed, difficulty='hard').data
        assert magnitude['squared_norm'] == dot(magnitude['vector'], magnitude['vector'])

        unit = kw.PDFUnitVector(seed=seed, difficulty='hard').data
        assert unit['norm']**2 == dot(unit['vector'], unit['vector'])
        assert unit['result'] == tuple(Fraction(value, unit['norm']) for value in unit['vector'])
        assert dot(unit['result'], unit['result']) == 1

        projection = kw.PDFVectorProjection(seed=seed, difficulty='hard').data
        assert projection['factor'] == Fraction(dot(projection['vector'], projection['onto']),
                                                dot(projection['onto'], projection['onto']))
        assert projection['result'] == tuple(projection['factor']*x for x in projection['onto'])
        residual = tuple(a-b for a, b in zip(projection['vector'], projection['result']))
        assert dot(residual, projection['onto']) == 0


def test_matrix_answer_invariants_across_many_seeds():
    for seed in range(30):
        arithmetic = kw.PDFMatrixArithmetic(seed=seed, difficulty='hard').data
        sign = 1 if arithmetic['operation'] == '+' else -1
        expected = tuple(tuple(a+sign*b for a, b in zip(left, right))
                         for left, right in zip(arithmetic['left'], arithmetic['right']))
        assert arithmetic['result'] == expected

        scalar = kw.PDFScalarMatrixMultiplication(seed=seed, difficulty='hard').data
        assert scalar['result'] == tuple(tuple(scalar['scalar']*value for value in row)
                                         for row in scalar['matrix'])

        product = kw.PDFMatrixMultiplicationExercise(seed=seed, difficulty='hard').data
        assert product['result'] == matmul(product['left'], product['right'])

        det = kw.PDFDeterminantExercise(seed=seed, difficulty='hard').data
        assert det['result'] == determinant(det['matrix'])

        inverse = kw.PDFInverseMatrix(seed=seed, difficulty='hard').data
        assert inverse['determinant'] == determinant(inverse['matrix']) != 0
        assert matmul(inverse['matrix'], inverse['result']) == ((1, 0), (0, 1))
        assert matmul(inverse['result'], inverse['matrix']) == ((1, 0), (0, 1))

        transform = kw.PDFLinearTransformationExercise(seed=seed, difficulty='hard').data
        assert transform['result'] == matvec(transform['matrix'], transform['vector'])


def test_system_space_and_eigen_answer_invariants_across_many_seeds():
    for seed in range(30):
        system = kw.PDFLinearSystemExercise(seed=seed, difficulty='hard').data
        assert matvec(system['matrix'], system['result']) == system['right']

        reduced = kw.PDFRowReduction(seed=seed, difficulty='hard').data
        size = len(reduced['result'])
        expected = tuple(tuple(Fraction(int(row == column)) for column in range(size))
                         +(Fraction(reduced['result'][row]),) for row in range(size))
        assert reduced['rref'] == expected
        assert reduced['pivots'] == tuple(range(size))

        rank = kw.PDFMatrixRank(seed=seed, difficulty='hard').data
        assert rank['result'] == len(rank['pivots'])
        assert rank['result'] == kw.Matrix(rank['matrix']).rank()

        independent = kw.PDFLinearIndependence(seed=seed, difficulty='hard').data
        assert independent['determinant'] == determinant(independent['matrix'])
        assert independent['result'] is (independent['determinant'] != 0)

        basis = kw.PDFBasisCoordinates(seed=seed, difficulty='hard').data
        assert matvec(basis['basis'], basis['result']) == basis['target']
        assert determinant(basis['basis']) != 0

        eigenvalues = kw.PDFEigenvaluesExercise(seed=seed, difficulty='hard').data
        diagonal = tuple(eigenvalues['matrix'][i][i] for i in range(len(eigenvalues['matrix'])))
        assert eigenvalues['result'] == diagonal

        eigenvector = kw.PDFEigenvectorExercise(seed=seed, difficulty='hard').data
        assert matvec(eigenvector['matrix'], eigenvector['result']) == tuple(
            eigenvector['eigenvalue']*value for value in eigenvector['result'])


def test_factory_aliases_and_validation():
    aliases = {
        'vector addition': 'vector_arithmetic', 'dot': 'dot_product',
        'norm': 'vector_magnitude', 'normalize': 'unit_vector',
        'matrix addition': 'matrix_arithmetic', 'scalar multiplication': 'scalar_matrix',
        'matmul': 'matrix_multiplication', 'det': 'determinant',
        'inverse': 'inverse_matrix', 'linear system': 'solve_linear_system',
        'rref': 'row_reduction', 'matrix rank': 'rank',
        'independence': 'linear_independence', 'coordinates': 'basis_coordinates',
        'eigenvalue': 'eigenvalues', 'eigenvectors': 'eigenvector',
        'vector projection': 'projection', 'transformation': 'linear_transformation',
    }
    for alias, expected in aliases.items():
        assert kw.linear_algebra_exercise(alias, seed=4).kind == expected
    with pytest.raises(ValueError, match='Unknown linear-algebra exercise'):
        kw.linear_algebra_exercise('telepathy')
    with pytest.raises(TypeError, match='kind'):
        kw.linear_algebra_exercise(1)
    with pytest.raises(ValueError, match='difficulty'):
        kw.linear_algebra_exercise('determinant', difficulty='expert')
    with pytest.raises(TypeError, match='seed'):
        kw.linear_algebra_exercise('determinant', seed=True)
    with pytest.raises(ValueError, match='cannot both'):
        kw.linear_algebra_exercise('determinant', seed=1,
                                   _rng=__import__('random').Random(1))


@pytest.mark.parametrize('kind', KINDS)
def test_every_family_integrates_with_batch_worksheet(kind, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(layout, 'render_pages', lambda *args, **kwargs: calls.append((args, kwargs)))
    kw.worksheet(tmp_path/f'{kind}.pdf', dtype=kind, equations_per_page=3,
                 get_solutions=True, difficulty='hard', seed=42, theme='assessment')
    _, titles, pages = calls[0][0]
    assert len(titles) == len(pages) == 2
    assert titles[1].endswith('Solutions')
    assert len(pages[0]) == len(pages[1]) == 3
    assert calls[0][1]['theme'] == 'assessment'


def test_batch_without_solutions_and_title_validation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(layout, 'render_pages', lambda *args, **kwargs: calls.append((args, kwargs)))
    kw.worksheet(tmp_path/'vectors.pdf', dtype='dot_product', equations_per_page=2,
                 get_solutions=False, seed=7)
    assert len(calls[0][0][1]) == 1
    with pytest.raises(ValueError, match='one title'):
        kw.worksheet(tmp_path/'bad.pdf', dtype='determinant', num_of_pages=2,
                     titles=['Only one'], seed=7)


def test_all_linear_algebra_math_renders_in_one_document(tmp_path):
    sheet = kw.PDFWorksheet('Linear Algebra', theme='academic')
    for index, kind in enumerate(KINDS):
        sheet.add_exercise(kw.linear_algebra_exercise(kind, difficulty='hard', seed=index))
    sheet.end_page().create(tmp_path/'linear-algebra.pdf')
    assert (tmp_path/'linear-algebra.pdf').read_bytes().startswith(b'%PDF')
