from fractions import Fraction
import math

import pytest

import kiwicalc as kw
from kiwicalc.pdf import layout


KINDS = kw.GEOMETRY_EXERCISE_TYPES


def dot(left, right):
    return sum(a*b for a, b in zip(left, right))


def cross(left, right):
    return (left[1]*right[2]-left[2]*right[1],
            left[2]*right[0]-left[0]*right[2],
            left[0]*right[1]-left[1]*right[0])


def parallel(left, right):
    products = tuple(left[i]*right[j]-left[j]*right[i]
                     for i in range(len(left)) for j in range(i+1, len(left)))
    return all(value == 0 for value in products)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('difficulty', ('easy', 'medium', 'hard'))
def test_every_geometry_generator_is_deterministic(kind, difficulty):
    first = kw.geometry_exercise(kind, difficulty=difficulty, seed=123)
    second = kw.geometry_exercise(kind, difficulty=difficulty, seed=123)
    assert isinstance(first, kw.PDFGeometryExercise)
    assert first.kind == kind and first.difficulty == difficulty
    assert str(first.exercise) == str(second.exercise)
    assert str(first.solution) == str(second.solution)
    assert first.data == second.data
    assert first.solution is not None
    assert kw.geometry_exercise(kind, difficulty=difficulty, seed=123,
                                with_solution=False).solution is None


def test_coordinate_geometry_answer_invariants_across_many_seeds():
    for seed in range(30):
        distance = kw.PDFDistanceBetweenPoints(seed=seed, difficulty='hard').data
        delta = tuple(b-a for a, b in zip(distance['first'], distance['second']))
        assert distance['delta'] == delta
        assert distance['squared_distance'] == sum(value*value for value in delta)
        assert distance['result']**2 == distance['squared_distance']

        midpoint = kw.PDFMidpoint(seed=seed, difficulty='hard').data
        assert midpoint['result'] == tuple(Fraction(a+b, 2) for a, b in
                                            zip(midpoint['first'], midpoint['second']))

        slope = kw.PDFSlope(seed=seed, difficulty='hard').data
        assert slope['run'] != 0
        assert slope['result'] == Fraction(slope['rise'], slope['run'])

        line = kw.PDFLineEquation(seed=seed, difficulty='hard').data
        a, b, c = line['coefficients']
        for x, y in (line['first'], line['second']):
            assert a*x+b*y+c == 0

        point_line = kw.PDFPointLineDistance(seed=seed, difficulty='hard').data
        a, b, c = point_line['line']
        x, y = point_line['point']
        assert point_line['numerator'] == abs(a*x+b*y+c)
        assert point_line['normal_length']**2 == a*a+b*b
        assert point_line['result'] == Fraction(point_line['numerator'],
                                                point_line['normal_length'])

        related = kw.PDFParallelPerpendicularLines(seed=seed, difficulty='hard').data
        expected = related['slope'] if related['relation'] == 'parallel' else -1/related['slope']
        assert related['result'] == expected

        area = kw.PDFTriangleArea(seed=seed, difficulty='hard').data
        (x1, y1), (x2, y2), (x3, y3) = area['vertices']
        shoelace = Fraction(abs(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)), 2)
        assert area['result'] == shoelace

        centroid = kw.PDFTriangleCentroid(seed=seed, difficulty='hard').data
        assert centroid['result'] == tuple(Fraction(sum(point[i] for point in centroid['vertices']), 3)
                                            for i in range(2))

        pythagorean = kw.PDFPythagoreanTheorem(seed=seed, difficulty='hard').data
        a, b, c = pythagorean['triple']
        assert a*a+b*b == c*c
        assert pythagorean['result'] in pythagorean['triple']

        circle = kw.PDFCircleEquation(seed=seed, difficulty='hard').data
        assert circle['radius'] > 0
        assert circle['result'].endswith(f'={circle["radius"]**2}')


def test_measurement_and_transformation_invariants_across_many_seeds():
    for seed in range(30):
        arc = kw.PDFArcAndSector(seed=seed, difficulty='hard').data
        assert arc['arc_pi_coefficient'] == Fraction(arc['angle_degrees']*arc['radius'], 180)
        assert arc['sector_pi_coefficient'] == Fraction(
            arc['angle_degrees']*arc['radius']**2, 360)
        assert arc['result'] == (arc['arc_pi_coefficient'], arc['sector_pi_coefficient'])

        polygon = kw.PDFPolygonAngles(seed=seed, difficulty='hard').data
        assert polygon['interior_sum'] == 180*(polygon['sides']-2)
        assert polygon['interior_angle'] == Fraction(polygon['interior_sum'], polygon['sides'])
        assert polygon['exterior_angle'] == Fraction(360, polygon['sides'])
        assert polygon['interior_angle']+polygon['exterior_angle'] == 180

        easy_solid = kw.PDFSolidMeasurement(seed=seed, difficulty='easy').data
        length, width, height = easy_solid['dimensions']
        assert easy_solid['volume'] == length*width*height
        assert easy_solid['surface_area'] == 2*(length*width+length*height+width*height)

        cylinder = kw.PDFSolidMeasurement(seed=seed, difficulty='medium').data
        assert cylinder['volume_pi_coefficient'] == cylinder['radius']**2*cylinder['height']

        cone = kw.PDFSolidMeasurement(seed=seed, difficulty='hard').data
        assert cone['volume_pi_coefficient'] == Fraction(cone['radius']**2*cone['height'], 3)

        transformed = kw.PDFCoordinateTransformation(seed=seed, difficulty='hard').data
        x, y = transformed['point']
        mappings = {
            'reflect_x': (x, -y), 'reflect_y': (-x, y),
            'reflect_origin': (-x, -y), 'rotate_90': (-y, x),
            'rotate_180': (-x, -y),
        }
        assert transformed['result'] == mappings[transformed['transformation']]


def test_vector_geometry_answer_invariants_across_many_seeds():
    for seed in range(30):
        displacement = kw.PDFVectorFromPoints(seed=seed, difficulty='hard').data
        assert displacement['result'] == tuple(b-a for a, b in
                                               zip(displacement['start'], displacement['end']))

        relationship = kw.PDFVectorRelationship(seed=seed, difficulty='hard').data
        if relationship['result'] == 'parallel':
            assert parallel(relationship['first'], relationship['second'])
        elif relationship['result'] == 'perpendicular':
            assert relationship['dot_product'] == 0
        else:
            assert not parallel(relationship['first'], relationship['second'])
            assert relationship['dot_product'] != 0

        angle = kw.PDFVectorAngle(seed=seed, difficulty='hard').data
        cosine = dot(angle['first'], angle['second'])/math.sqrt(
            dot(angle['first'], angle['first'])*dot(angle['second'], angle['second']))
        assert math.degrees(math.acos(round(cosine, 14))) == pytest.approx(angle['result'])

        product = kw.PDFCrossProduct(seed=seed, difficulty='hard').data
        assert product['result'] == cross(product['first'], product['second'])
        assert dot(product['result'], product['first']) == 0
        assert dot(product['result'], product['second']) == 0

        line = kw.PDFVectorLine(seed=seed, difficulty='hard').data
        point_at_two = tuple(point+2*direction for point, direction in
                             zip(line['point'], line['direction']))
        assert tuple(value-point for value, point in zip(point_at_two, line['point'])) == tuple(
            2*value for value in line['direction'])

        plane = kw.PDFPlaneEquation(seed=seed, difficulty='hard').data
        assert dot(plane['normal'], plane['point'])+plane['coefficients'][-1] == 0


def test_factory_aliases_and_validation():
    aliases = {
        'distance between points': 'distance', 'section midpoint': 'midpoint',
        'gradient': 'slope', 'line': 'line_equation',
        'distance to line': 'point_line_distance', 'parallel lines': 'parallel_perpendicular',
        'area of triangle': 'triangle_area', 'centroid': 'triangle_centroid',
        'right triangle': 'pythagorean', 'circle': 'circle_equation',
        'sector': 'arc_sector', 'regular polygon': 'polygon_angles', 'solid': 'solid_measurement',
        'transform point': 'coordinate_transformation', 'displacement vector': 'vector_from_points',
        'parallel vectors': 'vector_relationship', 'angle between vectors': 'vector_angle',
        'cross': 'cross_product', 'vector equation': 'vector_line', 'plane': 'plane_equation',
    }
    for alias, expected in aliases.items():
        assert kw.geometry_exercise(alias, seed=4).kind == expected
    with pytest.raises(ValueError, match='Unknown geometry exercise'):
        kw.geometry_exercise('telepathy')
    with pytest.raises(TypeError, match='kind'):
        kw.geometry_exercise(1)
    with pytest.raises(ValueError, match='difficulty'):
        kw.geometry_exercise('distance', difficulty='expert')
    with pytest.raises(TypeError, match='seed'):
        kw.geometry_exercise('distance', seed=True)
    with pytest.raises(ValueError, match='cannot both'):
        kw.geometry_exercise('distance', seed=1, _rng=__import__('random').Random(1))


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
    kw.worksheet(tmp_path/'circles.pdf', dtype='circle_equation', equations_per_page=2,
                 get_solutions=False, seed=7)
    assert len(calls[0][0][1]) == 1
    with pytest.raises(ValueError, match='one title'):
        kw.worksheet(tmp_path/'bad.pdf', dtype='distance', num_of_pages=2,
                     titles=['Only one'], seed=7)


def test_all_geometry_math_renders_in_one_document(tmp_path):
    sheet = kw.PDFWorksheet('Geometry and Vectors', theme='academic')
    for index, kind in enumerate(KINDS):
        sheet.add_exercise(kw.geometry_exercise(kind, difficulty='hard', seed=index))
    sheet.end_page().create(tmp_path/'geometry.pdf')
    assert (tmp_path/'geometry.pdf').read_bytes().startswith(b'%PDF')
