"""Deterministic coordinate-geometry and vector-geometry PDF exercises."""
from __future__ import annotations

from fractions import Fraction
from math import gcd
import random

from .arrays import PDFVector
from .formatting import PDFText, format_math
from .layout import PDFMath
from .worksheet import PDFExercise


DIFFICULTIES = ('easy', 'medium', 'hard')
GEOMETRY_EXERCISE_TYPES = (
    'distance', 'midpoint', 'slope', 'line_equation', 'point_line_distance',
    'parallel_perpendicular', 'triangle_area', 'triangle_centroid',
    'pythagorean', 'circle_equation', 'arc_sector', 'polygon_angles',
    'solid_measurement', 'coordinate_transformation', 'vector_from_points',
    'vector_relationship', 'vector_angle', 'cross_product', 'vector_line',
    'plane_equation',
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


def _random_point(rng, dimension, limit=8):
    return tuple(rng.randint(-limit, limit) for _ in range(dimension))


def _point(values):
    return PDFMath(r'\left('+','.join(format_math(value) for value in values)+r'\right)')


def _dot(left, right):
    return sum(a*b for a, b in zip(left, right))


def _cross(left, right):
    return (left[1]*right[2]-left[2]*right[1],
            left[2]*right[0]-left[0]*right[2],
            left[0]*right[1]-left[1]*right[0])


def _root(number):
    candidate = int(number**.5)
    return str(candidate) if candidate*candidate == number else rf'\sqrt{{{number}}}'


def _fraction(value):
    return format_math(value if isinstance(value, Fraction) else Fraction(value))


def _factor(variable, shift):
    return variable if shift == 0 else f'({variable}{"-" if shift > 0 else "+"}{abs(shift)})'


def _linear_equation(coefficients, variables=('x', 'y', 'z')):
    pieces = []
    for coefficient, variable in zip(coefficients[:-1], variables):
        if not coefficient:
            continue
        magnitude = abs(coefficient)
        body = ('' if magnitude == 1 else str(magnitude))+variable
        pieces.append(('-' if coefficient < 0 else '+' if pieces else '')+body)
    constant = coefficients[-1]
    if constant:
        pieces.append(('-' if constant < 0 else '+' if pieces else '')+str(abs(constant)))
    return ''.join(pieces or ['0'])+'=0'


def _normalize_line(a, b, c):
    divisor = gcd(gcd(abs(a), abs(b)), abs(c)) or 1
    a, b, c = a//divisor, b//divisor, c//divisor
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return a, b, c


def _pi_multiple(coefficient):
    if coefficient == 0:
        return '0'
    if coefficient == 1:
        return r'\pi'
    if coefficient == -1:
        return r'-\pi'
    return _fraction(coefficient)+r'\pi'


class PDFGeometryExercise(PDFExercise):
    """Base class exposing exact source data for answer checking."""
    __slots__ = ('kind', 'difficulty', 'data')

    def __init__(self, prompt, kind, solution, difficulty, data):
        category = 'vector geometry' if kind in {
            'vector_from_points', 'vector_relationship', 'vector_angle',
            'cross_product', 'vector_line', 'plane_equation',
        } else 'geometry'
        super().__init__(prompt, category, kind, solution=solution)
        self.kind = kind
        self.difficulty = difficulty
        self.data = data


class PDFDistanceBetweenPoints(PDFGeometryExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        dimension = _dimension(difficulty)
        triples = ((3, 4, 5), (5, 12, 13), (8, 15, 17)) if dimension == 2 else (
            (1, 2, 2, 3), (2, 3, 6, 7), (3, 4, 12, 13))
        values = rng.choice(triples)
        delta, distance = values[:-1], values[-1]
        delta = tuple(rng.choice((-value, value)) for value in delta)
        first = _random_point(rng, dimension, 6)
        second = tuple(value+change for value, change in zip(first, delta))
        squared = sum(change*change for change in delta)
        prompt = PDFText('Find the distance between ', _point(first), ' and ', _point(second), '.')
        solution = PDFText(PDFMath(rf'd=\sqrt{{{squared}}}={distance}'), '.') if with_solution else None
        super().__init__(prompt, 'distance', solution, difficulty,
                         {'first': first, 'second': second, 'delta': delta,
                          'squared_distance': squared, 'result': distance})


class PDFMidpoint(PDFGeometryExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        dimension = _dimension(difficulty)
        first, second = _random_point(rng, dimension), _random_point(rng, dimension)
        result = tuple(Fraction(a+b, 2) for a, b in zip(first, second))
        prompt = PDFText('Find the midpoint of the segment joining ', _point(first), ' and ', _point(second), '.')
        solution = PDFText('The midpoint is ', _point(result), '.') if with_solution else None
        super().__init__(prompt, 'midpoint', solution, difficulty,
                         {'first': first, 'second': second, 'result': result})


class PDFSlope(PDFGeometryExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        first = _random_point(rng, 2)
        dx, dy = _nonzero(rng, -7, 7), rng.randint(-9, 9)
        second = (first[0]+dx, first[1]+dy)
        result = Fraction(dy, dx)
        prompt = PDFText('Find the slope of the line through ', _point(first), ' and ', _point(second), '.')
        solution = PDFText(PDFMath(rf'm=\frac{{{dy}}}{{{dx}}}={_fraction(result)}'), '.') if with_solution else None
        super().__init__(prompt, 'slope', solution, difficulty,
                         {'first': first, 'second': second, 'rise': dy, 'run': dx, 'result': result})


class PDFLineEquation(PDFGeometryExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        first = _random_point(rng, 2, 6)
        delta = (_nonzero(rng, -6, 6), _nonzero(rng, -6, 6))
        second = tuple(value+change for value, change in zip(first, delta))
        a, b, c = _normalize_line(first[1]-second[1], second[0]-first[0],
                                  first[0]*second[1]-second[0]*first[1])
        equation = _linear_equation((a, b, c), ('x', 'y'))
        prompt = PDFText('Find an equation of the line through ', _point(first), ' and ', _point(second), '.')
        solution = PDFText('In standard form, ', PDFMath(equation), '.') if with_solution else None
        super().__init__(prompt, 'line_equation', solution, difficulty,
                         {'first': first, 'second': second, 'coefficients': (a, b, c),
                          'result': equation})


class PDFPointLineDistance(PDFGeometryExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        a, b, norm = rng.choice(((3, 4, 5), (5, 12, 13), (8, 15, 17)))
        a, b = rng.choice((-a, a)), rng.choice((-b, b))
        point = _random_point(rng, 2, 5)
        offset = rng.randint(1, 5)*norm
        c = offset-a*point[0]-b*point[1]
        numerator = abs(a*point[0]+b*point[1]+c)
        result = Fraction(numerator, norm)
        equation = _linear_equation((a, b, c), ('x', 'y'))
        prompt = PDFText('Find the distance from ', _point(point), ' to the line ', PDFMath(equation), '.')
        solution = PDFText(PDFMath(rf'd=\frac{{{numerator}}}{{\sqrt{{{a*a+b*b}}}}}={_fraction(result)}'), '.') if with_solution else None
        super().__init__(prompt, 'point_line_distance', solution, difficulty,
                         {'point': point, 'line': (a, b, c), 'numerator': numerator,
                          'normal_length': norm, 'result': result})


class PDFParallelPerpendicularLines(PDFGeometryExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        slope = Fraction(_nonzero(rng, -6, 6), _nonzero(rng, 1, 6))
        relation = rng.choice(('parallel', 'perpendicular')) if difficulty != 'easy' else 'parallel'
        result = slope if relation == 'parallel' else -1/slope
        prompt = PDFText('A line has slope ', PDFMath(_fraction(slope)), '. Find the slope of a ',
                         relation, ' line.')
        explanation = 'Parallel slopes are equal' if relation == 'parallel' else 'Perpendicular slopes are negative reciprocals'
        solution = PDFText(explanation, ': ', PDFMath(_fraction(result)), '.') if with_solution else None
        super().__init__(prompt, 'parallel_perpendicular', solution, difficulty,
                         {'slope': slope, 'relation': relation, 'result': result})


class PDFTriangleArea(PDFGeometryExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        first = _random_point(rng, 2, 5)
        base, height = 2*rng.randint(1, 6), _nonzero(rng, -8, 8)
        second = (first[0]+base, first[1])
        third = (rng.randint(first[0]-3, second[0]+3), first[1]+height)
        result = Fraction(abs(base*height), 2)
        prompt = PDFText('Find the area of the triangle with vertices ', _point(first), ', ',
                         _point(second), ', and ', _point(third), '.')
        solution = PDFText(PDFMath(rf'A=\frac{{1}}{{2}}\left|{base}\cdot {height}\right|={_fraction(result)}'), '.') if with_solution else None
        super().__init__(prompt, 'triangle_area', solution, difficulty,
                         {'vertices': (first, second, third), 'base': base,
                          'signed_height': height, 'result': result})


class PDFTriangleCentroid(PDFGeometryExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        vertices = tuple(_random_point(rng, 2, 8) for _ in range(3))
        result = tuple(Fraction(sum(point[index] for point in vertices), 3) for index in range(2))
        prompt = PDFText('Find the centroid of the triangle with vertices ', _point(vertices[0]), ', ',
                         _point(vertices[1]), ', and ', _point(vertices[2]), '.')
        solution = PDFText('Average the coordinates: ', _point(result), '.') if with_solution else None
        super().__init__(prompt, 'triangle_centroid', solution, difficulty,
                         {'vertices': vertices, 'result': result})


class PDFPythagoreanTheorem(PDFGeometryExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        a, b, c = rng.choice(((3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)))
        missing = rng.choice(('leg', 'hypotenuse')) if difficulty != 'easy' else 'hypotenuse'
        if missing == 'hypotenuse':
            known, result, equation = (a, b), c, rf'c=\sqrt{{{a}^2+{b}^2}}={c}'
        else:
            known, result, equation = (c, a), b, rf'b=\sqrt{{{c}^2-{a}^2}}={b}'
        prompt = PDFText(f'A right triangle has known side lengths {known[0]} and {known[1]}. '
                         f'Find the missing {missing}.')
        solution = PDFText(PDFMath(equation), '.') if with_solution else None
        super().__init__(prompt, 'pythagorean', solution, difficulty,
                         {'triple': (a, b, c), 'missing': missing, 'known': known, 'result': result})


class PDFCircleEquation(PDFGeometryExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        center, radius = _random_point(rng, 2, 6), rng.randint(1, 8)
        equation = rf'{_factor("x", center[0])}^2+{_factor("y", center[1])}^2={radius*radius}'
        prompt = PDFText('Write the equation of the circle with center ', _point(center),
                         ' and radius ', PDFMath(str(radius)), '.')
        solution = PDFText(PDFMath(equation), '.') if with_solution else None
        super().__init__(prompt, 'circle_equation', solution, difficulty,
                         {'center': center, 'radius': radius, 'result': equation})


class PDFArcAndSector(PDFGeometryExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        radius = rng.randint(2, 12)
        angle = rng.choice((30, 45, 60, 90, 120, 180))
        arc = Fraction(angle*radius, 180)
        sector = Fraction(angle*radius*radius, 360)
        prompt = PDFText('A circle has radius ', PDFMath(str(radius)), ' and central angle ',
                         PDFMath(rf'{angle}^\circ'), '. Find the arc length and sector area.')
        solution = PDFText('Arc length ', PDFMath(_pi_multiple(arc)), '; sector area ',
                           PDFMath(_pi_multiple(sector)), '.') if with_solution else None
        super().__init__(prompt, 'arc_sector', solution, difficulty,
                         {'radius': radius, 'angle_degrees': angle,
                          'arc_pi_coefficient': arc, 'sector_pi_coefficient': sector,
                          'result': (arc, sector)})


class PDFPolygonAngles(PDFGeometryExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        sides = rng.randint(5 if difficulty == 'hard' else 3, 14 if difficulty == 'hard' else 10)
        interior_sum = 180*(sides-2)
        interior = Fraction(interior_sum, sides)
        exterior = Fraction(360, sides)
        prompt = PDFText(f'For a regular {sides}-gon, find the sum of its interior angles '
                         'and each interior and exterior angle.')
        solution = PDFText('Sum: ', PDFMath(rf'{interior_sum}^\circ'), '; each interior angle: ',
                           PDFMath(rf'{_fraction(interior)}^\circ'), '; each exterior angle: ',
                           PDFMath(rf'{_fraction(exterior)}^\circ'), '.') if with_solution else None
        super().__init__(prompt, 'polygon_angles', solution, difficulty,
                         {'sides': sides, 'interior_sum': interior_sum,
                          'interior_angle': interior, 'exterior_angle': exterior,
                          'result': (interior_sum, interior, exterior)})


class PDFSolidMeasurement(PDFGeometryExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        shape = {'easy': 'rectangular prism', 'medium': 'cylinder', 'hard': 'cone'}[difficulty]
        if shape == 'rectangular prism':
            dimensions = tuple(rng.randint(2, 10) for _ in range(3))
            volume = dimensions[0]*dimensions[1]*dimensions[2]
            surface = 2*(dimensions[0]*dimensions[1]+dimensions[0]*dimensions[2]+dimensions[1]*dimensions[2])
            prompt = PDFText(f'A rectangular prism has dimensions {dimensions[0]}, {dimensions[1]}, and {dimensions[2]}. '
                             'Find its volume and surface area.')
            solution = PDFText('Volume ', PDFMath(str(volume)), '; surface area ', PDFMath(str(surface)), '.') if with_solution else None
            data = {'shape': shape, 'dimensions': dimensions, 'volume': volume,
                    'surface_area': surface, 'result': (volume, surface)}
        else:
            radius, height = rng.randint(2, 9), rng.randint(3, 12)
            volume_coefficient = Fraction(radius*radius*height, 1 if shape == 'cylinder' else 3)
            prompt = PDFText(f'A {shape} has radius {radius} and height {height}. Find its exact volume.')
            solution = PDFText('Volume ', PDFMath(_pi_multiple(volume_coefficient)), '.') if with_solution else None
            data = {'shape': shape, 'radius': radius, 'height': height,
                    'volume_pi_coefficient': volume_coefficient,
                    'result': volume_coefficient}
        super().__init__(prompt, 'solid_measurement', solution, difficulty, data)


class PDFCoordinateTransformation(PDFGeometryExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        point = _random_point(rng, 2, 8)
        choices = ('reflect_x', 'reflect_y') if difficulty == 'easy' else (
            'reflect_x', 'reflect_y', 'reflect_origin', 'rotate_90', 'rotate_180')
        transformation = rng.choice(choices)
        x, y = point
        mapping = {
            'reflect_x': ((x, -y), 'reflection across the x-axis'),
            'reflect_y': ((-x, y), 'reflection across the y-axis'),
            'reflect_origin': ((-x, -y), 'reflection through the origin'),
            'rotate_90': ((-y, x), 'a 90-degree counterclockwise rotation'),
            'rotate_180': ((-x, -y), 'a 180-degree rotation'),
        }
        result, description = mapping[transformation]
        prompt = PDFText('Find the image of ', _point(point), ' after ', description, '.')
        solution = PDFText('The image is ', _point(result), '.') if with_solution else None
        super().__init__(prompt, 'coordinate_transformation', solution, difficulty,
                         {'point': point, 'transformation': transformation, 'result': result})


class PDFVectorFromPoints(PDFGeometryExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        dimension = _dimension(difficulty)
        start, end = _random_point(rng, dimension), _random_point(rng, dimension)
        result = tuple(b-a for a, b in zip(start, end))
        prompt = PDFText('Find the vector from ', _point(start), ' to ', _point(end), '.')
        solution = PDFText('Subtract start from end: ', PDFVector(result), '.') if with_solution else None
        super().__init__(prompt, 'vector_from_points', solution, difficulty,
                         {'start': start, 'end': end, 'result': result})


class PDFVectorRelationship(PDFGeometryExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        dimension = _dimension(difficulty)
        first = tuple(_nonzero(rng, -4, 4) for _ in range(dimension))
        relation = rng.choice(('parallel', 'perpendicular', 'neither'))
        if relation == 'parallel':
            scalar = _nonzero(rng, -4, 4)
            second = tuple(scalar*value for value in first)
        elif relation == 'perpendicular':
            if dimension == 2:
                second = (-first[1], first[0])
            else:
                second = (-first[1], first[0], 0)
        else:
            second = tuple(value+index+1 for index, value in enumerate(first))
            while (_dot(first, second) == 0
                   or len({Fraction(b, a) for a, b in zip(first, second)}) == 1):
                second = (second[0]+1, *second[1:])
        dot_product = _dot(first, second)
        prompt = PDFText('Classify the vectors as parallel, perpendicular, or neither: ',
                         PDFVector(first), ' and ', PDFVector(second), '.')
        solution = PDFText(f'They are {relation}. ', PDFMath(rf'u\cdot v={dot_product}'), '.') if with_solution else None
        super().__init__(prompt, 'vector_relationship', solution, difficulty,
                         {'first': first, 'second': second, 'dot_product': dot_product,
                          'result': relation})


class PDFVectorAngle(PDFGeometryExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        first, second, angle = rng.choice((
            ((1, 0), (0, 1), 90), ((1, 0), (1, 1), 45),
            ((1, 0), (-1, 1), 135), ((1, 1), (1, -1), 90)))
        scale_left, scale_right = rng.randint(1, 4), rng.randint(1, 4)
        first = tuple(scale_left*value for value in first)
        second = tuple(scale_right*value for value in second)
        dot_product = _dot(first, second)
        prompt = PDFText('Find the angle between ', PDFVector(first), ' and ', PDFVector(second), '.')
        solution = PDFText(PDFMath(rf'\cos\theta=\frac{{u\cdot v}}{{\Vert u\Vert\Vert v\Vert}}'),
                           ', so ', PDFMath(rf'\theta={angle}^\circ'), '.') if with_solution else None
        super().__init__(prompt, 'vector_angle', solution, difficulty,
                         {'first': first, 'second': second, 'dot_product': dot_product,
                          'result_degrees': angle, 'result': angle})


class PDFCrossProduct(PDFGeometryExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        first = tuple(_nonzero(rng, -5, 5) for _ in range(3))
        second = tuple(_nonzero(rng, -5, 5) for _ in range(3))
        result = _cross(first, second)
        prompt = PDFText('Compute ', PDFVector(first), PDFMath(r'\times'), PDFVector(second), '.')
        solution = PDFText('The cross product is ', PDFVector(result), '.') if with_solution else None
        super().__init__(prompt, 'cross_product', solution, difficulty,
                         {'first': first, 'second': second, 'result': result})


class PDFVectorLine(PDFGeometryExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        dimension = _dimension(difficulty)
        point = _random_point(rng, dimension, 6)
        direction = tuple(_nonzero(rng, -5, 5) for _ in range(dimension))
        prompt = PDFText('Write a vector equation for the line through ', _point(point),
                         ' with direction ', PDFVector(direction), '.')
        solution = PDFText(PDFMath(r'r(t)='), PDFVector(point), PDFMath('+t'), PDFVector(direction), '.') if with_solution else None
        super().__init__(prompt, 'vector_line', solution, difficulty,
                         {'point': point, 'direction': direction,
                          'result': (point, direction)})


class PDFPlaneEquation(PDFGeometryExercise):
    def __init__(self, with_solution=True, difficulty='medium', *, seed=None, _rng=None):
        rng = _settings(difficulty, seed, _rng)
        point = _random_point(rng, 3, 5)
        normal = tuple(_nonzero(rng, -5, 5) for _ in range(3))
        constant = -_dot(normal, point)
        coefficients = (*normal, constant)
        equation = _linear_equation(coefficients)
        prompt = PDFText('Find the plane through ', _point(point), ' with normal vector ',
                         PDFVector(normal), '.')
        solution = PDFText('An equation is ', PDFMath(equation), '.') if with_solution else None
        super().__init__(prompt, 'plane_equation', solution, difficulty,
                         {'point': point, 'normal': normal, 'coefficients': coefficients,
                          'result': equation})


_FACTORIES = {
    'distance': PDFDistanceBetweenPoints,
    'midpoint': PDFMidpoint,
    'slope': PDFSlope,
    'line_equation': PDFLineEquation,
    'point_line_distance': PDFPointLineDistance,
    'parallel_perpendicular': PDFParallelPerpendicularLines,
    'triangle_area': PDFTriangleArea,
    'triangle_centroid': PDFTriangleCentroid,
    'pythagorean': PDFPythagoreanTheorem,
    'circle_equation': PDFCircleEquation,
    'arc_sector': PDFArcAndSector,
    'polygon_angles': PDFPolygonAngles,
    'solid_measurement': PDFSolidMeasurement,
    'coordinate_transformation': PDFCoordinateTransformation,
    'vector_from_points': PDFVectorFromPoints,
    'vector_relationship': PDFVectorRelationship,
    'vector_angle': PDFVectorAngle,
    'cross_product': PDFCrossProduct,
    'vector_line': PDFVectorLine,
    'plane_equation': PDFPlaneEquation,
}

_ALIASES = {
    'distance_between_points': 'distance', 'section_midpoint': 'midpoint',
    'gradient': 'slope', 'line': 'line_equation', 'distance_to_line': 'point_line_distance',
    'parallel_lines': 'parallel_perpendicular', 'perpendicular_lines': 'parallel_perpendicular',
    'area_of_triangle': 'triangle_area', 'centroid': 'triangle_centroid',
    'right_triangle': 'pythagorean', 'circle': 'circle_equation',
    'sector': 'arc_sector', 'regular_polygon': 'polygon_angles', 'solid': 'solid_measurement',
    'transform_point': 'coordinate_transformation', 'displacement_vector': 'vector_from_points',
    'parallel_vectors': 'vector_relationship', 'angle_between_vectors': 'vector_angle',
    'cross': 'cross_product', 'vector_equation': 'vector_line', 'plane': 'plane_equation',
}


def geometry_exercise(kind, *, difficulty='medium', seed=None,
                      with_solution=True, _rng=None):
    """Create a geometry or vector exercise by a canonical or friendly name."""
    if not isinstance(kind, str):
        raise TypeError('kind must be text')
    key = kind.strip().lower().replace('-', '_').replace(' ', '_')
    key = _ALIASES.get(key, key)
    try:
        factory = _FACTORIES[key]
    except KeyError as exc:
        choices = ', '.join(GEOMETRY_EXERCISE_TYPES)
        raise ValueError(f'Unknown geometry exercise {kind!r}; choose from {choices}') from exc
    return factory(with_solution=with_solution, difficulty=difficulty, seed=seed, _rng=_rng)


__all__ = [
    'PDFGeometryExercise', 'PDFDistanceBetweenPoints', 'PDFMidpoint', 'PDFSlope',
    'PDFLineEquation', 'PDFPointLineDistance', 'PDFParallelPerpendicularLines',
    'PDFTriangleArea', 'PDFTriangleCentroid', 'PDFPythagoreanTheorem',
    'PDFCircleEquation', 'PDFArcAndSector', 'PDFPolygonAngles',
    'PDFSolidMeasurement', 'PDFCoordinateTransformation', 'PDFVectorFromPoints',
    'PDFVectorRelationship', 'PDFVectorAngle', 'PDFCrossProduct', 'PDFVectorLine',
    'PDFPlaneEquation', 'GEOMETRY_EXERCISE_TYPES', 'geometry_exercise',
]
