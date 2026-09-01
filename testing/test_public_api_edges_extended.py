import pytest

import kiwicalc as kw


def test_function_alternate_constructors_and_y_intersection():
    assert kw.Function("lambda x: x + 1")(2) == 3
    assert kw.Function("x=>x + 1")(2) == 3
    assert kw.Function(kw.Poly("x^2+1"))(3) == 10
    assert kw.Function("f(x)=x^2+3").y_intersection() == 3
    with pytest.raises(TypeError):
        kw.Function(object())


def test_function_finite_integral_dispatch_and_validation():
    function = kw.Function("f(x)=x")
    assert function.finite_integral(0, 1, 1000, "trapz") == pytest.approx(0.5)
    assert function.finite_integral(0, 1, 1001, "simpson") == pytest.approx(0.5)
    assert function.finite_integral(0, 1, 1000, "reinman") == pytest.approx(0.5, abs=0.002)
    with pytest.raises(TypeError):
        function.finite_integral(0, 1, 10, 1)
    with pytest.raises(ValueError):
        function.finite_integral(0, 1, 10, "unknown")


def test_function_chain_reverse_and_collection_indexing():
    chain = kw.FunctionChain("f(x)=x+1", "g(x)=2x", "h(x)=x-3")
    assert chain.execute_reverse(4) == 3

    collection = kw.FunctionCollection("f(x)=x", "g(x)=x^2", "h(x)=x+2")
    assert isinstance(collection[0], kw.Function)
    assert collection[0](3) == 3
    subset = collection[1:]
    assert isinstance(subset, kw.FunctionCollection)
    assert subset.values(3) == [9, 5]


def test_point_right_hand_arithmetic_and_scaling_all_coordinates():
    point = kw.Point((1, 2, 3))
    assert point * 3 == kw.Point((3, 6, 9))
    assert (10, 20, 30) + point == kw.Point((11, 22, 33))
    assert (10, 20, 30) - point == kw.Point((9, 18, 27))
    assert point * kw.Point((4, 5, 6)) == 32


def test_point_metrics_and_dimension_validation():
    point = kw.Point((3, -1, 5))
    assert point.coord_at(1) == -1
    assert point.max_coord() == 5
    assert point.min_coord() == -1
    assert point.sum() == 7
    assert point.distance(kw.Point((0, -1, 1))) == 5
    with pytest.raises(ValueError):
        point.distance(kw.Point((1, 2)))


def test_vector_construction_products_and_validation():
    assert kw.Vector(start_coordinate=(1, 2), end_coordinate=(4, 6)) == kw.Vector((3, 4))
    assert kw.Vector(direction_vector=(3, 4), end_coordinate=(4, 6)).start_coordinate == [1, 2]
    assert kw.Vector((1, 2, 3)).scalar_product((4, 5, 6)) == 32
    assert kw.Vector((1, 2)) + 3 == kw.Vector((4, 5))
    assert kw.Vector((3, 4)) - 1 == kw.Vector((2, 3))
    with pytest.raises(ValueError):
        kw.Vector(start_coordinate=(1, 2), end_coordinate=(3, 4, 5))


def test_vector_collection_mutating_protocols():
    collection = kw.VectorCollection(kw.Vector((2, 4)), kw.Vector((6, 8)))
    divided = collection / 2
    assert divided == kw.VectorCollection(kw.Vector((1, 2)), kw.Vector((3, 4)))
    assert collection == kw.VectorCollection(kw.Vector((2, 4)), kw.Vector((6, 8)))

    del collection[0]
    assert collection == kw.Vector((6, 8))
    collection.vectors = [kw.Vector((1, 0)), kw.Vector((0, 1))]
    assert collection.num_of_vectors == 2
    assert collection.vectors == [kw.Vector((1, 0)), kw.Vector((0, 1))]
    with pytest.raises(ValueError):
        collection.nlongest(3)


def test_matrix_scalar_transforms_and_function_mapping():
    matrix = kw.Matrix([[1, 2], [3, 4]])
    matrix.subtract_from_all(1)
    assert matrix == [[0, 1], [2, 3]]
    matrix.apply_to_all(lambda value: value * value)
    assert matrix == [[0, 1], [4, 9]]
    assert kw.Function("f(x)=x+1").compute_value(kw.Matrix([[1, 2], [3, 4]])) == [[2, 3], [4, 5]]


def test_matrix_filtering_and_rectangular_reversal():
    matrix = kw.Matrix([[1, 2, 3], [4, 5, 6]])
    assert matrix.filter_by_indices(lambda row, column: (row + column) % 2 == 0) == [[1, 3], [5]]
    assert matrix.reversed_rows() == [[4, 5, 6], [1, 2, 3]]
    assert matrix.reversed_columns() == [[3, 2, 1], [6, 5, 4]]
    assert reversed(matrix) == [[4, 5, 6], [1, 2, 3]]
    assert list(matrix.iterate_by_columns()) == [1, 4, 2, 5, 3, 6]
    assert list(matrix.range()) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def test_matrix_mapping_filtering_and_unit_matrix():
    matrix = kw.Matrix([[1, 2], [3, 4]])
    assert matrix.mapped_matrix(lambda value: value * 10) == [[10, 20], [30, 40]]
    assert matrix.filtered_matrix(lambda value: value % 2 == 0, get_list=True) == [[2], [4]]
    assert matrix.foreach_item(lambda value: value + 1) == [[2, 3], [4, 5]]
    assert kw.Matrix.unit_matrix(3) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert kw.Matrix.is_unit_matrix(kw.Matrix.unit_matrix(3))


def test_fraction_evaluation_serialization_and_calculus():
    fraction = kw.Fraction(1, 2)
    assert fraction.try_evaluate() == 0.5
    assert fraction.python_syntax() == "(1)/(2)"
    assert kw.Fraction.from_dict(fraction.to_dict()) == fraction
    assert kw.Fraction(kw.Var("x"), 2).derivative().when(x=4).try_evaluate() == pytest.approx(0.5)
    with pytest.raises(ZeroDivisionError):
        kw.Fraction(1, 0).try_evaluate()


def test_polyfraction_analysis_and_reciprocal():
    fraction = kw.PolyFraction("x^2-1/x-2")
    assert sorted(round(complex(root).real, 6) for root in fraction.roots()) == [-1, 1]
    assert [round(complex(root).real, 6) for root in fraction.invalid_values()] == [2]
    assert fraction.reciprocal().reciprocal() == fraction
    assert kw.PolyFraction("x/x^2+1").horizontal_asymptote() == 0
