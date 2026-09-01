import math

import matplotlib.pyplot as plt
import pytest

import kiwicalc as kw


@pytest.mark.parametrize(
    ("source", "classification"),
    [
        ("f(x)=2x+1", kw.Function.Classification.linear),
        ("f(x)=x^2+1", kw.Function.Classification.quadratic),
        ("f(x)=x^3+1", kw.Function.Classification.polynomial),
        ("f(x)=sin(x)", kw.Function.Classification.trigonometric),
        ("f(x)=2^x", kw.Function.Classification.exponent),
        ("f(x)=7", kw.Function.Classification.constant),
        ("f(x,y)=x+y", kw.Function.Classification.linear_several_parameters),
        ("f(x,y)=x^2+y", kw.Function.Classification.non_linear_several_parameters),
        ("f(x,y)=x^y", kw.Function.Classification.exponent_several_parameters),
        ("f(x)=x==2", kw.Function.Classification.predicate),
    ],
)
def test_function_classification_matrix(source, classification):
    assert kw.Function(source).classification is classification


def test_function_conversion_by_classification():
    assert kw.Function("f(x)=x^2+1").toIExpression() == kw.Poly("x^2+1")
    assert kw.Function("f(x)=sin(x)").toIExpression().when(x=0).try_evaluate() == 0
    assert kw.Function("f(x)=2^x").toIExpression().when(x=3).try_evaluate() == 8
    with pytest.raises(ValueError):
        kw.Function("f(x)=3").toIExpression()


def test_function_derivative_and_integral_validation():
    assert kw.Function("f()=3").derivative() == 0
    assert kw.Function("f(x,y)=x+y").partial_derivative(("x",))(2, 3) == 1
    with pytest.raises(ValueError):
        kw.Function("f(x,y)=x+y").derivative()
    with pytest.raises(ValueError):
        kw.Function("f(x,y)=x+y").integral()


def test_function_ranges_filter_undefined_and_normalize_predicates():
    rational = kw.Function("f(x)=1/x")
    assert rational.range(-1, 1, 1) == ([-1, 1], [-1.0, 1.0])
    predicate = kw.Function("f(x)=x==1")
    assert predicate.range(0, 2, 1) == ([0, 1, 2], [0.0, 1.0, 0.0])
    rounded = kw.Function("f(x)=x/3").range(0, 1, 0.5, round_results=True)
    assert rounded[1] == [0, 0.16667, 0.33333]
    with pytest.warns(UserWarning):
        assert list(kw.Function("f(x,y)=x+y").range_gen(0, 1)) == []


def test_function_random_return_shapes_and_input_safety(monkeypatch):
    monkeypatch.setattr("random.randint", lambda a, b: 2)
    single = kw.Function("f(x)=x^2")
    assert single.random() == 4
    assert single.random(as_tuple=True) == (2, 4)
    assert single.random(as_point=True) == kw.Point2D(2, 4)

    values = [2, 3]
    multiple = kw.Function("f(x,y)=x+y")
    assert multiple.random(custom_values=values) == 5
    assert multiple.random(custom_values=values, as_tuple=True) == ([2, 3], 5)
    assert multiple.random(custom_values=values, as_point=True) == kw.Point3D(2, 3, 5)
    assert values == [2, 3]


def test_function_coefficients_and_extrema_validation():
    assert kw.Function("f(x)=x^3-2x+1").coefficients() == [1, 0, -2, 1]
    with pytest.raises(ValueError):
        kw.Function("f(x)=sin(x)").coefficients()
    with pytest.raises(NotImplementedError):
        kw.Function("f(x)=sin(x)").max_and_min()
    assert kw.Function("f(x)=2x+1").max_and_min() == ([], [])


def test_function_search_roots_and_validation():
    function = kw.Function("f(x)=x^2-1")
    with pytest.warns(UserWarning):
        roots = function.search_roots_in_range((-2, 2), step=0.25, epsilon=1e-6)
    assert (-1, 0) in roots
    assert (1, 0) in roots
    with pytest.raises(IndexError):
        function.search_roots_in_range((0, 1, 2), verbose=False)


def test_function_equality_dimension_and_type_contracts():
    assert kw.Function("f(x)=x+1") == "g(x)=x+1"
    assert kw.Function("f(x)=x+1") != kw.Function("g(x,y)=x+y")
    assert kw.Function("f(x)=x+1") == (lambda x: x + 1)
    with pytest.raises(TypeError):
        kw.Function("f(x)=x") == object()


def test_function_chain_invalid_inputs():
    function = kw.Function("f(x)=x")
    with pytest.raises(TypeError):
        function.chain(object())
    with pytest.raises(ValueError):
        function.chain("not valid (")


def test_function_collection_validation_and_modes(monkeypatch):
    collection = kw.FunctionCollection("f(x)=x", "g(x)=x+1")
    with pytest.raises(TypeError):
        collection.add_function(object())
    with pytest.raises(ValueError):
        collection.random_value(0, 1, mode="decimal")
    monkeypatch.setattr("random.choice", lambda values: values[0])
    monkeypatch.setattr("random.uniform", lambda a, b: 0.5)
    assert collection.random_value(1, 0, mode="float") == pytest.approx(0.5)
    with pytest.raises(ValueError):
        kw.FunctionCollection("f(x,y)=x+y").derivatives()
    assert "1. f(x)=x" in str(collection)


def test_function_plot_all_and_scatter_dispatch():
    kw.Function.plot_all("f(x)=x", kw.Function("g(x)=x^2"), start=0, end=1, step=1)
    with pytest.raises(TypeError):
        kw.Function.plot_all(object())
    kw.Function("f(x)=x").scatter(show=False)
    kw.Function("f(x,y)=x+y").scatter(start=0, stop=1, step=1, show=False)
    with pytest.raises(ValueError):
        kw.Function("f(x,y,z)=x+y+z").scatter(show=False)
    plt.close("all")
