import json

import matplotlib.pyplot as plt
import pytest

import kiwicalc as kw


def test_function_metadata_classification_and_calling():
    function = kw.Function("square(x)=x^2")
    assert function.function_signature == "square(x)"
    assert function.function_expression == "x**2"
    assert function.variables == ["x"]
    assert function.num_of_variables == 1
    assert function(4) == 16
    assert function.apply_on([-2, 0, 3]) == [4, 0, 9]
    assert "quadratic" in str(function.classification).lower()


def test_function_calculus_and_ranges():
    function = kw.Function("f(x)=x^2+2x+1")
    assert function.derivative()(3) == pytest.approx(8)
    integral = function.integral()
    assert integral.derivative().when(x=3).try_evaluate() == pytest.approx(function(3), abs=1e-4)
    assert list(function.range_gen(0, 2, 1)) == [(0, 1), (1, 4), (2, 9)]
    assert function.range(0, 2, 1) == ([0, 1, 2], [1, 4, 9])


def test_function_numerical_integration_contract():
    function = kw.Function("f(x)=x^2+2x+1")
    assert function.trapz(0, 1, 1000) == pytest.approx(7 / 3, rel=1e-4)
    assert function.simpson(0, 1, 1001) == pytest.approx(7 / 3, rel=1e-4)


def test_multivariate_function_range():
    function = kw.Function("f(x,y)=x^2+3y")
    x_values, y_values, results = function.range_3d(0, 1, 1, 0, 1, 1)
    assert x_values == [0, 0, 1, 1]
    assert y_values == [0, 1, 0, 1]
    assert results == [0, 3, 1, 4]


def test_multivariate_function_partial_derivative_contract():
    derivative = kw.Function("f(x,y)=x^2+3y").partial_derivative(("x",))
    assert derivative(4, 10) == pytest.approx(8)


def test_function_serialization_and_copy(tmp_path):
    function = kw.Function("f(x)=sin(x)+1")
    payload = function.to_dict()
    assert kw.Function.from_dict(payload) == function
    assert kw.Function.from_json(function.to_json()) == function

    output = tmp_path / "function.json"
    function.export_json(output)
    assert json.loads(output.read_text()) == payload

    copied = function.__copy__()
    assert copied == function
    assert copied is not function


def test_function_sequence_protocol_and_context_manager():
    function = kw.Function("f(x,y)=x+y")
    assert function[0] == "x"
    function[0] = "a"
    assert function(2, 3) == 5
    del function[1]
    assert function.variables == ["a"]

    with kw.Function("f(x)=2x") as active:
        assert active(3) == 6


def test_function_roots_and_intersections():
    quadratic = kw.Function("f(x)=x^2-4")
    roots = quadratic.roots()
    assert sorted(root.real for root in roots) == pytest.approx([-2, 2])
    intersections = kw.Function("f(x)=x").search_intersections(
        kw.Function("g(x)=2-x"), values_range=(-1, 3), step=0.1
    )
    assert any(abs(point[0] - 1) < 0.11 for point in intersections)


def test_function_extrema_contract():
    maxima, minima = kw.Function("f(x)=x^2").max_and_min()
    assert maxima == []
    assert minima[0] == pytest.approx((0, 0))


def test_function_newton_contract():
    assert kw.Function("f(x)=x^2-4").newton(3) == pytest.approx(2)


def test_function_chain_execution_and_indexing():
    chain = kw.FunctionChain("f(x)=x+1", "g(x)=2x", "h(x)=x-3")
    assert chain.execute_all(4) == 7
    assert chain(4) == 7
    assert chain.execute_indices([0, 2], 4) == 2
    assert chain[0:2](4) == 10
    assert chain.chain("q(x)=x^2")(2) == 9

    with pytest.raises(ValueError):
        kw.FunctionChain().execute_all(1)
    with pytest.raises(ValueError):
        kw.FunctionChain().execute_reverse(1)
    with pytest.raises(ValueError):
        chain.execute_indices([], 1)


def test_function_collection_operations(monkeypatch):
    collection = kw.FunctionCollection("f(x)=x", "g(x)=x^2")
    assert len(collection) == 2
    assert collection.variables == {"x"}
    assert collection.num_of_variables == 1
    derivatives = collection.derivatives()
    assert derivatives[0]() == 1
    assert derivatives[1](3) == 6
    assert len(list(collection.filter(lambda function: function(2) > 2))) == 1
    assert list(collection) == collection.functions

    collection.add_function("h(x)=x+3")
    collection.extend(["i(x)=2x"])
    assert len(collection) == 4
    assert collection.random_function() in collection.functions
    monkeypatch.setattr("random.choice", lambda values: values[0])
    monkeypatch.setattr("random.randint", lambda a, b: a)
    assert collection.random_value(2, 5) == collection.functions[0](2)
    collection.clear()
    assert collection.is_empty()


def test_function_collection_count_and_values_contract():
    collection = kw.FunctionCollection("f(x)=x", "g(x)=x^2")
    assert collection.num_of_functions == 2
    assert collection.values(3) == [3, 9]


def test_function_plotting_paths_without_showing():
    fig, ax = plt.subplots()
    kw.Function("f(x)=x^2").plot(values=[-1, 0, 1], show=False, fig=fig, ax=ax)
    assert ax.lines
    plt.close(fig)

    chain = kw.FunctionChain("f(x)=x+1", "g(x)=2x")
    fig, ax = plt.subplots()
    chain.plot(values=[0, 1], show=False, fig=fig, ax=ax)
    assert ax.lines
    plt.close(fig)
