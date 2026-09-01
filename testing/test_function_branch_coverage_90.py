import math

import matplotlib.pyplot as plt
import pytest

import kiwicalc as kw


MODULE_LAMBDA = lambda x: x + 1


def test_function_constructor_and_power_role_edges():
    assert kw.Function(MODULE_LAMBDA)(2) == 3
    with pytest.raises(TypeError):
        kw.Function(None)

    exponent = kw.Function("f(x)=x^n")
    assert exponent.classification is kw.Function.Classification.exponent
    malformed = kw.Function("f(x)=x^")
    assert malformed.classification is kw.Function.Classification.exponent
    with pytest.warns(UserWarning):
        assert malformed(2) is None


def test_function_conversion_and_calculus_unavailable_paths(monkeypatch):
    function = kw.Function("f(x)=x")
    function._Function__classification = kw.Function.Classification.logarithmic
    function._Function__func_expression = "log2(x)"
    assert isinstance(function.toIExpression(), kw.Log)

    no_expression = kw.Function("f(x)=x")
    monkeypatch.setattr(no_expression, "toIExpression", lambda: None)
    no_expression._Function__classification = kw.Function.Classification.trigonometric
    assert no_expression.derivative() is None
    assert no_expression.integral() is None

    command = kw.Function("f()=2")
    assert command.partial_derivative(("x",)) == 0
    assert kw.Function("f(x)=x").partial_derivative(("x",))() == 1

    multi = kw.Function("f(x,y)=x+y")
    monkeypatch.setattr(multi, "toIExpression", lambda: None)
    assert multi.partial_derivative(("x",)) is None


def test_function_range_random_and_plot_dispatch_edges(monkeypatch):
    rounded = kw.Function("f(x)=1/x").range(-1, 1, 1, round_results=True)
    assert rounded == ([-1, 1], [-1.0, 1.0])
    assert kw.Function("f(x,y)=x+y").range_3d(0, 1, 1, 0, 1, 1, round_results=True)[2] == [0, 1, 1, 2]
    assert kw.Function("f(a,b,c)=a+b+c").random(custom_values=(1, 2, 3), as_point=True) == kw.Point((1, 2, 3, 6))

    import kiwicalc.plotting.plots as plots
    calls = []
    monkeypatch.setattr(plots, "plot_function", lambda **kwargs: calls.append("one"))
    monkeypatch.setattr(plots, "plot_functions", lambda *args, **kwargs: calls.append("many"))
    monkeypatch.setattr(plots, "plot_function_3d", lambda **kwargs: calls.append("3d"))
    kw.Function("f(x)=x").plot(others=(kw.Function("g(x)=2x"),), show=False)
    kw.Function("f(x,y)=x+y").plot(show=False)
    kw.Function("f(x,y,z)=x+y+z").plot(show=False)
    assert calls == ["many", "3d"]


def test_function_scatter2d_nonbasic_display_branches(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    function = kw.Function("f(x)=x")
    assert function.scatter2d(start=0, stop=1, step=1, show_axis=False, show=False, basic=False) is None
    assert function.scatter2d(start=0, stop=1, step=1, show_axis=True, show=True, basic=False) is None
    plt.close("all")


def test_function_newton_extrema_chain_and_index_edges(monkeypatch):
    function = kw.Function("f(x)=x^2")
    function._Function__lambda_expression = None
    monkeypatch.setattr("kiwicalc.numeric.roots.newton_raphson", lambda *args: "fallback")
    assert function.newton(1) == "fallback"

    maxima, minima = kw.Function("f(x)=-x^2").max_and_min()
    assert maxima[0] == pytest.approx((0, 0))
    assert minima == []

    assert isinstance(kw.Function("f(x)=x").chain(kw.Function("g(x)=x+1")), kw.FunctionChain)
    sliced = kw.Function("f(x,y,z)=x+y+z")
    assert sliced[0:3:2] == ["x", "z"]
    assert sliced[object()] is None


def test_function_equality_and_intersection_edge_paths():
    function = kw.Function("f(x)=x")
    assert not (function == None)  # noqa: E711
    assert function != kw.Function("f(x)=x+1")
    assert function != (lambda x, y: x + y)
    assert function != (lambda x: x + 1)

    with pytest.raises(TypeError):
        function.search_intersections(math.sin)
    with pytest.raises(TypeError):
        function.search_intersections(object(), values_range=(0, 1))
    assert function.search_intersections("g(x)=x", values_range=(0, 0.3), step=0.05, precision=0.1)
    unavailable_first = kw.Function("f(x)=x")
    unavailable_first.compute_value = lambda value: None
    assert unavailable_first.search_intersections(
        kw.Function("g(x)=0"), values_range=(0, 0.2), step=0.1
    ) == []
    unavailable_second = kw.Function("g(x)=x")
    unavailable_second.compute_value = lambda value: None
    assert kw.Function("f(x)=0").search_intersections(
        unavailable_second, values_range=(0, 0.2), step=0.1
    ) == []


def test_function_collection_constructor_and_mutator_edges():
    original = kw.Function("f(x)=x")
    copied = kw.FunctionCollection(original, gen_copies=True)
    assert copied[0] == original and copied[0] is not original

    nested = kw.FunctionCollection("f(x)=x", "g(x)=2x")
    flattened = kw.FunctionCollection(nested)
    copied_flattened = kw.FunctionCollection(nested, gen_copies=True)
    assert len(flattened) == len(copied_flattened) == 2
    assert flattened[0] is nested[0]
    assert copied_flattened[0] is not nested[0]

    empty = kw.FunctionCollection()
    assert empty.num_of_variables == 0
    empty.add_function(original)
    empty.extend([original])
    assert empty.functions == [original, original]
