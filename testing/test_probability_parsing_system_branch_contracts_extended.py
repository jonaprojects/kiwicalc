import importlib
import json

import pytest
from anytree import Node

import kiwicalc as kw
from kiwicalc.parsing import parse_equation


def make_probability_tree():
    tree = kw.ProbabilityTree(root=kw.Occurrence(1, "root"))
    left = tree.add(0.4, "left")
    right = tree.add(0.6, "right")
    leaf = tree.add(0.25, "leaf", parent=left)
    return tree, left, right, leaf


def test_occurrence_property_and_object_arithmetic_branches():
    first = kw.Occurrence(0.5, "first")
    second = kw.Occurrence(0.2, "second")
    first.chance = 0.75
    first.identifier = "renamed"
    assert first.chance == 0.75
    assert first.identifier == "renamed"
    assert first.intersection(second, 0.5) == pytest.approx(0.075)
    assert first.union(second, 0.5) == pytest.approx(0.9)
    assert "renamed" in str(first)
    assert "Occurrence" in repr(first)


def test_probability_tree_default_invalid_and_warning_branches():
    default = kw.ProbabilityTree()
    assert default.root.name.identifier == "root"
    with pytest.raises(TypeError):
        kw.ProbabilityTree(root="root")
    with pytest.warns(UserWarning, match="expected 1 or less"):
        default.add(0.8, "one")
        default.add(0.8, "two")


def test_probability_tree_navigation_false_and_error_branches():
    tree, left, right, leaf = make_probability_tree()
    assert tree.get_probability() == 1
    assert tree.get_probability(["root", "left", "leaf"]) == pytest.approx(0.1)
    assert tree.get_probability(("root", "right")) == pytest.approx(0.6)
    assert tree.get_node_path(left) == "left"
    assert tree.get_node_path("leaf") == "left/leaf"
    assert tree.get_node_by_id("missing") is None
    assert Node(kw.Occurrence(1, "outside")) not in tree
    assert "missing" not in tree
    with pytest.raises(TypeError):
        _ = 3 in tree
    assert kw.ProbabilityTree.biggest_probability_node(tree.root) is tree.root
    assert "root:1" in str(tree)
    payload = tree.to_dict()
    assert payload["root"]["parent"] is None
    assert payload["leaf"]["parent"] == 'left'
    assert tree.remove(left) is tree
    assert 'left' not in tree and 'leaf' not in tree


def test_probability_json_unknown_parent_warning(tmp_path):
    path = tmp_path / "orphan.json"
    path.write_text(
        json.dumps(
            {
                "root": {"parent": None, "_identifier": "root", "_chance": 1},
                "orphan": {"parent": "missing", "_identifier": "orphan", "_chance": 0.2},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unresolved"):
        kw.ProbabilityTree.tree_from_json(path)


def test_probability_xml_unknown_parent_warning(tmp_path):
    path = tmp_path / "orphan.xml"
    path.write_text(
        "<Tree>"
        "<node><parent>None</parent><identifier>root</identifier><chance>1</chance></node>"
        "<node><parent>missing</parent><identifier>orphan</identifier><chance>0.2</chance></node>"
        "</Tree>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unresolved"):
        kw.ProbabilityTree.tree_from_xml(path)


def test_coefficient_list_alignment_modes_and_empty_result():
    assert parse_equation.add_or_sub_coefficients([1, 2, 3], [4]) == [1, 2, 7]
    assert parse_equation.add_or_sub_coefficients([1], [2, 3]) == [2, 4]
    target = [1, 2]
    result = parse_equation.add_or_sub_coefficients(target, [1, 2], mode="sub", copy_first=False)
    assert result is target and result == []
    assert parse_equation.add_or_sub_coefficients([1], [2], mode="noop") == [1]


def test_equation_string_parsing_and_validation_branches():
    assert parse_equation.extract_dict_from_equation("3=2") == {"number": 0}
    assert parse_equation.linear_expression_to_dict("x-y+2.5", ("x", "y")) == {
        "x": 1, "y": -1, "number": 2.5
    }
    assert parse_equation.equation_to_one_side("x=2-y") == "x-2+y"
    assert parse_equation.equation_to_one_side("x=-2") == "x+2"
    with pytest.raises(ValueError, match="two sides"):
        parse_equation.equation_to_one_side("x+1")


def test_simplify_expression_all_token_branches():
    assert parse_equation.simplify_expression("x-x+2", ("x",)) == {"x": 0, "number": 2}
    assert parse_equation.simplify_expression("-x+3!", {"x": 4}, format_factorial=True) == {
        "x": 3, "number": 6
    }
    assert parse_equation.simplify_expression("|3|+x", ("x",), format_abs=True) == {
        "x": 1, "number": 3
    }
    with pytest.raises(ValueError, match="Unrecognized expression"):
        parse_equation.simplify_expression("bad", ("x",))


def test_parse_equation_degree_validation_branches():
    with pytest.raises(ValueError, match="1 variable"):
        parse_equation.ParseEquation.parse_polynomial("x+y=1")
    assert parse_equation.ParseEquation.parse_quadratic("x^2-3x+2=0", strict_syntax=True) == [1, -3, 2]
    with pytest.raises(ValueError, match="degree-2"):
        parse_equation.ParseEquation.parse_quadratic("x^3-1=0", strict_syntax=True)
    assert parse_equation.coefficients_to_expressions([0, 2, 0, -1], "t")[0] == kw.Mono("2t^2")


def test_linear_system_constructor_mutation_and_print(capsys):
    equation = kw.LinearEquation("x+y=3")
    system = kw.LinearSystem((equation, "x-y=1"), variables=("x", "y"))
    assert system.equations[0] is equation
    assert system.variables == ["x", "y"]
    system.add_equation("2x=4")
    assert len(system.equations) == 3
    with pytest.raises(TypeError):
        kw.LinearSystem((object(),))
    two_equations = kw.LinearSystem(("x+y=3", "x-y=1"), variables=("x", "y"))
    two_equations.print_solutions()
    output = capsys.readouterr().out
    assert "x = 2" in output and "y = 1" in output
    assert two_equations.to_matrix_and_vector() is None
    assert two_equations.simplify() is None


def test_system_solvers_infer_variables_and_initial_values():
    assert kw.solve_linear_system(("x+y=3", "x-y=1")) == pytest.approx({"x": 2, "y": 1})
    assert kw.solve_poly_system(("x-2=0",), initial_vals=None) == pytest.approx({"x": 2})
    assert kw.solve_poly_system((kw.Poly("x-2"),), initial_vals={"x": 0}) == pytest.approx({"x": 2})


def test_random_linear_system_all_operation_branches(monkeypatch):
    system_module = importlib.import_module("kiwicalc.equations.system")
    operation_calls = iter((0, 1, 0, 1))

    def fake_randint(start, stop):
        if (start, stop) == (2, 5):
            return 2
        if (start, stop) == (0, 1):
            return next(operation_calls)
        if (start, stop) == (1, 3):
            return 2
        return 1

    monkeypatch.setattr(system_module.random, "uniform", lambda start, stop: 2)
    monkeypatch.setattr(system_module.random, "randint", fake_randint)
    monkeypatch.setattr(system_module.random, "choice", lambda values: values[0])
    equations, solutions = kw.random_linear_system(("x", "y"), get_solutions=True)
    assert len(equations) == 2
    assert solutions == [2, 2]
    assert all("=" in equation for equation in equations)
