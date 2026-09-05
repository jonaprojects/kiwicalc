import json
import math

from anytree import Node
import pytest

import kiwicalc as kw


@pytest.mark.parametrize('value', [-0.01, 1.01, math.inf, -math.inf, math.nan])
def test_occurrence_rejects_out_of_range_or_nonfinite_probability(value):
    with pytest.raises(ValueError):
        kw.Occurrence(value)


@pytest.mark.parametrize('value', [True, False, '0.5', None, object()])
def test_occurrence_rejects_non_real_probability(value):
    with pytest.raises(TypeError):
        kw.Occurrence(value)


def test_occurrence_assignment_preserves_legacy_warning_contract():
    occurrence = kw.Occurrence(0.4, 'rain')
    with pytest.warns(UserWarning, match='was not changed'):
        occurrence.chance = -1
    assert occurrence.chance == 0.4
    occurrence.chance = 0.25
    assert occurrence.chance == 0.25


def test_explicit_event_probability_operations():
    first = kw.Occurrence(0.5, 'first')
    second = kw.Occurrence(0.4, 'second')
    assert first.complement == 0.5
    assert first.independent_intersection(second, 0.2) == pytest.approx(0.04)
    assert first.intersection(second) == pytest.approx(0.2)
    assert first.independent_union(second, 0.2) == pytest.approx(0.76)
    assert first.union(second) == pytest.approx(0.7)
    assert first.union_with_overlap(second, 0.1) == pytest.approx(0.8)
    with pytest.raises(ValueError, match='inconsistent'):
        first.union_with_overlap(second, 0.6)
    with pytest.raises((TypeError, ValueError)):
        first.independent_union(2)


def test_occurrence_identity_and_identifier_validation():
    assert kw.Occurrence(0.5, 'event') == kw.Occurrence(0.5, 'event')
    assert kw.Occurrence(0.5, 'event') != kw.Occurrence(0.4, 'event')
    assert kw.Occurrence(0.5, 'event') != object()
    with pytest.raises(TypeError, match='identifier'):
        kw.Occurrence(0.5, 3)
    with pytest.raises(ValueError, match='separates paths'):
        kw.Occurrence(0.5, 'a/b')


def make_complete_tree():
    tree = kw.ProbabilityTree(root=kw.Occurrence(1, 'start'))
    left = tree.add(0.6, 'left')
    right = tree.add(0.4, 'right')
    low = tree.add(0.5, 'low', parent=left)
    high = tree.add(0.5, 'high', parent=left)
    return tree, left, right, low, high


def test_tree_rejects_ambiguous_nodes_and_foreign_parents():
    tree, left, _, _, _ = make_complete_tree()
    with pytest.raises(ValueError, match='already exists'):
        tree.add(0.1, 'left')
    with pytest.raises(ValueError, match='cannot be empty'):
        tree.add(0.1, '')
    with pytest.raises(ValueError, match='separates paths'):
        tree.add(0.1, 'bad/path')
    with pytest.raises(TypeError, match='parent'):
        tree.add(0.1, 'bad-parent', parent='left')
    other = kw.ProbabilityTree()
    with pytest.raises(ValueError, match='does not belong'):
        tree.add(0.1, 'foreign', parent=other.root)
    left.name.identifier = 'renamed-left'
    assert tree.get_node_by_id('renamed-left') is left
    with pytest.raises(ValueError, match='already exists'):
        left.name.identifier = 'right'


def test_tree_strict_add_is_atomic_and_legacy_add_warns():
    tree = kw.ProbabilityTree()
    tree.add(0.8, 'first')
    with pytest.raises(ValueError, match='expected 1 or less'):
        tree.add(0.3, 'strict', strict=True)
    assert 'strict' not in tree
    with pytest.warns(UserWarning, match='expected 1 or less'):
        tree.add(0.3, 'legacy')
    with pytest.raises(TypeError, match='strict'):
        tree.add(0.1, 'invalid-strict', strict='yes')


def test_tree_preserves_probability_precision_and_explicit_rounding():
    tree = kw.ProbabilityTree(root=kw.Occurrence(1, 'root'))
    first = tree.add(0.1234567, 'first')
    leaf = tree.add(0.1234567, 'leaf', parent=first)
    exact = 0.1234567**2
    assert tree.get_probability(leaf) == exact
    assert tree.get_probability('root/first/leaf', ndigits=5) == round(exact, 5)
    with pytest.raises(TypeError, match='ndigits'):
        tree.get_probability(leaf, ndigits=True)


def test_tree_paths_leaves_and_most_likely_outcome():
    tree, left, right, low, high = make_complete_tree()
    right_leaf = tree.add(0.9, 'right-leaf', parent=right)
    assert tree.leaves == (low, high, right_leaf)
    assert tree.get_node_path(high) == 'left/high'
    assert tree.path_probabilities() == pytest.approx({
        'left/low': 0.3, 'left/high': 0.3, 'right/right-leaf': 0.36,
    })
    assert tree.most_likely_leaf() is right_leaf
    assert set(tree.path_probabilities(leaves_only=False)) == {
        'start', 'left', 'left/low', 'left/high', 'right', 'right/right-leaf',
    }
    with pytest.raises(ValueError, match='Unknown'):
        tree.get_node_path('missing')
    with pytest.raises(ValueError, match='does not belong'):
        tree.get_node_path(Node(kw.Occurrence(1, 'outside')))


def test_tree_completeness_and_validation():
    tree, left, right, _, _ = make_complete_tree()
    assert tree.validate(require_complete=True)
    assert tree.is_complete()
    tree.add(0.4, 'partial', parent=right)
    assert tree.validate()
    assert not tree.is_complete()
    with pytest.raises(ValueError, match='do not sum'):
        tree.validate(require_complete=True)
    with pytest.raises(TypeError, match='require_complete'):
        tree.validate(require_complete='yes')
    for tolerance in (-1, math.nan, math.inf, True):
        with pytest.raises(ValueError):
            tree.validate(tolerance=tolerance)


def test_tree_removal_detaches_subtrees_and_protects_root():
    tree, left, _, low, _ = make_complete_tree()
    assert tree.remove('left') is tree
    assert 'left' not in tree and 'low' not in tree
    assert low.root is left
    with pytest.raises(ValueError, match='root'):
        tree.remove(tree.root)
    with pytest.raises(ValueError, match='Unknown'):
        tree.remove('missing')
    with pytest.raises(TypeError, match='nodes'):
        tree.remove(3)


def test_json_serialization_is_stable_order_independent_and_copyable(tmp_path):
    tree, _, _, _, _ = make_complete_tree()
    payload = tree.to_dict()
    assert payload['left']['parent'] == 'start'
    json.dumps(payload)
    reordered = dict(reversed(tuple(payload.items())))
    restored = kw.ProbabilityTree.from_dict(reordered)
    assert restored == tree
    assert kw.ProbabilityTree.from_json(tree.to_json()) == tree
    assert kw.ProbabilityTree.from_json(tree.to_json().encode()) == tree
    output = tmp_path/'tree.json'
    tree.export_json(output)
    assert kw.ProbabilityTree.from_json(output) == tree
    clone = tree.copy()
    assert clone == tree and clone is not tree
    clone.add(0.1, 'new-leaf', parent=clone.get_node_by_id('right'))
    assert clone != tree


@pytest.mark.parametrize('payload', [
    {},
    {'a': {'parent': None, '_identifier': 'a', '_chance': 1},
     'b': {'parent': None, '_identifier': 'b', '_chance': 1}},
    {'root': {'parent': None, '_identifier': 'different', '_chance': 1}},
    {'root': {'parent': None, '_identifier': 'root', '_chance': 1},
     'orphan': {'parent': 'missing', '_identifier': 'orphan', '_chance': .2}},
])
def test_from_dict_rejects_malformed_or_disconnected_trees(payload):
    with pytest.raises((TypeError, ValueError)):
        kw.ProbabilityTree.from_dict(payload)


def test_xml_roundtrip_escapes_text_and_accepts_empty_root_parent(tmp_path):
    tree = kw.ProbabilityTree(root=kw.Occurrence(1, 'root & origin'))
    tree.add(1, 'result <ok>')
    xml = tree.to_xml_str('ProbabilityTree')
    assert '&amp;' in xml and '&lt;' in xml
    output = tmp_path/'tree.xml'
    tree.export_xml(output, root_name='ProbabilityTree')
    assert kw.ProbabilityTree.from_xml(output) == tree
    empty_parent = tmp_path/'empty-parent.xml'
    empty_parent.write_text(
        '<Tree><node><parent/><identifier>root</identifier><chance>1</chance></node></Tree>',
        encoding='utf-8')
    assert kw.ProbabilityTree.from_xml(empty_parent).root.name.identifier == 'root'
    with pytest.raises(ValueError, match='valid XML'):
        tree.to_xml_str('not valid')


def test_tree_constructor_and_node_helper_validation():
    with pytest.raises(ValueError, match='only one'):
        kw.ProbabilityTree(root=kw.Occurrence(1, 'root'), json_path='tree.json')
    with pytest.raises(ValueError, match='cannot be empty'):
        kw.ProbabilityTree(root=kw.Occurrence(1, ''))
    with pytest.raises(TypeError, match='node'):
        kw.ProbabilityTree.get_depth('node')
    with pytest.raises(TypeError, match='node'):
        kw.ProbabilityTree.get_level_sum('node')
    with pytest.raises(TypeError, match='node'):
        kw.ProbabilityTree.biggest_probability_node('node')
