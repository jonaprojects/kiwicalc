import json

import pytest

import kiwicalc as kw


def test_occurrence_intersection_and_union():
    event = kw.Occurrence(0.5, 'event')
    assert event.intersection(0.4) == pytest.approx(0.2)
    assert event.union(0.4) == pytest.approx(0.7)


def test_invalid_probability_assignment_is_rejected():
    event = kw.Occurrence(0.5, 'event')
    with pytest.warns(UserWarning):
        event.chance = 1.5
    assert event.chance == 0.5


def make_tree():
    tree = kw.ProbabilityTree(root=kw.Occurrence(1, 'start'))
    pass_node = tree.add(0.4, 'pass')
    tree.add(0.6, 'fail')
    ace_node = tree.add(0.1, 'ace', parent=pass_node)
    return tree, pass_node, ace_node


def test_probability_tree_navigation_and_path_probability():
    tree, pass_node, ace_node = make_tree()
    assert tree.num_of_nodes() == 4
    assert 'ace' in tree
    assert ace_node in tree
    assert tree.get_node_by_id('pass') is pass_node
    assert tree.get_node_path(ace_node) == 'pass/ace'
    assert tree.get_probability(ace_node) == pytest.approx(0.04)
    assert tree.get_probability('start/pass/ace') == pytest.approx(0.04)


def test_probability_tree_json_round_trip(tmp_path):
    tree, _, _ = make_tree()
    output = tmp_path / 'tree.json'
    tree.export_json(output)

    payload = json.loads(output.read_text(encoding='utf-8'))
    assert payload['ace']['parent'] == 'pass'

    restored = kw.ProbabilityTree(json_path=output)
    assert restored.get_probability('start/pass/ace') == pytest.approx(0.04)


def test_probability_tree_xml_round_trip(tmp_path):
    tree, _, _ = make_tree()
    output = tmp_path / 'tree.xml'
    tree.export_xml(output, root_name='TestTree')

    restored = kw.ProbabilityTree(xml_path=output)
    assert restored.get_probability('start/pass/ace') == pytest.approx(0.04)


def test_create_pdf_writes_valid_pdf_header(tmp_path):
    output = tmp_path / 'worksheet.pdf'
    assert kw.create_pdf(output, title='Test Worksheet', lines=['x + 1 = 2']) is True
    assert output.read_bytes().startswith(b'%PDF')
    assert output.stat().st_size > 100


def test_create_pages_writes_valid_pdf_header(tmp_path):
    output = tmp_path / 'pages.pdf'
    kw.create_pages(output, 2, ['Page 1', 'Page 2'], [['one'], ['two']])
    assert output.read_bytes().startswith(b'%PDF')
