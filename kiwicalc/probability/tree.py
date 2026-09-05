"""Validated probability occurrences and conditional-probability trees."""
from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
import re
import warnings
from xml.etree.ElementTree import Element, SubElement, tostring

from anytree import Node, PreOrderIter, RenderTree

try:
    from defusedxml.ElementTree import parse
except ImportError:  # pragma: no cover - compatibility fallback
    from xml.etree.ElementTree import parse


_XML_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_.-]*$')


def _probability(value, *, name='probability') -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real number')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{name} must be finite and between 0 and 1')
    return value


def _identifier(value, *, allow_empty=True) -> str:
    if not isinstance(value, str):
        raise TypeError('identifier must be text')
    if not allow_empty and not value.strip():
        raise ValueError('identifier cannot be empty in a probability tree')
    if '/' in value:
        raise ValueError("identifier cannot contain '/' because it separates paths")
    return value


def _chance_of(value) -> float:
    if isinstance(value, Occurrence):
        return value.chance
    return _probability(value)


class Occurrence:
    """A named probability value.

    ``intersection`` and ``union`` retain their historical independent-event
    semantics. The explicit ``independent_*`` names are preferred in new code.
    """

    def __init__(self, chance: float = 1, identifier: str = ''):
        self._chance = _probability(chance, name='chance')
        self._identifier = _identifier(identifier)
        self._tree = None
        self._node = None

    @property
    def chance(self):
        return self._chance

    @chance.setter
    def chance(self, chance: float):
        # Preserve the legacy non-mutating warning contract for assignments.
        try:
            value = _probability(chance, name='chance')
        except (TypeError, ValueError) as exc:
            warnings.warn(f'Occurrence.chance was not changed: {exc}', UserWarning,
                          stacklevel=2)
            return
        self._chance = value

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, identifier: str):
        identifier = _identifier(identifier, allow_empty=self._tree is None)
        if self._tree is not None:
            existing = self._tree.get_node_by_id(identifier)
            if existing is not None and existing is not self._node:
                raise ValueError(f"identifier {identifier!r} already exists in this tree")
        self._identifier = identifier

    @property
    def complement(self):
        """Return ``1 - chance``."""
        return 1-self.chance

    def independent_intersection(self, *occurrences):
        """Multiply probabilities, assuming all supplied events are independent."""
        result = self.chance
        for occurrence in occurrences:
            result *= _chance_of(occurrence)
        return result

    def intersection(self, *occurrences):
        """Compatibility alias for :meth:`independent_intersection`."""
        return self.independent_intersection(*occurrences)

    def independent_union(self, *occurrences):
        """Return an independent-event union probability."""
        complement = 1-self.chance
        for occurrence in occurrences:
            complement *= 1-_chance_of(occurrence)
        return 1-complement

    def union(self, *occurrences):
        """Compatibility alias for :meth:`independent_union`."""
        return self.independent_union(*occurrences)

    def union_with_overlap(self, other, overlap):
        """Return ``P(A union B)`` when ``P(A intersection B)`` is known."""
        other_chance = _chance_of(other)
        overlap = _probability(overlap, name='overlap')
        lower, upper = max(0., self.chance+other_chance-1), min(self.chance, other_chance)
        if not lower <= overlap <= upper:
            raise ValueError('overlap is inconsistent with the two event probabilities')
        return self.chance+other_chance-overlap

    def __str__(self):
        return f'probability: {self.chance} , _identifier: {self.identifier}'

    def __repr__(self):
        return f'Occurrence(_chance={self.chance},_identifier={self.identifier})'

    def __eq__(self, other):
        if not isinstance(other, Occurrence):
            return NotImplemented
        return self.chance == other.chance and self.identifier == other.identifier


class ProbabilityTree:
    """A validated tree of conditional branch probabilities.

    Identifiers are unique across a tree and paths use ``/`` separators. Branch
    probabilities are conditional on their parent; cumulative path probability
    is their product. ``add`` preserves the legacy warning when siblings exceed
    one, while ``validate`` provides strict whole-tree checks.
    """

    def __init__(self, root=None, json_path=None, xml_path=None):
        sources = sum(value is not None for value in (root, json_path, xml_path))
        if sources > 1:
            raise ValueError('Pass only one of root, json_path, or xml_path')
        if json_path is not None:
            self.__root = self.tree_from_json(json_path)
        elif xml_path is not None:
            self.__root = self.tree_from_xml(xml_path)
        elif root is None:
            self.__root = Node(Occurrence(1, 'root'))
        elif isinstance(root, Occurrence):
            _identifier(root.identifier, allow_empty=False)
            self.__root = Node(root)
        else:
            raise TypeError('root must be an Occurrence or None')
        for node in PreOrderIter(self.__root):
            node.name._tree = self
            node.name._node = node

    @property
    def root(self):
        return self.__root

    @property
    def leaves(self):
        return tuple(self.__root.leaves)

    def __iter__(self):
        return iter(PreOrderIter(self.__root))

    def num_of_nodes(self):
        return sum(1 for _ in self)

    def add(self, probability: float, identifier: str, parent=None, *, strict=False):
        """Add and return a child node.

        Set ``strict=True`` to reject a sibling sum above one instead of warning.
        """
        if not isinstance(strict, bool):
            raise TypeError('strict must be boolean')
        identifier = _identifier(identifier, allow_empty=False)
        if self.get_node_by_id(identifier) is not None:
            raise ValueError(f"identifier {identifier!r} already exists in this tree")
        if parent is None:
            parent = self.__root
        if not isinstance(parent, Node):
            raise TypeError('parent must be an anytree Node or None')
        if parent.root is not self.__root:
            raise ValueError('parent does not belong to this probability tree')
        occurrence = Occurrence(probability, identifier)
        projected_sum = sum(child.name.chance for child in parent.children)+occurrence.chance
        if projected_sum > 1+1e-12:
            message = (f'Probability sum for children of {parent.name.identifier!r} is '
                       f'{projected_sum}, expected 1 or less')
            if strict:
                raise ValueError(message)
            warnings.warn(message, UserWarning, stacklevel=2)
        node = Node(name=occurrence, parent=parent)
        occurrence._tree, occurrence._node = self, node
        return node

    @staticmethod
    def get_depth(node: Node):
        if not isinstance(node, Node):
            raise TypeError('node must be an anytree Node')
        return node.depth

    @staticmethod
    def get_level_sum(node: Node):
        if not isinstance(node, Node):
            raise TypeError('node must be an anytree Node')
        return sum(current.name.chance for current in (node,)+node.siblings)

    def _resolve(self, path=None):
        if path is None:
            return self.__root
        if isinstance(path, Node):
            if path.root is not self.__root:
                raise ValueError('The node does not belong to this probability tree')
            return path
        if not isinstance(path, (list, tuple, str)):
            raise TypeError(f'Expected a path or Node, got {type(path)}')
        identifiers = path.split('/') if isinstance(path, str) else list(path)
        if not all(isinstance(identifier, str) for identifier in identifiers):
            raise TypeError('path identifiers must be text')
        identifiers = [identifier for identifier in identifiers if identifier]
        if identifiers and identifiers[0] == self.__root.name.identifier:
            identifiers.pop(0)
        current = self.__root
        for identifier in identifiers:
            current = next((child for child in current.children
                            if child.name.identifier == identifier), None)
            if current is None:
                raise ValueError(f"Invalid probability path at node {identifier!r}")
        return current

    def get_probability(self, path=None, *, ndigits=None) -> float:
        """Return full-precision cumulative probability for a node or path.

        Use ``ndigits`` only when explicit display rounding is desired.
        """
        node = self._resolve(path)
        result = math.prod(current.name.chance for current in node.path)
        if ndigits is not None:
            if isinstance(ndigits, bool) or not isinstance(ndigits, int):
                raise TypeError('ndigits must be an integer or None')
            return round(result, ndigits)
        return result

    def path_probabilities(self, *, leaves_only=True):
        """Return ``{path: cumulative_probability}`` for leaves or all nodes."""
        nodes = self.leaves if leaves_only else tuple(self)
        return {self.get_node_path(node): self.get_probability(node) for node in nodes}

    def most_likely_leaf(self):
        """Return the leaf with the greatest cumulative path probability."""
        if not self.leaves:
            return self.__root
        return max(self.leaves, key=self.get_probability)

    @staticmethod
    def biggest_probability_node(node: Node) -> Node:
        """Return the descendant with the largest local branch probability.

        This retains the historical behavior. Use ``most_likely_leaf`` for
        cumulative outcome probability.
        """
        if not isinstance(node, Node):
            raise TypeError('node must be an anytree Node')
        return max(PreOrderIter(node), key=lambda current: current.name.chance)

    def get_node_path(self, node):
        if isinstance(node, str):
            found = self.get_node_by_id(node)
            if found is None:
                raise ValueError(f'Unknown node identifier {node!r}')
            node = found
        if not isinstance(node, Node):
            raise TypeError('node must be an identifier or anytree Node')
        if node.root is not self.__root:
            raise ValueError('The node does not belong to this probability tree')
        identifiers = [current.name.identifier for current in node.path]
        if identifiers and identifiers[0] == self.__root.name.identifier:
            identifiers.pop(0)
        return '/'.join(identifiers) or self.__root.name.identifier

    def get_node_by_id(self, identifier: str):
        _identifier(identifier)
        return next((node for node in self if node.name.identifier == identifier), None)

    def remove(self, *nodes):
        """Detach one or more nodes and their descendants; return this tree."""
        resolved = []
        for node in nodes:
            if isinstance(node, str):
                node = self.get_node_by_id(node)
                if node is None:
                    raise ValueError('Unknown node identifier')
            if not isinstance(node, Node):
                raise TypeError('nodes must be identifiers or anytree Nodes')
            if node is self.__root:
                raise ValueError('Cannot remove the root node')
            if node.root is not self.__root:
                raise ValueError('The node does not belong to this probability tree')
            resolved.append(node)
        for node in resolved:
            if node.root is self.__root:
                node.parent = None
                for descendant in PreOrderIter(node):
                    descendant.name._tree = None
                    descendant.name._node = None
        return self

    def validate(self, *, require_complete=False, tolerance=1e-12):
        """Validate identifiers, ownership, and sibling probability sums."""
        if not isinstance(require_complete, bool):
            raise TypeError('require_complete must be boolean')
        if (isinstance(tolerance, bool) or not isinstance(tolerance, Real)
                or not math.isfinite(float(tolerance)) or tolerance < 0):
            raise ValueError('tolerance must be a finite nonnegative real number')
        identifiers = set()
        for node in self:
            identifier = _identifier(node.name.identifier, allow_empty=False)
            if identifier in identifiers:
                raise ValueError(f'duplicate identifier {identifier!r}')
            identifiers.add(identifier)
            _probability(node.name.chance, name=f'probability for {identifier!r}')
            if node.children:
                total = sum(child.name.chance for child in node.children)
                if total > 1+tolerance:
                    raise ValueError(f'child probabilities of {identifier!r} exceed 1')
                if require_complete and not math.isclose(total, 1., abs_tol=tolerance,
                                                         rel_tol=0.):
                    raise ValueError(f'child probabilities of {identifier!r} do not sum to 1')
        return True

    def is_complete(self, *, tolerance=1e-12):
        try:
            self.validate(require_complete=True, tolerance=tolerance)
        except ValueError:
            return False
        return True

    def to_dict(self):
        """Return a stable, JSON-serializable dictionary representation."""
        return {
            node.name.identifier: {
                'parent': node.parent.name.identifier if node.parent else None,
                '_identifier': node.name.identifier,
                '_chance': node.name.chance,
            }
            for node in self
        }

    @classmethod
    def from_dict(cls, payload):
        """Build a tree from ``to_dict`` data, regardless of node ordering."""
        if not isinstance(payload, dict) or not payload:
            raise ValueError('tree data must be a nonempty dictionary')
        records = {}
        for key, raw in payload.items():
            if not isinstance(raw, dict):
                raise ValueError('each tree node must be a dictionary')
            identifier = raw.get('_identifier', raw.get('identifier', key))
            identifier = _identifier(identifier, allow_empty=False)
            if identifier != key:
                raise ValueError('tree node key and identifier must match')
            if identifier in records:
                raise ValueError(f'duplicate identifier {identifier!r}')
            chance = _probability(raw.get('_chance', raw.get('chance')), name='chance')
            parent = raw.get('parent')
            if parent is not None:
                parent = _identifier(parent, allow_empty=False)
            records[identifier] = (chance, parent)
        roots = [identifier for identifier, (_, parent) in records.items() if parent is None]
        if len(roots) != 1:
            raise ValueError('tree data must contain exactly one root')
        root_id = roots[0]
        tree = cls(root=Occurrence(records[root_id][0], root_id))
        pending = {key: value for key, value in records.items() if key != root_id}
        while pending:
            progress = False
            for identifier, (chance, parent_id) in list(pending.items()):
                parent = tree.get_node_by_id(parent_id)
                if parent is not None:
                    tree.add(chance, identifier, parent=parent, strict=True)
                    del pending[identifier]
                    progress = True
            if not progress:
                unresolved = ', '.join(sorted(pending))
                raise ValueError(f'unresolved or cyclic tree parents for: {unresolved}')
        return tree

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, source):
        """Load from a JSON string, bytes, or path."""
        if isinstance(source, (str, bytes)):
            text = source.decode('utf-8') if isinstance(source, bytes) else source
            if text.lstrip().startswith('{'):
                return cls.from_dict(json.loads(text))
        path = Path(source)
        return cls.from_dict(json.loads(path.read_text(encoding='utf-8')))

    @staticmethod
    def tree_from_json(file_path: str) -> Node:
        return ProbabilityTree.from_json(file_path).root

    def export_json(self, path: str):
        Path(path).write_text(self.to_json(indent=4)+'\n', encoding='utf-8')

    def to_xml_str(self, root_name: str = 'MyTree'):
        if not isinstance(root_name, str) or not _XML_NAME.fullmatch(root_name):
            raise ValueError('root_name must be a valid XML element name')
        root = Element(root_name)
        for current in self:
            node = SubElement(root, 'node')
            SubElement(node, 'parent').text = (current.parent.name.identifier
                                               if current.parent else 'None')
            SubElement(node, 'identifier').text = current.name.identifier
            SubElement(node, 'chance').text = repr(current.name.chance)
        return tostring(root, encoding='unicode')

    def export_xml(self, file_path: str = '', root_name: str = 'MyTree'):
        Path(file_path).write_text(self.to_xml_str(root_name=root_name)+'\n', encoding='utf-8')

    @classmethod
    def from_xml(cls, source):
        document = parse(Path(source))
        payload = {}
        for node in document.getroot().findall('./node'):
            identifier_element = node.find('./identifier')
            chance_element = node.find('./chance')
            parent_element = node.find('./parent')
            identifier = identifier_element.text if identifier_element is not None else None
            if identifier is None or chance_element is None or chance_element.text is None:
                raise ValueError('each XML node requires identifier and chance values')
            parent_text = None if parent_element is None else parent_element.text
            parent = None if parent_text is None or parent_text.strip().lower() in ('', 'none') else parent_text.strip()
            if identifier in payload:
                raise ValueError(f'duplicate identifier {identifier!r}')
            payload[identifier] = {'parent': parent, '_identifier': identifier,
                                   '_chance': float(chance_element.text.strip())}
        return cls.from_dict(payload)

    @staticmethod
    def tree_from_xml(xml_file: str):
        return ProbabilityTree.from_xml(xml_file).root

    def copy(self):
        return type(self).from_dict(self.to_dict())

    def _signature(self):
        return tuple(sorted((node.name.identifier,
                             node.parent.name.identifier if node.parent else None,
                             node.name.chance) for node in self))

    def __str__(self):
        return ''.join(f'{prefix}{node.name.identifier}:{node.name.chance}\n'
                       for prefix, _, node in RenderTree(self.__root))

    def __contains__(self, node):
        if isinstance(node, Node):
            return any(current is node for current in self)
        if isinstance(node, str):
            return self.get_node_by_id(node) is not None
        raise TypeError(f"ProbabilityTree.__contains__(): expected type 'str' or 'Node', got {type(node)}")

    def __eq__(self, other):
        if not isinstance(other, ProbabilityTree):
            return NotImplemented
        return self._signature() == other._signature()

    def __ne__(self, other):
        result = self.__eq__(other)
        return NotImplemented if result is NotImplemented else not result
