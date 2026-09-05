# Probability foundation

KiwiCalc probability trees model conditional branch probabilities. Every branch
probability must be finite and between zero and one, node identifiers are unique,
and a child can only be attached to a parent in the same tree.

## Events and explicit assumptions

```python
rain = kw.Occurrence(0.4, 'rain')
traffic = kw.Occurrence(0.3, 'traffic')

rain.complement
rain.independent_intersection(traffic)
rain.independent_union(traffic)
rain.union_with_overlap(traffic, overlap=0.15)
```

The historical `intersection()` and `union()` methods remain compatible aliases
for the independent-event operations. Prefer the explicit names in new code.

## Build and inspect a tree

```python
tree = kw.ProbabilityTree(root=kw.Occurrence(1, 'start'))
pass_test = tree.add(0.7, 'pass')
tree.add(0.3, 'fail')
tree.add(0.2, 'excellent', parent=pass_test)
tree.add(0.8, 'ordinary', parent=pass_test)

tree.get_probability('start/pass/excellent')
tree.path_probabilities()
tree.most_likely_leaf()
tree.validate(require_complete=True)
```

`get_probability()` retains full floating-point precision. Use `ndigits=` only
when explicit presentation rounding is desired.

`add()` continues to warn when sibling probabilities exceed one for compatibility.
Use `strict=True` to reject the addition atomically:

```python
tree.add(0.5, 'another-result', strict=True)
```

## Mutation and identity

```python
clone = tree.copy()
assert clone == tree

clone.remove('fail')
assert clone != tree
```

Removing a node removes its complete subtree from the tree. The detached nodes
remain ordinary `anytree` nodes if an application still holds references to them.
The root cannot be removed.

## Serialization

```python
payload = tree.to_dict()       # JSON-serializable
text = tree.to_json()
restored = kw.ProbabilityTree.from_json(text)

tree.export_json('tree.json')
tree.export_xml('tree.xml', root_name='ProbabilityTree')
```

JSON parents are stored as identifier strings, not Python node objects. Import is
order-independent and rejects missing parents, cycles, multiple roots, invalid
probabilities, and mismatched identifiers. XML output escapes user text and
validates its root element name.
