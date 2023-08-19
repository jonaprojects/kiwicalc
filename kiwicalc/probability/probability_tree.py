class ProbabilityTree:
    """
    A class that is designed in order to represent probability trees and handle with them with greater ease.
    You can also import and export ProbabiltyTree objects in JSON and XML formats.
    """

    @staticmethod
    def tree_from_json(file_path: str) -> Node:
        with open(file_path, 'r') as json_file:
            tree_json = json_file.read()
        tree_structure: dict = json.loads(tree_json)
        tree_values = list(tree_structure.values())
        root_dict = tree_values[0]
        root = Node(name=Occurrence(
            root_dict["_chance"], identifier=root_dict["_identifier"]), parent=None)
        existing_nodes = [root]
        for node_dict in tree_values[1:]:
            parent_name = node_dict["parent"]
            if parent_name is None:
                parent_node = None
            else:
                parent_node = None
                for existing_node in existing_nodes:
                    if existing_node.name._identifier == parent_name:
                        parent_node = existing_node
                        break
                if parent_node is None:
                    warnings.warn(f"ProbabilityTree.tree_from_json(): couldn't find parent '{parent_name}."
                                  f"{node_dict['_identifier']} will be lost unless it is resolved.'")
            new_node = Node(name=Occurrence(node_dict["_chance"], identifier=node_dict["_identifier"]),
                            parent=parent_node)
            existing_nodes.append(new_node)
        return root

    @staticmethod
    def tree_from_xml(xml_file: str):
        """ import a ProbabilityTree object from an XML file.
         For example:
        <myCustomTree>
            <node>
               <parent>None</parent>
                <identifier>root</identifier>
                <chance>1</chance>
            </node>
            <node>
               <parent>root</parent>
                <identifier>son1</identifier>
                <chance>0.6</chance>
            </node>
            <node>
               <parent>root</parent>
                <identifier>son2</identifier>
                <chance>0.4</chance>
            </node>
        </myCustomTree>
         """
        tree = parse(xml_file)
        root = tree.getroot()  # first tag, contains all
        nodes = []
        for node in root.findall('./node'):
            name = node.find('./identifier').text
            parent_name = node.find('./parent').text
            if parent_name.lower().strip() in ("", "none") or parent_name is None:
                parent_node = None
            else:
                parent_node = None
                for existing_node in nodes:
                    if existing_node.name._identifier == parent_name:
                        parent_node = existing_node
                        break
                if parent_node is None:
                    warnings.warn(f"ProbabilityTree.tree_from_xml(): couldn't find parent '{parent_name}."
                                  f"{name} will be lost unless it is resolved.'")

            probability = node.find('./chance').text
            nodes.append(Node(name=Occurrence(chance=float(
                probability.strip()), identifier=name), parent=parent_node))
        return nodes[0]

    def to_dict(self):
        root = self.__root
        nodes = [children for children in
                 ZigZagGroupIter(root)]
        new_dict = {}
        for level_nodes in nodes:
            for node in level_nodes:
                new_dict[node.name.identifier] = {"parent": node.parent, "_identifier": node.name.identifier,
                                                  "_chance": node.name.chance}
        return new_dict

    @staticmethod
    def __to_dict_json(tree):
        root = tree.root()
        nodes = [children for children in
                 ZigZagGroupIter(root)]
        new_dict = {}
        for level_nodes in nodes:
            for node in level_nodes:
                new_dict[node.name._identifier] = {"parent": node.parent.name._identifier if node.parent else None,
                                                   "_identifier": node.name._identifier,
                                                   "_chance": node.name._chance}
        return new_dict

    @staticmethod
    def __tree_to_json(path: str, tree):
        dictionary = ProbabilityTree.__to_dict_json(tree)
        with open(path, 'w') as fp:
            json.dump(dictionary, fp, indent=4)

    def export_json(self, path: str):
        return ProbabilityTree.__tree_to_json(path=path, tree=self)

    def to_xml_str(self, root_name: str = "MyTree"):
        xml_accumulator = f"<{root_name}>\n"
        for children in ZigZagGroupIter(self.__root):
            for node in children:
                xml_accumulator += f"\t<node>\n"
                if node.parent is not None:
                    xml_accumulator += f"\t\t<parent>{node.parent.name.identifier}</parent>\n"
                else:
                    xml_accumulator += f"\t\t<parent>None</parent>\n"
                xml_accumulator += f"\t\t<identifier>{node.name.identifier}</identifier>\n"
                xml_accumulator += f"\t\t<chance>{node.name.chance}</chance>\n"
                xml_accumulator += f"\t</node>\n"
        xml_accumulator += f"</{root_name}>"
        return xml_accumulator

    def export_xml(self, file_path: str = "", root_name: str = "MyTree"):
        with open(f"{file_path}", "w") as f:
            f.write(self.to_xml_str(root_name=root_name))

    def __init__(self, root=None, json_path=None, xml_path=None):
        """
        Creating a new probability tree
        :param root: a string that describes the root occurrence of the tree (Optional)
        :param json_path: in order to import the tree from json file (Optional)
        :param xml_path: in order to import the tree from xml file (Optional)
        """
        if json_path:
            self.__root = ProbabilityTree.tree_from_json(json_path)
        elif xml_path:
            self.__root = ProbabilityTree.tree_from_xml(xml_path)
        else:
            if root is None:
                self.__root = Node(Occurrence(1, "root"))
            elif isinstance(root, Occurrence):
                self.__root = Node(root)
            else:
                raise TypeError

    @property
    def root(self):
        return self.__root

    def add(self, probability: float, identifier: str, parent=None):
        """
        creates a new node, adds it to the tree, and returns it
        :param probability: The probability of occurrence
        :param identifier: Unique string that represents the new node
        :param parent: the parent of node that will be created
        :return: returns the node that was created
        """
        occurrence = Occurrence(probability, identifier)
        if parent is None:
            parent = self.__root
        node = Node(name=occurrence, parent=parent, edge=2)
        level_sum = ProbabilityTree.get_level_sum(node)
        if level_sum > 1:
            depth = ProbabilityTree.get_depth(node)
            warnings.warn(f"ProbabilityTree.add_occurrence(): Probability sum in depth {depth} (sum is {level_sum})"
                          f") is bigger than 1, expected 1 or less.")
        return node

    @staticmethod
    def get_depth(node: Node):
        return node.depth

    @staticmethod
    def get_level_sum(node: Node):
        if not node.siblings:
            return node.name.chance
        return node.name.chance + reduce(lambda a, b: a.name.chance + b.name.chance,
                                         node.siblings).name.chance

    def num_of_nodes(self):
        return sum(len(children) for children in ZigZagGroupIter(self.__root))

    def get_probability(self, path=None) -> float:
        """

        :param path: gets a specific path of the names of the nodes, such as ["root","son","grandson"]
        or the node itself
        :type path: list or node
        :return: returns the probability up to that path or node
        :rtype: float
        """
        probability = 1

        if isinstance(path, (list, tuple, str)):
            if isinstance(path, str):
                path = path.split('/')
            if path is None:
                path = self.__root.name.identifier
            nodes = [[node.name for node in children] for children in
                     ZigZagGroupIter(self.__root, filter_=lambda n: n.name.identifier in path)]
            probability: float = 1
            for node in nodes:
                for occurrence in node:
                    probability *= occurrence.chance
        elif isinstance(path, Node):
            for node in path.iter_path_reverse():
                probability *= node.name.chance
        return round(probability, 5)

    @staticmethod
    def biggest_probability_node(node: Node) -> Node:
        # TODO: check whether the new implementation works
        return max(ZigZagGroupIter(node), key=operator.attrgetter(node.name.chance))

    def get_node_path(self, node: Union[str, Node]):
        accumulator = ""
        if isinstance(node, str):
            node = self.get_node_by_id(node)
        if len(node.ancestors) <= 1:
            return node.name.identifier
        for ancestor in node.ancestors[1:]:
            accumulator += f"/{ancestor.name.identifier}"
        accumulator += f"/{node.name.identifier}"
        return accumulator[1:]

    def get_node_by_id(self, identifier: str):
        for children in ZigZagGroupIter(self.__root):
            for node in children:
                if node.name.identifier == identifier:
                    return node
        return None

    def remove(self, *nodes):
        raise NotImplementedError

    def __str__(self):
        accumulator = ""
        for pre, fill, node in RenderTree(self.__root):
            accumulator += ("%s%s:%s\n" %
                            (pre, node.name.identifier, node.name.chance))
        return accumulator

    def __contains__(self, node):
        """

        :param node: a node or the _identifier of the occurrence object inside the name attribute of the node
        :return: returns True if found a match, else False
        """
        if isinstance(node, Node):
            for current_node in PreOrderIter(self.__root):
                if current_node is Node or current_node == node:
                    return True
                # TODO: implement equality with __eq__ so two duplicates will be considered equal as well
                return False
        elif isinstance(node, str):
            for current_node in PreOrderIter(self.__root):
                if current_node.name.identifier == node:
                    return True
            return False
        else:
            raise TypeError(
                f"ProbabilityTree.__contains__(): expected type 'str' or 'Node', got {type(node)}")

    def __eq__(self, other):
        pass

    def __ne__(self, other):
        return not self.__eq__(other)
