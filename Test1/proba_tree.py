import itertools, functools
from functools import reduce
import operator
from graphviz import Digraph

class Node:
    def __init__(self, weight, childs: ["Node"] = [], label:str = "", is_root = False):
        self.weight = weight
        self.childs = childs
        self.parent = None
        self.is_root = is_root
        self.label = label
        for c in childs:
            c.parent = self

    def prop_up(self)->float:
        if(self.is_root): 
            return 1
        else:
            return self.weight * self._parent_prop()

    def _parent_prop(self)->float:
        if(self.parent is None):
            return 1
        else:
            return self.parent.prop_up()

    def find(self, label:str) -> ["Node"]:
        l = []

        if(self.label == label): 
            l.append(self)

        if len(self.childs) > 0:
            for c in self.childs:
                cl = c.find(label)
                l = l + cl
                #if len(cl) > 0:
                #    l.append(cl)

        return l
        return reduce(operator.concat, l)

    def prop_label(self, label:str):
        l = self.find(label)
        return sum([n.prop_up() for n in l])

    def show(self):
        dot = Digraph()
        dot.node("0", label="root")
        self._draw(dot, 0)
        dot.render('graphviz/proba_tree', view=True)

    def _draw(self, dot:Digraph, i:int):
        k = i + 1
        for c in self.childs:
            if c.label == "":
                label = str(k)
            else:
                label = c.label

            dot.node(str(k), label=c.label)
            dot.edge(str(i), str(k), label=str(c.weight))
            k += c._draw(dot, k)

        return k

    def __add__(self, other:"Node"):
        return self.weight + other.weight

    def __mul__(self, other:"Node"):
        return self.weight * other.weight

    def __str__(self, **kwargs):
        l = [c.__str__() for c in self.childs]
        return f"({self.label} [{l}])"

n1 = Node(0.97, label="OK")
n2 = Node(0.03, label="_OK")
n3 = Node(0.96, label="OK")
n4 = Node(0.4,  label="_OK")
n5 = Node(0.98, label="OK")
n6 = Node(0.02, label="_OK")
n7 = Node(0.5,  childs=[n1, n2], label="1")
n8 = Node(0.25, childs=[n3, n4], label="2")
n9 = Node(0.25, childs=[n5, n6], label="3")
root = Node(0, childs=[n7, n8, n9], label="root", is_root=True)

print(n1.prop_up())
print(root.prop_label("OK"))
print(root.prop_label("_OK"))

root.show()