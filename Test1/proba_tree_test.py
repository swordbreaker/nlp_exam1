from proba_tree import Node

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


n1 = Node(0.95, label="B")
n2 = Node(0.05, label="_B")
n3 = Node(0.03, label="B")
n4 = Node(0.97, label="_B")

n5 = Node(0.6, label="A", childs=[n1, n2])
n6 = Node(0.4, label="_A", childs=[n3, n4])

root = Node(0, childs=[n5, n6], is_root=True)

root.prop_label("B")